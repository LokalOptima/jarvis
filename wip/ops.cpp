/**
 * ops.cpp - Built-in ops, OPS dict, engine singletons, pipeline execution.
 *
 * Each op factory returns a Step with (name, params, placement, op lambda).
 * The OPS dict maps op names to factories for server-side resolution.
 */

#include "ops.h"
#include "detect.h"
#include "vad_ggml.h"
#include "weather.hpp"
#include <audio_async.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

// ---- Engine singletons ----

static SileroVad         *g_vad     = nullptr;
static std::atomic<bool> *g_running = nullptr;

SileroVad         &vad()     { return *g_vad; }
std::atomic<bool> &running() { return *g_running; }

void set_engine_singletons(SileroVad *v, std::atomic<bool> *r) {
    g_vad = v;
    g_running = r;
}

// ---- Pipeline execution ----

void run_pipeline(const Pipeline &steps, Msg &msg) {
    for (const auto &step : steps) {
        step.op(msg);
    }
}

// ---- Audio playback ----

void play_wav(const uint8_t *data, size_t size, bool wait) {
    char tmp[] = "/tmp/jarvis-XXXXXX.wav";
    int fd = mkstemps(tmp, 4);
    if (fd < 0) return;
    if (write(fd, data, size) < 0) { ::close(fd); return; }
    ::close(fd);
    pid_t pid = fork();
    if (pid == 0) {
#ifdef __APPLE__
        execlp("afplay", "afplay", tmp, nullptr);
#else
        execlp("aplay", "aplay", "-q", tmp, nullptr);
        execlp("paplay", "paplay", tmp, nullptr);
#endif
        _exit(127);
    }
    if (wait && pid > 0) {
        waitpid(pid, nullptr, 0);
        unlink(tmp);
    }
}

// ---- Helpers ----

static bool save_wav(const std::string &path, const float *pcm, int n_samples) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    int data_bytes = n_samples * 2;
    int file_bytes = 36 + data_bytes;
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_bytes, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    int fmt_size = 16; fwrite(&fmt_size, 4, 1, f);
    short audio_fmt = 1; fwrite(&audio_fmt, 2, 1, f);
    short channels = 1;  fwrite(&channels, 2, 1, f);
    int rate = JARVIS_SAMPLE_RATE; fwrite(&rate, 4, 1, f);
    int byte_rate = rate * 2; fwrite(&byte_rate, 4, 1, f);
    short block_align = 2; fwrite(&block_align, 2, 1, f);
    short bits = 16; fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_bytes, 4, 1, f);
    std::vector<short> buf(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float s = pcm[i];
        if (s > 1.0f) s = 1.0f; if (s < -1.0f) s = -1.0f;
        buf[i] = (short)(s * 32767.0f);
    }
    fwrite(buf.data(), 2, n_samples, f);
    fclose(f);
    return true;
}

static std::string run_transcription(const std::string &cmd, const std::string &wav_path) {
    std::string full = cmd + " " + wav_path + " 2>/dev/null";
    FILE *pipe = popen(full.c_str(), "r");
    if (!pipe) return "";
    std::string last;
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
            line.pop_back();
        if (!line.empty()) last = line;
    }
    pclose(pipe);
    return last;
}

static std::string shell_escape(const std::string &s) {
    std::string out;
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    return out;
}

// ---- Op: transcribe (REMOTE) ----
// Cooldown-based eager transcription: record from msg.source, fire async
// transcription on speech->silence edge, commit after cooldown expires.

Step transcribe(const std::string &params) {
    return {"transcribe", params, REMOTE, [params](Msg &msg) {
        const int slide_ms = JARVIS_SLIDE_MS;
        const int cooldown_max_ms = 600;
        const int max_record_ms = 30000;
        int cooldown_ms = cooldown_max_ms;
        int total_ms = 0;

        std::vector<float> recording;
        recording.reserve(max_record_ms * JARVIS_SAMPLE_RATE / 1000);
        std::vector<float> chunk;
        std::future<std::string> pending;

        std::cout << "  Recording..." << std::endl;

        while (running()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(slide_ms));

            msg.source->get(slide_ms, chunk);
            if (chunk.empty()) continue;

            recording.insert(recording.end(), chunk.begin(), chunk.end());
            total_ms += slide_ms;
            if (total_ms >= max_record_ms) break;

            // VAD on this chunk
            bool has_speech = false;
            for (size_t i = 0; i + SileroVad::CHUNK_SAMPLES <= chunk.size();
                 i += SileroVad::CHUNK_SAMPLES) {
                if (vad().process(chunk.data() + i) > 0.5f) has_speech = true;
            }

            if (!has_speech) {
                // Fire speculative transcription on first silence tick after speech
                if (cooldown_ms == cooldown_max_ms &&
                    (int)recording.size() >= JARVIS_SAMPLE_RATE / 4) {
                    if (pending.valid()) pending.get();  // drain previous
                    save_wav("/tmp/jarvis_spec.wav", recording.data(), recording.size());
                    pending = std::async(std::launch::async,
                        run_transcription, params, "/tmp/jarvis_spec.wav");
                }
                cooldown_ms -= slide_ms;
                render_bar("recording", (float)cooldown_ms / cooldown_max_ms, 1.0f, 0, true);
                if (cooldown_ms <= 0) {
                    // Silence timeout: commit
                    msg.text = pending.valid() ? pending.get() : "";
                    return;
                }
            } else {
                cooldown_ms = cooldown_max_ms;  // speech resumed
                render_bar("recording", 1.0f, 1.0f, 0, true);
            }
        }

        // Interrupted
        if (pending.valid()) pending.get();
        msg.text = "";
    }};
}

// ---- Op: weather (REMOTE) ----

Step weather(const std::string &params) {

    return {"weather", params, REMOTE, [](Msg &msg) {
        msg.text = get_weather_text();
    }};
}

// ---- Op: tts (REMOTE) ----
// TTS msg.text -> msg.audio. Pauses mic during playback to avoid self-hearing.

Step tts(const std::string &params) {

    return {"tts", params, REMOTE, [](Msg &msg) {
        if (msg.text.empty()) return;
        auto wav = speak_to_wav(msg.text);
        if (wav.empty()) return;
        msg.audio = std::move(wav);

        // CLI mode (has_device): play locally, pause mic to avoid self-hearing.
        // Server mode (push audio, no device): just synthesize into msg.audio.
        bool has_mic = msg.source && msg.source->has_device();
        if (!has_mic) return;

        msg.source->pause();
        play_wav(msg.audio.data(), msg.audio.size(), true);
        msg.source->resume();
    }};
}

// ---- Op: print (LOCAL) ----

Step print(const std::string &params) {

    return {"print", params, LOCAL, [](Msg &msg) {
        if (!msg.text.empty()) {
            std::cout << "  > " << msg.text << std::endl;
        }
    }};
}

// ---- Op: tmux (LOCAL) ----

Step tmux(const std::string &params) {

    return {"tmux", params, LOCAL, [](Msg &msg) {
        if (msg.text.empty()) return;
        std::string cmd = "tmux send-keys -l -- '" + shell_escape(msg.text) + "' 2>/dev/null";
        int ret = system(cmd.c_str()); (void)ret;
    }};
}

// ---- Op: save (LOCAL) ----

Step save(const std::string &params) {
    return {"save", params, LOCAL, [params](Msg &msg) {
        if (msg.text.empty() || params.empty()) return;
        std::ofstream f(params, std::ios::app);
        if (f) f << msg.text << "\n";
    }};
}

// ---- Op: fire (LOCAL) ----
// Double-fork so the command runs detached (no zombie, no SIGCHLD needed).

Step fire(const std::string &params) {
    return {"fire", params, LOCAL, [params](Msg &) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child: fork again so grandchild is orphaned (reparented to init)
            if (fork() == 0) {
                execl("/bin/sh", "sh", "-c", params.c_str(), nullptr);
                _exit(127);
            }
            _exit(0);  // child exits immediately
        }
        if (pid > 0) waitpid(pid, nullptr, 0);  // reap child (instant)
    }};
}

// ---- Op: run (LOCAL) ----

Step run(const std::string &params) {
    return {"run", params, LOCAL, [params](Msg &) {
        int ret = system(params.c_str()); (void)ret;
    }};
}

// ---- OPS dict ----

const std::unordered_map<std::string, OpFactory> OPS = {
    {"transcribe", ::transcribe},
    {"weather",    ::weather},
    {"tts",        ::tts},
    {"print",      ::print},
    {"tmux",       ::tmux},
    {"save",       ::save},
    {"fire",       ::fire},
    {"run",        ::run},
};
