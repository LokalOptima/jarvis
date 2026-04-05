/**
 * jarvis.cpp - Wake word detection engine (local mode).
 *
 * Pipeline:
 *   SDL2 mic capture → ring buffer → detect_once() → callback
 */

#include "jarvis.h"
#include "detect.h"
#include "vad_ggml.h"
#include <audio_async.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <unistd.h>

// ---- Signal handling ----

static std::atomic<bool> g_running{true};
static void signal_handler(int) { g_running.store(false, std::memory_order_relaxed); }

// ---- Jarvis implementation ----

struct Jarvis::Impl {
    struct whisper_context *ctx = nullptr;
    SileroVad vad;
    std::vector<LoadedKeyword> keywords;
};

Jarvis::Jarvis(const std::string &whisper_model,
               const std::string &vad_model) : impl(std::make_unique<Impl>()) {
    if (!impl->vad.load(vad_model)) {
        throw std::runtime_error("Failed to load VAD model: " + vad_model);
    }
    std::cout << "VAD loaded: " << vad_model << std::endl;

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    impl->ctx = whisper_init_from_file_with_params(whisper_model.c_str(), cparams);
    if (!impl->ctx) {
        throw std::runtime_error("Failed to load whisper model: " + whisper_model);
    }
    whisper_set_encoder_only(impl->ctx, true);
    std::cout << "Whisper loaded: " << whisper_model << std::endl;
}

Jarvis::~Jarvis() {
    if (impl && impl->ctx) whisper_free(impl->ctx);
}

void Jarvis::add_keyword(Keyword kw) {
    LoadedKeyword lk;
    lk.name = kw.name;
    lk.template_path = kw.template_path;
    lk.threshold = kw.threshold;
    lk.refractory_ms = kw.refractory_ms;
    if (!lk.templates.load(lk.template_path)) {
        throw std::runtime_error("Failed to load templates: " + lk.template_path);
    }
    int total = 0;
    for (const auto &t : lk.templates.items) total += t.n_frames;
    std::cout << "  " << lk.name << ": "
              << lk.templates.items.size() << " template(s), " << total << " frames"
              << (kw.callback ? " [callback]" : "") << std::endl;
    // Store callback separately (not in LoadedKeyword which is shared with server)
    callbacks.push_back(kw.callback);
    impl->keywords.push_back(std::move(lk));
}

void Jarvis::stop() {
    g_running = false;
}

void Jarvis::listen() {
    if (impl->keywords.empty()) {
        std::cerr << "No keywords registered" << std::endl;
        return;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGCHLD, SIG_IGN);
    g_running = true;

    auto audio = std::make_shared<audio_async>(static_cast<int>(JARVIS_BUFFER_SEC * 1000));
    audio->init(-1, JARVIS_SAMPLE_RATE);
    audio->resume();

    // Wait for the audio buffer to fill, showing progress
    {
        const int steps = 20;
        const int step_ms = (int)(JARVIS_BUFFER_SEC * 1000) / steps;
        for (int i = 1; i <= steps && g_running; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
            render_bar("buffering", (float)i / steps, 1.0f, 0, true);
        }
        std::cerr << "\r\033[K" << std::flush;
    }

    std::cout << "Listening... (" << impl->keywords.size() << " keyword(s), Ctrl+C to stop)" << std::endl;

    DetectScratch scratch;
    scratch.init();
    std::vector<float> pcm_buffer;
    pcm_buffer.reserve(JARVIS_BUFFER_SAMPLES);

    int refractory = 0;
    int refractory_total = 0;
    float default_thr = impl->keywords[0].threshold;

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(JARVIS_SLIDE_MS));

        if (refractory > 0) {
            float frac = (float)refractory / refractory_total;
            render_bar("cooldown", frac, 1.0f, 0, true);
            refractory--;
            continue;
        }

        // Get latest 200ms for VAD, then full 2s for detection
        audio->get(JARVIS_SLIDE_MS, pcm_buffer);
        if (pcm_buffer.empty()) continue;

        bool has_speech = false;
        for (size_t i = 0; i + SileroVad::CHUNK_SAMPLES <= pcm_buffer.size(); i += SileroVad::CHUNK_SAMPLES) {
            if (impl->vad.process(pcm_buffer.data() + i) > 0.5f) has_speech = true;
        }
        if (!has_speech) {
            render_bar("", 0.0f, default_thr, 0, true);
            continue;
        }

        audio->get(static_cast<int>(JARVIS_BUFFER_SEC * 1000), pcm_buffer);
        if (pcm_buffer.empty()) continue;

        DetectResult det = detect_once(impl->ctx, impl->keywords,
                                       pcm_buffer.data(), pcm_buffer.size(), scratch);

        int show_kw = det.keyword_index >= 0 ? det.keyword_index : det.best_keyword;
        float show_score = det.keyword_index >= 0 ? det.score : det.best_score;
        render_bar(impl->keywords[show_kw].name.c_str(), show_score,
                   impl->keywords[show_kw].threshold, det.elapsed_ms, false);

        if (det.keyword_index >= 0) {
            const auto &kw = impl->keywords[det.keyword_index];

            auto now = std::chrono::system_clock::now();
            auto tt = std::chrono::system_clock::to_time_t(now);
            char time_buf[32];
            std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", std::localtime(&tt));
            std::cerr << "\r\033[K" << std::flush;
            std::cout << "  [" << time_buf << "] " << kw.name
                      << "  sim=" << det.score << std::endl;

            auto &cb = callbacks[det.keyword_index];
            if (cb) cb(kw.name, det.score);

            impl->vad.reset();
            audio->clear();
            refractory_total = kw.refractory_ms / JARVIS_SLIDE_MS;
            refractory = refractory_total;
        }
    }

    std::cout << "\nStopped." << std::endl;
}

// ---- run_command helper ----

std::function<void(const std::string &, float)> run_command(const std::string &cmd) {
    return [cmd](const std::string &, float) {
        pid_t pid = fork();
        if (pid == 0) {
            execl("/bin/sh", "sh", "-c", cmd.c_str(), nullptr);
            _exit(127);
        } else if (pid < 0) {
            perror("fork");
        }
    };
}
