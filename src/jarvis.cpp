/**
 * jarvis.cpp - Wake word detection engine.
 *
 * Pipeline:
 *   SDL2 mic capture → ring buffer → whisper mel + encode
 *   → CMVN → subsequence DTW against enrolled templates → detect → callback
 */

#include "jarvis.h"
#include "whisper.h"
#include <audio_async.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

// ---- Configuration ----

#define SAMPLE_RATE       16000
#define BUFFER_SEC        2.0f
#define SLIDE_MS          200
#define WHISPER_DIM       384
#define MEL_HOP           160
#define CONV_STRIDE       2
#define ONSET_SKIP        2
#define STEP_PENALTY      0.1f

// ---- Signal handling ----

static std::atomic<bool> g_running{true};
static void signal_handler(int) { g_running.store(false, std::memory_order_relaxed); }

// ---- Terminal visualizer ----

static void render_bar(const char *name, float score, float threshold, int ms, bool silent) {
    static char buf[768];
    char *p = buf;

    constexpr int NAME_W = 14;
    constexpr int W = 36;
    int thr = std::max(0, std::min(W - 1, (int)(threshold * W)));

    *p++ = '\r'; *p++ = ' '; *p++ = ' ';

    int nlen = (int)strlen(name);
    int copy = std::min(nlen, NAME_W);
    memcpy(p, name, copy); p += copy;
    for (int i = copy; i < NAME_W; i++) *p++ = ' ';

    if (silent) {
        int filled = std::max(0, std::min(W, (int)(score * W)));
        memcpy(p, "\033[2m", 4); p += 4;
        for (int i = 0; i < W; i++) {
            if      (i < filled) { memcpy(p, "\xe2\x96\x88", 3); p += 3; }  // █
            else if (i == thr)   { memcpy(p, "\xe2\x94\x82", 3); p += 3; }  // │
            else                 { memcpy(p, "\xc2\xb7", 2); p += 2; }       // ·
        }
        memcpy(p, "\033[0m\033[K", 7); p += 7;
    } else {
        int filled = std::max(0, std::min(W, (int)(score * W)));
        static const char *esc[]  = { "\033[32m", "\033[1;31m", "\033[2m", "\033[33m" };
        static const int esc_len[] = { 5, 7, 4, 5 };
        int color = -1;

        for (int i = 0; i < W; i++) {
            int want;
            if      (i == thr)              want = 3;
            else if (i < filled && i < thr) want = 0;
            else if (i < filled)            want = 1;
            else                            want = 2;

            if (want != color) {
                memcpy(p, esc[want], esc_len[want]);
                p += esc_len[want];
                color = want;
            }

            if (i == thr)        { memcpy(p, "\xe2\x94\x82", 3); p += 3; }
            else if (i < filled) { memcpy(p, "\xe2\x96\x88", 3); p += 3; }
            else                 { memcpy(p, "\xc2\xb7", 2); p += 2; }
        }

        p += std::snprintf(p, 64, "\033[0m  %4.2f  %3dms\033[K", score, ms);
    }

    fwrite(buf, 1, p - buf, stderr);
}

// ---- CMVN ----

static void apply_cmvn(float *frames, int n_frames) {
    float mean[WHISPER_DIM] = {};
    float var[WHISPER_DIM] = {};

    for (int t = 0; t < n_frames; t++) {
        const float *f = frames + t * WHISPER_DIM;
        for (int d = 0; d < WHISPER_DIM; d++) mean[d] += f[d];
    }
    float inv_n = 1.0f / n_frames;
    for (int d = 0; d < WHISPER_DIM; d++) mean[d] *= inv_n;

    for (int t = 0; t < n_frames; t++) {
        const float *f = frames + t * WHISPER_DIM;
        for (int d = 0; d < WHISPER_DIM; d++) {
            float diff = f[d] - mean[d];
            var[d] += diff * diff;
        }
    }
    for (int d = 0; d < WHISPER_DIM; d++) var[d] = 1.0f / (std::sqrt(var[d] * inv_n) + 1e-10f);

    for (int t = 0; t < n_frames; t++) {
        float *f = frames + t * WHISPER_DIM;
        for (int d = 0; d < WHISPER_DIM; d++) f[d] = (f[d] - mean[d]) * var[d];
    }
}

// ---- Template matching ----

struct Template {
    std::vector<float> data;
    int n_frames = 0;
    const float *frame(int t) const { return data.data() + t * WHISPER_DIM; }
};

struct Templates {
    std::vector<Template> items;

    bool load(const std::string &path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Failed to open templates: " << path << std::endl;
            return false;
        }
        int32_t n;
        f.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
        if (n <= 0 || n > 10000) return false;

        items.resize(n);
        for (int i = 0; i < n; i++) {
            int32_t nf;
            f.read(reinterpret_cast<char *>(&nf), sizeof(int32_t));
            if (nf <= 0 || nf > 10000) return false;
            items[i].n_frames = nf;
            items[i].data.resize(nf * WHISPER_DIM);
            f.read(reinterpret_cast<char *>(items[i].data.data()), nf * WHISPER_DIM * sizeof(float));
        }
        return !f.fail();
    }

    static void compute_inv_norms(const float *input, int n_frames, std::vector<float> &inv_norms) {
        inv_norms.resize(n_frames);
        for (int i = 0; i < n_frames; i++) {
            float na = 0;
            const float *a = input + i * WHISPER_DIM;
            for (int k = 0; k < WHISPER_DIM; k++) na += a[k] * a[k];
            inv_norms[i] = na > 0 ? 1.0f / std::sqrt(na) : 0;
        }
    }

    static float cosine_dot(const float *a, const float *b_unit, float inv_norm_a) {
        float dot = 0;
        for (int i = 0; i < WHISPER_DIM; i++) dot += a[i] * b_unit[i];
        return dot * inv_norm_a;
    }

    // Subsequence DTW: template must fully match, can start/end anywhere in input.
    static float subdtw(const float *input, int n_input,
                        const Template &tmpl,
                        const std::vector<float> &inv_norms,
                        std::vector<float> &prev_row,
                        std::vector<float> &curr_row) {
        int n_tmpl = tmpl.n_frames;
        prev_row.resize(n_tmpl + 1);
        curr_row.resize(n_tmpl + 1);
        std::fill(prev_row.begin(), prev_row.end(), 1e30f);
        prev_row[0] = 0.0f;
        float best = 1e30f;

        for (int i = 1; i <= n_input; i++) {
            curr_row[0] = 0.0f;
            for (int j = 1; j <= n_tmpl; j++) {
                float c = 1.0f - cosine_dot(input + (i - 1) * WHISPER_DIM, tmpl.frame(j - 1), inv_norms[i - 1]);
                float diag = prev_row[j - 1];
                float ins  = prev_row[j] + STEP_PENALTY;
                float del  = curr_row[j - 1] + STEP_PENALTY;
                float best_prev = diag;
                if (ins < best_prev) best_prev = ins;
                if (del < best_prev) best_prev = del;
                curr_row[j] = c + best_prev;
            }
            if (curr_row[n_tmpl] < best) best = curr_row[n_tmpl];
            std::swap(prev_row, curr_row);
        }
        return best / n_tmpl;
    }

    float match(const float *input, int n_frames,
                const std::vector<float> &inv_norms,
                std::vector<float> &row_a, std::vector<float> &row_b) const {
        float best_sim = -1.0f;
        for (const auto &tmpl : items) {
            float cost = subdtw(input, n_frames, tmpl, inv_norms, row_a, row_b);
            float sim = 1.0f - cost;
            if (sim > best_sim) best_sim = sim;
        }
        return best_sim;
    }
};

// ---- Jarvis implementation ----

struct LoadedKeyword {
    Keyword config;
    Templates templates;
};

struct Jarvis::Impl {
    struct whisper_context *ctx = nullptr;
    std::vector<LoadedKeyword> keywords;
};

Jarvis::Jarvis(const std::string &whisper_model) : impl(std::make_unique<Impl>()) {
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
    lk.config = std::move(kw);
    if (!lk.templates.load(lk.config.template_path)) {
        throw std::runtime_error("Failed to load templates: " + lk.config.template_path);
    }
    int total = 0;
    for (const auto &t : lk.templates.items) total += t.n_frames;
    std::cout << "  " << lk.config.name << ": "
              << lk.templates.items.size() << " template(s), " << total << " frames"
              << (lk.config.callback ? " [callback]" : "") << std::endl;
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

    const int buffer_samples = static_cast<int>(BUFFER_SEC * SAMPLE_RATE);
    auto audio = std::make_shared<audio_async>(static_cast<int>(BUFFER_SEC * 1000));
    audio->init(-1, SAMPLE_RATE);
    audio->resume();

    // Wait for the audio buffer to fill, showing progress
    {
        const int steps = 20;
        const int step_ms = (int)(BUFFER_SEC * 1000) / steps;
        for (int i = 1; i <= steps && g_running; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
            render_bar("buffering", (float)i / steps, 1.0f, 0, true);
        }
        std::cerr << "\r\033[K" << std::flush;
    }

    std::cout << "Listening... (" << impl->keywords.size() << " keyword(s), Ctrl+C to stop)" << std::endl;

    const int max_encoder_frames = (buffer_samples / MEL_HOP + 1) / CONV_STRIDE + 1;
    const int max_encoder_floats = max_encoder_frames * WHISPER_DIM;
    std::vector<float> encoder_output(max_encoder_floats);
    std::vector<float> inv_norms;
    std::vector<float> dtw_row_a, dtw_row_b;
    std::vector<float> pcm_buffer;
    pcm_buffer.reserve(buffer_samples);

    int refractory = 0;
    int refractory_total = 0;
    float default_thr = impl->keywords[0].config.threshold;

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(SLIDE_MS));

        if (refractory > 0) {
            float frac = (float)refractory / refractory_total;
            render_bar("cooldown", frac, 1.0f, 0, true);
            refractory--;
            continue;
        }

        audio->get(static_cast<int>(BUFFER_SEC * 1000), pcm_buffer);
        if (pcm_buffer.empty()) continue;

        // Energy-based VAD (computed before zero-padding to avoid dilution)
        float energy = 0;
        for (auto s : pcm_buffer) energy += s * s;
        energy /= pcm_buffer.size();
        if (energy < 1e-6f) {
            render_bar("", 0.0f, default_thr, 0, true);
            continue;
        }

        // Zero-pad short buffers so whisper gets a full-length input
        if ((int)pcm_buffer.size() < buffer_samples)
            pcm_buffer.resize(buffer_samples, 0.0f);

        // Whisper mel + encode
        auto t0 = std::chrono::steady_clock::now();
        if (whisper_pcm_to_mel(impl->ctx, pcm_buffer.data(), pcm_buffer.size(), 1) != 0) {
            render_bar("", 0.0f, default_thr, 0, true);
            continue;
        }

        int mel_frames = whisper_n_len(impl->ctx);
        int actual_frames = (mel_frames + 1) / CONV_STRIDE;
        if (actual_frames <= 0) actual_frames = 1;
        whisper_set_audio_ctx(impl->ctx, actual_frames);

        if (whisper_encode(impl->ctx, 0, 1) != 0) {
            render_bar("", 0.0f, default_thr, 0, true);
            continue;
        }

        int n_floats = whisper_encoder_output(impl->ctx, encoder_output.data(), max_encoder_floats);
        if (n_floats <= 0) {
            render_bar("", 0.0f, default_thr, 0, true);
            continue;
        }
        int n_enc_frames = n_floats / WHISPER_DIM;

        float *enc_ptr = encoder_output.data() + ONSET_SKIP * WHISPER_DIM;
        n_enc_frames -= ONSET_SKIP;
        if (n_enc_frames <= 0) {
            render_bar("", 0.0f, default_thr, 0, true);
            continue;
        }
        apply_cmvn(enc_ptr, n_enc_frames);

        // Pre-compute input inverse norms once (shared across all keywords)
        Templates::compute_inv_norms(enc_ptr, n_enc_frames, inv_norms);

        // Match each keyword against its own threshold
        int fired_kw = -1;
        float fired_score = 0;
        int best_kw = 0;
        float best_score = -1.0f;
        for (int k = 0; k < (int)impl->keywords.size(); k++) {
            float score = impl->keywords[k].templates.match(
                enc_ptr, n_enc_frames, inv_norms, dtw_row_a, dtw_row_b);
            if (score > best_score) {
                best_score = score;
                best_kw = k;
            }
            if (fired_kw < 0 && score >= impl->keywords[k].config.threshold) {
                fired_kw = k;
                fired_score = score;
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        int total_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        int show_kw = fired_kw >= 0 ? fired_kw : best_kw;
        float show_score = fired_kw >= 0 ? fired_score : best_score;
        render_bar(impl->keywords[show_kw].config.name.c_str(), show_score,
                   impl->keywords[show_kw].config.threshold, total_ms, false);

        if (fired_kw >= 0) {
            const auto &kw = impl->keywords[fired_kw];

            auto now = std::chrono::system_clock::now();
            auto tt = std::chrono::system_clock::to_time_t(now);
            char time_buf[32];
            std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", std::localtime(&tt));
            std::cerr << "\r\033[K" << std::flush;
            std::cout << "  [" << time_buf << "] " << kw.config.name
                      << "  sim=" << fired_score << std::endl;

            if (kw.config.callback) {
                kw.config.callback(kw.config.name, fired_score);
            }

            audio->clear();
            refractory_total = kw.config.refractory_ms / SLIDE_MS;
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
