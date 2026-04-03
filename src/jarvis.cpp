/**
 * jarvis - Personalized wake word detector using Whisper Tiny encoder.
 *
 * Pipeline:
 *   SDL2 mic capture → ring buffer → whisper mel + encode
 *   → subsequence DTW against enrolled templates → detect
 *
 * No training required. Just enroll with your own recordings (see jarvis/enroll.py).
 * Templates are stored as a binary file (models/templates.bin).
 */

#include <whisper.h>
#include <audio_async.hpp>

#include <algorithm>
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
#include <vector>

// ---- Configuration ----

#define SAMPLE_RATE       16000
#define BUFFER_SEC        2.0f   // Audio buffer length
#define SLIDE_MS          200    // Detection interval (milliseconds)
#define WHISPER_DIM       384    // Whisper Tiny encoder embedding dimension
#define MEL_HOP           160    // Mel spectrogram hop length in samples
#define CONV_STRIDE       2      // Whisper encoder conv2 stride
#define DEFAULT_THRESHOLD 0.35f  // DTW similarity threshold
#define ONSET_SKIP        2      // Skip first N encoder frames (Whisper "start of audio" artifact)
#define STEP_PENALTY      0.1f   // DTW non-diagonal transition penalty

// ---- Terminal visualizer ----

static void render_bar(float score, float threshold, int ms, bool silent) {
    static char buf[768];
    char *p = buf;

    constexpr int W = 50;
    int thr = std::max(0, std::min(W - 1, (int)(threshold * W)));

    *p++ = '\r'; *p++ = ' '; *p++ = ' ';

    if (silent) {
        memcpy(p, "\033[2m", 4); p += 4;
        for (int i = 0; i < W; i++) {
            if (i == thr) { memcpy(p, "\xe2\x94\x82", 3); p += 3; }  // │
            else          { memcpy(p, "\xc2\xb7", 2); p += 2; }       // ·
        }
        memcpy(p, "\033[0m  ----\033[K", 14); p += 14;
    } else {
        int filled = std::max(0, std::min(W, (int)(score * W)));
        static const char *esc[]  = { "\033[32m", "\033[1;31m", "\033[2m", "\033[33m" };
        static const int esc_len[] = { 5, 7, 4, 5 };
        int color = -1;

        for (int i = 0; i < W; i++) {
            int want;
            if      (i == thr)              want = 3;  // yellow
            else if (i < filled && i < thr) want = 0;  // green
            else if (i < filled)            want = 1;  // red
            else                            want = 2;  // dim

            if (want != color) {
                memcpy(p, esc[want], esc_len[want]);
                p += esc_len[want];
                color = want;
            }

            if (i == thr)        { memcpy(p, "\xe2\x94\x82", 3); p += 3; }  // │
            else if (i < filled) { memcpy(p, "\xe2\x96\x88", 3); p += 3; }  // █
            else                 { memcpy(p, "\xc2\xb7", 2); p += 2; }       // ·
        }

        p += std::snprintf(p, 64, "\033[0m  %4.2f  %3dms\033[K", score, ms);
    }

    fwrite(buf, 1, p - buf, stderr);
}

// ---- Signal handling ----

static volatile bool g_running = true;
static void signal_handler(int) { g_running = false; }

// ---- CMVN (Cepstral Mean and Variance Normalization) ----
// Removes per-dimension DC offset from encoder features.
// After CMVN, only relative variation between frames matters — the shared
// "average speech" direction is subtracted out, dramatically improving discrimination.

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

// ---- Template matching with subsequence DTW ----

struct Template {
    std::vector<float> data;  // [n_frames * WHISPER_DIM], L2-normalized per frame
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
        if (n <= 0 || n > 10000) {
            std::cerr << "Invalid template count: " << n << std::endl;
            return false;
        }

        items.resize(n);
        for (int i = 0; i < n; i++) {
            int32_t nf;
            f.read(reinterpret_cast<char *>(&nf), sizeof(int32_t));
            if (nf <= 0 || nf > 10000) {
                std::cerr << "Invalid frame count for template " << i << ": " << nf << std::endl;
                return false;
            }
            items[i].n_frames = nf;
            items[i].data.resize(nf * WHISPER_DIM);
            f.read(reinterpret_cast<char *>(items[i].data.data()), nf * WHISPER_DIM * sizeof(float));
        }

        if (f.fail()) {
            std::cerr << "Failed to read template data" << std::endl;
            return false;
        }

        int total_frames = 0;
        for (const auto &t : items) total_frames += t.n_frames;
        std::cout << "Loaded " << n << " templates (" << total_frames << " total frames)" << std::endl;
        return true;
    }

    // Pre-compute inverse norms for input frames (call once per cycle).
    static void compute_inv_norms(const float *input, int n_frames, std::vector<float> &inv_norms) {
        inv_norms.resize(n_frames);
        for (int i = 0; i < n_frames; i++) {
            float na = 0;
            const float *a = input + i * WHISPER_DIM;
            for (int k = 0; k < WHISPER_DIM; k++) na += a[k] * a[k];
            inv_norms[i] = na > 0 ? 1.0f / std::sqrt(na) : 0;
        }
    }

    // Dot product with pre-normalized template vector, scaled by pre-computed input inv norm.
    static float cosine_dot(const float *a, const float *b_unit, float inv_norm_a) {
        float dot = 0;
        for (int i = 0; i < WHISPER_DIM; i++) dot += a[i] * b_unit[i];
        return dot * inv_norm_a;
    }

    // Subsequence DTW: find the best-matching region within the input for the template.
    // The template must be fully matched, but the match can start/end anywhere in input.
    // Uses two-row sliding window instead of full matrix.
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

    // Best match across all templates. Returns similarity (1 - normalized_dtw_cost).
    float match(const float *input, int n_frames,
                std::vector<float> &inv_norms,
                std::vector<float> &row_a, std::vector<float> &row_b) const {
        compute_inv_norms(input, n_frames, inv_norms);
        float best_sim = -1.0f;
        for (const auto &tmpl : items) {
            float cost = subdtw(input, n_frames, tmpl, inv_norms, row_a, row_b);
            float sim = 1.0f - cost;
            if (sim > best_sim) best_sim = sim;
        }
        return best_sim;
    }
};

// ---- Main ----

static void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  -m <path>   Whisper model (default: models/ggml-tiny.bin)\n"
              << "  -e <path>   Enrolled templates (default: models/templates.bin)\n"
              << "  -t <float>  DTW similarity threshold (default: " << DEFAULT_THRESHOLD << ")\n"
              << "  -h          Show this help\n";
}

int main(int argc, char **argv) {
    std::string whisper_path   = "models/ggml-tiny.bin";
    std::string templates_path = "models/templates.bin";
    float threshold = DEFAULT_THRESHOLD;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) whisper_path = argv[++i];
        else if (arg == "-e" && i + 1 < argc) templates_path = argv[++i];
        else if (arg == "-t" && i + 1 < argc) threshold = std::stof(argv[++i]);
        else { print_usage(argv[0]); return 1; }
    }

    std::signal(SIGINT, signal_handler);

    // Load whisper
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    struct whisper_context *ctx = whisper_init_from_file_with_params(
        whisper_path.c_str(), cparams);
    if (!ctx) {
        std::cerr << "Failed to load whisper: " << whisper_path << std::endl;
        return 1;
    }
    std::cout << "Whisper loaded: " << whisper_path << std::endl;

    // Load templates
    Templates templates;
    if (!templates.load(templates_path)) {
        whisper_free(ctx);
        return 1;
    }

    // Init audio
    const int buffer_samples = static_cast<int>(BUFFER_SEC * SAMPLE_RATE);
    auto audio = std::make_shared<audio_async>(static_cast<int>(BUFFER_SEC * 1000));
    audio->init(-1, SAMPLE_RATE);
    audio->resume();
    audio->clear();

    std::cout << "Listening... (threshold=" << threshold << ", Ctrl+C to stop)" << std::endl;

    const int max_encoder_frames = (buffer_samples / MEL_HOP + 1) / CONV_STRIDE + 1;
    const int max_encoder_floats = max_encoder_frames * WHISPER_DIM;
    std::vector<float> encoder_output(max_encoder_floats);
    std::vector<float> inv_norms;             // pre-computed input frame inverse norms
    std::vector<float> dtw_row_a, dtw_row_b;  // reusable DTW row buffers
    std::vector<float> pcm_buffer;
    pcm_buffer.reserve(buffer_samples);

    int refractory = 0;
    const int refractory_cycles = 10;  // 10 * SLIDE_MS = 2 seconds

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(SLIDE_MS));

        if (refractory > 0) {
            refractory--;
            audio->clear();
            continue;
        }

        audio->get(static_cast<int>(BUFFER_SEC * 1000), pcm_buffer);
        if ((int)pcm_buffer.size() < buffer_samples / 2) continue;

        // Energy-based VAD
        float energy = 0;
        for (auto s : pcm_buffer) energy += s * s;
        energy /= pcm_buffer.size();
        if (energy < 1e-6f) {
            render_bar(0.0f, threshold, 0, true);
            continue;
        }

        // Whisper mel + encode (variable-length: only encode actual audio frames)
        auto t0 = std::chrono::steady_clock::now();
        if (whisper_pcm_to_mel(ctx, pcm_buffer.data(), pcm_buffer.size(), 1) != 0) continue;

        int mel_frames = whisper_n_len(ctx);
        int actual_frames = (mel_frames + 1) / CONV_STRIDE;
        if (actual_frames <= 0) actual_frames = 1;
        whisper_set_audio_ctx(ctx, actual_frames);

        if (whisper_encode(ctx, 0, 1) != 0) continue;

        int n_floats = whisper_encoder_output(ctx, encoder_output.data(), max_encoder_floats);
        if (n_floats <= 0) continue;
        int n_enc_frames = n_floats / WHISPER_DIM;

        float *enc_ptr = encoder_output.data() + ONSET_SKIP * WHISPER_DIM;
        n_enc_frames -= ONSET_SKIP;
        if (n_enc_frames <= 0) continue;
        apply_cmvn(enc_ptr, n_enc_frames);

        // DTW match against templates
        float score = templates.match(enc_ptr, n_enc_frames, inv_norms, dtw_row_a, dtw_row_b);
        auto t1 = std::chrono::steady_clock::now();
        int total_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        render_bar(score, threshold, total_ms, false);

        if (score >= threshold) {
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            char time_buf[32];
            std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", std::localtime(&t));
            std::cerr << "\r\033[K" << std::flush;
            std::cout << "  [" << time_buf << "] DETECTED  sim=" << score << std::endl;

            audio->clear();
            refractory = refractory_cycles;
        }
    }

    whisper_free(ctx);
    std::cout << "\nStopped." << std::endl;
    return 0;
}
