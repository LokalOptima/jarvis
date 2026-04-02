/**
 * jarvis - Personalized wake word detector using Whisper Tiny encoder.
 *
 * Pipeline:
 *   SDL2 mic capture → ring buffer → whisper mel + encode
 *   → DTW frame-level matching against enrolled templates → detect
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
#define DEFAULT_THRESHOLD 0.80f  // DTW similarity threshold

// ---- Signal handling ----

static volatile bool g_running = true;
static void signal_handler(int) { g_running = false; }

// ---- Template matching with DTW ----

struct Template {
    std::vector<float> data;  // [n_frames * WHISPER_DIM] flattened
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

        // Format: int32 n_templates, then per template: int32 n_frames, float[n_frames * 384]
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

    // Cosine similarity between two 384-dim vectors
    static float cosine_sim(const float *a, const float *b) {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < WHISPER_DIM; i++) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        float denom = std::sqrt(na) * std::sqrt(nb);
        return denom > 0 ? dot / denom : 0;
    }

    // Subsequence DTW: find the best-matching region within the input for the template.
    // The template must be fully matched, but the match can start/end anywhere in input.
    // Returns normalized cost (lower = better match).
    static float subdtw(const float *input, int n_input,
                        const Template &tmpl, std::vector<float> &cost_matrix) {
        int n_tmpl = tmpl.n_frames;
        int w = n_tmpl + 1;
        cost_matrix.assign((n_input + 1) * w, 1e30f);

        // First column = 0: match can start at any input frame
        for (int i = 0; i <= n_input; i++)
            cost_matrix[i * w + 0] = 0.0f;

        for (int i = 1; i <= n_input; i++) {
            for (int j = 1; j <= n_tmpl; j++) {
                float c = 1.0f - cosine_sim(input + (i - 1) * WHISPER_DIM, tmpl.frame(j - 1));
                float prev = std::min({
                    cost_matrix[(i - 1) * w + j],
                    cost_matrix[i * w + (j - 1)],
                    cost_matrix[(i - 1) * w + (j - 1)]
                });
                cost_matrix[i * w + j] = c + prev;
            }
        }

        // Best match: minimum of last column (template fully consumed, at any input position)
        float best = 1e30f;
        for (int i = 1; i <= n_input; i++) {
            float v = cost_matrix[i * w + n_tmpl];
            if (v < best) best = v;
        }
        return best / n_tmpl;
    }

    // Best match across all templates. Returns similarity (1 - normalized_dtw_cost).
    float match(const float *input, int n_frames, std::vector<float> &cost_buf) const {
        float best_sim = -1.0f;
        for (const auto &tmpl : items) {
            float cost = subdtw(input, n_frames, tmpl, cost_buf);
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
              << "  -d          Debug mode (print scores)\n"
              << "  -h          Show this help\n";
}

int main(int argc, char **argv) {
    std::string whisper_path   = "models/ggml-tiny.bin";
    std::string templates_path = "models/templates.bin";
    float threshold = DEFAULT_THRESHOLD;
    bool debug = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) whisper_path = argv[++i];
        else if (arg == "-e" && i + 1 < argc) templates_path = argv[++i];
        else if (arg == "-t" && i + 1 < argc) threshold = std::stof(argv[++i]);
        else if (arg == "-d") debug = true;
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

    const int max_encoder_floats = 1500 * WHISPER_DIM;
    std::vector<float> encoder_output(max_encoder_floats);
    std::vector<float> dtw_cost_buf;  // reusable buffer for DTW cost matrix
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
            if (debug) std::cerr << "\r  ----  [                    ]        " << std::flush;
            continue;
        }

        // Whisper mel + encode (variable-length: only encode actual audio frames)
        auto t0 = std::chrono::steady_clock::now();
        if (whisper_pcm_to_mel(ctx, pcm_buffer.data(), pcm_buffer.size(), 1) != 0) continue;

        // Set encoder to process only the frames we need (not the full 1500)
        int mel_frames = whisper_n_len(ctx);
        int actual_frames = (mel_frames + 1) / 2;  // ceil(mel_frames / 2) — conv stride is 2
        if (actual_frames <= 0) actual_frames = 1;
        whisper_set_audio_ctx(ctx, actual_frames);

        auto t1 = std::chrono::steady_clock::now();
        if (whisper_encode(ctx, 0, 1) != 0) continue;
        auto t2 = std::chrono::steady_clock::now();

        int n_floats = whisper_encoder_output(ctx, encoder_output.data(), max_encoder_floats);
        if (n_floats <= 0) continue;
        int n_enc_frames = n_floats / WHISPER_DIM;

        // DTW match against templates
        float score = templates.match(encoder_output.data(), n_enc_frames, dtw_cost_buf);
        auto t3 = std::chrono::steady_clock::now();

        if (debug) {
            auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count();
            // Score bar: 20 chars wide, filled proportional to score
            int bar_len = 20;
            int filled = std::max(0, std::min(bar_len, static_cast<int>(score * bar_len)));
            char bar[21];
            for (int i = 0; i < bar_len; i++) bar[i] = i < filled ? '|' : ' ';
            bar[bar_len] = '\0';
            char line[128];
            std::snprintf(line, sizeof(line), "\r  %4.2f [%s] %3ldms  ", score, bar, total_ms);
            std::cerr << line << std::flush;
        }

        if (score >= threshold) {
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            char time_buf[32];
            std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", std::localtime(&t));
            std::cerr << "\r\033[K" << std::flush;  // clear debug line
            std::cout << "  [" << time_buf << "] DETECTED  sim=" << score << std::endl;

            audio->clear();
            refractory = refractory_cycles;
        }
    }

    whisper_free(ctx);
    std::cout << "\nStopped." << std::endl;
    return 0;
}
