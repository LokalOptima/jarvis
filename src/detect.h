/**
 * detect.h - Shared detection pipeline used by both local mode and server.
 *
 * Extracts the core mel → encode → CMVN → DTW pipeline from the listen loop
 * so it can be called from jarvis.cpp (local) and server.cpp (remote).
 */

#pragma once

#include "whisper.h"

#include <string>
#include <vector>

// ---- Configuration ----

static constexpr int   JARVIS_SAMPLE_RATE = WHISPER_SAMPLE_RATE;
static constexpr float JARVIS_BUFFER_SEC  = 2.0f;
static constexpr int   JARVIS_SLIDE_MS    = 200;
static constexpr int   JARVIS_DIM         = 384;
static constexpr int   JARVIS_MEL_HOP     = WHISPER_HOP_LENGTH;
static constexpr int   JARVIS_CONV_STRIDE = 2;
static constexpr int   JARVIS_ONSET_SKIP  = 2;
static constexpr float JARVIS_STEP_PENALTY = 0.1f;
static constexpr int   JARVIS_BUFFER_SAMPLES = (int)(JARVIS_BUFFER_SEC * JARVIS_SAMPLE_RATE);
static constexpr int   JARVIS_SLIDE_SAMPLES  = (JARVIS_SLIDE_MS * JARVIS_SAMPLE_RATE) / 1000;

// ---- Template matching ----

struct Template {
    std::vector<float> data;
    int n_frames = 0;
    const float *frame(int t) const { return data.data() + t * JARVIS_DIM; }
};

struct Templates {
    std::vector<Template> items;
    uint8_t model_hash[16] = {};
    std::string model_name;

    bool load(const std::string &path);

    static void compute_inv_norms(const float *input, int n_frames, std::vector<float> &inv_norms);

    float match(const float *input, int n_frames,
                const std::vector<float> &inv_norms,
                std::vector<float> &row_a, std::vector<float> &row_b) const;
};

// ---- Detection result ----

struct LoadedKeyword {
    std::string name;
    std::string template_path;
    float threshold = 0.35f;
    Templates templates;
};

struct DetectResult {
    int keyword_index = -1;   // -1 = no detection
    float score = 0.0f;
    int best_keyword = 0;     // highest-scoring keyword (even if below threshold)
    float best_score = -1.0f;
    int elapsed_ms = 0;
};

// Scratch buffers pre-allocated by the caller to avoid per-call allocation.
struct DetectScratch {
    std::vector<float> encoder_output;
    std::vector<float> inv_norms;
    std::vector<float> dtw_row_a, dtw_row_b;
    std::vector<float> pcm_padded;

    void init() {
        int max_frames = (JARVIS_BUFFER_SAMPLES / JARVIS_MEL_HOP + 1) / JARVIS_CONV_STRIDE + 1;
        encoder_output.resize(max_frames * JARVIS_DIM);
        pcm_padded.reserve(JARVIS_BUFFER_SAMPLES);
    }
};

// Run one detection cycle: mel → encode → CMVN → DTW.
// pcm/n_samples is the audio buffer (may be shorter than BUFFER_SAMPLES, will be zero-padded).
// Returns detection result. Does NOT check VAD — caller should do that.
DetectResult detect_once(
    whisper_context *ctx,
    const std::vector<LoadedKeyword> &keywords,
    const float *pcm, int n_samples,
    DetectScratch &scratch);

// Terminal display.
void render_bar(const char *name, float score, float threshold, int ms, bool silent);
void render_status(const char *keyword, float score, const char *time_str);
void render_log(const char *msg);      // print a line above the display without clobbering it
void render_header_field(int lines_above_display, const char *label, const char *value);
void render_separator();
void render_clear();  // clean up on exit

// Model/template path helpers.
std::string cache_dir();  // $HOME/.cache/jarvis
std::string model_tag(const std::string &model_path);
std::string template_path(const std::string &keyword, const std::string &tag);
std::string diagnose_path(const std::string &path);
