/**
 * detect.cpp - Shared detection pipeline implementation.
 */

#include "detect.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

// ---- Templates ----

bool Templates::load(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    // New format: "JWTL" magic + 16-byte model MD5 + model name, then template data.
    // Legacy format: starts directly with int32 n_templates.
    char magic[4];
    f.read(magic, 4);
    if (memcmp(magic, "JWTL", 4) == 0) {
        f.read(reinterpret_cast<char *>(model_hash), 16);
        int32_t name_len;
        f.read(reinterpret_cast<char *>(&name_len), sizeof(int32_t));
        if (name_len > 0 && name_len < 256) {
            model_name.resize(name_len);
            f.read(model_name.data(), name_len);
        }
    } else {
        memset(model_hash, 0, 16);
        f.seekg(0);
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
        items[i].data.resize(nf * JARVIS_DIM);
        f.read(reinterpret_cast<char *>(items[i].data.data()), nf * JARVIS_DIM * sizeof(float));
    }
    return !f.fail();
}

void Templates::compute_inv_norms(const float *input, int n_frames, std::vector<float> &inv_norms) {
    inv_norms.resize(n_frames);
    for (int i = 0; i < n_frames; i++) {
        float na = 0;
        const float *a = input + i * JARVIS_DIM;
        for (int k = 0; k < JARVIS_DIM; k++) na += a[k] * a[k];
        inv_norms[i] = na > 0 ? 1.0f / std::sqrt(na) : 0;
    }
}

static float cosine_dot(const float *a, const float *b_unit, float inv_norm_a) {
    float dot = 0;
    for (int i = 0; i < JARVIS_DIM; i++) dot += a[i] * b_unit[i];
    return dot * inv_norm_a;
}

static float subdtw(const float *input, int n_input,
                    const Template &tmpl,
                    const std::vector<float> &inv_norms,
                    std::vector<float> &prev_row,
                    std::vector<float> &curr_row,
                    int *best_end_out = nullptr) {
    int n_tmpl = tmpl.n_frames;
    prev_row.resize(n_tmpl + 1);
    curr_row.resize(n_tmpl + 1);
    std::fill(prev_row.begin(), prev_row.end(), 1e30f);
    prev_row[0] = 0.0f;
    float best = 1e30f;
    int best_end = 0;

    int anchor_end = n_tmpl - JARVIS_ANCHOR_FRAMES;

    for (int i = 1; i <= n_input; i++) {
        curr_row[0] = 0.0f;
        for (int j = 1; j <= n_tmpl; j++) {
            float c = 1.0f - cosine_dot(input + (i - 1) * JARVIS_DIM, tmpl.frame(j - 1), inv_norms[i - 1]);
            if (j <= JARVIS_ANCHOR_FRAMES || j > anchor_end)
                c *= JARVIS_ANCHOR_WEIGHT;
            float diag = prev_row[j - 1];
            float ins  = prev_row[j] + JARVIS_STEP_PENALTY;
            float del  = curr_row[j - 1] + JARVIS_STEP_PENALTY;
            float best_prev = diag;
            if (ins < best_prev) best_prev = ins;
            if (del < best_prev) best_prev = del;
            curr_row[j] = c + best_prev;
        }
        if (curr_row[n_tmpl] < best) { best = curr_row[n_tmpl]; best_end = i; }
        std::swap(prev_row, curr_row);
    }
    if (best_end_out) *best_end_out = best_end;
    return best / n_tmpl;
}

float Templates::match(const float *input, int n_frames,
                       const std::vector<float> &inv_norms,
                       std::vector<float> &row_a, std::vector<float> &row_b,
                       int *end_frame) const {
    float best_sim = -1.0f;
    for (const auto &tmpl : items) {
        int end = 0;
        float cost = subdtw(input, n_frames, tmpl, inv_norms, row_a, row_b, &end);
        float sim = 1.0f - cost;
        if (sim > best_sim) { best_sim = sim; if (end_frame) *end_frame = end; }
    }
    return best_sim;
}

// ---- CMVN ----

static void apply_cmvn(float *frames, int n_frames) {
    float mean[JARVIS_DIM] = {};
    float var[JARVIS_DIM] = {};

    float inv_n = 1.0f / n_frames;
    for (int t = 0; t < n_frames; t++) {
        const float *f = frames + t * JARVIS_DIM;
        for (int d = 0; d < JARVIS_DIM; d++) mean[d] += f[d];
    }
    for (int d = 0; d < JARVIS_DIM; d++) mean[d] *= inv_n;

    for (int t = 0; t < n_frames; t++) {
        const float *f = frames + t * JARVIS_DIM;
        for (int d = 0; d < JARVIS_DIM; d++) {
            float diff = f[d] - mean[d];
            var[d] += diff * diff;
        }
    }
    for (int d = 0; d < JARVIS_DIM; d++) var[d] = 1.0f / (std::sqrt(var[d] * inv_n) + 1e-10f);

    for (int t = 0; t < n_frames; t++) {
        float *f = frames + t * JARVIS_DIM;
        for (int d = 0; d < JARVIS_DIM; d++) f[d] = (f[d] - mean[d]) * var[d];
    }
}

// ---- Detection pipeline ----

DetectResult detect_once(
    whisper_context *ctx,
    const std::vector<LoadedKeyword> &keywords,
    const float *pcm, int n_samples,
    DetectScratch &scratch,
    int skip_frames)
{
    DetectResult result;
    auto t0 = std::chrono::steady_clock::now();

    // Zero-pad short buffers
    const float *input = pcm;
    if (n_samples < JARVIS_BUFFER_SAMPLES) {
        scratch.pcm_padded.assign(pcm, pcm + n_samples);
        scratch.pcm_padded.resize(JARVIS_BUFFER_SAMPLES, 0.0f);
        input = scratch.pcm_padded.data();
        n_samples = JARVIS_BUFFER_SAMPLES;
    }

    if (whisper_pcm_to_mel(ctx, input, n_samples, 1) != 0) return result;

    int mel_frames = whisper_n_len(ctx);
    int actual_frames = (mel_frames + 1) / JARVIS_CONV_STRIDE;
    if (actual_frames <= 0) actual_frames = 1;
    whisper_set_audio_ctx(ctx, actual_frames);

    if (whisper_encode(ctx, 0, 1) != 0) return result;

    int max_floats = (int)scratch.encoder_output.size();
    int n_floats = whisper_encoder_output(ctx, scratch.encoder_output.data(), max_floats);
    if (n_floats <= 0) return result;

    int n_enc_frames = n_floats / JARVIS_DIM;
    int total_skip = JARVIS_ONSET_SKIP + skip_frames;
    float *enc_ptr = scratch.encoder_output.data() + total_skip * JARVIS_DIM;
    n_enc_frames -= total_skip;
    if (n_enc_frames <= 0) return result;

    apply_cmvn(enc_ptr, n_enc_frames);
    Templates::compute_inv_norms(enc_ptr, n_enc_frames, scratch.inv_norms);

    for (int k = 0; k < (int)keywords.size(); k++) {
        int end = 0;
        float score = keywords[k].templates.match(
            enc_ptr, n_enc_frames, scratch.inv_norms,
            scratch.dtw_row_a, scratch.dtw_row_b, &end);
        if (score > result.best_score) {
            result.best_score = score;
            result.best_keyword = k;
        }
        if (result.keyword_index < 0 && score >= keywords[k].threshold) {
            result.keyword_index = k;
            result.score = score;
            result.end_frame = end;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    result.elapsed_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return result;
}


// ---- Path helpers ----

std::string cache_dir() {
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    return std::string(home) + "/.cache/jarvis";
}

std::string model_tag(const std::string &model_path) {
    std::string stem = model_path;
    auto slash = stem.rfind('/');
    if (slash != std::string::npos) stem = stem.substr(slash + 1);
    auto dot = stem.rfind('.');
    if (dot != std::string::npos) stem = stem.substr(0, dot);
    if (stem.rfind("ggml-", 0) == 0) stem = stem.substr(5);
    return stem;
}

std::string template_path(const std::string &keyword, const std::string &tag) {
    return cache_dir() + "/templates/" + keyword + "." + tag + ".bin";
}

std::string diagnose_path(const std::string &path) {
    struct stat st;
    if (lstat(path.c_str(), &st) != 0)
        return path + ": file not found";
    if (S_ISLNK(st.st_mode)) {
        if (stat(path.c_str(), &st) != 0)
            return path + ": broken symlink";
    }
    if (st.st_size == 0)
        return path + ": file is empty";
    return path + ": file exists but failed to parse";
}
