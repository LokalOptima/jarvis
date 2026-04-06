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
#include <unistd.h>

// ---- Templates ----

bool Templates::load(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

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
            float c = 1.0f - cosine_dot(input + (i - 1) * JARVIS_DIM, tmpl.frame(j - 1), inv_norms[i - 1]);
            float diag = prev_row[j - 1];
            float ins  = prev_row[j] + JARVIS_STEP_PENALTY;
            float del  = curr_row[j - 1] + JARVIS_STEP_PENALTY;
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

float Templates::match(const float *input, int n_frames,
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
    DetectScratch &scratch)
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
    float *enc_ptr = scratch.encoder_output.data() + JARVIS_ONSET_SKIP * JARVIS_DIM;
    n_enc_frames -= JARVIS_ONSET_SKIP;
    if (n_enc_frames <= 0) return result;

    apply_cmvn(enc_ptr, n_enc_frames);
    Templates::compute_inv_norms(enc_ptr, n_enc_frames, scratch.inv_norms);

    for (int k = 0; k < (int)keywords.size(); k++) {
        float score = keywords[k].templates.match(
            enc_ptr, n_enc_frames, scratch.inv_norms, scratch.dtw_row_a, scratch.dtw_row_b);
        if (score > result.best_score) {
            result.best_score = score;
            result.best_keyword = k;
        }
        if (result.keyword_index < 0 && score >= keywords[k].threshold) {
            result.keyword_index = k;
            result.score = score;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    result.elapsed_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return result;
}

// ---- Terminal bar ----

static int g_bar_row = 0;  // 0 = not pinned (legacy \r mode)

void setup_scroll_region() {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) < 0) return;
    g_bar_row = ws.ws_row;
    // APT-style: newline, save cursor (DECSC), set scroll region, restore (DECRC), up 1
    fprintf(stdout, "\n\0337\033[1;%dr\0338\033[1A", ws.ws_row - 1);
    fflush(stdout);
}

void teardown_scroll_region() {
    if (g_bar_row <= 0) return;
    struct winsize ws;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws);
    // Restore full scroll region, jump to bar row, clear it, restore cursor
    fprintf(stdout, "\0337\033[1;%dr\033[%d;1H\033[2K\0338",
            ws.ws_row, ws.ws_row);
    fflush(stdout);
    g_bar_row = 0;
}

void render_bar(const char *name, float score, float threshold, int ms, bool silent) {
    static char buf[768];
    char *p = buf;

    constexpr int NAME_W = 14;
    constexpr int W = 36;
    int thr = std::max(0, std::min(W - 1, (int)(threshold * W)));

    if (g_bar_row > 0) {
        // DECSC + jump to pinned row
        p += std::snprintf(p, 32, "\0337\033[%d;1H", g_bar_row);
    } else {
        *p++ = '\r';
    }
    *p++ = ' '; *p++ = ' ';

    int nlen = (int)strlen(name);
    int copy = std::min(nlen, NAME_W);
    memcpy(p, name, copy); p += copy;
    for (int i = copy; i < NAME_W; i++) *p++ = ' ';

    if (silent) {
        int filled = std::max(0, std::min(W, (int)(score * W)));
        memcpy(p, "\033[2m", 4); p += 4;
        for (int i = 0; i < W; i++) {
            if      (i < filled) { memcpy(p, "\xe2\x96\x88", 3); p += 3; }
            else if (i == thr)   { memcpy(p, "\xe2\x94\x82", 3); p += 3; }
            else                 { memcpy(p, "\xc2\xb7", 2); p += 2; }
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

    if (g_bar_row > 0) {
        // DECRC: restore cursor to scroll region
        *p++ = '\033'; *p++ = '8';
    }

    FILE *out = g_bar_row > 0 ? stdout : stderr;
    fwrite(buf, 1, p - buf, out);
    fflush(out);
}
