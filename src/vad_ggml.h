/**
 * vad_ggml.h - Silero VAD (v5, 16kHz) using ggml ops for SIMD-accelerated inference.
 *
 * Graph is built once during load(), then reused per frame.
 */

#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstring>
#include <string>
#include <vector>

struct SileroVad {
    static constexpr int CHUNK_SAMPLES  = 512;
    static constexpr int CONTEXT_SIZE   = 64;
    static constexpr int INPUT_SAMPLES  = CHUNK_SAMPLES + CONTEXT_SIZE;  // 576
    static constexpr int STFT_N_FFT     = 256;
    static constexpr int STFT_HOP       = 128;
    static constexpr int STFT_PAD_RIGHT = 64;
    static constexpr int STFT_PADDED    = INPUT_SAMPLES + STFT_PAD_RIGHT;  // 640
    static constexpr int STFT_FRAMES    = (STFT_PADDED - STFT_N_FFT) / STFT_HOP + 1;  // 4
    static constexpr int STFT_BINS      = STFT_N_FFT / 2 + 1;  // 129
    static constexpr int LSTM_HIDDEN    = 128;

    // ggml weight context + backend
    ggml_context *ctx_w = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;

    // Pre-built compute graph (reused every frame)
    ggml_context *ctx_compute = nullptr;
    ggml_cgraph  *graph = nullptr;
    ggml_gallocr_t allocr = nullptr;

    // Pinned input/output tensors (pointers into the graph)
    ggml_tensor *t_input = nullptr;
    ggml_tensor *t_h_in = nullptr;
    ggml_tensor *t_c_in = nullptr;
    ggml_tensor *t_prob = nullptr;
    ggml_tensor *t_h_out = nullptr;
    ggml_tensor *t_c_out = nullptr;

    // Persistent state
    float h[LSTM_HIDDEN] = {};
    float c[LSTM_HIDDEN] = {};
    float context[CONTEXT_SIZE] = {};

    bool load(const std::string &path);
    float process(const float *pcm_512);
    void reset();
    ~SileroVad();
};
