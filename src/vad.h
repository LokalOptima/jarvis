/**
 * vad.h - Silero VAD (v5, 16kHz branch) ported to plain C.
 *
 * Architecture:
 *   Input:  512 float32 PCM samples at 16kHz (32ms)
 *   STFT:   reflect pad → conv1d(258, 256, stride=128) → magnitude spectrum
 *   Encoder: 4× Conv1D+ReLU (129→128→64→64→128)
 *   LSTM:   input=128, hidden=128 (manual gate computation)
 *   Output: ReLU → Conv1D(128→1) → sigmoid → mean → speech probability
 *
 *   State: 64 samples audio context + 128 LSTM h + 128 LSTM c
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

struct SileroVad {
    // Configuration
    static constexpr int CHUNK_SAMPLES  = 512;
    static constexpr int CONTEXT_SIZE   = 64;
    static constexpr int INPUT_SAMPLES  = CHUNK_SAMPLES + CONTEXT_SIZE;  // 576
    static constexpr int STFT_N_FFT     = 256;
    static constexpr int STFT_HOP       = 128;
    static constexpr int STFT_PAD_RIGHT = 64;  // reflect pad only on right
    static constexpr int STFT_PADDED    = INPUT_SAMPLES + STFT_PAD_RIGHT;  // 640
    static constexpr int STFT_FRAMES    = (STFT_PADDED - STFT_N_FFT) / STFT_HOP + 1;  // 4
    static constexpr int STFT_BINS      = STFT_N_FFT / 2 + 1;  // 129
    static constexpr int LSTM_HIDDEN    = 128;

    // Weight storage (flat arrays, loaded from binary file)
    std::vector<float> weights;  // single allocation holding all parameters

    // Pointers into weights (set during load)
    const float *stft_basis = nullptr;   // [258][256]
    const float *enc_w[4] = {};          // conv weights [Cout][Cin][K=3]
    const float *enc_b[4] = {};          // conv biases [Cout]
    int enc_cout[4] = {128, 64, 64, 128};
    int enc_cin[4]  = {129, 128, 64, 64};
    int enc_stride[4] = {1, 2, 2, 1};
    const float *rnn_wih = nullptr;      // [512][128]
    const float *rnn_whh = nullptr;      // [512][128]
    const float *rnn_bih = nullptr;      // [512]
    const float *rnn_bhh = nullptr;      // [512]
    const float *dec_w = nullptr;        // [128]
    float dec_b = 0;                     // scalar

    // Persistent state
    float h[LSTM_HIDDEN] = {};
    float c[LSTM_HIDDEN] = {};
    float context[CONTEXT_SIZE] = {};

    bool load(const std::string &path);
    float process(const float *pcm_512);  // feed 512 samples, returns P(speech)
    void reset();
};
