/**
 * vad.cpp - Silero VAD (v5, 16kHz) implementation.
 *
 * Pure C implementation (no ggml dependency). The model is small enough
 * (~300K params, ~1M multiply-adds per frame) that SIMD isn't needed.
 */

#include "vad.h"

#include <algorithm>
#include <cstdio>
#include <fstream>

// ---- Weight loading ----

bool SileroVad::load(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "vad: cannot open %s\n", path.c_str());
        return false;
    }

    char magic[4];
    uint32_t version, n_tensors;
    f.read(magic, 4);
    f.read((char *)&version, 4);
    f.read((char *)&n_tensors, 4);
    if (memcmp(magic, "SVAD", 4) != 0 || version != 1) {
        fprintf(stderr, "vad: invalid file format\n");
        return false;
    }

    struct TensorInfo {
        std::string name;
        size_t offset;
        size_t size;
    };
    std::vector<TensorInfo> tensors;

    // First pass: compute total size
    size_t total_floats = 0;
    auto start_pos = f.tellg();
    for (uint32_t i = 0; i < n_tensors; i++) {
        uint32_t name_len;
        f.read((char *)&name_len, 4);
        std::string name(name_len, '\0');
        f.read(name.data(), name_len);
        uint32_t n_dims;
        f.read((char *)&n_dims, 4);
        size_t sz = 1;
        for (uint32_t d = 0; d < n_dims; d++) {
            uint32_t dim;
            f.read((char *)&dim, 4);
            sz *= dim;
        }
        tensors.push_back({name, total_floats, sz});
        total_floats += sz;
        f.seekg(sz * sizeof(float), std::ios::cur);
    }

    // Second pass: read all weight data
    weights.resize(total_floats);
    f.seekg(start_pos);
    for (auto &t : tensors) {
        uint32_t name_len;
        f.read((char *)&name_len, 4);
        f.seekg(name_len, std::ios::cur);
        uint32_t n_dims;
        f.read((char *)&n_dims, 4);
        f.seekg(n_dims * 4, std::ios::cur);
        f.read((char *)(weights.data() + t.offset), t.size * sizeof(float));
    }

    // Map tensor pointers
    for (const auto &t : tensors) {
        const float *ptr = weights.data() + t.offset;
        if      (t.name == "stft.forward_basis_buffer")       stft_basis = ptr;
        else if (t.name == "encoder.0.reparam_conv.weight")   enc_w[0] = ptr;
        else if (t.name == "encoder.0.reparam_conv.bias")     enc_b[0] = ptr;
        else if (t.name == "encoder.1.reparam_conv.weight")   enc_w[1] = ptr;
        else if (t.name == "encoder.1.reparam_conv.bias")     enc_b[1] = ptr;
        else if (t.name == "encoder.2.reparam_conv.weight")   enc_w[2] = ptr;
        else if (t.name == "encoder.2.reparam_conv.bias")     enc_b[2] = ptr;
        else if (t.name == "encoder.3.reparam_conv.weight")   enc_w[3] = ptr;
        else if (t.name == "encoder.3.reparam_conv.bias")     enc_b[3] = ptr;
        else if (t.name == "decoder.rnn.weight_ih")           rnn_wih = ptr;
        else if (t.name == "decoder.rnn.weight_hh")           rnn_whh = ptr;
        else if (t.name == "decoder.rnn.bias_ih")             rnn_bih = ptr;
        else if (t.name == "decoder.rnn.bias_hh")             rnn_bhh = ptr;
        else if (t.name == "decoder.decoder.2.weight")        dec_w = ptr;
        else if (t.name == "decoder.decoder.2.bias")          dec_b = *ptr;
    }

    if (!stft_basis || !rnn_wih || !rnn_whh || !rnn_bih || !rnn_bhh || !dec_w) {
        fprintf(stderr, "vad: missing weight tensors\n");
        return false;
    }
    for (int i = 0; i < 4; i++) {
        if (!enc_w[i] || !enc_b[i]) {
            fprintf(stderr, "vad: missing encoder.%d weights\n", i);
            return false;
        }
    }

    reset();
    return true;
}

void SileroVad::reset() {
    memset(h, 0, sizeof(h));
    memset(c, 0, sizeof(c));
    memset(context, 0, sizeof(context));
}

// ---- Forward pass helpers ----

// Conv1D + ReLU. Kernel size = 3, padding = 1.
// Weight: [C_out][C_in][3], Input: [C_in][T_in], Output: [C_out][T_out]
static void conv1d_relu(
    float *output, int t_out,
    const float *input, int c_in, int t_in,
    const float *weight, const float *bias,
    int c_out, int stride)
{
    for (int oc = 0; oc < c_out; oc++) {
        for (int t = 0; t < t_out; t++) {
            float sum = bias[oc];
            for (int ic = 0; ic < c_in; ic++) {
                for (int k = 0; k < 3; k++) {
                    int pos = t * stride - 1 + k;
                    if (pos >= 0 && pos < t_in) {
                        sum += weight[(oc * c_in + ic) * 3 + k] * input[ic * t_in + pos];
                    }
                }
            }
            output[oc * t_out + t] = sum > 0 ? sum : 0;  // ReLU
        }
    }
}

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

float SileroVad::process(const float *pcm_512) {
    // 1. Build input: context (64) + new audio (512) = 576 samples
    float input[INPUT_SAMPLES];
    memcpy(input, context, CONTEXT_SIZE * sizeof(float));
    memcpy(input + CONTEXT_SIZE, pcm_512, CHUNK_SAMPLES * sizeof(float));
    memcpy(context, pcm_512 + CHUNK_SAMPLES - CONTEXT_SIZE, CONTEXT_SIZE * sizeof(float));

    // 2. Reflect-pad: 0 on left, 64 on right
    float padded[STFT_PADDED];
    memcpy(padded, input, INPUT_SAMPLES * sizeof(float));
    for (int i = 0; i < STFT_PAD_RIGHT; i++)
        padded[INPUT_SAMPLES + i] = input[INPUT_SAMPLES - 2 - i];

    // 3. STFT: Conv1D (258 filters, kernel=256, stride=128) → magnitude
    //    Output: (258, 4) → magnitude (129, 4)
    float stft_raw[258 * STFT_FRAMES];
    for (int oc = 0; oc < 258; oc++) {
        const float *kernel = stft_basis + oc * STFT_N_FFT;
        for (int t = 0; t < STFT_FRAMES; t++) {
            const float *frame = padded + t * STFT_HOP;
            float sum = 0;
            for (int k = 0; k < STFT_N_FFT; k++)
                sum += kernel[k] * frame[k];
            stft_raw[oc * STFT_FRAMES + t] = sum;
        }
    }

    float mag[STFT_BINS * STFT_FRAMES];
    for (int bin = 0; bin < STFT_BINS; bin++) {
        for (int t = 0; t < STFT_FRAMES; t++) {
            float re = stft_raw[bin * STFT_FRAMES + t];
            float im = stft_raw[(bin + STFT_BINS) * STFT_FRAMES + t];
            mag[bin * STFT_FRAMES + t] = sqrtf(re * re + im * im);
        }
    }

    // 4. Encoder: 4× Conv1D+ReLU
    // STFT_FRAMES=4 → enc0: (128,4) → enc1: (64,2) → enc2: (64,1) → enc3: (128,1)
    float enc0[128 * STFT_FRAMES];
    conv1d_relu(enc0, STFT_FRAMES, mag, 129, STFT_FRAMES,
                enc_w[0], enc_b[0], 128, 1);

    int t1 = (STFT_FRAMES + 2 * 1 - 3) / 2 + 1;  // = 2
    float enc1[64 * t1];
    conv1d_relu(enc1, t1, enc0, 128, STFT_FRAMES,
                enc_w[1], enc_b[1], 64, 2);

    int t2 = (t1 + 2 * 1 - 3) / 2 + 1;  // = 1
    float enc2[64 * t2];
    conv1d_relu(enc2, t2, enc1, 64, t1,
                enc_w[2], enc_b[2], 64, 2);

    int t3 = (t2 + 2 * 1 - 3) / 1 + 1;  // = 1
    float enc3[128 * t3];
    conv1d_relu(enc3, t3, enc2, 64, t2,
                enc_w[3], enc_b[3], 128, 1);

    // 5. LSTM: process t3 timesteps, save per-timestep hidden output
    float h_out[t3 * LSTM_HIDDEN];

    for (int t = 0; t < t3; t++) {
        // Gather x_t from enc3 (layout: [channel][time])
        float x_t[LSTM_HIDDEN];
        for (int ch = 0; ch < LSTM_HIDDEN; ch++)
            x_t[ch] = enc3[ch * t3 + t];

        // gates = W_ih @ x_t + b_ih + W_hh @ h + b_hh  → (512,)
        float gates[4 * LSTM_HIDDEN];
        for (int i = 0; i < 4 * LSTM_HIDDEN; i++) {
            float sum_x = rnn_bih[i];
            float sum_h = rnn_bhh[i];
            for (int j = 0; j < LSTM_HIDDEN; j++) {
                sum_x += rnn_wih[i * LSTM_HIDDEN + j] * x_t[j];
                sum_h += rnn_whh[i * LSTM_HIDDEN + j] * h[j];
            }
            gates[i] = sum_x + sum_h;
        }

        // Gate activations (PyTorch order: i, f, g, o)
        for (int i = 0; i < LSTM_HIDDEN; i++) {
            float ig = sigmoidf(gates[i]);
            float fg = sigmoidf(gates[LSTM_HIDDEN + i]);
            float gg = tanhf(gates[2 * LSTM_HIDDEN + i]);
            float og = sigmoidf(gates[3 * LSTM_HIDDEN + i]);

            c[i] = fg * c[i] + ig * gg;
            h[i] = og * tanhf(c[i]);
        }

        memcpy(h_out + t * LSTM_HIDDEN, h, LSTM_HIDDEN * sizeof(float));
    }

    // 6. Decoder: ReLU → Conv1D(128→1, k=1) → sigmoid → mean
    float prob_sum = 0;
    for (int t = 0; t < t3; t++) {
        float *ht = h_out + t * LSTM_HIDDEN;
        float logit = dec_b;
        for (int i = 0; i < LSTM_HIDDEN; i++) {
            float val = ht[i] > 0 ? ht[i] : 0;  // ReLU
            logit += dec_w[i] * val;
        }
        prob_sum += sigmoidf(logit);
    }

    return prob_sum / t3;
}
