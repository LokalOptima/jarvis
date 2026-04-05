/**
 * vad_ggml.cpp - Silero VAD using ggml for SIMD-accelerated inference.
 *
 * Graph is built once during load() and reused per frame — only input data
 * changes between calls, eliminating per-frame allocation overhead.
 */

#include "vad_ggml.h"

#include <cmath>
#include <cstdio>
#include <fstream>

// ---- Loading ----

bool SileroVad::load(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "vad: cannot open %s\n", path.c_str()); return false; }

    char magic[4]; uint32_t version, n_tensors;
    f.read(magic, 4); f.read((char *)&version, 4); f.read((char *)&n_tensors, 4);
    if (memcmp(magic, "SVAD", 4) != 0 || version != 1) {
        fprintf(stderr, "vad: invalid file format\n"); return false;
    }

    // First pass: read tensor metadata
    struct TInfo { std::string name; std::vector<uint32_t> dims; size_t n_floats; };
    std::vector<TInfo> infos;
    auto data_start = f.tellg();
    for (uint32_t i = 0; i < n_tensors; i++) {
        uint32_t nl; f.read((char *)&nl, 4);
        std::string name(nl, '\0'); f.read(name.data(), nl);
        uint32_t nd; f.read((char *)&nd, 4);
        std::vector<uint32_t> dims(nd);
        for (auto &d : dims) f.read((char *)&d, 4);
        size_t sz = 1; for (auto d : dims) sz *= d;
        infos.push_back({name, dims, sz});
        f.seekg(sz * sizeof(float), std::ios::cur);
    }

    // Create ggml context for weight tensors
    struct ggml_init_params wp = {
        ggml_tensor_overhead() * 20 + 1024, nullptr, true
    };
    ctx_w = ggml_init(wp);

    // Conv kernels must be F16 for ggml_conv_1d on CPU
    auto is_conv_weight = [](const std::string &name) {
        return name.find("conv.weight") != std::string::npos ||
               name.find("forward_basis_buffer") != std::string::npos;
    };

    auto make_tensor = [&](const TInfo &ti) -> ggml_tensor * {
        auto &d = ti.dims;
        ggml_type dtype = is_conv_weight(ti.name) ? GGML_TYPE_F16 : GGML_TYPE_F32;
        ggml_tensor *t = nullptr;

        if (ti.name.find("reparam_conv.bias") != std::string::npos)
            t = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, 1, d[0]);
        else if (ti.name == "decoder.decoder.2.weight")
            t = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, ti.n_floats);
        else if (d.size() == 1) t = ggml_new_tensor_1d(ctx_w, dtype, d[0]);
        else if (d.size() == 2) t = ggml_new_tensor_2d(ctx_w, dtype, d[1], d[0]);
        else if (d.size() == 3) t = ggml_new_tensor_3d(ctx_w, dtype, d[2], d[1], d[0]);

        if (t) ggml_set_name(t, ti.name.c_str());
        return t;
    };

    std::vector<ggml_tensor *> tensors;
    for (auto &ti : infos) tensors.push_back(make_tensor(ti));

    backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 1);  // single-thread: no dispatch overhead on tiny graph
    buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);

    // Second pass: load weight data
    f.seekg(data_start);
    for (size_t i = 0; i < infos.size(); i++) {
        uint32_t nl; f.read((char *)&nl, 4); f.seekg(nl, std::ios::cur);
        uint32_t nd; f.read((char *)&nd, 4); f.seekg(nd * 4, std::ios::cur);
        std::vector<float> tmp(infos[i].n_floats);
        f.read((char *)tmp.data(), infos[i].n_floats * sizeof(float));

        if (tensors[i]->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp16(infos[i].n_floats);
            ggml_fp32_to_fp16_row(tmp.data(), tmp16.data(), infos[i].n_floats);
            ggml_backend_tensor_set(tensors[i], tmp16.data(), 0, infos[i].n_floats * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(tensors[i], tmp.data(), 0, infos[i].n_floats * sizeof(float));
        }
    }

    // Map named weight tensors
    ggml_tensor *stft_basis = nullptr;
    ggml_tensor *enc_w[4] = {}, *enc_b[4] = {};
    ggml_tensor *rnn_wih = nullptr, *rnn_whh = nullptr, *rnn_bih = nullptr, *rnn_bhh = nullptr;
    ggml_tensor *dec_w = nullptr, *dec_b = nullptr;

    for (size_t i = 0; i < infos.size(); i++) {
        auto &name = infos[i].name; auto *t = tensors[i];
        if      (name == "stft.forward_basis_buffer")       stft_basis = t;
        else if (name == "encoder.0.reparam_conv.weight")   enc_w[0] = t;
        else if (name == "encoder.0.reparam_conv.bias")     enc_b[0] = t;
        else if (name == "encoder.1.reparam_conv.weight")   enc_w[1] = t;
        else if (name == "encoder.1.reparam_conv.bias")     enc_b[1] = t;
        else if (name == "encoder.2.reparam_conv.weight")   enc_w[2] = t;
        else if (name == "encoder.2.reparam_conv.bias")     enc_b[2] = t;
        else if (name == "encoder.3.reparam_conv.weight")   enc_w[3] = t;
        else if (name == "encoder.3.reparam_conv.bias")     enc_b[3] = t;
        else if (name == "decoder.rnn.weight_ih")           rnn_wih = t;
        else if (name == "decoder.rnn.weight_hh")           rnn_whh = t;
        else if (name == "decoder.rnn.bias_ih")             rnn_bih = t;
        else if (name == "decoder.rnn.bias_hh")             rnn_bhh = t;
        else if (name == "decoder.decoder.2.weight")        dec_w = t;
        else if (name == "decoder.decoder.2.bias")          dec_b = t;
    }

    // ---- Build compute graph (once) ----
    struct ggml_init_params cp = {
        ggml_tensor_overhead() * 64 + 1024 * 1024, nullptr, true
    };
    ctx_compute = ggml_init(cp);
    auto *ctx = ctx_compute;

    // Input tensors
    t_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, INPUT_SAMPLES);
    ggml_set_name(t_input, "input"); ggml_set_input(t_input);

    t_h_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, LSTM_HIDDEN);
    ggml_set_name(t_h_in, "h_in"); ggml_set_input(t_h_in);

    t_c_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, LSTM_HIDDEN);
    ggml_set_name(t_c_in, "c_in"); ggml_set_input(t_c_in);

    // STFT: reflect pad → conv → magnitude
    auto *padded = ggml_pad_reflect_1d(ctx, t_input, 0, STFT_PAD_RIGHT);
    auto *stft = ggml_conv_1d(ctx, stft_basis, padded, STFT_HOP, 0, 1);
    auto *real_part = ggml_view_2d(ctx, stft, STFT_FRAMES, STFT_BINS, stft->nb[1], 0);
    auto *imag_part = ggml_view_2d(ctx, stft, STFT_FRAMES, STFT_BINS, stft->nb[1], STFT_BINS * stft->nb[1]);
    auto *mag = ggml_sqrt(ctx, ggml_add(ctx, ggml_sqr(ctx, real_part), ggml_sqr(ctx, imag_part)));

    // Encoder: 4× conv+bias+relu
    auto conv_br = [&](ggml_tensor *w, ggml_tensor *b, ggml_tensor *x, int s) {
        return ggml_relu(ctx, ggml_add(ctx, ggml_conv_1d_ph(ctx, w, x, s, 1), b));
    };
    auto *e0 = conv_br(enc_w[0], enc_b[0], mag, 1);
    auto *e1 = conv_br(enc_w[1], enc_b[1], e0, 2);
    auto *e2 = conv_br(enc_w[2], enc_b[2], e1, 2);
    auto *e3 = conv_br(enc_w[3], enc_b[3], e2, 1);

    // LSTM
    auto *x_t = ggml_reshape_1d(ctx, e3, LSTM_HIDDEN);
    auto *gates_x = ggml_add(ctx, ggml_mul_mat(ctx, rnn_wih, x_t), rnn_bih);
    auto *gates_h = ggml_add(ctx, ggml_mul_mat(ctx, rnn_whh, t_h_in), rnn_bhh);
    auto *gates = ggml_add(ctx, gates_x, gates_h);

    size_t gs = sizeof(float);
    auto *gi = ggml_view_1d(ctx, gates, LSTM_HIDDEN, 0 * LSTM_HIDDEN * gs);
    auto *gf = ggml_view_1d(ctx, gates, LSTM_HIDDEN, 1 * LSTM_HIDDEN * gs);
    auto *gg = ggml_view_1d(ctx, gates, LSTM_HIDDEN, 2 * LSTM_HIDDEN * gs);
    auto *go = ggml_view_1d(ctx, gates, LSTM_HIDDEN, 3 * LSTM_HIDDEN * gs);

    t_c_out = ggml_add(ctx,
        ggml_mul(ctx, ggml_sigmoid(ctx, gf), t_c_in),
        ggml_mul(ctx, ggml_sigmoid(ctx, gi), ggml_tanh(ctx, gg)));
    ggml_set_name(t_c_out, "c_out"); ggml_set_output(t_c_out);

    t_h_out = ggml_mul(ctx, ggml_sigmoid(ctx, go), ggml_tanh(ctx, t_c_out));
    ggml_set_name(t_h_out, "h_out"); ggml_set_output(t_h_out);

    // Decoder: relu → linear → sigmoid
    auto *h_relu = ggml_relu(ctx, t_h_out);
    auto *logit = ggml_add(ctx, ggml_sum(ctx, ggml_mul(ctx, dec_w, h_relu)), dec_b);
    t_prob = ggml_sigmoid(ctx, logit);
    ggml_set_name(t_prob, "prob"); ggml_set_output(t_prob);

    // Pre-allocate graph
    graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, t_prob);
    ggml_build_forward_expand(graph, t_h_out);
    ggml_build_forward_expand(graph, t_c_out);

    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, graph);

    reset();
    return true;
}

SileroVad::~SileroVad() {
    if (allocr) ggml_gallocr_free(allocr);
    if (ctx_compute) ggml_free(ctx_compute);
    if (buf_w) ggml_backend_buffer_free(buf_w);
    if (backend) ggml_backend_free(backend);
    if (ctx_w) ggml_free(ctx_w);
}

void SileroVad::reset() {
    memset(h, 0, sizeof(h));
    memset(c, 0, sizeof(c));
    memset(context, 0, sizeof(context));
}

// ---- Forward pass (graph already built, just set inputs and compute) ----

float SileroVad::process(const float *pcm_512) {
    // Build input: context(64) + new audio(512)
    float input_buf[INPUT_SAMPLES];
    memcpy(input_buf, context, CONTEXT_SIZE * sizeof(float));
    memcpy(input_buf + CONTEXT_SIZE, pcm_512, CHUNK_SAMPLES * sizeof(float));
    memcpy(context, pcm_512 + CHUNK_SAMPLES - CONTEXT_SIZE, CONTEXT_SIZE * sizeof(float));

    // Set inputs
    ggml_backend_tensor_set(t_input, input_buf, 0, INPUT_SAMPLES * sizeof(float));
    ggml_backend_tensor_set(t_h_in, h, 0, LSTM_HIDDEN * sizeof(float));
    ggml_backend_tensor_set(t_c_in, c, 0, LSTM_HIDDEN * sizeof(float));

    // Compute
    ggml_backend_graph_compute(backend, graph);

    // Read outputs
    float result;
    ggml_backend_tensor_get(t_prob, &result, 0, sizeof(float));
    ggml_backend_tensor_get(t_h_out, h, 0, LSTM_HIDDEN * sizeof(float));
    ggml_backend_tensor_get(t_c_out, c, 0, LSTM_HIDDEN * sizeof(float));

    return result;
}
