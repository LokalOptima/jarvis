/**
 * bench.cpp - Benchmark mel + encode for different whisper models.
 *
 * Usage: ./build/bench <model_path> [n_iterations]
 */

#include "whisper.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr float BUFFER_SEC  = 2.0f;
static constexpr int   SAMPLE_RATE = 16000;
static constexpr int   MEL_HOP     = WHISPER_HOP_LENGTH;
static constexpr int   CONV_STRIDE = 2;
static constexpr int   WHISPER_DIM = 384;

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path> [n_iterations]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    int n_iter = argc >= 3 ? atoi(argv[2]) : 50;

    // Load model
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    whisper_context *ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }
    whisper_set_encoder_only(ctx, true);

    // Synthetic audio: 2s of sine wave at 440Hz
    const int buffer_samples = (int)(BUFFER_SEC * SAMPLE_RATE);
    std::vector<float> pcm(buffer_samples);
    for (int i = 0; i < buffer_samples; i++)
        pcm[i] = 0.3f * sinf(2.0f * M_PI * 440.0f * i / SAMPLE_RATE);

    const int max_enc_frames = (buffer_samples / MEL_HOP + 1) / CONV_STRIDE + 1;
    const int max_enc_floats = max_enc_frames * WHISPER_DIM;
    std::vector<float> enc_out(max_enc_floats);

    // Warmup
    whisper_pcm_to_mel(ctx, pcm.data(), pcm.size(), 1);
    int mel_frames = whisper_n_len(ctx);
    int audio_ctx = (mel_frames + 1) / CONV_STRIDE;
    if (audio_ctx <= 0) audio_ctx = 1;
    whisper_set_audio_ctx(ctx, audio_ctx);
    whisper_encode(ctx, 0, 1);

    // Benchmark
    std::vector<long> t_mel(n_iter), t_enc(n_iter), t_total(n_iter);

    for (int i = 0; i < n_iter; i++) {
        auto t0 = std::chrono::steady_clock::now();

        whisper_pcm_to_mel(ctx, pcm.data(), pcm.size(), 1);
        auto t1 = std::chrono::steady_clock::now();

        mel_frames = whisper_n_len(ctx);
        audio_ctx = (mel_frames + 1) / CONV_STRIDE;
        if (audio_ctx <= 0) audio_ctx = 1;
        whisper_set_audio_ctx(ctx, audio_ctx);
        whisper_encode(ctx, 0, 1);
        auto t2 = std::chrono::steady_clock::now();

        whisper_encoder_output(ctx, enc_out.data(), max_enc_floats);

        t_mel[i]   = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        t_enc[i]   = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        t_total[i] = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();
    }

    // Stats (skip first 5 for warmup)
    int skip = n_iter > 10 ? 5 : 0;
    int count = n_iter - skip;

    auto stats = [&](const char *name, std::vector<long> &v) {
        long sum = 0, mn = v[skip], mx = v[skip];
        for (int i = skip; i < n_iter; i++) {
            sum += v[i];
            if (v[i] < mn) mn = v[i];
            if (v[i] > mx) mx = v[i];
        }
        float avg = (float)sum / count;
        float rtf = (avg / 1e6f) / BUFFER_SEC;
        printf("  %-8s  avg=%7.0f µs  min=%6ld  max=%6ld  RTF=%.4f\n", name, avg, mn, mx, rtf);
    };

    printf("\n%s  (%d iters, skip %d)\n", model_path, n_iter, skip);
    printf("─────────────────────────────────────────────────────────\n");
    stats("mel", t_mel);
    stats("encode", t_enc);
    stats("total", t_total);

    float avg_total = 0;
    for (int i = skip; i < n_iter; i++) avg_total += t_total[i];
    avg_total /= count;
    printf("  RTx = %.0fx real-time\n\n", BUFFER_SEC / (avg_total / 1e6f));

    whisper_free(ctx);
    return 0;
}
