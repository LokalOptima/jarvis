/**
 * encode.cpp - Extract whisper encoder embeddings from WAV files.
 *
 * Usage: ./build/encode <model> <wav_file> [wav_file ...]
 *
 * For each WAV file, writes raw float32 embeddings to stdout:
 *   int32  n_frames
 *   int32  dim (384)
 *   float  data[n_frames * dim]
 *
 * WAV files must be mono 16-bit 16kHz PCM.
 */

#include "whisper.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr int CONV_STRIDE = 2;

static void write_empty_header(int dim) {
    int32_t header[2] = { 0, dim };
    fwrite(header, 4, 2, stdout);
}

static bool read_wav_pcm(const char *path, std::vector<float> &pcm) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;

    char riff[4];
    if (fread(riff, 1, 4, f) != 4 || memcmp(riff, "RIFF", 4) != 0) { fclose(f); return false; }

    fseek(f, 4, SEEK_CUR); // skip file_size
    char wave[4];
    if (fread(wave, 1, 4, f) != 4 || memcmp(wave, "WAVE", 4) != 0) { fclose(f); return false; }

    // Find "fmt " chunk
    uint16_t audio_format = 0, channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;
    while (true) {
        char chunk_id[4];
        uint32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) { fclose(f); return false; }
        if (fread(&chunk_size, 4, 1, f) != 1) { fclose(f); return false; }
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            if (fread(&audio_format, 2, 1, f) != 1) { fclose(f); return false; }
            if (fread(&channels, 2, 1, f) != 1) { fclose(f); return false; }
            if (fread(&sample_rate, 4, 1, f) != 1) { fclose(f); return false; }
            fseek(f, 6, SEEK_CUR); // skip byte_rate + block_align
            if (fread(&bits_per_sample, 2, 1, f) != 1) { fclose(f); return false; }
            if (chunk_size > 16) fseek(f, chunk_size - 16, SEEK_CUR);
            break;
        }
        fseek(f, chunk_size, SEEK_CUR);
    }

    if (audio_format != 1 || channels != 1 || sample_rate != WHISPER_SAMPLE_RATE || bits_per_sample != 16) {
        fprintf(stderr, "encode: %s must be mono 16-bit 16kHz PCM WAV\n", path);
        fclose(f);
        return false;
    }

    // Find "data" chunk
    uint32_t data_size = 0;
    while (true) {
        char chunk_id[4];
        if (fread(chunk_id, 1, 4, f) != 4) { fclose(f); return false; }
        if (fread(&data_size, 4, 1, f) != 1) { fclose(f); return false; }
        if (memcmp(chunk_id, "data", 4) == 0) break;
        fseek(f, data_size, SEEK_CUR);
    }

    int n_samples = data_size / 2;
    std::vector<int16_t> raw(n_samples);
    size_t read = fread(raw.data(), 2, n_samples, f);
    fclose(f);
    if ((int)read != n_samples) return false;

    pcm.resize(n_samples);
    for (int i = 0; i < n_samples; i++)
        pcm[i] = (float)raw[i] / 32768.0f;

    return true;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model> <wav_file> [wav_file ...]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    whisper_context *ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) {
        fprintf(stderr, "encode: failed to load model: %s\n", model_path);
        return 1;
    }
    whisper_set_encoder_only(ctx, true);

    const int dim = whisper_model_n_audio_state(ctx);
    // CoreML model expects exactly 200 mel frames (2s at 16kHz).
    // Pad short clips to 2s so CoreML always gets the right shape.
    static constexpr int MIN_SAMPLES = WHISPER_SAMPLE_RATE * 2;
    std::vector<float> pcm;
    std::vector<float> enc_out;

    for (int i = 2; i < argc; i++) {
        if (!read_wav_pcm(argv[i], pcm)) {
            fprintf(stderr, "encode: failed to read %s\n", argv[i]);
            write_empty_header(dim);
            continue;
        }

        // Compute actual encoder frames from original audio length,
        // then pad/clamp to exactly 2s for CoreML's fixed [1,80,200] input.
        int orig_samples = (int)pcm.size();
        int orig_mel = orig_samples / WHISPER_HOP_LENGTH;
        int orig_enc_frames = (orig_mel + 1) / CONV_STRIDE;

        if ((int)pcm.size() < MIN_SAMPLES)
            pcm.resize(MIN_SAMPLES, 0.0f);
        else if ((int)pcm.size() > MIN_SAMPLES)
            pcm.resize(MIN_SAMPLES);

        if (whisper_pcm_to_mel(ctx, pcm.data(), pcm.size(), 1) != 0) {
            fprintf(stderr, "encode: mel failed for %s\n", argv[i]);
            write_empty_header(dim);
            continue;
        }

        int mel_frames = whisper_n_len(ctx);
        int audio_ctx = (mel_frames + 1) / CONV_STRIDE;
        if (audio_ctx <= 0) audio_ctx = 1;
        whisper_set_audio_ctx(ctx, audio_ctx);

        if (whisper_encode(ctx, 0, 1) != 0) {
            fprintf(stderr, "encode: encode failed for %s\n", argv[i]);
            write_empty_header(dim);
            continue;
        }

        // Output only frames from actual audio, not zero-padding
        int n_frames = std::min(orig_enc_frames, audio_ctx);
        if (n_frames <= 0) n_frames = 1;
        enc_out.resize(n_frames * dim);
        whisper_encoder_output(ctx, enc_out.data(), n_frames * dim);

        int32_t header[2] = { n_frames, dim };
        fwrite(header, 4, 2, stdout);
        fwrite(enc_out.data(), sizeof(float), n_frames * dim, stdout);
    }

    fflush(stdout);
    whisper_free(ctx);
    return 0;
}
