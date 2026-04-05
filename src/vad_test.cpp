/**
 * vad_test.cpp - Test Silero VAD implementation against ONNX reference.
 *
 * Reads a 16kHz mono WAV file, runs the C VAD frame-by-frame,
 * and prints per-frame probabilities to stdout for comparison.
 *
 * Usage: ./build/vad_test models/silero_vad.bin data/recording.wav
 */

#include "vad_ggml.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

// Minimal WAV reader (16-bit PCM, mono, 16kHz)
static bool read_wav_16k(const char *path, std::vector<float> &audio) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;

    // Read RIFF header
    char riff[4];
    fread(riff, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) != 0) { fclose(f); return false; }

    uint32_t file_size;
    fread(&file_size, 4, 1, f);

    char wave[4];
    fread(wave, 1, 4, f);
    if (memcmp(wave, "WAVE", 4) != 0) { fclose(f); return false; }

    // Find data chunk
    uint16_t channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;

    while (!feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t format;
            fread(&format, 2, 1, f);
            fread(&channels, 2, 1, f);
            fread(&sample_rate, 4, 1, f);
            fseek(f, 6, SEEK_CUR);  // skip byte_rate + block_align
            fread(&bits_per_sample, 2, 1, f);
            if (chunk_size > 16) fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            if (sample_rate != 16000 || channels != 1 || bits_per_sample != 16) {
                fprintf(stderr, "Expected 16kHz mono 16-bit, got %uHz %uch %ubit\n",
                        sample_rate, channels, bits_per_sample);
                fclose(f);
                return false;
            }
            int n_samples = chunk_size / 2;
            std::vector<int16_t> raw(n_samples);
            fread(raw.data(), 2, n_samples, f);
            audio.resize(n_samples);
            for (int i = 0; i < n_samples; i++)
                audio[i] = raw[i] / 32768.0f;
            fclose(f);
            return true;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    fclose(f);
    return false;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.bin> <audio.wav>\n", argv[0]);
        return 1;
    }

    SileroVad vad;
    if (!vad.load(argv[1])) return 1;

    std::vector<float> audio;
    if (!read_wav_16k(argv[2], audio)) {
        fprintf(stderr, "Failed to read %s\n", argv[2]);
        return 1;
    }

    int n_frames = audio.size() / SileroVad::CHUNK_SAMPLES;
    fprintf(stderr, "Audio: %zu samples (%.1fs), %d frames\n",
            audio.size(), audio.size() / 16000.0, n_frames);

    for (int i = 0; i < n_frames; i++) {
        float prob = vad.process(audio.data() + i * SileroVad::CHUNK_SAMPLES);
        printf("%.6f\n", prob);
    }

    return 0;
}
