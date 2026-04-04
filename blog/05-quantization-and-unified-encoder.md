# Quantization and Unified Encoder

The [previous post](04-performance-profiling-and-ggml-mel.md) got the pipeline down to 45ms/cycle (46x real-time) by replacing the scalar FFT with ggml matmul DFT and skipping cross-attention. This post covers quantized models, switching to English-only weights, and eliminating the Python/C++ encoder divergence.

## Quantized Whisper Tiny

whisper.cpp supports integer quantization: Q4_0 through Q8_0, plus k-quants. For tiny (39M params), aggressive quantization destroys accuracy — Q2_K/Q3_K produce garbage. The sweet spot is Q5 or Q8.

I benchmarked three variants on the i7-12700 with a `bench` tool that runs 50 iterations of mel + encode on synthetic audio:

| Model | Size | Mel | Encode | Total | RTx |
|-------|------|-----|--------|-------|-----|
| f16 | 77MB | 1.5ms | 41.8ms | 43.3ms | 46x |
| **Q8_0** | **42MB** | **1.5ms** | **28.3ms** | **29.8ms** | **67x** |
| Q5_1 | 31MB | 1.6ms | 39.7ms | 41.3ms | 48x |

Q8_0 is the clear winner: **32% faster encode**, 67x real-time. Q5_1 barely beats f16 — the dequantization overhead nearly offsets the memory bandwidth savings when AVX-512 already has plenty of bandwidth.

The mel spectrogram is invariant across quantization levels (same code path, same float32 computation), confirming the bottleneck is the encoder's quantized matmuls.

## English-Only vs Multilingual

The original setup used `ggml-tiny.bin` (multilingual, 99 languages). For English wake words, is this right?

The whisper paper is explicit: *"For small models, there is negative transfer between tasks and languages."* At tiny scale, multilingual is ~2% WER worse on English than the English-only variant. At medium/large the gap vanishes, but we're at tiny where 384 dims is a tight capacity budget to spread across 99 languages.

The counterargument: a non-native speaker might benefit from multilingual's exposure to diverse phonology. For DTW template matching on encoder embeddings (not transcription), nobody has tested this directly — it's a gap in the literature.

I switched to `.en` since the empirical path is clear: just test both and compare DTW scores.

## The Encoder Mismatch Problem

Switching to `.en` broke detection completely — scores dropped to 0.02 (from ~0.4). The fix revealed a deeper problem.

The templates were built with Python's `openai-whisper` loading `whisper.load_model("tiny")` (multilingual), while the C++ runtime used whatever `models/ggml-tiny.bin` pointed to. Two different models, two different embedding spaces. DTW scores between them are meaningless.

But it's worse than that. Even with the "same" model, the Python and C++ implementations differ:
- Different mel spectrogram code (we replaced the FFT with ggml DFT matmul)
- Different floating point accumulation order
- f16 vs Q8_0 quantization changes every matmul output slightly

The templates and detector must use the exact same code path.

## Solution: C++ Encoder for Everything

I built `encode` — a standalone C++ binary that reads WAV files, runs our whisper.cpp mel + encode pipeline, and writes raw float32 embeddings to stdout:

```
./build/encode models/ggml-tiny.bin clip_01.wav clip_02.wav > embeddings.bin
```

Binary output format per file:
```
int32  n_frames
int32  dim (queried from model at runtime)
float  data[n_frames * dim]
```

It accepts multiple WAV files on one invocation, loading the model once. The Python enrollment calls it via `subprocess`:

```python
def extract_features_batch(wav_paths: list[str | Path]) -> list[np.ndarray]:
    result = subprocess.run(
        [ENCODE_BIN, MODEL_PATH] + [str(p) for p in wav_paths],
        stdout=subprocess.PIPE,
    )
    # ... parse concatenated binary output ...
```

For 18 enrollment clips, this is 1 model load instead of 18 (the earlier per-file approach spawned a new process per clip). The model load (~42MB ggml init) dominates subprocess overhead, so batching matters.

`features.py` went from 75 lines of Python whisper + torch + monkey-patched encoder forward pass to 60 lines of subprocess + struct parsing. No more `torch` import, no more `openai-whisper` for feature extraction.

## Dependency Cleanup

With feature extraction moved to C++, the heavy Python dependencies are only needed for recording new clips (Whisper Turbo transcription to find wake word timestamps):

```toml
[project]
dependencies = ["numpy"]

[project.optional-dependencies]
enroll = ["sounddevice", "openai-whisper>=20250625", "torch>=2.11.0"]
```

`make templates` (building templates from existing clips) needs only `numpy` + the C++ binaries. The multi-GB torch/whisper install is only required for `make enroll`.

## Results

| | Before | After |
|---|--------|-------|
| Model | multilingual f16 (77MB) | English-only Q8_0 (42MB) |
| Encode | 41.8ms | 28.3ms |
| Total cycle | 43.3ms | 29.8ms |
| RTx | 46x | **67x real-time** |
| Template/runtime match | no (Python vs C++) | yes (same binary) |
| Required Python deps | torch, openai-whisper | numpy |

The detection pipeline now runs at 67x real-time on an i7-12700, with guaranteed encoder consistency between enrollment and detection, and a 45% smaller model on disk.
