# Porting Silero VAD to ggml

The [previous post](06-client-server-split.md) mentioned VAD — voice activity detection — as the first gate in the pipeline. We were using a crude energy-based check: `sum(x²)/N > 1e-6`. It works for silence vs. speech but fires on any noise (keyboard, fan, music). This post covers replacing it with [Silero VAD](https://github.com/snakers4/silero-vad), a proper neural VAD, ported to run natively in C using ggml.

## Why Silero?

Silero VAD is a ~300K parameter model that runs in real-time on CPU. It processes 32ms audio frames (512 samples at 16kHz) and outputs a speech probability in [0, 1]. The architecture is compact:

```
Input: 512 samples (32ms at 16kHz)
  ↓ prepend 64-sample context from previous frame
STFT: reflect pad → Conv1D(258, kernel=256, stride=128) → magnitude
  ↓ 4 time frames × 129 frequency bins
Encoder: Conv1D+ReLU × 4 (129→128→64→64→128, stride 1/2/2/1)
  ↓ 1 time frame × 128 channels
LSTM: hidden=128 (stateful across frames)
  ↓
Decoder: ReLU → Linear(128→1) → sigmoid
  ↓
Speech probability [0, 1]
```

The LSTM state carries across frames, so the model builds temporal context — it knows the difference between a click and the onset of speech.

## Challenge 1: The 64-Sample Context Window

The first attempt gave near-zero probabilities on clear speech. The official `silero_vad.onnx` from the repo seemed broken.

The fix was in the official Python wrapper's `OnnxWrapper.__call__()`:

```python
context_size = 64 if sr == 16000 else 32
x = torch.cat([self._context, x], dim=1)  # prepend 64 zeros (or last 64 samples)
```

The model expects 576 samples (64 context + 512 new), not bare 512. The context carries the tail of the previous chunk, giving the STFT conv overlap information across frame boundaries. Without it, the model has no temporal continuity.

## Challenge 2: Asymmetric STFT Padding

Even with the context fix, the C implementation produced wrong values. The ONNX model's STFT applies reflect padding of **(0, 64)** — zero on the left, 64 on the right. Not the standard center padding of (128, 128) that you'd expect from `n_fft // 2`.

This was found by tracing the ONNX graph's Pad node through 15 nodes of constant folding and shape manipulation (ConstantOfShape → Concat → Reshape → Slice → Transpose → Reshape → Cast → Pad). The actual pad values computed to `[0, 0, 0, 64]` in ONNX's `[start_dim0, start_dim1, end_dim0, end_dim1]` format.

The asymmetric padding changes the STFT frame count from 5 to 4, which cascades through the encoder convolutions: the encoder output shrinks from 2 timesteps to 1. This means the LSTM processes a single timestep per frame rather than two, and there's no temporal mean at the output.

## Challenge 3: ggml Conv1D Requires F16 Kernels

The ggml CPU backend's `conv_1d` implementation requires F16 kernel weights — the code asserts `src0->type == GGML_TYPE_F16`. This matches what whisper.cpp does (its `vtype` for conv weights is always F16). The fix: convert F32 weights to F16 during loading with `ggml_fp32_to_fp16_row()`.

The F16 quantization introduces small errors (max ~0.036 on speech probabilities) but this is irrelevant for a 0.5 threshold.

## Implementation: Naive C vs ggml

The first implementation used naive scalar C loops. It was correct (max error 2.56e-6 vs ONNX) but slow:

| Implementation | µs/frame | Real-time factor |
|---|---|---|
| ONNX (Python) | 66 | 486× |
| ggml (pre-built graph) | 169 | 189× |
| C naive loops | 300 | 107× |

The ggml version builds the compute graph once during `load()` and reuses it every frame — `process()` just sets input tensor data and calls `ggml_backend_graph_compute()`. This eliminates per-frame allocation overhead.

The remaining 2.6× gap vs ONNX is likely onnxruntime's more aggressive SIMD for conv/matmul (dedicated AVX-512 FMA paths vs ggml's generalized F16 conv). At 169µs per 32ms frame, it's negligible in the pipeline — the whisper encoder takes 28ms.

## Weight Export

A Python script (`scripts/export_silero.py`) extracts the 16kHz branch weights from the ONNX model:

```
SVAD header (12 bytes): magic "SVAD" + version + n_tensors
Per tensor: name_len + name + n_dims + dims + float32 data
```

15 tensors, 309K parameters, 1.2 MB binary. The weights are stored in PyTorch layout `(Cout, Cin, K)` which happens to match ggml's column-major `(K, Cin, Cout)` — no transpose needed, just a dimension relabel.

## Integration

The VAD runs on each incoming 200ms audio chunk (6 frames of 512 samples). If any frame exceeds the 0.5 threshold, the expensive whisper encoder + DTW detection runs. Otherwise the chunk is silently dropped.

```
  200ms audio arrives
  ↓
  Feed 6 × 512-sample frames through Silero VAD
  ↓ any P(speech) > 0.5?
  No → skip (saves 28ms encoder + DTW)
  Yes → run full detection pipeline
```

The VAD state (LSTM h/c + 64-sample audio context) persists across slides, maintaining temporal context. It's reset after a detection (during the refractory period).

## Files

- `src/vad_ggml.h`, `src/vad_ggml.cpp` — ggml implementation (production)
- `src/vad.h`, `src/vad.cpp` — naive C reference implementation
- `src/vad_test.cpp` — validation binary
- `scripts/silero_ref.py` — ONNX reference runner
- `scripts/export_silero.py` — weight extractor
