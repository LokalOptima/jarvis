# Performance Profiling and the ggml Mel Spectrogram

The [previous post](03-deploying-the-winning-configuration.md) got the detection pipeline working with CMVN, DBA templates, and a terminal visualizer. This post covers profiling the pipeline, eliminating wasted computation, and replacing whisper.cpp's scalar FFT with a ggml-native mel spectrogram.

## Profiling the Pipeline

Before optimizing anything, instrument everything. I added microsecond-resolution timing to every stage of the detection loop:

| Stage | Avg time | % of total |
|-------|----------|------------|
| **encode** | **49,393 µs** | **80.3%** |
| **mel** | **11,523 µs** | **18.7%** |
| dtw | 445 µs | 0.7% |
| vad | 60 µs | 0.1% |
| audio_get | 33 µs | <0.1% |
| norms | 13 µs | <0.1% |
| cmvn | 10 µs | <0.1% |
| copy | 8 µs | <0.1% |
| render | 5 µs | <0.1% |
| **total** | **61,493 µs** | |

The pipeline is 99% whisper. Everything we wrote — DTW, CMVN, norms, VAD, rendering — is rounding error at 483µs total. The only things worth optimizing are inside `whisper.cpp`.

## Free Win: Skip Cross-Attention

`whisper_encode_internal` runs three ggml compute graphs per call:

1. **conv** — 2x Conv1D + GELU (mel → embeddings): ~4ms
2. **encoder** — 4 transformer layers: ~38ms
3. **cross** — pre-compute decoder cross-attention K,V: ~6ms

We never decode. The cross graph runs 4 layers × 2 matmuls (K and V projections) for a decoder we don't use. Adding a `skip_cross` flag to `whisper_state` and gating the cross computation on it:

```c
if (!wstate.skip_cross) {
    // ... cross-attention graph ...
}
```

Saves ~6ms per cycle. Free.

## The Mel Spectrogram Problem

The mel spectrogram was 11.5ms — 19% of the pipeline. Looking at the code, it was the original whisper.cpp implementation:

```c
// Cooley-Tukey FFT
// poor man's implementation - use something better
static void fft(float* in, int N, float* out) {
```

The comment says it all. A recursive Cooley-Tukey FFT with no SIMD, processing each of 200 frames independently. Each 400-point FFT recurses until it hits odd-sized subproblems (400 = 2⁴ × 5²), where it falls back to an O(N²) naive DFT. Then a hand-unrolled-by-4 scalar loop applies the 80×201 mel filterbank.

The filterbank application is literally a matrix multiply: for each frame, compute 80 dot products of length 201. Over 200 frames, that's a `(80, 201) × (201, 200)` matmul done with scalar arithmetic while ggml's AVX-512/NEON matmul sits right there.

## DFT as Matrix Multiply

The key insight: for small N, the DFT doesn't need FFT. The DFT is a matrix multiply:

```
Re[k] = Σ_n x[n] · cos(2πkn/N)    →   Re = cos_matrix @ frames
Im[k] = Σ_n x[n] · sin(2πkn/N)    →   Im = sin_matrix @ frames
```

For N=400, this is O(N²) instead of O(N log N) — normally a bad trade. But we can **batch all 200 frames** into a single matmul: `(201, 400) × (400, 200)`. That's 32M multiply-adds going through ggml's SIMD kernels instead of 700K operations through a scalar recursive FFT.

The full mel spectrogram becomes a ggml compute graph:

```
frames    (400, 200)  — Hann-windowed audio frames
    ↓ ggml_mul_mat (cos_basis)
re        (201, 200)  — real part of DFT
    ↓ ggml_mul_mat (sin_basis)
im        (201, 200)  — imaginary part of DFT
    ↓ ggml_mul + ggml_add
power     (201, 200)  — re² + im²
    ↓ ggml_mul_mat (mel_filters)
mel       (80, 200)   — mel filterbank output
    ↓ ggml_clamp + ggml_log + ggml_scale
mel_log10 (80, 200)   — log10 mel spectrogram
```

Three matmuls + element-wise ops. The DFT basis matrices (cos and sin, each 201×400) are precomputed once at init and stored as persistent ggml tensors. The mel filterbank comes straight from the whisper model file.

This automatically adapts to whatever SIMD the hardware supports — AVX-512 on x86, NEON on ARM, dotprod on newer ARM. No FFT library needed.

## Implementation

The implementation follows the existing whisper.cpp graph pattern (same as conv, encoder, cross):

1. **Init**: precompute DFT cos/sin matrices as ggml tensors, init `sched_mel` scheduler
2. **Per-call**: build graph, set windowed frames as input, compute, read output
3. **Post-graph**: transpose from ggml column-major to whisper's row-major layout, apply dynamic range compression

Pre-allocated buffers in `whisper_state` avoid per-call heap allocation. The graph is rebuilt each call (ggml metadata ops, not compute), but the basis tensors persist.

Deleted ~170 lines of dead code: `dft()`, `fft()`, `log_mel_spectrogram_worker_thread()`, sin/cos lookup tables.

## Results

| Stage | Before | After | Change |
|-------|--------|-------|--------|
| mel spectrogram | 11.5ms | 6.8ms | **-41%** |
| cross-attention | 6ms | 0ms | **eliminated** |
| **Total cycle** | **61ms** | **45ms** | **-26%** |
| RTF (2s buffer) | 0.031 | 0.023 | **44x real-time** |

The mel improvement comes entirely from replacing scalar loops with ggml's SIMD matmul. On an i7-12700 with AVX-512, three batched matmuls beat 200 recursive scalar FFTs despite doing more arithmetic (O(N²) vs O(N log N)).

The encoder transformer (38ms) is still the dominant cost, but it's pure ggml matmul — there's nothing to optimize there without changing the model.

## Raspberry Pi 4: Too Slow

I also tested on a Raspberry Pi 4 (Cortex-A72, 1.8GHz, ARMv8.0). The encoder alone takes 830ms — way over the 200ms slide budget.

The Pi 4 has NEON (128-bit SIMD, 4 floats/cycle) but **no dotprod instruction** (ARMv8.2+). ggml's fast quantized matmul kernels are all gated behind `__ARM_FEATURE_DOTPROD`. Without it, each dot product takes 6 instructions instead of 1. There's no software workaround — the hardware doesn't have the instruction.

A Pi 5 (Cortex-A76, ARMv8.2 with dotprod) should be 3-4x faster and likely viable.

## UX: Buffer Fill Indicator

One last thing: the first wake word after startup was consistently missed. The audio ring buffer needs 2 seconds to fill, but the detector was printing "Listening..." immediately. Any wake word spoken during the fill period would only be partially captured.

Fix: wait for the buffer to fill before claiming readiness, showing a filling progress bar (reusing the existing `render_bar` in dim/silent mode):

```c
const int steps = 20;
const int step_ms = (int)(BUFFER_SEC * 1000) / steps;
for (int i = 1; i <= steps && g_running; i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
    render_bar("buffering", (float)i / steps, 1.0f, 0, true);
}
```

Same dim bar as the cooldown indicator, but filling up instead of draining. "Listening..." only appears once the buffer is full.
