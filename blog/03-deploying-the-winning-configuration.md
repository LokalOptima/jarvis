# Deploying the Winning Configuration

The [previous post](02-improving-discrimination-with-feature-analysis.md) found a configuration that doubled the discrimination gap: CMVN + DBA + onset skip + step penalty took the positive/negative gap from 0.22 to 0.47. But all of that was prototyped in a Python ablation script. The actual C++ runtime and Python enrollment pipeline still used the baseline configuration.

This post covers porting those improvements into the production code, replacing the SDL2 GUI with a terminal visualizer, and cleaning up the codebase along the way.

## The Implementation

Four changes, each touching both the Python enrollment pipeline and the C++ runtime.

### 1. CMVN: The Biggest Win

CMVN (Cepstral Mean and Variance Normalization) subtracts the per-dimension mean and divides by the standard deviation across all frames in a window. It's the single most impactful change — it accounts for most of the gap improvement.

**Python side** (enrollment): apply CMVN to each clip's raw features before template averaging. Three lines:

```python
def cmvn(features):
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-10
    return (features - mean) / std
```

**C++ side** (runtime): apply CMVN to the encoder output every cycle, before DTW matching. Three passes over the ~100x384 frame buffer — compute mean, compute variance, normalize in-place:

```c
static void apply_cmvn(float *frames, int n_frames) {
    float mean[384] = {}, var[384] = {};

    // Pass 1: mean
    for (int t = 0; t < n_frames; t++)
        for (int d = 0; d < 384; d++) mean[d] += frames[t*384 + d];
    for (int d = 0; d < 384; d++) mean[d] /= n_frames;

    // Pass 2: variance
    for (int t = 0; t < n_frames; t++)
        for (int d = 0; d < 384; d++) {
            float diff = frames[t*384 + d] - mean[d];
            var[d] += diff * diff;
        }
    for (int d = 0; d < 384; d++) var[d] = 1.0f / (sqrtf(var[d] / n_frames) + 1e-10f);

    // Pass 3: normalize
    for (int t = 0; t < n_frames; t++)
        for (int d = 0; d < 384; d++) frames[t*384 + d] = (frames[t*384 + d] - mean[d]) * var[d];
}
```

Three passes instead of two because you can't start normalizing until you have the full variance. With ~100 frames x 384 dims, the data is ~150KB — fits in L2 cache, so the extra pass costs effectively nothing. The Whisper encoder call takes 10-20ms; CMVN takes microseconds.

Note the `var[d]` trick: we store `1/std` directly so the normalization pass uses multiply instead of divide. Small thing, but division is ~5x slower than multiplication on most CPUs.

### 2. Skip Onset Frames

Whisper's first 2 encoder frames encode "start of audio" — a content-independent representation that looks nearly identical for all inputs (cosine similarity 0.97-0.99). These frames inflate DTW scores for everything, reducing discrimination.

**C++ runtime**: offset the encoder output pointer past the first 2 frames. Zero cost — it's a pointer add.

```c
float *enc_ptr = encoder_output.data() + ONSET_SKIP * WHISPER_DIM;
n_enc_frames -= ONSET_SKIP;
```

**Python enrollment**: slice after extraction.

```python
features = extract_features(audio)[ONSET_SKIP:MAX_TEMPLATE_FRAMES]
```

### 3. Step Penalty in DTW

Standard DTW allows free insertion and deletion — one template frame can stretch to match many input frames. Adding a penalty of 0.1 for non-diagonal transitions discourages pathological warping. Negative speech relies more on warping to find spurious matches, so the penalty hurts it more than genuine matches.

The change is one line in the DTW inner loop:

```c
float diag = prev_row[j - 1];            // diagonal: no penalty
float ins  = prev_row[j] + STEP_PENALTY; // insertion: penalized
float del  = curr_row[j - 1] + STEP_PENALTY; // deletion: penalized
```

### 4. DBA Template Averaging

Dynamic Barycenter Averaging merges all 18 enrolled templates into a single representative template. Starting from the median-length template, it iteratively aligns all templates via full DTW and averages the aligned frames. After 5 iterations, the result captures the essential temporal pattern of "hey Jarvis" while smoothing out per-utterance variation.

This is enrollment-only — the C++ runtime just loads the single template from disk. The side benefit: matching against 1 template instead of 18 makes runtime DTW 18x faster.

The enrollment pipeline became:

```python
raw_features = [extract_features(audio)[ONSET_SKIP:MAX_TEMPLATE_FRAMES] for each clip]
cmvn_features = [cmvn(f) for f in raw_features]
template = dba(cmvn_features, n_iter=5)
template = l2_normalize(template)
# save 1 template instead of 18
```

### Threshold Update

With the old feature space, scores ranged from ~0.63 (negative) to ~0.85 (positive), threshold at 0.80. After CMVN, the score distribution shifts dramatically: positive drops to ~0.58, negative collapses to ~0.12, background to ~0.06. The threshold moved from 0.80 to 0.35.

The gap is now 0.46 instead of 0.22 — but more importantly, the noise floor is at 0.06 instead of 0.63. There's an enormous margin on both sides of the threshold.

## Replacing the SDL2 GUI

The initial detector had an SDL2 window showing a colored score bar. It worked, but it was janky — an SDL video subsystem dependency just to show a number, window focus issues, and it didn't work over SSH. On Wayland-only systems, SDL would probe for X11/XCB on startup and emit `xcb_connection_has_error()` warnings even though we only needed audio.

The replacement is a single-line ANSI terminal bar that overwrites itself in place:

```
  ████████████░░░░░░│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.12   8ms
```

Green fill below threshold, red above, yellow `│` marker at the threshold position. On detection, a timestamped line prints to stdout (so it can be piped or logged) while the bar continues on stderr.

Implementation details:

- **Zero allocation**: a static 768-byte `char` buffer, built with pointer arithmetic and `memcpy`. No `std::string`, no heap.
- **Single syscall**: one `fwrite` to stderr per frame. stderr is unbuffered, so this is a direct `write(2)`.
- **Color batching**: ANSI escape codes only emitted when the color changes, not per character.
- **UTF-8 literals**: the block characters (`█`, `·`, `│`) are written as raw byte sequences (`\xe2\x96\x88`, etc.) to avoid any encoding overhead.

All of this for a function that runs at 5 Hz. The Whisper encoder dominates the cycle time by 4 orders of magnitude. But the bar shouldn't allocate, because allocating in a render loop — even a slow one — is a habit that scales poorly.

We also added `SDL_SetHint(SDL_HINT_VIDEODRIVER, "dummy")` before `SDL_Init(SDL_INIT_AUDIO)` to prevent SDL from probing X11/XCB entirely. Audio-only means audio-only.

## Code Cleanup

The ablation script had accumulated its own copies of `cmvn`, `cosine_sim`, `dba`, and `subdtw`. The enrollment pipeline added more copies when we ported the improvements. Three files with the same functions.

We extracted everything into `jarvis/dtw.py` — a shared module with `l2norm`, `cmvn`, `cosine_sim`, `cosine_dist_matrix`, `subdtw`, and `dba`. Both `enroll.py` and `ablation.py` import from it.

While extracting, we vectorized the DBA inner loop. The original computed cosine similarity per cell in a Python loop — `cosine_sim(avg[i-1], tmpl[j-1])` called ~2500 times per template. The new version precomputes the full pairwise distance matrix with a single matrix multiply:

```python
def cosine_dist_matrix(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return 1.0 - a_norm @ b_norm.T
```

The DP recurrence still uses Python loops (it has sequential dependencies), but it indexes into the precomputed matrix instead of calling numpy per cell. The per-cell numpy overhead was the bottleneck — each call to `cosine_sim` on a 384-dim vector involves Python→C→Python round trips, temporary array allocation, and norm computation. The matmul does it all in one BLAS call.

Shared constants (`ONSET_SKIP`, `STEP_PENALTY`) moved to `jarvis/__init__.py` so they're defined once. The C++ side necessarily has its own `#define`s, but at least the Python side has a single source of truth.

## Final State

The hot path now runs every 200ms:

1. Grab 2s of audio from SDL ring buffer
2. Energy-based VAD gate (skip if silent)
3. Mel spectrogram + Whisper Tiny encode (~15ms)
4. Skip 2 onset frames (pointer offset)
5. CMVN across all frames (~microseconds)
6. Subsequence DTW with step penalty against 1 DBA template (~microseconds)
7. Render terminal bar (single `fwrite`)

Total cycle: ~15-20ms, dominated by the encoder. Everything else is noise.

The system went from "works but fragile" (0.22 gap, threshold at 0.80, 18 templates, SDL window) to "comfortable margins" (0.47 gap, threshold at 0.35, 1 template, terminal UI). No training, no GPU, no cloud. Just Whisper as a frozen feature extractor and careful signal processing on the output.
