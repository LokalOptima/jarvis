# Improving Wake Word Discrimination: From 0.22 to 0.47 Gap

The initial system worked — it detected "hey Jarvis" — but the margin between genuine wake words (score ~0.85) and random speech (score ~0.63) was uncomfortably thin. A 0.22 gap means ambient noise or casual conversation could easily trigger false positives.

This post covers the systematic analysis and improvements that more than doubled that gap.

## Setting Up the Test Bench

Before optimizing anything, we needed reproducible measurements. Three test recordings:

- **positive.wav** (3.2s): Contains "hey Jarvis"
- **negative.wav** (4.5s): Speech that isn't the wake word
- **background.wav** (19.5s): Room noise, no speech

A Python ablation script scores all three against the enrolled templates using the same subsequence DTW algorithm as the C++ runtime. Every proposed change gets measured against all three.

## Finding 1: Frame 0 Is Poisoning the Scores

The first discovery came from analyzing frame-level cosine similarity between negative audio and all templates:

```
template  0: max frame sim=0.980 at neg_frame=0 tmpl_frame=0
template  1: max frame sim=0.984 at neg_frame=0 tmpl_frame=0
...
template 17: max frame sim=0.998 at neg_frame=0 tmpl_frame=0
```

Frame 0 of *every* template matches frame 0 of the negative sample at 0.97-0.998 cosine similarity. This is Whisper's "beginning of audio" representation — content-independent. The DTW gets a near-free perfect match on at least one frame pair, inflating all scores.

**Fix: Skip the first 2 onset frames.** Baseline gap 0.224 → 0.240. Small but free.

## Finding 2: CMVN Crushes the Noise Floor

CMVN (Cepstral Mean and Variance Normalization) subtracts the per-dimension mean and divides by the per-dimension standard deviation across all frames in a window:

```python
def cmvn(features):
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-10
    return (features - mean) / std
```

This removes the "DC offset" of the feature space — the shared structure that makes all speech look similar to Whisper's encoder. After CMVN, only relative variation between frames matters.

The effect is dramatic:

| Config | Positive | Negative | Gap |
|--------|----------|----------|-----|
| Baseline | 0.851 | 0.627 | 0.224 |
| CMVN | 0.668 | 0.281 | 0.387 |

The positive score drops (from 0.85 to 0.67) but the negative score collapses (from 0.63 to 0.28). The gap nearly doubles.

Why? Whisper's encoder features have a strong shared component across all speech — the "average speech" direction in 384-dim space. CMVN subtracts this out, leaving only what's distinctive about each utterance.

## Finding 3: Step Penalties Punish Warping

Standard DTW allows free insertion and deletion — one template frame can match many input frames or vice versa. Adding a small penalty (0.1) for non-diagonal transitions discourages pathological warping where the DTW stretches one frame to cover a long region:

| Config | Gap |
|--------|-----|
| Baseline | 0.224 |
| Step penalty=0.1 | 0.265 |

The penalty reduces both positive and negative scores, but negative drops faster (it relies more on warping to find spurious matches).

## Finding 4: Template Averaging (DBA) Helps

Dynamic Barycenter Averaging merges multiple templates into a single representative sequence. Starting from the median-length template, it iteratively aligns all templates via DTW and averages the aligned frames.

With 18 templates merged into 1:

| Config | Gap |
|--------|-----|
| 18 individual templates | 0.224 |
| DBA (1 template) | 0.220 |
| DBA + CMVN | 0.449 |

DBA alone doesn't help (the max-over-templates already picks the best match). But DBA + CMVN is powerful — the averaged template captures the essential temporal pattern of "hey Jarvis" while CMVN removes the noise floor. And matching against 1 template instead of 18 is 18x faster.

## Finding 5: Whisper Layer Selection

Whisper Tiny has 4 encoder layers (0-3). The "final" output includes a layer norm after block 3. We tested all layers:

| Layer | Positive | Negative | Gap |
|-------|----------|----------|-----|
| 0 | 0.930 | 0.918 | 0.012 |
| 1 | 0.900 | 0.888 | 0.012 |
| 2 | 0.872 | 0.758 | 0.114 |
| 3 | 0.884 | 0.723 | 0.161 |
| Final (3 + ln) | 0.851 | 0.627 | 0.224 |

Early layers are useless — too generic, everything looks the same. Layer 3 (before layer norm) is the best raw layer, and the final layer norm actually helps discrimination in the raw case.

But with CMVN, layer 3 and final converge — CMVN does the same normalization work that ln_post does, making the layer norm redundant. Layer 3 is marginally better (0.465 vs 0.461 gap), and skipping the layer norm is free computation savings.

## The Winning Combination

Every improvement stacks:

| Config | Pos | Neg | BG | Gap |
|--------|-----|-----|-----|-----|
| Baseline | 0.851 | 0.627 | 0.627 | 0.224 |
| + skip 2 frames | 0.861 | 0.621 | 0.621 | 0.240 |
| + CMVN | 0.674 | 0.269 | 0.189 | 0.405 |
| + step penalty 0.1 | 0.653 | 0.229 | 0.131 | 0.424 |
| + DBA (1 template) | 0.594 | 0.146 | 0.088 | 0.449 |
| + layer 3 | 0.582 | 0.117 | 0.058 | **0.465** |

Final gap: **0.465** — more than double the baseline. Negative speech scores 0.12, background noise scores 0.06. The threshold can sit comfortably at 0.35-0.40 with wide margins on both sides.

## What We Learned

1. **Feature normalization matters more than feature selection.** CMVN alone accounts for most of the improvement. The choice of encoder layer barely matters once CMVN is applied.

2. **Analyze before optimizing.** The frame-0 similarity finding came from looking at the data, not from the literature. It's a Whisper-specific artifact that no keyword spotting paper would mention.

3. **Simple techniques stack.** Every improvement here is textbook signal processing or DTW — CMVN, step penalties, template averaging. No learned components, no training data, no GPU. Just careful analysis of what the features actually look like.

4. **Subsequence DTW is robust.** It handled noise frames gracefully from the start — the problem was never the matching algorithm, it was the feature space having too much shared structure. Fix the features, and DTW does its job.
