# Next Steps: Implementing the Winning Configuration

## Current State

The detector works with basic subsequence DTW on Whisper Tiny encoder features.
Ablation study found a configuration that doubles the discrimination gap (0.22 → 0.47).

Test scores with best config (layer3 + skip2 + DBA + CMVN + step=0.1):
- positive: 0.582, negative: 0.117, background: 0.058

## What Needs to Be Implemented

### 1. CMVN in C++ (biggest win)

Subtract per-dimension mean and divide by std across all frames in the encoder output,
before DTW matching. This is just two passes over the ~100x384 frame buffer.

In `jarvis.cpp`, after `whisper_encoder_output()`:
```
for each dim d: compute mean and std across all frames
for each frame t, dim d: output[t][d] = (output[t][d] - mean[d]) / std[d]
```

Then L2-normalize each frame (as we already do for templates).

### 2. DBA template generation in Python

In `enroll.py`'s `build_templates()`: after extracting all 18 clip features,
run DBA to merge them into 1 (or 2-3) representative templates.
Save the averaged template(s) instead of all 18.

This also makes DTW matching 18x faster at runtime (1 template instead of 18).

### 3. Skip 2 onset frames

In both Python (enrollment) and C++ (runtime):
- `enroll.py`: `features = extract_features(audio)[2:MAX_TEMPLATE_FRAMES]`
- `jarvis.cpp`: offset encoder output pointer by `2 * WHISPER_DIM` and subtract 2 from frame count

### 4. Step penalty in DTW

`jarvis.cpp` `subdtw()`: add a `step_penalty` parameter (0.1).
Non-diagonal transitions (insertion/deletion) get the penalty added to their cost.
Already prototyped in the ablation script.

### 5. Layer 3 extraction (optional, marginal gain)

Extract from encoder layer 3 instead of the final output (layer 3 + ln_post).
Requires modifying whisper.cpp's `whisper_encode_internal()` to stop before `ln_post`,
or adding a flag to skip the final layer norm.

Gap improvement: 0.461 → 0.465. Marginal, but saves the ln_post computation.

## Implementation Order

1. CMVN (Python enrollment + C++ runtime) — biggest discrimination gain
2. Skip onset frames (trivial, both sides)
3. Step penalty (C++ only, one constant)
4. DBA (Python enrollment only) — changes template format from N templates to 1-3
5. Layer 3 (optional) — only if we want the last bit of performance

## Test Procedure

After each change, run `uv run python -m jarvis.ablation` to verify scores match
the ablation predictions, then `make run` for live testing.

## Files to Modify

- `jarvis/enroll.py` — DBA, skip frames, CMVN on templates
- `src/jarvis.cpp` — CMVN, skip frames, step penalty
- `lib/whisper.cpp/src/whisper.cpp` — layer 3 extraction (optional)
- `jarvis/features.py` — already has layer selection support
