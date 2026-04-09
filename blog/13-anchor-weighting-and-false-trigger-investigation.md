# Anchor Weighting and False Trigger Investigation

The [previous post](12-unified-mode-and-detection-improvements.md) identified the DTW stretch problem — partial keywords like "jar" falsely matching the full "hey jarvis" template. This post covers the investigation into that problem, the approaches tried, and the fix that stuck.

## The Problem

Two types of false triggers:

1. **Partial keyword**: saying "jarvis" triggers the "hey jarvis" template. The "jarvis" portion aligns perfectly via diagonal DTW steps, while the "hey" portion is consumed via deletion steps (template advances without input) at low cost.

2. **Disrupted keyword**: saying "what is... [pause] ...the weather" triggers the "what's the weather" template. The pause disrupts the temporal alignment, forcing extra non-diagonal steps.

Both cases share the same root cause: the DTW cost is averaged over the full template length, so a well-matched majority dilutes a poorly-matched minority.

## What Didn't Work

**Span ratio check.** Track `best_start` and `best_end` in the DTW, reject matches where `(end - start) / template_length < 0.7`. Failed completely — all ratios were 0.82+ because the DTW inflates spans by absorbing surrounding silence into the match via deletion steps. Required parallel int arrays in the DP for start tracking, all for nothing.

**Higher step penalty alone.** Tripled `STEP_PENALTY` from 0.1 to 0.3. The DTW adapted by switching from deletion steps (penalty + distance) to diagonal steps (distance only, no penalty) through the mismatched portion. Total cost barely changed. The penalty and cosine distance operate on the same scale (~0.1-0.5), so the DTW can trade one for the other.

**Running CMVN.** Replaced per-buffer CMVN with an exponential moving average over ~10s of speech (Welford's online variance, alpha=0.002). The idea was more stable normalization would improve score separation. A/B testing showed it made scores worse — more variable, lower true positives. Reverted to per-buffer CMVN.

## What Worked: Anchor Weighting

The key observation from diagnostic logging: "hey" is only ~5 frames out of a 34-frame template (~15%). Any cost-averaging method will dilute a 15% mismatch into noise. The score gap between "jarvis" (0.38-0.40) and "hey jarvis" (0.41-0.60) was only 0.01-0.03.

The fix: **weight the first and last K template frames higher in the DTW cost function.** The cosine distance `c` at anchor positions is multiplied by a weight factor before being added to the DP accumulator:

```cpp
int anchor_end = n_tmpl - JARVIS_ANCHOR_FRAMES;
for (int j = 1; j <= n_tmpl; j++) {
    float c = 1.0f - cosine_dot(input[i-1], tmpl[j-1], inv_norms[i-1]);
    if (j <= JARVIS_ANCHOR_FRAMES || j > anchor_end)
        c *= JARVIS_ANCHOR_WEIGHT;
    // ... standard DP transition
    curr_row[j] = c + best_prev;
}
```

With `ANCHOR_FRAMES = 5` and `ANCHOR_WEIGHT = 3.0`:

- **True match** ("hey jarvis"): anchor frames align well, `c ≈ 0.05`, weighted to `0.15`. Negligible score impact.
- **Partial match** ("jarvis"): first 5 anchor frames mismatch, `c ≈ 0.5`, weighted to `1.5`. Five frames at 3x cost creates a significant gap.

The DTW can't route around this — the weighted distance is paid regardless of step type (diagonal, insertion, or deletion). Every cell computes `c` between its input and template frame, and the weight amplifies mismatches at template endpoints where partial keywords fail.

Constants: `STEP_PENALTY = 0.3`, `ANCHOR_FRAMES = 5`, `ANCHOR_WEIGHT = 3.0`, threshold lowered to `0.25` to accommodate the shifted score distribution.

## Other Changes

**SIGINT fix.** The signal handler in `server.cpp` set `g_running = false` but never called `j.stop()`. The detection thread loops on `m_running` (set by `stop()`), so Ctrl-C left it running. Fixed by storing a static `Jarvis*` and calling `stop()` from the handler.

**Static library extraction.** `CMakeLists.txt` now builds `jarvis_lib` as a static library. The `jarvis` executable just links it. Enables embedding the detection engine in other projects.
