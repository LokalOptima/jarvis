# Unified Mode, Detection Timing, and the Stretch Problem

The [previous post](11-public-release.md) made the repo public with auto-downloading models. This post covers three changes: merging the two runtime modes, fixing detection timing, and discovering a fundamental scoring issue.

## One Mode

Jarvis had two modes: "local" (hardcoded keywords, no server) and "server" (`--serve` flag, config-driven, socket). They shared the same detection engine but diverged at startup — local mode skipped config, recording, and client management.

Server mode was strictly a superset. Local mode was just the original code preserved for backward compatibility. We dropped it. Now there's one path:

```sh
./jarvis-cpu                          # standalone, auto-discovers templates
./jarvis-cpu --listen tcp:9090        # same + accept client connections
./jarvis-cpu --config custom.toml     # custom config
```

Without `--listen` (and no `listen` in config), detection runs in the main thread with no socket — functionally identical to old local mode but using the same config-driven code path. With `--listen`, detection moves to a background thread and the main thread runs the accept loop.

Config loading now handles missing files gracefully: defaults for model/VAD/ding, and keywords auto-discovered by scanning `~/.cache/jarvis/templates/` for files matching the model tag pattern (`hey_jarvis.tiny-Q8.bin` → keyword `hey_jarvis`).

## Detection Timing

Three timing issues were fixed:

**Warmup.** On startup, the ring buffer is empty. Detection on a mostly-empty buffer (zero-padded to 2s) produces garbage CMVN — the mean/variance is computed over mostly zeros, distorting the real speech frames. Fix: check `audio->available()` (a new zero-copy accessor on the ring buffer) and show "warming up" until 2s of audio has accumulated. About 10 iterations at 200ms, then never checked again.

**Refractory period.** After detection, `audio->clear()` zeroed the ring buffer, creating a 2-second refractory period while it refilled. Saying a keyword twice in quick succession missed the second one. Fix: don't clear the buffer. Instead, track the DTW end frame — the encoder frame where the matched keyword ends — and pass it as `skip_frames` to `detect_once`. The encoder still processes the full 2s (CoreML requires fixed-size input), but CMVN and DTW only operate on frames after the skip point. Each 200ms cycle, `skip_frames` decays as new audio slides into the window. Detection accuracy ramps up naturally without any artificial cooldown.

**DTW end tracking.** `subdtw` already checks `curr_row[n_tmpl]` at each input frame to find the minimum-cost completion. Adding `best_end` was one extra variable — the frame index where the template was fully consumed at minimum cost. This propagates through `Templates::match` → `DetectResult::end_frame` → the listen loop.

## The Stretch Problem

Subsequence DTW allows the template to match anywhere within the input window. The template must be fully consumed, but the input frames that match it can be shorter than the template — DTW handles this via insertions (repeating a template frame to match against consecutive input frames) and deletions (skipping input frames).

Each non-diagonal step costs `STEP_PENALTY = 0.1`. The problem: this penalty is too weak. A short input segment like "jar" can match the full "hey jarvis" template by stretching:

1. "jar" frames align well with the "jar" portion of the template (low cosine distance)
2. The remaining template frames ("hey " and "vis") are covered by insertions
3. 15 insertion steps × 0.1 penalty = 1.5 total cost
4. Normalized by template length (25 frames): 1.5/25 = 0.06
5. Score = 1 - 0.06 = 0.94 — well above the 0.35 threshold

This causes false triggers when sentences contain substrings of the keyword. Three potential fixes:

1. **Increase step penalty** — but this also hurts legitimate matches where natural speech tempo varies
2. **Penalize length mismatch** — compare the DTW path length against the template length, add a penalty proportional to the compression ratio
3. **Check match span** — track both `best_start` and `best_end`, reject matches where the input span is too short relative to the template duration

Option 3 is the most targeted: it directly measures whether the matched audio region is plausibly long enough to contain the full keyword, without affecting the DTW scoring of well-aligned matches.
