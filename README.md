# Jarvis — Personalized Wake Word Detector

A CPU-only, privacy-first wake word detector that responds to a specific person saying a specific phrase. No cloud, no training data, no GPU. Enroll with a handful of recordings, and it runs forever on a single core.

Uses **Whisper Tiny** as a frozen feature extractor and **subsequence DTW** for template matching. The Whisper encoder turns audio into 384-dimensional frame sequences; DTW finds the best-matching region within a sliding window. CMVN and DBA collapse 10-20 enrollment recordings into a single robust template.

## Architecture

```
Enrollment (Python, runs once):

  "hey Jarvis" x 18  →  Whisper Turbo (word timestamps)  →  clip extraction
                         →  Whisper Tiny encoder  →  [T, 384] features per clip
                         →  skip onset frames  →  CMVN per clip
                         →  DBA (18 clips → 1 template)  →  L2-normalize
                         →  templates.bin

Runtime (C++, runs always):

  SDL2 mic (16kHz)  →  2s ring buffer  →  energy VAD gate
                    →  mel spectrogram  →  Whisper Tiny encode (~15ms)
                    →  skip onset frames  →  CMVN
                    →  subsequence DTW vs template  →  score > threshold?
                    →  DETECTED
```

The enrollment pipeline uses **Whisper Turbo** (a much larger model) to find wake word timestamps with word-level precision, then **Whisper Tiny** to extract the features that the runtime will match against. The runtime only needs Tiny.

## Key Techniques

**CMVN** (Cepstral Mean and Variance Normalization) — Subtracts per-dimension mean and divides by std across all frames in a window. Removes the shared "average speech" direction that makes all utterances look similar in Whisper's feature space. This is the single biggest improvement: discrimination gap 0.22 → 0.39.

**DBA** (Dynamic Barycenter Averaging) — Merges N enrollment templates into one representative template by iteratively aligning all templates via DTW and averaging the aligned frames. Captures the essential temporal pattern while smoothing per-utterance variation. Also makes runtime matching N times faster.

**Subsequence DTW** — Unlike full DTW, the template can match *anywhere* within the input window. The template must be fully consumed, but it can start and end at any position. Two-row sliding window implementation: O(n*m) time, O(m) space.

**Step penalty** — Non-diagonal DTW transitions (insertion/deletion) incur a 0.1 penalty, discouraging pathological warping where one frame stretches to cover a long region.

**Onset skip** — Whisper's first 2 encoder frames encode "start of audio" regardless of content (cosine similarity 0.97+ across all inputs). Skipping them removes this free match that inflates all scores.

**Variable-length encoding** — Whisper.cpp was designed for 30-second chunks (1500 frames). We set `exp_n_audio_ctx` to the actual frame count (~100 for 2 seconds), encoding only what's needed. ~15x less work per cycle.

## Project Structure

```
src/
  jarvis.cpp          C++ runtime — the always-on detector (~370 lines)
  audio_async.cpp/hpp  SDL2 audio ring buffer (from whisper.cpp examples)

jarvis/
  __init__.py          Shared constants (sample rate, paths, thresholds)
  features.py          Whisper Tiny feature extraction (Python, variable-length + layer selection)
  enroll.py            Enrollment pipeline: record → detect → extract clips → build templates
  review.py            Web UI for enrollment: record, upload, review clips, build templates
  dtw.py               Shared DTW utilities: cmvn, dba, subdtw, l2norm
  ablation.py          Ablation study — tests all improvement combinations

lib/
  whisper.cpp/         Vendored, stripped whisper.cpp (encoder-only, 2859 lines from 7400)
                       Patched with whisper_encoder_output() and whisper_set_audio_ctx()

models/                (gitignored)
  ggml-tiny.bin        Whisper Tiny GGML weights for C++ runtime
  templates.bin        Built template (1 DBA template, ~50KB)

data/                  (gitignored)
  clips/               Extracted wake word clips (~18 WAV files)
  positive.wav         Test recording with wake word (for ablation)
  negative.wav         Test recording without wake word (for ablation)
  background.wav       Test recording of room noise (for ablation)

blog/                  Technical writeups of the development process
```

## Prerequisites

- **uv** — Python package manager
- **cmake** + C++17 compiler
- **SDL2** development libraries (`libsdl2-dev` on Debian/Ubuntu)
- **ffmpeg** — required by openai-whisper for audio format conversion

```sh
# Debian/Ubuntu
sudo apt install cmake libsdl2-dev ffmpeg

# The Whisper Tiny GGML model (40MB)
uvx hf download ggerganov/whisper.cpp ggml-tiny.bin --local-dir models
```

## Quick Start

### 1. Enroll

Record yourself saying the wake word 10-20 times:

```sh
make enroll                              # live mic recording
# or
uv run python -m jarvis.enroll file.wav  # from a pre-recorded file
```

Whisper Turbo detects each utterance and extracts individual clips to `data/clips/`.

### 2. Review

```sh
make review
```

Opens a web UI at `localhost:8457`. Listen to each clip, delete bad ones (cut off, noisy, misdetections). You can also record and upload directly from the browser.

### 3. Build Templates

```sh
make build
```

Runs CMVN + DBA on the clips and writes `models/templates.bin`.

### 4. Compile and Run

```sh
make compile
make run
```

The detector prints a colored score bar to stderr and detection events to stdout:

```
  ████████████░░░░░░│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.12   8ms
  [14:23:05] DETECTED  sim=0.58
```

Green fill = score below threshold. Red = above. Yellow `│` = threshold marker.

### CLI Options

```
./build/jarvis [options]
  -m <path>   Whisper model      (default: models/ggml-tiny.bin)
  -e <path>   Template file      (default: models/templates.bin)
  -t <float>  Detection threshold (default: 0.35)
```

## Ablation Results

Tested against three recordings: positive (contains wake word), negative (speech without wake word), background (room noise). Gap = positive score - negative score.

| Config | Positive | Negative | Background | Gap |
|--------|----------|----------|------------|-----|
| Baseline | 0.851 | 0.627 | 0.627 | 0.224 |
| + skip 2 frames | 0.861 | 0.621 | 0.621 | 0.240 |
| + CMVN | 0.674 | 0.269 | 0.189 | 0.405 |
| + step penalty 0.1 | 0.653 | 0.229 | 0.131 | 0.424 |
| + DBA (18 → 1 template) | 0.594 | 0.146 | 0.088 | 0.449 |
| **+ layer 3** | **0.582** | **0.117** | **0.058** | **0.465** |

Run `uv run python -m jarvis.ablation` to reproduce.

## Runtime Performance

On a modern x86 CPU, the hot loop takes ~15-20ms per 200ms cycle:
- Mel spectrogram: ~2ms
- Whisper Tiny encode: ~12-15ms (variable-length, ~100 frames instead of 1500)
- CMVN + DTW + render: <1ms

The detector uses ~150MB RAM (Whisper Tiny model + SDL2 audio buffer).
