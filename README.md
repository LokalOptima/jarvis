# Jarvis — Personalized Wake Word Detector

A CPU-only, privacy-first wake word detector that responds to a specific person saying a specific phrase. No cloud, no training data, no GPU. Enroll with a handful of recordings, and it runs forever on a single core.

Supports multiple keywords, composable action pipelines, and a client/server split for running detection on a remote machine.

Uses **Whisper Tiny** as a frozen feature extractor and **subsequence DTW** for template matching. The Whisper encoder turns audio into 384-dimensional frame sequences; DTW finds the best-matching region within a sliding window. **Silero VAD** gates the expensive detection path so encoding only runs when someone is speaking.

## Architecture

```
Enrollment (Python, runs once per keyword):

  "hey jarvis" x 15  →  Web UI: record, trim, review clips
                      →  C++ encoder  →  [T, 384] features per clip
                      →  skip onset frames  →  CMVN per clip
                      →  DBA (N clips → 1 template)  →  L2-normalize
                      →  models/templates/hey_jarvis.bin

Local mode (C++, runs always):

  SDL2 mic (16kHz)  →  2s ring buffer  →  200ms slide
                    →  Silero VAD gate (speech?)
                    →  mel spectrogram  →  Whisper Tiny encode (~10ms)
                    →  skip onset frames  →  CMVN
                    →  subsequence DTW vs all keyword templates
                    →  score > threshold?  →  DETECTED
                    →  run pipeline: transcribe → print → tmux

Server/client mode:

  Client: split pipeline at REMOTE/LOCAL boundary
          send REMOTE specs via MSG_PIPELINE  →  stream audio  →  TCP
  Server: ring buffer  →  same detection loop  →  MSG_DETECT back
                       →  resolve + run REMOTE pipeline  →  MSG_RESULT
  Client: play audio  →  run LOCAL pipeline steps
```

## Key Techniques

**Silero VAD** — A ported Silero v5 voice activity detector (LSTM-based, ~300K params) runs on every 200ms slide. Detection only fires when speech is present, keeping CPU usage near zero during silence.

**CMVN** (Cepstral Mean and Variance Normalization) — Subtracts per-dimension mean and divides by std across all frames in a window. Removes the shared "average speech" direction that makes all utterances look similar in Whisper's feature space. This is the single biggest improvement: discrimination gap 0.22 → 0.39.

**DBA** (Dynamic Barycenter Averaging) — Merges N enrollment templates into one representative template by iteratively aligning all templates via DTW and averaging the aligned frames. Captures the essential temporal pattern while smoothing per-utterance variation. Also makes runtime matching N times faster.

**Subsequence DTW** — Unlike full DTW, the template can match *anywhere* within the input window. The template must be fully consumed, but it can start and end at any position. Two-row sliding window implementation: O(n\*m) time, O(m) space.

**Step penalty** — Non-diagonal DTW transitions (insertion/deletion) incur a 0.1 penalty, discouraging pathological warping where one frame stretches to cover a long region.

**Onset skip** — Whisper's first 2 encoder frames encode "start of audio" regardless of content (cosine similarity 0.97+ across all inputs). Skipping them removes this free match that inflates all scores.

**Variable-length encoding** — Whisper.cpp was designed for 30-second chunks (1500 frames). We set `whisper_set_audio_ctx()` to the actual frame count (~100 for 2 seconds), encoding only what's needed. ~15x less work per cycle.

**Composable pipelines** — Each keyword has a pipeline of ops that mutate a `Msg` in place. Steps declare `REMOTE`/`LOCAL` placement for client/server splitting. Built-in ops: `transcribe` (eager record + STT), `weather`, `tts` (synthesize + play), `print`, `tmux` (type into active pane), `save`, `fire` (fork async), `run`. New ops: one factory function + one dict entry.

## Project Structure

```
src/
  main.cpp              CLI entry — mode dispatch (local/server/client), keyword + pipeline config
  jarvis.cpp/h          Detection engine — listen loop, VAD gate, on() convenience
  ops.cpp/h             Msg, Step, Pipeline, Placement, OPS dict, all built-in ops
  detect.cpp/h          Core detection — mel, encode, CMVN, subsequence DTW, scoring
  vad_ggml.cpp/h        Silero VAD — ggml graph-based variant (~300K params)
  server.cpp/h          TCP server — resolve client pipeline specs via OPS dict
  client.cpp/h          TCP client — pipeline split, audio streaming, local step dispatch
  net.h                 TCP framing protocol, payload helpers, dual-stack sockets
  audio_async.cpp/hpp   Ring buffer — SDL2 capture mode + push mode (for server)
  whisper.cpp/h         Vendored whisper.cpp (encoder-only, custom embedding extraction)
  encode.cpp            CLI: extract encoder embeddings from WAV files (used by enroll)
  weather.cpp/hpp       Weather fetch (wttr.in) + TTS via rokoko
  bench.cpp             Mel + encode latency benchmark

jarvis/
  __init__.py           Constants (sample rate, paths, thresholds), keyword utilities
  features.py           Feature extraction via C++ encode binary (subprocess)
  enroll.py             Web UI: record, trim, review clips, build templates
  dtw.py                DTW, DBA, CMVN, cosine distance (shared Python algorithms)

scripts/
  ablation.py           Ablation study — sweep feature combinations, report gaps
  export_silero.py      Export Silero VAD weights from ONNX to binary format
  silero_ref.py         Validate C++ VAD against ONNX reference

lib/
  ggml/                 Vendored ggml (CPU-only tensor library)
  json.hpp              nlohmann/json (single-header)

models/                 (gitignored)
  ggml-tiny.bin         Whisper Tiny GGML weights
  silero_vad.bin        Silero VAD weights
  templates/            Per-keyword DBA templates
    hey_jarvis.bin
    weather.bin

data/                   (gitignored)
  clips/                Per-keyword enrollment clips
    hey_jarvis/
    weather/
  positive.wav          Test recording with wake word (for ablation)
  negative.wav          Test recording without wake word (for ablation)
  background.wav        Test recording of room noise (for ablation)

blog/                   Technical writeups of the development process (9 posts)
```

## Prerequisites

- **uv** — Python package manager
- **cmake** + C++17 compiler
- **SDL2** development libraries (`libsdl2-dev` on Debian/Ubuntu)

```sh
# Debian/Ubuntu
sudo apt install cmake libsdl2-dev

# Whisper Tiny GGML model (40MB)
uvx hf download ggerganov/whisper.cpp ggml-tiny.bin --local-dir models
```

The Silero VAD model (`models/silero_vad.bin`) is exported from ONNX via `uv run python scripts/export_silero.py`.

## Quick Start

### 1. Enroll

```sh
make enroll
```

Opens a web UI at `localhost:8457`. Record yourself saying the wake word 10-20 times, trim each clip with the waveform editor, and save. Clips are stored in `data/clips/<keyword>/`.

### 2. Build Templates

```sh
make templates
```

Extracts features from all clips, runs CMVN + DBA, and writes per-keyword templates to `models/templates/`.

### 3. Run

```sh
make run
```

Compiles (if needed) and starts the detector. The terminal shows a colored score bar and detection events:

```
  ████████████░░░░░░│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.12   8ms
  [14:23:05] DETECTED hey_jarvis  sim=0.58
```

Green fill = score below threshold. Red = above. Yellow `│` = threshold marker.

### Server/Client Mode

Run detection on a remote machine, stream audio from a lightweight client:

```sh
# On the server
make server

# On the client
make client
./build/jarvis-client SERVER_HOST
```

The client streams 200ms audio chunks over TCP. The server runs the full detection pipeline and sends back detection events and audio responses.

### CLI Options

```
./build/jarvis [options]
  --model PATH       Whisper model      (default: models/ggml-tiny.bin)
  --vad PATH         VAD model          (default: models/silero_vad.bin)
  --server           Server mode: listen for audio over TCP
  --client HOST      Client mode: stream mic to HOST
  --port PORT        TCP port           (default: 7287)
  -h, --help         Show help
```

### Makefile Targets

```
make run              Local detection (compiles if needed)
make server           Start TCP server
make client           Build lightweight client binary
make enroll           Open enrollment web UI
make templates        Build templates from clips
make clean            Remove build directory
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

Run `uv run python scripts/ablation.py` to reproduce.

## Runtime Performance

On a modern x86 CPU, the hot loop takes ~10-15ms per 200ms cycle:
- Silero VAD: <1ms (512-sample frames)
- Mel spectrogram: ~2ms
- Whisper Tiny encode: ~8-12ms (variable-length, ~100 frames instead of 1500)
- CMVN + DTW: <1ms

The detector uses ~150MB RAM (Whisper Tiny model + Silero VAD + SDL2 audio buffer). Run `./build/bench models/ggml-tiny.bin` to measure on your hardware.

## Acknowledgments

This project builds on the work of:

- **[Whisper](https://github.com/openai/whisper)** by OpenAI — the Whisper Tiny model used as a frozen feature extractor (MIT License)
- **[whisper.cpp](https://github.com/ggerganov/whisper.cpp)** by Georgi Gerganov — vendored encoder-only fork with custom modifications (MIT License, Copyright 2023-2026 The ggml authors)
- **[ggml](https://github.com/ggml-org/ggml)** — vendored CPU tensor library (MIT License, Copyright 2023-2026 The ggml authors)
- **[Silero VAD](https://github.com/snakers4/silero-vad)** by the Silero Team — voice activity detection model ported to C++ (MIT License, Copyright 2020-present Silero Team)
- **[nlohmann/json](https://github.com/nlohmann/json)** by Niels Lohmann — JSON parsing (MIT License)
