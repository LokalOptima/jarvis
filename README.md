# Jarvis — Personalized Wake Word Detector

A CPU-only, privacy-first wake word detector that responds to a specific person saying a specific phrase. No cloud, no training data, no GPU. Enroll with a handful of recordings, and it runs forever on a single core.

Uses **Whisper Tiny** as a frozen feature extractor and **subsequence DTW** for template matching. The Whisper encoder turns audio into 384-dimensional frame sequences; DTW finds the best-matching region within a sliding window. **Silero VAD** gates the expensive detection path so encoding only runs when someone is speaking.

Optional **Core ML** backend for Apple Neural Engine acceleration on macOS.

## Architecture

```
Enrollment (Python, runs once per keyword):

  "hey jarvis" x 15  →  Web UI: record, trim, review clips
                      →  C++ encoder  →  [T, 384] features per clip
                      →  skip onset frames  →  CMVN per clip
                      →  DBA (N clips → 1 template)  →  L2-normalize
                      →  ~/.cache/jarvis/templates/hey_jarvis.bin

Detection (C++, runs always):

  SDL2 mic (16kHz)  →  2s ring buffer  →  200ms slide
                    →  Silero VAD gate (speech?)
                    →  mel spectrogram  →  Whisper Tiny encode (~10ms)
                    →  skip onset frames  →  CMVN
                    →  subsequence DTW vs all keyword templates
                    →  score > threshold?  →  DETECTED
                    →  on_detect callback
```

## Key Techniques

**Silero VAD** — A ported Silero v5 voice activity detector (LSTM-based, ~300K params) runs on every 200ms slide. Detection only fires when speech is present, keeping CPU usage near zero during silence.

**CMVN** (Cepstral Mean and Variance Normalization) — Subtracts per-dimension mean and divides by std across all frames in a window. Removes the shared "average speech" direction that makes all utterances look similar in Whisper's feature space. This is the single biggest improvement: discrimination gap 0.22 → 0.39.

**DBA** (Dynamic Barycenter Averaging) — Merges N enrollment templates into one representative template by iteratively aligning all templates via DTW and averaging the aligned frames. Captures the essential temporal pattern while smoothing per-utterance variation. Also makes runtime matching N times faster.

**Subsequence DTW** — Unlike full DTW, the template can match *anywhere* within the input window. The template must be fully consumed, but it can start and end at any position. Two-row sliding window implementation: O(n\*m) time, O(m) space.

**Step penalty** — Non-diagonal DTW transitions (insertion/deletion) incur a 0.1 penalty, discouraging pathological warping where one frame stretches to cover a long region.

**Onset skip** — Whisper's first 2 encoder frames encode "start of audio" regardless of content (cosine similarity 0.97+ across all inputs). Skipping them removes this free match that inflates all scores.

**Variable-length encoding** — Whisper.cpp was designed for 30-second chunks (1500 frames). We set `whisper_set_audio_ctx()` to the actual frame count (~100 for 2 seconds), encoding only what's needed. ~15x less work per cycle.

## Project Structure

```
src/
  main.cpp              CLI entry — local mode + server mode
  jarvis.cpp/h          Detection engine — listen loop, VAD gate, on_detect callback
  detect.cpp/h          Core detection — mel, encode, CMVN, subsequence DTW, scoring
  config.cpp/h          TOML config parser (for server mode)
  server.cpp/h          Unix socket server — accept, subscribe, broadcast events
  recorder.cpp/h        VAD-gated audio recording (for voice-mode keywords)
  playback.cpp/h        Audio playback (aplay/paplay/afplay)
  vad_ggml.cpp/h        Silero VAD — ggml graph-based variant (~300K params)
  audio_async.cpp/hpp   Ring buffer — SDL2 capture mode + push mode
  whisper.cpp/h         Vendored whisper.cpp (encoder-only, custom embedding extraction)
  coreml/               Core ML (Apple Neural Engine) encoder backend
  encode.cpp            CLI: extract encoder embeddings from WAV files (used by enroll)
  bench.cpp             Mel + encode latency benchmark

jarvis/
  __init__.py           Constants (sample rate, paths, thresholds), keyword utilities
  features.py           Feature extraction via C++ encode binary (subprocess)
  enroll.py             Web UI: record, trim, review clips, build templates
  dtw.py                DTW, DBA, CMVN, cosine distance (shared Python algorithms)

scripts/
  ablation.py           Ablation study — sweep feature combinations, report gaps
  convert_coreml.py     Convert Whisper encoder to Core ML format
  export_silero.py      Export Silero VAD weights from ONNX to binary format
  silero_ref.py         Validate C++ VAD against ONNX reference

tools/
  test_client.py        Test client for the Unix socket server

wip/                    Pipeline executor (client/server, ops) — being restructured

lib/
  ggml/                 Vendored ggml (CPU-only tensor library)
  json.hpp              nlohmann/json (single-header)
  toml.hpp              toml++ (single-header TOML parser)

~/.cache/jarvis/        Runtime data (downloaded by make)
  ggml-tiny-Q8.bin      Whisper Tiny GGML weights (CPU)
  ggml-tiny-FP16.bin    Whisper Tiny GGML weights (CoreML)
  silero_vad.bin        Silero VAD weights
  beep.wav, bling.wav   Detection notification sounds
  templates/            Per-keyword DBA templates (built by enrollment)

~/.config/jarvis/       Configuration (XDG_CONFIG_HOME)
  config.toml           Server config
```

## Server Mode

Jarvis runs as a long-lived detection server on a Unix socket. Clients connect, subscribe to keywords, and receive detection events as JSONL.

```sh
./build/jarvis-cpu --serve
```

Reads `~/.config/jarvis/config.toml`, binds `/tmp/jarvis.sock`, starts detecting.

### Config

```toml
whisper = "ggml-tiny-Q8.bin"
vad = "silero_vad.bin"
ding = "beep"
threshold = 0.35

[[keywords]]
name = "hey_jarvis"
mode = "voice"       # detection + VAD-gated recording

[[keywords]]
name = "weather"
mode = "keyword"     # detection event only
```

### Protocol (JSONL over Unix socket)

```
Client → server:   {"subscribe": ["hey_jarvis", "weather"]}
Server → client:   {"keyword": "weather", "score": 0.51}
Server → client:   {"keyword": "hey_jarvis", "score": 0.54, "audio_length": 48000}
                    <48000 × float32 raw PCM bytes>
```

For `voice` keywords, `audio_length` is the number of float32 samples (16kHz). The client reads `audio_length × 4` bytes of raw PCM immediately after the JSON line.

### Test Client

```sh
uv run tools/test_client.py                      # subscribe to all
uv run tools/test_client.py weather              # specific keyword
uv run tools/test_client.py --save-audio         # save recordings as WAV
```

## Library API

Jarvis is also a C++ library with a callback interface:

```cpp
#include "jarvis.h"

Jarvis j("~/.cache/jarvis/ggml-tiny-Q8.bin", "~/.cache/jarvis/silero_vad.bin");
j.set_ding("~/.cache/jarvis/beep.wav");

j.add_keyword({"hey_jarvis", "~/.cache/jarvis/templates/hey_jarvis.tiny-Q8.bin", 0.35f});

j.on_detect = [](const std::string &keyword, float score,
                  std::shared_ptr<audio_async> audio) {
    std::cout << keyword << " detected (score=" << score << ")" << std::endl;
    // audio is the live mic stream — use it for recording, etc.
};

j.listen();  // blocks until Ctrl+C
```

## Prerequisites

- **uv** — Python package manager
- **cmake** + C++17 compiler
- **SDL2** development libraries (`libsdl2-dev` on Debian/Ubuntu)

```sh
# Debian/Ubuntu
sudo apt install cmake libsdl2-dev
```

Models are downloaded automatically on first `make`.

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

Extracts features from all clips, runs CMVN + DBA, and writes per-keyword templates to `~/.cache/jarvis/templates/`.

### 3. Run

```sh
make run
```

Compiles (if needed) and starts the detector. The terminal shows a pinned detection bar at the bottom and detection events above:

```
  [14:23:05] hey_jarvis  sim=0.58
  [14:23:12] weather     sim=0.51

  hey_jarvis  ██████████████████│·················  0.58   12ms
```

Green fill = score below threshold. Red = above. Yellow `│` = threshold marker.

### Core ML (macOS)

```sh
# Convert model
uv run python scripts/convert_coreml.py

# Build with ANE support
cmake -B build -DWHISPER_COREML=ON && cmake --build build
```

### CLI Options

```
./build/jarvis-cpu [options]

Local mode (default):
  --model PATH       Whisper model (default: ggml-tiny.bin)
  --vad PATH         VAD model (default: silero_vad.bin)
  --ding NAME        Detection sound: beep, bling, none (default: beep)

Server mode:
  --serve            Start Unix socket server
  --config PATH      Config file (default: ~/.config/jarvis/config.toml)

Audio:
  --list-devices     List SDL2 capture devices and exit
  --device N         Capture device index (-1 = default)
```

### Makefile Targets

```
make run              Local detection (compiles if needed)
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

The detector uses ~150MB RAM (Whisper Tiny model + Silero VAD + SDL2 audio buffer). Run `./build/bench ~/.cache/jarvis/ggml-tiny-Q8.bin` to measure on your hardware.

## Acknowledgments

This project builds on the work of:

- **[Whisper](https://github.com/openai/whisper)** by OpenAI — the Whisper Tiny model used as a frozen feature extractor (MIT License)
- **[whisper.cpp](https://github.com/ggerganov/whisper.cpp)** by Georgi Gerganov — vendored encoder-only fork with custom modifications (MIT License, Copyright 2023-2026 The ggml authors)
- **[ggml](https://github.com/ggml-org/ggml)** — vendored CPU tensor library (MIT License, Copyright 2023-2026 The ggml authors)
- **[Silero VAD](https://github.com/snakers4/silero-vad)** by the Silero Team — voice activity detection model ported to C++ (MIT License, Copyright 2020-present Silero Team)
- **[nlohmann/json](https://github.com/nlohmann/json)** by Niels Lohmann — JSON parsing (MIT License)
- **[toml++](https://github.com/marzer/tomlplusplus)** by Mark Gillard — TOML parsing (MIT License)
