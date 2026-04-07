# Public Release: Zero-Setup Build System

The [previous post](10-unix-socket-server.md) added a Unix socket server with JSONL protocol. This post covers making the repo public and ensuring a fresh clone builds and runs without manual file shuffling.

## The Problem

Runtime files were scattered across two local directories:

- `models/` — whisper model, VAD model, keyword templates
- `data/` — ding sounds (beep.wav, bling.wav)

Both were gitignored. A fresh clone had no models, no sounds, and error messages that said "copy ggml-tiny.bin and silero_vad.bin" without saying where to get them. The Python enrollment tools wrote templates to `models/templates/`, but the C++ detector read from `~/.cache/jarvis/templates/`. Two different paths for the same files.

## The Fix

One directory: `~/.cache/jarvis/`. Models, ding sounds, and templates all live there. The Makefile downloads everything automatically on first build.

### Auto-Download in Make

The build target depends on model files in `~/.cache/jarvis/`. If they don't exist, a pattern rule fetches them from the GitHub release:

```makefile
CACHE   := $(HOME)/.cache/jarvis
RELEASE := https://github.com/LokalOptima/jarvis/releases/download/models-v1

$(CACHE)/%.bin $(CACHE)/%.wav:
    @mkdir -p $(dir $@)
    @curl -fsSL -o $@.tmp $(RELEASE)/$(notdir $@) && mv $@.tmp $@
```

The `mv` is important: if curl is interrupted, a partial file would trick make into thinking the download succeeded. Writing to a temp file and atomically renaming means a failed download leaves nothing behind.

### Platform-Aware Models

macOS with CoreML needs different files than Linux:

```makefile
ifeq ($(UNAME),Darwin)
    MODEL := ggml-tiny-FP16.bin       # full precision for ANE
else
    MODEL := ggml-tiny-Q8.bin         # quantized for CPU
endif
```

On macOS, the CoreML encoder bundle (`.mlmodelc` directory) is also downloaded and extracted from a tarball. The whisper runtime finds it by replacing `.bin` with `-encoder.mlmodelc` in the model path — so `ggml-tiny-FP16.bin` automatically looks for `ggml-tiny-FP16-encoder.mlmodelc` alongside it.

### Unified Path Resolution

Before, three systems each had their own idea of where files lived:

| System | Models | Templates |
|--------|--------|-----------|
| C++ detector | `~/.cache/jarvis/` | `~/.cache/jarvis/templates/` |
| Python enrollment | `<project>/models/` | `<project>/models/templates/` |
| Makefile | N/A | N/A |

Now everything points to `~/.cache/jarvis/`. The Python package exports `CACHE_DIR` from `jarvis/__init__.py`, and all scripts import it instead of hardcoding paths. The C++ side already had `cache_dir()` centralized in `detect.cpp`.

Ding sounds also moved out of `data/` into the cache dir. The `data/` directory is now purely for enrollment clips — user recordings that are always local and never distributed.

## Fresh Clone Workflow

```sh
git clone https://github.com/LokalOptima/jarvis
cd jarvis
sudo apt install cmake libsdl2-dev   # prerequisites
make                                  # downloads models, builds
make enroll                           # record wake word samples
make templates                        # build templates from clips
make run                              # detect
```

Five commands from zero to a running wake word detector. The first `make` takes a minute (42MB model download + compile). Subsequent builds skip the download entirely — make sees the files exist and goes straight to cmake.

## What's in the Release

All runtime assets are in a single GitHub release (`models-v1`):

| File | Size | Purpose |
|------|------|---------|
| `ggml-tiny-Q8.bin` | 42 MB | Whisper Tiny, 8-bit quantized (CPU) |
| `ggml-tiny-FP16.bin` | 74 MB | Whisper Tiny, FP16 (CoreML/ANE) |
| `ggml-tiny-FP16-encoder.mlmodelc.tar.gz` | 14 MB | CoreML encoder bundle |
| `silero_vad.bin` | 1.2 MB | Silero VAD weights |
| `beep.wav` | 193 KB | Detection ding sound |
| `bling.wav` | 338 KB | Alternative ding sound |
