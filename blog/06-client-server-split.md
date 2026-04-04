# Client/Server Split

The [previous post](05-quantization-and-unified-encoder.md) got the pipeline to 67x real-time with Q8_0 quantization and a unified C++ encoder. But the whole point of this project is a voice-activated home assistant — and the Raspberry Pi 4 can't run the whisper encoder (830ms, 4x over the 200ms slide budget). This post covers splitting the system into a lightweight client on the Pi and a server on the desktop.

## Architecture

The detection pipeline is:

```
mic → ring buffer → VAD → mel → encode (28ms) → DTW (<0.5ms) → callback
```

Everything except mel+encode is trivially fast. The natural split: the Pi captures audio and streams it to the desktop, which runs the expensive inference and sends detection events back.

```
Pi (client)                            Desktop (server)
┌──────────────┐      TCP/7287     ┌──────────────────┐
│ SDL2 mic     │ ── AUDIO_CHUNK → │ ring buffer → VAD │
│              │                   │ → mel → encode    │
│ log/callback │ ← DETECTION ──── │ → DTW → match     │
└──────────────┘                   └──────────────────┘
```

The client sends 200ms PCM chunks (3200 float32 samples = 12.5 KB) every slide. The server accumulates these into its own 2-second ring buffer and runs `detect_once()` on each slide — the exact same function the local mode uses.

## Wire Protocol

Minimal framed binary over TCP. Every message is a 5-byte header (uint8 type + uint32 length, little-endian) followed by payload:

| Direction | Type | Payload |
|-----------|------|---------|
| client → server | `AUDIO` (0x01) | 3200 × float32 PCM |
| server → client | `DETECT` (0x81) | null-terminated keyword + float32 score |
| server → client | `STATUS` (0x82) | uint8 (buffering/ready) |

`TCP_NODELAY` on both ends — at 62 KB/s, Nagle would add up to 200ms of buffering latency, blowing the entire budget.

## Refactoring the Detection Pipeline

The core detection logic (mel → encode → CMVN → DTW) lived inline in `Jarvis::listen()`. To share it between local mode and the server, I extracted it into `detect_once()` in a new `detect.cpp`:

```cpp
DetectResult detect_once(
    whisper_context *ctx,
    const std::vector<LoadedKeyword> &keywords,
    const float *pcm, int n_samples,
    DetectScratch &scratch);
```

`DetectScratch` holds pre-allocated buffers (encoder output, DTW rows, padded PCM) to avoid per-call heap allocation. Both `Jarvis::listen()` and the server's `handle_client()` call the same function — the detection behavior is identical.

The template matching structs (`Template`, `Templates`, `LoadedKeyword`) moved to `detect.h`, and the terminal visualizer (`render_bar`) became a shared function. `jarvis.cpp` went from 430 lines to 160 — just the SDL2 audio setup, the listen loop, and callbacks.

## The Lightweight Client

The Pi doesn't need whisper, ggml, or the model file. It only needs SDL2 for mic capture and POSIX sockets for networking. So I made a separate build target:

```cmake
add_executable(jarvis-client
    src/jarvis-client.cpp
    src/audio_async.cpp
)
target_link_libraries(jarvis-client ${SDL2_LIBRARIES} pthread)
```

Result: **40 KB binary** vs 218 KB for the full jarvis. Dependencies: SDL2 + libc. Build time: seconds. On the Pi:

```
make client
./build/jarvis-client gondola
```

No ggml checkout, no model download, no Python.

## Tailscale Networking

Both machines are on the same Tailscale network. MagicDNS gives stable hostnames — `gondola` resolves to the desktop's Tailscale IP from anywhere. No hardcoded IPs, no mDNS, no discovery protocol. The client just connects to `gondola:7287`.

## Latency Budget

| Stage | Time |
|-------|------|
| Network: Pi → server (12.5 KB) | ~1ms |
| Server: ring buffer + VAD | <0.1ms |
| Server: mel + encode (Q8_0) | 28ms |
| Server: DTW | <0.5ms |
| Network: server → Pi (detection event) | ~1ms |
| **Total** | **~31ms** |

Well within the 200ms slide interval. The limiting factor is still the slide period itself — in the worst case, a wake word completes just after a slide and waits 200ms for the next inference cycle. The network adds negligible latency.

## Server as Headless Service

The server doesn't need a terminal visualizer — it's a background service. It just logs connections and detections:

```
Listening on port 7287 (Ctrl+C to stop)
Client connected
  Detecting...
  [23:41:27] hey_jarvis  sim=0.533
  [23:41:35] hey_jarvis  sim=0.550
Client disconnected
```

The refractory period (cooldown after detection) is managed server-side. The client doesn't need to know about it — the server simply won't send detection events during cooldown.

## macOS Support

Three changes to make the build work on macOS:
1. Disabled all non-CPU ggml backends — macOS auto-detects BLAS via Accelerate, which fails to link with our CPU-only vendored ggml
2. Replaced `nproc` with `getconf _NPROCESSORS_ONLN` in the Makefile
3. Wrapped the PipeWire audio driver hint in `#ifdef __linux__` (macOS uses CoreAudio)
4. Added `MSG_NOSIGNAL`/`SO_NOSIGPIPE` portability for the networking code

## What's Next

The server detects the wake word but the callback runs locally — currently that means the response (weather TTS, etc.) plays on the server's speakers, not where the user is. The next step is having the server run callbacks and stream the response (text or audio) back to the client for playback.
