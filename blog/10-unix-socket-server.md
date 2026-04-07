# Unix Socket Server: Separating Jarvis from the Pipeline

The [previous post](09-ops-architecture.md) built a composable ops system for chaining detection → transcription → TTS → actions. But everything still lived in one binary: the detection engine, the pipeline executor, the TCP server, the weather fetcher. This post covers splitting jarvis into a standalone detection server and moving everything else out.

## Why Split

Jarvis, paraketto (ASR), and rokoko (TTS) are independent LokalOptima models. The old architecture linked them into one binary — jarvis couldn't run without pulling in transcription, TTS, weather fetching, and TCP framing. Adding a new consumer (a home automation trigger, a different ASR backend) meant modifying jarvis itself.

Every other wake word project — openWakeWord, Porcupine, Lowwi, Mycroft Precise — works the same way: the detector is a library or service, and the "what happens after detection" logic lives elsewhere. The Wyoming protocol (Home Assistant's voice component standard) formalizes this: wake word detectors are standalone services that emit detection events.

## What Moved

Everything that isn't detection moved to `wip/`:

- `ops.cpp/h` — Msg, Step, Pipeline, all built-in ops
- `server.cpp/h` — the old TCP server wrapping Jarvis
- `client.cpp/h` — TCP client, audio streaming
- `net.h` — TCP framing protocol
- `weather.cpp/hpp`, `jarvis-client.cpp`

What remained is the detection engine: `jarvis.cpp/h`, `detect.cpp/h`, `vad_ggml.cpp/h`, `audio_async.cpp/hpp`, `whisper.cpp/h`. The engine's API is a callback:

```cpp
Jarvis j(whisper_model, vad_model);
j.add_keyword({"hey_jarvis", template_path, 0.35f});
j.on_detect = [](const std::string &kw, float score,
                  std::shared_ptr<audio_async> audio) {
    // kw detected, audio is the live mic stream
};
j.listen();
```

The `on_detect` callback receives the audio source, so consumers can record follow-up speech without the engine knowing about recording, transcription, or anything downstream.

## Server Architecture

The new server is a Unix socket service. Clients connect, subscribe to keywords, and receive detection events as JSONL.

### Two Keyword Modes

The config (`~/.config/jarvis/config.toml`) assigns each keyword a mode:

```toml
[[keywords]]
name = "hey_jarvis"
mode = "voice"       # detection + VAD-gated recording

[[keywords]]
name = "weather"
mode = "keyword"     # detection event only
```

`keyword` mode sends a JSON line: `{"keyword":"weather","score":0.51}`. Done.

`voice` mode plays the ding, records until 600ms of silence (or 30s max), and sends: `{"keyword":"hey_jarvis","score":0.54,"audio_length":48000}` followed by `48000 × sizeof(float)` bytes of raw PCM. The client reads the JSON, sees `audio_length`, and knows to consume that many bytes of binary payload before the next JSON line.

### Why Unix Socket

TCP adds nothing here — jarvis runs on the same machine as its consumers. Unix sockets avoid port allocation, firewall rules, and accidental network exposure of a live mic stream. The socket at `/tmp/jarvis.sock` is cleaned up on shutdown. `--tcp PORT` can be added later if needed.

### Threading

Three thread types:

1. **Main thread**: `poll()` on the listen socket with 1s timeout, accept new clients, reap dead ones periodically.
2. **Detection thread**: `Jarvis::listen(audio)` — owns the mic, runs the VAD + detection loop. The `on_detect` callback runs here and blocks detection during recording (voice mode). This is intentional: you don't want to detect "hey jarvis" while recording the command that follows it.
3. **Per-client reader thread**: reads `{"subscribe": [...]}` messages, updates the client's subscription set, detects disconnect.

### Broadcast Without Blocking Detection

The first version held the global client list mutex during `write()` syscalls. A single stalled client (full kernel buffer, stopped reading) would block the entire broadcast, which blocks `on_detect`, which blocks detection. Every client suffers because one is slow.

Fix: snapshot the subscribed client list under the lock, release it, then send to each client using only the per-client mutex:

```cpp
static void broadcast(const std::string &keyword, const std::string &jsonl,
                      const float *pcm, int n_samples) {
    std::vector<std::shared_ptr<Client>> targets;
    {
        std::lock_guard<std::mutex> lk(g_clients_mu);
        for (auto &c : g_clients)
            if (c->alive && c->subscriptions.count(keyword))
                targets.push_back(c);
    }
    for (auto &c : targets)
        send_to(*c, jsonl, pcm, n_samples);
}
```

A slow client only blocks itself.

## Recording VAD

Voice-mode keywords need a second VAD instance. The detection loop's VAD tracks speech to gate the expensive encode path; the recording VAD tracks speech-to-silence transitions to know when the user finished talking. These are different state machines — detection VAD resets after every detection, recording VAD starts fresh when recording begins.

The recorder is extracted from the old `ops.cpp` transcribe step, stripped of speculative transcription and render calls:

```cpp
RecordResult vad_record(SileroVad &vad, std::shared_ptr<audio_async> audio);
```

200ms slide, accumulate PCM, VAD each chunk. 600ms of continuous silence → commit. 30s hard cap. Returns raw PCM. ~40 lines.

Since `on_detect` blocks the detection loop, there's no concurrency issue — recording naturally pauses detection. The detection VAD and audio buffer are reset after the callback returns.

## Config

The config file is TOML, parsed with toml++ (single-header, MIT). It's meant to be generated by enrollment, not hand-written — enrollment knows which keywords exist, which model was used, and what threshold works. The config just captures that state:

```toml
whisper = "ggml-tiny-Q8.bin"
vad = "silero_vad.bin"
ding = "beep"
threshold = 0.35

[[keywords]]
name = "hey_jarvis"
mode = "voice"

[[keywords]]
name = "weather"
mode = "keyword"
```

Model filenames resolve to `~/.cache/jarvis/` (XDG_CACHE_HOME). Template paths are derived from keyword name + model tag via existing `template_path()`. No per-keyword thresholds — a global threshold is simpler and the DTW scores are already normalized.

## What the CLI Looks Like

```
./build/jarvis-cpu [options]

Local mode (default):
  --model PATH       whisper model
  --ding NAME        beep, bling, none

Server mode:
  --serve            start Unix socket server
  --config PATH      config file

Audio:
  --list-devices     list SDL2 capture devices
  --device N         capture device index
```

Local mode still works with hardcoded keywords for quick testing. Server mode is config-driven.

## File Changes

```
Created:   src/config.h/cpp      Config struct + TOML parser (~90 lines)
           src/recorder.h/cpp    VAD-gated recording (~55 lines)
           src/server.h/cpp      Unix socket server (~290 lines)
           tools/test_client.py  Python test client
           lib/toml.hpp          Vendored toml++ v3.4.0

Modified:  src/main.cpp          +50 lines (--serve, --config, --list-devices, --device)
           CMakeLists.txt        +3 lines (new source files)
```

## What's Next

The pipeline executor in `wip/` becomes a client: connect to `/tmp/jarvis.sock`, subscribe to keywords, receive detections, orchestrate paraketto and rokoko. Jarvis doesn't know or care what happens downstream.
