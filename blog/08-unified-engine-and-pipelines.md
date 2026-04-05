# Unified Engine and Composable Pipelines

The [previous posts](07-silero-vad-port.md) left us with two separate detection loops: `Jarvis::listen()` for local mode and `handle_client()` in `server.cpp`. Both did the same thing (VAD -> detect_once -> act) but the server had hardcoded weather+TTS logic instead of using the pipeline system we'd built for local mode. This meant `record_follow_up`, composable pipelines, and the transcription flow only worked locally. This post covers unifying them.

## The Problem

The code duplication was substantial. `server.cpp` had its own `RingBuffer` struct (~35 lines), its own VAD loop, its own whisper/model loading, and inline weather+TTS handling. Any improvement to the detection loop had to be applied twice. Worse, the server couldn't use the pipeline system at all — no composable steps, no `record_follow_up`, no `transcribe -> print -> tmux_type` chains.

## Injectable Audio Source

The core insight: `Jarvis::listen()` only needs a source of audio. In local mode that's SDL2 mic capture; in server mode it's TCP audio from the client. If we make the audio source injectable, one detection loop serves both.

`audio_async` gained a push mode — three new methods:

```cpp
bool init_push(int sample_rate);           // allocate ring buffer, no SDL2
void push(const float *samples, int n);    // thread-safe write into ring buffer
bool has_device() const;                   // true if backed by SDL2 device
```

`init_push()` sets up the ring buffer without touching SDL2 at all — no `SDL_Init`, no device open. `push()` is the same ring buffer write as the SDL callback, extracted to avoid duplication (the callback now just calls `push()`). The guards in `get()` and `clear()` changed from checking `m_dev_id_in` to checking `m_audio.empty()` so they work in both modes.

`Jarvis::listen()` split into two overloads:

```cpp
void listen();                                    // local: creates SDL2 audio, installs SIGINT
void listen(std::shared_ptr<audio_async> audio);  // external audio: loops until stop()
```

The first creates the SDL2 audio source and delegates to the second. The second is the real detection loop, checking only the instance-level `m_running` flag — no global state, no signal handler installation, reusable from any context.

## Server Rewrite

With the injectable audio source, `server.cpp` collapsed from ~180 lines of detection logic to ~30 lines of setup:

```cpp
// Push-mode audio source for this connection
auto audio = make_shared<audio_async>(JARVIS_BUFFER_SEC * 1000);
audio->init_push(JARVIS_SAMPLE_RATE);
audio->resume();

// TCP receiver thread: read audio from client, push into ring buffer
std::thread receiver([&]() {
    while (g_running) {
        if (!recv_msg(client_fd, hdr, payload)) { j.stop(); break; }
        if (hdr.type == MSG_AUDIO)
            audio->push((float *)payload.data(), hdr.length / sizeof(float));
    }
});

j.listen(audio);  // blocks until disconnect or SIGINT
```

The TCP receiver thread reads audio from the client and pushes it into the ring buffer. `Jarvis::listen(audio)` runs the exact same detection loop as local mode — VAD, detect_once, pipeline execution — all working identically.

## Per-Connection Pipelines

The server needs different pipeline steps than local mode. Locally, `hey_jarvis` runs `transcribe -> print -> tmux_type`. On the server, `tmux_type` would type into the server's tmux, not the client's. Instead, the server uses `send_text_to_client` which sends the transcription back via `MSG_RESPONSE`:

```cpp
j.set_pipeline("hey_jarvis", {
    transcribe("flock --shared /tmp/gpu.lock paraketto.fp8"),
    print_step(),
    send_text_to_client(client_fd),
});
j.set_pipeline("weather", {weather_response(client_fd)});
```

`set_pipeline()` and `set_record_follow_up()` let the server swap in connection-specific steps before each `listen(audio)` call. The pipeline lambdas capture `client_fd` by value — they're valid for the duration of the connection and get replaced on the next one.

## PipeStep Descriptions

Pipeline steps are `std::function` lambdas — opaque at print time. To make startup output useful, `PipeStep` became a struct with a description field:

```cpp
struct PipeStep {
    std::string desc;
    std::function<std::string(const std::string &)> fn;
    std::string operator()(const std::string &input) const { return fn(input); }
};
```

`operator()` preserves the existing call syntax so `run_pipeline` doesn't change. The factory functions fill in descriptions automatically:

```
Keywords:
  hey_jarvis: 34 frames
    -> transcribe(paraketto.fp8)
    -> print
    -> tmux_type
  weather: 44 frames
    -> run(./build/weather)
```

## The SIGCHLD Trap

First test: keywords detected, but no weather audio played. The weather pipeline called `get_weather_text()` which runs `curl` via `popen()`, but the result was always empty.

Root cause: `SIGCHLD` was set to `SIG_IGN` (for auto-reaping forked child processes). With `SIG_IGN`, child processes are reaped immediately by the kernel. When `pclose()` later calls `waitpid()`, the child is already gone — `waitpid` returns -1/ECHILD, `pclose` returns -1, and `fetch_url` interprets that as failure:

```cpp
int status = pclose(pipe);
if (status != 0) return {};  // -1 != 0 -> "failed"
```

The fix: don't set `SIGCHLD SIG_IGN` in server mode. The server's pipeline steps use `popen/pclose`, not `fork`. `SIG_IGN` is only needed in local mode (for the `fire()` step) and in the client (for `aplay` forks).

## Client Reconnection

The client now retries connecting with exponential backoff (1s -> 2s -> ... -> 10s max) and automatically reconnects on server disconnect. The mic capture persists across reconnections — only the TCP connection cycles.

## What Got Deleted

- `server.cpp`: `RingBuffer` struct, `handle_client()`, whisper/VAD loading, inline detection logic (~135 lines)
- `jarvis.cpp`: file-level `g_running` static
- `main.cpp`: server-mode template pre-loading

## What's Next

The enrollment web UI now has clip editing (load an existing clip back into the waveform editor for re-trimming). The system is feature-complete for a two-keyword setup (wake word + weather). Next steps: more keywords, better template management, and possibly moving enrollment into the C++ binary.
