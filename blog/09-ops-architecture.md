# Ops Architecture: Separating Detection from Actions

The [previous post](08-unified-engine-and-pipelines.md) unified the server and local detection loops, but the `Keyword` struct was still doing too much. It carried detection config (`name`, `template_path`, `threshold`), pipeline behavior (`pipeline`, `record_follow_up`), transcription config (`transcribe_cmd`), and client/server wiring (`server_pipeline`). Adding a keyword meant understanding which fields mattered in which mode. This post covers the rewrite that separates the engine from ops.

## The Problem

Adding a new keyword required touching multiple files and knowing which combination of `record_follow_up`, `transcribe_cmd`, and `server_pipeline` to set depending on the mode. The server had hardcoded `apply_config`/`apply_default_config` functions that mapped pipeline names to specific steps. The pipeline steps were `string -> string` functions — fine for "transcribe returns text, print prints text" but awkward for ops that produce audio or need access to the mic stream.

## Msg as Pipeline Currency

The old `PipeStep` was `std::function<std::string(const std::string &)>` — each step received the previous step's string output. This broke down when we needed ops that produce WAV audio, read from the mic, or set multiple fields. The new currency is a mutable `Msg`:

```cpp
struct Msg {
    std::string                  keyword;   // which keyword triggered this
    std::string                  text;      // transcription, weather text, etc.
    std::vector<uint8_t>         audio;     // WAV bytes for playback
    std::shared_ptr<audio_async> source;    // audio stream for recording ops
};
```

Each step mutates `Msg` in place. The `source` field gives recording ops direct access to the audio stream — no special `record_follow_up` flag needed. The `audio` field carries WAV bytes from TTS to playback without serialization hacks.

## Steps with Placement

Each step declares where it runs:

```cpp
enum Placement { REMOTE, LOCAL };

struct Step {
    std::string                  name;
    std::string                  params;    // stored for wire serialization
    Placement                    placement;
    std::function<void(Msg&)>    op;
};
```

In CLI mode, all steps run in-process regardless of placement. In client/server mode, the client splits the pipeline at the REMOTE/LOCAL boundary: remote specs are sent to the server via `MSG_PIPELINE`, local steps run on results received back. The `params` field exists solely so the client can serialize step names and parameters for the wire protocol.

## The OPS Dict

All op factories share a uniform signature and live in a single registry:

```cpp
using OpFactory = Step(*)(const std::string &params);

const std::unordered_map<std::string, OpFactory> OPS = {
    {"transcribe", transcribe},
    {"weather",    weather},
    {"tts",        tts},
    {"print",      print},
    {"tmux",       tmux},
    {"save",       save},
    {"fire",       fire},
    {"run",        run},
};
```

The server resolves client pipeline specs with `OPS.at(step_name)(step_params)`. Adding a new op means writing one factory function and one dict entry.

## Engine Singletons

The `transcribe` op needs VAD (to detect silence during recording) and the running flag (to abort on Ctrl+C). Rather than passing these through every call, they're exposed as process-wide singletons set by the Jarvis constructor:

```cpp
SileroVad         &vad();
std::atomic<bool> &running();
```

This constrains the design to one Jarvis instance per process, which is fine — there's only ever one detection engine running.

## Eager Transcription

The transcribe op records audio from `msg.source` and fires speculative transcriptions on speech-to-silence edges:

1. Every 200ms, read audio from `msg.source`, feed to VAD
2. On first silence tick after speech: save buffer to WAV, fire async transcription
3. If speech resumes: mark speculative result as stale, reset cooldown
4. If silence continues for 600ms: commit the speculative result as `msg.text`

This means transcription is already running during the last 600ms of silence, so the result is available almost instantly when the user stops talking.

## Killing the Refractory Period

The old code had a 2-second refractory cooldown after each detection to prevent double-triggers. But `vad.reset() + audio.clear()` after detection already wipes the buffer — the wake word audio is gone. The refractory was dead wait for no reason. Removed entirely.

## The SIGCHLD Saga Continues

The tts op uses `speak_to_wav()` which calls rokoko via `popen`. The old CLI mode set `SIGCHLD = SIG_IGN` for fire-and-forget child processes. Same bug as before: `pclose` returns -1/ECHILD, `speak_to_wav` thinks it failed, returns empty.

Fix: removed `SIGCHLD = SIG_IGN` from CLI mode entirely. The `fire()` op now uses double-fork to avoid zombies — child forks grandchild, child exits immediately (reaped by parent via `waitpid`), grandchild is reparented to init.

## The Server TTS Hang

The tts op was unconditionally forking `aplay` on whatever machine it ran on — including the headless server. If `aplay` hung (no PulseAudio session, wrong device), `waitpid` blocked forever, the pipeline never finished, `on_result` never fired, and the client never received `MSG_RESULT`.

Fix: tts only plays locally when `msg.source->has_device()` is true (CLI mode with real mic). On the server (push audio, no device), it just synthesizes WAV into `msg.audio` and returns. The `on_result` hook sends it to the client for playback.

## What main.cpp Looks Like Now

```cpp
// CLI mode
Jarvis j(whisper_model, vad_model);

j.on("hey_jarvis", "models/templates/hey_jarvis.bin", {
    transcribe(PARAKETTO),
    print(""),
    tmux(""),
});
j.on("weather", "models/templates/weather.bin", {
    weather(""),
    tts(""),
});

j.listen();
```

Adding a keyword: one `j.on()` call. Adding an op: one factory function + one dict entry. No mode-specific branches, no `record_follow_up` flags, no `transcribe_cmd` strings.

## What Got Deleted

- `jarvis.h/cpp`: `PipeStep`, `record_follow_up`, `transcribe_cmd`, `server_pipeline`, `record_until_silence()`, `record_and_transcribe()`, all built-in step functions, `run_pipeline(string)`, refractory period
- `server.cpp`: `send_text_to_client()`, `weather_response()`, `apply_config()`, `apply_default_config()` with hardcoded pipeline logic
- `main.cpp`: `--detect-only` flag, `TRANSCRIBE_CMD` constant

## What Got Created

- `ops.h/cpp`: `Msg`, `Step`, `Pipeline`, `Placement`, `OPS` dict, `run_pipeline(Msg&)`, engine singletons, `play_wav()` helper, all built-in ops
- `client.h`: `ClientKeyword` struct with full pipeline (replaces old `Keyword` reuse)

## File Sizes

```
Before:  jarvis.cpp 503 lines, server.cpp 233 lines
After:   jarvis.cpp 199 lines, ops.cpp 286 lines, server.cpp 216 lines
```

The total line count is similar, but the code is distributed by concern rather than by mode.
