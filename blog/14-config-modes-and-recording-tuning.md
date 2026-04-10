# Config File, Keyword Modes, and Recording Tuning

The [previous post](13-anchor-weighting-and-false-trigger-investigation.md) nailed down false trigger suppression with anchor weighting. This post covers three changes: a persistent config file for keyword metadata, a per-keyword "mode" concept surfaced in the enrollment UI, and a recording silence threshold bump.

## Persistent Config

Keywords previously existed only as filesystem state — directories under `~/.cache/jarvis/clips/` and template binaries under `~/.cache/jarvis/templates/`. Metadata like detection behavior had nowhere to live.

Now there's a TOML config at `~/.config/jarvis/config.toml` (XDG-compliant, respects `$XDG_CONFIG_HOME`):

```toml
[[keywords]]
name = "hey_jarvis"
mode = "keyword"

[[keywords]]
name = "start_recording"
mode = "voice"
```

The read/write is hand-rolled — no TOML library dependency. `_read_config()` parses the subset we use (top-level scalars, `[[keywords]]` array-of-tables with string/numeric values). `_write_config()` serializes back. Round-trips cleanly for the fields we care about.

Config is managed automatically: adding a keyword in the enrollment UI writes an entry, deleting one removes it, building templates backfills any keywords that exist on disk but aren't in the config yet.

## Keyword Modes

Each keyword now has a `mode` field — `"keyword"` (default) or `"voice"`. The distinction:

- **keyword**: trigger detection only — the engine fires an event when the phrase is spoken.
- **voice**: trigger detection + start recording — after wake word detection, the engine begins VAD-gated recording and streams the subsequent utterance for transcription.

The mode is set per-keyword in the enrollment UI via a dropdown selector that appears when a keyword tab is selected. The `/api/keywords` endpoint now returns `[{name, mode}]` objects instead of bare strings, and a new `/api/set-mode` endpoint persists mode changes to the config file.

This is plumbing only — the C++ engine doesn't read modes yet. That wiring comes next.

## Recording Silence Threshold

The VAD-gated recorder's silence cooldown was bumped from 600ms to 1000ms. At 600ms, natural pauses mid-sentence (thinking, breathing) would trigger early cutoff — the recorder would commit a partial utterance. 1s gives enough margin for conversational pauses without noticeably delaying the end-of-speech detection.

## Build Guard

`CMakeLists.txt` wraps `add_subdirectory(lib/ggml)` in `if(NOT TARGET ggml)` to prevent double-inclusion when jarvis is consumed as a subdirectory of another CMake project.
