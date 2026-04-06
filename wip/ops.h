/**
 * ops.h - Pipeline types, built-in ops, and engine singletons.
 *
 * A pipeline is a vector of Steps. Each Step mutates a Msg in place.
 * Ops are created by factory functions with uniform signature:
 *     Step op(const std::string &params)
 *
 * To add a new op: write the factory in ops.cpp, add one entry to OPS.
 */

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class audio_async;
class SileroVad;

// ---- Core types ----

struct Msg {
    std::string                  keyword;   // which keyword triggered this
    std::string                  text;      // transcription, weather text, etc.
    std::vector<uint8_t>         audio;     // WAV bytes for playback (may be empty)
    std::shared_ptr<audio_async> source;    // audio stream for recording ops
};

enum Placement { REMOTE, LOCAL };

struct Step {
    std::string                  name;
    std::string                  params;    // stored for client->server serialization
    Placement                    placement;
    std::function<void(Msg&)>    op;
};

using Pipeline = std::vector<Step>;

// ---- Engine singletons (set by Jarvis constructor) ----
// Constraint: only one Jarvis instance per process.

void set_engine_singletons(SileroVad *v, std::atomic<bool> *r);
SileroVad         &vad();
std::atomic<bool> &running();

// ---- Pipeline execution ----

void run_pipeline(const Pipeline &steps, Msg &msg);

// ---- Audio playback ----

// Write WAV to temp file and play via aplay/paplay/afplay.
// If wait=true, blocks until playback finishes. If wait=false, fire-and-forget.
void play_wav(const uint8_t *data, size_t size, bool wait);

// ---- Op factories ----
// All have signature: Step op(const std::string &params)

using OpFactory = Step(*)(const std::string &params);
extern const std::unordered_map<std::string, OpFactory> OPS;

// Record + eager transcribe -> msg.text.  params = paraketto binary path.
Step transcribe(const std::string &params);
// Fetch weather forecast -> msg.text.  params ignored.
Step weather(const std::string &params);
// TTS msg.text -> msg.audio, play + pause/resume mic.  params = TTS binary path.
Step tts(const std::string &params);
// Print msg.text to stdout.  params ignored.
Step print(const std::string &params);
// Type msg.text into active tmux pane.  params ignored.
Step tmux(const std::string &params);
// Append msg.text to file.  params = file path.
Step save(const std::string &params);
// Fork+exec a shell command (async). params = command.
Step fire(const std::string &params);
// Run a shell command synchronously. params = command.
Step run(const std::string &params);
