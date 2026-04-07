/**
 * recorder.h - VAD-gated audio recording.
 *
 * Records from audio_async until silence timeout (600ms) or max duration (30s).
 * Uses its own SileroVad instance (separate from the detection VAD).
 */

#pragma once

#include "vad_ggml.h"
#include <memory>
#include <vector>

class audio_async;

struct RecordResult {
    std::vector<float> pcm;   // raw PCM float32 @ 16kHz
    int duration_ms = 0;
};

// Record VAD-gated audio. Blocks until silence or max duration.
// vad: pre-loaded VAD instance (caller owns, will be reset before use).
// audio: audio source to read from.
RecordResult vad_record(SileroVad &vad, std::shared_ptr<audio_async> audio);
