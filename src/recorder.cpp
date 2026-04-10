/**
 * recorder.cpp - VAD-gated audio recording.
 *
 * Adapted from wip/ops.cpp transcribe step.
 * 200ms slide, 1s silence cooldown, 30s max.
 */

#include "recorder.h"
#include "detect.h"
#include <audio_async.hpp>

#include <chrono>
#include <thread>

RecordResult vad_record(SileroVad &vad, std::shared_ptr<audio_async> audio) {
    constexpr int slide_ms       = JARVIS_SLIDE_MS;   // 200
    constexpr int cooldown_max   = 1000;              // ms of silence to commit
    constexpr int max_record_ms  = 30000;

    vad.reset();

    RecordResult result;
    result.pcm.reserve(max_record_ms * JARVIS_SAMPLE_RATE / 1000);

    std::vector<float> chunk;
    int cooldown_ms = cooldown_max;
    int total_ms = 0;

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(slide_ms));

        audio->get(slide_ms, chunk);
        if (chunk.empty()) continue;

        result.pcm.insert(result.pcm.end(), chunk.begin(), chunk.end());
        total_ms += slide_ms;
        if (total_ms >= max_record_ms) break;

        bool has_speech = false;
        for (size_t i = 0; i + SileroVad::CHUNK_SAMPLES <= chunk.size();
             i += SileroVad::CHUNK_SAMPLES) {
            if (vad.process(chunk.data() + i) > 0.5f) has_speech = true;
        }

        if (!has_speech) {
            cooldown_ms -= slide_ms;
            if (cooldown_ms <= 0) break;
        } else {
            cooldown_ms = cooldown_max;
        }
    }

    result.duration_ms = total_ms;
    return result;
}
