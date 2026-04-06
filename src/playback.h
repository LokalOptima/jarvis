#pragma once
#include <cstddef>
#include <cstdint>

// Write WAV to temp file and play via aplay/paplay/afplay.
// If wait=true, blocks until playback finishes. If wait=false, fire-and-forget.
void play_wav(const uint8_t *data, size_t size, bool wait);
