// SDL2 audio capture — ring buffer with async callback.
// Based on whisper.cpp example code by Georgi Gerganov (MIT license), stripped down.

#pragma once

#include <SDL.h>
#include <SDL_audio.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

class audio_async {
public:
    audio_async(int len_ms);
    ~audio_async();

    bool init(int capture_id, int sample_rate);  // SDL2 capture mode
    bool init_push(int sample_rate);              // push mode (no SDL2, feed via push())
    bool resume();
    bool pause();
    bool clear();

    // Push audio samples into ring buffer (thread-safe, used by both SDL callback and TCP)
    void push(const float *samples, int n);

    // Called by SDL audio thread
    void callback(uint8_t *stream, int len);

    // Copy last `ms` milliseconds from ring buffer into `result`
    void get(int ms, std::vector<float> &result);

    // Number of samples captured so far (capped at ring buffer size)
    size_t available();

    // True if backed by an SDL2 audio device (false in push mode)
    bool has_device() const;

private:
    SDL_AudioDeviceID m_dev_id_in = 0;
    int m_len_ms = 0;
    int m_sample_rate = 0;

    std::atomic_bool m_running;
    std::mutex       m_mutex;

    std::vector<float> m_audio;
    size_t             m_audio_pos = 0;
    size_t             m_audio_len = 0;
};
