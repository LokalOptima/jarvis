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

    bool init(int capture_id, int sample_rate);
    bool resume();
    bool pause();
    bool clear();

    // Called by SDL audio thread
    void callback(uint8_t *stream, int len);

    // Copy last `ms` milliseconds from ring buffer into `result`
    void get(int ms, std::vector<float> &result);

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
