// SDL2 audio capture — ring buffer with async callback.
// Based on whisper.cpp example code by Georgi Gerganov (MIT license), stripped down.

#include <audio_async.hpp>

#include <cstdlib>
#include <cstring>

audio_async::audio_async(int len_ms) : m_len_ms(len_ms), m_running(false) {}

audio_async::~audio_async() {
    if (m_dev_id_in) SDL_CloseAudioDevice(m_dev_id_in);
}

bool audio_async::init(int capture_id, int sample_rate) {
    // Prevent SDL from probing X11/XCB when we only need audio
    SDL_SetHint(SDL_HINT_VIDEODRIVER, "dummy");

#ifdef __linux__
    // Prefer native PipeWire over pulseaudio compat layer (pipewire-pulse),
    // which intermittently fails to route the capture source.
    setenv("SDL_AUDIODRIVER", "pipewire", 0);  // 0 = don't overwrite user's choice
#endif

    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return false;
    }

    SDL_AudioSpec want = {}, got = {};
    want.freq     = sample_rate;
    want.format   = AUDIO_F32;
    want.channels = 1;
    want.samples  = 1024;
    want.callback = [](void *userdata, uint8_t *stream, int len) {
        static_cast<audio_async *>(userdata)->callback(stream, len);
    };
    want.userdata = this;

    if (capture_id >= 0) {
        m_dev_id_in = SDL_OpenAudioDevice(
            SDL_GetAudioDeviceName(capture_id, SDL_TRUE), SDL_TRUE, &want, &got, 0);
    } else {
        m_dev_id_in = SDL_OpenAudioDevice(nullptr, SDL_TRUE, &want, &got, 0);
    }

    if (!m_dev_id_in) {
        fprintf(stderr, "SDL_OpenAudioDevice failed: %s\n", SDL_GetError());
        return false;
    }

    m_sample_rate = got.freq;
    m_audio.resize((m_sample_rate * m_len_ms) / 1000);
    return true;
}

bool audio_async::resume() {
    if (!m_dev_id_in || m_running) return false;
    SDL_PauseAudioDevice(m_dev_id_in, 0);
    m_running = true;
    return true;
}

bool audio_async::pause() {
    if (!m_dev_id_in || !m_running) return false;
    SDL_PauseAudioDevice(m_dev_id_in, 1);
    m_running = false;
    return true;
}

bool audio_async::clear() {
    if (!m_dev_id_in || !m_running) return false;
    std::lock_guard<std::mutex> lock(m_mutex);
    m_audio_pos = 0;
    m_audio_len = 0;
    return true;
}

void audio_async::callback(uint8_t *stream, int len) {
    if (!m_running) return;

    size_t n_samples = len / sizeof(float);
    if (n_samples > m_audio.size()) {
        n_samples = m_audio.size();
        stream += (len - (n_samples * sizeof(float)));
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_audio_pos + n_samples > m_audio.size()) {
        size_t n0 = m_audio.size() - m_audio_pos;
        memcpy(&m_audio[m_audio_pos], stream, n0 * sizeof(float));
        memcpy(&m_audio[0], stream + n0 * sizeof(float), (n_samples - n0) * sizeof(float));
        m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
        m_audio_len = m_audio.size();
    } else {
        memcpy(&m_audio[m_audio_pos], stream, n_samples * sizeof(float));
        m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
        m_audio_len = std::min(m_audio_len + n_samples, m_audio.size());
    }
}

void audio_async::get(int ms, std::vector<float> &result) {
    result.clear();
    if (!m_dev_id_in || !m_running) return;

    std::lock_guard<std::mutex> lock(m_mutex);

    if (ms <= 0) ms = m_len_ms;

    size_t n_samples = (m_sample_rate * ms) / 1000;
    if (n_samples > m_audio_len) n_samples = m_audio_len;

    result.resize(n_samples);

    size_t s0 = (m_audio_pos - n_samples + m_audio.size()) % m_audio.size();

    if (s0 + n_samples > m_audio.size()) {
        size_t n0 = m_audio.size() - s0;
        memcpy(result.data(), &m_audio[s0], n0 * sizeof(float));
        memcpy(&result[n0], &m_audio[0], (n_samples - n0) * sizeof(float));
    } else {
        memcpy(result.data(), &m_audio[s0], n_samples * sizeof(float));
    }
}
