/**
 * server.cpp - Jarvis server: receive PCM audio, run detection, send events.
 *
 * Protocol:
 *   Client sends AUDIO chunks (200ms = 3200 float32 samples).
 *   Server accumulates into a 2s ring buffer, runs detect_once() each slide,
 *   and sends DETECT messages on keyword detection.
 */

#include "server.h"
#include "net.h"
#include "vad_ggml.h"
#include "weather.hpp"
#include "whisper.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <sys/time.h>
#include <thread>

static std::atomic<bool> g_running{true};
static void signal_handler(int) { g_running.store(false, std::memory_order_relaxed); }

// Simple ring buffer for float PCM (no SDL2 dependency).
struct RingBuffer {
    std::vector<float> data;
    int write_pos = 0;
    int filled = 0;  // how many samples have been written total (capped at capacity)

    void init(int capacity) {
        data.resize(capacity, 0.0f);
        write_pos = 0;
        filled = 0;
    }

    void push(const float *samples, int n) {
        int cap = (int)data.size();
        for (int i = 0; i < n; i++) {
            data[write_pos] = samples[i];
            write_pos = (write_pos + 1) % cap;
        }
        filled = std::min(filled + n, cap);
    }

    // Copy the last `n` samples into `out` (linearized, oldest first).
    void get_last(int n, std::vector<float> &out) const {
        int cap = (int)data.size();
        n = std::min(n, filled);
        out.resize(n);
        int start = (write_pos - n + cap) % cap;
        for (int i = 0; i < n; i++) {
            out[i] = data[(start + i) % cap];
        }
    }

    void clear() {
        filled = 0;
        write_pos = 0;
        std::fill(data.begin(), data.end(), 0.0f);
    }
};

static bool send_detection(int fd, const std::string &name, float score) {
    // payload: null-terminated name + float32 score
    std::vector<uint8_t> payload(name.size() + 1 + sizeof(float));
    memcpy(payload.data(), name.c_str(), name.size() + 1);
    memcpy(payload.data() + name.size() + 1, &score, sizeof(float));
    return send_msg(fd, MSG_DETECT, payload.data(), payload.size());
}

static bool send_status(int fd, uint8_t status) {
    return send_msg(fd, MSG_STATUS, &status, 1);
}


static void handle_client(int client_fd,
                          whisper_context *ctx,
                          SileroVad &vad,
                          const std::vector<LoadedKeyword> &keywords)
{
    RingBuffer ring;
    ring.init(JARVIS_BUFFER_SAMPLES);

    DetectScratch scratch;
    scratch.init();
    std::vector<float> pcm_window;
    pcm_window.reserve(JARVIS_BUFFER_SAMPLES);

    int refractory = 0;
    int refractory_total = 0;
    int chunks_received = 0;
    int chunks_per_buffer = (int)(JARVIS_BUFFER_SEC * 1000) / JARVIS_SLIDE_MS;
    float default_thr = keywords.empty() ? 0.35f : keywords[0].threshold;

    // Set receive timeout so we can check g_running periodically
    struct timeval tv = { 0, 500000 };  // 500ms
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    send_status(client_fd, STATUS_BUFFERING);

    MsgHeader hdr;
    std::vector<uint8_t> payload;

    while (g_running) {
        if (!recv_msg(client_fd, hdr, payload)) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;  // timeout, check g_running
            break;  // client disconnected
        }

        if (hdr.type != MSG_AUDIO) continue;

        // Interpret payload as float32 PCM
        int n_samples = hdr.length / sizeof(float);
        const float *samples = (const float *)payload.data();
        ring.push(samples, n_samples);
        chunks_received++;

        // Wait until ring buffer is full before detecting
        if (chunks_received < chunks_per_buffer) {
            send_status(client_fd, STATUS_BUFFERING);
            continue;
        }

        if (chunks_received == chunks_per_buffer) {
            std::cout << "  Detecting..." << std::endl;
            send_status(client_fd, STATUS_READY);
        }

        if (refractory > 0) {
            refractory--;
            continue;
        }

        // Run Silero VAD on incoming audio chunk
        bool has_speech = false;
        for (int i = 0; i + SileroVad::CHUNK_SAMPLES <= n_samples; i += SileroVad::CHUNK_SAMPLES) {
            if (vad.process(samples + i) > 0.5f) has_speech = true;
        }
        if (!has_speech) continue;

        // Get the full 2s window
        ring.get_last(JARVIS_BUFFER_SAMPLES, pcm_window);

        DetectResult det = detect_once(ctx, keywords,
                                       pcm_window.data(), pcm_window.size(), scratch);

        if (det.keyword_index >= 0) {
            const auto &kw = keywords[det.keyword_index];

            auto now = std::chrono::system_clock::now();
            auto tt = std::chrono::system_clock::to_time_t(now);
            char time_buf[32];
            std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", std::localtime(&tt));
            std::cout << "  [" << time_buf << "] " << kw.name
                      << "  sim=" << det.score << std::endl;

            send_detection(client_fd, kw.name, det.score);

            // Generate response: fetch weather text, TTS, send both in one message
            std::string text = get_weather_text();
            if (text.empty()) {
                std::cerr << "  Weather fetch failed" << std::endl;
            } else {
                std::cout << "  " << text << std::endl;
                auto wav = speak_to_wav(text);
                if (wav.empty()) {
                    std::cerr << "  TTS failed" << std::endl;
                } else {
                    // Pack: uint32 text_len + text + wav_data
                    uint32_t text_len = text.size();
                    std::vector<uint8_t> payload(sizeof(text_len) + text_len + wav.size());
                    memcpy(payload.data(), &text_len, sizeof(text_len));
                    memcpy(payload.data() + sizeof(text_len), text.data(), text_len);
                    memcpy(payload.data() + sizeof(text_len) + text_len, wav.data(), wav.size());
                    std::cout << "  Sending " << payload.size() << " bytes (text + audio)" << std::endl;
                    send_msg(client_fd, MSG_RESPONSE, payload.data(), payload.size());
                }
            }

            ring.clear();
            vad.reset();
            refractory_total = kw.refractory_ms / JARVIS_SLIDE_MS;
            refractory = refractory_total;
        }
    }
}

void jarvis_server(const std::string &model_path,
                   const std::string &vad_model_path,
                   std::vector<LoadedKeyword> keywords,
                   int port)
{
    std::signal(SIGINT, signal_handler);
    g_running = true;

    // Load Silero VAD
    SileroVad vad;
    if (!vad.load(vad_model_path)) {
        std::cerr << "Failed to load VAD model: " << vad_model_path << std::endl;
        return;
    }
    std::cout << "VAD loaded: " << vad_model_path << std::endl;

    // Load whisper model
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    whisper_context *ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (!ctx) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return;
    }
    whisper_set_encoder_only(ctx, true);

    std::cout << "Keywords:" << std::endl;
    for (const auto &kw : keywords) {
        int total = 0;
        for (const auto &t : kw.templates.items) total += t.n_frames;
        std::cout << "  " << kw.name << ": "
                  << kw.templates.items.size() << " template(s), " << total << " frames" << std::endl;
    }

    int listen_fd = tcp_listen(port);
    if (listen_fd < 0) {
        std::cerr << "Failed to bind port " << port << std::endl;
        whisper_free(ctx);
        return;
    }

    std::cout << "Listening on port " << port << " (Ctrl+C to stop)" << std::endl;

    while (g_running) {
        // Poll with timeout so we can check g_running periodically
        struct pollfd pfd = { listen_fd, POLLIN, 0 };
        int ret = poll(&pfd, 1, 500);
        if (ret <= 0) continue;  // timeout or EINTR

        struct sockaddr_storage addr;
        socklen_t addr_len = sizeof(addr);
        int client_fd = accept(listen_fd, (struct sockaddr *)&addr, &addr_len);
        if (client_fd < 0) continue;

        int opt = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

        std::cout << "Client connected" << std::endl;
        vad.reset();
        handle_client(client_fd, ctx, vad, keywords);
        close(client_fd);
        std::cout << "Client disconnected" << std::endl;
    }

    close(listen_fd);
    whisper_free(ctx);
    std::cout << "\nServer stopped." << std::endl;
}
