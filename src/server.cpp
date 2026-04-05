/**
 * server.cpp - Jarvis server: receive PCM audio over TCP, run detection via
 * the shared Jarvis engine, send events back to the client.
 *
 * Uses Jarvis with push-mode audio_async: a TCP receiver thread feeds audio
 * into the ring buffer, and Jarvis::listen(audio) runs the same detection
 * loop as local mode, with server-specific pipeline steps that send results
 * back to the client via MSG_RESPONSE.
 */

#include "server.h"
#include "detect.h"
#include "net.h"
#include "weather.hpp"
#include <audio_async.hpp>

#include <atomic>
#include <csignal>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <thread>

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

// ---- Server-specific pipeline steps ----

// Send transcribed text to client via MSG_RESPONSE (text only, no WAV)
static PipeStep send_text_to_client(int fd) {
    return {"send_to_client", [fd](const std::string &text) -> std::string {
        uint32_t text_len = text.size();
        std::vector<uint8_t> payload(sizeof(text_len) + text_len);
        memcpy(payload.data(), &text_len, sizeof(text_len));
        memcpy(payload.data() + sizeof(text_len), text.data(), text_len);
        send_msg(fd, MSG_RESPONSE, payload.data(), payload.size());
        return text;
    }};
}

// Fetch weather, TTS, send text+WAV to client
static PipeStep weather_response(int fd) {
    return {"weather_response", [fd](const std::string &) -> std::string {
        std::string text = get_weather_text();
        if (text.empty()) return "";
        auto wav = speak_to_wav(text);
        if (wav.empty()) return "";
        uint32_t text_len = text.size();
        std::vector<uint8_t> payload(sizeof(text_len) + text_len + wav.size());
        memcpy(payload.data(), &text_len, sizeof(text_len));
        memcpy(payload.data() + sizeof(text_len), text.data(), text_len);
        memcpy(payload.data() + sizeof(text_len) + text_len, wav.data(), wav.size());
        send_msg(fd, MSG_RESPONSE, payload.data(), payload.size());
        return "";  // terminal step
    }};
}

// ---- Server main loop ----

static std::atomic<bool> g_running{true};
static Jarvis *g_jarvis = nullptr;

static void signal_handler(int) {
    g_running.store(false, std::memory_order_relaxed);
    if (g_jarvis) g_jarvis->stop();
}

void jarvis_server(const std::string &model_path,
                   const std::string &vad_model_path,
                   const std::vector<Keyword> &keywords,
                   int port)
{
    std::signal(SIGINT, signal_handler);
    // No SIGCHLD SIG_IGN here — server pipelines use popen/pclose which need
    // default SIGCHLD behavior (SIG_IGN causes pclose to return -1/ECHILD).
    g_running = true;

    Jarvis j(model_path, vad_model_path);
    g_jarvis = &j;

    std::cout << "Keywords:" << std::endl;
    for (const auto &kw : keywords) {
        j.add_keyword(kw);
    }

    int listen_fd = tcp_listen(port);
    if (listen_fd < 0) {
        std::cerr << "Failed to bind port " << port << std::endl;
        g_jarvis = nullptr;
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

        // Push-mode audio source for this connection
        auto audio = std::make_shared<audio_async>(
            static_cast<int>(JARVIS_BUFFER_SEC * 1000));
        audio->init_push(JARVIS_SAMPLE_RATE);
        audio->resume();

        // Per-connection pipelines (server-specific steps, NOT tmux_type)
        j.set_pipeline("hey_jarvis", {
            transcribe("flock --shared /tmp/gpu.lock "
                       "/home/lapo/git/LokalOptima/paraketto/paraketto.fp8"),
            print_step(),
            send_text_to_client(client_fd),
        });
        j.set_record_follow_up("hey_jarvis", true);
        j.set_pipeline("weather", {weather_response(client_fd)});

        j.on_detect = [client_fd](const std::string &name, float score) {
            send_detection(client_fd, name, score);
        };
        j.on_ready = [client_fd]() {
            send_status(client_fd, STATUS_READY);
        };

        // TCP receiver thread: read audio from client, push into ring buffer
        std::thread receiver([&j, client_fd, audio]() {
            MsgHeader hdr;
            std::vector<uint8_t> payload;
            struct timeval tv = {0, 500000};
            setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            while (g_running) {
                if (!recv_msg(client_fd, hdr, payload)) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
                    j.stop();  // client disconnected → unblock listen()
                    break;
                }
                if (hdr.type == MSG_AUDIO) {
                    audio->push((const float *)payload.data(),
                                hdr.length / sizeof(float));
                }
            }
        });

        j.listen(audio);  // blocks until j.stop() or SIGINT

        shutdown(client_fd, SHUT_RDWR);  // unblock receiver if still in recv()
        if (receiver.joinable()) receiver.join();
        close(client_fd);
        std::cout << "Client disconnected" << std::endl;
    }

    close(listen_fd);
    g_jarvis = nullptr;
    std::cout << "\nServer stopped." << std::endl;
}
