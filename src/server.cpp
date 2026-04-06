/**
 * server.cpp - Jarvis server: receive PCM audio over TCP, run detection via
 * the shared Jarvis engine, send events back to the client.
 *
 * Receives MSG_PIPELINE from client on connect, resolves step names via OPS
 * dict, then runs the detection loop. Results are sent back via on_result hook.
 *
 * SIGCHLD must NOT be SIG_IGN (popen/pclose needs default behavior).
 */

#include "server.h"
#include "detect.h"
#include "net.h"
#include <audio_async.hpp>

#include <atomic>
#include <csignal>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <thread>

// ---- Wire helpers ----

static bool send_detection(int fd, const std::string &name, float score) {
    std::vector<uint8_t> payload(name.size() + 1 + sizeof(float));
    memcpy(payload.data(), name.c_str(), name.size() + 1);
    memcpy(payload.data() + name.size() + 1, &score, sizeof(float));
    return send_msg(fd, MSG_DETECT, payload.data(), payload.size());
}

static bool send_result(int fd, const Msg &msg) {
    // MSG_RESULT: null-terminated keyword + uint32 text_len + text + audio
    size_t kw_len = msg.keyword.size() + 1;
    uint32_t text_len = msg.text.size();
    std::vector<uint8_t> payload(kw_len + sizeof(text_len) + text_len + msg.audio.size());
    size_t pos = 0;
    memcpy(payload.data() + pos, msg.keyword.c_str(), kw_len); pos += kw_len;
    memcpy(payload.data() + pos, &text_len, sizeof(text_len)); pos += sizeof(text_len);
    if (text_len > 0) { memcpy(payload.data() + pos, msg.text.data(), text_len); pos += text_len; }
    if (!msg.audio.empty()) { memcpy(payload.data() + pos, msg.audio.data(), msg.audio.size()); }
    return send_msg(fd, MSG_RESULT, payload.data(), payload.size());
}

static bool send_status(int fd, uint8_t status) {
    return send_msg(fd, MSG_STATUS, &status, 1);
}

// ---- Pipeline resolution from MSG_PIPELINE ----

static void apply_pipeline_config(const std::vector<uint8_t> &payload, Jarvis &j) {
    size_t pos = 0;
    if (pos >= payload.size()) return;
    uint8_t n_keywords = payload[pos++];

    for (uint8_t k = 0; k < n_keywords && pos < payload.size(); k++) {
        std::string name = read_cstr(payload, pos);

        uint8_t n_steps = 0;
        if (pos < payload.size()) n_steps = payload[pos++];

        Pipeline pipe;
        for (uint8_t s = 0; s < n_steps && pos < payload.size(); s++) {
            std::string step_name = read_cstr(payload, pos);
            std::string step_params = read_cstr(payload, pos);

            auto it = OPS.find(step_name);
            if (it != OPS.end()) {
                pipe.push_back(it->second(step_params));
            } else {
                std::cerr << "  Unknown op: " << step_name << std::endl;
            }
        }

        std::cout << "  " << name << ": " << pipe.size() << " step(s)" << std::endl;
        j.set_pipeline(name, std::move(pipe));
    }
}

// ---- Default pipelines for legacy clients ----

static void apply_default_config(Jarvis &j, int client_fd) {
    // Legacy: no MSG_PIPELINE received. Set up reasonable defaults.
    // Weather: just run the weather op (result sent via on_result hook)
    j.set_pipeline("weather", {weather("")});
    std::cout << "  Using default pipelines (legacy client)" << std::endl;
    (void)client_fd;
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
    // No SIGCHLD SIG_IGN — server pipelines use popen/pclose which need
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
        struct pollfd pfd = { listen_fd, POLLIN, 0 };
        int ret = poll(&pfd, 1, 500);
        if (ret <= 0) continue;

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

        // Read first message: MSG_PIPELINE or MSG_AUDIO (legacy)
        MsgHeader first_hdr;
        std::vector<uint8_t> first_payload;
        bool got_first = recv_msg(client_fd, first_hdr, first_payload);

        if (got_first && first_hdr.type == MSG_PIPELINE) {
            apply_pipeline_config(first_payload, j);
            std::cout << "  Pipeline configured by client" << std::endl;
        } else {
            apply_default_config(j, client_fd);
            if (got_first && first_hdr.type == MSG_AUDIO) {
                audio->push((const float *)first_payload.data(),
                            first_hdr.length / sizeof(float));
            }
        }

        // Set hooks for this client connection
        j.on_detect = [client_fd](const std::string &name, float score) {
            send_detection(client_fd, name, score);
        };
        j.on_result = [client_fd](Msg &msg) {
            if (!msg.text.empty() || !msg.audio.empty()) {
                send_result(client_fd, msg);
            }
        };
        j.on_ready = [client_fd]() {
            send_status(client_fd, STATUS_READY);
        };

        // TCP receiver thread: read audio, push into ring buffer
        std::thread receiver([&j, client_fd, audio]() {
            MsgHeader hdr;
            std::vector<uint8_t> payload;
            struct timeval tv = {0, 500000};
            setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            while (g_running) {
                if (!recv_msg(client_fd, hdr, payload)) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
                    j.stop();
                    break;
                }
                if (hdr.type == MSG_AUDIO) {
                    audio->push((const float *)payload.data(),
                                hdr.length / sizeof(float));
                }
            }
        });

        j.listen(audio);

        shutdown(client_fd, SHUT_RDWR);
        if (receiver.joinable()) receiver.join();
        close(client_fd);
        std::cout << "Client disconnected" << std::endl;
    }

    close(listen_fd);
    g_jarvis = nullptr;
    std::cout << "\nServer stopped." << std::endl;
}
