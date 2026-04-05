/**
 * client.cpp - Jarvis client: capture mic, stream PCM to server, run callbacks.
 *
 * Two threads:
 *   1. Main: capture 200ms audio chunks via SDL2, send to server
 *   2. Receiver: read detection events from server, dispatch callbacks
 */

#include "client.h"
#include "detect.h"
#include "net.h"
#include <audio_async.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

static std::atomic<bool> g_running{true};
static void signal_handler(int) { g_running.store(false, std::memory_order_relaxed); }

static void receiver_thread(int fd, const std::vector<Keyword> &keywords) {
    MsgHeader hdr;
    std::vector<uint8_t> payload;

    while (g_running) {
        if (!recv_msg(fd, hdr, payload)) break;

        if (hdr.type == MSG_DETECT) {
            // Parse: null-terminated name + float32 score
            const char *name = (const char *)payload.data();
            float score = 0;
            size_t name_len = strlen(name);
            if (name_len + 1 + sizeof(float) <= payload.size()) {
                memcpy(&score, payload.data() + name_len + 1, sizeof(float));
            }

            auto now = std::chrono::system_clock::now();
            auto tt = std::chrono::system_clock::to_time_t(now);
            char time_buf[32];
            std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", std::localtime(&tt));
            std::cerr << "\r\033[K" << std::flush;
            std::cout << "  [" << time_buf << "] " << name
                      << "  sim=" << score << std::endl;

            for (const auto &kw : keywords) {
                if (kw.name == name && !kw.pipeline.empty()) {
                    run_pipeline(kw.pipeline, kw.name);
                    break;
                }
            }
        } else if (hdr.type == MSG_RESPONSE && payload.size() > sizeof(uint32_t)) {
            uint32_t text_len;
            memcpy(&text_len, payload.data(), sizeof(text_len));
            if (text_len > 0 && sizeof(uint32_t) + text_len <= payload.size()) {
                std::cout << "  " << std::string((const char *)(payload.data() + sizeof(uint32_t)), text_len) << std::endl;
            }
            size_t wav_offset = sizeof(uint32_t) + text_len;
            if (wav_offset < payload.size()) {
                size_t wav_size = payload.size() - wav_offset;
                char tmp[] = "/tmp/jarvis-XXXXXX.wav";
                int tfd = mkstemps(tmp, 4);
                if (tfd >= 0) {
                    write(tfd, payload.data() + wav_offset, wav_size);
                    ::close(tfd);
                    pid_t pid = fork();
                    if (pid == 0) {
#ifdef __APPLE__
                        execlp("afplay", "afplay", tmp, nullptr);
#else
                        execlp("aplay", "aplay", "-q", tmp, nullptr);
                        execlp("paplay", "paplay", tmp, nullptr);
#endif
                        _exit(127);
                    }
                }
            }
        } else if (hdr.type == MSG_STATUS) {
            if (!payload.empty()) {
                uint8_t status = payload[0];
                if (status == STATUS_BUFFERING) {
                    render_bar("server buffering", 0.5f, 1.0f, 0, true);
                }
            }
        }
    }
}

void jarvis_client(const std::string &server_host, int port,
                   const std::vector<Keyword> &keywords)
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGCHLD, SIG_IGN);
    g_running = true;

    std::cout << "Connecting to " << server_host << ":" << port << "..." << std::endl;
    int fd = tcp_connect(server_host.c_str(), port);
    if (fd < 0) {
        std::cerr << "Failed to connect to " << server_host << ":" << port << std::endl;
        return;
    }
    std::cout << "Connected to server" << std::endl;

    // Start receiver thread
    std::thread rx(receiver_thread, fd, std::cref(keywords));

    // Start mic capture
    auto audio = std::make_shared<audio_async>(static_cast<int>(JARVIS_BUFFER_SEC * 1000));
    audio->init(-1, JARVIS_SAMPLE_RATE);
    audio->resume();

    std::cout << "Streaming audio... (Ctrl+C to stop)" << std::endl;

    std::vector<float> chunk;

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(JARVIS_SLIDE_MS));

        // Get last 200ms of audio
        audio->get(JARVIS_SLIDE_MS, chunk);
        if (chunk.empty()) continue;

        if (!send_msg(fd, MSG_AUDIO, chunk.data(), chunk.size() * sizeof(float))) {
            std::cerr << "\nServer disconnected" << std::endl;
            break;
        }
    }

    g_running = false;
    // Shutdown socket to unblock receiver thread
    shutdown(fd, SHUT_RDWR);
    if (rx.joinable()) rx.join();
    close(fd);

    std::cout << "\nClient stopped." << std::endl;
}
