/**
 * jarvis-client.cpp - Lightweight client: capture mic, stream to server.
 *
 * Dependencies: SDL2 only. No whisper, no ggml.
 *
 * Usage: ./build/jarvis-client <server-host> [--port PORT]
 */

#include "net.h"
#include <audio_async.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

static constexpr int   SAMPLE_RATE    = 16000;
static constexpr float BUFFER_SEC     = 2.0f;
static constexpr int   SLIDE_MS       = 200;

static std::atomic<bool> g_running{true};
static void signal_handler(int) { g_running.store(false, std::memory_order_relaxed); }

static void status_line(const char *msg) {
    fprintf(stderr, "\r  %-40s\033[K", msg);
}

static void play_wav(const uint8_t *data, size_t size) {
    char tmp[] = "/tmp/jarvis-XXXXXX.wav";
    int fd = mkstemps(tmp, 4);
    if (fd < 0) return;
    if (write(fd, data, size) < 0) { close(fd); return; }
    close(fd);

    fprintf(stderr, "  Playing response (%zu bytes)\n", size);
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
    // Don't unlink — SIG_IGN auto-reaps the child, so waitpid races with exec.
    // Temp files are ~350KB and /tmp is cleaned on reboot.
}

static void receiver_thread(int fd) {
    MsgHeader hdr;
    std::vector<uint8_t> payload;

    while (g_running) {
        if (!recv_msg(fd, hdr, payload)) break;

        if (hdr.type == MSG_DETECT) {
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
            fprintf(stderr, "\r\033[K");
            printf("  [%s] %s  sim=%.2f\n", time_buf, name, score);

        } else if (hdr.type == MSG_RESULT) {
            // MSG_RESULT: null-terminated keyword + uint32 text_len + text + wav
            size_t pos = 0;
            std::string keyword = read_cstr(payload, pos);

            uint32_t text_len = 0;
            if (pos + sizeof(text_len) <= payload.size()) {
                memcpy(&text_len, payload.data() + pos, sizeof(text_len));
                pos += sizeof(text_len);
            }
            if (text_len > 0 && pos + text_len <= payload.size()) {
                printf("  %.*s\n", (int)text_len, (const char *)(payload.data() + pos));
                pos += text_len;
            }
            if (pos < payload.size()) {
                play_wav(payload.data() + pos, payload.size() - pos);
            }

        } else if (hdr.type == MSG_STATUS && !payload.empty()) {
            switch (payload[0]) {
                case STATUS_BUFFERING: status_line("server buffering..."); break;
                case STATUS_READY:     status_line("ready"); break;
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        fprintf(stderr, "Usage: %s <server-host> [--port PORT]\n", argv[0]);
        return 1;
    }

    const char *host = argv[1];
    int port = JARVIS_PORT;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        }
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGCHLD, SIG_IGN);
    g_running = true;

    printf("Connecting to %s:%d...\n", host, port);
    int fd = tcp_connect(host, port);
    if (fd < 0) {
        fprintf(stderr, "Failed to connect to %s:%d\n", host, port);
        return 1;
    }
    printf("Connected\n");

    std::thread rx(receiver_thread, fd);

    auto audio = std::make_shared<audio_async>(static_cast<int>(BUFFER_SEC * 1000));
    audio->init(-1, SAMPLE_RATE);
    audio->resume();

    printf("Streaming audio... (Ctrl+C to stop)\n");

    std::vector<float> chunk;

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(SLIDE_MS));

        audio->get(SLIDE_MS, chunk);
        if (chunk.empty()) continue;

        if (!send_msg(fd, MSG_AUDIO, chunk.data(), chunk.size() * sizeof(float))) {
            fprintf(stderr, "\nServer disconnected\n");
            break;
        }
    }

    g_running = false;
    shutdown(fd, SHUT_RDWR);
    if (rx.joinable()) rx.join();
    close(fd);

    printf("\nStopped.\n");
    return 0;
}
