/**
 * client.cpp - Jarvis client: capture mic, stream PCM to server, run local steps.
 *
 * On connect:
 *   1. Split each keyword's pipeline into REMOTE prefix + LOCAL suffix
 *   2. Send remote specs to server as single MSG_PIPELINE message
 *   3. Stream mic audio as MSG_AUDIO
 *   4. Receive MSG_DETECT, MSG_RESULT, MSG_STATUS from server
 *   5. On MSG_RESULT: play audio (if any), run local pipeline steps
 *
 * SIGCHLD = SIG_IGN (fire-and-forget for aplay child processes).
 */

#include "client.h"
#include "detect.h"
#include "net.h"
#include <audio_async.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <unistd.h>

static std::atomic<bool> g_running{true};
static void signal_handler(int) { g_running.store(false, std::memory_order_relaxed); }

// ---- Pipeline split ----

struct SplitPipeline {
    // Remote: (name, params) pairs for the server
    std::vector<std::pair<std::string, std::string>> remote_specs;
    // Local: steps to run on MSG_RESULT
    Pipeline local_steps;
};

static SplitPipeline split_pipeline(const Pipeline &pipe) {
    SplitPipeline sp;
    bool seen_local = false;
    for (const auto &step : pipe) {
        if (step.placement == REMOTE) {
            if (seen_local) {
                std::cerr << "Pipeline error: REMOTE step '" << step.name
                          << "' after LOCAL steps (must be contiguous REMOTE prefix)"
                          << std::endl;
                continue;
            }
            sp.remote_specs.push_back({step.name, step.params});
        } else {
            seen_local = true;
            sp.local_steps.push_back(step);
        }
    }
    return sp;
}

// ---- Send MSG_PIPELINE (single message, all keywords) ----

static bool send_pipeline_config(int fd,
    const std::vector<std::pair<std::string, SplitPipeline>> &configs)
{
    std::vector<uint8_t> payload;
    payload.push_back((uint8_t)configs.size());

    for (const auto &[name, sp] : configs) {
        // null-terminated keyword name
        payload.insert(payload.end(), name.begin(), name.end());
        payload.push_back(0);
        // number of remote steps
        payload.push_back((uint8_t)sp.remote_specs.size());
        for (const auto &[step_name, step_params] : sp.remote_specs) {
            payload.insert(payload.end(), step_name.begin(), step_name.end());
            payload.push_back(0);
            payload.insert(payload.end(), step_params.begin(), step_params.end());
            payload.push_back(0);
        }
    }

    return send_msg(fd, MSG_PIPELINE, payload.data(), payload.size());
}

// ---- Receiver thread ----

static void receiver_thread(int fd,
    const std::unordered_map<std::string, Pipeline> &local_pipelines)
{
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
            // Detection display handled by server's render_bar;
            // client just notes it for pipeline dispatch

        } else if (hdr.type == MSG_RESULT) {
            size_t pos = 0;
            std::string keyword = read_cstr(payload, pos);

            uint32_t text_len = 0;
            if (pos + sizeof(text_len) <= payload.size()) {
                memcpy(&text_len, payload.data() + pos, sizeof(text_len));
                pos += sizeof(text_len);
            }
            std::string text;
            if (text_len > 0 && pos + text_len <= payload.size()) {
                text.assign((const char *)(payload.data() + pos), text_len);
                pos += text_len;
            }

            if (pos < payload.size()) {
                play_wav(payload.data() + pos, payload.size() - pos, false);
            }

            // Run local pipeline
            auto it = local_pipelines.find(keyword);
            if (it != local_pipelines.end() && !it->second.empty() && !text.empty()) {
                Msg msg;
                msg.keyword = keyword;
                msg.text = text;
                run_pipeline(it->second, msg);
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

// ---- Connection with retry ----

static int connect_with_retry(const char *host, int port) {
    while (g_running) {
        int fd = tcp_connect(host, port);
        if (fd >= 0) return fd;

        std::cerr << "Connection failed, retrying in 2s..." << std::endl;
        for (int waited = 0; waited < 2000 && g_running; waited += 200)
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return -1;
}

// ---- Client main loop ----

void jarvis_client(const std::string &server_host, int port,
                   const std::vector<ClientKeyword> &keywords)
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGCHLD, SIG_IGN);
    g_running = true;

    // Split pipelines and build local pipeline map
    std::vector<std::pair<std::string, SplitPipeline>> configs;
    std::unordered_map<std::string, Pipeline> local_pipelines;

    for (const auto &kw : keywords) {
        auto sp = split_pipeline(kw.pipeline);
        local_pipelines[kw.name] = sp.local_steps;
        configs.push_back({kw.name, std::move(sp)});
    }

    // Start mic capture (persists across reconnections)
    auto audio = std::make_shared<audio_async>(static_cast<int>(JARVIS_BUFFER_SEC * 1000));
    audio->init(-1, JARVIS_SAMPLE_RATE);
    audio->resume();

    while (g_running) {
        std::cout << "Connecting to " << server_host << ":" << port << "..." << std::endl;
        int fd = connect_with_retry(server_host.c_str(), port);
        if (fd < 0) break;
        std::cout << "Connected to server" << std::endl;

        // Send pipeline config
        if (!send_pipeline_config(fd, configs)) {
            std::cerr << "Failed to send pipeline config" << std::endl;
            close(fd);
            continue;
        }

        // Start receiver thread
        std::thread rx(receiver_thread, fd, std::cref(local_pipelines));

        std::cout << "Streaming audio... (Ctrl+C to stop)" << std::endl;

        std::vector<float> chunk;
        audio->clear();

        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(JARVIS_SLIDE_MS));

            audio->get(JARVIS_SLIDE_MS, chunk);
            if (chunk.empty()) continue;

            if (!send_msg(fd, MSG_AUDIO, chunk.data(), chunk.size() * sizeof(float))) {
                std::cerr << "\nServer disconnected" << std::endl;
                break;
            }
        }

        shutdown(fd, SHUT_RDWR);
        if (rx.joinable()) rx.join();
        close(fd);

        if (g_running) {
            std::cout << "Reconnecting..." << std::endl;
        }
    }

    std::cout << "\nClient stopped." << std::endl;
}
