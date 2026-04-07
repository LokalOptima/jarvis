/**
 * server.cpp - Jarvis detection engine with optional socket server.
 *
 * If listen_addr is empty: standalone detection, no clients.
 * If listen_addr is set: accepts client connections (Unix socket or TCP).
 *
 * Threading model (with listener):
 *   - Main thread: accept loop, reaps dead clients
 *   - Detection thread: Jarvis::listen(audio) — owns mic, VAD + detection
 *   - Per-client reader thread: reads subscribe messages, detects disconnect
 *
 * on_detect callback runs in the detection thread (blocks the loop):
 *   1. Look up keyword mode from config
 *   2. voice mode: play ding, vad_record(), build JSON + PCM
 *   3. keyword mode: build JSON only
 *   4. broadcast to subscribed clients (if any)
 */

#include "server.h"
#include "jarvis.h"
#include "detect.h"
#include "playback.h"
#include "recorder.h"
#include "vad_ggml.h"
#include <audio_async.hpp>
#include <json.hpp>

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;

// ---- Client state ----

struct Client {
    int fd;
    std::set<std::string> subscriptions;
    std::mutex mu;  // protects subscriptions and fd writes
    std::atomic<bool> alive{true};
    std::thread reader;
};

// ---- Server state ----

static std::mutex g_clients_mu;
static std::vector<std::shared_ptr<Client>> g_clients;
static std::atomic<bool> g_running{true};
static int g_listen_fd = -1;

static void add_client(std::shared_ptr<Client> c) {
    std::lock_guard<std::mutex> lk(g_clients_mu);
    g_clients.push_back(std::move(c));
}

// Update the "clients: N" header line only when the count changes.
static std::atomic<int> g_last_client_count{-1};

static void update_client_count() {
    int n;
    {
        std::lock_guard<std::mutex> lk(g_clients_mu);
        n = 0;
        for (auto &c : g_clients) if (c->alive) n++;
    }
    int prev = g_last_client_count.exchange(n);
    if (prev == n) return;
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", n);
    render_header_field(2, "clients", buf);
}

static void reap_clients() {
    std::lock_guard<std::mutex> lk(g_clients_mu);
    for (auto it = g_clients.begin(); it != g_clients.end(); ) {
        if (!(*it)->alive) {
            if ((*it)->reader.joinable()) (*it)->reader.join();
            close((*it)->fd);
            it = g_clients.erase(it);
        } else {
            ++it;
        }
    }
}

// Write all bytes, handling partial writes. Returns false on failure.
static bool write_all(int fd, const char *data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = write(fd, data + sent, len - sent);
        if (n <= 0) return false;
        sent += n;
    }
    return true;
}

static bool send_to(Client &c, const std::string &jsonl,
                    const float *pcm = nullptr, int n_samples = 0) {
    std::lock_guard<std::mutex> lk(c.mu);
    if (!c.alive) return false;

    if (!write_all(c.fd, jsonl.data(), jsonl.size())) { c.alive = false; return false; }

    if (pcm && n_samples > 0) {
        if (!write_all(c.fd, reinterpret_cast<const char *>(pcm), n_samples * sizeof(float))) {
            c.alive = false;
            return false;
        }
    }
    return true;
}

// Snapshot subscribed clients, then send outside the global lock.
static void broadcast(const std::string &keyword, const std::string &jsonl,
                      const float *pcm = nullptr, int n_samples = 0) {
    std::vector<std::shared_ptr<Client>> targets;
    {
        std::lock_guard<std::mutex> lk(g_clients_mu);
        for (auto &c : g_clients) {
            if (c->alive && c->subscriptions.count(keyword))
                targets.push_back(c);
        }
    }
    for (auto &c : targets)
        send_to(*c, jsonl, pcm, n_samples);
}

// ---- Client reader thread ----

static void client_reader(std::shared_ptr<Client> c) {
    char buf[4096];
    std::string line;

    while (c->alive && g_running) {
        ssize_t n = read(c->fd, buf, sizeof(buf));
        if (n <= 0) { c->alive = false; return; }

        line.append(buf, n);

        // Process complete lines
        size_t pos;
        while ((pos = line.find('\n')) != std::string::npos) {
            std::string msg = line.substr(0, pos);
            line.erase(0, pos + 1);

            if (msg.empty()) continue;

            try {
                auto j = json::parse(msg);
                if (j.contains("subscribe") && j["subscribe"].is_array()) {
                    std::lock_guard<std::mutex> lk(c->mu);
                    c->subscriptions.clear();
                    for (auto &k : j["subscribe"])
                        if (k.is_string()) c->subscriptions.insert(k.get<std::string>());
                }
            } catch (...) {
                // Ignore malformed messages
            }
        }
    }
}

// ---- Listener ----

static bool is_tcp(const std::string &addr) { return addr.rfind("tcp:", 0) == 0; }

static int create_listener(const std::string &addr) {
    if (is_tcp(addr)) {
        std::string rest = addr.substr(4);
        std::string host = "0.0.0.0";
        int port;

        auto colon = rest.rfind(':');
        if (colon != std::string::npos) {
            host = rest.substr(0, colon);
            port = std::stoi(rest.substr(colon + 1));
        } else {
            port = std::stoi(rest);
        }

        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) throw std::runtime_error(std::string("socket: ") + strerror(errno));

        int opt = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in sa = {};
        sa.sin_family = AF_INET;
        sa.sin_port = htons(port);
        inet_pton(AF_INET, host.c_str(), &sa.sin_addr);

        if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
            close(fd);
            throw std::runtime_error(std::string("bind: ") + strerror(errno));
        }
        if (listen(fd, 8) < 0) {
            close(fd);
            throw std::runtime_error(std::string("listen: ") + strerror(errno));
        }
        return fd;
    } else {
        std::string path = addr;
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) throw std::runtime_error(std::string("socket: ") + strerror(errno));

        struct sockaddr_un sa = {};
        sa.sun_family = AF_UNIX;
        strncpy(sa.sun_path, path.c_str(), sizeof(sa.sun_path) - 1);
        unlink(path.c_str());

        if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
            close(fd);
            throw std::runtime_error(std::string("bind: ") + strerror(errno));
        }
        if (listen(fd, 8) < 0) {
            close(fd);
            throw std::runtime_error(std::string("listen: ") + strerror(errno));
        }
        return fd;
    }
}

static void cleanup_listener(const std::string &addr) {
    if (!is_tcp(addr)) unlink(addr.c_str());
}

// ---- Server ----

void jarvis_serve(const Config &config,
                  const std::string &listen_addr,
                  int device_id) {
    // Resolve model paths
    std::string cd = cache_dir();
    std::string whisper_path = cd + "/" + config.whisper;
    std::string vad_path     = cd + "/" + config.vad;
    std::string tag          = model_tag(whisper_path);

    // Load recording VAD (separate instance from detection VAD)
    SileroVad record_vad;
    if (!record_vad.load(vad_path))
        throw std::runtime_error("Failed to load recording VAD: " + diagnose_path(vad_path));

    // Set up Jarvis detection engine
    Jarvis j(whisper_path, vad_path);

    // Load ding sound
    std::string ding_path;
    if (config.ding != "none") {
        ding_path = cd + "/" + config.ding + ".wav";
        j.set_ding(ding_path);
    }

    // Register keywords
    for (auto &kw : config.keywords)
        j.add_keyword({kw.name, template_path(kw.name, tag), config.threshold});

    bool has_listener = !listen_addr.empty();

    // Set up listener (Unix socket or TCP) if configured
    if (has_listener)
        g_listen_fd = create_listener(listen_addr);

    // Header extension
    j.on_header = [&]() {
        if (has_listener) {
            fprintf(stderr, "    listen: %s\n", listen_addr.c_str());
            fprintf(stderr, "   clients: 0\n");
        }
    };

    // Signal handling
    g_running = true;
    std::signal(SIGINT, [](int) {
        g_running = false;
        if (g_listen_fd >= 0) { shutdown(g_listen_fd, SHUT_RDWR); close(g_listen_fd); g_listen_fd = -1; }
    });
    std::signal(SIGPIPE, SIG_IGN);

    // on_detect callback: build event, optionally record, broadcast
    j.on_detect = [&](const std::string &name, float score,
                      std::shared_ptr<audio_async> audio) {
        KeywordMode mode = KeywordMode::KEYWORD;
        for (auto &kw : config.keywords)
            if (kw.name == name) { mode = kw.mode; break; }

        json evt;
        evt["keyword"] = name;
        evt["score"] = score;

        if (mode == KeywordMode::VOICE) {
            RecordResult rec = vad_record(record_vad, audio);
            evt["audio_length"] = (int)rec.pcm.size();
            float audio_sec = (float)rec.pcm.size() / JARVIS_SAMPLE_RATE;

            auto now = std::chrono::system_clock::now();
            auto tt = std::chrono::system_clock::to_time_t(now);
            char tbuf[32];
            std::strftime(tbuf, sizeof(tbuf), "%H:%M:%S", std::localtime(&tt));
            render_status(name.c_str(), score, tbuf, audio_sec);

            if (has_listener) {
                std::string line = evt.dump() + "\n";
                broadcast(name, line, rec.pcm.data(), rec.pcm.size());
            }
        } else if (has_listener) {
            std::string line = evt.dump() + "\n";
            broadcast(name, line);
        }
    };

    // Start detection
    auto audio_src = std::make_shared<audio_async>(JARVIS_BUFFER_MS);
    audio_src->init(device_id, JARVIS_SAMPLE_RATE);
    audio_src->resume();

    if (has_listener) {
        // Server mode: detection in background thread, accept loop in main thread
        std::thread detect_thread([&]() {
            j.listen(audio_src);
        });

        struct pollfd pfd = { g_listen_fd, POLLIN, 0 };
        while (g_running) {
            int ret = poll(&pfd, 1, 1000);
            if (ret <= 0 || !(pfd.revents & POLLIN)) {
                reap_clients();
                update_client_count();
                continue;
            }

            int cfd = accept(g_listen_fd, nullptr, nullptr);
            if (cfd < 0) continue;

            auto client = std::make_shared<Client>();
            client->fd = cfd;
            client->reader = std::thread(client_reader, client);
            add_client(client);
            update_client_count();

            reap_clients();
            update_client_count();
        }

        j.stop();
        if (detect_thread.joinable()) detect_thread.join();

        // Close all clients
        {
            std::lock_guard<std::mutex> lk(g_clients_mu);
            for (auto &c : g_clients) {
                c->alive = false;
                shutdown(c->fd, SHUT_RDWR);
            }
        }
        reap_clients();
        cleanup_listener(listen_addr);
        if (g_listen_fd >= 0) close(g_listen_fd);
    } else {
        // Standalone mode: detection in main thread, no socket
        j.listen(audio_src);
    }

    render_clear();
    fprintf(stderr, "Stopped.\n");
}
