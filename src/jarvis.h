/**
 * jarvis.h - Wake word detection API.
 *
 * Register keywords with callbacks, then call listen() to start detecting.
 *
 *     Jarvis j("models/ggml-tiny.bin");
 *     j.add_keyword({
 *         .name = "hey_jarvis",
 *         .template_path = "models/templates/hey_jarvis.bin",
 *         .callback = run_command("./build/weather"),
 *     });
 *     j.listen();
 */

#pragma once

#include <functional>
#include <memory>
#include <string>

struct Keyword {
    std::string name;
    std::string template_path;
    std::function<void(const std::string &keyword, float score)> callback;
    float threshold = 0.35f;
    int refractory_ms = 2000;
};

class Jarvis {
public:
    explicit Jarvis(const std::string &whisper_model);
    ~Jarvis();

    void add_keyword(Keyword kw);
    void listen();  // blocking — runs until stop() or SIGINT
    void stop();

    Jarvis(const Jarvis &) = delete;
    Jarvis &operator=(const Jarvis &) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Convenience: callback that fork+execs a shell command.
std::function<void(const std::string &, float)> run_command(const std::string &cmd);
