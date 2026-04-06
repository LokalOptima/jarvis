/**
 * jarvis.h - Wake word detection engine.
 *
 * Detection only. Register keywords with callbacks:
 *
 *     Jarvis j("models/ggml-tiny.bin", "models/silero_vad.bin");
 *     j.add_keyword({"hey_jarvis", "models/templates/hey_jarvis.bin"});
 *     j.on_detect = [](const std::string &kw, float score,
 *                      std::shared_ptr<audio_async> audio) {
 *         std::cout << kw << " " << score << std::endl;
 *     };
 *     j.listen();
 */

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class audio_async;

struct Keyword {
    std::string name;
    std::string template_path;
    float       threshold = 0.35f;
};

class Jarvis {
public:
    explicit Jarvis(const std::string &whisper_model,
                    const std::string &vad_model = "models/silero_vad.bin");
    ~Jarvis();

    void add_keyword(Keyword kw);
    void set_ding(const std::string &wav_path);

    void listen();                                      // CLI: creates SDL2 mic, installs signals
    void listen(std::shared_ptr<audio_async> audio);    // server/external: blocks until stop()
    void stop();

    // Callback on keyword detection.
    // Receives keyword name, similarity score, and audio source (for recording).
    std::function<void(const std::string &name, float score,
                       std::shared_ptr<audio_async> audio)> on_detect;
    std::function<void()> on_ready;

    Jarvis(const Jarvis &) = delete;
    Jarvis &operator=(const Jarvis &) = delete;

private:
    void print_header();
    struct Impl;
    std::unique_ptr<Impl> impl;
    std::atomic<bool> m_running{false};
};
