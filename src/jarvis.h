/**
 * jarvis.h - Wake word detection engine.
 *
 * Detection only. Pipelines are defined via ops.h and registered per-keyword.
 *
 *     Jarvis j("models/ggml-tiny.bin", "models/silero_vad.bin");
 *     j.on("hey_jarvis", "models/templates/hey_jarvis.bin", {
 *         transcribe(PARAKETTO), print(""), tmux(""),
 *     });
 *     j.listen();
 */

#pragma once

#include "ops.h"

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
    void set_pipeline(const std::string &name, Pipeline pipe);

    // Convenience: add_keyword + set_pipeline in one call
    void on(const std::string &name, const std::string &template_path,
            Pipeline pipe, float threshold = 0.35f);

    void set_ding(const std::string &wav_path);

    void listen();                                      // CLI: creates SDL2 mic, installs signals
    void listen(std::shared_ptr<audio_async> audio);    // server/external: blocks until stop()
    void stop();

    // Hooks (set by mode before calling listen)
    std::function<void(const std::string &name, float score)> on_detect;
    std::function<void(Msg &msg)>                             on_result;
    std::function<void()>                                     on_ready;

    Jarvis(const Jarvis &) = delete;
    Jarvis &operator=(const Jarvis &) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    std::vector<Pipeline> pipelines;
    std::atomic<bool> m_running{false};
};
