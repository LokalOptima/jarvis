/**
 * jarvis.h - Wake word detection API (local mode).
 *
 * Register keywords with pipelines, then call listen() to start detecting.
 * Each pipeline step is a string→string function chained in sequence.
 *
 *     Jarvis j("models/ggml-tiny.bin");
 *     j.add_keyword({
 *         .name = "hey_jarvis",
 *         .template_path = "models/templates/hey_jarvis.bin",
 *         .pipeline = {transcribe("paraketto"), print_step(), tmux_type()},
 *         .record_follow_up = true,
 *     });
 *     j.listen();
 */

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

// A pipeline step: string in, string out. Empty return stops the pipeline.
using PipeStep = std::function<std::string(const std::string &)>;

struct Keyword {
    std::string name;
    std::string template_path;
    std::vector<PipeStep> pipeline;
    // If true, after detection record audio until silence, pass WAV path to pipeline.
    // If false, pass keyword name to pipeline.
    bool record_follow_up = false;
    float threshold = 0.35f;
    int refractory_ms = 2000;
};

class Jarvis {
public:
    explicit Jarvis(const std::string &whisper_model,
                    const std::string &vad_model = "models/silero_vad.bin");
    ~Jarvis();

    void add_keyword(Keyword kw);
    void listen();  // blocking — runs until stop() or SIGINT
    void stop();

    Jarvis(const Jarvis &) = delete;
    Jarvis &operator=(const Jarvis &) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    std::vector<std::vector<PipeStep>> pipelines;
    std::vector<bool> record_follow_ups;
};

// Run a pipeline: feeds input through each step in sequence.
void run_pipeline(const std::vector<PipeStep> &steps, const std::string &input);

// ---- Built-in pipeline steps ----

// Run cmd + " " + input via popen, return last line of stdout (stderr suppressed).
PipeStep transcribe(const std::string &cmd);
// Print input with "  > " prefix, pass through.
PipeStep print_step();
// Type input into the active tmux pane, pass through.
PipeStep tmux_type();
// Fork+exec a shell command (async, non-blocking). Stops pipeline.
PipeStep fire(const std::string &cmd);
// Run a shell command synchronously, wait for completion. Passes input through.
PipeStep run(const std::string &cmd);
// Run cmd via popen with input as stdin, return stdout.
PipeStep shell_pipe(const std::string &cmd);
