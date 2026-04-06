/**
 * jarvis.cpp - Wake word detection engine.
 *
 * Detection loop: SDL2/push audio -> VAD gate -> detect_once() -> pipeline.
 * No recording, transcription, or domain logic — those live in ops.cpp.
 */

#include "jarvis.h"
#include "detect.h"
#include "vad_ggml.h"
#include <audio_async.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>

// ---- Jarvis implementation ----

struct Jarvis::Impl {
    struct whisper_context *ctx = nullptr;
    SileroVad vad;
    std::vector<LoadedKeyword> keywords;
};

Jarvis::Jarvis(const std::string &whisper_model,
               const std::string &vad_model) : impl(std::make_unique<Impl>()) {
    if (!impl->vad.load(vad_model)) {
        throw std::runtime_error("Failed to load VAD model: " + vad_model);
    }
    std::cout << "    VAD loaded: " << vad_model << std::endl;

    whisper_log_set([](ggml_log_level, const char *, void *) {}, nullptr);

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    impl->ctx = whisper_init_from_file_with_params(whisper_model.c_str(), cparams);
    if (!impl->ctx) {
        throw std::runtime_error("Failed to load whisper model: " + whisper_model);
    }
    whisper_set_encoder_only(impl->ctx, true);
    std::cout << "Whisper loaded: " << whisper_model << std::endl;
    std::cout << "------------------------------------" << std::endl;

    // Set engine singletons so ops can access VAD and running flag
    set_engine_singletons(&impl->vad, &m_running);
}

Jarvis::~Jarvis() {
    if (impl && impl->ctx) whisper_free(impl->ctx);
}

void Jarvis::add_keyword(Keyword kw) {
    LoadedKeyword lk;
    lk.name = kw.name;
    lk.template_path = kw.template_path;
    lk.threshold = kw.threshold;
    if (!lk.templates.load(lk.template_path)) {
        throw std::runtime_error("Failed to load templates: " + lk.template_path);
    }
    int total = 0;
    for (const auto &t : lk.templates.items) total += t.n_frames;
    std::cout << "  " << lk.name << ": " << total << " frames" << std::endl;

    pipelines.emplace_back();  // empty pipeline by default
    impl->keywords.push_back(std::move(lk));
}

void Jarvis::set_pipeline(const std::string &name, Pipeline pipe) {
    for (size_t i = 0; i < impl->keywords.size(); i++) {
        if (impl->keywords[i].name == name) {
            pipelines[i] = std::move(pipe);
            return;
        }
    }
}

void Jarvis::on(const std::string &name, const std::string &template_path,
                Pipeline pipe, float threshold) {
    add_keyword({name, template_path, threshold});
    // Pipeline goes to the last added keyword
    pipelines.back() = std::move(pipe);
    // Print pipeline steps
    for (const auto &step : pipelines.back()) {
        std::cout << "    -> " << step.name;
        if (!step.params.empty()) std::cout << "(" << step.params << ")";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Jarvis::stop() {
    m_running = false;
}

// ---- Main detection loop ----

void Jarvis::listen() {
    static Jarvis *s_instance = nullptr;
    s_instance = this;
    std::signal(SIGINT, [](int) { if (s_instance) s_instance->stop(); });
    // No SIGCHLD SIG_IGN — ops use popen/pclose which need default behavior.
    // fire() uses double-fork to avoid zombies.

    auto audio = std::make_shared<audio_async>(static_cast<int>(JARVIS_BUFFER_SEC * 1000));
    audio->init(-1, JARVIS_SAMPLE_RATE);
    audio->resume();

    listen(audio);

    std::cout << "\nStopped." << std::endl;
}

void Jarvis::listen(std::shared_ptr<audio_async> audio) {
    if (impl->keywords.empty()) {
        std::cerr << "No keywords registered" << std::endl;
        return;
    }

    m_running = true;
    impl->vad.reset();

    // Wait for audio buffer to fill
    {
        const int steps = 20;
        const int step_ms = (int)(JARVIS_BUFFER_SEC * 1000) / steps;
        for (int i = 1; i <= steps && m_running; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
            render_bar("buffering", (float)i / steps, 1.0f, 0, true);
        }
        std::cerr << "\r\033[K" << std::flush;
    }

    if (on_ready) on_ready();
    std::cout << "Listening... (" << impl->keywords.size() << " keyword(s), Ctrl+C to stop)" << std::endl;

    DetectScratch scratch;
    scratch.init();
    std::vector<float> pcm_buffer;
    pcm_buffer.reserve(JARVIS_BUFFER_SAMPLES);

    float default_thr = impl->keywords[0].threshold;

    while (m_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(JARVIS_SLIDE_MS));

        // VAD gate: check latest 200ms
        audio->get(JARVIS_SLIDE_MS, pcm_buffer);
        if (pcm_buffer.empty()) continue;

        bool has_speech = false;
        for (size_t i = 0; i + SileroVad::CHUNK_SAMPLES <= pcm_buffer.size();
             i += SileroVad::CHUNK_SAMPLES) {
            if (impl->vad.process(pcm_buffer.data() + i) > 0.5f) has_speech = true;
        }
        if (!has_speech) {
            render_bar("listening", 0.0f, default_thr, 0, true);
            continue;
        }

        // Speech detected: get full 2s buffer and run detection
        audio->get(static_cast<int>(JARVIS_BUFFER_SEC * 1000), pcm_buffer);
        if (pcm_buffer.empty()) continue;

        DetectResult det = detect_once(impl->ctx, impl->keywords,
                                       pcm_buffer.data(), pcm_buffer.size(), scratch);

        int show_kw = det.keyword_index >= 0 ? det.keyword_index : det.best_keyword;
        float show_score = det.keyword_index >= 0 ? det.score : det.best_score;
        render_bar(impl->keywords[show_kw].name.c_str(), show_score,
                   impl->keywords[show_kw].threshold, det.elapsed_ms, false);

        if (det.keyword_index >= 0) {
            const auto &kw = impl->keywords[det.keyword_index];

            auto now = std::chrono::system_clock::now();
            auto tt = std::chrono::system_clock::to_time_t(now);
            char time_buf[32];
            std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", std::localtime(&tt));
            std::cerr << "\r\033[K" << std::flush;
            std::cout << "  [" << time_buf << "] " << kw.name
                      << "  sim=" << det.score << std::endl;

            if (on_detect) on_detect(kw.name, det.score);

            impl->vad.reset();
            audio->clear();

            Msg msg;
            msg.keyword = kw.name;
            msg.source = audio;
            run_pipeline(pipelines[det.keyword_index], msg);

            if (on_result) on_result(msg);

            impl->vad.reset();
            audio->clear();

        }
    }
}
