/**
 * jarvis.cpp - Wake word detection engine.
 *
 * Detection loop: SDL2/push audio -> VAD gate -> detect_once() -> callback.
 */

#include "jarvis.h"
#include "detect.h"
#include "playback.h"
#include "vad_ggml.h"
#include <audio_async.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>

// ---- Jarvis implementation ----

struct Jarvis::Impl {
    struct whisper_context *ctx = nullptr;
    SileroVad vad;
    std::vector<LoadedKeyword> keywords;
    std::vector<uint8_t> ding_wav;
};

Jarvis::Jarvis(const std::string &whisper_model,
               const std::string &vad_model) : impl(std::make_unique<Impl>()) {
    if (!impl->vad.load(vad_model)) {
        throw std::runtime_error("Failed to load VAD model: " + diagnose_path(vad_model));
    }

    whisper_log_set([](ggml_log_level, const char *, void *) {}, nullptr);

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    impl->ctx = whisper_init_from_file_with_params(whisper_model.c_str(), cparams);
    if (!impl->ctx) {
        throw std::runtime_error("Failed to load whisper model: " + diagnose_path(whisper_model));
    }
    whisper_set_encoder_only(impl->ctx, true);
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
        throw std::runtime_error("Failed to load templates: " + diagnose_path(lk.template_path));
    }
    impl->keywords.push_back(std::move(lk));
}

void Jarvis::set_ding(const std::string &wav_path) {
    std::ifstream f(wav_path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "Warning: could not load ding: " << wav_path << std::endl;
        return;
    }
    auto sz = f.tellg();
    f.seekg(0);
    impl->ding_wav.resize(sz);
    f.read(reinterpret_cast<char *>(impl->ding_wav.data()), sz);
}

void Jarvis::stop() {
    m_running = false;
}

// ---- Header ----

void Jarvis::print_header() {
    // Collect keyword names
    std::string kw_list;
    for (size_t i = 0; i < impl->keywords.size(); i++) {
        if (i > 0) kw_list += ", ";
        kw_list += impl->keywords[i].name;
    }

    fprintf(stderr,
        "\033[2m───\033[0m jarvis \033[2m"
        "─────────────────────────────────────────\033[0m\n");
    fprintf(stderr, "    engine: %s \033[2m[%s]\033[0m\n", JARVIS_ENGINE, cache_dir().c_str());
    fprintf(stderr, "  keywords: %s\n", kw_list.c_str());
    if (on_header) on_header();
    fprintf(stderr, "\n");
    fflush(stderr);
}

// ---- Main detection loop ----

void Jarvis::listen() {
    static Jarvis *s_instance = nullptr;
    s_instance = this;
    std::signal(SIGINT, [](int) { if (s_instance) s_instance->stop(); });

    auto audio = std::make_shared<audio_async>(JARVIS_BUFFER_MS);
    audio->init(-1, JARVIS_SAMPLE_RATE);
    audio->resume();

    listen(audio);

    render_clear();
    fprintf(stderr, "Stopped.\n");
}

void Jarvis::listen(std::shared_ptr<audio_async> audio) {
    if (impl->keywords.empty()) {
        std::cerr << "No keywords registered" << std::endl;
        return;
    }

    m_running = true;
    impl->vad.reset();

    print_header();

    // Print initial display: bar + status + separator
    fprintf(stderr, "  warming up\033[K\n");
    fprintf(stderr, "  \033[2m—\033[0m\033[K\n");
    render_separator();

    if (on_ready) on_ready();

    DetectScratch scratch;
    scratch.init();
    std::vector<float> pcm_buffer;
    pcm_buffer.reserve(JARVIS_BUFFER_SAMPLES);

    float default_thr = impl->keywords[0].threshold;
    bool warmed_up = false;
    int skip_frames = 0;  // encoder frames to skip (after detection, excludes old keyword)

    while (m_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(JARVIS_SLIDE_MS));

        // Wait for ring buffer to accumulate a full detection window
        if (!warmed_up) {
            if (audio->available() < JARVIS_BUFFER_SAMPLES) {
                render_bar("warming up", 0.0f, default_thr, 0, true);
                continue;
            }
            warmed_up = true;
        }

        // Decay skip as new audio slides in (1 encoder frame ≈ 20ms ≈ SLIDE_MS/10)
        if (skip_frames > 0) {
            int decay = JARVIS_SLIDE_MS * JARVIS_SAMPLE_RATE / (1000 * JARVIS_MEL_HOP * JARVIS_CONV_STRIDE);
            skip_frames = std::max(0, skip_frames - decay);
        }

        // VAD gate: check latest 200ms
        audio->get(JARVIS_SLIDE_MS, pcm_buffer);
        if (pcm_buffer.empty()) continue;

        bool has_speech = false;
        for (size_t i = 0; i + SileroVad::CHUNK_SAMPLES <= pcm_buffer.size();
             i += SileroVad::CHUNK_SAMPLES) {
            if (impl->vad.process(pcm_buffer.data() + i) > 0.5f) { has_speech = true; break; }
        }
        if (!has_speech) {
            render_bar("silence", 0.0f, default_thr, 0, true);
            continue;
        }

        // Speech detected: always fetch full 2s, skip old keyword in encoder output
        audio->get(JARVIS_BUFFER_MS, pcm_buffer);

        DetectResult det = detect_once(impl->ctx, impl->keywords,
                                       pcm_buffer.data(), pcm_buffer.size(), scratch,
                                       skip_frames);

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

            render_status(kw.name.c_str(), det.score, time_buf);

            if (!impl->ding_wav.empty())
                play_wav(impl->ding_wav.data(), impl->ding_wav.size(), false);

            if (on_detect) on_detect(kw.name, det.score, audio);

            // Skip past the detected keyword on subsequent cycles
            skip_frames = det.end_frame;
            impl->vad.reset();
        }
    }
}
