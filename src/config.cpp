/**
 * config.cpp - Parse ~/.config/jarvis/config.toml
 */

#include "config.h"
#include "detect.h"

#include "toml.hpp"

#include <cstdlib>
#include <stdexcept>

static std::string default_config_path() {
    const char *xdg = getenv("XDG_CONFIG_HOME");
    if (xdg && xdg[0]) return std::string(xdg) + "/jarvis/config.toml";
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    return std::string(home) + "/.config/jarvis/config.toml";
}

Config load_config(const std::string &path) {
    std::string file = path.empty() ? default_config_path() : path;

    toml::table tbl;
    try {
        tbl = toml::parse_file(file);
    } catch (const toml::parse_error &e) {
        throw std::runtime_error("Failed to parse config " + file + ": " + std::string(e.description()));
    }

    Config cfg;
    cfg.whisper   = tbl["whisper"].value_or(std::string(JARVIS_DEFAULT_MODEL));
    cfg.vad       = tbl["vad"].value_or(std::string("silero_vad.bin"));
    cfg.ding      = tbl["ding"].value_or(std::string("beep"));
    cfg.threshold = tbl["threshold"].value_or(0.35f);

    if (auto arr = tbl["keywords"].as_array()) {
        for (auto &elem : *arr) {
            auto *t = elem.as_table();
            if (!t) continue;

            KeywordConfig kw;
            auto name = (*t)["name"].value<std::string>();
            if (!name) throw std::runtime_error("Keyword entry missing 'name' in " + file);
            kw.name = *name;

            auto mode_str = (*t)["mode"].value_or(std::string("keyword"));
            if (mode_str == "voice")        kw.mode = KeywordMode::VOICE;
            else if (mode_str == "keyword") kw.mode = KeywordMode::KEYWORD;
            else throw std::runtime_error("Unknown keyword mode '" + mode_str + "' for " + kw.name);

            cfg.keywords.push_back(std::move(kw));
        }
    }

    if (cfg.keywords.empty())
        throw std::runtime_error("No keywords defined in " + file);

    return cfg;
}
