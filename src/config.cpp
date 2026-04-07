/**
 * config.cpp - Parse ~/.config/jarvis/config.toml
 *
 * If no config file exists, returns defaults with keywords
 * auto-discovered from template files in ~/.cache/jarvis/templates/.
 */

#include "config.h"
#include "detect.h"

#include "toml.hpp"

#include <cstdlib>
#include <dirent.h>
#include <stdexcept>

std::string default_config_path() {
    const char *xdg = getenv("XDG_CONFIG_HOME");
    if (xdg && xdg[0]) return std::string(xdg) + "/jarvis/config.toml";
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    return std::string(home) + "/.config/jarvis/config.toml";
}

// Discover keywords from template files: hey_jarvis.tiny-Q8.bin → "hey_jarvis"
static std::vector<KeywordConfig> discover_keywords(const std::string &tag) {
    std::vector<KeywordConfig> keywords;
    std::string tpl_dir = cache_dir() + "/templates";
    std::string suffix = "." + tag + ".bin";

    DIR *dir = opendir(tpl_dir.c_str());
    if (!dir) return keywords;

    struct dirent *ent;
    while ((ent = readdir(dir)) != nullptr) {
        std::string name(ent->d_name);
        if (name.size() > suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            KeywordConfig kw;
            kw.name = name.substr(0, name.size() - suffix.size());
            keywords.push_back(std::move(kw));
        }
    }
    closedir(dir);
    return keywords;
}

Config load_config(const std::string &path) {
    std::string file = path.empty() ? default_config_path() : path;

    Config cfg;
    cfg.whisper   = JARVIS_DEFAULT_MODEL;
    cfg.vad       = "silero_vad.bin";
    cfg.ding      = "beep";
    cfg.threshold = 0.35f;

    // Try parsing config file — use defaults if it doesn't exist
    FILE *f = fopen(file.c_str(), "r");
    if (f) {
        fclose(f);
        toml::table tbl;
        try {
            tbl = toml::parse_file(file);
        } catch (const toml::parse_error &e) {
            throw std::runtime_error("Failed to parse config " + file + ": " + std::string(e.description()));
        }

        cfg.whisper   = tbl["whisper"].value_or(cfg.whisper);
        cfg.vad       = tbl["vad"].value_or(cfg.vad);
        cfg.ding      = tbl["ding"].value_or(cfg.ding);
        cfg.listen    = tbl["listen"].value_or(std::string(""));
        cfg.threshold = tbl["threshold"].value_or(cfg.threshold);

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
    }

    // Auto-discover keywords from templates if none configured
    if (cfg.keywords.empty()) {
        std::string tag = model_tag(cache_dir() + "/" + cfg.whisper);
        cfg.keywords = discover_keywords(tag);
    }

    return cfg;
}
