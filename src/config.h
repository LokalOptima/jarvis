/**
 * config.h - Jarvis server configuration (parsed from TOML).
 */

#pragma once

#include <string>
#include <vector>

enum class KeywordMode { KEYWORD, VOICE };

struct KeywordConfig {
    std::string name;
    KeywordMode mode = KeywordMode::KEYWORD;
};

struct Config {
    std::string whisper;      // model filename, resolved to cache_dir
    std::string vad;          // VAD model filename
    std::string ding;         // "beep", "bling", "none"
    std::string listen;       // "/tmp/jarvis.sock", "tcp:9090", or "" (no server)
    float threshold = 0.25f;
    std::vector<KeywordConfig> keywords;
};

// Default config path: ~/.config/jarvis/config.toml
std::string default_config_path();

// Load config from path, or ~/.config/jarvis/config.toml if empty.
// Throws std::runtime_error on parse failure.
Config load_config(const std::string &path = "");
