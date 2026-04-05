#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Fetch weather for Tübingen from wttr.in and format as a speakable sentence.
// Returns empty string on failure.
std::string get_weather_text();

// Speak text via rokoko --say. Returns true on success.
bool speak(const std::string &text);

// Run text through rokoko TTS, capture WAV output. Returns empty on failure.
std::vector<uint8_t> speak_to_wav(const std::string &text);
