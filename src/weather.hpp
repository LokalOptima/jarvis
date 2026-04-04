#pragma once

#include <string>

// Fetch weather for Tübingen from wttr.in and format as a speakable sentence.
// Returns empty string on failure.
std::string get_weather_text();

// Speak text via rokoko --say. Returns true on success.
bool speak(const std::string &text);
