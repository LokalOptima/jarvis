#pragma once

#include <mutex>
#include <string>

class Display {
public:
    // Update bar state and redraw all lines.
    void bar(const char *label, float score, float threshold, int ms, bool silent);

    // Update event line and redraw all lines.
    void status(const char *keyword, float score, const char *time_str,
                float audio_sec = 0, const char *result = nullptr);

    // Print a persistent line above the display (scrolls up, display re-created below).
    void log(const char *fmt, ...) __attribute__((format(printf, 2, 3)));

    // Update a header field above the display area (cursor patch, no full redraw).
    void header_field(int lines_above, const char *label, const char *value);

    // Print initial display area. Call once after printing the header.
    void init();

    // Wipe the display area.
    void clear();

private:
    void render();  // redraws all 3 lines from current state; caller holds mu_

    std::mutex mu_;

    std::string label_ = "silence";
    float score_ = 0;
    float threshold_ = 0.35f;
    int ms_ = 0;
    bool silent_ = true;

    std::string event_ = "  \033[2m\xe2\x80\x94\033[0m";  // dim dash

    static constexpr int LINES  = 3;   // bar + event + separator
    static constexpr int BAR_W  = 36;
    static constexpr int NAME_W = 14;
};
