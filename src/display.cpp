#include "display.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>

void Display::init() {
    std::lock_guard<std::mutex> lk(mu_);
    for (int i = 0; i < LINES; i++)
        fprintf(stderr, "\n");
    render();
}

void Display::bar(const char *label, float score, float threshold, int ms, bool silent) {
    std::lock_guard<std::mutex> lk(mu_);
    label_ = label;
    score_ = score;
    threshold_ = threshold;
    ms_ = ms;
    silent_ = silent;
    render();
}

void Display::status(const char *keyword, float score, const char *time_str,
                     float audio_sec, const char *result) {
    std::lock_guard<std::mutex> lk(mu_);
    char buf[512];
    if (result)
        std::snprintf(buf, sizeof(buf),
                      "  \033[2m[%s]\033[0m %s  %.2f \033[2m\xe2\x86\x92\033[0m %s",
                      time_str, keyword, score, result);
    else if (audio_sec > 0)
        std::snprintf(buf, sizeof(buf),
                      "  \033[2m[%s]\033[0m %s  %.2f \033[2m(%.1fs audio)\033[0m",
                      time_str, keyword, score, audio_sec);
    else
        std::snprintf(buf, sizeof(buf),
                      "  \033[2m[%s]\033[0m %s  %.2f",
                      time_str, keyword, score);
    event_ = buf;
    render();
}

void Display::log(const char *fmt, ...) {
    std::lock_guard<std::mutex> lk(mu_);
    // Erase the display area
    fprintf(stderr, "\033[%dA\r", LINES);
    for (int i = 0; i < LINES; i++)
        fprintf(stderr, "\033[2K\n");
    fprintf(stderr, "\033[%dA\r", LINES);
    // Print the message (becomes permanent scrollback)
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    // Re-create display area below
    for (int i = 0; i < LINES; i++)
        fprintf(stderr, "\n");
    render();
}

void Display::header_field(int lines_above, const char *label, const char *value) {
    std::lock_guard<std::mutex> lk(mu_);
    int up = LINES + lines_above;
    fprintf(stderr, "\033[%dA\r", up);
    fprintf(stderr, "%10s: %s\033[K", label, value);
    fprintf(stderr, "\033[%dB\r", up);
    fflush(stderr);
}

void Display::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    fprintf(stderr, "\033[%dA\r", LINES);
    for (int i = 0; i < LINES; i++)
        fprintf(stderr, "\033[2K\n");
    fprintf(stderr, "\033[%dA\r", LINES);
    fflush(stderr);
}

void Display::render() {
    char buf[1024];
    char *p = buf;

    // Move cursor up to bar line
    p += std::snprintf(p, 32, "\033[%dA\r", LINES);

    // --- Line 1: bar ---
    *p++ = ' '; *p++ = ' ';

    int nlen = (int)label_.size();
    int copy = std::min(nlen, NAME_W);
    memcpy(p, label_.data(), copy); p += copy;
    for (int i = copy; i < NAME_W; i++) *p++ = ' ';

    float range = threshold_ * 2.0f;
    float norm_score = score_ / range;
    float norm_thr = threshold_ / range;
    int thr = std::max(0, std::min(BAR_W - 1, (int)(norm_thr * BAR_W)));
    int filled = std::max(0, std::min(BAR_W, (int)(norm_score * BAR_W)));

    if (silent_) {
        memcpy(p, "\033[2m", 4); p += 4;
        for (int i = 0; i < BAR_W; i++) {
            if      (i < filled) { memcpy(p, "\xe2\x96\x88", 3); p += 3; }
            else if (i == thr)   { memcpy(p, "\xe2\x94\x82", 3); p += 3; }
            else                 { memcpy(p, "\xc2\xb7", 2); p += 2; }
        }
        memcpy(p, "\033[0m\033[K\n", 8); p += 8;
    } else {
        static const char *esc[]    = { "\033[32m", "\033[1;32m", "\033[2m", "\033[33m" };
        static const int   esc_len[] = { 5, 7, 4, 5 };
        int color = -1;

        for (int i = 0; i < BAR_W; i++) {
            int want;
            if      (i == thr)              want = 3;  // threshold = yellow
            else if (i < filled && i < thr) want = 0;  // below threshold = green
            else if (i < filled)            want = 1;  // above threshold = bright green
            else                            want = 2;  // unfilled = dim

            if (want != color) {
                memcpy(p, esc[want], esc_len[want]);
                p += esc_len[want];
                color = want;
            }

            if (i == thr)        { memcpy(p, "\xe2\x94\x82", 3); p += 3; }
            else if (i < filled) { memcpy(p, "\xe2\x96\x88", 3); p += 3; }
            else                 { memcpy(p, "\xc2\xb7", 2); p += 2; }
        }

        p += std::snprintf(p, 64, "\033[0m  %4.2f  %3dms\033[K\n", score_, ms_);
    }

    // --- Line 2: event ---
    int elen = (int)event_.size();
    memcpy(p, event_.data(), elen); p += elen;
    memcpy(p, "\033[K\n", 4); p += 4;

    // --- Line 3: separator ---
    static const char sep[] =
        "\033[2m────────────────────────────────────────────────────\033[0m\033[K\n";
    memcpy(p, sep, sizeof(sep) - 1); p += sizeof(sep) - 1;

    fwrite(buf, 1, p - buf, stderr);
    fflush(stderr);
}
