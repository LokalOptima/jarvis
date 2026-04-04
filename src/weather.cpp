#include "weather.hpp"

#include <json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

static constexpr const char *WTTR_URL   = "https://wttr.in/Tubingen?format=j1";
static constexpr const char *ROKOKO_BIN = "/home/lapo/git/LokalOptima/rokoko/rokoko";

// ---------- HTTP via curl subprocess ----------

static std::string fetch_url(const char *url) {
    std::string cmd = std::string("curl -sf '") + url + "'";
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) return {};

    std::string result;
    std::array<char, 4096> buf;
    while (size_t n = fread(buf.data(), 1, buf.size(), pipe))
        result.append(buf.data(), n);

    int status = pclose(pipe);
    if (status != 0) return {};
    return result;
}

// ---------- Formatting ----------

static int current_hour() {
    auto now  = std::chrono::system_clock::now();
    auto tt   = std::chrono::system_clock::to_time_t(now);
    auto *loc = std::localtime(&tt);
    return loc->tm_hour;
}

std::string get_weather_text() {
    std::string raw = fetch_url(WTTR_URL);
    if (raw.empty()) return {};

    json data;
    try {
        data = json::parse(raw);
    } catch (...) {
        return {};
    }

    std::ostringstream out;

    // ---- Current conditions ----
    auto &cur  = data["current_condition"][0];
    int temp   = std::stoi(cur["temp_C"].get<std::string>());
    int feels  = std::stoi(cur["FeelsLikeC"].get<std::string>());
    int humid  = std::stoi(cur["humidity"].get<std::string>());
    std::string desc = cur["weatherDesc"][0]["value"].get<std::string>();
    // trim + lowercase
    while (!desc.empty() && desc.back() == ' ') desc.pop_back();
    for (auto &c : desc) c = std::tolower(static_cast<unsigned char>(c));

    out << "Right now it's " << desc << ", " << temp << " degrees, feels like " << feels;
    if (humid > 70)
        out << ", with " << humid << " percent humidity";
    out << ". ";

    // ---- Rest of today (summary) ----
    int hour_now = current_hour();
    auto &today_hourly = data["weather"][0]["hourly"];
    {
        int lo = 999, hi = -999;
        std::unordered_map<std::string, int> desc_counts;
        struct RainSlot { int hour; int pct; };
        std::vector<RainSlot> rainy;
        bool any = false;

        for (auto &h : today_hourly) {
            int hour = std::stoi(h["time"].get<std::string>()) / 100;
            if (hour <= hour_now) continue;
            any = true;
            int t    = std::stoi(h["tempC"].get<std::string>());
            int rain = std::stoi(h["chanceofrain"].get<std::string>());
            std::string d = h["weatherDesc"][0]["value"].get<std::string>();
            while (!d.empty() && d.back() == ' ') d.pop_back();
            for (auto &c : d) c = std::tolower(static_cast<unsigned char>(c));

            if (t < lo) lo = t;
            if (t > hi) hi = t;
            desc_counts[d]++;
            if (rain >= 40) rainy.push_back({hour, rain});
        }

        if (any) {
            std::string dominant;
            int best_count = 0;
            for (auto &[d, cnt] : desc_counts) {
                if (cnt > best_count) { dominant = d; best_count = cnt; }
            }
            out << "Later, " << dominant << ", " << lo << " to " << hi << " degrees. ";
            if (!rainy.empty()) {
                out << "Rain expected around ";
                for (size_t i = 0; i < rainy.size(); i++) {
                    if (i > 0) out << ", ";
                    out << rainy[i].hour << " o'clock";
                }
                out << ". ";
            }
        }
    }


    return out.str();
}

// ---------- TTS ----------

bool speak(const std::string &text) {
    if (text.empty()) return false;
    // Escape single quotes for shell
    std::string escaped;
    for (char c : text) {
        if (c == '\'') escaped += "'\\''";
        else escaped += c;
    }
    std::string cmd = std::string(ROKOKO_BIN) + " '" + escaped + "' --say";
    return std::system(cmd.c_str()) == 0;
}

// ---------- CLI ----------

int main(int argc, char **argv) {
    std::string text = get_weather_text();
    if (text.empty()) {
        fprintf(stderr, "Failed to fetch weather\n");
        return 1;
    }

    bool dry_run = (argc > 1 && std::string(argv[1]) == "--dry-run");
    if (dry_run) {
        printf("%s\n", text.c_str());
    } else {
        printf("Speaking: %s\n", text.c_str());
        if (!speak(text)) return 1;
    }
    return 0;
}
