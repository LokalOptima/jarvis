#include "weather.hpp"

#include <cstdio>
#include <string>

int main(int argc, char **argv) {
    std::string text = get_weather_text();
    if (text.empty()) {
        fprintf(stderr, "Failed to fetch weather\n");
        return 1;
    }

    bool dry_run = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--dry-run") dry_run = true;
    }

    if (dry_run) {
        printf("%s\n", text.c_str());
    } else {
        printf("Speaking: %s\n", text.c_str());
        if (!speak(text)) return 1;
    }
    return 0;
}
