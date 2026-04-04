/**
 * main.cpp - Configure keywords and callbacks here.
 *
 * Edit this file to add/remove wake words and change what happens on detection.
 */

#include "jarvis.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

struct Args {
    std::string model = "models/ggml-tiny.bin";
    bool detect_only = false;
};

static Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--detect-only") == 0) {
            args.detect_only = true;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args.model = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "Usage: %s [options]\n"
                            "  --model PATH     whisper model (default: models/ggml-tiny.bin)\n"
                            "  --detect-only    log detections without running callbacks\n"
                            "  -h, --help       show this help\n", argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            exit(1);
        }
    }
    return args;
}

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);

    Jarvis j(args.model);

    j.add_keyword({
        .name = "hey_jarvis",
        .template_path = "models/templates/hey_jarvis.bin",
        .callback = args.detect_only ? nullptr : run_command("./build/weather"),
    });

    // Add more keywords:
    // j.add_keyword({
    //     .name = "hey_computer",
    //     .template_path = "models/templates/hey_computer.bin",
    //     .callback = [](const std::string &kw, float score) {
    //         std::cout << "Detected " << kw << " with score " << score << std::endl;
    //     },
    //     .threshold = 0.40f,
    // });

    j.listen();
}
