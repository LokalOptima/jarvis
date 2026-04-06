/**
 * main.cpp - Jarvis wake word detector CLI.
 *
 * Listens for wake words and prints detections to stdout.
 * Pipelines / actions are handled externally.
 */

#include "jarvis.h"
#include "detect.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/stat.h>

struct Args {
    std::string model = cache_dir() + "/" + JARVIS_DEFAULT_MODEL;
    std::string vad_model = cache_dir() + "/silero_vad.bin";
    std::string ding = "data/beep.wav";
};

static Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args.model = argv[++i];
        } else if (strcmp(argv[i], "--vad") == 0 && i + 1 < argc) {
            args.vad_model = argv[++i];
        } else if (strcmp(argv[i], "--ding") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if (strcmp(v, "none") == 0) args.ding = "";
            else args.ding = std::string("data/") + v + ".wav";
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "Usage: %s [options]\n"
                            "  --model PATH       whisper model (default: %s)\n"
                            "  --vad PATH         VAD model (default: models/silero_vad.bin)\n"
                            "  --ding NAME        detection sound: beep, bling, none (default: beep)\n"
                            "  -h, --help         show this help\n",
                            argv[0], JARVIS_DEFAULT_MODEL);
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

    struct stat st;
    if (stat(cache_dir().c_str(), &st) != 0) {
        fprintf(stderr, "Model directory not found: %s\n"
                        "Create it and place model files there:\n"
                        "  mkdir -p %s/templates\n"
                        "  # copy ggml-tiny.bin, silero_vad.bin, and templates\n",
                        cache_dir().c_str(), cache_dir().c_str());
        return 1;
    }

    std::string tag = model_tag(args.model);

    Jarvis j(args.model, args.vad_model);
    if (!args.ding.empty()) j.set_ding(args.ding);

    j.add_keyword({"hey_jarvis", template_path("hey_jarvis", tag)});
    j.add_keyword({"weather", template_path("weather", tag)});

    j.listen();
}
