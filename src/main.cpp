/**
 * main.cpp - Configure keywords and pipelines here.
 *
 * Modes:
 *   ./build/jarvis-{cpu,coreml}                     Local detection (mic + model)
 *   ./build/jarvis-{cpu,coreml} --server            Server: receive audio, run detection
 *   ./build/jarvis-{cpu,coreml} --client HOST       Client: stream mic to server
 */

#include "jarvis.h"
#include "ops.h"
#include "net.h"
#include "server.h"
#include "client.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

// Derive a short tag from the model filename.
// "ggml-tiny-FP16.bin" -> "tiny-FP16"
// "ggml-tiny-Q8.bin"   -> "tiny-Q8"
static std::string model_tag(const std::string &model_path) {
    std::string stem = model_path;
    auto slash = stem.rfind('/');
    if (slash != std::string::npos) stem = stem.substr(slash + 1);
    auto dot = stem.rfind('.');
    if (dot != std::string::npos) stem = stem.substr(0, dot);
    if (stem.rfind("ggml-", 0) == 0) stem = stem.substr(5);
    return stem;
}

static std::string template_path(const std::string &keyword, const std::string &tag) {
    return "models/templates/" + keyword + "." + tag + ".bin";
}

enum Mode { MODE_LOCAL, MODE_SERVER, MODE_CLIENT };

struct Args {
    std::string model = JARVIS_DEFAULT_MODEL;
    std::string vad_model = "models/silero_vad.bin";
    Mode mode = MODE_LOCAL;
    std::string server_host;
    std::string ding = "data/beep.wav";
    bool detect_only = false;
    int port = JARVIS_PORT;
};

static Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args.model = argv[++i];
        } else if (strcmp(argv[i], "--vad") == 0 && i + 1 < argc) {
            args.vad_model = argv[++i];
        } else if (strcmp(argv[i], "--server") == 0) {
            args.mode = MODE_SERVER;
        } else if (strcmp(argv[i], "--client") == 0 && i + 1 < argc) {
            args.mode = MODE_CLIENT;
            args.server_host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            args.port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ding") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if (strcmp(v, "none") == 0) args.ding = "";
            else args.ding = std::string("data/") + v + ".wav";
        } else if (strcmp(argv[i], "--detect-only") == 0) {
            args.detect_only = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "Usage: %s [options]\n"
                            "  --model PATH       whisper model (default: %s)\n"
                            "  --vad PATH         VAD model (default: models/silero_vad.bin)\n"
                            "  --server           server mode: listen for audio over TCP\n"
                            "  --client HOST      client mode: stream mic to HOST\n"
                            "  --port PORT        TCP port (default: %d)\n"
                            "  --detect-only      detection only, no pipeline actions\n"
                            "  --ding NAME        detection sound: beep, bling, none (default: beep)\n"
                            "  -h, --help         show this help\n",
                            argv[0], JARVIS_DEFAULT_MODEL, JARVIS_PORT);
            exit(0);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            exit(1);
        }
    }
    return args;
}

static const char *PARAKETTO =
    "flock --shared /tmp/gpu.lock "
    "/home/lapo/git/LokalOptima/paraketto/paraketto.fp8";

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);
    std::string tag = model_tag(args.model);

    if (args.mode == MODE_SERVER) {
        std::vector<Keyword> keywords;
        keywords.push_back({.name = "hey_jarvis",
                            .template_path = template_path("hey_jarvis", tag)});
        keywords.push_back({.name = "weather",
                            .template_path = template_path("weather", tag)});

        jarvis_server(args.model, args.vad_model, keywords, args.port);

    } else if (args.mode == MODE_CLIENT) {
        std::vector<ClientKeyword> keywords;
        keywords.push_back({
            .name = "hey_jarvis",
            .pipeline = {transcribe(PARAKETTO), print(""), tmux("")},
        });
        keywords.push_back({
            .name = "weather",
            .pipeline = {weather(""), tts("")},
        });

        jarvis_client(args.server_host, args.port, keywords);

    } else {
        Jarvis j(args.model, args.vad_model);
        if (!args.ding.empty()) j.set_ding(args.ding);

        std::cout << "------------------------------------\nKeywords:" << std::endl;
        if (args.detect_only) {
            j.on("hey_jarvis", template_path("hey_jarvis", tag), {});
            j.on("weather", template_path("weather", tag), {});
        } else {
            j.on("hey_jarvis", template_path("hey_jarvis", tag), {
                transcribe(PARAKETTO),
                print(""),
                tmux(""),
            });
            j.on("weather", template_path("weather", tag), {
                weather(""),
                tts(""),
            });
        }

        j.listen();
    }
}
