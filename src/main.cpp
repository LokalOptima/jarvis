/**
 * main.cpp - Configure keywords and pipelines here.
 *
 * Modes:
 *   ./build/jarvis                     Local detection (mic + model)
 *   ./build/jarvis --server            Server: receive audio, run detection
 *   ./build/jarvis --client HOST       Client: stream mic to server
 */

#include "jarvis.h"
#include "ops.h"
#include "net.h"
#include "server.h"
#include "client.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

enum Mode { MODE_LOCAL, MODE_SERVER, MODE_CLIENT };

struct Args {
    std::string model = "models/ggml-tiny.bin";
    std::string vad_model = "models/silero_vad.bin";
    Mode mode = MODE_LOCAL;
    std::string server_host;
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
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "Usage: %s [options]\n"
                            "  --model PATH       whisper model (default: models/ggml-tiny.bin)\n"
                            "  --vad PATH         VAD model (default: models/silero_vad.bin)\n"
                            "  --server           server mode: listen for audio over TCP\n"
                            "  --client HOST      client mode: stream mic to HOST\n"
                            "  --port PORT        TCP port (default: %d)\n"
                            "  -h, --help         show this help\n", argv[0], JARVIS_PORT);
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

    if (args.mode == MODE_SERVER) {
        std::vector<Keyword> keywords;
        keywords.push_back({.name = "hey_jarvis",
                            .template_path = "models/templates/hey_jarvis.bin"});
        keywords.push_back({.name = "weather",
                            .template_path = "models/templates/weather.bin"});

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

        j.on("hey_jarvis", "models/templates/hey_jarvis.bin", {
            transcribe(PARAKETTO),
            print(""),
            tmux(""),
        });
        j.on("weather", "models/templates/weather.bin", {
            weather(""),
            tts(""),
        });

        j.listen();
    }
}
