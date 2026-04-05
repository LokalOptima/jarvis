/**
 * main.cpp - Configure keywords and callbacks here.
 *
 * Modes:
 *   ./build/jarvis                     Local detection (mic + model)
 *   ./build/jarvis --server            Server: receive audio, run detection
 *   ./build/jarvis --client HOST       Client: stream mic to server
 */

#include "jarvis.h"
#include "detect.h"
#include "net.h"
#include "server.h"
#include "client.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

enum Mode { LOCAL, SERVER, CLIENT };

struct Args {
    std::string model = "models/ggml-tiny.bin";
    std::string vad_model = "models/silero_vad.bin";
    bool detect_only = false;
    Mode mode = LOCAL;
    std::string server_host;
    int port = JARVIS_PORT;
};

static Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--detect-only") == 0) {
            args.detect_only = true;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args.model = argv[++i];
        } else if (strcmp(argv[i], "--vad") == 0 && i + 1 < argc) {
            args.vad_model = argv[++i];
        } else if (strcmp(argv[i], "--server") == 0) {
            args.mode = SERVER;
        } else if (strcmp(argv[i], "--client") == 0 && i + 1 < argc) {
            args.mode = CLIENT;
            args.server_host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            args.port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "Usage: %s [options]\n"
                            "  --model PATH       whisper model (default: models/ggml-tiny.bin)\n"
                            "  --vad PATH         VAD model (default: models/silero_vad.bin)\n"
                            "  --detect-only      log detections without running callbacks\n"
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

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);

    if (args.mode == SERVER) {
        // Server: load model + templates, runs callbacks (weather + TTS) directly
        std::vector<LoadedKeyword> keywords;
        {
            LoadedKeyword lk;
            lk.name = "hey_jarvis";
            lk.template_path = "models/templates/hey_jarvis.bin";
            lk.threshold = 0.35f;
            lk.refractory_ms = 2000;
            if (!lk.templates.load(lk.template_path)) {
                fprintf(stderr, "Failed to load templates: %s\n", lk.template_path.c_str());
                return 1;
            }
            keywords.push_back(std::move(lk));
        }

        jarvis_server(args.model, args.vad_model, std::move(keywords), args.port);

    } else if (args.mode == CLIENT) {
        // Client: just keywords + callbacks, no model needed
        std::vector<Keyword> keywords;
        keywords.push_back({
            .name = "hey_jarvis",
            .callback = args.detect_only ? nullptr : run_command("./build/weather"),
        });

        jarvis_client(args.server_host, args.port, keywords);

    } else {
        // Local mode (unchanged)
        Jarvis j(args.model, args.vad_model);

        j.add_keyword({
            .name = "hey_jarvis",
            .template_path = "models/templates/hey_jarvis.bin",
            .callback = args.detect_only ? nullptr : run_command("./build/weather"),
        });

        j.listen();
    }
}
