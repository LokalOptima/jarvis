/**
 * main.cpp - Jarvis wake word detector.
 *
 * Config-driven detection with optional socket server.
 * Without --listen, runs standalone. With --listen, accepts clients.
 */

#include "config.h"
#include "detect.h"
#include "server.h"

#include <SDL.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

struct Args {
    std::string config_path;
    std::string listen_addr;
    bool list_devices = false;
    int  device_id    = -1;
};

static Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            args.config_path = argv[++i];
        } else if (strcmp(argv[i], "--listen") == 0 && i + 1 < argc) {
            args.listen_addr = argv[++i];
        } else if (strcmp(argv[i], "--list-devices") == 0) {
            args.list_devices = true;
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            args.device_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr,
                "Usage: %s [options]\n\n"
                "  --config PATH      config file (default: ~/.config/jarvis/config.toml)\n"
                "  --listen ADDR      /path/to/sock, tcp:PORT, tcp:HOST:PORT\n"
                "  --list-devices     list SDL2 capture devices and exit\n"
                "  --device N         capture device index (-1 = default)\n"
                "  -h, --help         show this help\n",
                argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            exit(1);
        }
    }
    return args;
}

static void list_sdl_devices() {
    SDL_Init(SDL_INIT_AUDIO);
    int n = SDL_GetNumAudioDevices(1);
    fprintf(stderr, "SDL2 capture devices:\n");
    for (int i = 0; i < n; i++)
        fprintf(stderr, "  %d: %s\n", i, SDL_GetAudioDeviceName(i, 1));
    if (n == 0)
        fprintf(stderr, "  (none found)\n");
    SDL_Quit();
}

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);

    if (args.list_devices) {
        list_sdl_devices();
        return 0;
    }

    Config cfg = load_config(args.config_path);

    if (cfg.keywords.empty()) {
        fprintf(stderr, "No keywords found.\n"
                        "Run 'make enroll' to record clips, then 'make templates' to build templates.\n");
        return 1;
    }

    // CLI --listen overrides config
    std::string addr = args.listen_addr.empty() ? cfg.listen : args.listen_addr;
    jarvis_serve(cfg, addr, args.device_id);
}
