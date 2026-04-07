/**
 * main.cpp - Jarvis wake word detector CLI.
 *
 * Two modes:
 *   (default)  Local detection — hardcoded keywords, prints to terminal.
 *   --serve    Unix socket server — config-driven, broadcasts to clients.
 */

#include "jarvis.h"
#include "detect.h"
#include "config.h"
#include "server.h"

#include <SDL.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

struct Args {
    std::string model       = cache_dir() + "/" + JARVIS_DEFAULT_MODEL;
    std::string vad_model   = cache_dir() + "/silero_vad.bin";
    std::string ding        = cache_dir() + "/beep.wav";
    std::string config_path;
    std::string listen_addr;
    bool serve        = false;
    bool list_devices = false;
    int  device_id    = -1;
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
            else args.ding = cache_dir() + "/" + v + ".wav";
        } else if (strcmp(argv[i], "--serve") == 0) {
            args.serve = true;
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
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
                "Local mode (default):\n"
                "  --model PATH       whisper model (default: %s)\n"
                "  --vad PATH         VAD model (default: silero_vad.bin)\n"
                "  --ding NAME        detection sound: beep, bling, none (default: beep)\n\n"
                "Server mode:\n"
                "  --serve            start detection server\n"
                "  --config PATH      config file (default: ~/.config/jarvis/config.toml)\n"
                "  --listen ADDR      /path/to/sock, tcp:PORT, tcp:HOST:PORT\n\n"
                "Audio:\n"
                "  --list-devices     list SDL2 capture devices and exit\n"
                "  --device N         capture device index (-1 = default)\n\n"
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

static void list_sdl_devices() {
    SDL_Init(SDL_INIT_AUDIO);
    int n = SDL_GetNumAudioDevices(1);  // 1 = capture devices
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

    if (args.serve) {
        Config cfg = load_config(args.config_path);
        std::string addr = args.listen_addr.empty() ? cfg.listen : args.listen_addr;
        if (addr.empty()) addr = "/tmp/jarvis.sock";
        jarvis_serve(cfg, addr, args.device_id);
    } else {
        std::string tag = model_tag(args.model);

        Jarvis j(args.model, args.vad_model);
        if (!args.ding.empty()) j.set_ding(args.ding);

        j.add_keyword({"hey_jarvis", template_path("hey_jarvis", tag)});
        j.add_keyword({"weather", template_path("weather", tag)});

        j.listen();
    }
}
