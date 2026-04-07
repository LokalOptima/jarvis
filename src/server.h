/**
 * server.h - Jarvis Unix socket server.
 *
 * Listens for keyword detections and broadcasts events to subscribed clients.
 * Voice-mode keywords include VAD-gated recorded audio after the JSON line.
 */

#pragma once

#include "config.h"
#include <string>

// Run the server. Blocks until SIGINT.
// socket_path: Unix socket path (default /tmp/jarvis.sock).
// device_id: SDL2 capture device (-1 = default).
void jarvis_serve(const Config &config,
                  const std::string &socket_path = "/tmp/jarvis.sock",
                  int device_id = -1);
