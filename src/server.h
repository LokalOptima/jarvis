/**
 * server.h - Jarvis detection server.
 *
 * Listens for keyword detections and broadcasts events to subscribed clients.
 * Supports Unix sockets and TCP — determined by listen_addr format:
 *   /tmp/jarvis.sock   → Unix socket
 *   tcp:9090           → TCP on all interfaces
 *   tcp:127.0.0.1:9090 → TCP on localhost only
 */

#pragma once

#include "config.h"
#include <string>

// Run the server. Blocks until SIGINT.
// listen_addr: Unix socket path or tcp:HOST:PORT.
// device_id: SDL2 capture device (-1 = default).
void jarvis_serve(const Config &config,
                  const std::string &listen_addr = "/tmp/jarvis.sock",
                  int device_id = -1);
