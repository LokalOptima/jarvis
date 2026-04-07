/**
 * server.h - Jarvis detection server.
 *
 * Runs detection with optional socket server for client subscriptions.
 * If listen_addr is empty, runs standalone (detect + ding, no clients).
 * If listen_addr is set, accepts clients and broadcasts detection events.
 *
 * Address formats:
 *   /tmp/jarvis.sock   → Unix socket
 *   tcp:9090           → TCP on all interfaces
 *   tcp:127.0.0.1:9090 → TCP on localhost only
 */

#pragma once

#include "config.h"
#include <string>

// Run detection. Blocks until SIGINT.
// listen_addr: socket address, or "" for standalone mode.
// device_id: SDL2 capture device (-1 = default).
void jarvis_serve(const Config &config,
                  const std::string &listen_addr = "",
                  int device_id = -1);
