/**
 * client.h - Jarvis client: capture mic, stream to server, execute callbacks.
 */

#pragma once

#include "jarvis.h"

#include <string>
#include <vector>

void jarvis_client(const std::string &server_host, int port,
                   const std::vector<Keyword> &keywords);
