/**
 * client.h - Jarvis client: capture mic, stream to server, run local pipeline steps.
 */

#pragma once

#include "ops.h"

#include <string>
#include <vector>

struct ClientKeyword {
    std::string name;
    Pipeline    pipeline;   // full pipeline (REMOTE + LOCAL steps)
};

void jarvis_client(const std::string &server_host, int port,
                   const std::vector<ClientKeyword> &keywords);
