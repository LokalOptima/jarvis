/**
 * server.h - Jarvis server: receive audio over TCP, run detection, send events.
 */

#pragma once

#include "jarvis.h"

#include <string>
#include <vector>

void jarvis_server(const std::string &model_path,
                   const std::string &vad_model_path,
                   const std::vector<Keyword> &keywords,
                   int port);
