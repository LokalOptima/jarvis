/**
 * server.h - Jarvis server: receive audio over TCP, run detection, send events.
 */

#pragma once

#include "detect.h"

#include <string>
#include <vector>

void jarvis_server(const std::string &model_path,
                   std::vector<LoadedKeyword> keywords,
                   int port);
