/**
 * main.cpp - Configure keywords and callbacks here.
 *
 * Edit this file to add/remove wake words and change what happens on detection.
 */

#include "jarvis.h"

#include <iostream>

int main() {
    Jarvis j("models/ggml-tiny.bin");

    j.add_keyword({
        .name = "hey_jarvis",
        .template_path = "models/templates/hey_jarvis.bin",
        .callback = run_command("./build/weather"),
    });

    // Add more keywords:
    // j.add_keyword({
    //     .name = "hey_computer",
    //     .template_path = "models/templates/hey_computer.bin",
    //     .callback = [](const std::string &kw, float score) {
    //         std::cout << "Detected " << kw << " with score " << score << std::endl;
    //     },
    //     .threshold = 0.40f,
    // });

    j.listen();
}
