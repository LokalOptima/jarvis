cmake_minimum_required(VERSION 3.14)
project(soundtools LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- ggml (vendored tensor library, CPU-only) ----
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA  OFF CACHE BOOL "" FORCE)
add_subdirectory(lib/ggml)

find_package(SDL2 REQUIRED)

# ---- Jarvis: wake word detector ----
# Includes our patched whisper.cpp (encoder output + variable-length context)
add_executable(jarvis
    src/main.cpp
    src/jarvis.cpp
    src/detect.cpp
    src/server.cpp
    src/client.cpp
    src/whisper.cpp
    src/audio_async.cpp
)
target_include_directories(jarvis PRIVATE
    ${SDL2_INCLUDE_DIRS}
    src
    lib
    lib/ggml/include
)
target_link_libraries(jarvis ggml ${SDL2_LIBRARIES})

# ---- Jarvis client: lightweight mic streamer (SDL2 only, no whisper/ggml) ----
add_executable(jarvis-client
    src/jarvis-client.cpp
    src/audio_async.cpp
)
target_include_directories(jarvis-client PRIVATE
    ${SDL2_INCLUDE_DIRS}
    src
)
target_link_libraries(jarvis-client ${SDL2_LIBRARIES} pthread)

# ---- Encode: extract encoder embeddings from WAV files ----
add_executable(encode
    src/encode.cpp
    src/whisper.cpp
)
target_include_directories(encode PRIVATE
    src
    lib
    lib/ggml/include
)
target_link_libraries(encode ggml)

# ---- Bench: model benchmark ----
add_executable(bench
    src/bench.cpp
    src/whisper.cpp
)
target_include_directories(bench PRIVATE
    src
    lib
    lib/ggml/include
)
target_link_libraries(bench ggml)

# ---- Weather: fetch + speak utility ----
add_executable(weather
    src/weather.cpp
)
target_include_directories(weather PRIVATE src lib)
