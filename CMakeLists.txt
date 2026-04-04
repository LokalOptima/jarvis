cmake_minimum_required(VERSION 3.13)
project(soundtools LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- whisper.cpp (local copy, patched for encoder output + variable-length) ----
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
add_subdirectory(lib/whisper.cpp)

find_package(SDL2 REQUIRED)

# ---- Jarvis: Whisper-based personalized wake word detector ----
add_executable(jarvis
    src/jarvis.cpp
    src/audio_async.cpp
)
target_include_directories(jarvis PRIVATE
    ${SDL2_INCLUDE_DIRS}
    src
    lib
    lib/whisper.cpp/include
)
target_link_libraries(jarvis whisper ${SDL2_LIBRARIES})

# ---- Weather: fetch + speak utility ----
add_executable(weather
    src/weather.cpp
)
target_include_directories(weather PRIVATE
    src
    lib
)

