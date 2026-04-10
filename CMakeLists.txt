cmake_minimum_required(VERSION 3.14)
project(jarvis LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- ggml (vendored tensor library, CPU-only) ----
# Disable all non-CPU backends — our vendored ggml only has CPU source.
# Without this, macOS auto-detects BLAS via Accelerate and fails to link.
foreach(_backend BLAS METAL CUDA VULKAN OPENCL SYCL MUSA HIP CANN)
    set(GGML_${_backend} OFF CACHE BOOL "" FORCE)
endforeach()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(lib/ggml)

find_package(SDL2 REQUIRED)

# ---- Engine configuration (compile-time decision) ----
option(WHISPER_COREML "Enable Core ML (ANE) encoder" OFF)
if(WHISPER_COREML)
    enable_language(OBJCXX)
    add_compile_definitions(WHISPER_USE_COREML WHISPER_COREML_ALLOW_FALLBACK)
    add_compile_definitions(JARVIS_DEFAULT_MODEL="ggml-tiny-FP16.bin")
    add_compile_definitions(JARVIS_ENGINE="coreml")
    set(JARVIS_SUFFIX "-coreml")
else()
    add_compile_definitions(JARVIS_DEFAULT_MODEL="ggml-tiny-Q8.bin")
    add_compile_definitions(JARVIS_ENGINE="cpu")
    set(JARVIS_SUFFIX "-cpu")
endif()

# ---- Jarvis: static library (for embedding in other projects) ----
add_library(jarvis_lib STATIC
    src/jarvis.cpp
    src/detect.cpp
    src/display.cpp
    src/playback.cpp
    src/config.cpp
    src/recorder.cpp
    src/server.cpp
    src/vad_ggml.cpp
    src/whisper.cpp
    src/audio_async.cpp
)
target_include_directories(jarvis_lib PUBLIC
    ${SDL2_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/ggml/include
)
target_link_libraries(jarvis_lib PUBLIC ggml ${SDL2_LIBRARIES})

if(WHISPER_COREML)
    target_sources(jarvis_lib PRIVATE src/coreml/whisper-encoder.mm)
    set_source_files_properties(src/coreml/whisper-encoder.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    target_link_libraries(jarvis_lib PUBLIC "-framework CoreML" "-framework Foundation")
endif()

# ---- Jarvis: wake word detector ----
add_executable(jarvis src/main.cpp)
target_link_libraries(jarvis jarvis_lib)
set_target_properties(jarvis PROPERTIES OUTPUT_NAME "jarvis${JARVIS_SUFFIX}")

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
set_target_properties(encode PROPERTIES OUTPUT_NAME "encode${JARVIS_SUFFIX}")

if(WHISPER_COREML)
    target_sources(encode PRIVATE src/coreml/whisper-encoder.mm)
    set_source_files_properties(src/coreml/whisper-encoder.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    target_link_libraries(encode "-framework CoreML" "-framework Foundation")
endif()

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

if(WHISPER_COREML)
    target_sources(bench PRIVATE src/coreml/whisper-encoder.mm)
    set_source_files_properties(src/coreml/whisper-encoder.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    target_link_libraries(bench "-framework CoreML" "-framework Foundation")
endif()

# ---- VAD test: validate Silero VAD C implementation ----
add_executable(vad_test
    src/vad_test.cpp
    src/vad_ggml.cpp
)
target_include_directories(vad_test PRIVATE src lib/ggml/include)
target_link_libraries(vad_test ggml)
