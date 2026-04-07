# Auto-detect macOS for CoreML support
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
    COREML_FLAG := -DWHISPER_COREML=ON
    JARVIS_BIN := build/jarvis-coreml
    ENCODE_BIN := build/encode-coreml
    MODEL      := ggml-tiny-FP16.bin
else
    COREML_FLAG := -DWHISPER_COREML=OFF
    JARVIS_BIN := build/jarvis-cpu
    ENCODE_BIN := build/encode-cpu
    MODEL      := ggml-tiny-Q8.bin
endif

CACHE   := $(HOME)/.cache/jarvis
RELEASE := https://github.com/LokalOptima/jarvis/releases/download/models-v1

# Files to download from the release
MODELS := $(CACHE)/$(MODEL) $(CACHE)/silero_vad.bin \
          $(CACHE)/beep.wav $(CACHE)/bling.wav

$(CACHE)/%.bin $(CACHE)/%.wav:
	@mkdir -p $(dir $@)
	@echo "  downloading $(notdir $@)..."
	@curl -fsSL -o $@.tmp $(RELEASE)/$(notdir $@) && mv $@.tmp $@

# CoreML: download tarball, extract mlmodelc directory alongside the .bin
ifeq ($(UNAME),Darwin)
COREML_DIR := $(CACHE)/ggml-tiny-FP16-encoder.mlmodelc
MODELS += $(COREML_DIR)

$(COREML_DIR):
	@echo "  downloading ggml-tiny-FP16-encoder.mlmodelc..."
	@curl -fsSL $(RELEASE)/ggml-tiny-FP16-encoder.mlmodelc.tar.gz | tar xz -C $(CACHE) && touch $@
endif

$(JARVIS_BIN): $(MODELS) src/*.cpp src/*.h src/*.hpp lib/ggml/src/**/*.c lib/ggml/src/**/*.cpp CMakeLists.txt
	@cmake -B build -DCMAKE_BUILD_TYPE=Release $(COREML_FLAG) > /dev/null
	@cmake --build build -j$$(getconf _NPROCESSORS_ONLN)

build/jarvis-client: src/jarvis-client.cpp src/audio_async.cpp src/net.h src/audio_async.hpp CMakeLists.txt
	@cmake -B build -DCMAKE_BUILD_TYPE=Release $(COREML_FLAG) > /dev/null
	@cmake --build build -j$$(getconf _NPROCESSORS_ONLN) --target jarvis-client

client: build/jarvis-client

run: $(JARVIS_BIN)
	./$(JARVIS_BIN)

run-detect: $(JARVIS_BIN)
	./$(JARVIS_BIN) --detect-only

server: $(JARVIS_BIN)
	./$(JARVIS_BIN) --server

enroll:
	uv run python -m jarvis.enroll

templates: $(ENCODE_BIN)
	uv run python -m jarvis.enroll --build

review:
	uv run python -m jarvis.review

coreml:
	uv run --extra coreml python scripts/convert_coreml.py --quantize-f16

clean:
	rm -rf build

.PHONY: client run run-detect server enroll templates review coreml clean
