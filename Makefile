build/jarvis: src/*.cpp src/*.h src/*.hpp lib/ggml/src/**/*.c lib/ggml/src/**/*.cpp CMakeLists.txt
	@cmake -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
	@cmake --build build -j$$(nproc)

build/jarvis-client: src/jarvis-client.cpp src/audio_async.cpp src/net.h src/audio_async.hpp CMakeLists.txt
	@cmake -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
	@cmake --build build -j$$(nproc) --target jarvis-client

client: build/jarvis-client

run: build/jarvis
	./build/jarvis

run-detect: build/jarvis
	./build/jarvis --detect-only

server: build/jarvis
	./build/jarvis --server

enroll:
	uv run python -m jarvis.enroll

templates:
	uv run python -m jarvis.enroll --build

review:
	uv run python -m jarvis.review

clean:
	rm -rf build

.PHONY: client run run-detect server enroll templates review clean
