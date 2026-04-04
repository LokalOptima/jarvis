build/jarvis: src/*.cpp src/*.h src/*.hpp lib/ggml/src/**/*.c lib/ggml/src/**/*.cpp CMakeLists.txt
	@cmake -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
	@cmake --build build -j$$(nproc)

run: build/jarvis
	./build/jarvis

run-detect: build/jarvis
	./build/jarvis --detect-only

enroll:
	uv run python -m jarvis.enroll

templates:
	uv run python -m jarvis.enroll --build

review:
	uv run python -m jarvis.review

clean:
	rm -rf build

.PHONY: run run-detect enroll templates review clean
