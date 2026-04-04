enroll:
	uv run python -m jarvis.enroll

review:
	uv run python -m jarvis.review

build:
	uv run python -m jarvis.enroll --build

run:
	./build/jarvis -m models/ggml-tiny.bin -e models/templates.bin -c ./build/weather

compile:
	cmake -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j$$(nproc)

.PHONY: enroll review build run compile
