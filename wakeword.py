#!/usr/bin/env python3
"""Wake word detection from microphone using openwakeword."""

import argparse
import sys
import time

import numpy as np
import pyaudio
from openwakeword.model import Model

RATE = 16000
CHUNK = 1280  # 80ms at 16kHz
FORMAT = pyaudio.paInt16
CHANNELS = 1

AVAILABLE_MODELS = [
    "alexa",
    "hey_mycroft",
    "hey_jarvis",
    "hey_rhasspy",
    "timer",
    "weather",
]


def list_audio_devices():
    """Print available audio input devices."""
    pa = pyaudio.PyAudio()
    print("Available input devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  [{i}] {info['name']} (channels: {info['maxInputChannels']})")
    pa.terminate()


def run(
    models: list[str] | None = None,
    threshold: float = 0.5,
    device_index: int | None = None,
    vad_threshold: float = 0.0,
    debug: bool = False,
):
    oww = Model(
        wakeword_models=models or [],
        inference_framework="onnx",
        vad_threshold=vad_threshold if vad_threshold > 0 else 0,
    )

    pa = pyaudio.PyAudio()
    stream_kwargs = dict(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    if device_index is not None:
        stream_kwargs["input_device_index"] = device_index

    stream = pa.open(**stream_kwargs)

    loaded = list(oww.models.keys())
    print(f"Listening for: {', '.join(loaded)}")
    print(f"Threshold: {threshold}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            audio = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            scores = oww.predict(audio)

            if debug:
                active = {n: s for n, s in scores.items() if s > 0.01}
                if active:
                    parts = " | ".join(f"{n}: {s:.3f}" for n, s in active.items())
                    print(f"\r  {parts}    ", end="", flush=True)

            for name, score in scores.items():
                if score > threshold:
                    t = time.strftime("%H:%M:%S")
                    print(f"\r[{t}] {name}: {score:.4f}")
                    oww.reset()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def main():
    parser = argparse.ArgumentParser(description="Wake word detection from microphone")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=AVAILABLE_MODELS,
        help=f"Wake word models to load (default: all). Choices: {', '.join(AVAILABLE_MODELS)}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (use --list-devices to see options)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.0,
        help="VAD threshold for filtering (0 = disabled, try 0.5)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show live scores for all models (useful for tuning)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    run(
        models=args.models,
        threshold=args.threshold,
        device_index=args.device,
        vad_threshold=args.vad_threshold,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
