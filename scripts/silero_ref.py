"""
Run Silero VAD (ONNX) frame-by-frame on a WAV file.
Save per-frame speech probabilities to .npz for validation against ggml port.

Usage:
    uv run python scripts/silero_ref.py data/recording.wav
    uv run python scripts/silero_ref.py data/recording.wav -o ref_output.npz
"""

import argparse
import wave
import numpy as np
import onnxruntime as ort


def load_wav_16k(path: str) -> np.ndarray:
    with wave.open(path) as w:
        assert w.getframerate() == 16000, f"Expected 16kHz, got {w.getframerate()}"
        assert w.getnchannels() == 1, f"Expected mono, got {w.getnchannels()}"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def run_silero_vad(model_path: str, audio: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    sess = ort.InferenceSession(model_path)

    # State: (2, 1, 128) — h and c for LSTM
    state = np.zeros((2, 1, 128), dtype=np.float32)
    sr = np.array(16000, dtype=np.int64)

    # Context: last 64 samples from previous chunk (matches official OnnxWrapper)
    context_size = 64 if chunk_size == 512 else 32
    context = np.zeros(context_size, dtype=np.float32)

    n_chunks = len(audio) // chunk_size
    probs = np.zeros(n_chunks, dtype=np.float32)

    for i in range(n_chunks):
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]
        # Prepend context (64 samples) to chunk — model expects 576 samples
        inp = np.concatenate([context, chunk]).reshape(1, -1)
        context = chunk[-context_size:]

        out, state_out = sess.run(
            ["output", "stateN"],
            {"input": inp, "sr": sr, "state": state},
        )

        probs[i] = out.item()
        state = state_out

    return probs


def main():
    parser = argparse.ArgumentParser(description="Silero VAD ONNX reference")
    parser.add_argument("wav", help="Input WAV file (16kHz mono)")
    parser.add_argument("-m", "--model", default=str(__import__("pathlib").Path.home() / "models/silero_vad.onnx"))
    parser.add_argument("-o", "--output", default=None, help="Output .npz path")
    parser.add_argument("--chunk-size", type=int, default=512, help="Samples per frame (512 = 32ms)")
    args = parser.parse_args()

    audio = load_wav_16k(args.wav)
    print(f"Audio: {len(audio)} samples ({len(audio)/16000:.1f}s)")

    probs = run_silero_vad(args.model, audio, args.chunk_size)
    print(f"Frames: {len(probs)}")
    print(f"Speech prob range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"Mean: {probs.mean():.4f}")

    # Show a simple text visualization
    n_bins = min(80, len(probs))
    bin_size = len(probs) // n_bins
    print(f"\nTimeline ({len(probs)} frames, {args.chunk_size/16000*1000:.0f}ms each):")
    for i in range(n_bins):
        chunk_probs = probs[i * bin_size : (i + 1) * bin_size]
        avg = chunk_probs.mean()
        bar = "#" if avg > 0.5 else "." if avg > 0.1 else " "
        print(bar, end="")
    print()

    out_path = args.output or args.wav.rsplit(".", 1)[0] + "_vad_ref.npz"
    np.savez(out_path, probs=probs, chunk_size=np.array(args.chunk_size), sample_rate=np.array(16000))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
