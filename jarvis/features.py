"""Extract Whisper encoder features from audio using our C++ encoder.

Calls build/encode-{cpu,coreml} which uses the same whisper.cpp code path as the
detector, ensuring templates match runtime embeddings exactly.
"""

import struct
import subprocess
from pathlib import Path

import numpy as np

from jarvis import MODELS_DIR

BUILD_DIR = Path(__file__).parent.parent / "build"


def _find_encode_bin() -> Path:
    for name in ("encode-coreml", "encode-cpu"):
        p = BUILD_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "No encode binary found. Run 'make' first to build encode-{cpu,coreml}."
    )


def _find_default_model() -> Path:
    for name in ("ggml-tiny-FP16.bin", "ggml-tiny-Q8.bin"):
        p = MODELS_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "No model found. Expected models/ggml-tiny-FP16.bin or models/ggml-tiny-Q8.bin."
    )


ENCODE_BIN = _find_encode_bin()
DEFAULT_MODEL = _find_default_model()


def model_tag(model_path: Path) -> str:
    """Derive a short tag from the model filename.

    ggml-tiny-FP16.bin -> "tiny-FP16"
    ggml-tiny-Q8.bin   -> "tiny-Q8"
    """
    stem = model_path.stem
    parts = stem.split("-")
    parts = [p for p in parts if p != "ggml"]
    return "-".join(parts) if parts else stem


def extract_features(wav_path: str | Path, model: Path = DEFAULT_MODEL) -> np.ndarray:
    return extract_features_batch([wav_path], model=model)[0]


def extract_features_batch(
    wav_paths: list[str | Path],
    model: Path = DEFAULT_MODEL,
) -> list[np.ndarray]:
    result = subprocess.run(
        [ENCODE_BIN, model] + [str(p) for p in wav_paths],
        stdout=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"encode failed (exit {result.returncode})")

    buf = result.stdout
    offset = 0
    features = []
    for wav_path in wav_paths:
        n_frames = struct.unpack_from("i", buf, offset)[0]; offset += 4
        dim = struct.unpack_from("i", buf, offset)[0]; offset += 4

        if n_frames == 0:
            raise RuntimeError(f"encode returned 0 frames for {wav_path}")

        data = np.frombuffer(buf, dtype=np.float32, count=n_frames * dim, offset=offset)
        offset += n_frames * dim * 4
        features.append(data.reshape(n_frames, dim))

    return features
