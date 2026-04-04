"""Extract Whisper encoder features from audio using our C++ encoder.

Calls build/encode which uses the same whisper.cpp code path as the
detector, ensuring templates match runtime embeddings exactly.
"""

import struct
import subprocess
from pathlib import Path

import numpy as np

from jarvis import MODELS_DIR

ENCODE_BIN = Path(__file__).parent.parent / "build" / "encode"
MODEL_PATH = MODELS_DIR / "ggml-tiny.bin"


def extract_features(wav_path: str | Path) -> np.ndarray:
    """Extract encoder features from a WAV file.

    Args:
        wav_path: path to mono 16-bit 16kHz PCM WAV file.

    Returns:
        Features array of shape [T, 384] where T depends on audio length.
    """
    return extract_features_batch([wav_path])[0]


def extract_features_batch(wav_paths: list[str | Path]) -> list[np.ndarray]:
    """Extract encoder features from multiple WAV files in a single model load.

    Args:
        wav_paths: list of paths to mono 16-bit 16kHz PCM WAV files.

    Returns:
        List of feature arrays, each shape [T, 384].
    """
    result = subprocess.run(
        [ENCODE_BIN, MODEL_PATH] + [str(p) for p in wav_paths],
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
