"""Extract Whisper Tiny encoder features from audio files.

Uses openai-whisper Python for training-time feature extraction.
The C++ runtime uses whisper.cpp for the same computation.
"""

import functools
from pathlib import Path

import numpy as np
import torch
import whisper


@functools.cache
def _load_encoder():
    """Load Whisper Tiny encoder with variable-length support."""
    model = whisper.load_model("tiny", device="cpu")
    encoder = model.encoder
    encoder.eval()

    # Monkey-patch forward to support variable-length input.
    # The original forward hardcodes positional_embedding[:1500].
    # Since Whisper uses sinusoidal (absolute) positional encoding,
    # slicing to actual sequence length is mathematically correct.
    original_blocks = encoder.blocks
    original_ln = encoder.ln_post
    original_conv1 = encoder.conv1
    original_conv2 = encoder.conv2
    original_pe = encoder.positional_embedding

    def flexible_forward(x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.gelu(original_conv1(x))
        x = torch.nn.functional.gelu(original_conv2(x))
        x = x.permute(0, 2, 1)  # [B, T, C]
        seq_len = x.shape[1]
        x = (x + original_pe[:seq_len]).to(x.dtype)
        for block in original_blocks:
            x = block(x)
        x = original_ln(x)
        return x

    encoder.forward = flexible_forward
    return encoder


def extract_features(audio_path: str | Path) -> np.ndarray:
    """Extract encoder features from a WAV file.

    Args:
        audio_path: Path to 16kHz mono WAV file.

    Returns:
        Features array of shape [T, 384] where T depends on audio length.
        For 1.5s audio: T ≈ 75 frames (20ms per frame).
    """
    encoder = _load_encoder()
    audio = whisper.load_audio(str(audio_path))
    mel = whisper.log_mel_spectrogram(audio, n_mels=80)
    mel = mel.unsqueeze(0)  # [1, 80, mel_frames]

    with torch.no_grad():
        features = encoder(mel)  # [1, T, 384]

    return features.squeeze(0).numpy()  # [T, 384]


def extract_features_from_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract encoder features from a raw audio array.

    Args:
        audio: float32 or int16 numpy array, mono.
        sr: Sample rate (must be 16000).

    Returns:
        Features array of shape [T, 384].
    """
    if sr != 16000:
        raise ValueError(f"Expected 16kHz audio, got {sr}Hz")

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    encoder = _load_encoder()
    mel = whisper.log_mel_spectrogram(torch.from_numpy(audio), n_mels=80)
    mel = mel.unsqueeze(0)

    with torch.no_grad():
        features = encoder(mel)

    return features.squeeze(0).numpy()
