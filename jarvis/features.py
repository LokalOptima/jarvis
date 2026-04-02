"""Extract Whisper Tiny encoder features from audio.

Uses openai-whisper Python for feature extraction at enrollment time.
The C++ runtime uses whisper.cpp for the same computation.
"""

import functools
from pathlib import Path

import numpy as np
import torch
import whisper

from jarvis import RATE


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


def extract_features(audio: np.ndarray | str | Path) -> np.ndarray:
    """Extract encoder features from audio.

    Args:
        audio: float32/int16 numpy array (mono 16kHz), or path to WAV file.

    Returns:
        Features array of shape [T, 384] where T depends on audio length.
    """
    if isinstance(audio, (str, Path)):
        audio = whisper.load_audio(str(audio))

    if isinstance(audio, np.ndarray):
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        audio = torch.from_numpy(audio)

    encoder = _load_encoder()
    mel = whisper.log_mel_spectrogram(audio, n_mels=80).unsqueeze(0)

    with torch.no_grad():
        features = encoder(mel)

    return features.squeeze(0).numpy()
