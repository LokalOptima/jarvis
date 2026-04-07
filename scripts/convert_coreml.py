"""Convert Whisper Tiny encoder to Core ML for Apple Neural Engine inference.

Produces a fixed-shape model for 2-second audio input:
  Input:  [1, 80, 200]  (80 mel channels x 200 frames = 2s at 16kHz)
  Output: [1, 100, 384] (100 encoder frames x 384 hidden dim)

Usage:
    uv run python scripts/convert_coreml.py [--quantize-f16]

Requires: coremltools, openai-whisper, torch
"""

import argparse
import subprocess
import shutil
from pathlib import Path

import torch
import coremltools as ct
from whisper import load_model
import whisper.model

# Disable SDPA for stable tracing across torch versions.
whisper.model.MultiHeadAttention.use_sdpa = False

from jarvis import CACHE_DIR

# 2 seconds of audio at 16kHz → 200 mel frames → 100 encoder frames after stride-2 conv
N_MELS = 80
MEL_FRAMES = 200
ENCODER_FRAMES = MEL_FRAMES // 2  # 100


def convert_encoder(model_name: str, quantize_f16: bool = False) -> Path:
    whisper_model = load_model(model_name).cpu()
    encoder = whisper_model.encoder
    encoder.eval()

    # Slice positional embedding from [1500, 384] to [100, 384].
    # The full embedding causes a shape mismatch assertion in forward()
    # when tracing with our shorter 2s input.
    original_pe = encoder.positional_embedding
    assert original_pe.shape[0] >= ENCODER_FRAMES, (
        f"positional_embedding has {original_pe.shape[0]} frames, need {ENCODER_FRAMES}"
    )
    encoder.register_buffer("positional_embedding", original_pe[:ENCODER_FRAMES])

    input_shape = (1, N_MELS, MEL_FRAMES)
    example_input = torch.randn(input_shape)

    traced = torch.jit.trace(encoder, example_input)

    ml_model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="logmel_data", shape=input_shape)],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16 if quantize_f16 else ct.precision.FLOAT32,
    )

    mlpackage_path = CACHE_DIR / f"ggml-{model_name}-FP16-encoder.mlpackage"
    ml_model.save(str(mlpackage_path))
    print(f"Saved mlpackage: {mlpackage_path}")

    return mlpackage_path


def compile_model(mlpackage_path: Path, model_name: str) -> Path:
    """Compile .mlpackage to .mlmodelc using coremltools."""
    target_path = CACHE_DIR / f"ggml-{model_name}-FP16-encoder.mlmodelc"
    if target_path.exists():
        shutil.rmtree(target_path)

    ct.utils.compile_model(str(mlpackage_path), str(target_path))
    print(f"Compiled model: {target_path}")

    return target_path


def main():
    parser = argparse.ArgumentParser(description="Convert Whisper encoder to Core ML")
    parser.add_argument(
        "--model", type=str, default="tiny",
        help="Whisper model name (default: tiny)",
    )
    parser.add_argument(
        "--quantize-f16", action="store_true",
        help="Quantize weights to FP16 (recommended for ANE)",
    )
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    mlpackage_path = convert_encoder(args.model, args.quantize_f16)
    compile_model(mlpackage_path, args.model)


if __name__ == "__main__":
    main()
