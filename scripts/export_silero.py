"""
Export Silero VAD (16kHz branch) weights from ONNX to a simple binary format.

Format:
  Header: "SVAD" (4 bytes) + version (uint32) + n_tensors (uint32)
  Per tensor: name_len (uint32) + name (bytes) + n_dims (uint32)
              + dims (uint32[n_dims]) + data (float32[product(dims)])

Usage:
    uv run python scripts/export_silero.py
    uv run python scripts/export_silero.py -m ~/models/silero_vad.onnx -o models/silero_vad.bin
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import onnx


# Weight tensor names we need (in the order they appear in the 16kHz branch)
WEIGHT_NAMES = [
    "stft.forward_basis_buffer",
    "encoder.0.reparam_conv.weight",
    "encoder.0.reparam_conv.bias",
    "encoder.1.reparam_conv.weight",
    "encoder.1.reparam_conv.bias",
    "encoder.2.reparam_conv.weight",
    "encoder.2.reparam_conv.bias",
    "encoder.3.reparam_conv.weight",
    "encoder.3.reparam_conv.bias",
    "decoder.rnn.weight_ih",
    "decoder.rnn.weight_hh",
    "decoder.rnn.bias_ih",
    "decoder.rnn.bias_hh",
    "decoder.decoder.2.weight",
    "decoder.decoder.2.bias",
]


def extract_16k_weights(model_path: str) -> dict[str, np.ndarray]:
    model = onnx.load(model_path)

    # Top-level: Constant(16000) → Equal(sr) → If(then=16kHz, else=8kHz)
    if_node = model.graph.node[2]
    assert if_node.op_type == "If"

    weights = {}
    for attr in if_node.attribute:
        if attr.name == "then_branch":
            for node in attr.g.node:
                if node.op_type == "Constant":
                    name = node.output[0]
                    # Strip the ONNX inline prefix
                    if "Inline_0__" in name:
                        name = name.split("Inline_0__")[-1]
                    if name in WEIGHT_NAMES:
                        t = onnx.numpy_helper.to_array(node.attribute[0].t)
                        weights[name] = t.astype(np.float32)

    # Verify we got everything
    missing = set(WEIGHT_NAMES) - set(weights.keys())
    if missing:
        raise RuntimeError(f"Missing weights: {missing}")

    return weights


def write_binary(weights: dict[str, np.ndarray], out_path: str):
    with open(out_path, "wb") as f:
        # Header
        f.write(b"SVAD")
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", len(weights)))

        total_params = 0
        for name in WEIGHT_NAMES:
            t = weights[name]
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(t.shape)))
            for d in t.shape:
                f.write(struct.pack("<I", d))
            f.write(t.tobytes())
            total_params += t.size
            print(f"  {name:50s} {str(t.shape):20s} {t.size:>8d} params")

        print(f"\nTotal: {total_params} params ({total_params * 4 / 1024:.1f} KB)")
        print(f"Written to {out_path} ({f.tell()} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Export Silero VAD weights")
    parser.add_argument("-m", "--model", default=str(Path.home() / "models/silero_vad.onnx"))
    parser.add_argument("-o", "--output", default="models/silero_vad.bin")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    weights = extract_16k_weights(args.model)
    write_binary(weights, args.output)


if __name__ == "__main__":
    main()
