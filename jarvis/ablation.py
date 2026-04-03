#!/usr/bin/env python3
"""Ablation study — top performer combinations.

Usage: uv run python -m jarvis.ablation
"""

import struct
from pathlib import Path

import numpy as np

from jarvis import RATE
from jarvis.features import extract_features

MODELS_DIR = Path(__file__).parent.parent / "models"
BASE = Path(__file__).parent.parent


def l2norm(f):
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-10)


def cosine_sim(a, b):
    dot = np.sum(a * b, axis=-1)
    return dot / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-10)


def subdtw(inp, tmpl, step_penalty=0.0):
    n_in, n_t = len(inp), len(tmpl)
    prev = np.full(n_t + 1, 1e30)
    prev[0] = 0.0
    best = 1e30
    for i in range(1, n_in + 1):
        curr = np.full(n_t + 1, 1e30)
        curr[0] = 0.0
        for j in range(1, n_t + 1):
            c = 1.0 - cosine_sim(inp[i - 1], tmpl[j - 1])
            curr[j] = c + min(prev[j - 1], prev[j] + step_penalty, curr[j - 1] + step_penalty)
        if curr[n_t] < best:
            best = curr[n_t]
        prev = curr
    return best / n_t


def score(inp, templates, step_penalty=0.0):
    best_sim = -1.0
    for tmpl in templates:
        cost = subdtw(inp, tmpl, step_penalty=step_penalty)
        sim = 1.0 - cost
        if sim > best_sim:
            best_sim = sim
    return best_sim


def cmvn(features):
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-10
    return (features - mean) / std


def dba(templates, n_iter=5):
    lengths = [len(t) for t in templates]
    median_idx = np.argsort(lengths)[len(lengths) // 2]
    avg = templates[median_idx].copy()
    for _ in range(n_iter):
        accum = np.zeros_like(avg)
        counts = np.zeros(len(avg))
        for tmpl in templates:
            n_a, n_t = len(avg), len(tmpl)
            cost = np.full((n_a + 1, n_t + 1), 1e30)
            cost[0, 0] = 0.0
            for i in range(1, n_a + 1):
                for j in range(1, n_t + 1):
                    c = 1.0 - cosine_sim(avg[i - 1], tmpl[j - 1])
                    cost[i, j] = c + min(cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
            i, j = n_a, n_t
            while i > 0 and j > 0:
                accum[i - 1] += tmpl[j - 1]
                counts[i - 1] += 1
                options = [(cost[i-1,j-1], i-1, j-1), (cost[i-1,j], i-1, j), (cost[i,j-1], i, j-1)]
                _, i, j = min(options, key=lambda x: x[0])
        for i in range(len(avg)):
            if counts[i] > 0:
                avg[i] = accum[i] / counts[i]
    return avg


def run(templates, pos, neg, bg, label, step_penalty=0.0):
    pos_s = score(pos, templates, step_penalty=step_penalty)
    neg_s = score(neg, templates, step_penalty=step_penalty)
    bg_s = score(bg, templates, step_penalty=step_penalty)
    gap = pos_s - neg_s
    print(f"  {label:55s}  pos={pos_s:.4f}  neg={neg_s:.4f}  bg={bg_s:.4f}  gap={gap:.4f}")


def main():
    clips_dir = BASE / "data" / "clips"
    clip_paths = sorted(clips_dir.glob("*.wav"))

    print(f"Extracting features from {len(clip_paths)} clips + 3 test files at layers 3 and final...\n")

    # Extract at layer 3 and final for all clips + test files
    feats = {}
    for layer, layer_name in [(-1, "final"), (3, "layer3")]:
        tmpl_list = []
        for p in clip_paths:
            tmpl_list.append(extract_features(str(p), layer=layer)[:100])
        feats[layer_name] = {
            "templates": tmpl_list,
            "pos": extract_features(str(BASE / "positive.wav"), layer=layer)[:100],
            "neg": extract_features(str(BASE / "negative.wav"), layer=layer)[:100],
            "bg": extract_features(str(BASE / "background.wav"), layer=layer)[:100],
        }

    print(f"  {'Configuration':55s}  {'pos':>8s}  {'neg':>8s}  {'bg':>8s}  {'gap':>8s}")
    print(f"  {'-'*55}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for layer_name in ["final", "layer3"]:
        d = feats[layer_name]
        raw_t = d["templates"]
        raw_p, raw_n, raw_b = d["pos"], d["neg"], d["bg"]

        t_norm = [l2norm(x) for x in raw_t]
        p_norm, n_norm, b_norm = l2norm(raw_p), l2norm(raw_n), l2norm(raw_b)

        # Baseline for this layer
        run(t_norm, p_norm, n_norm, b_norm, f"{layer_name}: baseline")

        # Skip 2
        t_s2 = [x[2:] for x in t_norm]
        run(t_s2, p_norm[2:], n_norm[2:], b_norm[2:], f"{layer_name}: skip 2")

        # CMVN
        t_cmvn = [l2norm(cmvn(x)) for x in raw_t]
        run(t_cmvn, l2norm(cmvn(raw_p)), l2norm(cmvn(raw_n)), l2norm(cmvn(raw_b)),
            f"{layer_name}: CMVN")

        # skip 2 + CMVN
        t_s2c = [l2norm(cmvn(x[2:])) for x in raw_t]
        run(t_s2c, l2norm(cmvn(raw_p[2:])), l2norm(cmvn(raw_n[2:])), l2norm(cmvn(raw_b[2:])),
            f"{layer_name}: skip 2 + CMVN")

        # skip 2 + CMVN + step=0.1
        run(t_s2c, l2norm(cmvn(raw_p[2:])), l2norm(cmvn(raw_n[2:])), l2norm(cmvn(raw_b[2:])),
            f"{layer_name}: skip 2 + CMVN + step=0.1", step_penalty=0.1)

        # DBA + CMVN
        dba_tmpl = dba([cmvn(x) for x in raw_t], n_iter=5)
        run([l2norm(dba_tmpl)], l2norm(cmvn(raw_p)), l2norm(cmvn(raw_n)), l2norm(cmvn(raw_b)),
            f"{layer_name}: DBA + CMVN")

        # skip 2 + DBA + CMVN
        dba_s2 = dba([cmvn(x[2:]) for x in raw_t], n_iter=5)
        run([l2norm(dba_s2)], l2norm(cmvn(raw_p[2:])), l2norm(cmvn(raw_n[2:])), l2norm(cmvn(raw_b[2:])),
            f"{layer_name}: skip 2 + DBA + CMVN")

        # skip 2 + DBA + CMVN + step=0.1
        run([l2norm(dba_s2)], l2norm(cmvn(raw_p[2:])), l2norm(cmvn(raw_n[2:])), l2norm(cmvn(raw_b[2:])),
            f"{layer_name}: skip 2 + DBA + CMVN + step=0.1", step_penalty=0.1)

        print()


if __name__ == "__main__":
    main()
