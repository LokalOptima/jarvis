#!/usr/bin/env python3
"""Ablation study — top performer combinations.

Usage: uv run python -m jarvis.ablation
"""

from pathlib import Path

import numpy as np

from jarvis import RATE, ONSET_SKIP, STEP_PENALTY
from jarvis.dtw import l2norm, cmvn, subdtw, dba
from jarvis.features import extract_features

BASE = Path(__file__).parent.parent


def score(inp, templates, step_penalty=0.0):
    best_sim = -1.0
    for tmpl in templates:
        cost = subdtw(inp, tmpl, step_penalty=step_penalty)
        sim = 1.0 - cost
        if sim > best_sim:
            best_sim = sim
    return best_sim


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
        t_s2 = [x[ONSET_SKIP:] for x in t_norm]
        run(t_s2, p_norm[ONSET_SKIP:], n_norm[ONSET_SKIP:], b_norm[ONSET_SKIP:], f"{layer_name}: skip 2")

        # CMVN
        t_cmvn = [l2norm(cmvn(x)) for x in raw_t]
        run(t_cmvn, l2norm(cmvn(raw_p)), l2norm(cmvn(raw_n)), l2norm(cmvn(raw_b)),
            f"{layer_name}: CMVN")

        # skip 2 + CMVN
        t_s2c = [l2norm(cmvn(x[ONSET_SKIP:])) for x in raw_t]
        run(t_s2c, l2norm(cmvn(raw_p[ONSET_SKIP:])), l2norm(cmvn(raw_n[ONSET_SKIP:])), l2norm(cmvn(raw_b[ONSET_SKIP:])),
            f"{layer_name}: skip 2 + CMVN")

        # skip 2 + CMVN + step=0.1
        run(t_s2c, l2norm(cmvn(raw_p[ONSET_SKIP:])), l2norm(cmvn(raw_n[ONSET_SKIP:])), l2norm(cmvn(raw_b[ONSET_SKIP:])),
            f"{layer_name}: skip 2 + CMVN + step=0.1", step_penalty=STEP_PENALTY)

        # DBA + CMVN
        dba_tmpl = dba([cmvn(x) for x in raw_t], n_iter=5)
        run([l2norm(dba_tmpl)], l2norm(cmvn(raw_p)), l2norm(cmvn(raw_n)), l2norm(cmvn(raw_b)),
            f"{layer_name}: DBA + CMVN")

        # skip 2 + DBA + CMVN
        dba_s2 = dba([cmvn(x[ONSET_SKIP:]) for x in raw_t], n_iter=5)
        run([l2norm(dba_s2)], l2norm(cmvn(raw_p[ONSET_SKIP:])), l2norm(cmvn(raw_n[ONSET_SKIP:])), l2norm(cmvn(raw_b[ONSET_SKIP:])),
            f"{layer_name}: skip 2 + DBA + CMVN")

        # skip 2 + DBA + CMVN + step=0.1
        run([l2norm(dba_s2)], l2norm(cmvn(raw_p[ONSET_SKIP:])), l2norm(cmvn(raw_n[ONSET_SKIP:])), l2norm(cmvn(raw_b[ONSET_SKIP:])),
            f"{layer_name}: skip 2 + DBA + CMVN + step=0.1", step_penalty=STEP_PENALTY)

        print()


if __name__ == "__main__":
    main()
