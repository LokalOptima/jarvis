"""Shared DTW utilities for enrollment and ablation."""

import numpy as np


def l2norm(features: np.ndarray) -> np.ndarray:
    return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)


def cmvn(features: np.ndarray) -> np.ndarray:
    """Cepstral Mean and Variance Normalization."""
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-10
    return (features - mean) / std


def cosine_dist_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance matrix between rows of a and b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return 1.0 - a_norm @ b_norm.T


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.sum(a * b, axis=-1)
    return dot / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-10)


def subdtw(inp: np.ndarray, tmpl: np.ndarray, step_penalty: float = 0.0) -> float:
    """Subsequence DTW: best match of tmpl anywhere within inp."""
    n_in, n_t = len(inp), len(tmpl)
    dist = cosine_dist_matrix(inp, tmpl)
    prev = np.full(n_t + 1, 1e30)
    prev[0] = 0.0
    best = 1e30
    for i in range(1, n_in + 1):
        curr = np.full(n_t + 1, 1e30)
        curr[0] = 0.0
        for j in range(1, n_t + 1):
            c = dist[i - 1, j - 1]
            curr[j] = c + min(prev[j - 1], prev[j] + step_penalty, curr[j - 1] + step_penalty)
        if curr[n_t] < best:
            best = curr[n_t]
        prev = curr
    return best / n_t


def dba(templates: list[np.ndarray], n_iter: int = 5) -> np.ndarray:
    """Dynamic Barycenter Averaging — merge multiple templates into one."""
    lengths = [len(t) for t in templates]
    median_idx = np.argsort(lengths)[len(lengths) // 2]
    avg = templates[median_idx].copy()
    for _ in range(n_iter):
        accum = np.zeros_like(avg)
        counts = np.zeros(len(avg))
        for tmpl in templates:
            n_a, n_t = len(avg), len(tmpl)
            dist = cosine_dist_matrix(avg, tmpl)
            cost = np.full((n_a + 1, n_t + 1), 1e30)
            cost[0, 0] = 0.0
            for i in range(1, n_a + 1):
                for j in range(1, n_t + 1):
                    cost[i, j] = dist[i - 1, j - 1] + min(
                        cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1],
                    )
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
