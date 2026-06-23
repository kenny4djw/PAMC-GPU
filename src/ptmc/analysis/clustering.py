"""Cluster sampled poses into orientation basins by their contact normals."""
from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2


def cluster_orientations(normals: np.ndarray, k: int = 2, seed: int = 0):
    """K-means on unit contact-normal vectors. SHAPE normals (N,3).

    Returns ``(labels (N,), centroids (k,3) re-normalized)``. Always returns
    exactly ``k`` centroids and labels in ``[0, k)`` so downstream code
    (population, free energy) can index by basin without bounds errors.

    Edge cases
    ----------
    * N == 0: returns empty labels and zero-vector centroids.
    * N < k:  populates the first N centroids with the input vectors, fills
              the remaining slots with the first vector (zeroes if any
              centroid was all-zero).
    * kmeans2 collapses to fewer than k unique clusters: the unused centroid
      rows stay as kmeans2 returned them; relabel still confined to [0, k).
    """
    x = np.asarray(normals, dtype=np.float64)
    N = x.shape[0]

    # Degenerate inputs: don't even call kmeans2 — it raises on N=0 and is
    # unreliable on N<k. Return a deterministic fallback.
    if N == 0:
        return (np.zeros(0, dtype=int),
                np.zeros((k, 3), dtype=np.float64))
    if N < k:
        labels = np.arange(N, dtype=int)
        centroids = np.zeros((k, 3), dtype=np.float64)
        centroids[:N] = x
        # Fill remaining centroids by repeating the first to keep shapes valid.
        if N < k:
            centroids[N:] = x[0]
        norm = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.where(norm > 0, norm, 1.0)
        return labels, centroids

    rng = np.random.default_rng(seed)
    centroids, labels = kmeans2(x, k, minit="++", seed=int(rng.integers(1 << 30)))
    labels = np.asarray(labels, dtype=int)
    # Defensive clip: kmeans2 should never return labels >= k, but a corrupt
    # init can. Clamp so downstream bincount/indexing is safe.
    np.clip(labels, 0, k - 1, out=labels)
    norm = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.where(norm > 0, norm, 1.0)
    return labels, centroids
