"""Per-basin free energies from cluster populations: dG = -kT ln p (relative)."""
from __future__ import annotations

import numpy as np

from ptmc.config import BOLTZMANN_KJ_PER_MOL_K


def basin_free_energies(labels: np.ndarray, beta: float, k: int | None = None):
    """Relative basin free energies (kJ/mol) from occupancy.

    Returns (populations (k,), dG (k,)) with dG = -(1/beta) ln p, shifted so
    min dG = 0. beta in (kJ/mol)^-1.

    Edge cases
    ----------
    * Empty labels: returns uniform zero populations and zero dG (no samples
      to score; caller should detect this via ``populations.sum() == 0``).
    * Single populated basin: that basin gets dG=0, all others +inf.
    """
    labels = np.asarray(labels, dtype=int)
    if k is None:
        k = int(labels.max()) + 1 if labels.size else 1
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    if k > counts.size:
        counts = np.concatenate([counts, np.zeros(k - counts.size)])
    elif k < counts.size:
        # labels.max() >= k — clip the overflow into the last basin so the
        # returned arrays still match the requested length.
        counts = counts[:k]
    total = counts.sum()
    if total == 0.0:
        # No samples — degenerate input. Return zeros instead of NaN/inf.
        return np.zeros(k, dtype=np.float64), np.zeros(k, dtype=np.float64)
    p = counts / total
    with np.errstate(divide="ignore"):
        dG = -(1.0 / beta) * np.log(p)
    finite = np.isfinite(dG)
    if finite.any():
        dG = dG - dG[finite].min()
    return p, dG
