"""Per-system exclusion / 1-4 scaling tables (§ 2.4 of design doc).

Builds:
    excl_mask  (N, N) bool   -- True for self and 1-2 / 1-3 exclusion pairs
                                (these contribute zero to intra-NB).
    scale_lj   (N, N) float32 -- multiplier on the LJ pair term:
                                 0.0 for self / 1-2 / 1-3,
                                 fudge_lj for 1-4,
                                 1.0 otherwise.
    scale_qq   (N, N) float32 -- same shape for Coulomb (with fudge_qq).
    pair14_idx (n14, 2) int32 -- the 1-4 pairs, kept for inspection.

Source: parmed's adjacency tables. ``top.bonds`` -> 1-2; ``top.angles``
(atom1, atom3) -> 1-3; ``top.adjusts`` -> explicit 1-4 list with per-pair
chgscale (we use the .top global fudgeLJ / fudgeQQ instead -- AMBER and
CHARMM both use a uniform fudge per system).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parmed as pmd


@dataclass(frozen=True)
class ExclusionTable:
    """Static per-system pair-scaling tables.

    SHAPES: scale_lj / scale_qq / excl_mask all (N, N); pair14_idx (n14, 2).
    """
    n_atoms: int
    scale_lj: np.ndarray
    scale_qq: np.ndarray
    excl_mask: np.ndarray
    pair14_idx: np.ndarray
    fudge_lj: float
    fudge_qq: float

    def __post_init__(self) -> None:
        N = self.n_atoms
        assert self.scale_lj.shape == (N, N)
        assert self.scale_qq.shape == (N, N)
        assert self.excl_mask.shape == (N, N)
        assert self.scale_lj.dtype == np.float32
        assert self.scale_qq.dtype == np.float32
        assert self.excl_mask.dtype == np.bool_


def build_exclusion_table(top_path: str) -> ExclusionTable:
    """Build the exclusion / 1-4 scaling tables from a GROMACS .top."""
    p = Path(top_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"GROMACS topology not found: {top_path!s}.")
    top = pmd.load_file(str(p), parametrize=True)
    n = len(top.atoms)

    fudge_lj = float(top.defaults.fudgeLJ)
    fudge_qq = float(top.defaults.fudgeQQ)

    scale_lj = np.ones((n, n), dtype=np.float32)
    scale_qq = np.ones((n, n), dtype=np.float32)
    excl_mask = np.zeros((n, n), dtype=np.bool_)

    # Self pairs always excluded.
    di = np.arange(n)
    scale_lj[di, di] = 0.0
    scale_qq[di, di] = 0.0
    excl_mask[di, di] = True

    # 1-2 (covalent bonds).
    for b in top.bonds:
        i, j = b.atom1.idx, b.atom2.idx
        scale_lj[i, j] = scale_lj[j, i] = 0.0
        scale_qq[i, j] = scale_qq[j, i] = 0.0
        excl_mask[i, j] = excl_mask[j, i] = True

    # 1-3 (bond angles: atom1, atom3).
    for ang in top.angles:
        i, k = ang.atom1.idx, ang.atom3.idx
        scale_lj[i, k] = scale_lj[k, i] = 0.0
        scale_qq[i, k] = scale_qq[k, i] = 0.0
        excl_mask[i, k] = excl_mask[k, i] = True

    # 1-4 (explicit [pairs] section -> parmed adjusts).
    pair14_list: list[tuple[int, int]] = []
    for p14 in top.adjusts:
        i, j = p14.atom1.idx, p14.atom2.idx
        # If a quad with peri=0 happened to also re-emit as a 1-4 pair, the
        # earlier 1-2/1-3 entries already set excl_mask -- 1-4 may NOT be in
        # excl_mask. Safety check:
        if excl_mask[i, j]:
            # The 1-4 pair was already excluded as 1-2 or 1-3 (small rings).
            # Keep the exclusion -- shorter-range listings dominate.
            continue
        scale_lj[i, j] = scale_lj[j, i] = fudge_lj
        scale_qq[i, j] = scale_qq[j, i] = fudge_qq
        pair14_list.append((i, j))

    pair14_idx = (np.asarray(pair14_list, dtype=np.int32)
                  if pair14_list else np.zeros((0, 2), dtype=np.int32))

    return ExclusionTable(
        n_atoms=n,
        scale_lj=scale_lj,
        scale_qq=scale_qq,
        excl_mask=excl_mask,
        pair14_idx=pair14_idx,
        fudge_lj=fudge_lj,
        fudge_qq=fudge_qq,
    )
