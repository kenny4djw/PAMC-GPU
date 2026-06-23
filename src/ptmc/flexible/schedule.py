"""Depth-sorted residue-parallel schedule for chi kinematics.

Given a ChiTopology, re-bucket the K chis into a (max_n_chi, n_flex_res, ...)
layout. § 4.2 of the design doc: 4 serial depth steps × residue-parallel.

Within depth d, masks across residues are disjoint (each chi only moves atoms
inside its own residue's sidechain), so all residues at depth d can be applied
in a single fused pass via per-atom gather of (R, pivot) by owner-residue index.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ptmc.flexible.topology import ChiTopology


@dataclass(frozen=True)
class ChiSchedule:
    """Depth-bucketed chi tensors for residue-parallel kinematics.

    SHAPES (D = max_n_chi, R = n_flex_res, N = n_atoms):
        chi_by_depth    (D, R, 4) int32  -- four atom indices per chi
        mask_by_depth   (D, R, N) bool   -- downstream atoms per chi
        valid_by_depth  (D, R)    bool   -- True if residue r has a chi at depth d
        chi_global_idx  (D, R)    int32  -- flat chi index for chi[d, r] (0 if invalid)
        owner_per_depth (D, N)    int32  -- residue idx owning each atom at depth d
                                           (-1 if atom does not rotate at this depth)
        flex_resids     (R,)      int32  -- the residue.idx values for the R slots
        n_atoms, max_n_chi, n_flex_res    -- static ints
    """
    chi_by_depth: np.ndarray
    mask_by_depth: np.ndarray
    valid_by_depth: np.ndarray
    chi_global_idx: np.ndarray
    owner_per_depth: np.ndarray
    flex_resids: np.ndarray
    n_atoms: int
    max_n_chi: int
    n_flex_res: int

    def __post_init__(self) -> None:
        D, R, N = self.max_n_chi, self.n_flex_res, self.n_atoms
        assert self.chi_by_depth.shape == (D, R, 4)
        assert self.mask_by_depth.shape == (D, R, N)
        assert self.valid_by_depth.shape == (D, R)
        assert self.chi_global_idx.shape == (D, R)
        assert self.owner_per_depth.shape == (D, N)
        assert self.flex_resids.shape == (R,)
        assert self.chi_by_depth.dtype == np.int32
        assert self.mask_by_depth.dtype == np.bool_
        assert self.valid_by_depth.dtype == np.bool_
        assert self.chi_global_idx.dtype == np.int32
        assert self.owner_per_depth.dtype == np.int32


def build_chi_schedule(topo: ChiTopology) -> ChiSchedule:
    """Re-bucket a ChiTopology into a depth-sorted residue-parallel schedule."""
    K = topo.k
    N = topo.n_atoms

    if K == 0:
        empty_i32 = np.zeros((0, 0, 4), dtype=np.int32)
        empty_b = np.zeros((0, 0, N), dtype=np.bool_)
        empty_valid = np.zeros((0, 0), dtype=np.bool_)
        empty_gidx = np.zeros((0, 0), dtype=np.int32)
        empty_owner = -np.ones((0, N), dtype=np.int32)
        return ChiSchedule(
            chi_by_depth=empty_i32, mask_by_depth=empty_b,
            valid_by_depth=empty_valid, chi_global_idx=empty_gidx,
            owner_per_depth=empty_owner,
            flex_resids=np.zeros((0,), dtype=np.int32),
            n_atoms=N, max_n_chi=0, n_flex_res=0,
        )

    unique_resids, inverse = np.unique(topo.chi_resid, return_inverse=True)
    inverse = inverse.astype(np.int32, copy=False)
    n_flex_res = int(unique_resids.shape[0])
    max_n_chi = int(topo.chi_depth.max()) + 1

    chi_by_depth = np.zeros((max_n_chi, n_flex_res, 4), dtype=np.int32)
    mask_by_depth = np.zeros((max_n_chi, n_flex_res, N), dtype=np.bool_)
    valid_by_depth = np.zeros((max_n_chi, n_flex_res), dtype=np.bool_)
    chi_global_idx = np.zeros((max_n_chi, n_flex_res), dtype=np.int32)
    owner_per_depth = -np.ones((max_n_chi, N), dtype=np.int32)

    for k in range(K):
        d = int(topo.chi_depth[k])
        r = int(inverse[k])
        if valid_by_depth[d, r]:
            raise ValueError(
                f"duplicate chi at depth={d}, residue slot={r} "
                f"(residue idx {int(unique_resids[r])}). "
                f"ChiTopology malformed — each residue should have at most one "
                f"chi per depth.")
        chi_by_depth[d, r, :] = topo.chi_idx[k]
        mask_by_depth[d, r, :] = topo.chi_mask[k]
        valid_by_depth[d, r] = True
        chi_global_idx[d, r] = k
        owned = topo.chi_mask[k]
        prev_owner = owner_per_depth[d, owned]
        if not np.all((prev_owner == -1) | (prev_owner == r)):
            conflicting = np.where(owned & (owner_per_depth[d] >= 0)
                                   & (owner_per_depth[d] != r))[0]
            raise ValueError(
                f"chi masks overlap at depth={d}: residues "
                f"{set(int(owner_per_depth[d, a]) for a in conflicting)} and {r} "
                f"both claim atoms {conflicting.tolist()[:5]}... "
                f"Disjoint-mask invariant for residue-parallel scheduling broken.")
        owner_per_depth[d, owned] = r

    return ChiSchedule(
        chi_by_depth=chi_by_depth,
        mask_by_depth=mask_by_depth,
        valid_by_depth=valid_by_depth,
        chi_global_idx=chi_global_idx,
        owner_per_depth=owner_per_depth,
        flex_resids=unique_resids.astype(np.int32, copy=False),
        n_atoms=N,
        max_n_chi=max_n_chi,
        n_flex_res=n_flex_res,
    )
