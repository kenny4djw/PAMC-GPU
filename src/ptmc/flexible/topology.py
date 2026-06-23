"""Build per-system chi-topology static tensors (CPU/numpy, one-shot).

Given a GROMACS .top, walk residues and produce:
    chi_idx    (K, 4)  int32  -- four atom indices defining each dihedral
    chi_mask   (K, N)  bool   -- downstream atoms that move when chi rotates
    chi_resid  (K,)    int32  -- residue index of each chi
    chi_depth  (K,)    int32  -- 0=chi1, 1=chi2, ... within its residue

K = sum of n_chi(resname) across all residues, skipping residues that lack
required heavy atoms (truncated termini, unusual PDBs).

The chi_mask is the BFS-reachable set from atom k while blocking traversal
through j. It INCLUDES k (pivot — rotation leaves it fixed, harmless) but
EXCLUDES j and the entire upstream side of the molecule.

This is § 2.1 - 2.3 of the design doc. Exclusion tables (§ 2.4), dihedral
tables (§ 2.5) and the depth-sorted schedule (§ 2.6) are built by later
stages on demand.
"""
from __future__ import annotations

import logging

_LOGGER = logging.getLogger(__name__)

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parmed as pmd

from ptmc.flexible.chi_table import chi_atom_names


@dataclass(frozen=True)
class ChiTopology:
    """Static per-system chi-topology tensors.

    SHAPE: chi_idx (K,4) int32; chi_mask (K,N) bool; chi_resid (K,) int32;
    chi_depth (K,) int32; n_atoms = N.
    """
    chi_idx: np.ndarray
    chi_mask: np.ndarray
    chi_resid: np.ndarray
    chi_depth: np.ndarray
    n_atoms: int

    def __post_init__(self) -> None:
        k = self.chi_idx.shape[0]
        assert self.chi_idx.shape == (k, 4)
        assert self.chi_mask.shape == (k, self.n_atoms)
        assert self.chi_resid.shape == (k,)
        assert self.chi_depth.shape == (k,)
        assert self.chi_idx.dtype == np.int32
        assert self.chi_mask.dtype == np.bool_
        assert self.chi_resid.dtype == np.int32
        assert self.chi_depth.dtype == np.int32

    @property
    def k(self) -> int:
        return int(self.chi_idx.shape[0])


def _build_adjacency(top: "pmd.Structure") -> list[list[int]]:
    """Build undirected adjacency from parmed bonds list."""
    n = len(top.atoms)
    adj: list[list[int]] = [[] for _ in range(n)]
    for b in top.bonds:
        i = b.atom1.idx
        j = b.atom2.idx
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _bfs_downstream(adj: list[list[int]], k: int, j: int, n: int) -> np.ndarray:
    """BFS from k while refusing to traverse j. Returns (n,) bool mask.

    The mask is the set of atoms that move when the j-k bond is rotated.
    It includes k itself (harmless: k is the pivot, R @ 0 + pivot = pivot).
    """
    mask = np.zeros(n, dtype=bool)
    visited = {j, k}
    mask[k] = True
    queue = [k]
    while queue:
        cur = queue.pop()
        for nbr in adj[cur]:
            if nbr in visited:
                continue
            visited.add(nbr)
            mask[nbr] = True
            queue.append(nbr)
    return mask


def build_chi_topology(top_path: str) -> ChiTopology:
    """Build the static chi-topology tensors from a GROMACS .top.

    Raises ``FileNotFoundError`` / ``ValueError`` mirroring
    ``parse_topology`` for consistency.
    """
    p = Path(top_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"GROMACS topology not found: {top_path!s} "
            f"(resolved to {p.resolve()!s}).")
    try:
        # parametrize=True mirrors parse_topology — empirically the only stable
        # parmed path; parametrize=False crashes inside _process_normal_dihedral
        # for some real-world tops (e.g. data/1UBQ.top).
        top = pmd.load_file(str(p), parametrize=True)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse GROMACS topology {top_path!s}: {exc}."
        ) from exc

    # Strip disulfide (CYS SG-SG) bonds so the chi-mask BFS does not
    # cross into partner residues.  Disulfide bonds create cycles in the
    # bond graph that cause the chi_mask propagation to leak between
    # residues, breaking the disjoint-mask invariant required by the
    # residue-parallel schedule (§ 4.2 design doc).
    ss_removed = 0
    for b in list(top.bonds):
        a1, a2 = b.atom1, b.atom2
        r1 = a1.residue.name.strip().upper()
        r2 = a2.residue.name.strip().upper()
        if r1 in ("CYS", "CYX") and r2 in ("CYS", "CYX"):
            if a1.name.strip().upper() == "SG" and a2.name.strip().upper() == "SG":
                top.bonds.remove(b)
                ss_removed += 1
    if ss_removed:
        _LOGGER.info("Removed %d disulfide bond(s) from topology for chi-mask BFS.", ss_removed)

    n = len(top.atoms)
    adj = _build_adjacency(top)

    chi_idx_rows: list[tuple[int, int, int, int]] = []
    chi_mask_rows: list[np.ndarray] = []
    chi_resid_rows: list[int] = []
    chi_depth_rows: list[int] = []

    for res in top.residues:
        resname = res.name.strip().upper()
        defs = chi_atom_names(resname)
        if not defs:
            continue
        name2idx = {a.name.strip(): a.idx for a in res.atoms}
        for depth, (na, nb, nc, nd) in enumerate(defs):
            if not all(name in name2idx for name in (na, nb, nc, nd)):
                continue
            i, j, k, l = (name2idx[na], name2idx[nb],
                         name2idx[nc], name2idx[nd])
            mask = _bfs_downstream(adj, k=k, j=j, n=n)
            # Invariants enforced by BFS construction; assert to catch
            # malformed inputs (a cycle through j-k closing the upstream
            # side onto the downstream side, e.g. PRO if mistakenly enabled).
            assert not mask[j], (
                f"BFS leaked through j={j} (axis upstream) for chi at "
                f"resid={res.idx} ({resname}) depth={depth}. Likely an "
                f"unexpected ring closure — refusing to build a corrupt mask.")
            assert mask[l], (
                f"BFS missed atom l={l} (downstream) for chi at "
                f"resid={res.idx} ({resname}) depth={depth}.")
            chi_idx_rows.append((i, j, k, l))
            chi_mask_rows.append(mask)
            chi_resid_rows.append(int(res.idx))
            chi_depth_rows.append(depth)

    if chi_idx_rows:
        chi_idx = np.asarray(chi_idx_rows, dtype=np.int32)
        chi_mask = np.stack(chi_mask_rows, axis=0)
        chi_resid = np.asarray(chi_resid_rows, dtype=np.int32)
        chi_depth = np.asarray(chi_depth_rows, dtype=np.int32)
    else:
        chi_idx = np.zeros((0, 4), dtype=np.int32)
        chi_mask = np.zeros((0, n), dtype=bool)
        chi_resid = np.zeros((0,), dtype=np.int32)
        chi_depth = np.zeros((0,), dtype=np.int32)

    return ChiTopology(
        chi_idx=chi_idx,
        chi_mask=chi_mask,
        chi_resid=chi_resid,
        chi_depth=chi_depth,
        n_atoms=n,
    )
