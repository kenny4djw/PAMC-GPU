"""F1 — chi lookup table + chi_idx/chi_mask BFS construction.

Static-table sanity checks plus end-to-end build on the 1UBQ topology.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ptmc.flexible import (
    CHI_TABLE,
    FLEXIBLE_RESIDUES,
    N_CHI,
    build_chi_topology,
    chi_atom_names,
    n_chi_for,
)

DATA = Path(__file__).resolve().parent.parent / "data"
UBQ_TOP = DATA / "1UBQ.top"


# ---------------------------------------------------------------------------
# Static CHI_TABLE invariants
# ---------------------------------------------------------------------------

EXPECTED_N_CHI = {
    "ALA": 0, "GLY": 0, "PRO": 0,
    "SER": 1, "THR": 1, "CYS": 1, "VAL": 1,
    "ASP": 2, "ASN": 2, "HIS": 2, "PHE": 2, "TYR": 2,
    "TRP": 2, "LEU": 2, "ILE": 2,
    "MET": 3, "GLU": 3, "GLN": 3, "LYS": 3,
    "ARG": 4,
}


def test_n_chi_matches_plan():
    """§ 1.2 of the design doc."""
    for res, n in EXPECTED_N_CHI.items():
        assert N_CHI[res] == n, f"{res}: expected n_chi={n}, got {N_CHI[res]}"


def test_chi_table_covers_20_canonical_aas():
    canonical = set(EXPECTED_N_CHI)
    assert canonical.issubset(CHI_TABLE.keys())


def test_flexible_residues_excludes_rigid():
    rigid = {"ALA", "GLY", "PRO"}
    assert rigid.isdisjoint(FLEXIBLE_RESIDUES)
    for res in EXPECTED_N_CHI:
        if EXPECTED_N_CHI[res] > 0:
            assert res in FLEXIBLE_RESIDUES


def test_chi_tuples_have_four_distinct_atoms():
    for res, defs in CHI_TABLE.items():
        for k, tup in enumerate(defs):
            assert len(tup) == 4, f"{res} chi{k+1}: expected 4 atoms"
            assert len(set(tup)) == 4, (
                f"{res} chi{k+1}: duplicate atom names in {tup}")


def test_chi_tuples_share_central_bond():
    """Successive chis share the central two atoms (j_k = k_{k-1}, k_k = l_{k-1}).

    This is the IUPAC chain structure: chi_n rotates the bond shifted one
    along the backbone-to-tip direction.
    """
    for res, defs in CHI_TABLE.items():
        for d in range(1, len(defs)):
            prev = defs[d - 1]
            cur = defs[d]
            # cur (i,j,k,l) — its (i,j) should be prev's (j,k)
            assert cur[0] == prev[1] and cur[1] == prev[2], (
                f"{res}: chi{d+1}={cur} does not chain off chi{d}={prev}")


def test_his_aliases_map_to_his():
    for alias in ("HID", "HIE", "HIP", "HSD", "HSE", "HSP"):
        assert chi_atom_names(alias) == chi_atom_names("HIS")
        assert n_chi_for(alias) == 2


def test_unknown_resname_is_rigid():
    assert chi_atom_names("LIG") == ()
    assert n_chi_for("XYZ") == 0


# ---------------------------------------------------------------------------
# 1UBQ end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ubq_pmd():
    """Cached parmed Structure. parmed 4.3.1 has stateful internals — reloading
    the same .top in a single process triggers TypeError in Atom.sigma. Load
    once and share."""
    if not UBQ_TOP.is_file():
        pytest.skip(f"missing fixture: {UBQ_TOP}")
    import parmed as pmd
    return pmd.load_file(str(UBQ_TOP), parametrize=True)


@pytest.fixture(scope="module")
def ubq_topo():
    if not UBQ_TOP.is_file():
        pytest.skip(f"missing fixture: {UBQ_TOP}")
    return build_chi_topology(str(UBQ_TOP))


def test_ubq_shapes(ubq_topo):
    K = ubq_topo.k
    N = ubq_topo.n_atoms
    assert K > 0
    assert ubq_topo.chi_idx.shape == (K, 4)
    assert ubq_topo.chi_mask.shape == (K, N)
    assert ubq_topo.chi_resid.shape == (K,)
    assert ubq_topo.chi_depth.shape == (K,)


def test_ubq_chi_indices_in_range(ubq_topo):
    N = ubq_topo.n_atoms
    assert ubq_topo.chi_idx.min() >= 0
    assert ubq_topo.chi_idx.max() < N


def test_ubq_chi_atoms_distinct_per_row(ubq_topo):
    for k, row in enumerate(ubq_topo.chi_idx):
        assert len(set(row.tolist())) == 4, (
            f"chi {k}: duplicate atom indices {row}")


def test_ubq_mask_excludes_j(ubq_topo):
    """The j atom (axis-upstream end) must NOT be downstream."""
    j_idx = ubq_topo.chi_idx[:, 1]
    for k in range(ubq_topo.k):
        assert not ubq_topo.chi_mask[k, j_idx[k]], (
            f"chi {k}: mask includes j={j_idx[k]} (axis upstream)")


def test_ubq_mask_includes_l(ubq_topo):
    """The l atom (downstream tip) must BE downstream."""
    l_idx = ubq_topo.chi_idx[:, 3]
    for k in range(ubq_topo.k):
        assert ubq_topo.chi_mask[k, l_idx[k]], (
            f"chi {k}: mask missing l={l_idx[k]} (downstream tip)")


def test_ubq_mask_includes_k_pivot(ubq_topo):
    """k (pivot) is included; rotation about k leaves k fixed (identity)."""
    k_idx = ubq_topo.chi_idx[:, 2]
    for kk in range(ubq_topo.k):
        assert ubq_topo.chi_mask[kk, k_idx[kk]]


def test_ubq_mask_excludes_i_upstream(ubq_topo):
    """i (chi-defining atom on upstream side) must NOT be downstream."""
    i_idx = ubq_topo.chi_idx[:, 0]
    for k in range(ubq_topo.k):
        assert not ubq_topo.chi_mask[k, i_idx[k]], (
            f"chi {k}: mask includes upstream-defining atom i={i_idx[k]}")


def test_ubq_chi_depth_starts_at_zero_per_residue(ubq_topo):
    """Within each residue's chi run, depth = 0, 1, 2, ..."""
    seen_resid = -1
    expected_depth = 0
    for resid, depth in zip(ubq_topo.chi_resid, ubq_topo.chi_depth):
        if int(resid) != seen_resid:
            assert depth == 0, (
                f"first chi of residue {resid} has depth {depth} != 0")
            seen_resid = int(resid)
            expected_depth = 1
        else:
            assert depth == expected_depth
            expected_depth += 1


def test_ubq_k_matches_residue_n_chi_sum(ubq_topo, ubq_pmd):
    """K equals sum of n_chi(resname) over residues that have all needed atoms.

    All standard residues in 1UBQ should have complete heavy-atom sidechains,
    so K = sum of n_chi over residues.
    """
    top = ubq_pmd
    expected = 0
    for res in top.residues:
        resname = res.name.strip().upper()
        defs = chi_atom_names(resname)
        if not defs:
            continue
        name2idx = {a.name.strip(): a.idx for a in res.atoms}
        for tup in defs:
            if all(n in name2idx for n in tup):
                expected += 1
    assert ubq_topo.k == expected


def test_ubq_pro_residues_contribute_zero_chi(ubq_topo, ubq_pmd):
    """PRO is rigid (n_chi=0); no chi row should reference a PRO residue."""
    top = ubq_pmd
    pro_resids = {res.idx for res in top.residues
                  if res.name.strip().upper() == "PRO"}
    if not pro_resids:
        pytest.skip("1UBQ has no PRO — vacuously satisfied")
    assert pro_resids.isdisjoint(set(ubq_topo.chi_resid.tolist()))


# ---------------------------------------------------------------------------
# Spot-check: MET at residue 0 in 1UBQ (first atom is N MET A 1)
# Verify the three chis have the documented downstream sets.
# ---------------------------------------------------------------------------

def test_ubq_met1_chi_downstream(ubq_topo, ubq_pmd):
    """MET residue 0 has 3 chis. The chi tuples are (i,j,k,l); axis = j-k,
    pivot = k. BFS-downstream from k while blocking j:

        chi1 (N, CA, CB, CG) axis CA-CB :
            downstream = {CB, HB1, HB2, CG, HG1, HG2, SD, CE, HE1, HE2, HE3}
            excludes   = {CA, HA, N, H*, C, O, ...}
        chi2 (CA, CB, CG, SD) axis CB-CG :
            downstream = {CG, HG1, HG2, SD, CE, HE1, HE2, HE3}
            excludes   = {CB, HB1, HB2, CA, ...}
        chi3 (CB, CG, SD, CE) axis CG-SD :
            downstream = {SD, CE, HE1, HE2, HE3}
            excludes   = {CG, HG1, HG2, CB, ...}
    """
    top = ubq_pmd
    met = top.residues[0]
    assert met.name.strip().upper() == "MET"

    name2idx = {a.name.strip(): a.idx for a in met.atoms}
    chi_rows_for_met = np.flatnonzero(ubq_topo.chi_resid == met.idx)
    assert chi_rows_for_met.size == 3

    # chi1: axis CA-CB
    chi1_row = chi_rows_for_met[0]
    mask1 = ubq_topo.chi_mask[chi1_row]
    for n in ("CB", "HB1", "HB2", "CG", "HG1", "HG2", "SD", "CE",
              "HE1", "HE2", "HE3"):
        if n in name2idx:
            assert mask1[name2idx[n]], f"chi1 missing {n}"
    for n in ("CA", "HA", "N", "H1", "H2", "H3", "C", "O"):
        if n in name2idx:
            assert not mask1[name2idx[n]], f"chi1 wrongly includes {n}"

    # chi2: axis CB-CG
    chi2_row = chi_rows_for_met[1]
    mask2 = ubq_topo.chi_mask[chi2_row]
    for n in ("CG", "HG1", "HG2", "SD", "CE", "HE1", "HE2", "HE3"):
        if n in name2idx:
            assert mask2[name2idx[n]], f"chi2 missing {n}"
    for n in ("CB", "HB1", "HB2", "CA", "HA", "N", "C", "O"):
        if n in name2idx:
            assert not mask2[name2idx[n]], f"chi2 wrongly includes {n}"

    # chi3: axis CG-SD
    chi3_row = chi_rows_for_met[2]
    mask3 = ubq_topo.chi_mask[chi3_row]
    for n in ("SD", "CE", "HE1", "HE2", "HE3"):
        if n in name2idx:
            assert mask3[name2idx[n]], f"chi3 missing {n}"
    for n in ("CG", "HG1", "HG2", "CB", "HB1", "HB2", "CA"):
        if n in name2idx:
            assert not mask3[name2idx[n]], f"chi3 wrongly includes {n}"
