"""Bonded dihedral energy (§ 5.1 of design doc).

Three GROMACS dihedral forms are supported:

    type 1 / 9  (proper periodic)   :  E = k * (1 + cos(n * phi - phi0))
    type 4      (improper periodic) :  same form as proper
    type 3      (Ryckaert-Bellemans):  E = sum_n C_n * cos^n(phi - pi)
    type 2      (improper harmonic) :  E = 0.5 * k_xi * (xi - xi_eq)^2

Units: angles in radians, k_phi / C_n / k_xi in kJ/mol (already kJ/mol when read
from a GROMACS .top via parmed; AMBER .prmtop tops would need a kcal->kJ
conversion which we DO NOT do here -- this module assumes the .top file is
GROMACS-native).

Bond / bond-angle bonded terms are NOT included: per § 1.3, chi rotation
preserves all bond lengths and bond angles, so those terms are constant under
chi moves and can be dropped from the MC energy.

Per § 5.4 the F4 acceptance gate is "vs gmx single-point < 0.1 kJ/mol".  Since
gromacs is not part of the test rig, we validate against an independent numpy
reference instead (same physical formula, different implementation), plus the
physical-symmetry invariants (2π periodicity, sign symmetry around phi0).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import parmed as pmd

from ptmc.flexible.kinematics import measure_dihedrals

_DEG_TO_RAD = np.pi / 180.0


@dataclass(frozen=True)
class BondedParams:
    """Per-system bonded dihedral parameter tables.

    Stored as three independent ragged tables (one per form). All k values are
    in kJ/mol; all phases in radians.

    SHAPES:
        periodic_idx    (Mp, 4) int32
        periodic_phase  (Mp,)   float32 rad
        periodic_k      (Mp,)   float32 kJ/mol
        periodic_per    (Mp,)   int32   multiplicity
        rb_idx          (Mr, 4) int32
        rb_c            (Mr, 6) float32 C0..C5 in kJ/mol
        harmonic_idx    (Mh, 4) int32
        harmonic_phase  (Mh,)   float32 rad
        harmonic_k      (Mh,)   float32 kJ/mol/rad^2
        fudge_lj, fudge_qq  : 1-4 scaling factors (for F5)
    """
    periodic_idx: np.ndarray
    periodic_phase: np.ndarray
    periodic_k: np.ndarray
    periodic_per: np.ndarray
    rb_idx: np.ndarray
    rb_c: np.ndarray
    harmonic_idx: np.ndarray
    harmonic_phase: np.ndarray
    harmonic_k: np.ndarray
    fudge_lj: float
    fudge_qq: float

    @property
    def m_periodic(self) -> int:
        return int(self.periodic_idx.shape[0])

    @property
    def m_rb(self) -> int:
        return int(self.rb_idx.shape[0])

    @property
    def m_harmonic(self) -> int:
        return int(self.harmonic_idx.shape[0])

    def __post_init__(self) -> None:
        assert self.periodic_idx.shape[1] == 4
        assert self.periodic_phase.shape == (self.m_periodic,)
        assert self.periodic_k.shape == (self.m_periodic,)
        assert self.periodic_per.shape == (self.m_periodic,)
        assert self.rb_idx.shape[1] == 4
        assert self.rb_c.shape == (self.m_rb, 6)
        assert self.harmonic_idx.shape[1] == 4
        assert self.harmonic_phase.shape == (self.m_harmonic,)
        assert self.harmonic_k.shape == (self.m_harmonic,)


def _read_periodic_one(dih) -> list[tuple]:
    """Pull (phi_k_kJ, per, phase_rad) from a parmed Dihedral, expanding
    DihedralTypeList into individual terms."""
    out = []
    t = dih.type
    if hasattr(t, "__len__") and not hasattr(t, "phi_k"):
        # DihedralTypeList -> iterable of DihedralType
        for term in t:
            out.append((float(term.phi_k), int(term.per),
                        float(term.phase) * _DEG_TO_RAD))
    else:
        # Single DihedralType
        out.append((float(t.phi_k), int(t.per),
                    float(t.phase) * _DEG_TO_RAD))
    return out


def build_bonded_params(top_path: str) -> BondedParams:
    """Read a parametrized GROMACS topology and build the bonded parameter
    tables. Periodic propers + impropers are merged into ``periodic_*`` (same
    energy form); RB stored in ``rb_*``; harmonic impropers in ``harmonic_*``.
    """
    p = Path(top_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"GROMACS topology not found: {top_path!s} "
            f"(resolved to {p.resolve()!s}).")
    top = pmd.load_file(str(p), parametrize=True)

    periodic_idx: list[tuple[int, int, int, int]] = []
    periodic_phase: list[float] = []
    periodic_k: list[float] = []
    periodic_per: list[int] = []
    harmonic_idx: list[tuple[int, int, int, int]] = []
    harmonic_phase: list[float] = []
    harmonic_k: list[float] = []

    for d in top.dihedrals:
        quad = (d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx)
        for k_phi, per, phase in _read_periodic_one(d):
            # parmed sometimes emits zero-amplitude placeholder terms
            # (per=0, phi_k=0) for atom-type quads with no explicit
            # parameter entry. They contribute identically zero — drop
            # them to keep the table tight.
            if k_phi == 0.0:
                continue
            periodic_idx.append(quad)
            periodic_phase.append(phase)
            periodic_k.append(k_phi)
            periodic_per.append(per)

    # Harmonic impropers (type 2). Parmed exposes these in top.impropers with
    # ImproperType (psi_k, psi_eq). 1UBQ has 0.
    for imp in getattr(top, "impropers", []):
        quad = (imp.atom1.idx, imp.atom2.idx, imp.atom3.idx, imp.atom4.idx)
        harmonic_idx.append(quad)
        harmonic_phase.append(float(imp.type.psi_eq) * _DEG_TO_RAD)
        harmonic_k.append(float(imp.type.psi_k))

    # Ryckaert-Bellemans (type 3). 1UBQ has 0.
    rb_idx: list[tuple[int, int, int, int]] = []
    rb_c: list[list[float]] = []
    for rb in getattr(top, "rb_torsions", []):
        quad = (rb.atom1.idx, rb.atom2.idx, rb.atom3.idx, rb.atom4.idx)
        rb_idx.append(quad)
        rb_c.append([float(rb.type.c0), float(rb.type.c1), float(rb.type.c2),
                     float(rb.type.c3), float(rb.type.c4), float(rb.type.c5)])

    fudge_lj = float(getattr(top.defaults, "fudgeLJ", 0.5))
    fudge_qq = float(getattr(top.defaults, "fudgeQQ", 0.833333))

    return BondedParams(
        periodic_idx=np.asarray(periodic_idx or [], dtype=np.int32
                                ).reshape(-1, 4),
        periodic_phase=np.asarray(periodic_phase or [], dtype=np.float32),
        periodic_k=np.asarray(periodic_k or [], dtype=np.float32),
        periodic_per=np.asarray(periodic_per or [], dtype=np.int32),
        rb_idx=np.asarray(rb_idx or [], dtype=np.int32).reshape(-1, 4),
        rb_c=np.asarray(rb_c or [], dtype=np.float32).reshape(-1, 6),
        harmonic_idx=np.asarray(harmonic_idx or [], dtype=np.int32
                                ).reshape(-1, 4),
        harmonic_phase=np.asarray(harmonic_phase or [], dtype=np.float32),
        harmonic_k=np.asarray(harmonic_k or [], dtype=np.float32),
        fudge_lj=fudge_lj,
        fudge_qq=fudge_qq,
    )


# ---------------------------------------------------------------------------
# JAX kernel
# ---------------------------------------------------------------------------

def _periodic_energy(pos: jnp.ndarray, idx: jnp.ndarray,
                     phase: jnp.ndarray, k: jnp.ndarray,
                     per: jnp.ndarray) -> jnp.ndarray:
    """E = sum_m k_m * (1 + cos(n_m * phi_m - phi0_m))."""
    if idx.shape[0] == 0:
        return jnp.asarray(0.0, dtype=pos.dtype)
    phi = measure_dihedrals(pos, idx)             # (M,)
    return jnp.sum(k * (1.0 + jnp.cos(per * phi - phase)))


def _harmonic_energy(pos: jnp.ndarray, idx: jnp.ndarray,
                     phase: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    """E = sum_m 0.5 * k_m * wrap(phi_m - phi0_m)^2.

    The shortest signed distance to phi0 is used (the dihedral lives on the
    circle, so a naive (phi - phi0)^2 would mis-penalize when phi wraps).
    """
    if idx.shape[0] == 0:
        return jnp.asarray(0.0, dtype=pos.dtype)
    phi = measure_dihedrals(pos, idx)
    diff = (phi - phase + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    return jnp.sum(0.5 * k * diff * diff)


def _rb_energy(pos: jnp.ndarray, idx: jnp.ndarray,
               c: jnp.ndarray) -> jnp.ndarray:
    """E = sum_m sum_{n=0..5} C_{m,n} * cos^n(psi_m),   psi = phi - pi."""
    if idx.shape[0] == 0:
        return jnp.asarray(0.0, dtype=pos.dtype)
    phi = measure_dihedrals(pos, idx)               # (M,)
    psi = phi - jnp.pi
    cospsi = jnp.cos(psi)
    # Horner: ((((C5*c + C4)*c + C3)*c + C2)*c + C1)*c + C0
    e = c[:, 5]
    for j in (4, 3, 2, 1, 0):
        e = e * cospsi + c[:, j]
    return jnp.sum(e)


def E_bonded(pos: jnp.ndarray, params: BondedParams) -> jnp.ndarray:
    """Total bonded-dihedral energy in kJ/mol.

    pos : (N, 3); params : prebuilt BondedParams.
    Returns scalar (same dtype as pos).
    """
    e_per = _periodic_energy(
        pos,
        jnp.asarray(params.periodic_idx),
        jnp.asarray(params.periodic_phase),
        jnp.asarray(params.periodic_k),
        jnp.asarray(params.periodic_per),
    )
    e_harm = _harmonic_energy(
        pos,
        jnp.asarray(params.harmonic_idx),
        jnp.asarray(params.harmonic_phase),
        jnp.asarray(params.harmonic_k),
    )
    e_rb = _rb_energy(
        pos,
        jnp.asarray(params.rb_idx),
        jnp.asarray(params.rb_c),
    )
    return e_per + e_harm + e_rb


# ---------------------------------------------------------------------------
# Numpy reference (for F4 cross-validation)
# ---------------------------------------------------------------------------

def _measure_dihedrals_np(pos: np.ndarray, idx: np.ndarray) -> np.ndarray:
    i = idx[:, 0]
    j = idx[:, 1]
    k = idx[:, 2]
    l = idx[:, 3]
    b1 = pos[j] - pos[i]
    b2 = pos[k] - pos[j]
    b3 = pos[l] - pos[k]
    b2n = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-30)
    v = b1 - (b1 * b2n).sum(-1, keepdims=True) * b2n
    w = b3 - (b3 * b2n).sum(-1, keepdims=True) * b2n
    x = (v * w).sum(-1)
    y = (np.cross(b2n, v) * w).sum(-1)
    return np.arctan2(y, x)


def E_bonded_numpy(pos: np.ndarray, params: BondedParams) -> float:
    """Reference numpy implementation for F4 cross-validation."""
    total = 0.0
    if params.m_periodic:
        phi = _measure_dihedrals_np(pos, params.periodic_idx)
        total += float(np.sum(params.periodic_k * (
            1.0 + np.cos(params.periodic_per * phi - params.periodic_phase))))
    if params.m_harmonic:
        phi = _measure_dihedrals_np(pos, params.harmonic_idx)
        diff = (phi - params.harmonic_phase + np.pi) % (2.0 * np.pi) - np.pi
        total += float(np.sum(0.5 * params.harmonic_k * diff * diff))
    if params.m_rb:
        phi = _measure_dihedrals_np(pos, params.rb_idx)
        psi = phi - np.pi
        cospsi = np.cos(psi)
        e = params.rb_c[:, 5].copy()
        for jj in (4, 3, 2, 1, 0):
            e = e * cospsi + params.rb_c[:, jj]
        total += float(np.sum(e))
    return total
