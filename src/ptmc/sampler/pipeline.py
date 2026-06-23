"""End-to-end pipeline: I/O → surface setup → sampler → trajectory → results.

Returns (out_dict, betas_list) so the caller (run.py) handles summarization
without creating a circular import.

Supported combinations
----------------------
sampler  surface_type                                  runner
pa       continuum / patterned / crystal / full_atom   run_pa  (generic energy_fn closure)
pt       continuum / patterned / crystal / full_atom   run_pt  (generic energy_fn closure)

Agarose / agarose_surface are NOT supported by PA / PT; they were previously
reachable only through the HT batch path which has been removed. The
high-throughput vmap kernels still live in ``ptmc.sampler.highthroughput``
for direct use by tests and benchmarks; they are no longer wired into the
user-facing CLI.

Flexible (chi-aware) protein DOF
--------------------------------
``config.mc.flexible = True`` wires in chi-aware MC kernels, combined
energy closures, and chi-aware PA / PT drivers. The integration was
completed in the "flexible CLI" sprint.

Trajectory (--save-traj)
------------------------
Runs one additional chain with collect_traj=True using the generic
scan_metropolis kernel and writes PDB + XTC via write_trajectory.
Works for any surface type that has an energy_fn closure.
"""
from __future__ import annotations

import numpy as np

from ptmc.config import SimConfig, beta as beta_of
from ptmc.io.parse_pdb import parse_pdb
from ptmc.io.parse_topology import parse_topology, build_atoms
from ptmc.log import logger


def _safe_init_z(atoms, z_min, sigma_trans, user_init_z=None):
    """Compute a safe initial z-height so no atom penetrates the hard wall.

    The protein ``pos0`` is centred internally to obtain the true geometric
    radius, yielding a conservative ``init_z`` valid for any random
    orientation.  The margin ``3 * sigma_trans`` ensures a single upward
    translation proposal cannot land below the wall.
    """
    pos_centered = atoms.pos0 - atoms.pos0.mean(axis=0)
    R_max = float(np.linalg.norm(pos_centered, axis=1).max())
    init_z_safe = R_max + z_min + 3.0 * float(sigma_trans)
    if user_init_z is not None and user_init_z > 0:
        if user_init_z < init_z_safe:
            logger.warning(
                "User-supplied init_z = %.3f nm may be too small; "
                "safe minimum ≈ %.3f nm (R_max=%.3f + z_min=%.3f + 3σ=%.3f). "
                "Expect inf energies for some initial orientations.",
                user_init_z, init_z_safe, R_max, z_min, 3.0 * float(sigma_trans))
        return max(user_init_z, init_z_safe)
    return init_z_safe


def _active_z_min(config: SimConfig) -> float:
    """Return the hard-wall z_min that matches ``config.surface_type``.

    Each surface model has its own hard-wall parameter; the legacy
    ``config.surface.z_min`` only applies to the homogeneous continuum
    and patterned models. Crystal / full-atom / agarose-surface paths
    store z_min on their own sub-config. The 3D agarose gel has no hard
    wall and falls back to a small positive value so init_z still has a
    sensible lower bound.
    """
    st = config.surface_type
    if st in ("continuum", "patterned"):
        return float(config.surface.z_min)
    if st == "crystal":
        return float(config.crystal.z_min)
    if st == "full_atom":
        return float(config.full_atom.z_min)
    if st == "agarose_surface":
        return float(config.agarose.z_min)
    if st == "agarose":
        # 3D gel has no hard wall; use 0 so _safe_init_z falls back to
        # R_max + 3 sigma_trans (still a safe lower bound).
        return 0.0
    return float(config.surface.z_min)


def _validate_energies(out: dict, label: str = "sampler") -> tuple[int, int]:
    """Audit sampler output for non-finite energies.

    Returns ``(n_inf, n_nan)`` counts so the caller can record per-system
    health flags. Logs at warning level for partial corruption. Raises
    ``RuntimeError`` when EVERY energy is non-finite — in that regime the
    downstream orientation summary is completely unreliable, so failing
    loudly is more honest than writing a parquet of garbage.
    """
    e = np.asarray(out.get("energies", np.zeros(0)))
    n_inf = int(np.sum(np.isinf(e)))
    n_nan = int(np.sum(np.isnan(e)))
    if n_inf == 0 and n_nan == 0:
        return 0, 0
    logger.warning(
        "%s output contains non-finite energies: %d inf, %d NaN (%.2f%% of total). "
        "Results may be unreliable. Consider increasing z_min or init_z.",
        label, n_inf, n_nan,
        100.0 * (n_inf + n_nan) / max(e.size, 1),
    )
    if (n_inf + n_nan) == e.size and e.size > 0:
        raise RuntimeError(
            f"{label}: ALL {e.size} energy entries are non-finite "
            f"({n_inf} inf, {n_nan} NaN). Every chain hit the hard wall or "
            f"diverged. Refusing to write a parquet summary built on garbage. "
            f"Increase init_z (currently too close to z_min) and re-run.")
    return n_inf, n_nan


# ---------------------------------------------------------------------------
# Energy-function closures (JAX, one-pose scalar → used by pt/pa and traj)
# ---------------------------------------------------------------------------

def _efn_continuum(atoms, surf):
    """JAX closure for homogeneous Steele 9-3 + screened Coulomb.

    Honours ``surf.eps_surf`` / ``surf.sigma_surf`` (ε-σ Lorentz-Berthelot path)
    when both are set; otherwise the geometric C6/C12 surface params. The same
    ``steele_energy_jax`` kernel evaluates both via effective coefficients.
    """
    import jax.numpy as jnp
    from ptmc.energy.reference import steele_coefficients_for, steele_energy_jax

    c6p, c12p, cA, cB = steele_coefficients_for(
        atoms.c6, atoms.c12, surf.c6_surf, surf.c12_surf,
        surf.eps_surf, surf.sigma_surf, surf.rho_s)
    pos0 = jnp.asarray(atoms.pos0, dtype=jnp.float32)
    q    = jnp.asarray(atoms.q,    dtype=jnp.float32)
    c6p  = jnp.asarray(c6p,        dtype=jnp.float32)
    c12p = jnp.asarray(c12p,       dtype=jnp.float32)
    cA = float(cA); cB = float(cB)
    lamD = float(surf.lambda_D); z_min = float(surf.z_min); psi0 = float(surf.psi0)

    def energy_fn(quat, trans):
        return steele_energy_jax(quat, trans, pos0, q, c6p, c12p,
                                  cA, cB, lamD, z_min, psi0)
    return energy_fn


def _efn_patterned(atoms, surf, phi_field, grid_origin, grid_spacing):
    """JAX closure for Steele 9-3 vdW + grid-interpolated phi(x,y,z).

    vdW uses the continuum params, including the ε-σ Lorentz-Berthelot path when
    ``surf.eps_surf`` / ``surf.sigma_surf`` are set (same rule as continuum).
    """
    import jax.numpy as jnp
    from ptmc.energy.reference import steele_coefficients_for
    from ptmc.energy.grid_energy import patterned_energy_jax

    c6p, c12p, cA, cB = steele_coefficients_for(
        atoms.c6, atoms.c12, surf.c6_surf, surf.c12_surf,
        surf.eps_surf, surf.sigma_surf, surf.rho_s)
    pos0 = jnp.asarray(atoms.pos0,   dtype=jnp.float32)
    q    = jnp.asarray(atoms.q,      dtype=jnp.float32)
    c6p  = jnp.asarray(c6p,          dtype=jnp.float32)
    c12p = jnp.asarray(c12p,         dtype=jnp.float32)
    phi  = jnp.asarray(phi_field,    dtype=jnp.float32)
    org  = jnp.asarray(grid_origin,  dtype=jnp.float32)
    spc  = jnp.asarray(grid_spacing, dtype=jnp.float32)
    cA = float(cA); cB = float(cB)
    lamD = float(surf.lambda_D); z_min = float(surf.z_min)

    def energy_fn(quat, trans):
        return patterned_energy_jax(quat, trans, pos0, q, c6p, c12p,
                                     cA, cB, lamD, z_min, phi, org, spc)
    return energy_fn


def _efn_agarose(atoms, U_field, phi_field, origin, spacing):
    """JAX closure for 3D agarose gel (Gaussian steric + Yukawa, no hard wall)."""
    import jax.numpy as jnp
    from ptmc.energy.grid_energy import agarose_energy_jax

    pos0 = jnp.asarray(atoms.pos0, dtype=jnp.float32)
    q    = jnp.asarray(atoms.q,    dtype=jnp.float32)
    U    = jnp.asarray(U_field,    dtype=jnp.float32)
    phi  = jnp.asarray(phi_field,  dtype=jnp.float32)
    org  = jnp.asarray(origin,     dtype=jnp.float32)
    spc  = jnp.asarray(spacing,    dtype=jnp.float32)

    def energy_fn(quat, trans):
        return agarose_energy_jax(quat, trans, pos0, q, U, phi, org, spc)
    return energy_fn


def _efn_grid(atoms, grids, cell_Lx: float = 0.0, cell_Ly: float = 0.0):
    """JAX closure for full-atom surface energy via (G12, G6, phi) grids."""
    import jax.numpy as jnp
    from ptmc.energy.grid_energy import grid_energy_jax

    pos0    = jnp.asarray(atoms.pos0, dtype=jnp.float32)
    sc12    = jnp.asarray(np.sqrt(atoms.c12), dtype=jnp.float32)
    sc6     = jnp.asarray(np.sqrt(atoms.c6),  dtype=jnp.float32)
    q       = jnp.asarray(atoms.q,     dtype=jnp.float32)
    G12     = jnp.asarray(grids.G12,   dtype=jnp.float32)
    G6      = jnp.asarray(grids.G6,    dtype=jnp.float32)
    phi     = jnp.asarray(grids.phi,   dtype=jnp.float32)
    origin  = jnp.asarray(grids.origin,  dtype=jnp.float32)
    spacing = jnp.asarray(grids.spacing, dtype=jnp.float32)
    z_min   = float(grids.z_min)
    Lx, Ly  = float(cell_Lx), float(cell_Ly)

    def energy_fn(quat, trans):
        return grid_energy_jax(quat, trans, pos0, sc12, sc6, q,
                               G12, G6, phi, origin, spacing, z_min, Lx, Ly)
    return energy_fn


# ---------------------------------------------------------------------------
# Flexible (chi-aware) energy-function closures.
#
# Each factory builds a combined energy_fn(quat, trans, chi) → scalar that:
#   1. applies chi rotations to intrinsic (pos0) coords via apply_all_chi
#   2. applies rigid-body rotation + translation to get lab-frame positions
#   3. evaluates the surface energy via the _from_pos kernel
#   4. evaluates bonded dihedral energy E_bonded(chi) at the chi-rotated coords
#   5. evaluates intra-protein non-bonded energy E_intra_nb at those same coords
#   6. sums all three contributions
#
# All closures are JIT-compatible: only numpy arrays are captured by the
# closure; the call path is pure JAX.
# ---------------------------------------------------------------------------


def _make_flexible_closure(atoms, surface_efn_from_pos, chi_schedule,
                            bonded_params, intra_nb_params):
    """Build a chi-aware combined energy closure from pre-built surface/intra components.

    Parameters
    ----------
    atoms : Atoms
        Protein atoms (provides pos0, q, c6, c12).
    surface_efn_from_pos : callable (pos) -> scalar
        Surface energy kernel that accepts lab-frame positions.
    chi_schedule : ChiSchedule
        Depth-sorted chi kinematics schedule.
    bonded_params : BondedParams
        Bonded dihedral parameters (for E_bonded).
    intra_nb_params : IntraNBParams
        Intra-protein non-bonded parameters (for E_intra_nb).

    Returns
    -------
    energy_fn : callable (quat, trans, chi) -> scalar
    """
    import jax.numpy as jnp
    from ptmc.mc.moves import quat_rotate
    from ptmc.flexible.kinematics import apply_all_chi
    from ptmc.flexible.bonded import E_bonded
    from ptmc.flexible.intra_nb import E_intra_nb

    pos0_j = jnp.asarray(atoms.pos0, dtype=jnp.float32)

    def energy_fn(quat, trans, chi):
        # 1. Apply chi rotations to intrinsic coords
        pos_chi = apply_all_chi(pos0_j, chi, chi_schedule)
        # 2. Apply rigid-body transform to get lab-frame positions
        pos_lab = quat_rotate(quat, pos_chi) + trans
        # 3. Surface energy at lab positions
        e_surf = surface_efn_from_pos(pos_lab)
        # 4. Bonded dihedral energy at chi-rotated (body-frame) coords
        e_bond = E_bonded(pos_chi, bonded_params)
        # 5. Intra-protein non-bonded energy (excludes 1-2,1-3; scales 1-4)
        e_intra = E_intra_nb(pos_chi, intra_nb_params)
        return e_surf + e_bond + e_intra

    return energy_fn


# ---------------------------------------------------------------------------
# Surface-type-specific flexible energy closures.
# Each returns an energy_fn(quat, trans, chi) → scalar.
# ---------------------------------------------------------------------------

def _efn_continuum_flexible(atoms, surf, chi_schedule,
                               bonded_params, intra_nb_params):
    """Flexible closure for homogeneous Steele 9-3 + screened Coulomb."""
    import jax.numpy as jnp
    from ptmc.energy.reference import (
        steele_coefficients_for, steele_energy_from_pos_jax,
    )

    c6p, c12p, cA, cB = steele_coefficients_for(
        atoms.c6, atoms.c12, surf.c6_surf, surf.c12_surf,
        surf.eps_surf, surf.sigma_surf, surf.rho_s)
    q_j    = jnp.asarray(atoms.q,    dtype=jnp.float32)
    c6p_j  = jnp.asarray(c6p,        dtype=jnp.float32)
    c12p_j = jnp.asarray(c12p,       dtype=jnp.float32)
    cA_f = float(cA); cB_f = float(cB)
    lamD = float(surf.lambda_D); z_min_f = float(surf.z_min); psi0_f = float(surf.psi0)

    def surface_from_pos(pos):
        return steele_energy_from_pos_jax(pos, q_j, c6p_j, c12p_j,
                                           cA_f, cB_f, lamD, z_min_f, psi0_f)

    return _make_flexible_closure(atoms, surface_from_pos, chi_schedule,
                                   bonded_params, intra_nb_params)


def _efn_patterned_flexible(atoms, surf, phi_field, grid_origin, grid_spacing,
                               chi_schedule, bonded_params, intra_nb_params):
    """Flexible closure for Steele 9-3 vdW + grid-interpolated phi."""
    import jax.numpy as jnp
    from ptmc.energy.reference import steele_coefficients_for
    from ptmc.energy.grid_energy import patterned_energy_from_pos_jax

    c6p, c12p, cA, cB = steele_coefficients_for(
        atoms.c6, atoms.c12, surf.c6_surf, surf.c12_surf,
        surf.eps_surf, surf.sigma_surf, surf.rho_s)
    q_j    = jnp.asarray(atoms.q,      dtype=jnp.float32)
    c6p_j  = jnp.asarray(c6p,          dtype=jnp.float32)
    c12p_j = jnp.asarray(c12p,         dtype=jnp.float32)
    phi_j  = jnp.asarray(phi_field,    dtype=jnp.float32)
    org_j  = jnp.asarray(grid_origin,  dtype=jnp.float32)
    spc_j  = jnp.asarray(grid_spacing, dtype=jnp.float32)
    cA_f = float(cA); cB_f = float(cB)
    lamD_f = float(surf.lambda_D); z_min_f = float(surf.z_min)

    def surface_from_pos(pos):
        return patterned_energy_from_pos_jax(pos, q_j, c6p_j, c12p_j,
                                               cA_f, cB_f, lamD_f, z_min_f,
                                               phi_j, org_j, spc_j)

    return _make_flexible_closure(atoms, surface_from_pos, chi_schedule,
                                   bonded_params, intra_nb_params)


def _efn_grid_flexible(atoms, grids, cell_Lx, cell_Ly,
                         chi_schedule, bonded_params, intra_nb_params):
    """Flexible closure for full-atom/crystal surface energy via (G12, G6, phi) grids."""
    import jax.numpy as jnp
    from ptmc.energy.grid_energy import grid_energy_from_pos_jax

    sc12_j   = jnp.asarray(np.sqrt(atoms.c12), dtype=jnp.float32)
    sc6_j    = jnp.asarray(np.sqrt(atoms.c6),  dtype=jnp.float32)
    q_j      = jnp.asarray(atoms.q,     dtype=jnp.float32)
    G12_j    = jnp.asarray(grids.G12,   dtype=jnp.float32)
    G6_j     = jnp.asarray(grids.G6,    dtype=jnp.float32)
    phi_j    = jnp.asarray(grids.phi,   dtype=jnp.float32)
    origin_j = jnp.asarray(grids.origin,  dtype=jnp.float32)
    spc_j    = jnp.asarray(grids.spacing, dtype=jnp.float32)
    z_min_f  = float(grids.z_min)
    Lx_f, Ly_f = float(cell_Lx), float(cell_Ly)

    def surface_from_pos(pos):
        return grid_energy_from_pos_jax(pos, sc12_j, sc6_j, q_j,
                                         G12_j, G6_j, phi_j,
                                         origin_j, spc_j, z_min_f,
                                         Lx_f, Ly_f)

    return _make_flexible_closure(atoms, surface_from_pos, chi_schedule,
                                   bonded_params, intra_nb_params)


def _efn_agarose_surface(atoms, U_field, phi_field, origin, spacing, z_min):
    """JAX closure for flat gel-coated surface (hard wall at z_min)."""
    import jax.numpy as jnp
    from ptmc.energy.grid_energy import agarose_surface_energy_jax

    pos0  = jnp.asarray(atoms.pos0, dtype=jnp.float32)
    q     = jnp.asarray(atoms.q,    dtype=jnp.float32)
    U     = jnp.asarray(U_field,    dtype=jnp.float32)
    phi   = jnp.asarray(phi_field,  dtype=jnp.float32)
    org   = jnp.asarray(origin,     dtype=jnp.float32)
    spc   = jnp.asarray(spacing,    dtype=jnp.float32)
    z_min = float(z_min)

    def energy_fn(quat, trans):
        return agarose_surface_energy_jax(quat, trans, pos0, q, U, phi, org, spc, z_min)
    return energy_fn


# ---------------------------------------------------------------------------
# Trajectory recording
# ---------------------------------------------------------------------------

def _write_traj(config: SimConfig, atoms, energy_fn):
    """Run one chain with collect_traj=True and write PDB + XTC."""
    import jax
    import jax.numpy as jnp
    from ptmc.mc.metropolis import run_chain
    from ptmc.io.write_traj import write_trajectory

    mc = config.mc
    key = jax.random.PRNGKey(mc.seed ^ 0xDEADBEEF)
    qn = jax.random.normal(key, (4,))
    init_q = qn / jnp.linalg.norm(qn)
    init_z = _safe_init_z(atoms, _active_z_min(config), config.mc.sigma_trans,
                          config.mc.init_z or None)
    init_t = jnp.array([0.0, 0.0, float(init_z)])

    out = run_chain(
        key, init_q, init_t, energy_fn,
        float(beta_of(mc.temperature)),
        float(mc.sigma_rot), float(mc.sigma_trans),
        jnp.asarray(mc.axis_mask, dtype=jnp.float32),
        int(mc.n_steps),
    )
    quats  = np.asarray(out["quat"])   # (n_steps, 4)
    transs = np.asarray(out["trans"])  # (n_steps, 3)

    pdb_out = config.traj_prefix + ".pdb"
    xtc_out = config.traj_prefix + ".xtc"
    n = write_trajectory(pdb_out, xtc_out, atoms, quats, transs)
    logger.info("Trajectory: %d frames → %s, %s", n, pdb_out, xtc_out)


# ---------------------------------------------------------------------------
# PT runner
# ---------------------------------------------------------------------------

def _run_pt(config: SimConfig, atoms, energy_fn) -> tuple[dict, list]:
    import jax
    import jax.numpy as jnp
    from ptmc.sampler.parallel_tempering import run_pt
    from ptmc.config import INIT_QUAT_KEY_OFFSET

    pt = config.pt
    mc = config.mc

    T_ladder = np.geomspace(pt.T_min, pt.T_max, pt.n_replicas)
    betas    = np.array([beta_of(T) for T in T_ladder])
    beta_low = float(betas[np.argmax(betas)])   # beta at T_min (highest beta)

    key = jax.random.PRNGKey(mc.seed)
    init_key = jax.random.fold_in(key, 0)
    run_key = jax.random.fold_in(key, 1)
    R   = pt.n_replicas

    # Initial poses: S=1 system, R replicas, random quats. Use the same
    # fold_in(k, INIT_QUAT_KEY_OFFSET) convention as the high-throughput
    # runner and ``run_multi_pt`` so identical seeds yield identical initial
    # quaternions across drivers.
    init_keys = jax.random.split(init_key, R)
    qn = jax.vmap(lambda k: jax.random.normal(
        jax.random.fold_in(k, INIT_QUAT_KEY_OFFSET), (4,)))(init_keys)
    init_quats = (qn / jnp.linalg.norm(qn, axis=-1, keepdims=True))[None]  # (1,R,4)
    init_z = _safe_init_z(atoms, _active_z_min(config), config.mc.sigma_trans,
                          config.mc.init_z or None)
    init_transs = jnp.zeros((1, R, 3)).at[0, :, 2].set(float(init_z))       # (1,R,3)

    pt_out = run_pt(
        key=run_key,
        init_quats=init_quats,
        init_transs=init_transs,
        energy_fn=energy_fn,
        betas=betas,
        sigma_rot=mc.sigma_rot,
        sigma_trans=mc.sigma_trans,
        axis_mask=jnp.asarray(mc.axis_mask, dtype=jnp.float32),
        n_rounds=pt.n_rounds,
        n_sweep=pt.n_sweep,
    )

    logger.info("PT done. MC accept (per T): %s",
                np.array2string(np.asarray(pt_out["mc_accept_rate"]), precision=3))
    logger.info("PT swap accept (per gap): %s",
                np.array2string(np.asarray(pt_out["swap_accept_rate"]), precision=3))

    # Low-T trajectory: (n_rounds, 1, 4/3) → (1, n_rounds, 4/3)
    low_q = np.asarray(pt_out["low_T_quat"]).transpose(1, 0, 2)   # (1, n_rounds, 4)
    low_t = np.asarray(pt_out["low_T_trans"]).transpose(1, 0, 2)  # (1, n_rounds, 3)

    # Compute real energies for the low-T trajectory frames.
    pt_energies = np.asarray(
        jax.vmap(energy_fn)(
            jnp.asarray(low_q[0]), jnp.asarray(low_t[0]),
        )
    )
    out = {
        "quats":      low_q,                                    # (1, n_rounds, 4)
        "transs":     low_t,                                    # (1, n_rounds, 3)
        "energies":   pt_energies[None],                        # (1, n_rounds)
        "accept":     np.full((1, pt.n_rounds), np.nan),        # not tracked per frame
        "system_ids": [0],
        "_pt_raw":    pt_out,
    }
    return out, [beta_low]


# ---------------------------------------------------------------------------
# PA runner
# ---------------------------------------------------------------------------

def _pa_z_range(config: SimConfig, atoms, phi_data: dict | None = None) -> tuple[float, float]:
    """Resolve the (z_lo, z_hi) slab covered by the PA walker ensemble.

    z_lo is the smallest z at which the protein's COM cannot place ANY atom
    below the hard wall, regardless of orientation — i.e. the geometric
    safe boundary. z_hi is config-controlled (``pa.z_max``) or auto from
    the surface field's upper z limit.

    The slab [z_lo, z_hi] is what ``adsorption_free_energy`` integrates
    over to extract K_ads; choosing it too narrow underestimates K_ads
    (bulk baseline not reached), too wide just wastes population.
    """
    z_min = _active_z_min(config)
    pos_centered = atoms.pos0 - atoms.pos0.mean(axis=0)
    R_max = float(np.linalg.norm(pos_centered, axis=1).max())
    z_lo = z_min + R_max

    z_hi_user = float(config.pa.z_max)
    if z_hi_user > 0.0:
        z_hi = z_hi_user
    elif config.surface_type == "crystal":
        z_hi = float(config.crystal.z_grid_max)
    elif config.surface_type == "full_atom":
        z_hi = float(config.full_atom.z_grid_max)
    else:
        # continuum / patterned: surface energy decays exponentially in z;
        # 3 nm above the safe COM covers ~4·λ_D for the default 0.785 nm.
        z_hi = z_lo + 3.0
    if z_hi <= z_lo:
        raise ValueError(
            f"PA z slab is empty: z_lo={z_lo:.3f} ≥ z_hi={z_hi:.3f} nm. "
            f"Increase pa.z_max (or the surface field's z_grid_max for "
            f"crystal / full_atom).")
    return z_lo, z_hi


def _pa_lateral_area(config: SimConfig, phi_data: dict | None = None) -> float:
    """Lateral cell area (nm²) for the PA system. 0.0 if surface is treated
    as laterally infinite homogeneous (continuum / patterned) — the area
    factor cancels in the Z_ads / Z_bulk ratio in that case."""
    phi_data = phi_data or {}
    if config.surface_type in ("crystal", "full_atom"):
        return float(phi_data.get("_cell_Lx", 0.0)
                     * phi_data.get("_cell_Ly", 0.0))
    return 0.0


def _run_pa(config: SimConfig, atoms, energy_fn,
            phi_data: dict | None = None) -> tuple[dict, list]:
    import jax
    import jax.numpy as jnp
    from ptmc.sampler.population_annealing import run_pa
    from ptmc.analysis.adsorption import adsorption_free_energy, beta0_adequacy
    from ptmc.config import INIT_QUAT_KEY_OFFSET

    pa = config.pa
    mc = config.mc
    M  = pa.n_walkers

    key  = jax.random.PRNGKey(mc.seed)
    init_key = jax.random.fold_in(key, 0)
    run_key  = jax.random.fold_in(key, 1)
    # Same quat-init convention as HT / PT so identical seeds give identical
    # initial pose distributions across the three drivers.
    qn = jax.random.normal(jax.random.fold_in(init_key, INIT_QUAT_KEY_OFFSET),
                           (M, 4))
    init_quats  = qn / jnp.linalg.norm(qn, axis=-1, keepdims=True)

    # Distribute initial z UNIFORMLY over the slab [z_lo, z_hi] across walkers,
    # not all at one z. This is what makes PA's logZ_ratio interpretable as
    # log[Z(β_target) / (Δz · 8π²)] — the uniform-in-slab reference state.
    # Old behaviour (all walkers at one z) needed MCMC to diffuse them across
    # the slab, which requires ~ (Δz/σ_trans)² ≈ 3600 steps at default σ; the
    # 50-step default n_sweep was nowhere near enough, so Z(β₀) was *not* the
    # uniform prior and dG_box was biased by an unknown offset.
    z_lo, z_hi = _pa_z_range(config, atoms, phi_data)
    z_key = jax.random.fold_in(init_key, INIT_QUAT_KEY_OFFSET + 1)
    zs = jax.random.uniform(z_key, (M,), minval=z_lo, maxval=z_hi)
    init_transs = jnp.zeros((M, 3)).at[:, 2].set(zs)

    lateral_area = _pa_lateral_area(config, phi_data)
    logger.info(
        "PA initial slab: z ∈ [%.3f, %.3f] nm (Δz=%.3f), "
        "lateral area = %.3f nm² (0 = laterally infinite homogeneous), "
        "%d walkers uniformly distributed.",
        z_lo, z_hi, z_hi - z_lo, lateral_area, M)

    pa_out = run_pa(
        key=run_key,
        init_quats=init_quats,
        init_transs=init_transs,
        energy_fn=energy_fn,
        beta0=beta_of(pa.T_start),
        beta_target=beta_of(mc.temperature),
        n_sweep=mc.n_steps,
        sigma_rot=mc.sigma_rot,
        sigma_trans=mc.sigma_trans,
        axis_mask=jnp.asarray(mc.axis_mask, dtype=jnp.float32),
        target_ess=pa.target_ess,
        max_steps=pa.max_annealing_steps,
    )

    logger.info("PA done. logZ_ratio = %.4f, cooling steps = %d, converged = %s",
                pa_out["logZ_ratio"], pa_out["n_steps"], pa_out["converged"])

    # β₀ adequacy: did the initial walker ensemble actually sample the
    # uniform-in-slab prior? Use the energies BEFORE the first sweep —
    # those are what define Z(β₀).
    init_E = jax.vmap(energy_fn)(init_quats, init_transs)
    beta0_check = beta0_adequacy(np.asarray(init_E), beta_of(pa.T_start))
    log_fn = logger.info if beta0_check["ok"] else logger.warning
    log_fn("PA β₀-adequacy: %s (mean |β₀E|=%.3f, max=%.3f).",
           beta0_check["msg"], beta0_check["mean"], beta0_check["max"])

    # Drift check: how many final walkers are outside the slab? At β_target
    # most weight should sit near z_lo (bound state); walkers far from the
    # slab indicate either no binding or a slab too narrow. Drift past z_hi
    # means K_ads's bulk-baseline assumption may not hold.
    ft = np.asarray(pa_out["final_transs"])  # (M, 3)
    z_final = ft[:, 2]
    n_above = int(np.sum(z_final > z_hi))
    n_below = int(np.sum(z_final < z_lo))
    if n_above > 0 or n_below > 0:
        logger.warning(
            "PA final population: %d/%d walkers above z_hi=%.3f, "
            "%d/%d below z_lo=%.3f. Drift outside the slab biases K_ads — "
            "raise pa.z_max if walkers escape upward (insufficient slab top).",
            n_above, M, z_hi, n_below, M, z_lo)

    # Physical adsorption analysis. The result lives on pa_out so run.py can
    # surface it in the parquet without re-doing the conversion.
    ads = adsorption_free_energy(
        logZ_ratio=float(pa_out["logZ_ratio"]),
        temperature=float(mc.temperature),
        z_lo=z_lo, z_hi=z_hi,
        c_std=float(pa.c_std),
    )
    logger.info(
        "Adsorption free energy: dG_box = %.3f kJ/mol, "
        "K_ads = %.3f nm, dG°_ads = %.3f kJ/mol "
        "(T=%.1f K, slab Δz=%.3f nm, c°=%.2f M, λ_std=%.3f nm).",
        ads.dG_box_kJ_per_mol, ads.K_ads_nm, ads.dG_std_kJ_per_mol,
        ads.temperature_K, z_hi - z_lo, ads.c_std_M, ads.lambda_std_nm)
    if not np.isfinite(ads.dG_std_kJ_per_mol):
        logger.warning(
            "K_ads = %.3f nm ≤ 0 — sampled slab shows no net surface excess. "
            "Either binding is unfavourable, or z_lo (=safe-COM) excludes the "
            "actual bound state.", ads.K_ads_nm)

    fq = np.asarray(pa_out["final_quats"])   # (M, 4)
    fe = np.asarray(pa_out["final_energies"])  # (M,)
    out = {
        "quats":      fq[None],              # (1, M, 4)
        "transs":     ft[None],              # (1, M, 3)
        "energies":   fe[None],              # (1, M) — real final energies per walker
        "accept":     np.full((1, M), np.nan),  # per-walker accept not tracked
        "system_ids": [0],
        "_pa_raw":    pa_out,
        "pa_adsorption": ads,
        "_pa_beta0_check": beta0_check,
        "_pa_slab":   {"z_lo": z_lo, "z_hi": z_hi,
                       "lateral_area_nm2": lateral_area,
                       "n_drift_above": n_above, "n_drift_below": n_below},
    }
    return out, [beta_of(mc.temperature)]


# ---------------------------------------------------------------------------
# Flexible PA runner
# ---------------------------------------------------------------------------

def _run_pa_flexible(config: SimConfig, atoms, energy_fn,
                       flex_data: dict, phi_data: dict | None = None
                       ) -> tuple[dict, list]:
    import jax
    import jax.numpy as jnp
    from ptmc.sampler._pa_flexible import run_pa_flexible
    from ptmc.analysis.adsorption import adsorption_free_energy, beta0_adequacy
    from ptmc.config import INIT_QUAT_KEY_OFFSET

    pa = config.pa
    mc = config.mc
    M  = pa.n_walkers
    chi_schedule = flex_data["chi_schedule"]
    chi_topo = flex_data["chi_topo"]
    K = chi_topo.k

    key  = jax.random.PRNGKey(mc.seed)
    init_key = jax.random.fold_in(key, 0)
    run_key  = jax.random.fold_in(key, 1)

    # Initial random quaternions.
    qn = jax.random.normal(jax.random.fold_in(init_key, INIT_QUAT_KEY_OFFSET),
                           (M, 4))
    init_quats = qn / jnp.linalg.norm(qn, axis=-1, keepdims=True)

    # Uniform distribution of initial z over the slab.
    z_lo, z_hi = _pa_z_range(config, atoms, phi_data)
    z_key = jax.random.fold_in(init_key, INIT_QUAT_KEY_OFFSET + 1)
    zs = jax.random.uniform(z_key, (M,), minval=z_lo, maxval=z_hi)
    init_transs = jnp.zeros((M, 3)).at[:, 2].set(zs)

    # Initial chi: zero (all chi angles start at 0 = protein's original conformation).
    init_chis = jnp.zeros((M, K), dtype=jnp.float32)

    lateral_area = _pa_lateral_area(config, phi_data)
    logger.info(
        "PA-flexible initial slab: z ∈ [%.3f, %.3f] nm (Δz=%.3f), "
        "lateral area = %.3f nm², %d walkers, K=%d chi DOFs.",
        z_lo, z_hi, z_hi - z_lo, lateral_area, M, K)

    pa_out = run_pa_flexible(
        key=run_key,
        init_quats=init_quats,
        init_transs=init_transs,
        init_chis=init_chis,
        energy_fn=energy_fn,
        beta0=beta_of(pa.T_start),
        beta_target=beta_of(mc.temperature),
        n_sweep=mc.n_steps,
        sigma_rot=mc.sigma_rot,
        sigma_trans=mc.sigma_trans,
        sigma_chi=mc.sigma_chi,
        axis_mask=jnp.asarray(mc.axis_mask, dtype=jnp.float32),
        target_ess=pa.target_ess,
        max_steps=pa.max_annealing_steps,
    )

    logger.info("PA-flexible done. logZ_ratio = %.4f, cooling steps = %d, converged = %s",
                pa_out["logZ_ratio"], pa_out["n_steps"], pa_out["converged"])

    # β₀ adequacy check.
    init_E = jax.vmap(energy_fn)(init_quats, init_transs, init_chis)
    beta0_check = beta0_adequacy(np.asarray(init_E), beta_of(pa.T_start))
    log_fn = logger.info if beta0_check["ok"] else logger.warning
    log_fn("PA β₀-adequacy: %s (mean |β₀E|=%.3f, max=%.3f).",
           beta0_check["msg"], beta0_check["mean"], beta0_check["max"])

    # Drift check.
    ft = np.asarray(pa_out["final_transs"])
    z_final = ft[:, 2]
    n_above = int(np.sum(z_final > z_hi))
    n_below = int(np.sum(z_final < z_lo))
    if n_above > 0 or n_below > 0:
        logger.warning(
            "PA final population: %d/%d walkers above z_hi=%.3f, "
            "%d/%d below z_lo=%.3f. Drift outside the slab biases K_ads.",
            n_above, M, z_hi, n_below, M, z_lo)

    # Adsorption analysis.
    ads = adsorption_free_energy(
        logZ_ratio=float(pa_out["logZ_ratio"]),
        temperature=float(mc.temperature),
        z_lo=z_lo, z_hi=z_hi,
        c_std=float(pa.c_std),
    )
    logger.info(
        "Adsorption free energy: dG_box = %.3f kJ/mol, "
        "K_ads = %.3f nm, dG°_ads = %.3f kJ/mol "
        "(T=%.1f K, slab Δz=%.3f nm, c°=%.2f M, λ_std=%.3f nm).",
        ads.dG_box_kJ_per_mol, ads.K_ads_nm, ads.dG_std_kJ_per_mol,
        ads.temperature_K, z_hi - z_lo, ads.c_std_M, ads.lambda_std_nm)
    if not np.isfinite(ads.dG_std_kJ_per_mol):
        logger.warning(
            "K_ads = %.3f nm ≤ 0 — sampled slab shows no net surface excess.",
            ads.K_ads_nm)

    fq = np.asarray(pa_out["final_quats"])
    fe = np.asarray(pa_out["final_energies"])
    fc = np.asarray(pa_out["final_chis"])
    out = {
        "quats":      fq[None],              # (1, M, 4)
        "transs":     ft[None],              # (1, M, 3)
        "energies":   fe[None],              # (1, M)
        "accept":     np.full((1, M), np.nan),
        "system_ids": [0],
        "_pa_raw":    pa_out,
        "pa_adsorption": ads,
        "_pa_beta0_check": beta0_check,
        "_pa_slab":   {"z_lo": z_lo, "z_hi": z_hi,
                       "lateral_area_nm2": lateral_area,
                       "n_drift_above": n_above, "n_drift_below": n_below},
        "_flex_chis": fc[None],              # (1, M, K) for inspection
    }
    return out, [beta_of(mc.temperature)]


# ---------------------------------------------------------------------------
# Flexible PT runner
# ---------------------------------------------------------------------------

def _run_pt_flexible(config: SimConfig, atoms, energy_fn,
                       flex_data: dict) -> tuple[dict, list]:
    import jax
    import jax.numpy as jnp
    from ptmc.sampler._pt_flexible import run_pt_flexible
    from ptmc.config import INIT_QUAT_KEY_OFFSET

    pt = config.pt
    mc = config.mc
    chi_topo = flex_data["chi_topo"]
    K = chi_topo.k

    T_ladder = np.geomspace(pt.T_min, pt.T_max, pt.n_replicas)
    betas    = np.array([beta_of(T) for T in T_ladder])
    beta_low = float(betas[np.argmax(betas)])
    R   = pt.n_replicas

    key = jax.random.PRNGKey(mc.seed)
    init_key = jax.random.fold_in(key, 0)
    run_key = jax.random.fold_in(key, 1)

    # Initial poses: R replicas, random quats.
    init_keys = jax.random.split(init_key, R)
    qn = jax.vmap(lambda k: jax.random.normal(
        jax.random.fold_in(k, INIT_QUAT_KEY_OFFSET), (4,)))(init_keys)
    init_quats = qn / jnp.linalg.norm(qn, axis=-1, keepdims=True)  # (R, 4)
    init_z = _safe_init_z(atoms, _active_z_min(config), config.mc.sigma_trans,
                          config.mc.init_z or None)
    init_transs = jnp.zeros((R, 3)).at[:, 2].set(float(init_z))    # (R, 3)
    init_chis = jnp.zeros((R, K), dtype=jnp.float32)                # (R, K)

    logger.info("PT-flexible: %d replicas at T ∈ [%.0f, %.0f] K, K=%d chi DOFs, "
                "%d rounds × %d sweeps.",
                R, pt.T_min, pt.T_max, K, pt.n_rounds, pt.n_sweep)

    pt_out = run_pt_flexible(
        key=run_key,
        init_quats=init_quats,
        init_transs=init_transs,
        init_chis=init_chis,
        energy_fn=energy_fn,
        betas=betas,
        sigma_rot=mc.sigma_rot,
        sigma_trans=mc.sigma_trans,
        sigma_chi=mc.sigma_chi,
        axis_mask=jnp.asarray(mc.axis_mask, dtype=jnp.float32),
        n_rounds=pt.n_rounds,
        n_sweep=pt.n_sweep,
    )

    logger.info("PT-flexible done. MC accept (per T): %s",
                np.array2string(np.asarray(pt_out["mc_accept_rate"]), precision=3))
    logger.info("PT-flexible swap accept (per gap): %s",
                np.array2string(np.asarray(pt_out["swap_accept_rate"]), precision=3))

    # Low-T trajectory: (n_rounds, 4/3/K)
    low_q = np.asarray(pt_out["low_T_quat"])     # (n_rounds, 4)
    low_t = np.asarray(pt_out["low_T_trans"])    # (n_rounds, 3)
    low_c = np.asarray(pt_out["low_T_chi"])      # (n_rounds, K)

    # Compute real energies for the low-T trajectory frames.
    pt_energies = np.asarray(
        jax.vmap(energy_fn)(
            jnp.asarray(low_q), jnp.asarray(low_t), jnp.asarray(low_c),
        )
    )
    out = {
        "quats":      low_q[None],                              # (1, n_rounds, 4)
        "transs":     low_t[None],                              # (1, n_rounds, 3)
        "energies":   pt_energies[None],                        # (1, n_rounds)
        "accept":     np.full((1, pt.n_rounds), np.nan),
        "system_ids": [0],
        "_pt_raw":    pt_out,
        "_flex_chis": low_c[None],                              # (1, n_rounds, K)
    }
    return out, [beta_low]


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(config: SimConfig) -> tuple[dict, list]:
    """Parse input, build surface, run sampler, optionally write trajectory.

    Returns
    -------
    out : dict
        Contains at minimum 'quats' (S, C, 4), 'transs' (S, C, 3),
        'energies', 'accept', 'system_ids'.
    betas : list[float]
        Per-system inverse temperature in mol/kJ, used for orientation
        summarization (passed back to the caller).
    """
    # 1. Parse structure ────────────────────────────────────────────────────
    logger.info("Parsing PDB: %s", config.pdb_path)
    pdb  = parse_pdb(config.pdb_path)
    logger.info("Parsing topology: %s", config.top_path)
    topo = parse_topology(config.top_path)
    atoms = build_atoms(pdb, topo)
    logger.info("Protein: %d atoms, net charge %.2f e", atoms.n, atoms.net_charge)

    st      = config.surface_type
    sampler = config.sampler

    # 1b. Flexible-DOF setup ─────────────────────────────────────────────────
    # Build chi topology, schedule, bonded/intra-NB parameters once so the
    # flexible closures (and runners) can consume them.
    _flex_data = {}
    if config.mc.flexible:
        from ptmc.flexible.topology import build_chi_topology
        from ptmc.flexible.schedule import build_chi_schedule
        from ptmc.flexible.bonded import build_bonded_params
        from ptmc.flexible.excl_table import build_exclusion_table
        from ptmc.flexible.intra_nb import build_intra_nb_params

        logger.info("Building chi topology from %s", config.top_path)
        _chi_topo = build_chi_topology(config.top_path)
        _chi_schedule = build_chi_schedule(_chi_topo)
        _bonded_params = build_bonded_params(config.top_path)
        _excl_table = build_exclusion_table(config.top_path)
        # Use the surface-type-specific Debye length for intra-NB screening.
        _intra_lamD = _active_z_min(config)
        # For the active Debye length, grab it from the surface config.
        if st in ("continuum", "patterned"):
            _intra_lamD = float(config.surface.lambda_D)
        elif st == "crystal":
            _intra_lamD = float(config.crystal.lambda_D)
        elif st == "full_atom":
            _intra_lamD = float(config.full_atom.lambda_D)
        elif st in ("agarose", "agarose_surface"):
            _intra_lamD = float(config.agarose.lambda_D)
        _intra_nb_params = build_intra_nb_params(config.top_path, lambda_D=_intra_lamD,
                                                  excl=_excl_table)
        logger.info("Flexible setup: K=%d chi DOFs, %d flexible residues, "
                    "max_n_chi=%d, %d periodic+%d RB+%d harmonic dihedrals, "
                    "%d 1-4 pairs.",
                    _chi_topo.k, _chi_schedule.n_flex_res, _chi_schedule.max_n_chi,
                    _bonded_params.m_periodic, _bonded_params.m_rb,
                    _bonded_params.m_harmonic,
                    _excl_table.pair14_idx.shape[0])
        _flex_data = dict(
            chi_topo=_chi_topo,
            chi_schedule=_chi_schedule,
            bonded_params=_bonded_params,
            excl_table=_excl_table,
            intra_nb_params=_intra_nb_params,
        )

    # 2. Build energy_fn (needed for pt/pa and trajectory recording) ───────
    energy_fn = None
    _phi_data = {}   # shared between patterned HT and energy_fn

    if st == "continuum":
        if config.mc.flexible:
            energy_fn = _efn_continuum_flexible(
                atoms, config.surface,
                _flex_data["chi_schedule"],
                _flex_data["bonded_params"],
                _flex_data["intra_nb_params"])
        else:
            energy_fn = _efn_continuum(atoms, config.surface)

    elif st == "patterned":
        if not config.phi_grid_path:
            raise ValueError("--surface-type patterned requires --phi-grid FILE")
        phi_field   = np.load(config.phi_grid_path).astype(np.float32)
        grid_origin = np.asarray(config.phi_grid_origin, dtype=np.float32)
        grid_spc    = np.asarray(config.phi_grid_spacing, dtype=np.float32)
        _phi_data   = dict(phi_field=phi_field, grid_origin=grid_origin,
                           grid_spacing=grid_spc)
        if config.mc.flexible:
            energy_fn = _efn_patterned_flexible(
                atoms, config.surface, phi_field, grid_origin, grid_spc,
                _flex_data["chi_schedule"],
                _flex_data["bonded_params"],
                _flex_data["intra_nb_params"])
        else:
            energy_fn = _efn_patterned(atoms, config.surface,
                                          phi_field, grid_origin, grid_spc)

    elif st == "agarose":
        from ptmc.energy.agarose import build_agarose_system_dict
        from ptmc.model.structures import AgaroseGel
        ag  = config.agarose
        gel = AgaroseGel(
            L=ag.L, n_seeds=ag.n_seeds, sigma=ag.sigma, A=ag.A,
            doping_frac=ag.doping_frac, q_ligand=ag.q_ligand,
            lambda_D=ag.lambda_D, dielectric=ag.dielectric, seed=ag.gel_seed,
        )
        _sys = build_agarose_system_dict(atoms, gel,
                                          grid_spacing=ag.grid_spacing,
                                          margin=ag.margin,
                                          sample_density=ag.sample_density,
                                          doping_correlation=ag.doping_correlation,
                                          floor_pctile=ag.floor_pctile,
                                          beta=beta_of(config.mc.temperature))
        energy_fn = _efn_agarose(atoms,
                                   _sys["U_steric_field"], _sys["phi_elec_field"],
                                   _sys["grid_origin"],    _sys["grid_spacing"])
        _phi_data["_agarose_system"] = _sys

    elif st == "agarose_surface":
        from ptmc.energy.agarose import build_agarose_surface_grids
        ag  = config.agarose
        L, th = ag.L, ag.thickness
        n_lat = max(2, int(np.ceil(L / ag.grid_spacing)))
        n_z   = max(2, int(np.ceil(th * 1.5 / ag.grid_spacing)))
        xs = -L / 2 + np.arange(n_lat) * ag.grid_spacing
        ys = -L / 2 + np.arange(n_lat) * ag.grid_spacing
        zs = ag.z_min + np.arange(n_z) * ag.grid_spacing
        U, phi_e, origin, spacing = build_agarose_surface_grids(
            xs, ys, zs, L=L, thickness=th,
            n_seeds=ag.n_seeds, sigma=ag.sigma, A=ag.A,
            doping_frac=ag.doping_frac, q_ligand=ag.q_ligand,
            lambda_D=ag.lambda_D, dielectric=ag.dielectric, seed=ag.gel_seed,
            margin=ag.margin, sample_density=ag.sample_density,
            doping_correlation=ag.doping_correlation,
        )
        energy_fn = _efn_agarose_surface(atoms, U, phi_e, origin, spacing, ag.z_min)
        _phi_data["_agarose_surface_grids"] = (U, phi_e, origin, spacing)

    elif st == "crystal":
        cr = config.crystal
        logger.info("Building crystal surface: %s hkl=%s n_layers=%d",
                    cr.crystal, cr.hkl, cr.n_layers)
        from ptmc.surface.builder import build_surface
        _disc_surf, _grids = build_surface(
            crystal=cr.crystal,
            hkl=tuple(int(x) for x in cr.hkl),
            n_layers=cr.n_layers,
            vacuum=cr.vacuum,
            hydroxylate=cr.hydroxylate,
            lambda_D=cr.lambda_D,
            z_min=cr.z_min,
            r_cut=cr.r_cut,
            spacing=cr.grid_spacing,
            z_range=(cr.z_min, cr.z_grid_max),
        )
        # Cell dimensions from the slab lattice
        import math
        from ptmc.surface.lattice import get_crystal
        from ptmc.surface.slab import cut_slab
        _bulk = get_crystal(cr.crystal)
        _slab = cut_slab(_bulk, hkl=tuple(int(x) for x in cr.hkl),
                         n_layers=cr.n_layers, vacuum=cr.vacuum)
        cell_Lx = float(_slab.cell[0, 0])
        cell_Ly = float(_slab.cell[1, 1])
        logger.info("Surface cell: Lx=%.3f nm, Ly=%.3f nm, %d atoms",
                    cell_Lx, cell_Ly, _disc_surf.m)
        logger.info("Grid shape: %s, z_min=%.3f nm",
                    _grids.G12.shape, _grids.z_min)
        if config.mc.flexible:
            energy_fn = _efn_grid_flexible(
                atoms, _grids, cell_Lx, cell_Ly,
                _flex_data["chi_schedule"],
                _flex_data["bonded_params"],
                _flex_data["intra_nb_params"])
        else:
            energy_fn = _efn_grid(atoms, _grids, cell_Lx, cell_Ly)
        _phi_data["_disc_surf"] = _disc_surf
        _phi_data["_grids"]     = _grids
        _phi_data["_cell_Lx"]   = cell_Lx
        _phi_data["_cell_Ly"]   = cell_Ly

    else:  # full_atom
        fa = config.full_atom
        if not fa.surface_pdb:
            raise ValueError("--surface-type full_atom requires --surface-pdb")
        if fa.surface_ff_json and fa.surface_top:
            raise ValueError("Provide either --surface-ff-json or --surface-top, not both")
        if not fa.surface_ff_json and not fa.surface_top:
            raise ValueError("--surface-type full_atom requires --surface-ff-json or --surface-top")

        logger.info("Loading full-atom surface: %s", fa.surface_pdb)
        if fa.surface_ff_json:
            from ptmc.surface.forcefield import load_surface_from_pdb_json
            _disc_surf = load_surface_from_pdb_json(
                fa.surface_pdb, fa.surface_ff_json, fa.lambda_D, fa.z_min)
        else:
            from ptmc.surface.forcefield import load_surface_from_gromacs
            _disc_surf = load_surface_from_gromacs(
                fa.surface_pdb, fa.surface_top, fa.lambda_D, fa.z_min)

        # Shift so topmost atom sits at z=0.  Use dataclasses.replace so the
        # surface object returned by the loader stays untouched in case it's
        # cached/shared.
        from dataclasses import replace
        z_top = float(_disc_surf.pos[:, 2].max())
        new_pos = _disc_surf.pos.copy()
        new_pos[:, 2] -= z_top
        _disc_surf = replace(_disc_surf, pos=new_pos)
        logger.info("Surface: %d atoms, z_min=%.3f nm", _disc_surf.m, fa.z_min)

        # Build field grids (JAX GPU-accelerated, PBC minimum-image)
        from ptmc.energy.grid_build_jax import build_grids_cutoff
        cell_Lx, cell_Ly = fa.cell_xy
        _grids = build_grids_cutoff(
            _disc_surf,
            x_range=(0.0, cell_Lx),
            y_range=(0.0, cell_Ly),
            z_range=(fa.z_min, fa.z_grid_max),
            spacing=fa.grid_spacing,
            cell=(cell_Lx, cell_Ly),
            r_cut=fa.r_cut,
        )
        logger.info("Grid shape: %s, spacing=%.3f nm", _grids.G12.shape, fa.grid_spacing)
        if config.mc.flexible:
            energy_fn = _efn_grid_flexible(
                atoms, _grids, float(cell_Lx), float(cell_Ly),
                _flex_data["chi_schedule"],
                _flex_data["bonded_params"],
                _flex_data["intra_nb_params"])
        else:
            energy_fn = _efn_grid(atoms, _grids, float(cell_Lx), float(cell_Ly))
        _phi_data["_disc_surf"] = _disc_surf
        _phi_data["_grids"]     = _grids
        _phi_data["_cell_Lx"]   = float(cell_Lx)
        _phi_data["_cell_Ly"]   = float(cell_Ly)

    # 3. Run sampler ────────────────────────────────────────────────────────
    if sampler == "pt":
        if st not in ("continuum", "patterned", "crystal", "full_atom"):
            raise ValueError(
                f"--sampler pt does not support surface type {st!r}. "
                f"Only continuum / patterned / crystal / full_atom are wired.")
        if config.mc.flexible:
            out, betas = _run_pt_flexible(config, atoms, energy_fn, _flex_data)
        else:
            out, betas = _run_pt(config, atoms, energy_fn)

    elif sampler == "pa":
        if st not in ("continuum", "patterned", "crystal", "full_atom"):
            raise ValueError(
                f"--sampler pa does not support surface type {st!r}. "
                f"Only continuum / patterned / crystal / full_atom are wired.")
        if config.mc.flexible:
            out, betas = _run_pa_flexible(config, atoms, energy_fn, _flex_data,
                                           _phi_data)
        else:
            out, betas = _run_pa(config, atoms, energy_fn, _phi_data)

    else:
        raise ValueError(f"Unknown sampler: {sampler!r}")

    # 4. Post-simulation audit ───────────────────────────────────────────────
    n_inf, n_nan = _validate_energies(
        out, label=f"sampler={sampler}, surface={st}")
    out["n_nonfinite_energy"] = n_inf + n_nan

    # 5. Trajectory recording ───────────────────────────────────────────────
    if config.save_traj:
        if energy_fn is None:
            logger.warning("Trajectory recording skipped: no energy_fn for %r", st)
        else:
            _write_traj(config, atoms, energy_fn)

    return out, betas
