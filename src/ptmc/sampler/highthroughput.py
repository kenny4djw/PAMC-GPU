"""High-throughput multi-system batched MC (JAX), with deterministic seeding
and checkpoint/resume.

Inner vmap iterates chains (C) of a single system; outer vmap iterates systems
(S). Per-chain inputs (keys, init quat/trans) have a leading (S, C, ...) axis;
per-system inputs (pos0, charges, LJ coefficients, surface params, fields) have
a leading (S, ...) axis with ``in_axes=None`` on the inner vmap so they are
broadcast across chains WITHOUT materializing an (S*C, N, 3) tensor. Memory
scales as O(S * N) for the protein arrays, not O(S * C * N) as in the
flattened-batch version. RNG is fully positional:

    chain_key   = fold_in(fold_in(master, system_id), chain_index)
    step t key  = fold_in(chain_key, absolute_step_index)

so (a) a system's result is independent of how systems are batched together
(batch == individual, same seed) and (b) running [0,k) then resuming [k,N) from
the saved state is identical to running [0,N) continuously (checkpoint/resume).
"""
from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from ptmc.mc.metropolis import scan_one
from ptmc.config import INIT_QUAT_KEY_OFFSET
from ptmc.energy.reference import steele_coefficients_for


# ===========================================================================
# Continuum-surface (Steele 9-3 + screened Coulomb) hot loop
# ===========================================================================

def _ht_chain(chain_key, quat0, trans0, beta, pos0, q, c6p, c12p, psi0,
              cA, cB, lamD, z_min, sigma_rot, sigma_trans, axis_mask,
              start_step, n_steps):
    """One chain on a homogeneous continuum surface, positional per-step RNG."""
    from ptmc.energy.reference import steele_energy_jax

    def efn(quat, trans):
        return steele_energy_jax(quat, trans, pos0, q, c6p, c12p, cA, cB,
                                 lamD, z_min, psi0)
    return scan_one(chain_key, quat0, trans0, efn, beta,
                    sigma_rot, sigma_trans, axis_mask, n_steps,
                    start_step=start_step)


# Inner vmap: over chains of ONE system. Per-chain: chain_key, quat0, trans0.
# All other arrays are per-system (None ⇒ broadcast across chains).
_ht_chain_inner = jax.vmap(
    _ht_chain,
    in_axes=(0, 0, 0,                       # chain_key, quat0, trans0
             None, None, None, None, None,  # beta, pos0, q, c6p, c12p
             None, None, None, None, None,  # psi0, cA, cB, lamD, z_min
             None, None, None,              # sigma_rot, sigma_trans, axis_mask
             None, None))                   # start_step, n_steps

# Outer vmap: over systems. Per-system: chain_key (S,C,2), quat0 (S,C,4),
# trans0 (S,C,3), and the scalar/vector params with leading S axis.
_ht_chain_outer = jax.vmap(
    _ht_chain_inner,
    in_axes=(0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             None, None, None,
             None, None))


@partial(jax.jit, static_argnums=(17,))
def _ht_run_ps(chain_keys, quat0, trans0, beta, pos0, q, c6p, c12p, psi0,
               cA, cB, lamD, z_min, sigma_rot, sigma_trans, axis_mask,
               start_step, n_steps):
    """Double-vmap runner over (S systems) × (C chains/system).

    SHAPE: chain_keys (S,C,2), quat0 (S,C,4), trans0 (S,C,3); beta (S,);
    pos0 (S,N,3), q (S,N), c6p (S,N), c12p (S,N); psi0/cA/cB/lamD/z_min (S,).
    """
    return _ht_chain_outer(
        chain_keys, quat0, trans0, beta, pos0, q, c6p, c12p, psi0,
        cA, cB, lamD, z_min, sigma_rot, sigma_trans, axis_mask,
        start_step, n_steps)


@partial(jax.jit, static_argnums=(2,))
def _chain_keys_jit(master, sids, C):
    """Per-chain PRNGKeys (S, C, 2) via vmap(vmap(fold_in)).

    Byte-identical to the legacy ``S*C`` Python-level loop because
    ``jax.random.fold_in`` is a pure function of (key, int).
    """
    cidx = jnp.arange(C, dtype=jnp.uint32)
    fold_c = jax.vmap(jax.random.fold_in, in_axes=(0, 0))

    def per_system(sid):
        sk = jax.random.fold_in(master, sid)
        sk_b = jnp.broadcast_to(sk, (C, 2))
        return fold_c(sk_b, cidx)

    return jax.vmap(per_system)(sids)


def _chain_keys(master_seed, system_ids, C):
    """Per-chain PRNGKeys (S, C, 2) from fold_in(fold_in(master, sid), c).

    Native (S, C, 2) layout feeds the double-vmap runner directly — no flat→sc
    reshape needed at the call site. Tests / external code that want the flat
    (B, 2) view can call ``.reshape(-1, 2)``.
    """
    master = jax.random.PRNGKey(master_seed)
    sids = jnp.asarray([int(s) for s in system_ids], dtype=jnp.uint32)
    return _chain_keys_jit(master, sids, int(C))


# ---------------------------------------------------------------------------
# Per-system batch builder. Heavy arrays (pos0, q, c6p, c12p) are stored ONCE
# per system (S × N), not (S × C × N), so memory scales independently of C.
# The double-vmap runner broadcasts these arrays across chains via
# ``in_axes=None`` on the inner vmap — no materialization required.
# ---------------------------------------------------------------------------

def build_batch(systems, C):
    """Build per-system batch arrays for S systems × C chains.

    Returns
    -------
    dict of arrays keyed with the ``_ps`` suffix and shape ``(S, …)``. The
    runners consume these directly via double vmap; broadcast over chains
    happens inside the JIT via ``in_axes=None`` on the inner vmap, so there
    is never an ``(S*C, N, 3)`` tensor in memory.
    """
    S = len(systems)
    N = systems[0]["pos0"].shape[0]

    # eps_surf/sigma_surf (optional, both-or-neither) select the ε-σ
    # Lorentz-Berthelot Steele path; otherwise the geometric C6/C12 path.
    coeffs = [steele_coefficients_for(s["c6"], s["c12"],
                                      s.get("c6_surf", 1.0), s.get("c12_surf", 1.0),
                                      s.get("eps_surf"), s.get("sigma_surf"),
                                      s["rho_s"]) for s in systems]

    pos0_ps  = jnp.asarray(np.stack([np.asarray(s["pos0"]) for s in systems]))  # (S,N,3)
    q_ps     = jnp.asarray(np.stack([np.asarray(s["q"]) for s in systems]))     # (S,N)
    c6p_ps   = jnp.asarray(np.stack([c[0] for c in coeffs]))                    # (S,N)
    c12p_ps  = jnp.asarray(np.stack([c[1] for c in coeffs]))                    # (S,N)
    psi0_ps  = jnp.asarray(np.array([s["psi0"]    for s in systems]))           # (S,)
    beta_ps  = jnp.asarray(np.array([s["beta"]    for s in systems]))           # (S,)
    cA_ps    = jnp.asarray(np.array([c[2] for c in coeffs]))                    # (S,)
    cB_ps    = jnp.asarray(np.array([c[3] for c in coeffs]))                    # (S,)
    lamD_ps  = jnp.asarray(np.array([s["lambda_D"] for s in systems]))          # (S,)
    z_min_ps = jnp.asarray(np.array([s["z_min"]    for s in systems]))          # (S,)
    z0_ps    = jnp.asarray(np.array([s.get("init_z", 0.95) for s in systems]))  # (S,)

    return dict(
        S=S, N=N, C=C,
        pos0_ps=pos0_ps, q_ps=q_ps, c6p_ps=c6p_ps, c12p_ps=c12p_ps,
        psi0_ps=psi0_ps, beta_ps=beta_ps, cA_ps=cA_ps, cB_ps=cB_ps,
        lamD_ps=lamD_ps, z_min_ps=z_min_ps, z0_ps=z0_ps,
    )


def initial_state(chain_keys, z0):
    """Deterministic initial poses: random unit quat per chain, trans=[0,0,z0].

    Shape-polymorphic: ``chain_keys`` and ``z0`` share leading axes (e.g. ``(M,)``
    or ``(S, C)``); outputs preserve the same lead. So this works equally well
    on flat ``(M, 2)`` keys and on native ``(S, C, 2)`` double-vmap keys.
    """
    chain_keys = jnp.asarray(chain_keys)
    z0 = jnp.asarray(z0)
    lead = chain_keys.shape[:-1]
    flat_keys = chain_keys.reshape(-1, 2)
    flat_z = z0.reshape(-1)
    qn = jax.vmap(lambda k: jax.random.normal(
        jax.random.fold_in(k, INIT_QUAT_KEY_OFFSET), (4,)))(flat_keys)
    quat_flat = qn / jnp.linalg.norm(qn, axis=1, keepdims=True)
    trans_flat = jnp.stack([jnp.zeros_like(flat_z),
                            jnp.zeros_like(flat_z), flat_z], axis=1)
    return quat_flat.reshape(lead + (4,)), trans_flat.reshape(lead + (3,))


def run_systems(master_seed, systems, n_chains, n_steps, sigma_rot,
                sigma_trans, axis_mask, start_step=0, init=None):
    """Run S systems × n_chains in one GPU batch.

    Returns per-system final states (quats (S,C,4), transs (S,C,3),
    energies (S,C), accept (S,C)) and the raw flat state for checkpointing.
    """
    S = len(systems)
    C = n_chains
    sids = [s["system_id"] for s in systems]

    b = build_batch(systems, C)
    keys_sc = _chain_keys(master_seed, sids, C)  # (S, C, 2)
    if init is None:
        z0_sc = jnp.broadcast_to(b["z0_ps"][:, None], (S, C))
        quat0_sc, trans0_sc = initial_state(keys_sc, z0_sc)
    else:
        quat0, trans0 = init
        quat0_sc = quat0.reshape(S, C, 4)
        trans0_sc = trans0.reshape(S, C, 3)

    qf_sc, tf_sc, ef_sc, ar_sc = _ht_run_ps(
        keys_sc, quat0_sc, trans0_sc,
        b["beta_ps"], b["pos0_ps"], b["q_ps"], b["c6p_ps"], b["c12p_ps"],
        b["psi0_ps"], b["cA_ps"], b["cB_ps"], b["lamD_ps"], b["z_min_ps"],
        float(sigma_rot), float(sigma_trans), jnp.asarray(axis_mask),
        int(start_step), int(n_steps))
    jax.block_until_ready((qf_sc, tf_sc, ef_sc, ar_sc))

    quats = np.asarray(qf_sc)      # (S, C, 4)
    transs = np.asarray(tf_sc)     # (S, C, 3)
    energies = np.asarray(ef_sc)   # (S, C)
    accept = np.asarray(ar_sc)     # (S, C)
    flat_state = (qf_sc.reshape(S * C, 4), tf_sc.reshape(S * C, 3))
    return dict(quats=quats, transs=transs, energies=energies, accept=accept,
                system_ids=sids, flat_state=flat_state)


# ===========================================================================
# Patterned-electrostatics surface (Steele 9-3 + grid-interpolated phi)
# ===========================================================================

def _ht_chain_patterned(chain_key, quat0, trans0, beta, pos0, q, c6p, c12p,
                        cA, cB, lamD, z_min,
                        sigma_rot, sigma_trans, axis_mask, start_step, n_steps,
                        phi_field, grid_origin, grid_spacing):
    """One chain with grid-interpolated patterned-surface electrostatics."""
    from ptmc.energy.grid_energy import patterned_energy_jax

    def efn(quat, trans):
        return patterned_energy_jax(quat, trans, pos0, q, c6p, c12p, cA, cB,
                                     lamD, z_min, phi_field,
                                     grid_origin, grid_spacing)
    return scan_one(chain_key, quat0, trans0, efn, beta,
                    sigma_rot, sigma_trans, axis_mask, n_steps,
                    start_step=start_step)


_ht_pattern_inner = jax.vmap(
    _ht_chain_patterned,
    in_axes=(0, 0, 0,
             None, None, None, None, None,
             None, None, None, None,
             None, None, None, None, None,
             None, None, None))

_ht_pattern_outer = jax.vmap(
    _ht_pattern_inner,
    in_axes=(0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0,
             None, None, None, None, None,
             0, 0, 0))


@partial(jax.jit, static_argnums=(16,))
def _ht_run_patterned_ps(chain_keys, quat0, trans0, beta, pos0, q, c6p, c12p,
                          cA, cB, lamD, z_min,
                          sigma_rot, sigma_trans, axis_mask, start_step, n_steps,
                          phi_field, grid_origin, grid_spacing):
    return _ht_pattern_outer(
        chain_keys, quat0, trans0, beta, pos0, q, c6p, c12p,
        cA, cB, lamD, z_min,
        sigma_rot, sigma_trans, axis_mask, start_step, n_steps,
        phi_field, grid_origin, grid_spacing)


def run_systems_patterned(master_seed, systems, n_chains, n_steps,
                           sigma_rot, sigma_trans, axis_mask,
                           start_step=0, init=None):
    """Run S systems × n_chains with patterned phi(x,y,z) electrostatics.

    Each system dict must contain the surface parameters required by
    ``run_systems()`` plus a precomputed phi grid:
        phi_field      : (nx, ny, nz)  electrostatic potential (kJ/mol/e)
        grid_origin    : (3,)  grid origin (nm)
        grid_spacing   : (3,)  grid spacing (nm)
    """
    S = len(systems)
    C = n_chains
    sids = [s["system_id"] for s in systems]

    b = build_batch(systems, C)
    keys_sc = _chain_keys(master_seed, sids, C)
    if init is None:
        z0_sc = jnp.broadcast_to(b["z0_ps"][:, None], (S, C))
        quat0_sc, trans0_sc = initial_state(keys_sc, z0_sc)
    else:
        quat0, trans0 = init
        quat0_sc = quat0.reshape(S, C, 4)
        trans0_sc = trans0.reshape(S, C, 3)

    # Per-system grids (S, nx, ny, nz), (S, 3), (S, 3). One copy per system,
    # broadcast over chains via in_axes=None on the inner vmap.
    phi_field_ps    = jnp.asarray(np.stack([np.asarray(s["phi_field"]) for s in systems]))
    grid_origin_ps  = jnp.asarray(np.stack([
        np.asarray(s.get("grid_origin", np.array([0.0, 0.0, 0.0]))) for s in systems]))
    grid_spacing_ps = jnp.asarray(np.stack([
        np.asarray(s.get("grid_spacing", np.array([0.1, 0.1, 0.1]))) for s in systems]))

    qf_sc, tf_sc, ef_sc, ar_sc = _ht_run_patterned_ps(
        keys_sc, quat0_sc, trans0_sc,
        b["beta_ps"], b["pos0_ps"], b["q_ps"], b["c6p_ps"], b["c12p_ps"],
        b["cA_ps"], b["cB_ps"], b["lamD_ps"], b["z_min_ps"],
        float(sigma_rot), float(sigma_trans), jnp.asarray(axis_mask),
        int(start_step), int(n_steps),
        phi_field_ps, grid_origin_ps, grid_spacing_ps)
    jax.block_until_ready((qf_sc, tf_sc, ef_sc, ar_sc))

    quats = np.asarray(qf_sc)
    transs = np.asarray(tf_sc)
    energies = np.asarray(ef_sc)
    accept = np.asarray(ar_sc)
    flat_state = (qf_sc.reshape(S * C, 4), tf_sc.reshape(S * C, 3))
    return dict(quats=quats, transs=transs, energies=energies, accept=accept,
                system_ids=sids, flat_state=flat_state)


# ===========================================================================
# 3D agarose hydrogel: Gaussian soft-core + Yukawa elec via dual grids
# ===========================================================================

def _ht_chain_agarose(chain_key, quat0, trans0, beta, pos0, q,
                       sigma_rot, sigma_trans, axis_mask, start_step, n_steps,
                       U_steric_field, phi_elec_field, grid_origin, grid_spacing):
    """One chain in the 3D agarose gel (no hard wall, dual potentials)."""
    from ptmc.energy.grid_energy import agarose_energy_jax

    def efn(quat, trans):
        return agarose_energy_jax(quat, trans, pos0, q,
                                   U_steric_field, phi_elec_field,
                                   grid_origin, grid_spacing)
    return scan_one(chain_key, quat0, trans0, efn, beta,
                    sigma_rot, sigma_trans, axis_mask, n_steps,
                    start_step=start_step)


_ht_agarose_inner = jax.vmap(
    _ht_chain_agarose,
    in_axes=(0, 0, 0,
             None, None, None,
             None, None, None, None, None,
             None, None, None, None))

_ht_agarose_outer = jax.vmap(
    _ht_agarose_inner,
    in_axes=(0, 0, 0,
             0, 0, 0,
             None, None, None, None, None,
             0, 0, 0, 0))


@partial(jax.jit, static_argnums=(10,))
def _ht_run_agarose_ps(chain_keys, quat0, trans0, beta, pos0, q,
                        sigma_rot, sigma_trans, axis_mask, start_step, n_steps,
                        U_steric_field, phi_elec_field, grid_origin, grid_spacing):
    return _ht_agarose_outer(
        chain_keys, quat0, trans0, beta, pos0, q,
        sigma_rot, sigma_trans, axis_mask, start_step, n_steps,
        U_steric_field, phi_elec_field, grid_origin, grid_spacing)


def _build_agarose_batch_ps(systems):
    """Per-system arrays for the agarose runner (no Steele 9-3 params)."""
    pos0_ps = jnp.asarray(np.stack([np.asarray(s["pos0"]) for s in systems]))
    q_ps    = jnp.asarray(np.stack([np.asarray(s["q"]) for s in systems]))
    z0_ps   = np.array([s.get("init_z", 0.0) for s in systems])
    beta_ps = jnp.asarray(np.array([s["beta"] for s in systems]))
    return dict(pos0_ps=pos0_ps, q_ps=q_ps, z0_ps=z0_ps, beta_ps=beta_ps)


def run_systems_agarose(master_seed, systems, n_chains, n_steps,
                         sigma_rot, sigma_trans, axis_mask,
                         start_step=0, init=None):
    """Run S gel systems × n_chains in one GPU batch."""
    S = len(systems)
    C = n_chains
    sids = [s["system_id"] for s in systems]

    b = _build_agarose_batch_ps(systems)
    keys_sc = _chain_keys(master_seed, sids, C)
    if init is None:
        z0_sc = jnp.broadcast_to(jnp.asarray(b["z0_ps"])[:, None], (S, C))
        quat0_sc, trans0_sc = initial_state(keys_sc, z0_sc)
    else:
        quat0, trans0 = init
        quat0_sc = quat0.reshape(S, C, 4)
        trans0_sc = trans0.reshape(S, C, 3)

    U_ps   = jnp.asarray(np.stack([np.asarray(s["U_steric_field"]) for s in systems]))
    phi_ps = jnp.asarray(np.stack([np.asarray(s["phi_elec_field"]) for s in systems]))
    org_ps = jnp.asarray(np.stack([np.asarray(s["grid_origin"])    for s in systems]))
    spc_ps = jnp.asarray(np.stack([np.asarray(s["grid_spacing"])   for s in systems]))

    qf_sc, tf_sc, ef_sc, ar_sc = _ht_run_agarose_ps(
        keys_sc, quat0_sc, trans0_sc, b["beta_ps"], b["pos0_ps"], b["q_ps"],
        float(sigma_rot), float(sigma_trans), jnp.asarray(axis_mask),
        int(start_step), int(n_steps),
        U_ps, phi_ps, org_ps, spc_ps)
    jax.block_until_ready((qf_sc, tf_sc, ef_sc, ar_sc))

    quats = np.asarray(qf_sc)
    transs = np.asarray(tf_sc)
    energies = np.asarray(ef_sc)
    accept = np.asarray(ar_sc)
    flat_state = (qf_sc.reshape(S * C, 4), tf_sc.reshape(S * C, 3))
    return dict(quats=quats, transs=transs, energies=energies, accept=accept,
                system_ids=sids, flat_state=flat_state)


# ===========================================================================
# Flat agarose-coated surface: fibers at z<0, protein at z>0, hard wall at z_min
# ===========================================================================

def _ht_chain_agarose_surface(chain_key, quat0, trans0, beta, pos0, q, z_min,
                               sigma_rot, sigma_trans, axis_mask,
                               start_step, n_steps,
                               U_steric_field, phi_elec_field,
                               grid_origin, grid_spacing):
    """One chain above a flat gel-coated surface (hard wall at z_min)."""
    from ptmc.energy.grid_energy import agarose_surface_energy_jax

    def efn(quat, trans):
        return agarose_surface_energy_jax(quat, trans, pos0, q,
                                           U_steric_field, phi_elec_field,
                                           grid_origin, grid_spacing, z_min)
    return scan_one(chain_key, quat0, trans0, efn, beta,
                    sigma_rot, sigma_trans, axis_mask, n_steps,
                    start_step=start_step)


_ht_agarose_surf_inner = jax.vmap(
    _ht_chain_agarose_surface,
    in_axes=(0, 0, 0,
             None, None, None, None,
             None, None, None, None, None,
             None, None, None, None))

_ht_agarose_surf_outer = jax.vmap(
    _ht_agarose_surf_inner,
    in_axes=(0, 0, 0,
             0, 0, 0, 0,
             None, None, None, None, None,
             0, 0, 0, 0))


@partial(jax.jit, static_argnums=(10, 11))
def _ht_run_agarose_surface_ps(chain_keys, quat0, trans0, beta, pos0, q, z_min,
                                sigma_rot, sigma_trans, axis_mask,
                                start_step, n_steps,
                                U_steric_field, phi_elec_field,
                                grid_origin, grid_spacing):
    return _ht_agarose_surf_outer(
        chain_keys, quat0, trans0, beta, pos0, q, z_min,
        sigma_rot, sigma_trans, axis_mask, start_step, n_steps,
        U_steric_field, phi_elec_field, grid_origin, grid_spacing)


def _build_agarose_surface_batch_ps(systems):
    pos0_ps  = jnp.asarray(np.stack([np.asarray(s["pos0"]) for s in systems]))
    q_ps     = jnp.asarray(np.stack([np.asarray(s["q"]) for s in systems]))
    z0_ps    = np.array([s.get("init_z", 1.0) for s in systems])
    beta_ps  = jnp.asarray(np.array([s["beta"]  for s in systems]))
    z_min_ps = jnp.asarray(np.array([s["z_min"] for s in systems]))
    return dict(pos0_ps=pos0_ps, q_ps=q_ps, z0_ps=z0_ps,
                beta_ps=beta_ps, z_min_ps=z_min_ps)


def run_systems_agarose_surface(master_seed, systems, n_chains, n_steps,
                                sigma_rot, sigma_trans, axis_mask,
                                start_step=0, init=None):
    """Run S gel-coated surface systems × n_chains in one GPU batch."""
    S = len(systems)
    C = n_chains
    sids = [s["system_id"] for s in systems]

    b = _build_agarose_surface_batch_ps(systems)
    keys_sc = _chain_keys(master_seed, sids, C)
    if init is None:
        z0_sc = jnp.broadcast_to(jnp.asarray(b["z0_ps"])[:, None], (S, C))
        quat0_sc, trans0_sc = initial_state(keys_sc, z0_sc)
    else:
        quat0, trans0 = init
        quat0_sc = quat0.reshape(S, C, 4)
        trans0_sc = trans0.reshape(S, C, 3)

    U_ps   = jnp.asarray(np.stack([np.asarray(s["U_steric_field"]) for s in systems]))
    phi_ps = jnp.asarray(np.stack([np.asarray(s["phi_elec_field"]) for s in systems]))
    org_ps = jnp.asarray(np.stack([np.asarray(s["grid_origin"])    for s in systems]))
    spc_ps = jnp.asarray(np.stack([np.asarray(s["grid_spacing"])   for s in systems]))

    qf_sc, tf_sc, ef_sc, ar_sc = _ht_run_agarose_surface_ps(
        keys_sc, quat0_sc, trans0_sc, b["beta_ps"], b["pos0_ps"], b["q_ps"],
        b["z_min_ps"],
        float(sigma_rot), float(sigma_trans), jnp.asarray(axis_mask),
        int(start_step), int(n_steps),
        U_ps, phi_ps, org_ps, spc_ps)
    jax.block_until_ready((qf_sc, tf_sc, ef_sc, ar_sc))

    quats = np.asarray(qf_sc)
    transs = np.asarray(tf_sc)
    energies = np.asarray(ef_sc)
    accept = np.asarray(ar_sc)
    flat_state = (qf_sc.reshape(S * C, 4), tf_sc.reshape(S * C, 3))
    return dict(quats=quats, transs=transs, energies=energies, accept=accept,
                system_ids=sids, flat_state=flat_state)


# ===========================================================================
# Full-atom surface: precomputed (G12, G6, phi) grids + trilinear interpolation
# ===========================================================================

def _ht_chain_grid(chain_key, quat0, trans0, beta,
                   pos0, sqrt_c12, sqrt_c6, q,
                   sigma_rot, sigma_trans, axis_mask, start_step, n_steps,
                   G12_field, G6_field, phi_field,
                   grid_origin, grid_spacing, z_min, cell_Lx, cell_Ly):
    """One chain on a full-atom surface (trilinear grid interpolation)."""
    from ptmc.energy.grid_energy import grid_energy_jax

    def efn(quat, trans):
        return grid_energy_jax(quat, trans, pos0, sqrt_c12, sqrt_c6, q,
                               G12_field, G6_field, phi_field,
                               grid_origin, grid_spacing, z_min,
                               cell_Lx, cell_Ly)
    return scan_one(chain_key, quat0, trans0, efn, beta,
                    sigma_rot, sigma_trans, axis_mask, n_steps,
                    start_step=start_step)


# Inner vmap: C chains of one system; all per-system args to None axes.
_ht_grid_inner = jax.vmap(
    _ht_chain_grid,
    in_axes=(0, 0, 0,
             None, None, None, None, None,
             None, None, None, None, None,
             None, None, None,
             None, None, None, None, None))

# Outer vmap: S systems; per-system args to axis 0.
_ht_grid_outer = jax.vmap(
    _ht_grid_inner,
    in_axes=(0, 0, 0,
             0, 0, 0, 0, 0,
             None, None, None, None, None,
             0, 0, 0,
             0, 0, 0, 0, 0))


@partial(jax.jit, static_argnums=(12,))
def _ht_run_grid_ps(chain_keys, quat0, trans0, beta_ps,
                    pos0_ps, sqrt_c12_ps, sqrt_c6_ps, q_ps,
                    sigma_rot, sigma_trans, axis_mask, start_step, n_steps,
                    G12_ps, G6_ps, phi_ps,
                    origin_ps, spacing_ps, z_min_ps, Lx_ps, Ly_ps):
    """Double-vmap batch runner for full-atom grid energy."""
    return _ht_grid_outer(
        chain_keys, quat0, trans0, beta_ps,
        pos0_ps, sqrt_c12_ps, sqrt_c6_ps, q_ps,
        sigma_rot, sigma_trans, axis_mask, start_step, n_steps,
        G12_ps, G6_ps, phi_ps,
        origin_ps, spacing_ps, z_min_ps, Lx_ps, Ly_ps)


def run_systems_grid(master_seed, systems, n_chains, n_steps,
                     sigma_rot, sigma_trans, axis_mask,
                     start_step=0, init=None):
    """Run S full-atom surface systems x n_chains in one GPU batch.

    Each system dict must contain
    --------------------------------
    pos0       (N,3)  protein intrinsic coordinates (nm)
    c6, c12    (N,)   protein LJ coefficients (kJ/mol nm^6, nm^12)
    q          (N,)   protein atom charges (e)
    G12        (nx,ny,nz)  surface G12 field (precomputed by build_grids*)
    G6         (nx,ny,nz)  surface G6 field
    phi        (nx,ny,nz)  surface electrostatic potential (kJ/mol/e)
    grid_origin   (3,)  field grid origin (nm)
    grid_spacing  (3,)  grid spacing (nm)
    z_min      float   hard-wall distance (nm)
    cell_Lx    float   lateral PBC cell in x (nm); 0 = no wrapping
    cell_Ly    float   lateral PBC cell in y (nm); 0 = no wrapping
    beta       float   inverse temperature (mol/kJ)
    init_z     float   initial z for chains (nm)
    system_id  int
    """
    S = len(systems)
    C = n_chains
    sids = [s["system_id"] for s in systems]

    pos0_ps    = jnp.asarray(np.stack([np.asarray(s["pos0"]) for s in systems]))
    q_ps       = jnp.asarray(np.stack([np.asarray(s["q"])    for s in systems]))
    sc12_ps    = jnp.asarray(np.stack([np.sqrt(np.asarray(s["c12"])) for s in systems]))
    sc6_ps     = jnp.asarray(np.stack([np.sqrt(np.asarray(s["c6"]))  for s in systems]))
    beta_ps    = jnp.asarray(np.array([s["beta"]              for s in systems]))
    z_min_ps   = jnp.asarray(np.array([s["z_min"]             for s in systems]))
    Lx_ps      = jnp.asarray(np.array([s.get("cell_Lx", 0.0) for s in systems]))
    Ly_ps      = jnp.asarray(np.array([s.get("cell_Ly", 0.0) for s in systems]))
    z0_ps      = np.array([s.get("init_z", 0.95)              for s in systems])

    G12_ps     = jnp.asarray(np.stack([np.asarray(s["G12"])          for s in systems]))
    G6_ps      = jnp.asarray(np.stack([np.asarray(s["G6"])           for s in systems]))
    phi_ps     = jnp.asarray(np.stack([np.asarray(s["phi"])          for s in systems]))
    origin_ps  = jnp.asarray(np.stack([np.asarray(s["grid_origin"])  for s in systems]))
    spacing_ps = jnp.asarray(np.stack([np.asarray(s["grid_spacing"]) for s in systems]))

    keys_sc = _chain_keys(master_seed, sids, C)
    if init is None:
        z0_sc = jnp.broadcast_to(jnp.asarray(z0_ps)[:, None], (S, C))
        quat0_sc, trans0_sc = initial_state(keys_sc, z0_sc)
    else:
        quat0, trans0 = init
        quat0_sc = quat0.reshape(S, C, 4)
        trans0_sc = trans0.reshape(S, C, 3)

    qf_sc, tf_sc, ef_sc, ar_sc = _ht_run_grid_ps(
        keys_sc, quat0_sc, trans0_sc, beta_ps,
        pos0_ps, sc12_ps, sc6_ps, q_ps,
        float(sigma_rot), float(sigma_trans), jnp.asarray(axis_mask),
        int(start_step), int(n_steps),
        G12_ps, G6_ps, phi_ps,
        origin_ps, spacing_ps, z_min_ps, Lx_ps, Ly_ps)
    jax.block_until_ready((qf_sc, tf_sc, ef_sc, ar_sc))

    quats    = np.asarray(qf_sc)
    transs   = np.asarray(tf_sc)
    energies = np.asarray(ef_sc)
    accept   = np.asarray(ar_sc)
    flat_state = (qf_sc.reshape(S * C, 4), tf_sc.reshape(S * C, 3))
    return dict(quats=quats, transs=transs, energies=energies, accept=accept,
                system_ids=sids, flat_state=flat_state)
