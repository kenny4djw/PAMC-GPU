"""Symmetric MC moves on rigid-body pose (quaternion + translation), in JAX."""
from __future__ import annotations
import jax
import jax.numpy as jnp

def quat_mul(a, b):
    w1,x1,y1,z1 = a[...,0],a[...,1],a[...,2],a[...,3]
    w2,x2,y2,z2 = b[...,0],b[...,1],b[...,2],b[...,3]
    return jnp.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=-1)

def normalize_quat(q):
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)

def axisangle_to_quat(omega):
    angle = jnp.linalg.norm(omega, axis=-1, keepdims=True)
    safe = jnp.where(angle < 1e-12, 1.0, angle)
    axis = omega / safe
    half = 0.5 * angle
    return jnp.concatenate([jnp.cos(half), axis*jnp.sin(half)], axis=-1)

def quat_rotate(q, v):
    w = q[...,0:1]; u = q[...,1:4]
    uv = jnp.cross(u, v)
    return v + 2.0*(w*uv + jnp.cross(u, uv))

def propose_rotation(key, quat, sigma_rot):
    """Symmetric small rotation q' = normalize(dq(omega)*q), omega~N(0,sigma^2 I_3).

    Output dtype matches ``quat.dtype`` so the carry through ``lax.scan``
    stays type-stable even when jax_enable_x64 is on globally (the default
    ``jax.random.normal`` dtype is float64 in that mode).
    """
    omega = sigma_rot * jax.random.normal(key, (3,), dtype=quat.dtype)
    return normalize_quat(quat_mul(axisangle_to_quat(omega), quat))

def propose_translation(key, trans, sigma_trans, axis_mask):
    """Symmetric Gaussian translation on masked axes. Output dtype = trans.dtype."""
    return trans + sigma_trans * jax.random.normal(key, (3,), dtype=trans.dtype) * axis_mask


def propose_chi(key, chi, sigma_chi):
    """Symmetric Gaussian perturbation of a single chi component.

    Picks k ~ Uniform({0, ..., K-1}) and proposes chi'[k] = chi[k] + N(0, sigma_chi^2);
    all other components are unchanged. The detailed-balance proposal is
    symmetric, so no Hastings correction is needed.

    Returns (new_chi, k_index). When K == 0 (no chi DOFs), the original chi
    is returned and k_index is 0 (sentinel): callers should mask the chi
    move out via move_weights so this branch never selects.
    """
    K = chi.shape[0]
    if K == 0:
        return chi, jnp.zeros((), dtype=jnp.int32)
    k_idx_key, k_d_key = jax.random.split(key, 2)
    k = jax.random.randint(k_idx_key, (), 0, K)
    dchi = sigma_chi * jax.random.normal(k_d_key, (), dtype=chi.dtype)
    new_chi = chi.at[k].add(dchi)
    return new_chi, k
