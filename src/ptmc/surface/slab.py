"""Slab cutting from a bulk crystal + surface termination."""
from __future__ import annotations
import numpy as np
import logging
from .lattice import Lattice

logger = logging.getLogger(__name__)


def _reciprocal(cell):
    return np.linalg.inv(cell.T) * 2 * np.pi

def _hkl_normal(hkl, cell):
    h, k, l = hkl; b = _reciprocal(cell)
    n = h*b[:,0] + k*b[:,1] + l*b[:,2]
    return n / np.linalg.norm(n)

def _plane_distance(hkl, cell):
    h, k, l = hkl; b = _reciprocal(cell)
    G = h*b[:,0] + k*b[:,1] + l*b[:,2]
    return float(2 * np.pi / np.linalg.norm(G))


def cut_slab(lattice: Lattice, hkl=(0,0,1), n_layers: int = 4,
             vacuum: float = 3.0) -> Lattice:
    """Cut a slab from a crystal lattice along Miller indices (hkl)."""
    cell = lattice.cell.copy()
    cart = lattice.frac_to_cart(lattice.frac_pos)
    n = _hkl_normal(hkl, cell)
    d_hkl = _plane_distance(hkl, cell)

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    ex = np.cross(n, ref); ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex);  ey /= np.linalg.norm(ey)
    R = np.column_stack([ex, ey, n])

    proj = cart @ n
    thickness = n_layers * d_hkl
    z_min_slab = proj.min()

    mask0 = (proj >= z_min_slab) & (proj < z_min_slab + thickness)
    xy0 = cart[mask0] @ np.column_stack([ex, ey])
    in_plane_extent = xy0.max(axis=0) - xy0.min(axis=0)
    ip_cell = np.array([[np.linalg.norm(cell @ ex), 0.0],
                        [0.0, np.linalg.norm(cell @ ey)]])
    rep_x = max(1, int(np.ceil(1.5 * in_plane_extent[0] / max(ip_cell[0,0], 1e-10))))
    rep_y = max(1, int(np.ceil(1.5 * in_plane_extent[1] / max(ip_cell[1,1], 1e-10))))

    sup = lattice.replicate(rep_x, rep_y, 2)
    sc = sup.frac_to_cart(sup.frac_pos)
    sp = sc @ n
    mask = (sp >= z_min_slab) & (sp < z_min_slab + thickness)
    slab_cart = sc[mask]
    slab_species = [s for s, m in zip(sup.species, mask) if m]

    slab_rot = slab_cart @ R
    slab_rot[:, 2] -= slab_rot[:, 2].min()
    # The in-plane cell is forced to a diagonal (orthorhombic) box.
    # For non-orthogonal surface lattices this approximates the true
    # periodicity; check how much of the in-plane cell is being discarded.
    off_diag = abs(np.dot(cell @ ex, ey))
    diag_avg = 0.5 * (abs(np.dot(cell @ ex, ex)) + abs(np.dot(cell @ ey, ey)))
    if off_diag > 0.05 * diag_avg + 1e-10:
        logger.warning(
            "Slab in-plane cell is significantly non-orthogonal "
            "(off-diagonal / avg-diagonal = %.3f). "
            "The output cell is diagonal (orthorhombic), which loses the "
            "true 2D periodicity for this surface orientation.",
            off_diag / max(diag_avg, 1e-10))
    new_cell = np.array([
        [ip_cell[0,0] * rep_x, 0.0, 0.0],
        [0.0, ip_cell[1,1] * rep_y, 0.0],
        [0.0, 0.0, thickness + vacuum],
    ])
    return Lattice(cell=new_cell, species=slab_species,
                   frac_pos=slab_rot @ np.linalg.inv(new_cell))


def hydroxylate_rutile(slab: Lattice, oh_bond: float = 0.180,
                        oh_z_cutoff: float = 0.25) -> Lattice:
    """Add hydroxyl groups to a rutile TiO₂ slab (110) surface."""
    cart = slab.frac_to_cart(slab.frac_pos)
    new_species = list(slab.species); new_cart = list(cart)
    z_top = cart[:, 2].max()
    for i, (spec, pos) in enumerate(zip(slab.species, cart)):
        if pos[2] <= z_top - oh_z_cutoff:
            continue
        if spec == "Ob_ti":
            new_species[i] = "OH"
            new_species.append("H_oh")
            new_cart.append(pos + np.array([0.0, 0.0, 0.096]))
        elif spec == "Ti":
            new_species[i] = "Ti_surf"
            oh_pos = pos + np.array([0.0, 0.0, oh_bond])
            new_species += ["OH", "H_oh"]
            new_cart += [oh_pos, oh_pos + np.array([0.0, 0.0, 0.096])]
    if len(new_species) == len(slab.species):
        return slab
    arr = np.array(new_cart)
    return Lattice(cell=slab.cell.copy(), species=new_species,
                   frac_pos=arr @ np.linalg.inv(slab.cell))


def hydroxylate_quartz(slab: Lattice, oh_z_cutoff: float = 0.35) -> Lattice:
    """Add hydroxyl groups to a quartz (SiO₂) slab surface."""
    cart = slab.frac_to_cart(slab.frac_pos)
    thickness = slab.cell[2, 2]
    new_species = list(slab.species); new_cart = list(cart); oh_count = 0
    for i, (spec, pos) in enumerate(zip(slab.species, cart)):
        if spec != "Ob":
            continue
        if not ((pos[2] < oh_z_cutoff) or (pos[2] > thickness - oh_z_cutoff)):
            continue
        new_species[i] = "OH"
        outward = 1.0 if pos[2] < thickness / 2 else -1.0
        new_species.append("H_oh")
        new_cart.append(pos + np.array([0.0, 0.0, outward * 0.096]))
        oh_count += 1
    if oh_count == 0:
        return slab
    arr = np.array(new_cart)
    return Lattice(cell=slab.cell.copy(), species=new_species,
                   frac_pos=arr @ np.linalg.inv(slab.cell))
