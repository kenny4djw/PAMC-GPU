"""3D agarose hydrogel field builder: Voronoi fiber network + dual grids.

Two independent fields on a shared regular lattice:

  U_steric(x,y,z) = sum_i A * exp(-|r - r_i|^2 / 2 sigma^2)
  phi_elec(x,y,z) = sum_i delta_i * (f/eps_r) * q_i * exp(-|r - r_i|/lamD) / |r - r_i|

so that E_atom(r, q) = U_steric(r) + q * phi_elec(r).

U_steric is built by scattering node positions to a density grid and applying
a 3D Gaussian blur (scipy.ndimage.gaussian_filter).  phi_elec uses real-space
KD-tree summation over doped nodes only (cutoff = 3.5 * lambda_D).

Units (GROMACS): length nm, energy kJ/mol, charge e.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree, Voronoi

from ptmc.config import COULOMB_FACTOR_KJ_NM_PER_E2
from ptmc.log import logger


def generate_fiber_network(
    L: float, n_seeds: int, sigma: float,
    margin: float = 0.3, sample_density: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Poisson-Voronoi fiber network in a box of side L (nm).

    Seeds are placed in a larger box of side L * (1 + margin) to ensure the
    central L^3 region has a proper bulk-like network (no surface artifacts).
    Fiber nodes are densely sampled along Voronoi ridge edges at intervals
    of sample_density * sigma.

    Parameters
    ----------
    L : float
        Central box side length (nm).
    n_seeds : int
        Number of Voronoi seeds in the extended box.
    sigma : float
        Gaussian soft-core radius (nm).  Used to set edge sample density.
    margin : float
        Fractional extension beyond L to avoid boundary artifacts (0.3 = 30%).
    sample_density : float
        Node spacing along edges as fraction of sigma (0.5 = 2 nodes per sigma).
    seed : int
        Random seed.

    Returns
    -------
    nodes : (N, 3) ndarray
        Fiber node coordinates (nm) within the central box plus a sigma buffer.
    """
    rng = np.random.default_rng(seed)
    L_total = L * (1.0 + margin)
    half = L_total / 2.0

    seeds = rng.uniform(-half, half, (n_seeds, 3))
    vor = Voronoi(seeds)

    # Filter vertices: only those within or near the central box
    verts = vor.vertices
    v_ok = np.all(np.abs(verts) < L / 2.0 + 3.0 * sigma, axis=1)

    # Extract unique edges from ridge polygons  (3D Voronoi ridges are polygons)
    seen = set()
    edges = []
    for ridge in vor.ridge_vertices:
        n = len(ridge)
        for i in range(n):
            j = (i + 1) % n
            iv, jv = ridge[i], ridge[j]
            if iv < 0 or jv < 0:
                continue
            if not (v_ok[iv] or v_ok[jv]):
                continue
            key = (min(iv, jv), max(iv, jv))
            if key not in seen:
                seen.add(key)
                edges.append((iv, jv))

    # Sample points densely along each edge
    nodes = []
    step = sigma * sample_density
    for iv, jv in edges:
        p1, p2 = verts[iv], verts[jv]
        d = np.linalg.norm(p2 - p1)
        n = max(2, int(d / step) + 1)
        for t in np.linspace(0, 1, n):
            p = p1 + t * (p2 - p1)
            if np.all(np.abs(p) < L / 2.0 + 3.0 * sigma):
                nodes.append(p)

    # Always return (N, 3); empty cases get (0, 3) so downstream code can
    # safely index nodes[:, 2] / call KDTree without shape errors.
    if not nodes:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(nodes, dtype=np.float64)


def dope_nodes(
    nodes: np.ndarray,
    doping_frac: float,
    q_ligand: float,
    correlation: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Dope fiber nodes with charge, clustered along fiber.

    Implements the degree-of-substitution model: only a fraction of nodes carry
    charge (delta_i = 1).  Clustered doping (correlation > 1) creates contiguous
    charged patches along the fiber, matching the realistic random-walk grafting
    chemistry of ion-exchange resins.

    Parameters
    ----------
    nodes : (N, 3)
        Fiber node positions.
    doping_frac : float
        Fraction of nodes to dope (dimensionless, e.g. 0.05 = 5%).
    q_ligand : float
        Charge per doped node (e). +1 for Q-Sepharose, -1 for SP-Sepharose.
    correlation : int
        Number of consecutive nodes to dope per seed point (cluster size).
    seed : int
        Random seed.

    Returns
    -------
    q_doped : (N,) ndarray
        Effective charge per node (e).  Most entries are 0; doped entries = q_ligand.
    """
    rng = np.random.default_rng(seed)
    n = len(nodes)
    theta = np.zeros(n, dtype=np.float64)

    # Empty network: nothing to dope. Avoid rng.choice(0, ...) ValueError.
    if n == 0:
        return theta

    n_seeds = max(1, min(n, int(n * doping_frac / correlation)))
    # Clustered: pick seed nodes, then expand to neighbors along the array
    idx = rng.choice(n, n_seeds, replace=False)
    for i in idx:
        lo = max(0, i - correlation)
        hi = min(n, i + correlation + 1)
        theta[lo:hi] = q_ligand

    # Adjust to match exact target doping fraction
    target_charged = int(n * doping_frac)
    current_charged = int(np.count_nonzero(theta))
    if current_charged > target_charged:
        # Remove excess, preferring isolated (single) doped nodes
        excess = np.where(theta != 0)[0]
        rng.shuffle(excess)
        for i in excess[:current_charged - target_charged]:
            theta[i] = 0.0
    elif current_charged < target_charged:
        # Add more, preferring undoped nodes near existing clusters
        candidates = np.where(theta == 0)[0]
        rng.shuffle(candidates)
        for i in candidates[:target_charged - current_charged]:
            theta[i] = q_ligand

    return theta


def _accumulate_kernel(pts, source_nodes, neighbor_lists, kernel_fn,
                       kernel_args=()):
    """Vectorized accumulation over a ragged neighbor-list KDTree result.

    Flatten the ``query_ball_point`` ragged list into two flat arrays
    ``(point_idx, node_idx)`` of length ``sum(len(L))`` and call
    ``kernel_fn(r, *kernel_args)`` once on ALL pairs at once.  Per-point sums
    are then formed with ``np.bincount`` — no Python-level loop over grid
    points.

    Parameters
    ----------
    pts : (P, 3) ndarray  grid points being evaluated
    source_nodes : (N, 3) ndarray  the kdtree's nodes
    neighbor_lists : list[list[int]] of length P  output of query_ball_point
    kernel_fn : callable  r (M,) -> contribution (M,)
    kernel_args : tuple  extra args forwarded to kernel_fn

    Returns
    -------
    out : (P,) ndarray  per-grid-point summed contribution
    """
    counts = np.fromiter((len(L) for L in neighbor_lists), dtype=np.int64,
                          count=len(neighbor_lists))
    total = int(counts.sum())
    P = len(neighbor_lists)
    if total == 0:
        return np.zeros(P, dtype=np.float64)

    point_idx = np.repeat(np.arange(P, dtype=np.int64), counts)        # (total,)
    node_idx = np.concatenate([np.asarray(L, dtype=np.int64)
                                for L in neighbor_lists if L])         # (total,)
    dr = pts[point_idx] - source_nodes[node_idx]                       # (total, 3)
    r2 = np.einsum("ij,ij->i", dr, dr)
    r = np.sqrt(r2)
    contrib = kernel_fn(r2, r, node_idx, *kernel_args)                 # (total,)
    return np.bincount(point_idx, weights=contrib, minlength=P).astype(np.float64)


def build_agarose_grids(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
    nodes: np.ndarray, A: float, sigma: float,
    q_doped: np.ndarray, lambda_D: float,
    dielectric: float = 78.5,
    coulomb_factor: float = COULOMB_FACTOR_KJ_NM_PER_E2,
    floor_pctile: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build U_steric and phi_elec 3D grids from fiber nodes.

    U_steric : Gaussian filter of node density — fast, O(N log N).
    phi_elec : real-space KD-tree sum over doped nodes — O(N * n_nearby).

    Parameters
    ----------
    xs, ys, zs : (nx,), (ny,), (nz,)  coordinate arrays (nm)
    nodes : (N, 3)  fiber node positions
    A : float  steric amplitude (kJ/mol)
    sigma : float  Gaussian effective radius (nm)
    q_doped : (N,)  charge per node (e)
    lambda_D : float  Debye length (nm)
    dielectric : float  relative permittivity
    coulomb_factor : float  = 1/(4 pi eps0) in kJ nm / (mol e^2)
    floor_pctile : float
        Percentile of U_steric used as the reference zero (default 5.0).
        For a 3D gel every grid point sits in some fiber field, so the
        "vacuum" reference is not directly available — using the cleanest 5%
        of the grid as zero gives a well-defined pore reference that is
        insensitive to local node-density fluctuations. Pass 0.0 to disable
        the shift (raw values, vacuum-only reference like the surface model).

    Returns
    -------
    U_steric : (nx, ny, nz) ndarray  steric potential (kJ/mol)
    phi_elec : (nx, ny, nz) ndarray  electrostatic potential (kJ/(mol*e))
    origin : (3,)  grid origin
    spacing : (3,)  grid spacing
    """
    nx, ny, nz = len(xs), len(ys), len(zs)
    origin = np.array([xs[0], ys[0], zs[0]])
    spacing = np.array([xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]])

    # Degenerate input: an empty fiber network can't build a meaningful field.
    # Fail loudly so the caller knows to increase n_seeds or the box size.
    if nodes.shape[0] == 0:
        raise ValueError(
            "build_agarose_grids received an empty node array. The Voronoi "
            "fiber network produced 0 nodes inside the central box — increase "
            "n_seeds, enlarge L, or check that sigma is positive.")

    # =================================================================
    # KD-tree on ALL nodes, shared by steric and electrostatic sums.
    # =================================================================
    kdt = KDTree(nodes)
    doped_mask = q_doped != 0.0
    doped_nodes = nodes[doped_mask]
    doped_q = q_doped[doped_mask]
    n_doped = doped_nodes.shape[0]
    kdt_doped = KDTree(doped_nodes) if n_doped > 0 else None

    kappa = 1.0 / lambda_D
    prefac = coulomb_factor / dielectric
    cutoff_steric = 3.0 * sigma
    cutoff_elec = 3.5 * lambda_D

    U_steric = np.zeros((nx, ny, nz), dtype=np.float64)
    phi_elec = np.zeros((nx, ny, nz), dtype=np.float64)

    total_pts = nx * ny * nz
    chunk = max(1, min(100000, total_pts // 4))
    sigma2_inv = -0.5 / (sigma * sigma)  # used in exp(r^2 * (-1/2σ²))

    def steric_kernel(r2, r, _idx):
        return A * np.exp(r2 * sigma2_inv)

    def elec_kernel(r2, r, idx, charges):
        # Guard r=0 (grid point on a node) - mask zero distance contribution.
        safe = r > 0.0
        return np.where(safe, charges[idx] * np.exp(-kappa * r) / np.where(safe, r, 1.0),
                         0.0) * prefac

    for start in range(0, total_pts, chunk):
        end = min(start + chunk, total_pts)
        ii = np.arange(start, end)
        iz = ii % nz
        iy = (ii // nz) % ny
        ix = ii // (ny * nz)
        pts = np.column_stack([xs[ix], ys[iy], zs[iz]])

        # --- Steric: vectorized accumulation over ragged neighbor lists ---
        nearby_s = kdt.query_ball_point(pts, r=cutoff_steric)
        u_chunk = _accumulate_kernel(pts, nodes, nearby_s, steric_kernel)
        U_steric[ix, iy, iz] = u_chunk

        # --- Electrostatic: vectorized over doped nodes ---
        if kdt_doped is not None:
            nearby_e = kdt_doped.query_ball_point(pts, r=cutoff_elec)
            phi_chunk = _accumulate_kernel(pts, doped_nodes, nearby_e,
                                            elec_kernel, (doped_q,))
            phi_elec[ix, iy, iz] = phi_chunk

    # Normalize steric: set the floor_pctile-th percentile (clean pore volume)
    # to 0. Using a percentile rather than the global minimum ensures that even
    # with a dense fiber network where every grid point has some fiber
    # contribution, the reference zero corresponds to a genuine pore region.
    if floor_pctile > 0.0:
        floor = np.percentile(U_steric, floor_pctile)
        U_steric = np.maximum(U_steric - floor, 0.0)

    return U_steric, phi_elec, origin, spacing


def build_agarose_system_dict(
    atoms,
    gel,
    grid_spacing: float = 0.15,
    margin: float = 0.3,
    sample_density: float = 0.5,
    doping_correlation: int = 5,
    floor_pctile: float = 5.0,
    beta: float = 0.401,
) -> dict:
    """Build a system dict ready for run_systems_agarose.

    Generates the Voronoi fiber network, dopes charges, builds the two
    3D fields (U_steric + phi_elec), and returns a system dict.

    Parameters
    ----------
    atoms : Atoms  protein model
    gel : AgaroseGel  gel parameters
    grid_spacing : float  grid resolution in nm (default 0.15)
    margin : float  Voronoi box margin (default 0.3)
    beta : float  thermodynamic beta = 1/(kB T) in mol/kJ (default 0.401 for 300 K)

    Returns
    -------
    system : dict
    """
    L = gel.L
    box_origin = np.array([-L / 2.0, -L / 2.0, -L / 2.0])
    nx = ny = nz = max(2, int(np.ceil(L / grid_spacing)))
    xs = box_origin[0] + np.arange(nx) * grid_spacing
    ys = box_origin[1] + np.arange(ny) * grid_spacing
    zs = box_origin[2] + np.arange(nz) * grid_spacing

    # Generate fiber network
    logger.info("[agarose] Generating Voronoi fiber network: %d seeds, "
                "L=%.1f nm, sigma=%.2f nm",
                gel.n_seeds, L, gel.sigma)
    nodes = generate_fiber_network(
        L, gel.n_seeds, gel.sigma, margin=margin,
        sample_density=sample_density, seed=gel.seed)

    # Convert A from kcal/mol to kJ/mol if needed: A [kJ/mol] input assumed
    A_kj = gel.A

    # Dope charges
    q_doped = dope_nodes(
        nodes, gel.doping_frac, gel.q_ligand,
        correlation=doping_correlation, seed=gel.seed + 1)
    n_doped = np.count_nonzero(q_doped)
    logger.info("[agarose] %d fiber nodes, %d doped (%.1f%%)",
                len(nodes), n_doped,
                100.0 * n_doped / max(len(nodes), 1))

    # Build grids
    logger.info("[agarose] Building U_steric + phi_elec grids (%dx%dx%d)...",
                nx, ny, nz)
    U_steric, phi_elec, origin, spacing = build_agarose_grids(
        xs, ys, zs, nodes, A_kj, gel.sigma, q_doped,
        gel.lambda_D, gel.dielectric, floor_pctile=floor_pctile,
    )
    diagnostic = U_steric[nx // 2, ny // 2, nz // 2]
    logger.info("[agarose] U_steric range: [%.2f, %.2f] kJ/mol",
                float(U_steric.min()), float(U_steric.max()))
    if diagnostic < 1.0:
        logger.info("[agarose] U_steric at box center: %.2f kJ/mol (OK)",
                    diagnostic)
    else:
        logger.warning("[agarose] U_steric at box center: %.2f kJ/mol "
                       "(>1 kJ/mol — consider larger box or fewer seeds)",
                       diagnostic)
    logger.info("[agarose] phi_elec range: [%.4f, %.4f] kJ/(mol*e)",
                float(phi_elec.min()), float(phi_elec.max()))

    system = {
        "system_id": 0,
        "pos0": atoms.pos0,
        "q": atoms.q,
        "c6": atoms.c6,
        "c12": atoms.c12,
        "U_steric_field": U_steric,
        "phi_elec_field": phi_elec,
        "grid_origin": origin,
        "grid_spacing": spacing,
        "lambda_D": gel.lambda_D,
        "beta": beta,
        "init_z": 0.0,  # center of box
    }
    return system


def build_agarose_surface_grids(
    xs, ys, zs,
    L, thickness, n_seeds, sigma, A,
    doping_frac, q_ligand, lambda_D,
    dielectric=78.5,
    coulomb_factor=COULOMB_FACTOR_KJ_NM_PER_E2,
    seed=42,
    margin=0.3,
    sample_density=0.5,
    doping_correlation=5,
):
    """Build U_steric + phi_elec grids for a flat gel-coated surface.

    Fibers are generated in the coating slab z<0; the grid covers z>0.
    Uses the same physics as build_agarose_grids (Gaussian steric + Yukawa
    electrostatic), but the Voronoi seeds are placed in an asymmetric box
    z ∈ [-thickness * (1+margin), thickness * margin] so fibers are dense
    near the surface and decay into the bulk.

    Parameters
    ----------
    xs, ys, zs : (nx,), (ny,), (nz,)  coordinate arrays (nm), z>=0
    L : float  lateral box size (nm)
    thickness : float  coating thickness (nm)
    n_seeds : int  Voronoi seeds
    sigma : float  Gaussian soft-core radius (nm)
    A : float  steric amplitude (kJ/mol)
    doping_frac : float  fraction of nodes charged
    q_ligand : float  charge per doped node (e)
    lambda_D : float  Debye length (nm)
    dielectric : float  relative permittivity
    coulomb_factor : float  = 1/(4 pi eps0) in kJ nm / (mol e^2)
    seed : int  master seed

    Returns
    -------
    U_steric, phi_elec : (nx, ny, nz) ndarrays
    origin, spacing : (3,) arrays
    """
    nx, ny, nz = len(xs), len(ys), len(zs)
    origin = np.array([xs[0], ys[0], zs[0]])
    spacing = np.array([xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]])

    # Generate fibers in an asymmetric slab z<0.
    # Seeds are placed in a box extending into the coating (z<0) with a margin.
    L_total = L * (1.0 + margin)
    half = L_total / 2.0
    z_lo = -thickness * (1.0 + margin)
    z_hi = thickness * margin

    rng = np.random.default_rng(seed)
    seeds = np.column_stack([
        rng.uniform(-half, half, n_seeds),
        rng.uniform(-half, half, n_seeds),
        rng.uniform(z_lo, z_hi, n_seeds),
    ])

    from scipy.spatial import Voronoi
    vor = Voronoi(seeds)
    verts = vor.vertices
    v_ok = (
        (np.abs(verts[:, 0]) < half + 3.0 * sigma) &
        (np.abs(verts[:, 1]) < half + 3.0 * sigma) &
        (verts[:, 2] < z_hi + 3.0 * sigma) &
        (verts[:, 2] > z_lo - 3.0 * sigma)
    )

    seen = set()
    edges = []
    for ridge in vor.ridge_vertices:
        n = len(ridge)
        for i in range(n):
            j = (i + 1) % n
            iv, jv = ridge[i], ridge[j]
            if iv < 0 or jv < 0:
                continue
            if not (v_ok[iv] or v_ok[jv]):
                continue
            key = (min(iv, jv), max(iv, jv))
            if key not in seen:
                seen.add(key)
                edges.append((iv, jv))

    nodes = []
    step = sigma * sample_density
    for iv, jv in edges:
        p1, p2 = verts[iv], verts[jv]
        d = np.linalg.norm(p2 - p1)
        n_seg = max(2, int(d / step) + 1)
        for t in np.linspace(0, 1, n_seg):
            p = p1 + t * (p2 - p1)
            if (abs(p[0]) < half and abs(p[1]) < half and
                    z_lo < p[2] < z_hi):
                nodes.append(p)

    if not nodes:
        raise ValueError(
            "build_agarose_surface_grids produced 0 fiber nodes inside the "
            "coating slab. Increase n_seeds, enlarge L, or check that sigma "
            "and thickness are positive.")
    nodes = np.asarray(nodes, dtype=np.float64)

    # Shift nodes so the topmost fiber is at z=0 (clean surface).
    z_top = nodes[:, 2].max()
    if z_top > 0.0:
        nodes[:, 2] -= z_top
    logger.info("[agarose surface] %d fiber nodes in coating slab, "
                "z ∈ [%.1f, %.1f] nm",
                len(nodes), float(nodes[:, 2].min()), float(nodes[:, 2].max()))

    # Dope charges
    q_doped = dope_nodes(nodes, doping_frac, q_ligand,
                         correlation=doping_correlation, seed=seed + 1)
    n_doped = np.count_nonzero(q_doped)
    logger.info("[agarose surface] %d doped (%.1f%%)",
                n_doped, 100.0 * n_doped / max(len(nodes), 1))

    # KD-trees
    from scipy.spatial import KDTree
    kdt = KDTree(nodes)
    doped_mask = q_doped != 0.0
    kdt_doped = KDTree(nodes[doped_mask]) if doped_mask.any() else None
    doped_q = q_doped[doped_mask]

    kappa = 1.0 / lambda_D
    prefac = coulomb_factor / dielectric
    cutoff_steric = 3.0 * sigma
    cutoff_elec = 3.5 * lambda_D
    sigma2_inv = -0.5 / (sigma * sigma)

    U_steric = np.zeros((nx, ny, nz), dtype=np.float64)
    phi_elec = np.zeros((nx, ny, nz), dtype=np.float64)

    total_pts = nx * ny * nz
    chunk = max(1, min(50000, total_pts // 4))

    logger.info("[agarose surface] Building fields on %dx%dx%d grid "
                "(z ∈ [%.2f, %.2f] nm)...", nx, ny, nz, zs[0], zs[-1])

    # Pre-extract doped node positions once (consistent with build_agarose_grids).
    doped_pos = nodes[doped_mask]

    def steric_kernel(r2, r, _idx):
        return A * np.exp(r2 * sigma2_inv)

    def elec_kernel(r2, r, idx, charges):
        safe = r > 0.0
        return np.where(safe,
                         charges[idx] * np.exp(-kappa * r)
                         / np.where(safe, r, 1.0),
                         0.0) * prefac

    for start in range(0, total_pts, chunk):
        end = min(start + chunk, total_pts)
        ii = np.arange(start, end)
        iz = ii % nz
        iy = (ii // nz) % ny
        ix = ii // (ny * nz)
        pts = np.column_stack([xs[ix], ys[iy], zs[iz]])

        nearby_s = kdt.query_ball_point(pts, r=cutoff_steric)
        u_chunk = _accumulate_kernel(pts, nodes, nearby_s, steric_kernel)
        U_steric[ix, iy, iz] = u_chunk

        if kdt_doped is not None:
            nearby_e = kdt_doped.query_ball_point(pts, r=cutoff_elec)
            phi_chunk = _accumulate_kernel(pts, doped_pos, nearby_e,
                                            elec_kernel, (doped_q,))
            phi_elec[ix, iy, iz] = phi_chunk

    # Reference zero is vacuum (z→∞), no floor normalization needed.
    # Hard wall is enforced by agarose_surface_energy_jax, not the grid.
    # (For unified zero-point with build_agarose_grids, pass floor_pctile>0 to
    # that function; defaults differ because the 3D model has no vacuum region
    # while the surface model does.)

    z_mid = nz // 2
    diagnostic = U_steric[nx // 2, ny // 2, z_mid]
    logger.info("[agarose surface] U range [%.2f, %.2f]",
                float(U_steric.min()), float(U_steric.max()))
    if diagnostic < 20:
        logger.info("[agarose surface] U at mid-height (z=%.2f): %.2f (OK)",
                    float(zs[z_mid]), diagnostic)
    else:
        logger.warning("[agarose surface] U at mid-height (z=%.2f): %.2f "
                       "(high steric at mid-height)",
                       float(zs[z_mid]), diagnostic)
    logger.info("[agarose surface] phi range [%.4f, %.4f]",
                float(phi_elec.min()), float(phi_elec.max()))

    return U_steric, phi_elec, origin, spacing
