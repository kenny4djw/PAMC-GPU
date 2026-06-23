"""Global configuration: units, physical constants, device, precision, RNG.

Unit convention (GROMACS):
    length       nm
    energy       kJ / mol
    charge       e (elementary charge)
    temperature  K
    mass         amu

All physical constants below are expressed in these units so that every
formula in the codebase is unit-consistent without ad-hoc conversion factors.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import jax

# ---------------------------------------------------------------------------
# Physical constants in GROMACS units.
#   BOLTZMANN k_B : R = 8.314462618 J mol^-1 K^-1 = 8.314462618e-3 kJ mol^-1 K^-1
#   COULOMB factor f = 1/(4 pi eps0) = 138.935458 kJ mol^-1 nm e^-2
#       (GROMACS ONE_4PI_EPS0; see GROMACS reference manual, "Coulomb interaction").
# Coulomb energy (vacuum) between charges q_i, q_j (in e) at distance r (nm):
#       E = f * q_i q_j / r            [kJ/mol]
# Screened (Debye-Hueckel) Coulomb adds a factor exp(-kappa r):
#       E = f * q_i q_j / r * exp(-r / lambda_D)
# with Debye length lambda_D (nm). NOTE: vacuum Coulomb is FORBIDDEN for the
# physical model (see architecture); electrostatics MUST be screened.
# ---------------------------------------------------------------------------
BOLTZMANN_KJ_PER_MOL_K: float = 8.314462618e-3
COULOMB_FACTOR_KJ_NM_PER_E2: float = 138.935458

# Default solvent / electrolyte parameters for the screened Coulomb model.
# Debye length (1:1 electrolyte, 25 C water):  lambda_D[nm] = 0.304 / sqrt(I[M])
#   (Israelachvili, "Intermolecular and Surface Forces"). 0.15 M -> ~0.785 nm.
DEFAULT_IONIC_STRENGTH_M: float = 0.15

# Relative permittivity of liquid water at 25 C (~78.5). Used by the patterned
# continuum surface model (FFT-based 2D convolution electrostatics).
WATER_DIELECTRIC: float = 78.5


def beta(temperature_K: float) -> float:
    """Inverse temperature 1/(k_B T) in (kJ/mol)^-1. SHAPE: scalar."""
    return 1.0 / (BOLTZMANN_KJ_PER_MOL_K * float(temperature_K))


def debye_length_nm(ionic_strength_M: float = DEFAULT_IONIC_STRENGTH_M) -> float:
    """Debye screening length (nm) for a 1:1 electrolyte in water at 25 C.

    lambda_D = 0.304 / sqrt(I)  with I in mol/L (Israelachvili). SHAPE: scalar.
    """
    import math
    return 0.304 / math.sqrt(float(ionic_strength_M))


# ---------------------------------------------------------------------------
# Device handling: prefer GPU, fall back to CPU without error.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeviceInfo:
    backend: str          # "gpu" or "cpu"
    n_devices: int
    device_repr: str
    on_gpu: bool


def device_info() -> DeviceInfo:
    """Detect the active JAX backend; never raises if no GPU is present."""
    devices = jax.devices()
    on_gpu = any(d.platform == "gpu" for d in devices)
    backend = "gpu" if on_gpu else "cpu"
    return DeviceInfo(
        backend=backend,
        n_devices=len(devices),
        device_repr=", ".join(str(d) for d in devices),
        on_gpu=on_gpu,
    )


def print_device_info() -> DeviceInfo:
    from ptmc.log import logger
    info = device_info()
    logger.info("JAX backend = %s | %d device(s): %s",
                info.backend, info.n_devices, info.device_repr)
    return info


# ---------------------------------------------------------------------------
# Reproducibility: a single entry point for PRNG keys (JAX) so that every
# stochastic component is seedable from one integer seed.
# ---------------------------------------------------------------------------
def make_key(seed: int) -> jax.Array:
    """Return a JAX PRNGKey from an integer seed. SHAPE: (2,) uint32."""
    return jax.random.PRNGKey(int(seed))


# Throughput threshold for the GPU benchmark (stage 4). Configurable via env so
# CPU fallback can use a lower bar. Units: configurations / second.
THROUGHPUT_TARGET_CONFIGS_PER_SEC: float = float(
    os.environ.get("PTMC_THROUGHPUT_TARGET", "1.0e6")
)

# RNG seed offset for initial-quaternion generation, decoupled from the per-chain
# keys so that varying chain count / system count does not change the initial
# pose distribution. Arbitrary large constant; any large integer works.
INIT_QUAT_KEY_OFFSET: int = 2_000_000_000

# Maximum floating-point elements per chunk in grid-build distance computations.
# Chunk size = max(1, GRID_BUILD_CHUNK_ELEMS // max(surface_atoms, 1)).
# Keeps peak memory bounded regardless of resolution. Tune via env if needed.
GRID_BUILD_CHUNK_ELEMS: int = int(
    os.environ.get("PTMC_GRID_BUILD_CHUNK_ELEMS", "4_000_000")
)


# ---------------------------------------------------------------------------
# Structured simulation configuration (dataclasses replace loose dicts/floats).
# ---------------------------------------------------------------------------
@dataclass
class SurfaceConfig:
    """Homogeneous continuum surface (Steele 9-3 vdW + linearized-PB elec).

    Attributes
    ----------
    rho_s : float
        Surface atom number density (nm^-3). Default 30.0 (~graphite).
    c6_surf : float
        Surface LJ C6 coefficient (kJ/mol nm^6). Default 1.0.
    c12_surf : float
        Surface LJ C12 coefficient (kJ/mol nm^12). Default 1.0.
    lambda_D : float
        Debye screening length (nm). Default 0.785 (~0.15 M NaCl).
    z_min : float
        Hard-wall repulsion distance (nm). Default 0.15 (matches the
        continuum/patterned CLI ``--z-min`` default; the per-surface
        crystal/full_atom/agarose hard walls default to 0.2).
    psi0 : float
        Surface electrostatic potential at z=0 (kJ/mol/e). Default 0.0.
    eps_surf : float | None
        Surface LJ ε (kJ/mol). When provided (not None), overrides c6/c12_surf
        for the ε-σ Steele 9-3 path. Default None (use C6/C12 legacy path).
    sigma_surf : float | None
        Surface LJ σ (nm). Paired with eps_surf. Default None.
    """
    rho_s: float = 30.0
    c6_surf: float = 1.0
    c12_surf: float = 1.0
    lambda_D: float = field(default_factory=debye_length_nm)
    z_min: float = 0.15
    psi0: float = 0.0
    eps_surf: float | None = None
    sigma_surf: float | None = None

    def __post_init__(self) -> None:
        assert self.rho_s > 0, f"rho_s must be > 0, got {self.rho_s}"
        assert self.lambda_D > 0, f"lambda_D must be > 0, got {self.lambda_D}"
        assert self.z_min >= 0, f"z_min must be >= 0, got {self.z_min}"
        if (self.eps_surf is None) != (self.sigma_surf is None):
            raise ValueError(
                "eps_surf and sigma_surf must both be None (use C6/C12 path) "
                "or both set (use epsilon-sigma LB path).")


@dataclass
class MCConfig:
    """Monte Carlo sampling parameters for the PA / PT drivers.

    Attributes
    ----------
    n_chains : int
        Per-system MC chain count. Retained for legacy callers of the batched
        runners in ``ptmc.sampler.highthroughput`` (these are still used by
        the test suite); the user-facing PA / PT pipeline ignores it. Default 64.
    n_steps : int
        Number of MC sweeps per chain / walker. Default 10000.
    sigma_rot : float
        Rotation proposal stddev (rad). Default 0.1.
    sigma_trans : float
        Translation proposal stddev (nm). Default 0.05.
    axis_mask : tuple
        Allowed translation axes (x, y, z). Default (False, False, True).
    seed : int
        Master PRNG seed for reproducibility. Default 42.
    temperature : float
        Simulation temperature (K). Default 300.0.
    init_z : float
        Initial protein z-height (nm). Default 0.0 = auto-compute a safe value
        (R_protein + z_min + 3·sigma_trans). A positive value overrides the
        auto height; if it is below the safe minimum a warning is emitted and
        the safe value is used instead. Applies to surface models (the 3D
        agarose gel always initialises at the box centre).
    flexible : bool
        EXPERIMENTAL — Enable per-residue side-chain χ dihedrals in addition
        to rigid-body rotation + translation. Default False (rigid body).
        The χ-aware MC kernel exists (``ptmc.mc.flex_metropolis``,
        ``ptmc.sampler.flexible_run``) but: (1) the combined surface+intra
        energy closure and chi-aware PA / PT drivers are not yet wired
        through ``run_pipeline``; and (2) the intra-protein non-bonded path
        (``ptmc.flexible.intra_nb``) combines per-atom C6/C12 *geometrically*,
        which is inconsistent with the Lorentz–Berthelot rule used by the
        amber99sb-ildn topology — quantitative results from the flexible
        path are not validated and should not be reported without an LB
        upgrade to ``intra_nb``. See SI §S1.2 of the manuscript for the full
        disclosure. The validated rigid-body §S4–§S5 results are unaffected.
    sigma_chi : float
        χ-dihedral proposal stddev (rad) when ``flexible`` is True. Default 0.2.
    flexible_ack_experimental : bool
        Required acknowledgement for the experimental chi-DOF path. Setting
        ``flexible=True`` without also setting this to True raises
        ``NotImplementedError`` — quantitative free energies and orientations
        from the flexible path are not validated against literature and must
        not be reported as such (see SI §S1.2). This gate is intentional and
        independent of any warning filter.
    """
    n_chains: int = 64
    n_steps: int = 10000
    sigma_rot: float = 0.1
    sigma_trans: float = 0.05
    axis_mask: tuple = (False, False, True)
    seed: int = 42
    temperature: float = 300.0
    init_z: float = 0.0
    flexible: bool = False
    sigma_chi: float = 0.2
    flexible_ack_experimental: bool = False

    def __post_init__(self) -> None:
        assert self.n_chains >= 1, f"n_chains must be >= 1, got {self.n_chains}"
        assert self.n_steps >= 1, f"n_steps must be >= 1, got {self.n_steps}"
        assert self.sigma_rot >= 0, f"sigma_rot must be >= 0, got {self.sigma_rot}"
        assert self.sigma_trans >= 0, f"sigma_trans must be >= 0, got {self.sigma_trans}"
        assert self.temperature > 0, f"temperature must be > 0, got {self.temperature}"
        assert self.init_z >= 0, f"init_z must be >= 0 (0 = auto), got {self.init_z}"
        assert len(self.axis_mask) == 3, f"axis_mask must have 3 elements, got {len(self.axis_mask)}"
        assert self.sigma_chi >= 0, f"sigma_chi must be >= 0, got {self.sigma_chi}"
        if self.flexible and not self.flexible_ack_experimental:
            raise NotImplementedError(
                "MCConfig.flexible=True is an EXPERIMENTAL chi-DOF kernel. "
                "The intra-NB path uses geometric combining (rule 1/3), "
                "inconsistent with the amber99sb-ildn Lorentz-Berthelot "
                "topology (rule 2); quantitative DG_ads / orientation "
                "results are not validated against literature and must not "
                "be reported as such. To use this path for development, "
                "set MCConfig(flexible=True, flexible_ack_experimental=True) "
                "explicitly. See manuscript SI for details."
            )
        if self.flexible:
            import warnings
            warnings.warn(
                "MCConfig.flexible=True enables the EXPERIMENTAL chi-DOF "
                "kernel: intra-NB uses geometric combining, inconsistent "
                "with the amber99sb-ildn LB topology. Results are not "
                "validated and must not be quoted.",
                stacklevel=2,
            )


@dataclass
class PTConfig:
    """Parallel Tempering configuration.

    n_replicas : int
        Number of temperature replicas in the ladder. Default 8.
    T_min : float
        Lowest replica temperature (K). Default 200.0.
    T_max : float
        Highest replica temperature (K). Default 600.0.
    n_rounds : int
        PT rounds (one MC sweep + one replica-exchange pass). Default 200.
    n_sweep : int
        MC sweeps per round per replica. Default 50.
    """
    n_replicas: int = 8
    T_min: float = 200.0
    T_max: float = 600.0
    n_rounds: int = 200
    n_sweep: int = 50

    def __post_init__(self) -> None:
        assert self.n_replicas >= 2, f"n_replicas must be >= 2, got {self.n_replicas}"
        assert self.T_max > self.T_min > 0, f"require T_max > T_min > 0, got T_min={self.T_min}, T_max={self.T_max}"
        assert self.n_rounds >= 1, f"n_rounds must be >= 1, got {self.n_rounds}"
        assert self.n_sweep >= 1, f"n_sweep must be >= 1, got {self.n_sweep}"


@dataclass
class PAConfig:
    """Population Annealing configuration.

    n_walkers : int
        Population size (number of walkers). Default 512.
    T_start : float
        Starting (hot) temperature (K). Default 5000.0. Raised from 1000 K
        so that ⟨|β₀·E|⟩ ≪ 1 for typical kJ/mol-scale adsorption energies —
        a prerequisite for interpreting ``logZ_ratio`` as ΔG of going from
        a uniform-in-slab reference to the canonical distribution at
        ``mc.temperature``. The β₀-adequacy check in ``_run_pa`` warns if
        the initial walkers still see significant β·E here.
    target_ess : float
        Minimum ESS fraction for adaptive cooling bisection. Default 0.7.
    max_annealing_steps : int
        Maximum number of cooling schedule steps. Default 400.
    z_max : float
        Upper z (nm) of the slab covered by the PA walker ensemble. Used
        to define the reference state for ``adsorption_free_energy``.
        Set to 0 (default) to auto-detect from the surface type:
          * crystal / full_atom : the precomputed field's ``z_grid_max``
          * continuum / patterned : safe lower z + 3.0 nm (covers typical
            screened-Coulomb + Steele 9-3 attractive range; raise to
            check bulk-baseline convergence of K_ads).
    c_std : float
        Standard reference concentration for ΔG⁰_ads (mol/L). Default 1.0.
    """
    n_walkers: int = 512
    T_start: float = 5000.0
    target_ess: float = 0.7
    max_annealing_steps: int = 400
    z_max: float = 0.0
    c_std: float = 1.0

    def __post_init__(self) -> None:
        assert self.n_walkers >= 2, f"n_walkers must be >= 2, got {self.n_walkers}"
        assert self.T_start > 0, f"T_start must be > 0, got {self.T_start}"
        assert 0 < self.target_ess <= 1, f"target_ess must be in (0, 1], got {self.target_ess}"
        assert self.max_annealing_steps >= 1, f"max_annealing_steps must be >= 1, got {self.max_annealing_steps}"
        assert self.z_max >= 0, f"z_max must be >= 0 (0 = auto), got {self.z_max}"
        assert self.c_std > 0, f"c_std must be > 0 mol/L, got {self.c_std}"


@dataclass
class AgaroseConfig:
    """Agarose hydrogel / gel-coated surface parameters.

    L : float
        Box (3D gel) or lateral (surface coating) size in nm. Default 20.0.
    thickness : float
        Coating thickness for agarose_surface model (nm). Default 10.0.
    n_seeds : int
        Number of Voronoi seeds for fiber network generation. Default 50.
    sigma : float
        Gaussian soft-core radius (nm). Default 2.0.
    A : float
        Steric repulsion amplitude (kJ/mol). Default 500.0.
    doping_frac : float
        Fraction of fiber nodes carrying ligand charge [0,1]. Default 0.1.
    q_ligand : float
        Charge per doped node (e). +1 = cation exchanger, -1 = anion. Default -1.0.
    lambda_D : float
        Debye screening length (nm). Default 3.0 (~10 mM salt).
    dielectric : float
        Solvent relative permittivity. Default 78.5.
    gel_seed : int
        Master seed for Voronoi + doping. Default 42.
    grid_spacing : float
        Field grid resolution (nm). Default 0.15.
    z_min : float
        Hard-wall distance from surface, agarose_surface only (nm). Default 0.2.
    margin : float
        Fractional box extension for the Voronoi seed box, to avoid boundary
        artifacts in the central region (0.3 = 30%). Default 0.3.
    sample_density : float
        Fiber-node spacing along Voronoi edges as a fraction of sigma
        (0.5 = 2 nodes per sigma). Smaller = denser sampling. Default 0.5.
    doping_correlation : int
        Number of consecutive nodes doped per seed point (clustered grafting
        patch size along the fiber). Default 5.
    floor_pctile : float
        Percentile of U_steric used as the zero reference for the 3D gel
        (pore-volume reference); 0 disables the shift. Default 5.0. Unused by
        the agarose_surface model (vacuum reference).
    """
    L: float = 20.0
    thickness: float = 10.0
    n_seeds: int = 50
    sigma: float = 2.0
    A: float = 500.0
    doping_frac: float = 0.1
    q_ligand: float = -1.0
    lambda_D: float = 3.0
    dielectric: float = 78.5
    gel_seed: int = 42
    grid_spacing: float = 0.15
    z_min: float = 0.2
    margin: float = 0.3
    sample_density: float = 0.5
    doping_correlation: int = 5
    floor_pctile: float = 5.0

    def __post_init__(self) -> None:
        assert self.L > 0, f"L must be > 0, got {self.L}"
        assert self.sigma > 0, f"sigma must be > 0, got {self.sigma}"
        assert self.A >= 0, f"A must be >= 0, got {self.A}"
        assert 0 <= self.doping_frac <= 1, f"doping_frac must be in [0, 1], got {self.doping_frac}"
        assert self.lambda_D > 0, f"lambda_D must be > 0, got {self.lambda_D}"
        assert self.z_min >= 0, f"z_min must be >= 0, got {self.z_min}"
        assert self.grid_spacing > 0, f"grid_spacing must be > 0, got {self.grid_spacing}"
        assert self.margin >= 0, f"margin must be >= 0, got {self.margin}"
        assert self.sample_density > 0, f"sample_density must be > 0, got {self.sample_density}"
        assert self.doping_correlation >= 1, f"doping_correlation must be >= 1, got {self.doping_correlation}"
        assert 0 <= self.floor_pctile < 100, f"floor_pctile must be in [0, 100), got {self.floor_pctile}"


@dataclass
class CrystalConfig:
    """Built-in crystal full-atom surface parameters.

    crystal : str
        Crystal name: alpha_quartz | rutile_tio2 | anatase_tio2 | gold_fcc.
    hkl : tuple
        Miller indices of the surface plane. Default (0, 0, 1).
    n_layers : int
        Number of crystal unit-cell layers in the slab. Default 4.
    hydroxylate : bool
        Add -OH termination to silica / rutile surfaces. Default True.
    r_cut : float
        PBC interaction cutoff radius for grid building (nm). Default 1.4.
    grid_spacing : float
        Field grid resolution (nm). Default 0.05.
    lambda_D : float
        Debye screening length (nm). Default 0.785.
    z_min : float
        Hard-wall distance from topmost surface atom (nm). Default 0.2.
    vacuum : float
        Vacuum gap above slab for slab cutting (nm). Default 3.0.
    z_grid_max : float
        Absolute upper z (nm) of the field grid above the surface plane (z=0).
        Grid covers z ∈ [z_min, z_grid_max]. Must exceed z_min + R_protein.
        Default 3.2 (covers protein radius ≲ 3 nm above the 0.2 nm hard wall).
    """
    crystal: str = "alpha_quartz"
    hkl: tuple = (0, 0, 1)
    n_layers: int = 4
    hydroxylate: bool = True
    r_cut: float = 1.4
    grid_spacing: float = 0.05
    lambda_D: float = 0.785
    z_min: float = 0.2
    vacuum: float = 3.0
    z_grid_max: float = 3.2

    def __post_init__(self) -> None:
        assert self.grid_spacing > 0, f"grid_spacing must be > 0, got {self.grid_spacing}"
        assert self.lambda_D > 0, f"lambda_D must be > 0, got {self.lambda_D}"
        assert self.z_min >= 0, f"z_min must be >= 0, got {self.z_min}"
        assert self.n_layers >= 1, f"n_layers must be >= 1, got {self.n_layers}"
        assert self.r_cut > 0, f"r_cut must be > 0, got {self.r_cut}"
        assert self.vacuum >= 0, f"vacuum must be >= 0, got {self.vacuum}"
        assert self.z_grid_max > self.z_min, (
            f"z_grid_max ({self.z_grid_max}) must exceed z_min ({self.z_min})")


@dataclass
class FullAtomConfig:
    """External full-atom surface: PDB coordinates + force field parameters.

    Two force field import paths are supported (mutually exclusive):
      JSON  --  provide surface_ff_json; atom names in PDB are looked up
                directly in the JSON dict (no parmed/GROMACS needed).
      GROMACS -- provide surface_top; uses the same parmed-based parser
                as the protein path (handles comb-rules 1/2/3, #includes).

    surface_pdb : str
        Path to PDB file with surface atom coordinates (Å, ATOM/HETATM).
    surface_top : str
        GROMACS .top file for the surface (parmed path).
    surface_ff_json : str
        JSON force field file (json path).  Format::

            {
              "_units": "epsilon kJ/mol, sigma nm, q e",
              "Au":  {"q": 0.0,   "sigma": 0.254, "epsilon": 2.483},
              "Si":  {"q": 2.10,  "sigma": 0.330, "epsilon": 7.7e-6}
            }

    cell_xy : tuple
        Lateral periodic cell (Lx, Ly) in nm. Required.
    r_cut : float
        PBC cutoff for grid building (nm). Default 1.4.
    grid_spacing : float
        Field grid resolution (nm). Default 0.05.
    lambda_D : float
        Debye screening length (nm). Default 0.785.
    z_min : float
        Hard-wall distance (nm). Default 0.2.
    z_grid_max : float
        Absolute upper z (nm) of the field grid above the surface plane (z=0).
        Grid covers z ∈ [z_min, z_grid_max]. Must exceed z_min + R_protein.
        Default 3.2 (covers protein radius ≲ 3 nm above the 0.2 nm hard wall).
    """
    surface_pdb: str = ""
    surface_top: str = ""
    surface_ff_json: str = ""
    cell_xy: tuple = (3.0, 3.0)
    r_cut: float = 1.4
    grid_spacing: float = 0.05
    lambda_D: float = 0.785
    z_min: float = 0.2
    z_grid_max: float = 3.2

    def __post_init__(self) -> None:
        assert self.grid_spacing > 0, f"grid_spacing must be > 0, got {self.grid_spacing}"
        assert self.lambda_D > 0, f"lambda_D must be > 0, got {self.lambda_D}"
        assert self.z_min >= 0, f"z_min must be >= 0, got {self.z_min}"
        assert self.r_cut > 0, f"r_cut must be > 0, got {self.r_cut}"
        assert self.cell_xy[0] > 0 and self.cell_xy[1] > 0, f"cell_xy must be positive, got {self.cell_xy}"
        assert self.z_grid_max > self.z_min, (
            f"z_grid_max ({self.z_grid_max}) must exceed z_min ({self.z_min})")


@dataclass
class SimConfig:
    """Top-level simulation configuration.

    surface_type : str
        Surface potential model: continuum | patterned | agarose | agarose_surface.
    sampler : str
        Sampling algorithm: pa (population annealing, default) | pt (parallel tempering).
    surface : SurfaceConfig
        Continuum / patterned surface parameters.
    mc : MCConfig
        Monte Carlo move and chain parameters.
    pt : PTConfig
        Parallel Tempering parameters (only used when sampler='pt').
    pa : PAConfig
        Population Annealing parameters (only used when sampler='pa').
    agarose : AgaroseConfig
        Agarose gel parameters (only used when surface_type='agarose'/'agarose_surface').
    pdb_path : str
        Path to input PDB structure file.
    top_path : str
        Path to input GROMACS topology (.top) file.
    output : str
        Output parquet summary table path.
    save_traj : bool
        Record MC trajectory and write PDB + XTC files. Default False.
    traj_prefix : str
        Prefix for trajectory files ({prefix}.pdb, {prefix}.xtc). Default 'traj'.
    phi_grid_path : str
        Path to precomputed phi(x,y,z) .npy field for patterned surface.
    phi_grid_origin : tuple
        Grid origin (x, y, z) in nm. Default (0, 0, 0).
    phi_grid_spacing : tuple
        Grid spacing (dx, dy, dz) in nm. Default (0.1, 0.1, 0.1).
    n_clusters : int
        Number of orientation clusters (k-means k) for the output summary.
        Default 2.
    top_k : int
        Number of basins reported per system in the parquet. Default 2.
    """
    surface_type: str = "continuum"
    sampler: str = "pa"
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    mc: MCConfig = field(default_factory=MCConfig)
    pt: PTConfig = field(default_factory=PTConfig)
    pa: PAConfig = field(default_factory=PAConfig)
    agarose: AgaroseConfig = field(default_factory=AgaroseConfig)
    crystal: CrystalConfig = field(default_factory=CrystalConfig)
    full_atom: FullAtomConfig = field(default_factory=FullAtomConfig)
    pdb_path: str = ""
    top_path: str = ""
    output: str = "output.parquet"
    save_traj: bool = False
    traj_prefix: str = "traj"
    phi_grid_path: str = ""
    phi_grid_origin: tuple = (0.0, 0.0, 0.0)
    phi_grid_spacing: tuple = (0.1, 0.1, 0.1)
    n_clusters: int = 2
    top_k: int = 2
