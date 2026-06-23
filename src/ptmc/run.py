"""CLI entry point for PTMC-GPU.

Usage
-----
ptmc --pdb PROTEIN.pdb --top PROTEIN.top [OPTIONS]

Surface types  (--surface-type):
  continuum        Homogeneous half-space: Steele 9-3 vdW + screened Coulomb
  patterned        Same vdW + laterally patterned phi(x,y,z) from a .npy grid
  agarose          3D agarose gel: Gaussian soft-core + Yukawa (no hard wall)
  agarose_surface  Flat gel coating: same physics + hard wall at z_min

Samplers  (--sampler):
  pa   Population Annealing with adaptive cooling — gives absolute / ΔΔG
       adsorption free energy via ``ptmc.analysis.adsorption`` (default)
  pt   Parallel Tempering replica exchange — orientation sampling only

Protein DOF  (--flexible):
  Off (default): rigid body. 3 translation + SO(3) rotation.
  On            : EXPERIMENTAL — rigid backbone + per-residue sidechain χ
                  dihedrals. The chi-aware PA / PT drivers are wired in
                  ``run_pipeline``, but the intra-NB kernel currently uses
                  geometric combining (comb-rule 1/3), inconsistent with
                  the amber99sb-ildn topology (comb-rule 2, Lorentz–
                  Berthelot). Quantitative ΔG⁰_ads and orientation from
                  this path are NOT validated and must not be quoted. To
                  prevent accidental use, ``--flexible`` is **disabled at
                  the CLI level** in the present release; the path can
                  only be reached from the Python API by setting
                  ``MCConfig(flexible=True,
                  flexible_ack_experimental=True)`` (see SI §S1.2).

Trajectory  (--save-traj):
  Writes one representative chain as PDB topology + XTC trajectory.
  Requires MDAnalysis. Works with any surface type.
"""
from __future__ import annotations

import argparse
import logging
import numpy as np

from ptmc.analysis.orientation import contact_normal
from ptmc.analysis.clustering import cluster_orientations
from ptmc.analysis.free_energy import basin_free_energies
from ptmc.log import logger, configure_logger
from ptmc.config import beta as beta_of


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------

def _triple_float(s: str) -> tuple:
    """Parse '1.0 2.0 3.0' or '1.0,2.0,3.0' → (float, float, float)."""
    parts = s.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected 3 space/comma-separated values, got: {s!r}")
    return tuple(float(x) for x in parts)


def _triple_bool(s: str) -> tuple:
    """Parse '0 0 1' → (False, False, True) for axis mask."""
    parts = s.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected 3 0/1 values for axis mask, got: {s!r}")
    return tuple(bool(int(x)) for x in parts)


def _parse_cell_xy(s: str) -> tuple:
    """Parse 'Lx Ly' or 'Lx,Ly' → (float, float) for lateral cell dimensions."""
    parts = s.replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected 2 values (Lx Ly) for --surface-cell, got: {s!r}")
    return tuple(float(x) for x in parts)


# ---------------------------------------------------------------------------
# Output helpers (used by main and importable by tests)
# ---------------------------------------------------------------------------

def summarize_systems(out: dict, betas: list, k: int = 2, top_k: int = 2):
    """Build orientation summary DataFrame from run_systems-compatible output.

    Parameters
    ----------
    out : dict
        Must contain 'quats' (S, C, 4) and 'system_ids'.
    betas : list[float]
        Per-system inverse temperature (mol/kJ).
    k : int
        Number of orientation clusters.
    top_k : int
        Basins to report per system.

    Returns
    -------
    pandas.DataFrame. Every row carries the per-basin orientation columns
    (``system_id, basin_rank, population, dG_kJ_mol, normal_*, tilt_deg``)
    plus health flags ``n_bad_chains`` (chains with non-finite final energy)
    and ``pa_converged`` (PA reached β_target; True for HT/PT).

    When ``out`` is a PA run (``pa_adsorption`` present), the per-system
    absolute adsorption energetics are repeated on every basin row::

        dG_box_ads_kJ_mol   — raw box-conditional ΔG = -kT·logZ_ratio
        K_ads_nm            — surface excess equilibrium constant
        dG_std_ads_kJ_mol   — standard-state-corrected ΔG⁰_ads (1 M default)
        pa_logZ_ratio       — echo of the raw PA estimator
        pa_slab_z_lo / z_hi — slab used for the conversion (nm)
        pa_beta0_ok         — β₀-adequacy check passed (bool)

    These let the parquet stand alone: the adsorption free energy is read
    off the per-system constant columns; the orientation breakdown lives
    in the per-basin rows.

    For PT runs (no PA-derived adsorption columns), only the per-basin
    orientation summary and ``pa_converged=True`` placeholder are written.
    """
    import pandas as pd

    quats = out["quats"]       # (S, C, 4)
    sids = out["system_ids"]
    # Per-chain non-finite mask (S, C). Used to attach per-system n_bad_chains
    # to the summary — gives users a one-line diagnostic without having to
    # peek into the raw sampler output.
    energies = np.asarray(out.get("energies", np.zeros((len(sids), 0))))
    bad_per_chain = ~np.isfinite(energies) if energies.size else np.zeros_like(energies, dtype=bool)
    pa_raw = out.get("_pa_raw", {})
    pa_converged_flag = bool(pa_raw.get("converged", True)) if pa_raw else True  # PT/non-PA: placeholder True
    ads = out.get("pa_adsorption")
    slab = out.get("_pa_slab", {})
    beta0_check = out.get("_pa_beta0_check", {})
    ads_cols = {}
    if ads is not None:
        ads_cols = dict(
            dG_box_ads_kJ_mol=float(ads.dG_box_kJ_per_mol),
            K_ads_nm=float(ads.K_ads_nm),
            dG_std_ads_kJ_mol=float(ads.dG_std_kJ_per_mol),
            pa_logZ_ratio=float(ads.logZ_ratio),
            pa_slab_z_lo=float(slab.get("z_lo", ads.z_lo)),
            pa_slab_z_hi=float(slab.get("z_hi", ads.z_hi)),
            pa_beta0_ok=bool(beta0_check.get("ok", True)),
        )
    rows = []
    for s, sid in enumerate(sids):
        n_bad = int(bad_per_chain[s].sum()) if bad_per_chain.ndim >= 2 else 0
        if quats[s].shape[0] == 0:
            logger.warning("System %s has no sampled poses; skipping summary.", sid)
            continue
        cn = np.array([contact_normal(q) for q in quats[s]])  # (C, 3)
        labels, cents = cluster_orientations(cn, k=k, seed=0)
        p, dG = basin_free_energies(labels, float(betas[s]), k=k)
        if p.sum() == 0.0:
            logger.warning(
                "System %s clustering yielded zero population (all %d samples "
                "collapsed). Skipping summary row.", sid, cn.shape[0])
            continue
        order = np.argsort(-p)[:top_k]
        for rank, b in enumerate(order):
            n = cents[b]
            nrm = np.linalg.norm(n)
            n = n / nrm if nrm > 0 else n
            tilt = np.degrees(np.arccos(np.clip(-n[2], -1.0, 1.0)))
            row = dict(
                system_id=int(sid), basin_rank=int(rank),
                population=float(p[b]), dG_kJ_mol=float(dG[b]),
                normal_x=float(n[0]), normal_y=float(n[1]), normal_z=float(n[2]),
                tilt_deg=float(tilt),
                n_bad_chains=n_bad,
                pa_converged=pa_converged_flag)
            row.update(ads_cols)
            rows.append(row)
    return pd.DataFrame(rows)


def write_parquet(df, path: str) -> str:
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ptmc",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── I/O ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("I/O")
    g.add_argument("--pdb", required=True, metavar="FILE",
                   help="Input PDB structure file")
    g.add_argument("--top", required=True, metavar="FILE",
                   help="Input GROMACS topology (.top) file")
    g.add_argument("--output", default="output.parquet", metavar="FILE",
                   help="Output parquet path (default: output.parquet)")
    g.add_argument("--verbose", "-v", action="store_true",
                   help="Enable debug logging")

    # ── Surface type ───────────────────────────────────────────────────────
    g = p.add_argument_group("Surface model")
    g.add_argument(
        "--surface-type",
        choices=["continuum", "patterned", "agarose", "agarose_surface",
                 "crystal", "full_atom"],
        default="continuum",
        metavar="TYPE",
        help=("Surface potential model:\n"
              "  continuum       Homogeneous Steele 9-3 + screened Coulomb (default)\n"
              "  patterned       Same vdW + patterned phi(x,y,z) from .npy grid\n"
              "  agarose         3D agarose gel (Gaussian soft-core + Yukawa)\n"
              "  agarose_surface Flat gel coating + hard wall\n"
              "  crystal         Built-in crystal slab with CLAYFF force field\n"
              "  full_atom       External surface PDB + JSON / GROMACS force field"))

    # ── Continuum / patterned surface params ───────────────────────────────
    g = p.add_argument_group(
        "Continuum surface params  [--surface-type continuum | patterned]")
    g.add_argument("--rho-s", type=float, default=30.0, metavar="NM⁻³",
                   help="Surface atom number density nm⁻³ (default: 30.0)")
    g.add_argument("--c6-surf", type=float, default=1.0, metavar="KJ·NM⁶",
                   help="Surface LJ C6 coefficient kJ/mol·nm⁶ (default: 1.0)")
    g.add_argument("--c12-surf", type=float, default=1.0, metavar="KJ·NM¹²",
                   help="Surface LJ C12 coefficient kJ/mol·nm¹² (default: 1.0)")
    g.add_argument("--lambda-d", type=float, default=0.785, metavar="NM",
                   help="Debye screening length nm (default: 0.785 ≈ 0.15 M NaCl)")
    g.add_argument("--z-min", type=float, default=0.15, metavar="NM",
                   help="Hard-wall distance from surface nm (default: 0.15)")
    g.add_argument("--psi0", type=float, default=0.0, metavar="KJ/(MOL·E)",
                   help="Surface electrostatic potential ψ₀ kJ/mol/e (default: 0.0)")
    g.add_argument("--eps-surf", type=float, default=None, metavar="KJ/MOL",
                   help=("Surface LJ ε kJ/mol. When given together with "
                         "--sigma-surf, the ε-σ Lorentz-Berthelot Steele path is "
                         "used instead of C6/C12 (default: unset → C6/C12 path)"))
    g.add_argument("--sigma-surf", type=float, default=None, metavar="NM",
                   help=("Surface LJ σ nm. Must be paired with --eps-surf "
                         "(default: unset)"))

    # ── Patterned surface params ───────────────────────────────────────────
    g = p.add_argument_group(
        "Patterned surface params  [--surface-type patterned]",
        "Precomputed phi(x,y,z) field; vdW uses continuum params above.")
    g.add_argument("--phi-grid", default="", metavar="FILE",
                   help="Path to phi field .npy file (shape nx×ny×nz, kJ/mol/e)")
    g.add_argument("--phi-grid-origin", type=_triple_float,
                   default=(0.0, 0.0, 0.0), metavar="'X Y Z'",
                   help="Grid origin (nm), e.g. '0.0 0.0 0.15' (default: 0 0 0)")
    g.add_argument("--phi-grid-spacing", type=_triple_float,
                   default=(0.1, 0.1, 0.1), metavar="'DX DY DZ'",
                   help="Grid spacing (nm) (default: 0.1 0.1 0.1)")

    # ── Built-in crystal params ────────────────────────────────────────────
    g = p.add_argument_group(
        "Crystal surface params  [--surface-type crystal]",
        "Uses CLAYFF force field automatically. "
        "All 4 built-in materials: alpha_quartz, rutile_tio2, anatase_tio2, gold_fcc.")
    g.add_argument("--crystal",
                   choices=["alpha_quartz", "rutile_tio2",
                            "anatase_tio2", "gold_fcc"],
                   default="alpha_quartz",
                   help="Crystal material (default: alpha_quartz)")
    g.add_argument("--crystal-hkl", type=_triple_float,
                   default=(0.0, 0.0, 1.0), metavar="'H K L'",
                   help="Miller indices e.g. '0 0 1' (default: 0 0 1)")
    g.add_argument("--crystal-n-layers", type=int, default=4, metavar="INT",
                   help="Slab thickness in unit-cell layers (default: 4)")
    g.add_argument("--no-hydroxylate", action="store_true",
                   help="Skip -OH termination of silica/rutile surfaces")
    g.add_argument("--crystal-r-cut", type=float, default=1.4, metavar="NM",
                   help="PBC cutoff for grid building nm (default: 1.4)")
    g.add_argument("--crystal-grid-spacing", type=float, default=0.05, metavar="NM",
                   help="Field grid resolution nm (default: 0.05)")
    g.add_argument("--crystal-lambda-d", type=float, default=0.785, metavar="NM",
                   help="Debye length nm (default: 0.785)")
    g.add_argument("--crystal-z-min", type=float, default=0.2, metavar="NM",
                   help="Hard-wall distance from surface nm (default: 0.2)")
    g.add_argument("--crystal-vacuum", type=float, default=3.0, metavar="NM",
                   help="Vacuum gap above the slab for slab cutting nm (default: 3.0)")
    g.add_argument("--crystal-z-grid-max", type=float, default=3.2, metavar="NM",
                   help=("Absolute upper z of the field grid above surface (nm). "
                         "Grid covers z ∈ [crystal-z-min, crystal-z-grid-max]. "
                         "Must exceed crystal-z-min + R_protein. Default: 3.2"))

    # ── External full-atom surface params ─────────────────────────────────
    g = p.add_argument_group(
        "Full-atom surface params  [--surface-type full_atom]",
        "Two FF import paths (choose one):\n"
        "  --surface-ff-json  JSON dict {AtomName: {q, sigma, epsilon}}\n"
        "  --surface-top      GROMACS .top (parmed, same as protein path)")
    g.add_argument("--surface-pdb", default="", metavar="FILE",
                   help="PDB file with surface atom positions (Å)")
    g.add_argument("--surface-top", default="", metavar="FILE",
                   help="GROMACS .top for the surface (parmed-based FF import)")
    g.add_argument("--surface-ff-json", default="", metavar="FILE",
                   help=(
                       "JSON force field file. Format:\n"
                       '  {"Au": {"q": 0.0, "sigma": 0.254, "epsilon": 2.483}, ...}\n'
                       "Keys = atom names in PDB. Units: epsilon kJ/mol, sigma nm, q e."))
    g.add_argument("--surface-cell", type=_parse_cell_xy,
                   default=(3.0, 3.0), metavar="'LX LY'",
                   help="Lateral periodic cell (Lx Ly) nm (default: 3.0 3.0)")
    g.add_argument("--fa-r-cut", type=float, default=1.4, metavar="NM",
                   help="PBC cutoff for grid building nm (default: 1.4)")
    g.add_argument("--fa-grid-spacing", type=float, default=0.05, metavar="NM",
                   help="Field grid resolution nm (default: 0.05)")
    g.add_argument("--fa-lambda-d", type=float, default=0.785, metavar="NM",
                   help="Debye length nm (default: 0.785)")
    g.add_argument("--fa-z-min", type=float, default=0.2, metavar="NM",
                   help="Hard-wall distance nm (default: 0.2)")
    g.add_argument("--fa-z-grid-max", type=float, default=3.2, metavar="NM",
                   help=("Absolute upper z of the field grid above surface (nm). "
                         "Grid covers z ∈ [fa-z-min, fa-z-grid-max]. "
                         "Must exceed fa-z-min + R_protein. Default: 3.2"))

    # ── Agarose gel params ─────────────────────────────────────────────────
    g = p.add_argument_group(
        "Agarose gel params  [--surface-type agarose | agarose_surface]")
    g.add_argument("--gel-L", type=float, default=20.0, metavar="NM",
                   help="Box/lateral size nm (default: 20.0)")
    g.add_argument("--gel-thickness", type=float, default=10.0, metavar="NM",
                   help="Coating thickness nm, agarose_surface only (default: 10.0)")
    g.add_argument("--gel-n-seeds", type=int, default=50, metavar="INT",
                   help="Voronoi seed count for fiber network (default: 50)")
    g.add_argument("--gel-sigma", type=float, default=2.0, metavar="NM",
                   help="Gaussian soft-core radius nm (default: 2.0)")
    g.add_argument("--gel-A", type=float, default=500.0, metavar="KJ/MOL",
                   help="Steric repulsion amplitude kJ/mol (default: 500.0)")
    g.add_argument("--gel-doping-frac", type=float, default=0.1, metavar="FRAC",
                   help="Fraction of fiber nodes charged [0,1] (default: 0.1)")
    g.add_argument("--gel-q-ligand", type=float, default=-1.0, metavar="E",
                   help="Charge per doped node e (+1 cation, -1 anion, default: -1.0)")
    g.add_argument("--gel-lambda-d", type=float, default=3.0, metavar="NM",
                   help="Debye length for gel nm (default: 3.0 ≈ 10 mM)")
    g.add_argument("--gel-dielectric", type=float, default=78.5,
                   help="Solvent relative permittivity (default: 78.5)")
    g.add_argument("--gel-seed", type=int, default=42,
                   help="Master seed for fiber network + doping (default: 42)")
    g.add_argument("--gel-grid-spacing", type=float, default=0.15, metavar="NM",
                   help="Agarose field grid resolution nm (default: 0.15)")
    g.add_argument("--gel-z-min", type=float, default=0.2, metavar="NM",
                   help="Hard-wall distance nm, agarose_surface only (default: 0.2)")
    g.add_argument("--gel-margin", type=float, default=0.3, metavar="FRAC",
                   help="Voronoi seed-box fractional extension (default: 0.3)")
    g.add_argument("--gel-sample-density", type=float, default=0.5, metavar="FRAC",
                   help="Fiber node spacing along edges as fraction of sigma (default: 0.5)")
    g.add_argument("--gel-doping-correlation", type=int, default=5, metavar="INT",
                   help="Consecutive nodes doped per seed (cluster size) (default: 5)")
    g.add_argument("--gel-floor-pctile", type=float, default=5.0, metavar="PCT",
                   help=("U_steric zero-reference percentile for 3D gel; "
                         "0 disables (default: 5.0)"))

    # ── Sampler ────────────────────────────────────────────────────────────
    g = p.add_argument_group("Sampler")
    g.add_argument(
        "--sampler", choices=["pt", "pa"], default="pa",
        help=("pa = population annealing (default; emits ΔG⁰_ads); "
              "pt = parallel tempering"))

    # ── Common MC params ───────────────────────────────────────────────────
    g = p.add_argument_group("Common MC parameters")
    g.add_argument("--n-steps", type=int, default=10000,
                   help="MC sweeps per chain / walker (default: 10000)")
    g.add_argument("--temperature", type=float, default=300.0, metavar="K",
                   help="Target temperature K (default: 300.0)")
    g.add_argument("--sigma-rot", type=float, default=0.1, metavar="RAD",
                   help="Rotation proposal stddev rad (default: 0.1)")
    g.add_argument("--sigma-trans", type=float, default=0.05, metavar="NM",
                   help="Translation proposal stddev nm (default: 0.05)")
    g.add_argument("--axis-mask", type=_triple_bool,
                   default=(False, False, True), metavar="'X Y Z'",
                   help=("Translation axes enabled as 0/1 (default: '0 0 1' = z only). "
                         "E.g. '1 1 1' for free 3D translation."))
    g.add_argument("--seed", type=int, default=42,
                   help="Master PRNG seed (default: 42)")
    g.add_argument("--init-z", type=float, default=0.0, metavar="NM",
                   help=("Initial protein z-height nm; 0 = auto-compute a safe "
                         "value (R_protein + z_min + 3σ). A positive value "
                         "overrides it (default: 0.0 = auto)"))

    # ── Protein degrees of freedom ─────────────────────────────────────────
    g = p.add_argument_group("Protein degrees of freedom")
    g.add_argument("--flexible", action="store_true",
                   help=("EXPERIMENTAL — DISABLED at the CLI level. The "
                         "chi-DOF kernel exists in ptmc.flexible.* but the "
                         "intra-NB path uses geometric combining "
                         "(comb-rule 1/3), inconsistent with the "
                         "amber99sb-ildn LB topology. Quantitative results "
                         "from this path are not validated and must not be "
                         "quoted. Setting --flexible on the CLI will refuse "
                         "to run; the path is reachable only via the Python "
                         "API with MCConfig(flexible=True, "
                         "flexible_ack_experimental=True). See SI §S1.2."))
    g.add_argument("--sigma-chi", type=float, default=0.2, metavar="RAD",
                   help=("χ-dihedral proposal stddev (rad) when --flexible is "
                         "set. Default 0.2 ≈ 11°."))

    # ── PT-specific ────────────────────────────────────────────────────────
    g = p.add_argument_group("Parallel Tempering options  [--sampler pt]")
    g.add_argument("--pt-n-replicas", type=int, default=8, metavar="INT",
                   help="Number of temperature replicas (default: 8)")
    g.add_argument("--pt-T-min", type=float, default=200.0, metavar="K",
                   help="Lowest replica temperature K (default: 200.0)")
    g.add_argument("--pt-T-max", type=float, default=600.0, metavar="K",
                   help="Highest replica temperature K (default: 600.0)")
    g.add_argument("--pt-n-rounds", type=int, default=200, metavar="INT",
                   help="PT rounds = MC sweep + replica exchange (default: 200)")
    g.add_argument("--pt-n-sweep", type=int, default=50, metavar="INT",
                   help="MC sweeps per round per replica (default: 50)")

    # ── PA-specific ────────────────────────────────────────────────────────
    g = p.add_argument_group("Population Annealing options  [--sampler pa]")
    g.add_argument("--pa-n-walkers", type=int, default=512, metavar="INT",
                   help="Population size (default: 512)")
    g.add_argument("--pa-T-start", type=float, default=5000.0, metavar="K",
                   help=("Starting (hot) temperature K (default: 5000.0). "
                         "Must be high enough that ⟨|β₀·E|⟩ ≪ 1 over the "
                         "initial slab — the β₀-adequacy check warns if not."))
    g.add_argument("--pa-target-ess", type=float, default=0.7, metavar="FRAC",
                   help="Minimum ESS fraction for adaptive cooling (default: 0.7)")
    g.add_argument("--pa-max-steps", type=int, default=400, metavar="INT",
                   help="Maximum adaptive cooling steps (default: 400)")
    g.add_argument("--pa-z-max", type=float, default=0.0, metavar="NM",
                   help=("Slab top (nm) for the PA reference state used in "
                         "ΔG⁰_ads. 0 = auto: crystal/full_atom use z_grid_max; "
                         "continuum/patterned default to safe-z + 3.0 nm. "
                         "Raise if K_ads has not converged to a bulk baseline."))
    g.add_argument("--pa-c-std", type=float, default=1.0, metavar="M",
                   help=("Standard reference concentration for ΔG⁰_ads "
                         "(mol/L). Default 1.0 M."))

    # ── Trajectory ─────────────────────────────────────────────────────────
    g = p.add_argument_group("Trajectory output")
    g.add_argument("--save-traj", action="store_true",
                   help=("Record and save one representative MC chain as "
                         "PDB topology + XTC trajectory. Requires MDAnalysis."))
    g.add_argument("--traj-prefix", default="traj", metavar="PREFIX",
                   help=("Prefix for trajectory output files "
                         "({prefix}.pdb + {prefix}.xtc, default: traj)"))

    # ── Analysis output ────────────────────────────────────────────────────
    g = p.add_argument_group("Orientation analysis output")
    g.add_argument("--n-clusters", type=int, default=2, metavar="INT",
                   help="Number of orientation clusters (k-means k) (default: 2)")
    g.add_argument("--top-k", type=int, default=2, metavar="INT",
                   help="Basins reported per system in the parquet (default: 2)")

    return p


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logger(logging.DEBUG if args.verbose else logging.INFO)

    if args.flexible:
        raise SystemExit(
            "ERROR: --flexible is disabled at the CLI level in this release.\n"
            "The chi-DOF kernel is EXPERIMENTAL: its intra-NB path uses "
            "geometric combining, inconsistent with the amber99sb-ildn LB "
            "topology. Quantitative results are not validated and must not "
            "be quoted. To use this path for development, drive PTMC-GPU "
            "from the Python API with "
            "MCConfig(flexible=True, flexible_ack_experimental=True) and "
            "acknowledge the EXPERIMENTAL status. See SI §S1.2."
        )

    from ptmc.config import (
        SurfaceConfig, MCConfig, SimConfig,
        PTConfig, PAConfig, AgaroseConfig,
        CrystalConfig, FullAtomConfig,
        device_info,
    )
    from ptmc.sampler.pipeline import run_pipeline

    info = device_info()
    logger.info("JAX backend = %s | %d device(s): %s",
                info.backend, info.n_devices, info.device_repr)

    config = SimConfig(
        surface_type=args.surface_type,
        sampler=args.sampler,
        surface=SurfaceConfig(
            rho_s=args.rho_s,
            c6_surf=args.c6_surf,
            c12_surf=args.c12_surf,
            lambda_D=args.lambda_d,
            z_min=args.z_min,
            psi0=args.psi0,
            eps_surf=args.eps_surf,
            sigma_surf=args.sigma_surf,
        ),
        mc=MCConfig(
            n_steps=args.n_steps,
            sigma_rot=args.sigma_rot,
            sigma_trans=args.sigma_trans,
            axis_mask=args.axis_mask,
            seed=args.seed,
            temperature=args.temperature,
            init_z=args.init_z,
            flexible=args.flexible,
            sigma_chi=args.sigma_chi,
        ),
        pt=PTConfig(
            n_replicas=args.pt_n_replicas,
            T_min=args.pt_T_min,
            T_max=args.pt_T_max,
            n_rounds=args.pt_n_rounds,
            n_sweep=args.pt_n_sweep,
        ),
        pa=PAConfig(
            n_walkers=args.pa_n_walkers,
            T_start=args.pa_T_start,
            target_ess=args.pa_target_ess,
            max_annealing_steps=args.pa_max_steps,
            z_max=args.pa_z_max,
            c_std=args.pa_c_std,
        ),
        agarose=AgaroseConfig(
            L=args.gel_L,
            thickness=args.gel_thickness,
            n_seeds=args.gel_n_seeds,
            sigma=args.gel_sigma,
            A=args.gel_A,
            doping_frac=args.gel_doping_frac,
            q_ligand=args.gel_q_ligand,
            lambda_D=args.gel_lambda_d,
            dielectric=args.gel_dielectric,
            gel_seed=args.gel_seed,
            grid_spacing=args.gel_grid_spacing,
            z_min=args.gel_z_min,
            margin=args.gel_margin,
            sample_density=args.gel_sample_density,
            doping_correlation=args.gel_doping_correlation,
            floor_pctile=args.gel_floor_pctile,
        ),
        crystal=CrystalConfig(
            crystal=args.crystal,
            hkl=tuple(int(x) for x in args.crystal_hkl),
            n_layers=args.crystal_n_layers,
            hydroxylate=not args.no_hydroxylate,
            r_cut=args.crystal_r_cut,
            grid_spacing=args.crystal_grid_spacing,
            lambda_D=args.crystal_lambda_d,
            z_min=args.crystal_z_min,
            vacuum=args.crystal_vacuum,
            z_grid_max=args.crystal_z_grid_max,
        ),
        full_atom=FullAtomConfig(
            surface_pdb=args.surface_pdb,
            surface_top=args.surface_top,
            surface_ff_json=args.surface_ff_json,
            cell_xy=args.surface_cell,
            r_cut=args.fa_r_cut,
            grid_spacing=args.fa_grid_spacing,
            lambda_D=args.fa_lambda_d,
            z_min=args.fa_z_min,
            z_grid_max=args.fa_z_grid_max,
        ),
        pdb_path=args.pdb,
        top_path=args.top,
        output=args.output,
        save_traj=args.save_traj,
        traj_prefix=args.traj_prefix,
        phi_grid_path=args.phi_grid,
        phi_grid_origin=args.phi_grid_origin,
        phi_grid_spacing=args.phi_grid_spacing,
        n_clusters=args.n_clusters,
        top_k=args.top_k,
    )

    out, betas = run_pipeline(config)

    logger.info("Summarizing orientations → %s", config.output)
    df = summarize_systems(out, betas, k=config.n_clusters, top_k=config.top_k)
    write_parquet(df, config.output)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
