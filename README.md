# PTMC-GPU

<!-- editorial: replace ptmc-gpu/ptmc-gpu once the public repository is created -->
[![CI](https://github.com/ptmc-gpu/ptmc-gpu/actions/workflows/test.yml/badge.svg)](https://github.com/ptmc-gpu/ptmc-gpu/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The 24-file pytest suite runs on Linux/macOS/Windows under GitHub Actions on
every push and pull request (`.github/workflows/test.yml`), with line coverage
reported via `pytest --cov`. Install with `pip install -e ".[dev]"`.

GPU-accelerated **Parallel-Tempering (PT) / Population-Annealing (PA) Monte
Carlo** for predicting the adsorption orientation of proteins on solid
surfaces. The numerical core is written in **JAX** (`jit` + `lax.scan`), fully
vectorized over chains (batch axis = number of chains); there are no Python
loops over the chain dimension and no hand-written CUDA.


## Layout (architecture §9)

```
src/ptmc/
├── config.py                 # units, constants, device detection, RNG, precision
├── run.py                    # CLI entry point (full runner: stage 9)
├── io/                       # PDB + GROMACS top/itp parsing (parmed), trajectory I/O
├── model/                    # Atoms / Surface / Pose data models
├── energy/                   # direct-sum reference, surface grids, grid energy
├── mc/                       # trial moves, Metropolis kernel, batched sweep
├── sampler/                  # parallel_tempering, population_annealing
└── analysis/                 # orientation invariants, clustering, free energy, heatmaps
tests/
├── fixtures/                 # tiny real peptide (alanine dipeptide) + GROMACS top/itp
└── test_*.py
```

> Note on the package name: the architecture lists the sub-packages as
> `io/ model/ energy/ …`. They live under a top-level `ptmc` package
> (src layout) so that `ptmc.io` never shadows Python's stdlib `io`. The
> sub-package names match the architecture exactly.

## Units

GROMACS convention: length **nm**, energy **kJ/mol**, charge **e**,
temperature **K**. See [`UNITS.md`](UNITS.md) for constants and the non-bonded
interaction conventions (LJ comb-rules, screened Coulomb, grid factorization).

## Install

```bash
pip install -e ".[dev]"          # CPU
# On a GPU host, install a CUDA build of JAX first, e.g.:
#   pip install -U "jax[cuda12]"
```

JAX runs on GPU when available and **falls back to CPU** otherwise (no error).

## Test

```bash
pytest                            # runs tests/
python -m ptmc.run                # prints the active JAX device
```

## Stage progress

| Stage | Gate | Status |
|------:|------|:------:|
| 0 | pytest green; reproducible env; all modules import | ✅ |
| 1 | direct-sum energy validated against analytic values | ✅ |
| 2 | grid convergence + factorization correct | ✅ |
| 3 | toy-potential stationary distribution = Boltzmann | ✅ |
| 4 | throughput target + chain independence | ✅ |
| 5 | PT beats single-T + per-system isolation | ✅ |
| 6 | free-energy accuracy + PA/PT agreement | ✅ |
| 7 | pose round-trip + orientation invariance + readable traj | ✅ |
| 8 | reproduce lysozyme / protein G B1 (physical realism) | ✅ |
| 9 | batch equivalence + checkpoint/resume | ✅ |
