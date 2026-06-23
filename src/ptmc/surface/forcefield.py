"""Force-field parameter assignment for surface atoms (CLAYFF).

Cygan et al. 2004, J. Phys. Chem. B 108, 1255.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ptmc.model.structures import DiscreteSurface
from .lattice import Lattice


@dataclass
class FFParams:
    q: float
    sigma: float
    epsilon: float

    @property
    def c6(self) -> float:
        return 4.0 * self.epsilon * self.sigma ** 6

    @property
    def c12(self) -> float:
        return 4.0 * self.epsilon * self.sigma ** 12


CLAYFF: dict[str, FFParams] = {
    "Si":      FFParams(q=+2.1000, sigma=0.330,   epsilon=7.700e-6),
    "Ob":      FFParams(q=-1.0500, sigma=0.3166,   epsilon=0.6506),
    "Ot":      FFParams(q=-0.9500, sigma=0.3166,   epsilon=0.6506),
    "OH":      FFParams(q=-0.9500, sigma=0.3166,   epsilon=0.6506),
    "H_oh":    FFParams(q=+0.4250, sigma=0.0000,   epsilon=0.0000),
    "Ti":      FFParams(q=+1.9500, sigma=0.3120,   epsilon=0.5234),
    "Ti_surf": FFParams(q=+1.5750, sigma=0.3120,   epsilon=0.5234),
    "Ob_ti":   FFParams(q=-0.9750, sigma=0.3166,   epsilon=0.6506),
    "Au":      FFParams(q=+0.0000, sigma=0.2540,   epsilon=2.4830),
}
CLAYFF["Ob_silica"] = CLAYFF["Ob"]
CLAYFF["Ot_silica"] = CLAYFF["Ot"]


def assign_ff(species_list: list[str],
              ff: dict[str, FFParams] | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ff is None:
        ff = CLAYFF
    q_l, c6_l, c12_l = [], [], []
    missing = set()
    for s in species_list:
        if s not in ff:
            missing.add(s); q_l.append(0.0); c6_l.append(0.0); c12_l.append(0.0)
        else:
            p = ff[s]; q_l.append(p.q); c6_l.append(p.c6); c12_l.append(p.c12)
    if missing:
        raise ValueError(f"missing FF params for species: {sorted(missing)}")
    return (np.array(q_l, dtype=np.float64),
            np.array(c6_l, dtype=np.float64),
            np.array(c12_l, dtype=np.float64))


def lattice_to_surface(lattice: Lattice, ff=None,
                       lambda_D: float = 0.785, z_min: float = 0.2,
                       name: str = "surface") -> DiscreteSurface:
    """Convert a Lattice to a DiscreteSurface with CLAYFF parameters."""
    cart = lattice.frac_to_cart(lattice.frac_pos)
    q, c6, c12 = assign_ff(lattice.species, ff)
    return DiscreteSurface(pos=cart, q=q, c6=c6, c12=c12,
                           lambda_D=lambda_D, z_min=z_min)


# ---------------------------------------------------------------------------
# External force field / surface loaders
# ---------------------------------------------------------------------------

def load_ff_json(path: str) -> dict[str, FFParams]:
    """Load a surface force field from a JSON file.

    JSON format
    -----------
    ::

        {
            "_comment": "optional comment keys (any key starting with _ is ignored)",
            "_units": "epsilon kJ/mol, sigma nm, q elementary charge",
            "Au":  {"q": 0.0,   "sigma": 0.2540, "epsilon": 2.4830},
            "Si":  {"q": 2.10,  "sigma": 0.3300, "epsilon": 7.7e-6},
            "Ob":  {"q": -1.05, "sigma": 0.3166, "epsilon": 0.6506}
        }

    Keys are atom *names* exactly as they appear in the PDB ATOM/HETATM
    records (columns 13-16, stripped).  Keys beginning with ``_`` are
    treated as comments and silently ignored.

    Parameters
    ----------
    path : str
        Path to the JSON force field file.

    Returns
    -------
    dict[str, FFParams]
        Mapping from atom name → FFParams(q, sigma, epsilon).
    """
    import json
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    ff: dict[str, FFParams] = {}
    for key, val in raw.items():
        if key.startswith("_"):
            continue
        try:
            ff[key] = FFParams(
                q=float(val["q"]),
                sigma=float(val["sigma"]),
                epsilon=float(val["epsilon"]),
            )
        except KeyError as e:
            raise ValueError(
                f"FF JSON entry {key!r} is missing field {e}. "
                f"Each entry must have 'q', 'sigma', 'epsilon'."
            ) from e
    return ff


def load_surface_from_pdb_json(pdb_path: str, ff_json_path: str,
                                lambda_D: float, z_min: float,
                                ) -> DiscreteSurface:
    """Build a DiscreteSurface from a PDB + JSON force field.

    Atom names in the PDB (columns 13-16) are looked up in the JSON FF.
    This path is fully independent of parmed and GROMACS.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file with surface atom coordinates (Å).
    ff_json_path : str
        Path to JSON force field file (see :func:`load_ff_json`).
    lambda_D : float
        Debye screening length (nm).
    z_min : float
        Hard-wall distance (nm).

    Returns
    -------
    DiscreteSurface
    """
    ff = load_ff_json(ff_json_path)

    # Minimal PDB parser — no parmed dependency.
    pos_list, name_list = [], []
    with open(pdb_path, encoding="utf-8") as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            name = line[12:16].strip()
            pos_list.append([x * 0.1, y * 0.1, z * 0.1])   # Å → nm
            name_list.append(name)

    if not pos_list:
        raise ValueError(f"No ATOM/HETATM records found in {pdb_path!r}")

    pos = np.array(pos_list, dtype=np.float64)

    missing = sorted({n for n in name_list if n not in ff})
    if missing:
        raise ValueError(
            f"Atom name(s) {missing} not found in FF JSON {ff_json_path!r}. "
            f"Available: {sorted(ff)}"
        )

    q_l, c6_l, c12_l = [], [], []
    for name in name_list:
        p = ff[name]
        q_l.append(p.q); c6_l.append(p.c6); c12_l.append(p.c12)

    return DiscreteSurface(
        pos=pos,
        q=np.array(q_l, dtype=np.float64),
        c6=np.array(c6_l, dtype=np.float64),
        c12=np.array(c12_l, dtype=np.float64),
        lambda_D=lambda_D,
        z_min=z_min,
    )


def load_surface_from_gromacs(pdb_path: str, top_path: str,
                               lambda_D: float, z_min: float,
                               ) -> DiscreteSurface:
    """Build a DiscreteSurface from GROMACS PDB + topology files.

    Uses the same parmed-based parsers as the protein path, so all
    GROMACS comb-rules (1, 2, 3) and #include chains are supported.
    The surface must have been prepared with ``gmx pdb2gmx`` (or
    equivalent), producing a ``*_processed.pdb`` and a ``*.top``.

    Parameters
    ----------
    pdb_path : str
        Processed PDB with surface atom coordinates (Å, same order as top).
    top_path : str
        GROMACS .top file for the surface molecule.
    lambda_D : float
        Debye screening length (nm).
    z_min : float
        Hard-wall distance (nm).

    Returns
    -------
    DiscreteSurface
    """
    from ptmc.io.parse_pdb import parse_pdb
    from ptmc.io.parse_topology import parse_topology

    pdb  = parse_pdb(pdb_path)
    topo = parse_topology(top_path)

    if pdb.n != len(topo.q):
        raise ValueError(
            f"Atom count mismatch: PDB has {pdb.n} atoms, topology has {len(topo.q)}."
        )

    return DiscreteSurface(
        pos=pdb.pos,
        q=topo.q,
        c6=topo.c6,
        c12=topo.c12,
        lambda_D=lambda_D,
        z_min=z_min,
    )
