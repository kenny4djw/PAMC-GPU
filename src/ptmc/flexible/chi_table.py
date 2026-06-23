"""IUPAC chi-dihedral definitions for the 20 canonical amino acids.

Each chi_k is defined by an ordered four-atom tuple (i, j, k, l) of PDB atom
names; the dihedral rotates the bond j-k. Atom names follow the standard PDB
convention used by GROMACS pdb2gmx and the AMBER99SB-ILDN force field
(matches the names in [1UBQ_processed.pdb](data/1UBQ_processed.pdb)).

Per the design doc § 1.2:
- ALA, GLY, PRO: n_chi = 0  (PRO ring fixed in v1)
- SER, THR, CYS, VAL: n_chi = 1
- ASP, ASN, HIS, PHE, TYR, TRP, LEU, ILE: n_chi = 2
- MET, GLU, GLN, LYS: n_chi = 3
- ARG: n_chi = 4

LYS chi4 (CG-CD-CE-NZ) is intentionally omitted in v1 — terminal NH3+
rotation is folded into the implicit-solvent average. May be promoted to a
real DOF in a later revision.

Hydrogens never appear in chi definitions (the IUPAC convention uses heavy
atoms only). All names here are heavy atoms.
"""
from __future__ import annotations

from typing import Mapping, Sequence


# Each tuple is (i, j, k, l). The dihedral angle is measured between the
# i-j-k and j-k-l planes; rotating chi_k means rotating about the j-k bond.
CHI_TABLE: Mapping[str, Sequence[tuple[str, str, str, str]]] = {
    # n_chi = 0 (rigid)
    "ALA": (),
    "GLY": (),
    "PRO": (),
    # n_chi = 1
    "SER": (("N", "CA", "CB", "OG"),),
    "THR": (("N", "CA", "CB", "OG1"),),
    "CYS": (("N", "CA", "CB", "SG"),),
    "VAL": (("N", "CA", "CB", "CG1"),),
    # n_chi = 2
    "ASP": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "OD1"),
    ),
    "ASN": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "OD1"),
    ),
    "HIS": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "ND1"),
    ),
    "PHE": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ),
    "TYR": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ),
    "TRP": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ),
    "LEU": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ),
    "ILE": (
        ("N", "CA", "CB", "CG1"),
        ("CA", "CB", "CG1", "CD1"),
    ),
    # n_chi = 3
    "MET": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "SD"),
        ("CB", "CG", "SD", "CE"),
    ),
    "GLU": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ),
    "GLN": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ),
    "LYS": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "CE"),
    ),
    # n_chi = 4
    "ARG": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "NE"),
        ("CG", "CD", "NE", "CZ"),
    ),
}


N_CHI: Mapping[str, int] = {res: len(chis) for res, chis in CHI_TABLE.items()}


FLEXIBLE_RESIDUES: frozenset[str] = frozenset(
    res for res, n in N_CHI.items() if n > 0
)


# Common HIS protonation-state aliases (AMBER / CHARMM). All share heavy-atom
# topology so the chi definitions are identical. v1 treats protonation as
# fixed so we only need name aliases.
_HIS_ALIASES = ("HID", "HIE", "HIP", "HSD", "HSE", "HSP")
for _alias in _HIS_ALIASES:
    CHI_TABLE[_alias] = CHI_TABLE["HIS"]  # type: ignore[index]
    N_CHI[_alias] = N_CHI["HIS"]  # type: ignore[index]

# CYX = disulfide-bonded cysteine. Same chi as CYS (chi1 = N-CA-CB-SG); the
# disulfide bond removes downstream-from-SG atoms (none anyway). Whether the
# CYX SG should be considered rotatable in a disulfide is a separate
# question — we leave it on and let bonded energy penalize bad geometry.
CHI_TABLE["CYX"] = CHI_TABLE["CYS"]  # type: ignore[index]
N_CHI["CYX"] = N_CHI["CYS"]  # type: ignore[index]


def chi_atom_names(resname: str) -> Sequence[tuple[str, str, str, str]]:
    """Return the chi definitions for a residue, or () for unknown / rigid.

    Unknown residues (ligands, modified AAs, ions) are treated as rigid.
    """
    return CHI_TABLE.get(resname.upper(), ())


def n_chi_for(resname: str) -> int:
    """Return the number of chi dihedrals for a residue (0 if unknown / rigid)."""
    return N_CHI.get(resname.upper(), 0)
