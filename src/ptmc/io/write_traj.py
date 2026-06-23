"""Write sampled poses as a PDB topology + XTC trajectory via MDAnalysis.

Coordinates are converted nm -> Angstrom (MDAnalysis convention). The PDB stores
the topology (frame 0); the XTC stores all frames.
"""
from __future__ import annotations

import numpy as np
import MDAnalysis as mda

from ptmc.model.structures import Pose, Atoms

_NM_TO_ANGSTROM = 10.0


def write_trajectory(pdb_path: str, xtc_path: str, atoms: Atoms,
                     quats: np.ndarray, transs: np.ndarray) -> int:
    """Write F frames (poses applied to atoms) to PDB+XTC. Returns n_frames.

    SHAPE: quats (F,4), transs (F,3). Atom order follows `atoms`.
    """
    quats = np.asarray(quats); transs = np.asarray(transs)
    n = atoms.n
    resids = np.asarray(atoms.resids)
    uniq = np.unique(resids)
    resindex = np.searchsorted(uniq, resids)

    # One-pass resid → resname lookup. The previous version did
    # np.where(resids == r)[0][0] per unique residue (O(N·R)); for chains
    # with thousands of atoms and hundreds of residues this dominated the
    # PDB-write cost.
    first_resname: dict = {}
    for rid, rname in zip(resids.tolist(), atoms.resnames):
        first_resname.setdefault(int(rid), rname)
    resname_list = [first_resname[int(r)] for r in uniq]

    u = mda.Universe.empty(n, n_residues=uniq.size, atom_resindex=resindex,
                           trajectory=True)
    u.add_TopologyAttr("names", list(atoms.names))
    u.add_TopologyAttr("types", [e or "X" for e in atoms.elements])
    u.add_TopologyAttr("resids", [int(r) for r in uniq])
    u.add_TopologyAttr("resnames", resname_list)

    pos0 = Pose(quat=quats[0], trans=transs[0]).apply(atoms.pos0)
    u.atoms.positions = pos0 * _NM_TO_ANGSTROM
    u.atoms.write(pdb_path)

    F = quats.shape[0]
    with mda.Writer(xtc_path, n_atoms=n) as W:
        for f in range(F):
            u.atoms.positions = (
                Pose(quat=quats[f], trans=transs[f]).apply(atoms.pos0)
                * _NM_TO_ANGSTROM)
            W.write(u.atoms)
    return F


def read_trajectory(pdb_path: str, xtc_path: str):
    """Read back a PDB+XTC; returns (n_atoms, n_frames)."""
    u = mda.Universe(pdb_path, xtc_path)
    return u.atoms.n_atoms, u.trajectory.n_frames
