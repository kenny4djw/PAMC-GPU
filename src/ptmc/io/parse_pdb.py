"""Parse a PDB file (via parmed) into coordinate + identity arrays.

Coordinates are converted from Angstrom (PDB) to nm (GROMACS convention).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parmed as pmd

_ANGSTROM_TO_NM = 0.1


@dataclass
class PdbAtoms:
    """Raw per-atom data read from a PDB.

    SHAPE/UNITS: pos (N,3) nm; resids (N,) int; names/resnames/elements lists.
    """
    pos: np.ndarray
    names: list
    resids: np.ndarray
    resnames: list
    elements: list

    @property
    def n(self) -> int:
        return self.pos.shape[0]


# Module-level cache + bounded retry, for the same reason as in
# ``parse_topology``: ParmEd's reader intermittently raises an internal
# UnboundLocalError ("Atom") on otherwise-valid PDBs when load_file is called
# repeatedly in one process (e.g. once per PA seed). Caching by
# (resolved path, mtime) parses each file once; the retry covers a first-call
# flake. PdbAtoms is treated as read-only by all callers.
_PDB_CACHE: dict = {}


def _load_parmed(p: Path, *, retries: int = 4):
    """``parmed.load_file`` with a bounded retry against the transient
    ParmEd UnboundLocalError fault (see ``parse_topology._load_parmed``)."""
    last_exc: Exception | None = None
    for _attempt in range(retries):
        try:
            return pmd.load_file(str(p))
        except Exception as exc:  # retried; re-raised below if persistent
            last_exc = exc
    assert last_exc is not None
    raise last_exc


def parse_pdb(path: str) -> PdbAtoms:
    """Read a PDB file into a PdbAtoms record (coords in nm).

    Raises ``FileNotFoundError`` with a friendly message when ``path`` does
    not exist, instead of leaving the user to interpret parmed's lower-level
    error. The result is cached per (resolved path, mtime).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"PDB file not found: {path!s} "
            f"(resolved to {p.resolve()!s}). "
            f"Pass an absolute path or check the working directory.")
    cache_key = (str(p.resolve()), p.stat().st_mtime_ns)
    cached = _PDB_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        s = _load_parmed(p)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse PDB file {path!s}: {exc}. "
            f"Verify the file is a valid PDB (e.g. produced by gmx pdb2gmx)."
        ) from exc
    pos = np.asarray(s.coordinates, dtype=np.float64) * _ANGSTROM_TO_NM
    names = [a.name for a in s.atoms]
    resids = np.array([a.residue.number for a in s.atoms], dtype=np.int64)
    resnames = [a.residue.name for a in s.atoms]
    elements = [a.element_name for a in s.atoms]
    result = PdbAtoms(pos=pos, names=names, resids=resids,
                      resnames=resnames, elements=elements)
    _PDB_CACHE[cache_key] = result
    return result
