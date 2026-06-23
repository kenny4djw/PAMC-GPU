"""Parse GROMACS topology (top/itp) into per-atom q, C6, C12.

parmed gives the atom list, charges, type names and connectivity. LJ self-
coefficients C6/C12 are obtained comb-rule-aware:
  * comb-rule 1: the two [atomtypes] columns ARE C6, C12 (kJ/mol nm^6, nm^12).
    parmed mis-reads them as sigma/epsilon, so we read the raw columns ourselves
    (only two numeric columns from [atomtypes] -- not a full GROMACS parser).
  * comb-rule 2/3: columns are sigma (nm), epsilon (kJ/mol); parmed reads them
    correctly and we form C6_i=4 eps sig^6, C12_i=4 eps sig^12.
Geometric pair combination (rule 1/3); the fixture is comb-rule 1.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parmed as pmd

from ptmc.model.structures import Atoms
from ptmc.io.parse_pdb import PdbAtoms

logger = logging.getLogger(__name__)

_NM_PER_ANGSTROM = 0.1
_KJ_PER_KCAL = 4.184
_PTYPES = {"A", "S", "V", "D"}


@dataclass
class TopologyParams:
    """Per-atom non-bonded parameters from a GROMACS topology.

    SHAPE/UNITS: q (N,) e; c6 (N,) kJ/mol nm^6; c12 (N,) kJ/mol nm^12.
    """
    q: np.ndarray
    c6: np.ndarray
    c12: np.ndarray
    types: list
    names: list
    comb_rule: int


def _expand_includes(path: str, _seen: set | None = None) -> list:
    """Return all lines of `path` with #include files inlined (text-level)."""
    _seen = _seen or set()
    path = os.path.abspath(path)
    if path in _seen:
        return []
    _seen.add(path)
    base = os.path.dirname(path)
    out: list = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r'\s*#include\s+"([^"]+)"', line)
            if m:
                inc = os.path.join(base, m.group(1))
                if os.path.exists(inc):
                    out.extend(_expand_includes(inc, _seen))
                else:
                    logger.warning("#include file not found: %s (referenced from %s)",
                                   inc, path)
                continue
            out.append(line)
    return out


def _parse_atomtypes_raw(top_path: str) -> dict:
    """Extract {type_name: (c6, c12)} from the raw [atomtypes] section.

    Locate the ptype token (single letter in {A,S,V,D}); the next two tokens are
    the LJ columns (C6, C12 for comb-rule 1).
    """
    types: dict = {}
    in_block = False
    continuation = ""
    for raw in _expand_includes(top_path):
        line = raw.split(";", 1)[0].strip()
        # GROMACS continuation lines: a trailing backslash joins the next line.
        if line.endswith("\\"):
            continuation += line[:-1].strip() + " "
            continue
        if continuation:
            line = continuation + line
            continuation = ""
        if not line:
            continue
        if line.startswith("["):
            in_block = (line.replace("[", "").replace("]", "").strip()
                        == "atomtypes")
            continue
        if not in_block:
            continue
        toks = line.split()
        pidx = next((i for i, t in enumerate(toks) if t in _PTYPES), None)
        if pidx is None or pidx + 2 >= len(toks):
            continue
        types[toks[0]] = (float(toks[pidx + 1]), float(toks[pidx + 2]))
    return types


# Module-level cache. Parsing the same topology repeatedly in one process
# (e.g. once per PA seed in a multi-seed sweep) both wastes work and exposes
# an intermittent UnboundLocalError ("needs_indexing") inside ParmEd's
# GROMACS reader that is triggered by state accumulated across repeated
# load_file calls (observed: 4 successful parses then a failure on the 5th in
# a single process). Caching by (resolved path, mtime) means ParmEd is
# invoked once per unique file, so a multi-seed run parses the topology a
# single time. TopologyParams is treated as read-only by all callers.
_TOPO_CACHE: dict = {}


def _load_parmed(p: Path, *, retries: int = 4):
    """``parmed.load_file`` with a bounded retry.

    ParmEd's GROMACS reader intermittently raises an internal
    ``UnboundLocalError`` ("needs_indexing") on otherwise-valid topologies; the
    fault is transient (a fresh attempt typically succeeds), so retry a few
    times before surfacing the error to the caller.
    """
    last_exc: Exception | None = None
    for _attempt in range(retries):
        try:
            return pmd.load_file(str(p), parametrize=True)
        except Exception as exc:  # retried; re-raised below if persistent
            last_exc = exc
    assert last_exc is not None
    raise last_exc


def parse_topology(top_path: str) -> TopologyParams:
    """Parse a GROMACS top into per-atom q, C6, C12 (comb-rule aware).

    Raises ``FileNotFoundError`` if ``top_path`` does not exist and
    ``ValueError`` if parmed cannot parse the file (instead of letting the
    user see a raw parmed exception).

    The result is cached per (resolved path, mtime); repeated calls in the
    same process (e.g. one per seed) reuse the first parse.
    """
    p = Path(top_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"GROMACS topology not found: {top_path!s} "
            f"(resolved to {p.resolve()!s}). "
            f"Pass an absolute path or check the working directory.")
    cache_key = (str(p.resolve()), p.stat().st_mtime_ns)
    cached = _TOPO_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        top = _load_parmed(p)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse GROMACS topology {top_path!s}: {exc}. "
            f"Verify the file is a valid .top (with #include resolvable from "
            f"its directory) — typically produced by gmx pdb2gmx."
        ) from exc
    comb_rule = int(top.defaults.comb_rule)
    names = [a.name for a in top.atoms]
    types = [a.type for a in top.atoms]
    q = np.array([a.charge for a in top.atoms], dtype=np.float64)

    if comb_rule == 1:
        type_lj = _parse_atomtypes_raw(top_path)
        missing = sorted({t for t in types if t not in type_lj})
        if missing:
            raise ValueError(f"atom types missing from [atomtypes]: {missing}")
        c6 = np.array([type_lj[t][0] for t in types], dtype=np.float64)
        c12 = np.array([type_lj[t][1] for t in types], dtype=np.float64)
    elif comb_rule in (2, 3):
        sig_nm = np.array([a.sigma * _NM_PER_ANGSTROM for a in top.atoms])
        eps_kj = np.array([a.epsilon * _KJ_PER_KCAL for a in top.atoms])
        c6 = 4.0 * eps_kj * sig_nm ** 6
        c12 = 4.0 * eps_kj * sig_nm ** 12
    else:
        raise ValueError(f"unsupported comb-rule {comb_rule}")

    result = TopologyParams(q=q, c6=c6, c12=c12, types=types,
                            names=names, comb_rule=comb_rule)
    _TOPO_CACHE[cache_key] = result
    return result


def build_atoms(pdb: PdbAtoms, topo: TopologyParams) -> Atoms:
    """Combine PDB coords with topology params into an Atoms model.

    Asserts atom-count and name-order alignment between PDB and topology.
    """
    if pdb.n != len(topo.names):
        top_n = len(topo.names)
        extra_in_top = top_n > pdb.n and topo.names[pdb.n:pdb.n+5]
        extra_in_pdb = pdb.n > top_n and pdb.names[top_n:top_n+5]
        hint = ""
        if extra_in_top:
            hint = f" (top has extra atoms e.g. {extra_in_top})"
        elif extra_in_pdb:
            hint = f" (PDB has extra atoms e.g. {extra_in_pdb})"
        raise ValueError(
            f"atom count mismatch: PDB {pdb.n} vs topology {len(topo.names)}{hint}. "
            f"Topology may have been generated from a different PDB "
            f"(e.g. GROMACS pdb2gmx adds hydrogens; use the _processed.pdb)."
        )
    if pdb.names != topo.names:
        diffs = [(i, pdb.names[i], topo.names[i])
                 for i in range(len(pdb.names))
                 if pdb.names[i] != topo.names[i]]
        n_show = min(5, len(diffs))
        detail = "; ".join(f"pos {i}: PDB={p} vs top={t}"
                          for i, p, t in diffs[:n_show])
        raise ValueError(
            f"atom name order mismatch at {len(diffs)} positions: {detail}"
        )
    return Atoms(
        pos0=pdb.pos - pdb.pos.mean(axis=0), q=topo.q, c6=topo.c6, c12=topo.c12,
        names=pdb.names, resids=pdb.resids,
        resnames=pdb.resnames, elements=pdb.elements,
    )
