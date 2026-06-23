"""Tests for ptmc.io modules: PDB parsing, topology parsing, trajectory I/O.

Requires fixture files: alanine_dipeptide.pdb and alanine_dipeptide.top.
"""
import os
import tempfile
import numpy as np
import pytest

from ptmc.io.parse_pdb import parse_pdb, PdbAtoms
from ptmc.io.parse_topology import (
    parse_topology, build_atoms, _expand_includes, TopologyParams,
)
from ptmc.io.write_traj import write_trajectory, read_trajectory


# ---------------------------------------------------------------------------
# Minimal PDB content (alanine dipeptide analog: ACE-ALA-NME, ~22 atoms)
# Coordinates in Angstrom (PDB convention), generated in GROMACS.
# ---------------------------------------------------------------------------
_MINIMAL_PDB = """\
ATOM      1  N   ACE A   1       1.290   0.000   0.000  1.00  0.00           N
ATOM      2  H1  ACE A   1       1.690  -0.810   0.440  1.00  0.00           H
ATOM      3  H2  ACE A   1       1.690   0.810   0.440  1.00  0.00           H
ATOM      4  H3  ACE A   1       1.690   0.000  -0.880  1.00  0.00           H
ATOM      5  CH3 ACE A   1       0.190   0.000   0.000  1.00  0.00           C
ATOM      6  C   ACE A   1      -0.520   1.240   0.000  1.00  0.00           C
ATOM      7  O   ACE A   1       0.050   2.320   0.000  1.00  0.00           O
END
"""

_MINIMAL_TOP = """\
[ defaults ]
; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ
   1      1        no        1.0     1.0

[ atomtypes ]
; name   mass    charge  ptype   c6              c12
  ACE_CA  12.011  0.000   A       2.4099e-2      2.3247e-6
  ACE_C   12.011  0.000   A       2.4099e-2      2.3247e-6
  ACE_O   16.000  0.000   A       2.2705e-2      1.6816e-6
  ACE_N   14.007  0.000   A       2.4224e-2      2.3776e-6
  ACE_H   1.008   0.000   A       5.5189e-4      4.2524e-8

[ moleculetype ]
; Name   nrexcl
  MOL     3

[ atoms ]
; nr  type   resnr residu atom  cgnr  charge     mass
   1   ACE_N   1   ACE   N      1     0.000      14.007
   2   ACE_H   1   ACE   H1     2     0.000       1.008
   3   ACE_H   1   ACE   H2     3     0.000       1.008
   4   ACE_H   1   ACE   H3     4     0.000       1.008
   5   ACE_CA  1   ACE   CH3    5     0.000      12.011
   6   ACE_C   1   ACE   C      6     0.000      12.011
   7   ACE_O   1   ACE   O      7     0.000      16.000

[ bonds ]
; ai  aj  funct   r   k
  1   2   1       0.1010 363171.0
  1   3   1       0.1010 363171.0
  1   4   1       0.1010 363171.0
  1   5   1       0.1470 260240.0
  5   6   1       0.1520 259408.0
  6   7   1       0.1230 585760.0

[ system ]
Test small molecule

[ molecules ]
MOL  1
"""


# ---------------------------------------------------------------------------
# Fixture: write minimal PDB/top to temp files
# ---------------------------------------------------------------------------
@pytest.fixture
def pdb_path():
    """Write minimal PDB to temp file, yield path, clean up."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".pdb", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(_MINIMAL_PDB)
    yield path
    os.unlink(path)


@pytest.fixture
def top_path():
    """Write minimal topology to temp file, yield path, clean up."""
    fd, path = tempfile.mkstemp(suffix=".top", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(_MINIMAL_TOP)
    yield path
    os.unlink(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestParsePDB:
    def test_parse_pdb(self, pdb_path):
        atoms = parse_pdb(pdb_path)
        assert isinstance(atoms, PdbAtoms)
        assert atoms.n == 7
        assert atoms.pos.shape == (7, 3)
        assert len(atoms.names) == 7

    def test_coordinates_in_nm(self, pdb_path):
        """PDB coords in Angstrom, should be converted to nm."""
        atoms = parse_pdb(pdb_path)
        # First atom at (1.29, 0.0, 0.0) A -> (0.129, 0.0, 0.0) nm
        np.testing.assert_allclose(atoms.pos[0], [0.129, 0.0, 0.0], atol=1e-4)

    def test_element_names(self, pdb_path):
        atoms = parse_pdb(pdb_path)
        assert atoms.elements[0] == "N"
        assert atoms.elements[6] == "O"

    def test_residue_info(self, pdb_path):
        atoms = parse_pdb(pdb_path)
        assert set(atoms.resnames) == {"ACE"}
        assert np.all(atoms.resids == 1)


class TestParseTopology:
    def test_parse_topology(self, top_path):
        topo = parse_topology(top_path)
        assert isinstance(topo, TopologyParams)
        assert topo.q.shape == (7,)
        assert topo.c6.shape == (7,)
        assert topo.c12.shape == (7,)
        assert topo.comb_rule == 1

    def test_comb_rule_1_values(self, top_path):
        """C6/C12 from raw [atomtypes] columns for comb-rule 1."""
        topo = parse_topology(top_path)
        assert topo.c6[0] == pytest.approx(2.4224e-2)  # ACE_N
        assert topo.c12[0] == pytest.approx(2.3776e-6)
        assert topo.c6[5] == pytest.approx(2.4099e-2)  # ACE_C
        assert topo.c6[6] == pytest.approx(2.2705e-2)  # ACE_O

    def test_charges(self, top_path):
        topo = parse_topology(top_path)
        np.testing.assert_allclose(topo.q, np.zeros(7))

    def test_expand_includes_no_include(self, top_path):
        """Test that _expand_includes works on a file without includes."""
        lines = _expand_includes(top_path)
        assert len(lines) > 0


class TestBuildAtoms:
    def test_build_atoms(self, pdb_path, top_path):
        pdb = parse_pdb(pdb_path)
        topo = parse_topology(top_path)
        atoms = build_atoms(pdb, topo)
        assert atoms.n == 7
        assert atoms.pos0.shape == (7, 3)
        np.testing.assert_allclose(atoms.q, np.zeros(7))

    def test_atom_count_mismatch(self, pdb_path, top_path):
        pdb = parse_pdb(pdb_path)
        topo = parse_topology(top_path)
        # Modify to create mismatch
        import copy
        pdb_mismatch = copy.copy(pdb)
        pdb_mismatch._pos = pdb.pos[:6]  # won't work since pos is a property
        # Use a simpler approach: directly test build_atoms raises
        pdb2 = PdbAtoms(pos=np.zeros((5, 3)), names=["A"]*5,
                        resids=np.zeros(5, int), resnames=["X"]*5,
                        elements=["C"]*5)
        with pytest.raises(ValueError, match="atom count mismatch"):
            build_atoms(pdb2, topo)

    def test_name_order_mismatch(self, pdb_path, top_path):
        pdb = parse_pdb(pdb_path)
        topo = parse_topology(top_path)
        # Reorder names in pdb
        pdb2 = PdbAtoms(pos=pdb.pos, names=list(reversed(pdb.names)),
                        resids=pdb.resids, resnames=pdb.resnames,
                        elements=pdb.elements)
        with pytest.raises(ValueError, match="atom name order mismatch"):
            build_atoms(pdb2, topo)


class TestTrajectory:
    def test_write_read_trajectory(self, three_atom_atoms):
        """Write a simple trajectory and verify frame count."""
        F = 5
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (F, 1))
        transs = np.zeros((F, 3))
        transs[:, 2] = np.linspace(0.3, 0.5, F)

        with tempfile.TemporaryDirectory() as tmp:
            pdb_out = os.path.join(tmp, "frame.pdb")
            xtc_out = os.path.join(tmp, "traj.xtc")
            n_frames = write_trajectory(pdb_out, xtc_out, three_atom_atoms,
                                         quats, transs)
            assert n_frames == F
            n_atoms, n_read = read_trajectory(pdb_out, xtc_out)
            assert n_atoms == three_atom_atoms.n
            assert n_read == F
