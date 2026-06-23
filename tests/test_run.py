"""Tests for ptmc.run: summarization, parquet output, CLI."""
import numpy as np
import pytest

from ptmc.run import summarize_systems, write_parquet


class FakeBetas:
    """Make a mock-out object to fake beta()."""
    pass


class TestSummarizeSystems:
    def test_single_system_two_basins(self):
        """Verify output DataFrame structure from a minimal run_systems output."""
        S, C = 1, 4  # 1 system, 4 chains
        out = {
            "quats": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (S, C, 1)),
            "system_ids": [42],
            # Note: cluster_orientations needs diverse normals.
            # With identity quats all same, kmeans may produce 1 cluster.
            # Use diverse quats to get 2 clusters.
        }
        # Use random-ish final quats to get multiple basins
        rng = np.random.default_rng(42)
        quats = rng.normal(size=(S, C, 4))
        quats = quats / np.linalg.norm(quats, axis=2, keepdims=True)
        out["quats"] = quats

        betas = [1.0]
        df = summarize_systems(out, betas, k=2, top_k=2)
        assert len(df) == 2  # 1 system x 2 basins
        # Required base columns must all be present (order-insensitive so
        # later additions of health-flag columns don't break the test).
        required = {
            "system_id", "basin_rank", "population", "dG_kJ_mol",
            "normal_x", "normal_y", "normal_z", "tilt_deg",
        }
        assert required.issubset(set(df.columns))
        assert df["system_id"].iloc[0] == 42
        assert df["basin_rank"].iloc[0] == 0
        assert df["basin_rank"].iloc[1] == 1

    def test_multiple_systems(self):
        S, C = 3, 8
        rng = np.random.default_rng(42)
        quats = rng.normal(size=(S, C, 4))
        quats = quats / np.linalg.norm(quats, axis=2, keepdims=True)
        out = {"quats": quats, "system_ids": [10, 20, 30]}
        betas = [1.0, 1.5, 2.0]
        df = summarize_systems(out, betas, k=2, top_k=2)
        assert len(df) == 3 * 2  # S * top_k
        assert set(df["system_id"]) == {10, 20, 30}

    def test_populations_sum_to_one(self):
        S, C = 1, 16
        rng = np.random.default_rng(42)
        quats = rng.normal(size=(S, C, 4))
        quats = quats / np.linalg.norm(quats, axis=2, keepdims=True)
        out = {"quats": quats, "system_ids": [0]}
        df = summarize_systems(out, [1.0], k=2, top_k=2)
        assert df["population"].sum() == pytest.approx(1.0, abs=0.01)

    def test_tilt_angle_range(self):
        S, C = 1, 8
        rng = np.random.default_rng(42)
        quats = rng.normal(size=(S, C, 4))
        quats = quats / np.linalg.norm(quats, axis=2, keepdims=True)
        out = {"quats": quats, "system_ids": [0]}
        df = summarize_systems(out, [1.0], k=2, top_k=2)
        assert df["tilt_deg"].between(0, 180).all()


class TestWriteParquet:
    def test_write_parquet(self):
        import pandas as pd
        import tempfile, os
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            path = f.name
            result = write_parquet(df, path)
            assert os.path.exists(path)
            assert result == path
            # Read back
            df2 = pd.read_parquet(path)
            assert len(df2) == 2


class TestCLI:
    @pytest.mark.skip(reason="requires real PDB/top files")
    def test_main(self):
        """Placeholder for CLI integration test with real fixtures."""
        pass

    def test_main_help(self):
        """At minimum, --help should not error."""
        from ptmc.run import main
        import sys
        try:
            main(["--help"])
        except SystemExit as e:
            assert e.code in (0, None)

    def test_parser_drops_ht(self):
        """--sampler ht is no longer a valid choice."""
        from ptmc.run import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--pdb", "x", "--top", "y", "--sampler", "ht"])

    def test_parser_default_sampler_is_pa(self):
        """With HT gone, PA is the sensible default — it emits ΔG⁰_ads."""
        from ptmc.run import build_parser
        parser = build_parser()
        args = parser.parse_args(["--pdb", "x", "--top", "y"])
        assert args.sampler == "pa"

    def test_parser_flexible_off_by_default(self):
        from ptmc.run import build_parser
        parser = build_parser()
        args = parser.parse_args(["--pdb", "x", "--top", "y"])
        assert args.flexible is False
        assert args.sigma_chi == pytest.approx(0.2)

    def test_parser_flexible_on(self):
        from ptmc.run import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--pdb", "x", "--top", "y",
            "--flexible", "--sigma-chi", "0.05",
        ])
        assert args.flexible is True
        assert args.sigma_chi == pytest.approx(0.05)


class TestFlexibleGate:
    """--flexible is now wired into PA/PT. Verify it no longer raises
    NotImplementedError and the chi-aware path is reachable."""

    def test_run_pipeline_flexible_no_longer_raises_notimplemented(self):
        """run_pipeline(config.mc.flexible=True) used to raise NotImplementedError.
        Now the gate is removed — it should proceed past the flexible setup
        and fail later on a real file-not-found or energy evaluation (not a
        NotImplementedError), proving the chi-aware path is wired.
        """
        pytest.importorskip("parmed")

        from ptmc.config import SimConfig, MCConfig
        from ptmc.sampler.pipeline import run_pipeline
        import ptmc.sampler.pipeline as pipeline_mod

        cfg = SimConfig(
            surface_type="continuum",
            sampler="pa",
            mc=MCConfig(flexible=True, flexible_ack_experimental=True),
        )
        cfg.pdb_path = "/no/such/file.pdb"
        cfg.top_path = "/no/such/file.top"

        # Stub parse + build so we reach the flexible setup without real files.
        class _DummyAtoms:
            import numpy as _np
            pos0 = _np.zeros((1, 3))
            q = _np.zeros((1,))
            c6 = _np.zeros((1,))
            c12 = _np.zeros((1,))
            n = 1
            net_charge = 0.0

        orig_parse_pdb = pipeline_mod.parse_pdb
        orig_parse_top = pipeline_mod.parse_topology
        orig_build = pipeline_mod.build_atoms
        pipeline_mod.parse_pdb = lambda p: None
        pipeline_mod.parse_topology = lambda p: None
        pipeline_mod.build_atoms = lambda pdb, top: _DummyAtoms()
        try:
            # The flexible setup calls build_chi_topology which calls parmed.
            # With a stub .top path, parmed will raise FileNotFoundError or
            # ValueError — NOT NotImplementedError.
            with pytest.raises(Exception) as excinfo:
                run_pipeline(cfg)
            msg = str(excinfo.value)
            # The gate is gone — must NOT be NotImplementedError any more.
            assert "notimplemented" not in msg.lower()
            assert "flexible=True is not yet wired" not in msg
            # The error should come from parmed trying to load the fake .top.
            assert "topology not found" in msg.lower() or "not found" in msg.lower() or "no such file" in msg.lower()
        finally:
            pipeline_mod.parse_pdb = orig_parse_pdb
            pipeline_mod.parse_topology = orig_parse_top
            pipeline_mod.build_atoms = orig_build

    def test_mcconfig_flexible_defaults(self):
        from ptmc.config import MCConfig
        mc = MCConfig()
        assert mc.flexible is False
        assert mc.sigma_chi == pytest.approx(0.2)

    def test_simconfig_default_sampler_is_pa(self):
        from ptmc.config import SimConfig
        assert SimConfig().sampler == "pa"
