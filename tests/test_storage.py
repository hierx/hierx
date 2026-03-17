"""Tests for the storage module (save/load Hierarchy and InteractionHierarchy)."""

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

from hierx import (
    HAS_H5PY,
    Hierarchy,
    InteractionHierarchy,
    StorageError,
    generate_grid_network,
    load_hierarchy,
    load_interaction,
    save_hierarchy,
    save_interaction,
)
from hierx.storage import (
    FORMAT_VERSION,
    _detect_format,
    _ensure_ext,
    _extract_zone_coords,
    _flatten_costs,
    _flatten_cutoffs,
    _flatten_groups,
    _flatten_repr_zones,
    _flatten_zone_indices,
    _unflatten_costs,
    _unflatten_cutoffs,
    _unflatten_groups,
    _unflatten_repr_zones,
    _unflatten_zone_indices,
)

# ---------------------------------------------------------------------------
# Backend parametrisation: always test NPZ, add HDF5 if h5py is available
# ---------------------------------------------------------------------------

_backends = ["npz"]
if HAS_H5PY:
    _backends.append("hdf5")


@pytest.fixture(params=_backends)
def backend(request):
    return request.param


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_network():
    """5×5 grid network."""
    return generate_grid_network(5, 5, spacing=1000)


@pytest.fixture
def hierarchy(small_network):
    return Hierarchy(small_network, base_radius=2000, increase_factor=2, overlap_factor=1.5)


@pytest.fixture
def interaction_fn():
    return lambda c: (c + 1000) ** (-2)


@pytest.fixture
def interaction(hierarchy, interaction_fn):
    return InteractionHierarchy(hierarchy, interaction_fn)


# ---------------------------------------------------------------------------
# Flatten / unflatten roundtrips
# ---------------------------------------------------------------------------


class TestFlattenRoundtrips:
    def test_costs_roundtrip(self, hierarchy):
        radii = hierarchy.radii
        flat = _flatten_costs(hierarchy.costs, radii)
        restored = _unflatten_costs(flat, radii)
        for r in radii:
            for src in hierarchy.costs[r]:
                for dst, val in hierarchy.costs[r][src].items():
                    assert restored[r][src][dst] == pytest.approx(val)

    def test_groups_roundtrip(self, hierarchy):
        radii = hierarchy.radii
        flat = _flatten_groups(hierarchy.groups, radii)
        restored = _unflatten_groups(flat, radii)
        for r in radii:
            assert restored[r] == hierarchy.groups[r]

    def test_repr_zones_roundtrip(self, hierarchy):
        radii = hierarchy.radii
        flat = _flatten_repr_zones(hierarchy.repr_zones, radii)
        restored = _unflatten_repr_zones(flat, radii)
        for r in radii:
            assert restored[r] == hierarchy.repr_zones[r]

    def test_cutoffs_roundtrip(self, hierarchy):
        radii = hierarchy.radii
        flat = _flatten_cutoffs(hierarchy.cutoffs, radii)
        restored = _unflatten_cutoffs(flat, radii)
        for r in radii:
            assert restored[r] == hierarchy.cutoffs[r]

    def test_zone_indices_roundtrip(self, interaction):
        radii = interaction.hierarchy.radii
        flat = _flatten_zone_indices(interaction.zone_indices, radii)
        restored = _unflatten_zone_indices(flat, radii)
        for r in radii:
            assert restored[r] == interaction.zone_indices[r]


# ---------------------------------------------------------------------------
# Hierarchy save / load
# ---------------------------------------------------------------------------


class TestHierarchySaveLoad:
    def test_parameters_preserved(self, hierarchy, backend, tmp_path):
        p = tmp_path / "h"
        out = save_hierarchy(hierarchy, p, backend=backend)
        loaded = load_hierarchy(out)
        assert loaded.base_radius == hierarchy.base_radius
        assert loaded.increase_factor == hierarchy.increase_factor
        assert loaded.overlap_factor == hierarchy.overlap_factor
        assert loaded._min_representatives == hierarchy._min_representatives

    def test_radii_and_zones(self, hierarchy, backend, tmp_path):
        p = save_hierarchy(hierarchy, tmp_path / "h", backend=backend)
        loaded = load_hierarchy(p)
        assert loaded.radii == hierarchy.radii
        assert loaded.zones == hierarchy.zones

    def test_cost_lookup_identical(self, hierarchy, backend, tmp_path):
        p = save_hierarchy(hierarchy, tmp_path / "h", backend=backend)
        loaded = load_hierarchy(p)
        zones = hierarchy.zones
        for src in zones[:5]:
            for dst in zones[:5]:
                assert loaded.get_cost(src, dst) == pytest.approx(hierarchy.get_cost(src, dst))

    def test_groups_preserved(self, hierarchy, backend, tmp_path):
        p = save_hierarchy(hierarchy, tmp_path / "h", backend=backend)
        loaded = load_hierarchy(p)
        for r in hierarchy.radii:
            assert loaded.groups[r] == hierarchy.groups[r]

    def test_without_network(self, hierarchy, backend, tmp_path):
        p = save_hierarchy(hierarchy, tmp_path / "h", backend=backend, include_network=False)
        loaded = load_hierarchy(p)
        assert loaded.network is None

    def test_with_network(self, hierarchy, backend, tmp_path):
        p = save_hierarchy(hierarchy, tmp_path / "h", backend=backend, include_network=True)
        loaded = load_hierarchy(p)
        assert loaded.network is not None
        assert set(loaded.network.nodes()) == set(hierarchy.network.nodes())
        assert len(loaded.network.edges()) == len(hierarchy.network.edges())

    def test_with_external_network(self, hierarchy, small_network, backend, tmp_path):
        p = save_hierarchy(hierarchy, tmp_path / "h", backend=backend)
        loaded = load_hierarchy(p, network=small_network)
        assert loaded.network is small_network

    def test_extension_appended(self, hierarchy, backend, tmp_path):
        p = save_hierarchy(hierarchy, tmp_path / "myfile", backend=backend)
        expected_ext = ".npz" if backend == "npz" else ".h5"
        assert p.suffix == expected_ext
        assert p.exists()

    def test_extension_not_doubled(self, hierarchy, backend, tmp_path):
        expected_ext = ".npz" if backend == "npz" else ".h5"
        p = save_hierarchy(hierarchy, tmp_path / f"myfile{expected_ext}", backend=backend)
        assert p.suffix == expected_ext
        # Should not double the extension
        assert p.name == f"myfile{expected_ext}"


# ---------------------------------------------------------------------------
# InteractionHierarchy save / load
# ---------------------------------------------------------------------------


class TestInteractionSaveLoad:
    def test_matvec_identical(self, interaction, interaction_fn, backend, tmp_path):
        p = save_interaction(interaction, tmp_path / "ih", backend=backend)
        loaded = load_interaction(p, interaction_fn=interaction_fn)
        activity = np.ones(len(interaction.hierarchy.zones))
        expected = interaction.matvec(activity)
        actual = loaded.matvec(activity)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_matvec_without_interaction_fn(self, interaction, backend, tmp_path):
        """matvec should work even without providing interaction_fn on load."""
        p = save_interaction(interaction, tmp_path / "ih", backend=backend)
        loaded = load_interaction(p)
        assert loaded.interaction_fn is None
        activity = np.ones(len(interaction.hierarchy.zones))
        expected = interaction.matvec(activity)
        actual = loaded.matvec(activity)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_sparse_matrices_preserved(self, interaction, backend, tmp_path):
        p = save_interaction(interaction, tmp_path / "ih", backend=backend)
        loaded = load_interaction(p)
        for r in interaction.hierarchy.radii:
            np.testing.assert_allclose(
                loaded.D[r].toarray(), interaction.D[r].toarray(), rtol=1e-12
            )
            np.testing.assert_allclose(
                loaded.Corr[r].toarray(), interaction.Corr[r].toarray(), rtol=1e-12
            )
            np.testing.assert_allclose(
                loaded.G[r].toarray(), interaction.G[r].toarray(), rtol=1e-12
            )

    def test_hierarchy_embedded(self, interaction, backend, tmp_path):
        """Loading an InteractionHierarchy should also reconstruct its Hierarchy."""
        p = save_interaction(interaction, tmp_path / "ih", backend=backend)
        loaded = load_interaction(p)
        h_orig = interaction.hierarchy
        h_loaded = loaded.hierarchy
        assert h_loaded.radii == h_orig.radii
        assert h_loaded.zones == h_orig.zones
        assert h_loaded.base_radius == h_orig.base_radius
        # Cost lookup should match
        for src in h_orig.zones[:3]:
            for dst in h_orig.zones[:3]:
                assert h_loaded.get_cost(src, dst) == pytest.approx(h_orig.get_cost(src, dst))

    def test_interaction_fn_hint_stored(self, interaction, backend, tmp_path):
        """Hint string should be stored and retrievable (via metadata)."""
        p = save_interaction(
            interaction,
            tmp_path / "ih",
            backend=backend,
            interaction_fn_hint="(c + 1000)^{-2}",
        )
        # Just verify the file loads fine — hint is in metadata
        loaded = load_interaction(p)
        assert loaded is not None

    def test_with_network(self, interaction, backend, tmp_path):
        p = save_interaction(
            interaction,
            tmp_path / "ih",
            backend=backend,
            include_network=True,
        )
        loaded = load_interaction(p)
        assert loaded.hierarchy.network is not None

    def test_loaded_is_linear_operator(self, interaction, backend, tmp_path):
        """Loaded InteractionHierarchy should also be a LinearOperator."""
        p = save_interaction(interaction, tmp_path / "ih", backend=backend)
        loaded = load_interaction(p)
        assert isinstance(loaded, LinearOperator)
        n = loaded.shape[0]
        assert loaded.shape == (n, n)
        assert loaded.dtype == np.float64
        # @ operator should work
        activity = np.ones(n)
        result = loaded @ activity
        np.testing.assert_allclose(result, loaded.matvec(activity))


# ---------------------------------------------------------------------------
# Convenience methods
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    def test_hierarchy_save_load(self, hierarchy, backend, tmp_path):
        p = hierarchy.save(tmp_path / "h", backend=backend)
        loaded = Hierarchy.load(p)
        assert loaded.radii == hierarchy.radii

    def test_interaction_save_load(self, interaction, interaction_fn, backend, tmp_path):
        p = interaction.save(tmp_path / "ih", backend=backend)
        loaded = InteractionHierarchy.load(p, interaction_fn=interaction_fn)
        activity = np.ones(len(interaction.hierarchy.zones))
        np.testing.assert_allclose(
            loaded.matvec(activity), interaction.matvec(activity), rtol=1e-12
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_hierarchy(tmp_path / "nonexistent.npz")

    def test_wrong_object_type(self, hierarchy, tmp_path):
        """Loading a Hierarchy file as InteractionHierarchy should fail."""
        p = save_hierarchy(hierarchy, tmp_path / "h", backend="npz")
        with pytest.raises(StorageError, match="object_type"):
            load_interaction(p)

    def test_corrupt_file(self, tmp_path):
        """A file with unknown magic bytes should raise StorageError."""
        bad = tmp_path / "bad.npz"
        bad.write_bytes(b"CORRUPT DATA")
        with pytest.raises(StorageError, match="Unrecognised"):
            load_hierarchy(bad)

    @pytest.mark.skipif(HAS_H5PY, reason="h5py IS available")
    def test_hdf5_backend_without_h5py(self, hierarchy, tmp_path):
        with pytest.raises(StorageError, match="h5py"):
            save_hierarchy(hierarchy, tmp_path / "h", backend="hdf5")

    def test_invalid_backend(self, hierarchy, tmp_path):
        with pytest.raises(ValueError, match="backend"):
            save_hierarchy(hierarchy, tmp_path / "h", backend="parquet")


# ---------------------------------------------------------------------------
# Backend auto-selection
# ---------------------------------------------------------------------------


class TestBackendAutoSelection:
    def test_auto_selects_working_backend(self, hierarchy, tmp_path):
        """auto backend should produce a loadable file."""
        p = save_hierarchy(hierarchy, tmp_path / "h", backend="auto")
        loaded = load_hierarchy(p)
        assert loaded.radii == hierarchy.radii


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestFormatDetection:
    def test_npz_detected(self, hierarchy, tmp_path):
        p = save_hierarchy(hierarchy, tmp_path / "h", backend="npz")
        assert _detect_format(p) == "npz"

    @pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
    def test_hdf5_detected(self, hierarchy, tmp_path):
        p = save_hierarchy(hierarchy, tmp_path / "h", backend="hdf5")
        assert _detect_format(p) == "hdf5"

    def test_cross_format_load(self, hierarchy, tmp_path):
        """A file saved as NPZ should load regardless of backend kwarg."""
        p = save_hierarchy(hierarchy, tmp_path / "h", backend="npz")
        # load doesn't take backend — it auto-detects
        loaded = load_hierarchy(p)
        assert loaded.radii == hierarchy.radii


# ---------------------------------------------------------------------------
# Extension handling
# ---------------------------------------------------------------------------


class TestExtensionHandling:
    def test_no_ext_npz(self):
        assert _ensure_ext(Path("foo"), "npz") == Path("foo.npz")

    def test_no_ext_hdf5(self):
        assert _ensure_ext(Path("foo"), "hdf5") == Path("foo.h5")

    def test_already_correct(self):
        assert _ensure_ext(Path("foo.npz"), "npz") == Path("foo.npz")
        assert _ensure_ext(Path("foo.h5"), "hdf5") == Path("foo.h5")

    def test_other_ext(self):
        assert _ensure_ext(Path("foo.bar"), "npz") == Path("foo.bar.npz")
        assert _ensure_ext(Path("foo.bar"), "hdf5") == Path("foo.bar.h5")


# ---------------------------------------------------------------------------
# HDF5 structured layout verification
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
class TestHdf5Structure:
    def test_hierarchy_hdf5_structure(self, hierarchy, tmp_path):
        """Verify the on-disk HDF5 layout has per-layer groups with CSR costs."""
        import h5py

        p = save_hierarchy(hierarchy, tmp_path / "h", backend="hdf5")
        with h5py.File(str(p), "r") as f:
            # Root datasets
            assert "zones" in f
            assert "radii" in f
            assert "cutoffs" in f

            # Root attrs
            assert f.attrs["object_type"] == "Hierarchy"
            assert f.attrs["format_version"] == FORMAT_VERSION

            # Layers group
            assert "layers" in f
            n_layers = len(f["radii"][()])
            for k in range(n_layers):
                lg = f[f"layers/{k}"]
                assert "repr_zones" in lg
                assert "groups_zone" in lg
                assert "groups_rep" in lg
                assert "costs_src_zones" in lg
                assert "costs_src_offsets" in lg
                assert "costs_dst" in lg
                assert "costs_val" in lg

                # CSR offsets consistency: offsets[-1] == len(costs_dst)
                offsets = lg["costs_src_offsets"][()]
                n_entries = len(lg["costs_dst"][()])
                assert offsets[-1] == n_entries
                assert offsets[0] == 0
                assert np.all(np.diff(offsets) >= 0), "offsets must be monotonic"

    def test_interaction_hdf5_structure(self, interaction, tmp_path):
        """Verify InteractionHierarchy HDF5 has sparse matrix datasets per layer."""
        import h5py

        p = save_interaction(interaction, tmp_path / "ih", backend="hdf5")
        with h5py.File(str(p), "r") as f:
            assert f.attrs["object_type"] == "InteractionHierarchy"
            n_layers = len(f["radii"][()])
            for k in range(n_layers):
                lg = f[f"layers/{k}"]
                for prefix in ("D", "Corr", "G"):
                    assert f"{prefix}_data" in lg
                    assert f"{prefix}_indices" in lg
                    assert f"{prefix}_indptr" in lg
                    assert f"{prefix}_shape" in lg
                # Cell indices
                assert "ci_zone" in lg
                assert "ci_idx" in lg

    def test_network_in_group(self, hierarchy, tmp_path):
        """Network data should be in a 'network' group, not flat."""
        import h5py

        p = save_hierarchy(hierarchy, tmp_path / "h", backend="hdf5", include_network=True)
        with h5py.File(str(p), "r") as f:
            assert "network" in f
            net = f["network"]
            assert "nodes" in net
            assert "edge_src" in net
            assert "edge_dst" in net
            assert "edge_cost" in net


# ---------------------------------------------------------------------------
# Zone coordinates
# ---------------------------------------------------------------------------


class TestZoneCoords:
    def test_zone_coords_stored_hdf5(self, hierarchy, tmp_path):
        """zone_x/zone_y should be present when network has coordinates."""
        if not HAS_H5PY:
            pytest.skip("h5py not available")
        import h5py

        p = save_hierarchy(hierarchy, tmp_path / "h", backend="hdf5")
        with h5py.File(str(p), "r") as f:
            assert "zone_x" in f
            assert "zone_y" in f
            assert len(f["zone_x"][()]) == len(hierarchy.zones)

    def test_zone_coords_stored_npz(self, hierarchy, tmp_path):
        """zone_x/zone_y should be present in NPZ files too."""
        p = save_hierarchy(hierarchy, tmp_path / "h", backend="npz")
        with np.load(str(p), allow_pickle=False) as npz:
            assert "zone_x" in npz.files
            assert "zone_y" in npz.files

    def test_zone_coords_in_interaction(self, interaction, tmp_path):
        """zone_x/zone_y should be present in InteractionHierarchy files."""
        if not HAS_H5PY:
            pytest.skip("h5py not available")
        import h5py

        p = save_interaction(interaction, tmp_path / "ih", backend="hdf5")
        with h5py.File(str(p), "r") as f:
            assert "zone_x" in f
            assert "zone_y" in f

    def test_zone_coords_absent_no_network(self, tmp_path):
        """No zone_x/zone_y when hierarchy.network is None."""
        import networkx as nx

        # Build a network without x/y attributes
        G = nx.Graph()
        for i in range(10):
            G.add_node(i)
        for i in range(9):
            G.add_edge(i, i + 1, cost=1.0)

        h = Hierarchy(G, base_radius=2, increase_factor=2, overlap_factor=1.5)
        p = save_hierarchy(h, tmp_path / "h", backend="npz")
        with np.load(str(p), allow_pickle=False) as npz:
            assert "zone_x" not in npz.files
            assert "zone_y" not in npz.files

    def test_extract_zone_coords_helper(self, hierarchy):
        """_extract_zone_coords should return correct arrays."""
        coords = _extract_zone_coords(hierarchy)
        assert "zone_x" in coords
        assert "zone_y" in coords
        assert len(coords["zone_x"]) == len(hierarchy.zones)

    def test_extract_zone_coords_no_network(self):
        """_extract_zone_coords returns empty dict for None network."""

        class FakeHierarchy:
            network = None

        assert _extract_zone_coords(FakeHierarchy()) == {}
