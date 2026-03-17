"""Tests for the zones subset feature of Hierarchy and InteractionHierarchy."""

import networkx as nx
import numpy as np
import pytest

from hierx import (
    HAS_H5PY,
    Hierarchy,
    InteractionHierarchy,
    generate_grid_network,
    load_hierarchy,
    load_interaction,
    save_hierarchy,
    save_interaction,
)
from hierx.storage import _extract_zone_coords

# ---------------------------------------------------------------------------
# Backend parametrisation
# ---------------------------------------------------------------------------

_backends = ["npz"]
if HAS_H5PY:
    _backends.append("hdf5")


@pytest.fixture(params=_backends)
def backend(request):
    return request.param


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestZonesSubsetConstruction:
    def test_zones_subset_basic(self):
        """Only specified zones should appear in the hierarchy."""
        G = generate_grid_network(5, 5, spacing=1000)
        zones = [0, 5, 10, 15, 20]
        h = Hierarchy(G, base_radius=2000, increase_factor=2, overlap_factor=1.5, zones=zones)

        assert h.zones == sorted(zones)
        for radius in h.radii:
            # All group keys should be zones
            assert set(h.groups[radius].keys()) == set(zones)
            # All representatives should be zones
            for rep in h.repr_zones[radius]:
                assert rep in zones

    def test_zones_sorting(self):
        """Unsorted zones input should produce sorted zones list."""
        G = generate_grid_network(5, 5, spacing=1000)
        zones = [20, 0, 15, 5, 10]
        h = Hierarchy(G, base_radius=2000, increase_factor=2, overlap_factor=1.5, zones=zones)
        assert h.zones == [0, 5, 10, 15, 20]

    def test_zones_none_equals_all_nodes(self):
        """zones=None and zones=list(G.nodes()) should produce identical results."""
        G = generate_grid_network(3, 3, spacing=1000)
        h_none = Hierarchy(G, base_radius=2000, increase_factor=2, overlap_factor=1.5, zones=None)
        h_all = Hierarchy(
            G,
            base_radius=2000,
            increase_factor=2,
            overlap_factor=1.5,
            zones=list(G.nodes()),
        )

        assert h_none.zones == h_all.zones
        assert h_none.radii == h_all.radii
        for src in h_none.zones[:3]:
            for dst in h_none.zones[:3]:
                assert h_none.get_cost(src, dst) == pytest.approx(h_all.get_cost(src, dst))

    def test_validates_invalid_nodes(self):
        """zones with non-existent node should raise ValueError."""
        G = generate_grid_network(3, 3, spacing=1000)
        with pytest.raises(ValueError, match="not in network"):
            Hierarchy(G, base_radius=2000, zones=[0, 1, 9999])

    def test_validates_empty(self):
        """zones=[] should raise ValueError."""
        G = generate_grid_network(3, 3, spacing=1000)
        with pytest.raises(ValueError, match="non-empty"):
            Hierarchy(G, base_radius=2000, zones=[])

    def test_routing_through_non_zone_nodes(self):
        """Dijkstra should route through non-zone nodes for correct costs."""
        # Linear graph: 0 --10-- 1 --10-- 2 --10-- 3
        G = nx.Graph()
        G.add_edge(0, 1, cost=10)
        G.add_edge(1, 2, cost=10)
        G.add_edge(2, 3, cost=10)

        # Only endpoints are zones; nodes 1, 2 are intermediate routing nodes
        h = Hierarchy(G, base_radius=50, increase_factor=2, overlap_factor=2.0, zones=[0, 3])

        assert h.zones == [0, 3]
        cost = h.get_cost(0, 3)
        assert cost == pytest.approx(30.0), f"Expected 30.0, got {cost}"


# ---------------------------------------------------------------------------
# Interaction tests
# ---------------------------------------------------------------------------


class TestZonesSubsetInteraction:
    @pytest.fixture
    def setup(self):
        G = generate_grid_network(5, 5, spacing=1000)
        zones = [0, 5, 10, 15, 20]
        h = Hierarchy(G, base_radius=2000, increase_factor=2, overlap_factor=1.5, zones=zones)

        def fn(c):
            return (c + 1000) ** (-2)

        ih = InteractionHierarchy(h, fn)
        return h, ih, zones

    def test_matvec_shape(self, setup):
        """Activity vector should be sized to zone count, not node count."""
        h, ih, zones = setup
        activity = np.ones(len(zones))
        result = ih.matvec(activity)
        assert result.shape == (len(zones),)

    def test_group_matrix_shape(self, setup):
        """G[k] should have columns == len(zones)."""
        h, ih, zones = setup
        for radius in h.radii:
            assert ih.G[radius].shape[1] == len(zones)

    def test_non_negativity(self, setup):
        """matvec results should be non-negative."""
        h, ih, zones = setup
        activity = np.ones(len(zones))
        result = ih.matvec(activity)
        assert np.all(result >= 0)

    def test_wrong_shape_rejected(self, setup):
        """Activity vector with wrong size should be rejected."""
        h, ih, zones = setup
        n_nodes = len(h.network.nodes())
        if n_nodes != len(zones):
            with pytest.raises((ValueError, IndexError)):
                ih.matvec(np.ones(n_nodes))

    def test_get_row(self, setup):
        """get_row should return array of shape (n_zones,)."""
        h, ih, zones = setup
        row = ih.get_row(0)
        assert row.shape == (len(zones),)


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------


class TestZonesSubsetStorage:
    @pytest.fixture
    def setup(self, tmp_path):
        G = generate_grid_network(5, 5, spacing=1000)
        zones = [0, 5, 10, 15, 20]
        h = Hierarchy(G, base_radius=2000, increase_factor=2, overlap_factor=1.5, zones=zones)
        return h, G, zones, tmp_path

    def test_save_load_zones_preserved(self, setup, backend):
        """Round-trip should preserve zones list."""
        h, G, zones, tmp_path = setup
        p = save_hierarchy(h, tmp_path / f"h_{backend}", backend=backend, include_network=True)
        loaded = load_hierarchy(p)
        assert loaded.zones == h.zones

    def test_save_load_costs_identical(self, setup, backend):
        """get_cost should match after round-trip."""
        h, G, zones, tmp_path = setup
        p = save_hierarchy(h, tmp_path / f"h_{backend}", backend=backend, include_network=True)
        loaded = load_hierarchy(p)
        for src in zones:
            for dst in zones:
                assert loaded.get_cost(src, dst) == pytest.approx(h.get_cost(src, dst))

    def test_zone_coords_subset_only(self, setup):
        """_extract_zone_coords should return len(zones) coords, not len(nodes)."""
        h, G, zones, tmp_path = setup
        coords = _extract_zone_coords(h)
        assert len(coords["zone_x"]) == len(zones)

    def test_interaction_save_load_matvec(self, setup, backend):
        """InteractionHierarchy round-trip should preserve matvec."""
        h, G, zones, tmp_path = setup

        def fn(c):
            return (c + 1000) ** (-2)

        ih = InteractionHierarchy(h, fn)

        p = save_interaction(ih, tmp_path / f"ih_{backend}", backend=backend)
        loaded = load_interaction(p, interaction_fn=fn)

        activity = np.ones(len(zones))
        np.testing.assert_allclose(loaded.matvec(activity), ih.matvec(activity), rtol=1e-12)
