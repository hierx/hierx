"""Regression tests for input validation contracts (Phase 1 hardening)."""

import networkx as nx
import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

from hierx import Hierarchy, InteractionHierarchy
from hierx.utils import (
    exponential_interaction,
    generate_grid_network,
    generate_large_spatial_network,
    generate_random_spatial_network,
    generate_transport_network,
)

# -----------------------------------------------------------------------
# Hierarchy constructor validation
# -----------------------------------------------------------------------


class TestHierarchyConstructorValidation:
    """Tests for Hierarchy.__init__ parameter validation."""

    def test_non_graph_raises_type_error(self):
        with pytest.raises(TypeError, match="networkx.Graph"):
            Hierarchy("not a graph")

    def test_empty_network_raises_value_error(self):
        G = nx.Graph()
        with pytest.raises(ValueError, match="empty"):
            Hierarchy(G)

    def test_base_radius_zero_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1, cost=10)
        with pytest.raises(ValueError, match="base_radius"):
            Hierarchy(G, base_radius=0)

    def test_base_radius_negative_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1, cost=10)
        with pytest.raises(ValueError, match="base_radius"):
            Hierarchy(G, base_radius=-5)

    def test_increase_factor_one_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1, cost=10)
        with pytest.raises(ValueError, match="increase_factor"):
            Hierarchy(G, increase_factor=1)

    def test_increase_factor_below_one_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1, cost=10)
        with pytest.raises(ValueError, match="increase_factor"):
            Hierarchy(G, increase_factor=0.5)

    def test_overlap_factor_zero_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1, cost=10)
        with pytest.raises(ValueError, match="overlap_factor"):
            Hierarchy(G, overlap_factor=0)

    def test_overlap_factor_negative_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1, cost=10)
        with pytest.raises(ValueError, match="overlap_factor"):
            Hierarchy(G, overlap_factor=-1)

    def test_n_workers_zero_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1, cost=10)
        with pytest.raises(ValueError, match="n_workers"):
            Hierarchy(G, n_workers=0)


# -----------------------------------------------------------------------
# Invalid-zone contract
# -----------------------------------------------------------------------


class TestInvalidZoneContract:
    """Tests for get_cost and get_row with invalid zone IDs."""

    def test_get_cost_invalid_source(self, toy_network):
        h = Hierarchy(toy_network, base_radius=20)
        with pytest.raises(ValueError, match="source_zone"):
            h.get_cost(999, 0)

    def test_get_cost_invalid_dest(self, toy_network):
        h = Hierarchy(toy_network, base_radius=20)
        with pytest.raises(ValueError, match="dest_zone"):
            h.get_cost(0, 999)

    def test_get_row_invalid_zone(self, toy_network):
        h = Hierarchy(toy_network, base_radius=20)

        def interaction_fn(c):
            return (c + 1000) ** (-2)

        ih = InteractionHierarchy(h, interaction_fn)
        with pytest.raises(ValueError, match="source_zone"):
            ih.get_row(999)


# -----------------------------------------------------------------------
# LinearOperator integration
# -----------------------------------------------------------------------


class TestLinearOperatorIntegration:
    """Tests that InteractionHierarchy is a proper scipy LinearOperator."""

    def _make_ih(self, toy_network):
        h = Hierarchy(toy_network, base_radius=20)
        return InteractionHierarchy(h, lambda c: (c + 1000) ** (-2))

    def test_is_linear_operator(self, toy_network):
        ih = self._make_ih(toy_network)
        assert isinstance(ih, LinearOperator)

    def test_shape_property(self, toy_network):
        ih = self._make_ih(toy_network)
        n = len(ih.hierarchy.zones)
        assert ih.shape == (n, n)

    def test_dtype_property(self, toy_network):
        ih = self._make_ih(toy_network)
        assert ih.dtype == np.float64

    def test_matmul_operator(self, toy_network):
        ih = self._make_ih(toy_network)
        n = ih.shape[0]
        activity = np.ones(n)
        result_matmul = ih @ activity
        result_matvec = ih.matvec(activity)
        np.testing.assert_array_equal(result_matmul, result_matvec)

    def test_matmat(self, toy_network):
        ih = self._make_ih(toy_network)
        n = ih.shape[0]
        # ih @ I should produce the dense representation
        dense = ih @ np.eye(n)
        assert dense.shape == (n, n)
        # Each column should match get_row (via one-hot matvec)
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            np.testing.assert_allclose(dense[:, i], ih.matvec(e_i))

    def test_rmatvec(self, toy_network):
        ih = self._make_ih(toy_network)
        n = ih.shape[0]
        x = np.arange(n, dtype=float)
        np.testing.assert_array_equal(ih.rmatvec(x), ih.matvec(x))

    def test_matvec_wrong_shape(self, toy_network):
        ih = self._make_ih(toy_network)
        with pytest.raises(ValueError):
            ih.matvec(np.ones(2))


# -----------------------------------------------------------------------
# Missing edge cost contract
# -----------------------------------------------------------------------


class TestUnsupportedGraphTypes:
    """Tests that DiGraph, MultiGraph, and MultiDiGraph are rejected."""

    def _make_graph(self, cls):
        G = cls()
        G.add_node(0, x=0, y=0)
        G.add_node(1, x=10, y=0)
        G.add_edge(0, 1, cost=5)
        return G

    def test_digraph_raises(self):
        with pytest.raises(TypeError, match="undirected simple"):
            Hierarchy(self._make_graph(nx.DiGraph), base_radius=20)

    def test_multigraph_raises(self):
        with pytest.raises(TypeError, match="undirected simple"):
            Hierarchy(self._make_graph(nx.MultiGraph), base_radius=20)

    def test_multidigraph_raises(self):
        with pytest.raises(TypeError, match="undirected simple"):
            Hierarchy(self._make_graph(nx.MultiDiGraph), base_radius=20)


# -----------------------------------------------------------------------
# Edge cost validation (extended)
# -----------------------------------------------------------------------


class TestEdgeCostValidation:
    """Tests that invalid edge costs are rejected at construction time."""

    def test_negative_cost_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1, cost=-1)
        with pytest.raises(ValueError, match="invalid cost"):
            Hierarchy(G, base_radius=20)

    def test_nan_cost_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1, cost=float("nan"))
        with pytest.raises(ValueError, match="invalid cost"):
            Hierarchy(G, base_radius=20)

    def test_inf_cost_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1, cost=float("inf"))
        with pytest.raises(ValueError, match="invalid cost"):
            Hierarchy(G, base_radius=20)

    def test_string_cost_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1, cost="ten")
        with pytest.raises(ValueError, match="invalid cost"):
            Hierarchy(G, base_radius=20)

    def test_numpy_int_cost_accepted(self):
        G = nx.Graph()
        G.add_edge(0, 1, cost=np.int64(5))
        Hierarchy(G, base_radius=20)  # should not raise

    def test_numpy_float32_cost_accepted(self):
        G = nx.Graph()
        G.add_edge(0, 1, cost=np.float32(5.0))
        Hierarchy(G, base_radius=20)  # should not raise


# -----------------------------------------------------------------------
# Node-ID contract
# -----------------------------------------------------------------------


class TestNodeIDContract:
    """Tests that non-integer node IDs are rejected."""

    def test_string_node_ids_raises(self):
        G = nx.Graph()
        G.add_node("a", x=0, y=0)
        G.add_node("b", x=10, y=0)
        G.add_edge("a", "b", cost=5)
        with pytest.raises(TypeError, match="integers"):
            Hierarchy(G, base_radius=20)

    def test_tuple_node_ids_raises(self):
        G = nx.Graph()
        G.add_node((0, 0), x=0, y=0)
        G.add_node((1, 0), x=10, y=0)
        G.add_edge((0, 0), (1, 0), cost=5)
        with pytest.raises(TypeError, match="integers"):
            Hierarchy(G, base_radius=20)

    def test_float_node_ids_raises(self):
        G = nx.Graph()
        G.add_node(0.5, x=0, y=0)
        G.add_node(1.5, x=10, y=0)
        G.add_edge(0.5, 1.5, cost=5)
        with pytest.raises(TypeError, match="integers"):
            Hierarchy(G, base_radius=20)

    def test_non_integer_routing_node_raises(self):
        """Non-integer routing nodes (not in zones) must also be rejected."""
        G = nx.Graph()
        G.add_node(0, x=0, y=0)
        G.add_node("mid", x=5, y=0)
        G.add_node(1, x=10, y=0)
        G.add_edge(0, "mid", cost=5)
        G.add_edge("mid", 1, cost=5)
        with pytest.raises(TypeError, match="integers"):
            Hierarchy(G, base_radius=20, zones=[0, 1])


# -----------------------------------------------------------------------
# Missing edge cost contract
# -----------------------------------------------------------------------


class TestMissingEdgeCostContract:
    """Tests for _validate_edge_costs fail-fast behaviour."""

    def test_missing_cost_attribute_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1)  # no cost attribute
        with pytest.raises(ValueError, match="missing a 'cost'"):
            Hierarchy(G, base_radius=20)

    def test_none_cost_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1, cost=None)
        with pytest.raises(ValueError, match="missing a 'cost'"):
            Hierarchy(G, base_radius=20)


# -----------------------------------------------------------------------
# Utility function edge cases
# -----------------------------------------------------------------------


class TestUtilityEdgeCases:
    """Tests for generator and interaction function validation."""

    def test_grid_zero_nx(self):
        with pytest.raises(ValueError, match="n_x"):
            generate_grid_network(0, 5)

    def test_grid_zero_ny(self):
        with pytest.raises(ValueError, match="n_y"):
            generate_grid_network(5, 0)

    def test_grid_negative_spacing(self):
        with pytest.raises(ValueError, match="spacing"):
            generate_grid_network(3, 3, spacing=-1)

    def test_transport_too_small(self):
        with pytest.raises(ValueError, match="n_zones"):
            generate_transport_network(2)

    def test_random_zero_zones(self):
        with pytest.raises(ValueError, match="n_zones"):
            generate_random_spatial_network(0)

    def test_large_zero_zones(self):
        with pytest.raises(ValueError, match="n_zones"):
            generate_large_spatial_network(0)

    def test_exponential_negative_a(self):
        with pytest.raises(ValueError, match="non-negative"):
            exponential_interaction(100.0, a=-1)


# -----------------------------------------------------------------------
# Total nodes explored
# -----------------------------------------------------------------------


class TestTotalNodesExplored:
    """Tests for Hierarchy.total_nodes_explored property."""

    def test_total_nodes_explored_is_positive(self):
        G = generate_grid_network(10, 10, spacing=1000)
        h = Hierarchy(G, base_radius=2000)
        assert isinstance(h.total_nodes_explored, int)
        assert h.total_nodes_explored > 0

    def test_total_nodes_explored_grows_with_network(self):
        G_small = generate_grid_network(5, 5, spacing=1000)
        G_large = generate_grid_network(10, 10, spacing=1000)
        h_small = Hierarchy(G_small, base_radius=2000)
        h_large = Hierarchy(G_large, base_radius=2000)
        assert h_large.total_nodes_explored > h_small.total_nodes_explored


# -----------------------------------------------------------------------
# Within-group cost correction
# -----------------------------------------------------------------------


class TestWithinGroupCostCorrection:
    """Tests that _correct_costs fills gaps left by cutoff-limited Dijkstra.

    With a small overlap_factor, the finest layer's cutoff may not reach
    all pairs within a coarser group.  Without correction, get_cost would
    fall through to the coarser layer where both zones share a representative
    and incorrectly return 0.
    """

    def test_same_group_not_zero(self):
        """Two distinct zones in the same coarser group must not get cost 0."""
        # Use a long linear chain where small overlap_factor creates gaps
        G = nx.Graph()
        n = 20
        for i in range(n - 1):
            G.add_node(i, x=float(i * 100), y=0.0)
            G.add_edge(i, i + 1, cost=100.0)
        G.add_node(n - 1, x=float((n - 1) * 100), y=0.0)

        # overlap_factor=1.0 gives cutoff = base_radius at finest layer,
        # which is tight enough that some within-group pairs are beyond cutoff.
        h = Hierarchy(G, base_radius=300, increase_factor=2, overlap_factor=1.0)

        # Every pair of distinct zones must have cost > 0
        for src in h.zones:
            for dst in h.zones:
                cost = h.get_cost(src, dst)
                if src == dst:
                    assert cost == 0, f"Self-cost should be 0 for zone {src}"
                else:
                    assert cost > 0, (
                        f"get_cost({src}, {dst}) returned 0 — "
                        f"within-group cost correction is missing"
                    )

    def test_corrected_costs_are_exact_shortest_paths(self):
        """Costs added by _correct_costs should be exact Dijkstra distances.

        At the finest layer all zones are representatives, so within-group
        corrections at this layer should produce exact shortest-path costs.
        """
        G = nx.Graph()
        n = 15
        for i in range(n - 1):
            G.add_node(i, x=float(i * 100), y=0.0)
            G.add_edge(i, i + 1, cost=100.0)
        G.add_node(n - 1, x=float((n - 1) * 100), y=0.0)

        h = Hierarchy(G, base_radius=300, increase_factor=2, overlap_factor=1.0)

        # Check that costs added at the finest layer are exact
        finest_radius = h.radii[0]
        costs = h.costs[finest_radius]
        for src in costs:
            for dst, cost in costs[src].items():
                expected = abs(src - dst) * 100.0
                assert cost == pytest.approx(expected), (
                    f"costs[{finest_radius}][{src}][{dst}] = {cost}, expected {expected}"
                )

    def test_correction_fills_gap_across_layers(self):
        """Zones in the same coarser group get a real cost at the finer layer,
        preventing fall-through to the zero-cost same-representative lookup."""
        G = nx.Graph()
        n = 10
        for i in range(n - 1):
            G.add_node(i, x=float(i * 100), y=0.0)
            G.add_edge(i, i + 1, cost=100.0)
        G.add_node(n - 1, x=float((n - 1) * 100), y=0.0)

        h = Hierarchy(G, base_radius=200, increase_factor=2, overlap_factor=1.0)

        # For every coarser layer, check that zones sharing a representative
        # have a cost entry at the finer layer
        for k in range(len(h.radii) - 1):
            fine_radius = h.radii[k]
            coarse_radius = h.radii[k + 1]
            for zone_a in h.repr_zones[fine_radius]:
                for zone_b in h.repr_zones[fine_radius]:
                    if zone_a >= zone_b:
                        continue
                    rep_a = h.groups[coarse_radius].get(zone_a)
                    rep_b = h.groups[coarse_radius].get(zone_b)
                    if rep_a == rep_b:
                        # Must have cost at fine layer
                        assert zone_b in h.costs[fine_radius].get(zone_a, {}), (
                            f"Missing cost[{fine_radius}][{zone_a}][{zone_b}] "
                            f"— both in group {rep_a} at layer {coarse_radius}"
                        )
