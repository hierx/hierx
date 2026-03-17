"""Tests for backend abstraction and parallelization."""

import networkx as nx
import numpy as np
import pytest

from hierx import HAS_GRAPH_TOOL, Hierarchy, generate_grid_network
from hierx.backends import (
    _dists_to_dicts,
    select_backend,
    shortest_paths_nx,
)

needs_gt = pytest.mark.skipif(not HAS_GRAPH_TOOL, reason="graph-tool not installed")


# ---------------------------------------------------------------------------
# _dists_to_dicts vectorized helper
# ---------------------------------------------------------------------------


class TestDistsToDict:
    def test_all_inf_returns_empty_dicts(self):
        dist = np.full((3, 5), np.inf)
        chunk = [0, 1, 2]
        idx_to_node = list(range(5))
        col_node_arr = np.array(idx_to_node)
        result, explored = _dists_to_dicts(dist, chunk, idx_to_node, col_node_arr)
        assert result == {0: {}, 1: {}, 2: {}}
        assert explored == 0

    def test_single_row(self):
        dist = np.array([[0.0, 10.0, np.inf, 25.0]])
        chunk = [0]
        idx_to_node = list(range(4))
        col_node_arr = np.array(idx_to_node)
        result, explored = _dists_to_dicts(dist, chunk, idx_to_node, col_node_arr)
        assert result == {0: {0: 0.0, 1: 10.0, 3: 25.0}}
        assert explored == 3

    def test_mixed_finite_inf(self):
        dist = np.array(
            [
                [0.0, np.inf, np.inf],
                [np.inf, 0.0, 5.0],
            ]
        )
        chunk = [0, 1]
        idx_to_node = list(range(3))
        col_node_arr = np.array(idx_to_node)
        result, explored = _dists_to_dicts(dist, chunk, idx_to_node, col_node_arr)
        assert result == {0: {0: 0.0}, 1: {1: 0.0, 2: 5.0}}
        assert explored == 3

    def test_all_finite(self):
        dist = np.array([[1.0, 2.0], [3.0, 4.0]])
        chunk = [0, 1]
        idx_to_node = list(range(2))
        col_node_arr = np.array(idx_to_node)
        result, explored = _dists_to_dicts(dist, chunk, idx_to_node, col_node_arr)
        assert result == {0: {0: 1.0, 1: 2.0}, 1: {0: 3.0, 1: 4.0}}
        assert explored == 4

    def test_col_node_arr_with_rep_subset(self):
        """Simulates the case where dist is already column-sliced to representatives."""
        dist = np.array([[5.0, np.inf], [np.inf, 10.0]])
        chunk = [0, 3]
        idx_to_node = [10, 20, 30, 40]  # node IDs are not 0-indexed
        col_node_arr = np.array([10, 40])  # only 2 rep columns
        result, explored = _dists_to_dicts(dist, chunk, idx_to_node, col_node_arr)
        assert result == {10: {10: 5.0}, 40: {40: 10.0}}
        assert explored == 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_grid():
    """6x6 grid network (36 nodes) — fast but enough for parallelization."""
    return generate_grid_network(6, 6, spacing=1000)


@pytest.fixture
def tiny_network():
    """4-node linear graph."""
    G = nx.Graph()
    G.add_edge(0, 1, cost=10)
    G.add_edge(1, 2, cost=15)
    G.add_edge(2, 3, cost=20)
    return G


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestSelectBackend:
    def test_auto_selects_scipy(self):
        assert select_backend("auto") == "scipy"

    def test_explicit_scipy(self):
        assert select_backend("scipy") == "scipy"

    def test_explicit_networkx(self):
        assert select_backend("networkx") == "networkx"

    def test_explicit_gt_raises_without_gt(self):
        if not HAS_GRAPH_TOOL:
            with pytest.raises(ValueError, match="not installed"):
                select_backend("graph-tool")

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            select_backend("igraph")

    @needs_gt
    def test_auto_still_selects_scipy_with_gt_available(self):
        """auto always selects scipy (fastest C backend) regardless of graph-tool."""
        assert select_backend("auto") == "scipy"

    @needs_gt
    def test_explicit_gt_when_available(self):
        assert select_backend("graph-tool") == "graph-tool"


# ---------------------------------------------------------------------------
# Shortest-paths wrapper (NetworkX)
# ---------------------------------------------------------------------------


class TestShortestPathsNx:
    def test_basic(self, tiny_network):
        result, explored = shortest_paths_nx(tiny_network, 0)
        assert result[0] == 0
        assert result[1] == 10
        assert result[3] == pytest.approx(45)
        assert explored == 4

    def test_cutoff(self, tiny_network):
        result, explored = shortest_paths_nx(tiny_network, 0, cutoff=20)
        assert 0 in result
        assert 1 in result
        # node 2 is at distance 25, should be excluded
        assert 2 not in result
        assert explored == 2


# ---------------------------------------------------------------------------
# Parallelization equivalence
# ---------------------------------------------------------------------------


class TestParallelization:
    def _assert_costs_equal(self, h1: Hierarchy, h2: Hierarchy):
        """Assert two hierarchies have identical cost dicts."""
        assert h1.radii == h2.radii
        for r in h1.radii:
            for s in h1.costs[r]:
                assert s in h2.costs[r], f"source {s} missing at radius {r}"
                for d in h1.costs[r][s]:
                    assert d in h2.costs[r][s], f"dest {d} missing for source {s} at radius {r}"
                    assert abs(h1.costs[r][s][d] - h2.costs[r][s][d]) < 1e-10, (
                        f"cost mismatch at r={r}, s={s}, d={d}: "
                        f"{h1.costs[r][s][d]} vs {h2.costs[r][s][d]}"
                    )

    def test_parallel_matches_sequential(self, small_grid):
        h1 = Hierarchy(small_grid, base_radius=2000, backend="networkx", n_workers=1)
        h2 = Hierarchy(small_grid, base_radius=2000, backend="networkx", n_workers=2)
        self._assert_costs_equal(h1, h2)

    def test_parallel_with_4_workers(self, small_grid):
        h1 = Hierarchy(small_grid, base_radius=2000, backend="networkx", n_workers=1)
        h2 = Hierarchy(small_grid, base_radius=2000, backend="networkx", n_workers=4)
        self._assert_costs_equal(h1, h2)

    def test_parallel_more_workers_than_sources(self, tiny_network):
        """n_workers=8 on a 4-node graph should not crash."""
        h1 = Hierarchy(tiny_network, base_radius=20, backend="networkx", n_workers=1)
        h2 = Hierarchy(tiny_network, base_radius=20, backend="networkx", n_workers=8)
        self._assert_costs_equal(h1, h2)

    def test_n_workers_1_is_default(self, tiny_network):
        h = Hierarchy(tiny_network, base_radius=20)
        assert h._n_workers == 1

    def test_get_cost_matches(self, small_grid):
        h1 = Hierarchy(small_grid, base_radius=2000, backend="networkx", n_workers=1)
        h2 = Hierarchy(small_grid, base_radius=2000, backend="networkx", n_workers=2)
        nodes = list(small_grid.nodes())
        for src in nodes[:5]:
            for dst in nodes[-5:]:
                assert abs(h1.get_cost(src, dst) - h2.get_cost(src, dst)) < 1e-10


# ---------------------------------------------------------------------------
# Graph-tool conversion tests
# ---------------------------------------------------------------------------


@needs_gt
class TestGraphConversion:
    def test_node_count_preserved(self, small_grid):
        from hierx.backends import convert_nx_to_gt

        gt_graph, node_to_vertex, _ = convert_nx_to_gt(small_grid)
        assert gt_graph.num_vertices() == small_grid.number_of_nodes()

    def test_edge_count_preserved(self, small_grid):
        from hierx.backends import convert_nx_to_gt

        gt_graph, _, _ = convert_nx_to_gt(small_grid)
        assert gt_graph.num_edges() == small_grid.number_of_edges()

    def test_node_mapping_bijective(self, small_grid):
        from hierx.backends import convert_nx_to_gt

        _, node_to_vertex, _ = convert_nx_to_gt(small_grid)
        assert len(node_to_vertex) == small_grid.number_of_nodes()
        assert len(set(node_to_vertex.values())) == len(node_to_vertex)

    def test_weights_preserved(self, tiny_network):
        from hierx.backends import convert_nx_to_gt

        gt_graph, node_to_vertex, weight_prop = convert_nx_to_gt(tiny_network)
        # Check edge (0, 1) has cost 10
        v0 = gt_graph.vertex(node_to_vertex[0])
        v1 = gt_graph.vertex(node_to_vertex[1])
        e = gt_graph.edge(v0, v1)
        assert weight_prop[e] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Graph-tool backend equivalence
# ---------------------------------------------------------------------------


@needs_gt
class TestBackendEquivalence:
    def test_shortest_paths_match(self, small_grid):
        from hierx.backends import (
            convert_nx_to_gt,
            shortest_paths_gt,
        )

        gt_graph, node_to_vertex, weight_prop = convert_nx_to_gt(small_grid)
        sources = list(small_grid.nodes())[:5]
        for src in sources:
            nx_result, _ = shortest_paths_nx(small_grid, src)
            gt_result, _ = shortest_paths_gt(src, gt_graph, node_to_vertex, weight_prop)
            for node in nx_result:
                assert node in gt_result, f"node {node} missing in gt result"
                assert abs(nx_result[node] - gt_result[node]) < 1e-10

    def test_hierarchy_costs_match(self, small_grid):
        h_nx = Hierarchy(small_grid, base_radius=2000, backend="networkx")
        h_gt = Hierarchy(small_grid, base_radius=2000, backend="graph-tool")
        assert h_nx.radii == h_gt.radii
        for r in h_nx.radii:
            for s in h_nx.costs[r]:
                for d in h_nx.costs[r][s]:
                    assert abs(h_nx.costs[r][s][d] - h_gt.costs[r][s][d]) < 1e-10

    def test_hierarchy_get_cost_match(self, small_grid):
        h_nx = Hierarchy(small_grid, base_radius=2000, backend="networkx")
        h_gt = Hierarchy(small_grid, base_radius=2000, backend="graph-tool")
        nodes = list(small_grid.nodes())
        for src in nodes[:5]:
            for dst in nodes[-5:]:
                assert abs(h_nx.get_cost(src, dst) - h_gt.get_cost(src, dst)) < 1e-10


# ---------------------------------------------------------------------------
# Integration: InteractionHierarchy matvec
# ---------------------------------------------------------------------------


class TestScipyParallelization:
    """Verify shared-memory scipy parallel path matches serial."""

    def _assert_costs_equal(self, h1: Hierarchy, h2: Hierarchy):
        """Assert two hierarchies have identical cost dicts."""
        assert h1.radii == h2.radii
        for r in h1.radii:
            for s in h1.costs[r]:
                assert s in h2.costs[r], f"source {s} missing at radius {r}"
                for d in h1.costs[r][s]:
                    assert d in h2.costs[r][s], f"dest {d} missing for source {s} at radius {r}"
                    assert abs(h1.costs[r][s][d] - h2.costs[r][s][d]) < 1e-10, (
                        f"cost mismatch at r={r}, s={s}, d={d}: "
                        f"{h1.costs[r][s][d]} vs {h2.costs[r][s][d]}"
                    )

    def test_scipy_parallel_matches_serial(self, small_grid):
        h1 = Hierarchy(small_grid, base_radius=2000, backend="scipy", n_workers=1)
        h2 = Hierarchy(small_grid, base_radius=2000, backend="scipy", n_workers=2)
        self._assert_costs_equal(h1, h2)

    def test_scipy_parallel_4_workers(self, small_grid):
        h1 = Hierarchy(small_grid, base_radius=2000, backend="scipy", n_workers=1)
        h2 = Hierarchy(small_grid, base_radius=2000, backend="scipy", n_workers=4)
        self._assert_costs_equal(h1, h2)

    def test_scipy_parallel_more_workers_than_sources(self, tiny_network):
        h1 = Hierarchy(tiny_network, base_radius=20, backend="scipy", n_workers=1)
        h2 = Hierarchy(tiny_network, base_radius=20, backend="scipy", n_workers=8)
        self._assert_costs_equal(h1, h2)

    def test_scipy_parallel_get_cost_matches(self, small_grid):
        h1 = Hierarchy(small_grid, base_radius=2000, backend="scipy", n_workers=1)
        h2 = Hierarchy(small_grid, base_radius=2000, backend="scipy", n_workers=2)
        nodes = list(small_grid.nodes())
        for src in nodes[:5]:
            for dst in nodes[-5:]:
                assert abs(h1.get_cost(src, dst) - h2.get_cost(src, dst)) < 1e-10

    def test_scipy_parallel_matvec_matches(self, small_grid):
        from hierx import InteractionHierarchy

        def f(c):
            return (c + 1000) ** (-2)

        h1 = Hierarchy(small_grid, base_radius=2000, backend="scipy", n_workers=1)
        h2 = Hierarchy(small_grid, base_radius=2000, backend="scipy", n_workers=2)
        ih1, ih2 = InteractionHierarchy(h1, f), InteractionHierarchy(h2, f)
        activity = np.ones(len(small_grid.nodes()))
        np.testing.assert_allclose(ih1.matvec(activity), ih2.matvec(activity), atol=1e-12)

    def test_scipy_parallel_no_cutoff_layer(self, small_grid):
        """Coarsest layer has cutoff=None; verify parallel handles it."""
        h = Hierarchy(small_grid, base_radius=2000, backend="scipy", n_workers=2)
        coarsest = h.radii[-1]
        assert h.cutoffs[coarsest] is None
        assert len(h.costs[coarsest]) > 0


@needs_gt
class TestGraphToolParallelization:
    """Verify chunked graph-tool parallel path matches serial."""

    def _assert_costs_equal(self, h1: Hierarchy, h2: Hierarchy):
        assert h1.radii == h2.radii
        for r in h1.radii:
            for s in h1.costs[r]:
                assert s in h2.costs[r], f"source {s} missing at radius {r}"
                for d in h1.costs[r][s]:
                    assert d in h2.costs[r][s], f"dest {d} missing for source {s} at radius {r}"
                    assert abs(h1.costs[r][s][d] - h2.costs[r][s][d]) < 1e-10, (
                        f"cost mismatch at r={r}, s={s}, d={d}: "
                        f"{h1.costs[r][s][d]} vs {h2.costs[r][s][d]}"
                    )

    def test_gt_parallel_matches_serial(self, small_grid):
        h1 = Hierarchy(small_grid, base_radius=2000, backend="graph-tool", n_workers=1)
        h2 = Hierarchy(small_grid, base_radius=2000, backend="graph-tool", n_workers=2)
        self._assert_costs_equal(h1, h2)

    def test_gt_parallel_4_workers(self, small_grid):
        h1 = Hierarchy(small_grid, base_radius=2000, backend="graph-tool", n_workers=1)
        h2 = Hierarchy(small_grid, base_radius=2000, backend="graph-tool", n_workers=4)
        self._assert_costs_equal(h1, h2)

    def test_gt_parallel_get_cost_matches(self, small_grid):
        h1 = Hierarchy(small_grid, base_radius=2000, backend="graph-tool", n_workers=1)
        h2 = Hierarchy(small_grid, base_radius=2000, backend="graph-tool", n_workers=2)
        nodes = list(small_grid.nodes())
        for src in nodes[:5]:
            for dst in nodes[-5:]:
                assert abs(h1.get_cost(src, dst) - h2.get_cost(src, dst)) < 1e-10


# ---------------------------------------------------------------------------
# Integration: InteractionHierarchy matvec
# ---------------------------------------------------------------------------


class TestCorrectionParallelization:
    """Verify parallel _correct_costs matches serial across backends.

    Uses overlap_factor=1.0 to force heavy correction (many missing pairs).
    """

    def _assert_costs_equal(self, h1: Hierarchy, h2: Hierarchy):
        assert h1.radii == h2.radii
        for r in h1.radii:
            for s in h1.costs[r]:
                assert s in h2.costs[r], f"source {s} missing at radius {r}"
                for d in h1.costs[r][s]:
                    assert d in h2.costs[r][s], f"dest {d} missing for source {s} at radius {r}"
                    assert abs(h1.costs[r][s][d] - h2.costs[r][s][d]) < 1e-10, (
                        f"cost mismatch at r={r}, s={s}, d={d}: "
                        f"{h1.costs[r][s][d]} vs {h2.costs[r][s][d]}"
                    )

    def test_correction_nx_parallel_matches_serial(self, small_grid):
        h1 = Hierarchy(
            small_grid, base_radius=2000, overlap_factor=1.0, backend="networkx", n_workers=1
        )
        h2 = Hierarchy(
            small_grid, base_radius=2000, overlap_factor=1.0, backend="networkx", n_workers=4
        )
        self._assert_costs_equal(h1, h2)

    def test_correction_scipy_parallel_matches_serial(self, small_grid):
        h1 = Hierarchy(
            small_grid, base_radius=2000, overlap_factor=1.0, backend="scipy", n_workers=1
        )
        h2 = Hierarchy(
            small_grid, base_radius=2000, overlap_factor=1.0, backend="scipy", n_workers=4
        )
        self._assert_costs_equal(h1, h2)

    @needs_gt
    def test_correction_gt_parallel_matches_serial(self, small_grid):
        h1 = Hierarchy(
            small_grid, base_radius=2000, overlap_factor=1.0, backend="graph-tool", n_workers=1
        )
        h2 = Hierarchy(
            small_grid, base_radius=2000, overlap_factor=1.0, backend="graph-tool", n_workers=4
        )
        self._assert_costs_equal(h1, h2)

    def test_correction_linear_chain(self):
        """Linear chain: worst case for missing pairs."""
        G = nx.path_graph(20)
        nx.set_edge_attributes(G, 100, "cost")
        h1 = Hierarchy(G, base_radius=200, overlap_factor=1.0, backend="scipy", n_workers=1)
        h2 = Hierarchy(G, base_radius=200, overlap_factor=1.0, backend="scipy", n_workers=4)
        self._assert_costs_equal(h1, h2)

    def test_correction_get_cost_matches(self, small_grid):
        """get_cost lookups must agree after parallel correction."""
        h1 = Hierarchy(
            small_grid, base_radius=2000, overlap_factor=1.0, backend="scipy", n_workers=1
        )
        h2 = Hierarchy(
            small_grid, base_radius=2000, overlap_factor=1.0, backend="scipy", n_workers=4
        )
        nodes = list(small_grid.nodes())
        for src in nodes[:6]:
            for dst in nodes[-6:]:
                assert abs(h1.get_cost(src, dst) - h2.get_cost(src, dst)) < 1e-10


class TestInteractionIntegration:
    def test_matvec_same_parallel_vs_sequential(self, small_grid):
        from hierx import InteractionHierarchy

        def f(c):
            return (c + 1000) ** (-2)

        h1 = Hierarchy(small_grid, base_radius=2000, backend="networkx", n_workers=1)
        h2 = Hierarchy(small_grid, base_radius=2000, backend="networkx", n_workers=2)
        ih1 = InteractionHierarchy(h1, f)
        ih2 = InteractionHierarchy(h2, f)

        activity = np.ones(len(small_grid.nodes()))
        result1 = ih1.matvec(activity)
        result2 = ih2.matvec(activity)
        np.testing.assert_allclose(result1, result2, atol=1e-12)
