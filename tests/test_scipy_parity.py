"""Backend parity tests: scipy vs networkx produce identical results."""

import numpy as np
import pytest

from hierx import Hierarchy, InteractionHierarchy, generate_grid_network


@pytest.fixture
def grid_8x8():
    """8x8 grid for parity testing."""
    return generate_grid_network(8, 8, spacing=1000)


class TestScipyNetworkxParity:
    """Verify scipy and networkx backends produce identical results."""

    def test_cost_matrix_parity(self, grid_8x8):
        """All get_cost() values should match between backends."""
        h_scipy = Hierarchy(grid_8x8, base_radius=2000, backend="scipy")
        h_nx = Hierarchy(grid_8x8, base_radius=2000, backend="networkx")

        zones = h_scipy.zones
        for src in zones[:10]:
            for dst in zones[:10]:
                cost_scipy = h_scipy.get_cost(src, dst)
                cost_nx = h_nx.get_cost(src, dst)
                assert cost_scipy == pytest.approx(cost_nx, abs=1e-10), (
                    f"Mismatch for ({src}, {dst}): scipy={cost_scipy}, nx={cost_nx}"
                )

    def test_get_cost_pointwise(self, grid_8x8):
        """Spot-check specific zone pairs."""
        h_scipy = Hierarchy(grid_8x8, base_radius=2000, backend="scipy")
        h_nx = Hierarchy(grid_8x8, base_radius=2000, backend="networkx")

        pairs = [(0, 63), (0, 0), (7, 56), (31, 32)]
        for src, dst in pairs:
            assert h_scipy.get_cost(src, dst) == pytest.approx(h_nx.get_cost(src, dst), abs=1e-10)

    def test_matvec_parity(self, grid_8x8):
        """matvec results should match between backends."""

        def interaction_fn(c):
            return (c + 1000) ** (-2)

        h_scipy = Hierarchy(grid_8x8, base_radius=2000, backend="scipy")
        h_nx = Hierarchy(grid_8x8, base_radius=2000, backend="networkx")

        ih_scipy = InteractionHierarchy(h_scipy, interaction_fn)
        ih_nx = InteractionHierarchy(h_nx, interaction_fn)

        rng = np.random.RandomState(42)
        activity = rng.uniform(0, 100, size=len(h_scipy.zones))

        result_scipy = ih_scipy.matvec(activity)
        result_nx = ih_nx.matvec(activity)

        np.testing.assert_allclose(result_scipy, result_nx, atol=1e-12)

    def test_scipy_parallel_vs_serial_parity(self, grid_8x8):
        """scipy n_workers=2 must match scipy n_workers=1."""
        h1 = Hierarchy(grid_8x8, base_radius=2000, backend="scipy", n_workers=1)
        h2 = Hierarchy(grid_8x8, base_radius=2000, backend="scipy", n_workers=2)
        zones = h1.zones
        for src in zones[:10]:
            for dst in zones[:10]:
                assert h1.get_cost(src, dst) == pytest.approx(h2.get_cost(src, dst), abs=1e-10)
