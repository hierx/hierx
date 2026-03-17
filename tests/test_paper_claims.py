"""
Tests directly validating paper claims for Sections 4-5.

These tests are marked @pytest.mark.slow because they run benchmarks
on moderately large networks to validate complexity and error claims.

Run with:
    python -m pytest tests/test_paper_claims.py -m slow -v
"""

import time

import numpy as np
import pytest
from scipy.optimize import curve_fit

from hierx import (
    Hierarchy,
    InteractionHierarchy,
    compute_dense_cost_matrix,
    compute_dense_interaction_matrix,
    generate_grid_network,
    generate_large_spatial_network,
)


def _nlogn(n: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * n * np.log(n) + b


def _fit_r2(sizes: np.ndarray, times: np.ndarray) -> float:
    """Fit t = a*n*log(n) + b and return R^2."""
    popt, _ = curve_fit(_nlogn, sizes, times, p0=[1e-6, 0], maxfev=10000)
    predicted = _nlogn(sizes, *popt)
    ss_res = np.sum((times - predicted) ** 2)
    ss_tot = np.sum((times - np.mean(times)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ===================================================================
# Section 4: Complexity claims
# ===================================================================


class TestComplexityClaims:
    """Validate O(n log n) scaling claims."""

    @pytest.mark.slow
    def test_hierarchy_build_nlogn(self) -> None:
        """Hierarchy build time fits O(n log n) with R^2 > 0.95."""
        sizes_n = [100, 500, 1000, 2000, 5000, 10000]
        times = []

        for n in sizes_n:
            area_size = np.sqrt(n) * 5000
            base_radius = area_size / np.sqrt(n) * 2
            G = generate_large_spatial_network(n, area_size=area_size, seed=42)

            # Warm up
            Hierarchy(G, base_radius=base_radius, increase_factor=2, overlap_factor=1.5)

            trials = []
            for _ in range(3):
                t0 = time.perf_counter()
                Hierarchy(G, base_radius=base_radius, increase_factor=2, overlap_factor=1.5)
                trials.append(time.perf_counter() - t0)
            times.append(np.median(trials))

        sizes_arr = np.array(sizes_n, dtype=float)
        times_arr = np.array(times)
        r2 = _fit_r2(sizes_arr, times_arr)

        assert r2 > 0.95, f"Hierarchy build time O(n log n) fit R^2 = {r2:.4f} (expected > 0.95)"

    @pytest.mark.slow
    def test_matvec_nlogn(self) -> None:
        """Matvec time fits O(n log n) with R^2 > 0.95."""
        sizes_n = [100, 500, 1000, 2000, 5000, 10000]
        times = []

        for n in sizes_n:
            area_size = np.sqrt(n) * 5000
            base_radius = area_size / np.sqrt(n) * 2
            G = generate_large_spatial_network(n, area_size=area_size, seed=42)
            hierarchy = Hierarchy(G, base_radius=base_radius, increase_factor=2, overlap_factor=1.5)

            def interaction_fn(c):
                return (c + 1000) ** (-2)

            ih = InteractionHierarchy(hierarchy, interaction_fn)
            activity = np.random.rand(n)

            # Warm up
            ih.matvec(activity)

            trials = []
            for _ in range(5):
                t0 = time.perf_counter()
                for _ in range(10):
                    ih.matvec(activity)
                trials.append((time.perf_counter() - t0) / 10)
            times.append(np.median(trials))

        sizes_arr = np.array(sizes_n, dtype=float)
        times_arr = np.array(times)
        r2 = _fit_r2(sizes_arr, times_arr)

        assert r2 > 0.95, f"Matvec time O(n log n) fit R^2 = {r2:.4f} (expected > 0.95)"


# ===================================================================
# Section 5: Error claims
# ===================================================================


class TestErrorClaims:
    """Validate approximation error claims."""

    def _compute_error(
        self,
        G,
        base_radius: float,
        overlap_factor: float,
        interaction_fn,
        n_trials: int = 5,
    ) -> float:
        zones = sorted(G.nodes())
        n_zones = len(zones)
        hierarchy = Hierarchy(
            G, base_radius=base_radius, increase_factor=2, overlap_factor=overlap_factor
        )
        ih = InteractionHierarchy(hierarchy, interaction_fn)
        cost_matrix = compute_dense_cost_matrix(G, zones)
        interaction_matrix = compute_dense_interaction_matrix(cost_matrix, interaction_fn)

        rng = np.random.RandomState(42)
        errors = []
        for _ in range(n_trials):
            activity = rng.rand(n_zones) * 100 + 1
            h = ih.matvec(activity)
            d = interaction_matrix @ activity
            rel_err = np.mean(np.abs(h - d) / np.maximum(np.abs(d), 1e-10))
            errors.append(rel_err)
        return float(np.median(errors))

    @pytest.mark.slow
    def test_error_below_10pct_defaults(self) -> None:
        """Error < 10% with default parameters across grid sizes."""

        def interaction_fn(c):
            return (c + 2000) ** (-1.5)

        for n_x, base_r in [(10, 3000), (15, 5000), (20, 6000)]:
            G = generate_grid_network(n_x, n_x, spacing=1000)
            err = self._compute_error(G, base_r, overlap_factor=1.5, interaction_fn=interaction_fn)
            assert err < 0.10, f"Error {err:.4f} >= 10% for {n_x}x{n_x} grid"

    @pytest.mark.slow
    def test_error_decreases_with_overlap(self) -> None:
        """Error decreases as overlap_factor increases."""
        G = generate_grid_network(10, 10, spacing=1000)

        def interaction_fn(c):
            return (c + 2000) ** (-1.5)

        prev_err = float("inf")
        for ov in [1.0, 1.5, 2.0, 3.0]:
            err = self._compute_error(
                G, base_radius=3000, overlap_factor=ov, interaction_fn=interaction_fn
            )
            assert err <= prev_err + 0.005, (
                f"Error increased from {prev_err:.4f} to {err:.4f} when overlap went to {ov}"
            )
            prev_err = err

    @pytest.mark.slow
    def test_steep_decay_higher_error(self) -> None:
        """Steeper decay functions produce larger errors."""
        G = generate_grid_network(10, 10, spacing=1000)

        def steep_fn(c):
            return (c + 500) ** (-2)

        def shallow_fn(c):
            return (c + 5000) ** (-1)

        err_steep = self._compute_error(G, 3000, 1.5, steep_fn)
        err_shallow = self._compute_error(G, 3000, 1.5, shallow_fn)

        assert err_steep > err_shallow - 0.01, (
            f"Steep error ({err_steep:.4f}) not greater than shallow error ({err_shallow:.4f})"
        )
