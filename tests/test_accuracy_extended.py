"""
Extended accuracy tests validating specific paper claims (Section 5).

Parametrised pytest tests checking that:
- Mean error < 10% with default parameters across grid sizes
- Larger overlap_factor reduces error
- Steeper decay functions produce larger errors
"""

import numpy as np
import pytest

from hierx import (
    Hierarchy,
    InteractionHierarchy,
    compute_dense_cost_matrix,
    compute_dense_interaction_matrix,
    generate_grid_network,
)


def _matvec_relative_error(
    G,
    base_radius: float,
    overlap_factor: float,
    increase_factor: float,
    interaction_fn,
    n_trials: int = 5,
) -> float:
    """Compute median mean-relative-error of hierarchical matvec vs dense."""
    zones = sorted(G.nodes())
    n_zones = len(zones)

    hierarchy = Hierarchy(
        G,
        base_radius=base_radius,
        increase_factor=increase_factor,
        overlap_factor=overlap_factor,
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


# -----------------------------------------------------------------------
# Test: default parameters produce < 10% mean error across grid sizes
# -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_x,n_y,base_radius",
    [
        (5, 5, 2000),
        (10, 10, 3000),
        (15, 15, 5000),
        (20, 20, 6000),
    ],
)
def test_default_error_below_threshold(n_x: int, n_y: int, base_radius: float) -> None:
    """Mean error < 10% with default overlap_factor=1.5, increase_factor=2."""
    G = generate_grid_network(n_x, n_y, spacing=1000)

    def interaction_fn(c):
        return (c + 2000) ** (-1.5)

    err = _matvec_relative_error(
        G,
        base_radius=base_radius,
        overlap_factor=1.5,
        increase_factor=2.0,
        interaction_fn=interaction_fn,
    )
    assert err < 0.10, (
        f"Mean relative error {err:.4f} exceeds 10% for "
        f"{n_x}x{n_y} grid with base_radius={base_radius}"
    )


# -----------------------------------------------------------------------
# Test: larger overlap_factor reduces error
# -----------------------------------------------------------------------
def test_overlap_factor_reduces_error() -> None:
    """Increasing overlap_factor should decrease approximation error."""
    G = generate_grid_network(10, 10, spacing=1000)

    def interaction_fn(c):
        return (c + 2000) ** (-1.5)

    errors = {}
    for ov in [1.0, 1.5, 2.0, 3.0]:
        errors[ov] = _matvec_relative_error(
            G,
            base_radius=3000,
            overlap_factor=ov,
            increase_factor=2.0,
            interaction_fn=interaction_fn,
        )

    # Each step should not increase error (allow small tolerance)
    overlaps = sorted(errors.keys())
    for i in range(len(overlaps) - 1):
        assert errors[overlaps[i + 1]] <= errors[overlaps[i]] + 0.005, (
            f"Error did not decrease: overlap={overlaps[i]} err={errors[overlaps[i]]:.4f} "
            f"vs overlap={overlaps[i + 1]} err={errors[overlaps[i + 1]]:.4f}"
        )


# -----------------------------------------------------------------------
# Test: steeper decay functions produce larger errors
# -----------------------------------------------------------------------
def test_steep_decay_larger_error() -> None:
    """Steeper decay functions should produce larger approximation errors."""
    G = generate_grid_network(10, 10, spacing=1000)

    def steep_fn(c):
        return (c + 500) ** (-2)

    def moderate_fn(c):
        return (c + 2000) ** (-1.5)

    def shallow_fn(c):
        return (c + 5000) ** (-1)

    err_steep = _matvec_relative_error(
        G,
        base_radius=3000,
        overlap_factor=1.5,
        increase_factor=2.0,
        interaction_fn=steep_fn,
    )
    err_moderate = _matvec_relative_error(
        G,
        base_radius=3000,
        overlap_factor=1.5,
        increase_factor=2.0,
        interaction_fn=moderate_fn,
    )
    err_shallow = _matvec_relative_error(
        G,
        base_radius=3000,
        overlap_factor=1.5,
        increase_factor=2.0,
        interaction_fn=shallow_fn,
    )

    # Steep should have at least as much error as shallow (with tolerance)
    assert err_steep >= err_shallow - 0.01, (
        f"Expected steep ({err_steep:.4f}) >= shallow ({err_shallow:.4f})"
    )
    assert err_moderate >= err_shallow - 0.01, (
        f"Expected moderate ({err_moderate:.4f}) >= shallow ({err_shallow:.4f})"
    )
