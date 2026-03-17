"""Regression tests that compare current output against saved reference results."""

import json
from pathlib import Path

import pytest

from hierx import Hierarchy
from hierx.utils import generate_grid_network

TESTS_DIR = Path(__file__).parent


@pytest.fixture(params=["grid_20x20", "grid_50x50"])
def reference(request):
    path = TESTS_DIR / f"reference_{request.param}.json"
    with open(path) as f:
        return json.load(f)


def test_repr_zones_match(reference):
    """Representative sets must be identical at every layer."""
    cfg = reference["config"]
    G = generate_grid_network(cfg["rows"], cfg["cols"], spacing=cfg["spacing"])
    h = Hierarchy(G, base_radius=cfg["base_radius"], backend="scipy")

    assert h.radii == reference["radii"]
    for r in h.radii:
        actual = sorted(h.repr_zones[r])
        expected = reference["repr_zones"][str(r)]
        assert actual == expected, f"repr_zones mismatch at r={r}"


def test_groups_match(reference):
    """Group assignments must be identical at every layer."""
    cfg = reference["config"]
    G = generate_grid_network(cfg["rows"], cfg["cols"], spacing=cfg["spacing"])
    h = Hierarchy(G, base_radius=cfg["base_radius"], backend="scipy")

    for r in h.radii:
        ref_groups = reference["groups"][str(r)]
        for zone_str, expected_rep in ref_groups.items():
            zone = int(zone_str)
            actual_rep = h.groups[r][zone]
            assert actual_rep == expected_rep, (
                f"group mismatch at r={r}, zone={zone}: got {actual_rep}, expected {expected_rep}"
            )


def test_costs_summary_match(reference):
    """Number of cost entries must match at every layer."""
    cfg = reference["config"]
    G = generate_grid_network(cfg["rows"], cfg["cols"], spacing=cfg["spacing"])
    h = Hierarchy(G, base_radius=cfg["base_radius"], backend="scipy")

    for r in h.radii:
        actual_sources = len(h.costs[r])
        actual_entries = sum(len(h.costs[r][s]) for s in h.costs[r])
        ref = reference["costs_summary"][str(r)]
        assert actual_sources == ref["n_sources"], (
            f"n_sources mismatch at r={r}: {actual_sources} vs {ref['n_sources']}"
        )
        assert actual_entries == ref["n_entries"], (
            f"n_entries mismatch at r={r}: {actual_entries} vs {ref['n_entries']}"
        )


def test_get_cost_match(reference):
    """Sampled get_cost lookups must return the same values."""
    cfg = reference["config"]
    G = generate_grid_network(cfg["rows"], cfg["cols"], spacing=cfg["spacing"])
    h = Hierarchy(G, base_radius=cfg["base_radius"], backend="scipy")

    for pair_str, expected in reference["sample_get_cost"].items():
        src, dst = pair_str.split("->")
        src, dst = int(src), int(dst)
        actual = h.get_cost(src, dst)
        if expected == "inf":
            assert actual == float("inf"), f"Expected inf for {src}->{dst}"
        else:
            assert abs(actual - expected) < 1e-10, f"get_cost({src}, {dst}): {actual} vs {expected}"
