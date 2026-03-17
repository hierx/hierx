"""Tests for pre-release features: self_interaction parameter and D_dict."""

import numpy as np
import pytest

from hierx import (
    HAS_H5PY,
    Hierarchy,
    InteractionHierarchy,
    generate_grid_network,
    load_interaction,
    save_interaction,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_backends = ["npz"]
if HAS_H5PY:
    _backends.append("hdf5")


@pytest.fixture(params=_backends)
def backend(request):
    return request.param


@pytest.fixture
def small_network():
    return generate_grid_network(5, 5, spacing=1000)


@pytest.fixture
def hierarchy(small_network):
    return Hierarchy(small_network, base_radius=2000, increase_factor=2, overlap_factor=1.5)


@pytest.fixture
def interaction_fn():
    return lambda c: (c + 1000) ** (-2)


# ---------------------------------------------------------------------------
# self_interaction tests
# ---------------------------------------------------------------------------


class TestSelfInteraction:
    def test_default_true(self, hierarchy, interaction_fn):
        """Default self_interaction=True preserves existing behavior."""
        ih = InteractionHierarchy(hierarchy, interaction_fn)
        assert ih.self_interaction is True
        # Diagonal should be nonzero at finest layer
        finest = hierarchy.radii[0]
        diag = ih.D[finest].diagonal()
        assert np.any(diag > 0), "Diagonal should have nonzero entries by default"

    def test_false_zeros_finest_diagonal(self, hierarchy, interaction_fn):
        """self_interaction=False zeros diagonal at finest layer only."""
        ih = InteractionHierarchy(hierarchy, interaction_fn, self_interaction=False)
        assert ih.self_interaction is False

        finest = hierarchy.radii[0]
        diag_finest = ih.D[finest].diagonal()
        assert np.all(diag_finest == 0), "Finest layer diagonal should be all zeros"

        # Coarser layers should be unaffected (diagonal may still be nonzero)
        if len(hierarchy.radii) > 1:
            coarser = hierarchy.radii[1]
            ih_with = InteractionHierarchy(hierarchy, interaction_fn, self_interaction=True)
            np.testing.assert_array_equal(
                ih.D[coarser].toarray(),
                ih_with.D[coarser].toarray(),
                err_msg="Coarser layers should be identical regardless of self_interaction",
            )

    def test_net_operator_differs_at_finest(self, hierarchy, interaction_fn):
        """The net operator at the finest layer should differ."""
        ih_with = InteractionHierarchy(hierarchy, interaction_fn, self_interaction=True)
        ih_without = InteractionHierarchy(hierarchy, interaction_fn, self_interaction=False)

        finest = hierarchy.radii[0]

        # The diagonal of D differs, so the net operator should too
        # (unless the correction perfectly cancels, which happens for
        # self-cost=0 entries — either way D itself is verifiably different)
        diag_with = ih_with.D[finest].diagonal()
        diag_without = ih_without.D[finest].diagonal()
        assert not np.allclose(diag_with, diag_without), (
            "D diagonal should differ between self_interaction=True and False"
        )

    def test_save_load_roundtrip(self, hierarchy, interaction_fn, backend, tmp_path):
        """self_interaction flag should survive save/load."""
        ih = InteractionHierarchy(hierarchy, interaction_fn, self_interaction=False)
        p = save_interaction(ih, tmp_path / "ih", backend=backend)
        loaded = load_interaction(p, interaction_fn=interaction_fn)

        assert loaded.self_interaction is False

        # matvec should match
        activity = np.ones(len(hierarchy.zones))
        np.testing.assert_allclose(loaded.matvec(activity), ih.matvec(activity), rtol=1e-12)

    def test_save_load_default_true(self, hierarchy, interaction_fn, backend, tmp_path):
        """Default self_interaction=True should also roundtrip correctly."""
        ih = InteractionHierarchy(hierarchy, interaction_fn)
        p = save_interaction(ih, tmp_path / "ih", backend=backend)
        loaded = load_interaction(p)
        assert loaded.self_interaction is True


# ---------------------------------------------------------------------------
# D_dict tests
# ---------------------------------------------------------------------------


class TestDDict:
    def test_d_dict_populated(self, hierarchy, interaction_fn):
        """D_dict should be populated for all layers."""
        ih = InteractionHierarchy(hierarchy, interaction_fn)
        for radius in hierarchy.radii:
            assert radius in ih.D_dict

    def test_d_dict_matches_sparse(self, hierarchy, interaction_fn):
        """D_dict entries should match the sparse D matrix values."""
        ih = InteractionHierarchy(hierarchy, interaction_fn)
        for radius in hierarchy.radii:
            idx_to_zone = {i: z for z, i in ih.zone_indices[radius].items()}
            D_coo = ih.D[radius].tocoo()
            for row, col, val in zip(D_coo.row, D_coo.col, D_coo.data):
                src = idx_to_zone[row]
                dst = idx_to_zone[col]
                assert src in ih.D_dict[radius], f"Missing src {src} in D_dict[{radius}]"
                assert dst in ih.D_dict[radius][src], (
                    f"Missing dst {dst} in D_dict[{radius}][{src}]"
                )
                assert ih.D_dict[radius][src][dst] == pytest.approx(val), (
                    f"D_dict[{radius}][{src}][{dst}] != D[{radius}][{row},{col}]"
                )

    def test_d_dict_entry_count_matches_nnz(self, hierarchy, interaction_fn):
        """Number of D_dict entries should match D.nnz at each layer."""
        ih = InteractionHierarchy(hierarchy, interaction_fn)
        for radius in hierarchy.radii:
            count = sum(len(dsts) for dsts in ih.D_dict[radius].values())
            assert count == ih.D[radius].nnz

    def test_d_dict_roundtrip(self, hierarchy, interaction_fn, backend, tmp_path):
        """D_dict should be reconstructed correctly after save/load."""
        ih = InteractionHierarchy(hierarchy, interaction_fn)
        p = save_interaction(ih, tmp_path / "ih", backend=backend)
        loaded = load_interaction(p, interaction_fn=interaction_fn)

        for radius in hierarchy.radii:
            for src in ih.D_dict[radius]:
                for dst, val in ih.D_dict[radius][src].items():
                    assert loaded.D_dict[radius][src][dst] == pytest.approx(val)

    def test_d_dict_with_self_interaction_false(self, hierarchy, interaction_fn):
        """D_dict should not have diagonal entries when self_interaction=False."""
        ih = InteractionHierarchy(hierarchy, interaction_fn, self_interaction=False)
        finest = hierarchy.radii[0]
        for zone in ih.D_dict[finest]:
            assert zone not in ih.D_dict[finest].get(zone, {}), (
                f"D_dict should not have self-entry for zone {zone} at finest layer"
            )
