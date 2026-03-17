"""
Compact, versioned save/load for Hierarchy and InteractionHierarchy objects.

Two backends are supported:

* **NPZ** (zero extra dependencies) — uses ``numpy.savez_compressed``.
* **HDF5** (optional, requires ``h5py>=3.0``) — gzip-compressed datasets,
  preferred for spatial-database interoperability.

Files use standard extensions (``.npz`` or ``.h5``).  The format is
auto-detected on load via magic bytes (HDF5 ``\\x89HDF``, NPZ/zip ``PK``).

Interaction functions are *not* serialised (they are arbitrary callables).
A human-readable ``interaction_fn_hint`` string can be stored instead.
Loaded :class:`InteractionHierarchy` objects can still call
:meth:`~InteractionHierarchy.matvec` because the net operator
``_net = D - Corr`` is recomputed from the stored sparse matrices.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
HDF5_CHUNK_SIZE = 65536  # elements per HDF5 chunk (~512 KB for float64)

try:
    import h5py  # type: ignore[import-untyped]  # noqa: F401

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def _extract_zone_coords(hierarchy: Any) -> dict[str, np.ndarray]:
    """Extract zone x/y coordinates from hierarchy.network node attributes.

    Returns a dict with ``zone_x`` and ``zone_y`` arrays (sorted by zone id),
    or an empty dict if the network has no coordinates.
    """
    net = getattr(hierarchy, "network", None)
    if net is None:
        return {}
    zones = list(getattr(hierarchy, "zones", None) or sorted(net.nodes()))
    if not zones:
        return {}
    first = zones[0]
    if "x" not in net.nodes[first] or "y" not in net.nodes[first]:
        return {}
    return {
        "zone_x": np.array([net.nodes[z]["x"] for z in zones], dtype=np.float64),
        "zone_y": np.array([net.nodes[z]["y"] for z in zones], dtype=np.float64),
    }


class StorageError(Exception):
    """Raised for save/load failures."""


# ---------------------------------------------------------------------------
# Flatten helpers — nested dicts → columnar arrays
# ---------------------------------------------------------------------------


def _flatten_costs(
    costs: dict[float, dict[int, dict[int, float]]],
    radii: list[float],
) -> dict[str, np.ndarray]:
    """Flatten ``costs[radius][src][dst] = val`` to four columnar arrays."""
    radius_to_idx = {r: i for i, r in enumerate(radii)}
    layer_idx_list: list[int] = []
    src_list: list[int] = []
    dst_list: list[int] = []
    val_list: list[float] = []
    for radius, src_dict in costs.items():
        if radius not in radius_to_idx:
            continue
        li = radius_to_idx[radius]
        for src, dst_dict in src_dict.items():
            for dst, val in dst_dict.items():
                layer_idx_list.append(li)
                src_list.append(src)
                dst_list.append(dst)
                val_list.append(val)
    return {
        "costs_layer_idx": np.asarray(layer_idx_list, dtype=np.int32),
        "costs_src": np.asarray(src_list, dtype=np.int64),
        "costs_dst": np.asarray(dst_list, dtype=np.int64),
        "costs_val": np.asarray(val_list, dtype=np.float64),
    }


def _unflatten_costs(
    arrays: dict[str, np.ndarray],
    radii: list[float],
) -> dict[float, dict[int, dict[int, float]]]:
    """Reconstruct nested costs dict from columnar arrays."""
    costs: dict[float, dict[int, dict[int, float]]] = {r: {} for r in radii}
    layer_idx = arrays["costs_layer_idx"]
    src = arrays["costs_src"]
    dst = arrays["costs_dst"]
    val = arrays["costs_val"]
    for i in range(len(layer_idx)):
        r = radii[int(layer_idx[i])]
        s = int(src[i])
        d = int(dst[i])
        costs[r].setdefault(s, {})[d] = float(val[i])
    return costs


def _flatten_groups(
    groups: dict[float, dict[int, int]],
    radii: list[float],
) -> dict[str, np.ndarray]:
    """Flatten ``groups[radius][zone] = rep`` to three columnar arrays."""
    radius_to_idx = {r: i for i, r in enumerate(radii)}
    layer_idx_list: list[int] = []
    zone_list: list[int] = []
    rep_list: list[int] = []
    for radius, zone_dict in groups.items():
        if radius not in radius_to_idx:
            continue
        li = radius_to_idx[radius]
        for zone, rep in zone_dict.items():
            layer_idx_list.append(li)
            zone_list.append(zone)
            rep_list.append(rep)
    return {
        "groups_layer_idx": np.asarray(layer_idx_list, dtype=np.int32),
        "groups_zone": np.asarray(zone_list, dtype=np.int64),
        "groups_rep": np.asarray(rep_list, dtype=np.int64),
    }


def _unflatten_groups(
    arrays: dict[str, np.ndarray],
    radii: list[float],
) -> dict[float, dict[int, int]]:
    """Reconstruct nested groups dict from columnar arrays."""
    groups: dict[float, dict[int, int]] = {r: {} for r in radii}
    layer_idx = arrays["groups_layer_idx"]
    zone = arrays["groups_zone"]
    rep = arrays["groups_rep"]
    for i in range(len(layer_idx)):
        r = radii[int(layer_idx[i])]
        groups[r][int(zone[i])] = int(rep[i])
    return groups


def _flatten_repr_zones(
    repr_zones: dict[float, list[int]],
    radii: list[float],
) -> dict[str, np.ndarray]:
    """Flatten ``repr_zones[radius] = [zones]`` to two columnar arrays."""
    radius_to_idx = {r: i for i, r in enumerate(radii)}
    layer_idx_list: list[int] = []
    zone_list: list[int] = []
    for radius, zones in repr_zones.items():
        if radius not in radius_to_idx:
            continue
        li = radius_to_idx[radius]
        for z in zones:
            layer_idx_list.append(li)
            zone_list.append(z)
    return {
        "repr_layer_idx": np.asarray(layer_idx_list, dtype=np.int32),
        "repr_zone": np.asarray(zone_list, dtype=np.int64),
    }


def _unflatten_repr_zones(
    arrays: dict[str, np.ndarray],
    radii: list[float],
) -> dict[float, list[int]]:
    """Reconstruct repr_zones dict from columnar arrays."""
    repr_zones: dict[float, list[int]] = {r: [] for r in radii}
    layer_idx = arrays["repr_layer_idx"]
    zone = arrays["repr_zone"]
    for i in range(len(layer_idx)):
        r = radii[int(layer_idx[i])]
        repr_zones[r].append(int(zone[i]))
    return repr_zones


def _flatten_cutoffs(
    cutoffs: dict[float, float | None],
    radii: list[float],
) -> np.ndarray:
    """Encode cutoffs as float64 array with NaN for None entries."""
    return np.array(
        [cutoffs[r] if cutoffs[r] is not None else np.nan for r in radii],
        dtype=np.float64,
    )


def _unflatten_cutoffs(
    arr: np.ndarray,
    radii: list[float],
) -> dict[float, float | None]:
    """Reconstruct cutoffs dict (NaN → None)."""
    return {r: (None if np.isnan(arr[i]) else float(arr[i])) for i, r in enumerate(radii)}


def _flatten_zone_indices(
    zone_indices: dict[float, dict[int, int]],
    radii: list[float],
) -> dict[str, np.ndarray]:
    """Flatten ``zone_indices[radius][zone] = idx`` to three arrays."""
    radius_to_idx = {r: i for i, r in enumerate(radii)}
    layer_idx_list: list[int] = []
    zone_list: list[int] = []
    idx_list: list[int] = []
    for radius, zone_dict in zone_indices.items():
        li = radius_to_idx[radius]
        for zone, idx in zone_dict.items():
            layer_idx_list.append(li)
            zone_list.append(zone)
            idx_list.append(idx)
    return {
        "ci_layer_idx": np.asarray(layer_idx_list, dtype=np.int32),
        "ci_zone": np.asarray(zone_list, dtype=np.int64),
        "ci_idx": np.asarray(idx_list, dtype=np.int32),
    }


def _unflatten_zone_indices(
    arrays: dict[str, np.ndarray],
    radii: list[float],
) -> dict[float, dict[int, int]]:
    """Reconstruct zone_indices dict from columnar arrays."""
    zone_indices: dict[float, dict[int, int]] = {r: {} for r in radii}
    layer_idx = arrays["ci_layer_idx"]
    zone = arrays["ci_zone"]
    idx = arrays["ci_idx"]
    for i in range(len(layer_idx)):
        r = radii[int(layer_idx[i])]
        zone_indices[r][int(zone[i])] = int(idx[i])
    return zone_indices


def _flatten_sparse_matrices(
    matrices: dict[float, sp.csr_matrix],
    radii: list[float],
    prefix: str,
) -> dict[str, np.ndarray]:
    """Flatten a dict of sparse CSR matrices to arrays."""
    arrays: dict[str, np.ndarray] = {}
    for k, radius in enumerate(radii):
        m = matrices[radius].tocsr()
        arrays[f"{prefix}_{k}_data"] = np.asarray(m.data)
        arrays[f"{prefix}_{k}_indices"] = np.asarray(m.indices)
        arrays[f"{prefix}_{k}_indptr"] = np.asarray(m.indptr)
        arrays[f"{prefix}_{k}_shape"] = np.array(m.shape, dtype=np.int64)
    return arrays


def _unflatten_sparse_matrices(
    arrays: dict[str, np.ndarray],
    radii: list[float],
    prefix: str,
) -> dict[float, sp.csr_matrix]:
    """Reconstruct dict of sparse CSR matrices from arrays."""
    matrices: dict[float, sp.csr_matrix] = {}
    for k, radius in enumerate(radii):
        data = arrays[f"{prefix}_{k}_data"]
        indices = arrays[f"{prefix}_{k}_indices"]
        indptr = arrays[f"{prefix}_{k}_indptr"]
        shape = tuple(arrays[f"{prefix}_{k}_shape"])
        matrices[radius] = sp.csr_matrix((data, indices, indptr), shape=shape)
    return matrices


def _network_to_arrays(network: Any) -> dict[str, np.ndarray]:
    """Serialise a NetworkX graph to arrays (edge list + node attributes).

    Only node IDs, optional ``x``/``y`` coordinates, and the edge ``cost``
    attribute are preserved.  Other node and edge attributes are **not**
    serialised.
    """
    nodes = sorted(network.nodes())
    node_ids = np.asarray(nodes, dtype=np.int64)

    # Node attributes (x, y if present)
    has_xy = all("x" in network.nodes[n] and "y" in network.nodes[n] for n in nodes)
    arrays: dict[str, np.ndarray] = {"net_nodes": node_ids}
    if has_xy:
        arrays["net_x"] = np.array([network.nodes[n]["x"] for n in nodes], dtype=np.float64)
        arrays["net_y"] = np.array([network.nodes[n]["y"] for n in nodes], dtype=np.float64)

    # Edges
    src_list: list[int] = []
    dst_list: list[int] = []
    cost_list: list[float] = []
    for u, v, d in network.edges(data=True):
        src_list.append(u)
        dst_list.append(v)
        cost_list.append(d.get("cost", 1.0))
    arrays["net_edge_src"] = np.asarray(src_list, dtype=np.int64)
    arrays["net_edge_dst"] = np.asarray(dst_list, dtype=np.int64)
    arrays["net_edge_cost"] = np.asarray(cost_list, dtype=np.float64)
    return arrays


def _arrays_to_network(arrays: dict[str, np.ndarray]) -> Any:
    """Reconstruct a NetworkX graph from arrays."""
    import networkx as nx

    G = nx.Graph()
    nodes = arrays["net_nodes"]
    has_xy = "net_x" in arrays and "net_y" in arrays
    for i, n in enumerate(nodes):
        n = int(n)
        attrs: dict[str, Any] = {}
        if has_xy:
            attrs["x"] = float(arrays["net_x"][i])
            attrs["y"] = float(arrays["net_y"][i])
        G.add_node(n, **attrs)

    src = arrays["net_edge_src"]
    dst = arrays["net_edge_dst"]
    cost = arrays["net_edge_cost"]
    for i in range(len(src)):
        G.add_edge(int(src[i]), int(dst[i]), cost=float(cost[i]))
    return G


# ---------------------------------------------------------------------------
# Backend classes
# ---------------------------------------------------------------------------


class _NpzBackend:
    """Save/load using ``numpy.savez_compressed``."""

    @staticmethod
    def save(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
        # Encode metadata as arrays
        save_dict: dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, str):
                save_dict[f"_meta_{k}"] = np.array(v)
            elif isinstance(v, (int, float)):
                save_dict[f"_meta_{k}"] = np.array(v)
            else:
                save_dict[f"_meta_{k}"] = np.asarray(v)
        save_dict.update(arrays)
        # numpy.savez_compressed appends .npz — save to a temp name then rename
        tmp = path.with_suffix(".npz")
        np.savez_compressed(str(tmp), **save_dict)
        # np.savez_compressed may or may not re-append .npz depending on
        # whether the path already ends in .npz.  Handle both cases.
        actual = tmp if tmp.exists() else Path(str(tmp) + ".npz")
        if actual != path:
            actual.rename(path)

    @staticmethod
    def load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        with np.load(str(path), allow_pickle=False) as npz:
            arrays: dict[str, np.ndarray] = {}
            metadata: dict[str, Any] = {}
            for key in npz.files:
                if key.startswith("_meta_"):
                    val = npz[key]
                    # Scalar arrays → Python scalars
                    if val.ndim == 0:
                        item = val.item()
                        metadata[key[6:]] = item
                    else:
                        metadata[key[6:]] = val
                else:
                    arrays[key] = npz[key]
        return arrays, metadata


class _Hdf5Backend:
    """Save/load using ``h5py`` with hierarchical groups and chunked datasets.

    On-disk layout::

        /
          attrs: format_version, object_type, base_radius, ...
          zones           int64[n_zones]
          zone_x          float64[n_zones]   (if coordinates exist)
          zone_y          float64[n_zones]
          radii           float64[n_layers]
          cutoffs         float64[n_layers]
          layers/
            0/                                ← finest layer
              repr_zones      int64[n_reps]
              groups_zone     int64[n]        sorted by zone
              groups_rep      int64[n]
              costs_src_zones int64[n_src]    sorted unique sources (CSR row keys)
              costs_src_offsets int64[n_src+1] CSR-style offsets
              costs_dst       int64[nnz]      chunked
              costs_val       float64[nnz]    chunked
              D_data, D_indices, D_indptr, D_shape   (InteractionHierarchy only)
              Corr_*, G_*     ...same pattern
              ci_zone         int64[n_reps]   sorted
              ci_idx          int32[n_reps]
            1/
              ...
          network/                            (optional)
            nodes, x, y, edge_src, edge_dst, edge_cost
    """

    @staticmethod
    def _chunked_ds(
        group: Any,
        name: str,
        data: np.ndarray,
    ) -> None:
        """Create a chunked, gzip-compressed dataset (skip for tiny arrays)."""
        if data.size == 0:
            group.create_dataset(name, data=data)
            return
        chunk = min(HDF5_CHUNK_SIZE, data.shape[0])
        group.create_dataset(
            name,
            data=data,
            chunks=(chunk,),
            compression="gzip",
            compression_opts=4,
        )

    @staticmethod
    def _small_ds(group: Any, name: str, data: np.ndarray) -> None:
        """Create a small, gzip-compressed dataset (no chunking)."""
        group.create_dataset(name, data=data, compression="gzip", compression_opts=4)

    @classmethod
    def save(cls, path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
        import h5py

        radii = arrays["radii"]
        n_layers = len(radii)

        with h5py.File(str(path), "w") as f:
            # -- Root attributes (metadata) --
            for k, v in metadata.items():
                f.attrs[k] = v

            # -- Root datasets --
            cls._small_ds(f, "zones", arrays["zones"])
            cls._small_ds(f, "radii", radii)
            cls._small_ds(f, "cutoffs", arrays["cutoffs"])

            if "zone_x" in arrays:
                cls._small_ds(f, "zone_x", arrays["zone_x"])
                cls._small_ds(f, "zone_y", arrays["zone_y"])

            # -- Per-layer groups --
            layers_grp = f.create_group("layers")

            # Pre-index columnar arrays by layer for costs, groups, repr, zone_indices
            has_costs = "costs_layer_idx" in arrays
            has_groups = "groups_layer_idx" in arrays
            has_repr = "repr_layer_idx" in arrays
            has_ci = "ci_layer_idx" in arrays

            # Determine which sparse matrix prefixes exist
            sparse_prefixes: list[str] = []
            for prefix in ("D", "Corr", "G"):
                if f"{prefix}_0_data" in arrays:
                    sparse_prefixes.append(prefix)

            for k in range(n_layers):
                lg = layers_grp.create_group(str(k))

                # --- Representative zones ---
                if has_repr:
                    mask = arrays["repr_layer_idx"] == k
                    repr_zones = np.sort(arrays["repr_zone"][mask])
                    cls._small_ds(lg, "repr_zones", repr_zones)

                # --- Groups (sorted by zone) ---
                if has_groups:
                    mask = arrays["groups_layer_idx"] == k
                    g_zone = arrays["groups_zone"][mask]
                    g_rep = arrays["groups_rep"][mask]
                    order = np.argsort(g_zone)
                    cls._small_ds(lg, "groups_zone", g_zone[order])
                    cls._small_ds(lg, "groups_rep", g_rep[order])

                # --- Costs in CSR-style layout ---
                if has_costs:
                    mask = arrays["costs_layer_idx"] == k
                    c_src = arrays["costs_src"][mask]
                    c_dst = arrays["costs_dst"][mask]
                    c_val = arrays["costs_val"][mask]

                    if len(c_src) > 0:
                        # Sort by (src, dst) for CSR layout
                        order = np.lexsort((c_dst, c_src))
                        c_src = c_src[order]
                        c_dst = c_dst[order]
                        c_val = c_val[order]

                        # Build CSR-style offsets
                        unique_src, first_idx, counts = np.unique(
                            c_src,
                            return_index=True,
                            return_counts=True,
                        )
                        offsets = np.empty(len(unique_src) + 1, dtype=np.int64)
                        offsets[0] = 0
                        np.cumsum(counts, out=offsets[1:])

                        cls._small_ds(lg, "costs_src_zones", unique_src)
                        cls._small_ds(lg, "costs_src_offsets", offsets)
                        cls._chunked_ds(lg, "costs_dst", c_dst)
                        cls._chunked_ds(lg, "costs_val", c_val)
                    else:
                        # Empty layer
                        cls._small_ds(lg, "costs_src_zones", np.array([], dtype=np.int64))
                        cls._small_ds(lg, "costs_src_offsets", np.array([0], dtype=np.int64))
                        cls._chunked_ds(lg, "costs_dst", np.array([], dtype=np.int64))
                        cls._chunked_ds(lg, "costs_val", np.array([], dtype=np.float64))

                # --- Sparse matrices (D, Corr, G) ---
                for prefix in sparse_prefixes:
                    data_key = f"{prefix}_{k}_data"
                    cls._chunked_ds(lg, f"{prefix}_data", arrays[data_key])
                    cls._chunked_ds(lg, f"{prefix}_indices", arrays[f"{prefix}_{k}_indices"])
                    cls._small_ds(lg, f"{prefix}_indptr", arrays[f"{prefix}_{k}_indptr"])
                    cls._small_ds(lg, f"{prefix}_shape", arrays[f"{prefix}_{k}_shape"])

                # --- Cell indices (sorted by zone) ---
                if has_ci:
                    mask = arrays["ci_layer_idx"] == k
                    ci_zone = arrays["ci_zone"][mask]
                    ci_idx = arrays["ci_idx"][mask]
                    order = np.argsort(ci_zone)
                    cls._small_ds(lg, "ci_zone", ci_zone[order])
                    cls._small_ds(lg, "ci_idx", ci_idx[order])

            # -- Network group (optional) --
            if "net_nodes" in arrays:
                net_grp = f.create_group("network")
                cls._small_ds(net_grp, "nodes", arrays["net_nodes"])
                if "net_x" in arrays:
                    cls._small_ds(net_grp, "x", arrays["net_x"])
                    cls._small_ds(net_grp, "y", arrays["net_y"])
                cls._chunked_ds(net_grp, "edge_src", arrays["net_edge_src"])
                cls._chunked_ds(net_grp, "edge_dst", arrays["net_edge_dst"])
                cls._chunked_ds(net_grp, "edge_cost", arrays["net_edge_cost"])

    @classmethod
    def load(cls, path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Read structured HDF5 layout and assemble into flat arrays dict."""
        import h5py

        arrays: dict[str, np.ndarray] = {}
        metadata: dict[str, Any] = {}

        with h5py.File(str(path), "r") as f:
            # -- Metadata from root attrs --
            for k, v in f.attrs.items():
                metadata[k] = v if not isinstance(v, bytes) else v.decode()

            # -- Root datasets --
            arrays["zones"] = f["zones"][()]
            arrays["radii"] = f["radii"][()]
            arrays["cutoffs"] = f["cutoffs"][()]

            if "zone_x" in f:
                arrays["zone_x"] = f["zone_x"][()]
                arrays["zone_y"] = f["zone_y"][()]

            # -- Per-layer data --
            n_layers = len(arrays["radii"])
            layers_grp = f["layers"]

            # Accumulators for columnar arrays
            costs_layer: list[np.ndarray] = []
            costs_src: list[np.ndarray] = []
            costs_dst: list[np.ndarray] = []
            costs_val: list[np.ndarray] = []

            groups_layer: list[np.ndarray] = []
            groups_zone: list[np.ndarray] = []
            groups_rep: list[np.ndarray] = []

            repr_layer: list[np.ndarray] = []
            repr_zone: list[np.ndarray] = []

            ci_layer: list[np.ndarray] = []
            ci_zone: list[np.ndarray] = []
            ci_idx: list[np.ndarray] = []

            # Detect sparse prefixes from first layer
            first_layer = layers_grp["0"] if "0" in layers_grp else None
            sparse_prefixes: list[str] = []
            if first_layer is not None:
                for prefix in ("D", "Corr", "G"):
                    if f"{prefix}_data" in first_layer:
                        sparse_prefixes.append(prefix)

            for k in range(n_layers):
                lg = layers_grp[str(k)]

                # --- Representative zones ---
                if "repr_zones" in lg:
                    rz = lg["repr_zones"][()]
                    repr_layer.append(np.full(len(rz), k, dtype=np.int32))
                    repr_zone.append(rz)

                # --- Groups ---
                if "groups_zone" in lg:
                    gz = lg["groups_zone"][()]
                    gr = lg["groups_rep"][()]
                    groups_layer.append(np.full(len(gz), k, dtype=np.int32))
                    groups_zone.append(gz)
                    groups_rep.append(gr)

                # --- Costs (CSR → columnar) ---
                if "costs_src_zones" in lg:
                    src_zones = lg["costs_src_zones"][()]
                    offsets = lg["costs_src_offsets"][()]
                    c_dst = lg["costs_dst"][()]
                    c_val = lg["costs_val"][()]

                    # Expand CSR offsets back to per-entry source zones
                    if len(c_dst) > 0:
                        n_entries = len(c_dst)
                        expanded_src = np.empty(n_entries, dtype=np.int64)
                        for i, sz in enumerate(src_zones):
                            expanded_src[offsets[i] : offsets[i + 1]] = sz

                        costs_layer.append(np.full(n_entries, k, dtype=np.int32))
                        costs_src.append(expanded_src)
                        costs_dst.append(c_dst)
                        costs_val.append(c_val)

                # --- Sparse matrices ---
                for prefix in sparse_prefixes:
                    arrays[f"{prefix}_{k}_data"] = lg[f"{prefix}_data"][()]
                    arrays[f"{prefix}_{k}_indices"] = lg[f"{prefix}_indices"][()]
                    arrays[f"{prefix}_{k}_indptr"] = lg[f"{prefix}_indptr"][()]
                    arrays[f"{prefix}_{k}_shape"] = lg[f"{prefix}_shape"][()]

                # --- Cell indices ---
                if "ci_zone" in lg:
                    cz = lg["ci_zone"][()]
                    cx = lg["ci_idx"][()]
                    ci_layer.append(np.full(len(cz), k, dtype=np.int32))
                    ci_zone.append(cz)
                    ci_idx.append(cx)

            # Concatenate columnar arrays
            if costs_layer:
                arrays["costs_layer_idx"] = np.concatenate(costs_layer)
                arrays["costs_src"] = np.concatenate(costs_src)
                arrays["costs_dst"] = np.concatenate(costs_dst)
                arrays["costs_val"] = np.concatenate(costs_val)
            else:
                arrays["costs_layer_idx"] = np.array([], dtype=np.int32)
                arrays["costs_src"] = np.array([], dtype=np.int64)
                arrays["costs_dst"] = np.array([], dtype=np.int64)
                arrays["costs_val"] = np.array([], dtype=np.float64)

            if groups_layer:
                arrays["groups_layer_idx"] = np.concatenate(groups_layer)
                arrays["groups_zone"] = np.concatenate(groups_zone)
                arrays["groups_rep"] = np.concatenate(groups_rep)
            else:
                arrays["groups_layer_idx"] = np.array([], dtype=np.int32)
                arrays["groups_zone"] = np.array([], dtype=np.int64)
                arrays["groups_rep"] = np.array([], dtype=np.int64)

            if repr_layer:
                arrays["repr_layer_idx"] = np.concatenate(repr_layer)
                arrays["repr_zone"] = np.concatenate(repr_zone)
            else:
                arrays["repr_layer_idx"] = np.array([], dtype=np.int32)
                arrays["repr_zone"] = np.array([], dtype=np.int64)

            if ci_layer:
                arrays["ci_layer_idx"] = np.concatenate(ci_layer)
                arrays["ci_zone"] = np.concatenate(ci_zone)
                arrays["ci_idx"] = np.concatenate(ci_idx)

            # -- Network --
            if "network" in f:
                net_grp = f["network"]
                arrays["net_nodes"] = net_grp["nodes"][()]
                if "x" in net_grp:
                    arrays["net_x"] = net_grp["x"][()]
                    arrays["net_y"] = net_grp["y"][()]
                arrays["net_edge_src"] = net_grp["edge_src"][()]
                arrays["net_edge_dst"] = net_grp["edge_dst"][()]
                arrays["net_edge_cost"] = net_grp["edge_cost"][()]

        return arrays, metadata


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _detect_format(path: Path) -> str:
    """Detect file format from magic bytes.

    Returns ``"hdf5"`` or ``"npz"``.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic[:4] == b"\x89HDF":
        return "hdf5"
    if magic[:2] == b"PK":
        return "npz"
    raise StorageError(f"Unrecognised file format (magic bytes: {magic!r}): {path}")


def _choose_backend(backend: str) -> str:
    """Resolve ``"auto"`` to a concrete backend name."""
    if backend == "auto":
        return "hdf5" if HAS_H5PY else "npz"
    if backend == "hdf5" and not HAS_H5PY:
        raise StorageError("h5py is not installed; install it with: pip install h5py>=3.0")
    if backend not in ("npz", "hdf5"):
        raise ValueError(f"backend must be 'auto', 'npz', or 'hdf5', got {backend!r}")
    return backend


def _get_backend(name: str) -> _NpzBackend | _Hdf5Backend:
    """Return the storage backend instance for the given name."""
    if name == "npz":
        return _NpzBackend()
    return _Hdf5Backend()


_BACKEND_EXT: dict[str, str] = {"npz": ".npz", "hdf5": ".h5"}


def _ensure_ext(path: Path, backend_name: str) -> Path:
    """Append the standard extension for *backend_name* if not already present."""
    ext = _BACKEND_EXT[backend_name]
    if path.suffix != ext:
        path = path.with_suffix(path.suffix + ext)
    return path


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_hierarchy(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    network: Any | None,
) -> Any:
    """Build a Hierarchy from deserialised data, bypassing __init__."""
    from hierx.backends import convert_nx_to_csr
    from hierx.hierarchy import Hierarchy

    h = object.__new__(Hierarchy)

    # Scalar parameters
    h.base_radius = float(metadata["base_radius"])
    h.increase_factor = float(metadata["increase_factor"])
    h.overlap_factor = float(metadata["overlap_factor"])
    h._min_representatives = int(metadata["min_representatives"])
    h._backend = str(metadata.get("sp_backend", "scipy"))
    h._n_workers = 1

    # Radii and zones
    h.radii = [float(r) for r in arrays["radii"]]
    h.zones = [int(z) for z in arrays["zones"]]
    h._zone_set = set(h.zones)

    # Nested structures
    h.costs = _unflatten_costs(arrays, h.radii)
    h.groups = _unflatten_groups(arrays, h.radii)
    h.repr_zones = _unflatten_repr_zones(arrays, h.radii)
    h.cutoffs = _unflatten_cutoffs(arrays["cutoffs"], h.radii)

    # Network
    if network is not None:
        h.network = network
    elif "net_nodes" in arrays:
        h.network = _arrays_to_network(arrays)
    else:
        h.network = None

    # Backend caches
    h._csr = None
    h._idx_to_node = None
    h._node_to_idx = None
    h._gt_graph = None
    h._node_to_vertex = None
    h._weight_prop = None
    h._pool = None
    h._shm_pool = None
    h._total_nodes_explored = 0  # not meaningful for loaded hierarchies

    if h.network is not None:
        try:
            h._csr, h._idx_to_node, h._node_to_idx = convert_nx_to_csr(h.network)
        except Exception:
            pass  # non-critical; only needed for new cost computation

    return h


def _reconstruct_interaction(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    interaction_fn: Callable[[float], float] | None,
    network: Any | None,
) -> Any:
    """Build an InteractionHierarchy from deserialised data, bypassing __init__."""
    from hierx.interaction import InteractionHierarchy

    # Reconstruct the embedded hierarchy first
    h = _reconstruct_hierarchy(arrays, metadata, network)

    ih = object.__new__(InteractionHierarchy)
    ih.hierarchy = h
    ih.interaction_fn = interaction_fn
    ih.self_interaction = bool(metadata.get("self_interaction", True))

    # Sparse matrices
    ih.D = _unflatten_sparse_matrices(arrays, h.radii, "D")
    ih.Corr = _unflatten_sparse_matrices(arrays, h.radii, "Corr")
    ih.G = _unflatten_sparse_matrices(arrays, h.radii, "G")

    # Cell indices
    ih.zone_indices = _unflatten_zone_indices(arrays, h.radii)

    # Reconstruct D_dict from sparse D matrices + zone_indices
    ih.D_dict = {}
    for radius in h.radii:
        idx_to_zone = {i: zone for zone, i in ih.zone_indices[radius].items()}
        D_coo = ih.D[radius].tocoo()
        d_dict_layer: dict[int, dict[int, float]] = {}
        for row, col, val in zip(D_coo.row, D_coo.col, D_coo.data):
            src_zone = idx_to_zone[row]
            dst_zone = idx_to_zone[col]
            d_dict_layer.setdefault(src_zone, {})[dst_zone] = float(val)
        ih.D_dict[radius] = d_dict_layer

    # Recompute net operator: _net[k] = D[k] - Corr[k]
    ih._net = {}
    for radius in h.radii:
        ih._net[radius] = (ih.D[radius] - ih.Corr[radius]).tocsr()

    # Initialize LinearOperator base class
    n_zones = len(h.zones)
    LinearOperator.__init__(ih, shape=(n_zones, n_zones), dtype=np.float64)

    return ih


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_hierarchy(
    hierarchy: Any,
    path: str | Path,
    *,
    include_network: bool = False,
    backend: str = "auto",
) -> Path:
    """Save a :class:`~hierx.Hierarchy` to disk.

    Parameters
    ----------
    hierarchy : Hierarchy
        The hierarchy to save.
    path : str or Path
        Destination path.  ``.npz`` or ``.h5`` is appended if missing.
    include_network : bool, default False
        Whether to embed the NetworkX graph topology (nodes and edges).
        Zone coordinates (``zone_x``, ``zone_y``) are always included when
        available regardless of this flag — they are needed for spatial queries
        and visualisation and add negligible file size.

        When ``True``, only node IDs, ``x``/``y`` coordinates, and the edge
        ``cost`` attribute are serialised.  Other node and edge attributes
        (e.g. labels, road type) are **not** preserved.  If full attribute
        fidelity is needed, pass the original graph to
        ``load_hierarchy(path, network=G)`` instead.
    backend : str, default "auto"
        ``"auto"`` (HDF5 if h5py installed, else NPZ), ``"npz"``, or ``"hdf5"``.

    Returns
    -------
    Path
        The path the file was written to.
    """
    backend_name = _choose_backend(backend)
    path = _ensure_ext(Path(path), backend_name)

    arrays: dict[str, np.ndarray] = {}
    arrays["radii"] = np.asarray(hierarchy.radii, dtype=np.float64)
    arrays["zones"] = np.asarray(hierarchy.zones, dtype=np.int64)
    arrays["cutoffs"] = _flatten_cutoffs(hierarchy.cutoffs, hierarchy.radii)
    arrays.update(_flatten_costs(hierarchy.costs, hierarchy.radii))
    arrays.update(_flatten_groups(hierarchy.groups, hierarchy.radii))
    arrays.update(_flatten_repr_zones(hierarchy.repr_zones, hierarchy.radii))

    # Zone coordinates (always, if available — tiny and needed for spatial queries)
    arrays.update(_extract_zone_coords(hierarchy))

    if include_network and hierarchy.network is not None:
        arrays.update(_network_to_arrays(hierarchy.network))

    metadata = {
        "format_version": FORMAT_VERSION,
        "object_type": "Hierarchy",
        "base_radius": hierarchy.base_radius,
        "increase_factor": hierarchy.increase_factor,
        "overlap_factor": hierarchy.overlap_factor,
        "min_representatives": hierarchy._min_representatives,
        "sp_backend": hierarchy._backend,
        "backend": backend_name,
    }

    _get_backend(backend_name).save(path, arrays, metadata)
    logger.info("Saved Hierarchy to %s (%s backend)", path, backend_name)
    return path


def load_hierarchy(
    path: str | Path,
    *,
    network: Any | None = None,
) -> Any:
    """Load a :class:`~hierx.Hierarchy` from a ``.npz`` or ``.h5`` file.

    Parameters
    ----------
    path : str or Path
        File to load.
    network : nx.Graph or None, optional
        If provided, used as the network (avoids storing/loading it).

    Returns
    -------
    Hierarchy
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    fmt = _detect_format(path)
    arrays, metadata = _get_backend(fmt).load(path)

    file_version = int(metadata.get("format_version", 0))
    if file_version > FORMAT_VERSION:
        raise StorageError(
            f"File format version {file_version} is newer than supported version "
            f"{FORMAT_VERSION}. Upgrade hierx to read this file."
        )

    obj_type = str(metadata.get("object_type", ""))
    if obj_type not in ("Hierarchy", "InteractionHierarchy"):
        raise StorageError(f"Expected object_type 'Hierarchy', got {obj_type!r} in {path}")

    return _reconstruct_hierarchy(arrays, metadata, network)


def save_interaction(
    interaction: Any,
    path: str | Path,
    *,
    include_network: bool = False,
    backend: str = "auto",
    interaction_fn_hint: str = "",
) -> Path:
    """Save an :class:`~hierx.InteractionHierarchy` to disk.

    The full :class:`Hierarchy` data is embedded so that the resulting file
    is self-contained.

    Parameters
    ----------
    interaction : InteractionHierarchy
        The interaction hierarchy to save.
    path : str or Path
        Destination path.  ``.npz`` or ``.h5`` is appended if missing.
    include_network : bool, default False
        Whether to embed the NetworkX graph topology (nodes and edges).
        Zone coordinates (``zone_x``, ``zone_y``) are always included when
        available regardless of this flag — they are needed for spatial queries
        and visualisation and add negligible file size.

        When ``True``, only node IDs, ``x``/``y`` coordinates, and the edge
        ``cost`` attribute are serialised.  Other node and edge attributes
        (e.g. labels, road type) are **not** preserved.  If full attribute
        fidelity is needed, pass the original graph to
        ``load_interaction(path, network=G)`` instead.
    backend : str, default "auto"
        ``"auto"``, ``"npz"``, or ``"hdf5"``.
    interaction_fn_hint : str, default ""
        Human-readable description of the interaction function.

    Returns
    -------
    Path
        The path the file was written to.
    """
    backend_name = _choose_backend(backend)
    path = _ensure_ext(Path(path), backend_name)
    h = interaction.hierarchy

    arrays: dict[str, np.ndarray] = {}

    # Hierarchy data (embedded)
    arrays["radii"] = np.asarray(h.radii, dtype=np.float64)
    arrays["zones"] = np.asarray(h.zones, dtype=np.int64)
    arrays["cutoffs"] = _flatten_cutoffs(h.cutoffs, h.radii)
    arrays.update(_flatten_costs(h.costs, h.radii))
    arrays.update(_flatten_groups(h.groups, h.radii))
    arrays.update(_flatten_repr_zones(h.repr_zones, h.radii))

    # Interaction-specific: sparse matrices and zone indices
    arrays.update(_flatten_sparse_matrices(interaction.D, h.radii, "D"))
    arrays.update(_flatten_sparse_matrices(interaction.Corr, h.radii, "Corr"))
    arrays.update(_flatten_sparse_matrices(interaction.G, h.radii, "G"))
    arrays.update(_flatten_zone_indices(interaction.zone_indices, h.radii))

    # Zone coordinates (always, if available — tiny and needed for spatial queries)
    arrays.update(_extract_zone_coords(h))

    if include_network and h.network is not None:
        arrays.update(_network_to_arrays(h.network))

    metadata = {
        "format_version": FORMAT_VERSION,
        "object_type": "InteractionHierarchy",
        "base_radius": h.base_radius,
        "increase_factor": h.increase_factor,
        "overlap_factor": h.overlap_factor,
        "min_representatives": h._min_representatives,
        "sp_backend": h._backend,
        "backend": backend_name,
        "interaction_fn_hint": interaction_fn_hint,
        "self_interaction": int(interaction.self_interaction),
    }

    _get_backend(backend_name).save(path, arrays, metadata)
    logger.info("Saved InteractionHierarchy to %s (%s backend)", path, backend_name)
    return path


def load_interaction(
    path: str | Path,
    *,
    interaction_fn: Callable[[float], float] | None = None,
    network: Any | None = None,
) -> Any:
    """Load an :class:`~hierx.InteractionHierarchy` from a ``.npz`` or ``.h5`` file.

    Parameters
    ----------
    path : str or Path
        File to load.
    interaction_fn : callable or None, optional
        Interaction function to attach.  Not required for ``matvec()`` to work
        (the net operator is recomputed from stored matrices).
    network : nx.Graph or None, optional
        If provided, used as the network.

    Returns
    -------
    InteractionHierarchy
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    fmt = _detect_format(path)
    arrays, metadata = _get_backend(fmt).load(path)

    file_version = int(metadata.get("format_version", 0))
    if file_version > FORMAT_VERSION:
        raise StorageError(
            f"File format version {file_version} is newer than supported version "
            f"{FORMAT_VERSION}. Upgrade hierx to read this file."
        )

    obj_type = str(metadata.get("object_type", ""))
    if obj_type != "InteractionHierarchy":
        raise StorageError(
            f"Expected object_type 'InteractionHierarchy', got {obj_type!r} in {path}"
        )

    return _reconstruct_interaction(arrays, metadata, interaction_fn, network)
