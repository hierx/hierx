"""Generate reference results for regression testing.

Saves hierarchy outputs (groups, costs, repr_zones) for several network
sizes so we can verify that algorithmic changes don't alter behavior.
"""

import json

import numpy as np

from hierx import Hierarchy
from hierx.utils import generate_grid_network


def hierarchy_to_dict(h):
    """Serialize a Hierarchy's key outputs to a JSON-safe dict."""
    result = {
        "radii": h.radii,
        "repr_zones": {str(r): sorted(h.repr_zones[r]) for r in h.radii},
        "groups": {
            str(r): {str(z): int(rep) for z, rep in sorted(h.groups[r].items())} for r in h.radii
        },
        "costs_summary": {},
        "sample_get_cost": {},
    }
    # Per-layer cost stats
    for r in h.radii:
        n_entries = sum(len(h.costs[r][s]) for s in h.costs[r])
        result["costs_summary"][str(r)] = {
            "n_sources": len(h.costs[r]),
            "n_entries": n_entries,
        }

    # Sample get_cost lookups
    zones = h.zones
    pairs = [
        (zones[0], zones[-1]),
        (zones[0], zones[len(zones) // 2]),
        (zones[len(zones) // 4], zones[3 * len(zones) // 4]),
        (zones[len(zones) // 3], zones[2 * len(zones) // 3]),
    ]
    for src, dst in pairs:
        cost = h.get_cost(src, dst)
        result["sample_get_cost"][f"{src}->{dst}"] = float(cost) if np.isfinite(cost) else "inf"

    return result


def main():
    configs = [
        ("grid_20x20", 20, 20, 1000, 2000),
        ("grid_50x50", 50, 50, 1000, 2000),
    ]

    for name, rows, cols, spacing, base_radius in configs:
        print(f"Generating reference for {name} ({rows * cols} nodes)...")
        G = generate_grid_network(rows, cols, spacing=spacing)
        h = Hierarchy(G, base_radius=base_radius, backend="scipy")

        ref = hierarchy_to_dict(h)
        ref["config"] = {
            "rows": rows,
            "cols": cols,
            "spacing": spacing,
            "base_radius": base_radius,
            "increase_factor": 2,
            "overlap_factor": 1.5,
        }

        path = f"tests/reference_{name}.json"
        with open(path, "w") as f:
            json.dump(ref, f, indent=2, sort_keys=True)
        print(f"  Saved to {path}")
        print(f"  Layers: {len(h.radii)}, radii: {h.radii}")
        for r in h.radii:
            print(f"    r={r}: {len(h.repr_zones[r])} reps")

    print("\nDone. Reference files generated.")


if __name__ == "__main__":
    main()
