"""
Toy example: 4-zone linear network from documentation.

This example recreates the exact network from HIERARCHIES_CONCEPTUAL_DOCUMENTATION.md
to demonstrate and validate the hierarchical approach with concrete numbers.

Network topology:
    A ──10── B ──15── C ──20── D

Expected outputs match documentation Section "Worked Numerical Example"
"""

import networkx as nx
import numpy as np

from hierx import Hierarchy, InteractionHierarchy


def create_toy_network():
    """
    Create 4-zone linear network A-B-C-D.

    Returns
    -------
    nx.Graph
        Network with zones 0=A, 1=B, 2=C, 3=D
    """
    G = nx.Graph()

    # Add nodes with labels
    nodes = {0: "A", 1: "B", 2: "C", 3: "D"}
    for node_id, label in nodes.items():
        G.add_node(node_id, label=label, x=node_id * 20, y=0)

    # Add edges with costs (minutes)
    G.add_edge(0, 1, cost=10)  # A-B: 10 minutes
    G.add_edge(1, 2, cost=15)  # B-C: 15 minutes
    G.add_edge(2, 3, cost=20)  # C-D: 20 minutes

    return G, nodes


def print_section(title):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}\n")


def print_cost_matrix(hierarchy, layer_idx, zones, zone_labels):
    """Print cost matrix for a layer."""
    radius = hierarchy.radii[layer_idx]
    representatives = hierarchy.repr_zones[radius]
    costs = hierarchy.costs[radius]

    print(f"Layer {layer_idx}: radius={radius:.0f}, cutoff={hierarchy.cutoffs[radius]}")
    print(f"Representatives: {[zone_labels[r] for r in representatives]}")
    print(f"\nCost matrix C_{layer_idx}:")

    # Print header
    print("     ", end="")
    for dest in representatives:
        print(f"{zone_labels[dest]:>6}", end="")
    print()

    # Print matrix
    for source in representatives:
        print(f"{zone_labels[source]:>4} ", end="")
        for dest in representatives:
            if dest in costs.get(source, {}):
                cost = costs[source][dest]
                print(f"{cost:>6.0f}", end="")
            else:
                print(f"{'−':>6}", end="")
        print()


def main():
    """Run toy example demonstration."""
    print_section("4-Zone Toy Network Example")

    # Create network
    G, zone_labels = create_toy_network()
    zones = sorted(G.nodes())

    print("Network topology: A ──10── B ──15── C ──20── D")
    print(f"\nZones: {[zone_labels[z] for z in zones]}")
    print("\nEdge costs:")
    for u, v, data in G.edges(data=True):
        print(f"  {zone_labels[u]}-{zone_labels[v]}: {data['cost']} minutes")

    # Build hierarchy
    print_section("Building Hierarchy")

    print("Parameters:")
    print("  base_radius = 10")
    print("  increase_factor = 2")
    print("  overlap_factor = 1.5")

    hierarchy = Hierarchy(G, base_radius=10, increase_factor=2, overlap_factor=1.5)

    print(f"\nLayers created: {len(hierarchy.radii)}")
    for i, radius in enumerate(hierarchy.radii):
        n_reps = len(hierarchy.repr_zones[radius])
        print(f"  Layer {i}: radius={radius:.0f}, {n_reps} representatives")

    # Print cost matrices
    print_section("Cost Matrices")

    for layer_idx in range(len(hierarchy.radii)):
        print_cost_matrix(hierarchy, layer_idx, zones, zone_labels)
        print()

    # Test cost lookup
    print_section("Cost Lookup Tests")

    print("Testing get_cost() method:\n")
    test_pairs = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3)]

    for source, dest in test_pairs:
        cost = hierarchy.get_cost(source, dest)
        print(f"  {zone_labels[source]} → {zone_labels[dest]}: {cost:.0f} minutes")

    # Print all-pairs costs from hierarchy
    print("\nAll-pairs shortest paths (via hierarchy):")
    print("     ", end="")
    for dest in zones:
        print(f"{zone_labels[dest]:>6}", end="")
    print()

    for source in zones:
        print(f"{zone_labels[source]:>4} ", end="")
        for dest in zones:
            cost = hierarchy.get_cost(source, dest)
            if np.isinf(cost):
                print(f"{'∞':>6}", end="")
            else:
                print(f"{cost:>6.0f}", end="")
        print()

    # Hierarchy statistics
    print_section("Hierarchy Statistics")

    total_density = hierarchy.get_density()
    finest_density = hierarchy.get_finest_layer_density()
    print(f"Finest layer density: {finest_density:.2%} (sparsity measure)")
    print(f"Total storage (all layers): {total_density:.2%} of dense")
    print(f"Total zones: {len(zones)}")
    print(f"Dense storage: {len(zones) ** 2} costs")
    print(f"Finest layer: {int(finest_density * len(zones) ** 2)} costs")
    print(f"All layers: {int(total_density * len(zones) ** 2)} costs")

    print("\nLayer details:")
    for layer_info in hierarchy.get_layer_info():
        print(
            f"  Radius {layer_info['radius']:.0f}: "
            f"{layer_info['n_representatives']} reps, "
            f"{layer_info['n_costs']} costs stored"
        )

    print("\n" + "=" * 60)
    print("Hierarchy construction successful!")
    print("=" * 60)

    # Part 2: InteractionHierarchy
    print_section("Building InteractionHierarchy")

    # Define interaction function: f(c) = (c + 5)^(-1)
    def interaction_fn(c):
        return (c + 5) ** (-1)

    print("Interaction function: f(c) = (c + 5)^(-1)")

    # Build InteractionHierarchy
    ih = InteractionHierarchy(hierarchy, interaction_fn)

    print(f"\nInteraction matrices built for {len(ih.hierarchy.radii)} layers")

    # Print interaction matrices
    print_section("Interaction Matrices")

    for layer_idx in range(min(2, len(hierarchy.radii))):  # Show first 2 layers
        radius = hierarchy.radii[layer_idx]
        representatives = hierarchy.repr_zones[radius]

        print(f"Layer {layer_idx}: D_{layer_idx} = f(C_{layer_idx})")
        print("     ", end="")
        for dest in representatives:
            print(f"{zone_labels[dest]:>8}", end="")
        print()

        D_matrix = ih.D[radius].toarray()
        for i, source in enumerate(representatives):
            print(f"{zone_labels[source]:>4} ", end="")
            for j in range(len(representatives)):
                value = D_matrix[i, j]
                if value > 0:
                    print(f"{value:>8.3f}", end="")
                else:
                    print(f"{'−':>8}", end="")
            print()
        print()

    # Print correction matrix for finest layer
    print("Correction Matrix Corr_0:")
    radius = hierarchy.radii[0]
    representatives = hierarchy.repr_zones[radius]
    print("     ", end="")
    for dest in representatives:
        print(f"{zone_labels[dest]:>8}", end="")
    print()

    Corr_matrix = ih.Corr[radius].toarray()
    for i, source in enumerate(representatives):
        print(f"{zone_labels[source]:>4} ", end="")
        for j in range(len(representatives)):
            value = Corr_matrix[i, j]
            if value > 0:
                print(f"{value:>8.3f}", end="")
            else:
                print(f"{'0':>8}", end="")
        print()

    # Matrix-vector product example
    print_section("Matrix-Vector Product Example")

    activity = np.array([5, 10, 8, 3], dtype=float)
    print("Activity vector (population in thousands):")
    for i, zone in enumerate(zones):
        print(f"  Zone {zone_labels[zone]}: {activity[i]:.0f}")

    print("\nStep 1: Aggregate activity to each layer")
    for layer_idx in range(min(2, len(hierarchy.radii))):
        radius = hierarchy.radii[layer_idx]
        aHier = ih.G[radius].dot(activity)
        representatives = hierarchy.repr_zones[radius]
        print(f"  Layer {layer_idx}: aHier_{layer_idx} = ", end="")
        print("[" + ", ".join(f"{v:.0f}" for v in aHier) + "]")
        print(f"            Representatives: {[zone_labels[r] for r in representatives]}")

    print("\nStep 2: Compute contributions at each layer")
    for layer_idx in range(min(2, len(hierarchy.radii))):
        radius = hierarchy.radii[layer_idx]
        aHier = ih.G[radius].dot(activity)
        prod = ih.D[radius].dot(aHier)
        corr = ih.Corr[radius].dot(aHier)
        contrib = prod - corr
        print(f"  Layer {layer_idx}:")
        prod_str = ", ".join(f"{v:.2f}" for v in prod)
        print(f"    prod  = D_{layer_idx} × aHier_{layer_idx} = [{prod_str}]")
        corr_str = ", ".join(f"{v:.2f}" for v in corr)
        print(f"    corr  = Corr_{layer_idx} × aHier_{layer_idx} = [{corr_str}]")
        print(f"    contrib = prod - corr = [{', '.join(f'{v:.2f}' for v in contrib)}]")

    print("\nStep 3: Compute final accessibility")
    accessibility = ih.matvec(activity)

    print("Final accessibility:")
    for i, zone in enumerate(zones):
        print(f"  Zone {zone_labels[zone]}: {accessibility[i]:.2f}")

    print("\nInterpretation:")
    max_zone = zones[np.argmax(accessibility)]
    print(f"Zone {zone_labels[max_zone]} has highest accessibility because it:")
    print("- Has good local activity")
    print("- Is well-connected to other zones")
    print("- Benefits from being central in the network")

    # Hierarchy statistics
    print_section("InteractionHierarchy Statistics")

    density = ih.get_density()
    print(f"Interaction matrix density: {density:.2%} of dense matrix")

    print("\nLayer details:")
    for layer_info in ih.get_layer_info():
        print(
            f"  Radius {layer_info['radius']:.0f}: "
            f"{layer_info['n_representatives']} reps, "
            f"{layer_info['n_interactions']} interactions, "
            f"{layer_info['n_corrections']} corrections"
        )

    print("\n" + "=" * 60)
    print("Complete hierarchical computation successful!")
    print("=" * 60)


if __name__ == "__main__":
    main()
