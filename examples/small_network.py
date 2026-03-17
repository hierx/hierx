"""
Small network example: 50-zone grid for visualization.

This example demonstrates the hierarchical approach on a small but realistic
network, showing:
- Hierarchy construction with multiple layers
- Accessibility computation
- Visualization of network structure and results
"""

import matplotlib.pyplot as plt
import numpy as np

from hierx import Hierarchy, InteractionHierarchy, generate_grid_network


def visualize_network(G, hierarchy, title="Network Structure"):
    """Visualize network with zones colored by group membership."""
    fig, axes = plt.subplots(1, min(3, len(hierarchy.radii)), figsize=(15, 5))
    if len(hierarchy.radii) == 1:
        axes = [axes]

    # Get positions
    pos = {node: (G.nodes[node]["x"], G.nodes[node]["y"]) for node in G.nodes()}

    # Plot first few layers
    for idx, radius in enumerate(hierarchy.radii[:3]):
        ax = axes[idx] if len(hierarchy.radii) > 1 else axes[0]

        # Color nodes by group
        representatives = hierarchy.repr_zones[radius]
        groups = hierarchy.groups[radius]

        # Assign colors to groups
        rep_to_color = {rep: i for i, rep in enumerate(representatives)}
        node_colors = [rep_to_color[groups[node]] for node in G.nodes()]

        # Draw network
        import networkx as nx

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap="tab20", ax=ax, node_size=100)

        # Highlight representatives
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=representatives,
            node_color="red",
            node_size=200,
            node_shape="s",
            ax=ax,
            alpha=0.5,
        )

        ax.set_title(f"Layer {idx}: r={radius:.0f}, {len(representatives)} reps")
        ax.set_aspect("equal")
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def visualize_accessibility(G, accessibility, title="Accessibility"):
    """Visualize accessibility values on the network."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Get positions
    pos = {node: (G.nodes[node]["x"], G.nodes[node]["y"]) for node in G.nodes()}

    # Draw network
    import networkx as nx

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2)

    # Draw nodes colored by accessibility
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=accessibility,
        cmap="YlOrRd",
        node_size=200,
        ax=ax,
        vmin=np.min(accessibility),
        vmax=np.max(accessibility),
    )

    # Add colorbar
    plt.colorbar(nodes, ax=ax, label="Accessibility")

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig


def main():
    """Run small network example."""
    x_size = 10
    y_size = 10
    spacing = 1000

    print("=" * 60)
    print(f"Small Network Example: {x_size}×{y_size} grid")
    print("=" * 60)

    # Generate network

    print(f"\n1. Generating {x_size}×{y_size} grid network...")
    G = generate_grid_network(x_size, y_size, spacing=spacing)
    n_zones = len(G.nodes())
    print(f"   Created network with {n_zones} zones and {len(G.edges())} edges")

    # Build hierarchy
    print("\n2. Building hierarchical cost matrix...")
    hierarchy = Hierarchy(G, base_radius=2500, increase_factor=2, overlap_factor=1.5)

    print(f"   Hierarchy built with {len(hierarchy.radii)} layers:")
    for i, layer_info in enumerate(hierarchy.get_layer_info()):
        print(
            f"     Layer {i}: radius={layer_info['radius']:.0f}m, "
            f"{layer_info['n_representatives']} representatives, "
            f"{layer_info['n_costs']} costs"
        )

    total_density = hierarchy.get_density()
    finest_density = hierarchy.get_finest_layer_density()
    print(f"   Finest layer density: {finest_density:.2%} (sparsity measure)")
    print(f"   Total storage (all layers): {total_density:.2%} of dense")
    print(
        f"   ({int(finest_density * n_zones**2)} costs at finest layer, "
        f"{int(total_density * n_zones**2)} total across all layers)"
    )

    # Build interaction hierarchy
    print("\n3. Building interaction hierarchy...")

    # Use less aggressive decay: offset=2000 instead of 1000, exponent=-1 instead of -2
    def interaction_fn(c):
        return (c + 2000) ** (-1)  # Power law with offset

    ih = InteractionHierarchy(hierarchy, interaction_fn)

    print("   Interaction matrices built")
    print(f"   Interaction density: {ih.get_density():.2%}")

    # Compute accessibility with uniform activity
    print("\n4. Computing accessibility (uniform activity)...")
    activity_uniform = np.ones(n_zones)
    accessibility_uniform = ih.matvec(activity_uniform)

    print(
        f"   Accessibility range: [{np.min(accessibility_uniform):.6f}, "
        f"{np.max(accessibility_uniform):.6f}]"
    )
    print(f"   Mean accessibility: {np.mean(accessibility_uniform):.6f}")
    print(f"   Std deviation: {np.std(accessibility_uniform):.6f}")

    # Compute accessibility with non-uniform activity
    print("\n5. Computing accessibility (non-uniform activity)...")
    # Create activity concentrated in center
    activity_nonuniform = np.zeros(n_zones)
    center_zones = [
        z
        for z in G.nodes()
        if 3 <= G.nodes[z]["x"] / 1000 <= 6 and 1 <= G.nodes[z]["y"] / 1000 <= 3
    ]
    activity_nonuniform[center_zones] = 10.0
    activity_nonuniform += 1.0  # Background activity

    accessibility_nonuniform = ih.matvec(activity_nonuniform)

    print(
        f"   Accessibility range: [{np.min(accessibility_nonuniform):.6f}, "
        f"{np.max(accessibility_nonuniform):.6f}]"
    )
    print(f"   Mean accessibility: {np.mean(accessibility_nonuniform):.6f}")
    print(f"   Std deviation: {np.std(accessibility_nonuniform):.6f}")

    # Find zones with highest accessibility
    top_zones = np.argsort(accessibility_nonuniform)[-5:][::-1]
    print("\n   Top 5 zones by accessibility:")
    for rank, zone in enumerate(top_zones, 1):
        x, y = G.nodes[zone]["x"], G.nodes[zone]["y"]
        acc = accessibility_nonuniform[zone]
        print(f"     {rank}. Zone {zone} at ({x:.0f}, {y:.0f}): {acc:.6f}")

    # Visualize
    print("\n6. Creating visualizations...")
    try:
        # Visualize network structure
        fig1 = visualize_network(G, hierarchy, "Hierarchical Groups by Layer")
        fig1.savefig("small_network_groups.png", dpi=150, bbox_inches="tight")
        print("   Saved: small_network_groups.png")

        # Visualize accessibility (uniform)
        fig2 = visualize_accessibility(G, accessibility_uniform, "Accessibility (Uniform Activity)")
        fig2.savefig("small_network_accessibility_uniform.png", dpi=150, bbox_inches="tight")
        print("   Saved: small_network_accessibility_uniform.png")

        # Visualize accessibility (non-uniform)
        fig3 = visualize_accessibility(
            G, accessibility_nonuniform, "Accessibility (Concentrated Activity)"
        )
        fig3.savefig("small_network_accessibility_nonuniform.png", dpi=150, bbox_inches="tight")
        print("   Saved: small_network_accessibility_nonuniform.png")

        plt.close("all")
    except Exception as e:
        print(f"   Visualization error (matplotlib display issue): {e}")
        print("   (Figures would be saved in interactive environment)")

    print("\n" + "=" * 60)
    print("Small network example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
