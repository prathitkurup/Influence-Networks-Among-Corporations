from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


def plot_circular(H, title, path):
    """
    Circular layout plot of subgraph H. Node size ∝ weighted degree.
    Parent directory is created automatically if it doesn't exist.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    pos = nx.circular_layout(H)

    strengths = np.array([H.degree(n, weight="weight") for n in H.nodes()])
    sizes = (1600 + 3000 * (strengths - strengths.min())
             / (strengths.max() - strengths.min() + 1))

    edges = list(H.edges())
    weights = np.array([H[u][v]["weight"] for u, v in edges]) if edges else np.array([])
    widths = (1 + 25 * (weights - weights.min())
              / (weights.max() - weights.min() + 1)) if weights.size else []

    fig, ax = plt.subplots(figsize=(16, 16))
    nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color="#337aff",
                           edgecolors="black", linewidths=1.5, ax=ax)
    if edges:
        nx.draw_networkx_edges(H, pos, width=widths, alpha=0.7,
                               edge_color="gray", ax=ax)
    nx.draw_networkx_labels(H, pos, font_size=12, font_weight="bold",
                            bbox=dict(facecolor="white", edgecolor="none",
                                      alpha=0.7),
                            ax=ax)

    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved -> {path}")


def plot_directed_circular(G, title, path, top_k=20):
    """Circular directed plot for a DiGraph, top-k nodes by total involvement.

    Node color: green = net influencer (net_influence > 0), red = net follower (net_influence < 0), gray = net influence = 0.
    Node size ∝ out_strength + in_strength. Arrow width ∝ edge weight.
    Curved edges (arc3,rad=0.15) prevent A→B and B→A from overlapping.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if G.number_of_nodes() == 0:
        print(f"  (empty graph — PNG skipped)")
        return

    involvement = {n: (G.nodes[n].get("out_strength", 0)
                       + G.nodes[n].get("in_strength", 0)) for n in G.nodes()}
    H = G.subgraph(sorted(involvement, key=involvement.get, reverse=True)[:top_k]).copy()
    if H.number_of_nodes() == 0:
        return

    pos    = nx.circular_layout(H)
    colors = ["#27ae60" if H.nodes[n].get("net_influence", 0) > 0
              else "#e74c3c" if H.nodes[n].get("net_influence", 0) < 0
              else "#95a5a6" for n in H.nodes()]
    inv    = np.array([involvement.get(n, 0) for n in H.nodes()])
    sizes  = 1200 + 3000 * (inv - inv.min()) / (inv.max() - inv.min() + 1e-9)

    edges   = list(H.edges(data=True))
    weights = np.array([d.get("weight", 1) for _, _, d in edges]) if edges else np.array([])
    widths  = (1 + 4 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
               if weights.size else [])

    fig, ax = plt.subplots(figsize=(16, 16))
    nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=colors,
                           edgecolors="black", linewidths=1.5, ax=ax)
    if edges:
        nx.draw_networkx_edges(H, pos, edgelist=[(u, v) for u, v, _ in edges],
                               width=list(widths), alpha=0.65, edge_color="#555555",
                               arrows=True, arrowsize=20,
                               connectionstyle="arc3,rad=0.15", ax=ax)
    nx.draw_networkx_labels(H, pos, font_size=11, font_weight="bold",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
                            ax=ax)
    ax.legend(handles=[
        mpatches.Patch(facecolor="#27ae60", label="Net influencer (net_influence > 0)"),
        mpatches.Patch(facecolor="#e74c3c", label="Net follower (net_influence < 0)"),
        mpatches.Patch(facecolor="#95a5a6", label="Net neutral (net_influence = 0)"),
    ], loc="lower right", fontsize=12)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  PNG saved -> {path}")
