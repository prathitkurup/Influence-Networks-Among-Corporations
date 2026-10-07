"""
Community detection via the Leiden algorithm (Traag et al., 2019).
Leiden guarantees internally connected communities; use γ=1.0 as default.
"""

import warnings
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg


# -- Conversion --

def networkx_to_igraph(G_nx, weight_attr="weight"):
    """Convert weighted undirected NetworkX graph to igraph, preserving node names and edge weights."""
    nodes = list(G_nx.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    edges = []
    weights = []
    for u, v, data in G_nx.edges(data=True):
        edges.append((node_idx[u], node_idx[v]))
        weights.append(float(data.get(weight_attr, 1.0)))

    G_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)
    G_ig.vs["name"] = nodes
    G_ig.es["weight"] = weights
    return G_ig


# -- Leiden detection --

def detect_communities(G_nx, resolution=1.0, seed=42, n_iterations=10,
                       weight_attr="weight"):
    """Run Leiden on a NetworkX graph; returns (partition dict, modularity Q, summary dict)."""
    if len(G_nx.nodes()) == 0:
        return {}, 0.0, {}

    G_ig = networkx_to_igraph(G_nx, weight_attr=weight_attr)

    part = leidenalg.find_partition(
        G_ig,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=n_iterations,
    )

    # Map igraph vertex indices back to node names
    partition = {}
    for comm_id, members in enumerate(part):
        for idx in members:
            name = G_ig.vs[idx]["name"]
            partition[name] = comm_id

    modularity = part.modularity

    # Build summary dict
    summary = {}
    for node, comm_id in partition.items():
        summary.setdefault(comm_id, []).append(node)
    for comm_id in summary:
        summary[comm_id] = sorted(summary[comm_id])

    return partition, modularity, summary


# -- Resolution sweep --

def sweep_resolution(G_nx, resolutions=None, seed=42, n_iterations=10,
                     weight_attr="weight", verbose=True):
    """Run Leiden at multiple γ values; returns list of dicts with resolution, n_communities, modularity."""
    if resolutions is None:
        resolutions = [0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 3.0, 4.0]

    results = []
    for gamma in resolutions:
        part, Q, summ = detect_communities(G_nx, resolution=gamma, seed=seed,
                                           n_iterations=n_iterations,
                                           weight_attr=weight_attr)
        row = {"resolution": gamma, "n_communities": len(summ), "modularity": Q}
        results.append(row)
        if verbose:
            sizes = sorted([len(v) for v in summ.values()], reverse=True)
            print(f"  γ={gamma:.2f}  →  {len(summ):2d} communities  "
                  f"Q={Q:.4f}  sizes: {sizes}")

    return results


# -- Diagnostics --

def print_community_summary(summary, partition, G_nx=None, label="",
                            weight_attr="weight"):
    """Print per-community size, members, and (if G_nx given) intra-community edge density and total weight."""
    header = f"-- Community Summary{' (' + label + ')' if label else ''} --"
    print(f"\n{header}")
    print(f"  Total communities: {len(summary)}")
    print(f"  Total nodes:       {sum(len(v) for v in summary.values())}")

    for comm_id in sorted(summary.keys()):
        members = summary[comm_id]
        line = f"\n  Community {comm_id}  ({len(members)} firms)"
        if G_nx is not None:
            sub = G_nx.subgraph(members)
            n, m = sub.number_of_nodes(), sub.number_of_edges()
            max_edges = n * (n - 1) / 2 if n > 1 else 1
            density = m / max_edges if max_edges > 0 else 0.0
            total_w = sum(d.get(weight_attr, 1.0)
                          for _, _, d in sub.edges(data=True))
            line += (f"  |  intra-edges={m}  "
                     f"density={density:.3f}  total_weight={total_w:.2f}")
        print(line)
        # Print members in two columns
        for i in range(0, len(members), 2):
            left  = members[i]
            right = members[i + 1] if i + 1 < len(members) else ""
            print(f"    {left:<42} {right}")


def community_sizes(summary):
    """Return a sorted list of (community_id, size) tuples, largest first."""
    return sorted(((cid, len(members)) for cid, members in summary.items()),
                  key=lambda x: x[1], reverse=True)
