"""
Centrality utilities: global metrics (Tier 1) and community-scoped measures (Tier 2).

Tier 2 includes within-community eigenvector, Guimerà-Amaral z/P, PageRank, and Katz.
Reference: Guimerà & Amaral (2005). Nature, 433, 895-900.
"""

import numpy as np
import pandas as pd
import networkx as nx


# -- Tier 1: global centrality --

def compute_katz_centrality(G, weight_attr="weight", normalized=True, max_iter=2000):
    """
    Katz-Bonacich centrality. alpha auto-set to 0.85 / spectral_radius.
    Falls back to 0.50/rho, then weighted degree if convergence fails.
    Katz (1953); Bonacich (1987).
    """
    if G.number_of_nodes() == 0:
        return {}

    A = nx.to_numpy_array(G, weight=weight_attr)
    try:
        eigenvalues = np.linalg.eigvals(A)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
    except np.linalg.LinAlgError:
        spectral_radius = 0.0

    if spectral_radius == 0.0:
        return {n: 0.0 for n in G.nodes()}

    alpha = 0.85 / spectral_radius

    try:
        katz = nx.katz_centrality(
            G, alpha=alpha, beta=1.0, weight=weight_attr,
            max_iter=max_iter, normalized=normalized,
        )
    except nx.PowerIterationFailedConvergence:
        # More conservative alpha if convergence fails at 0.85/rho
        alpha = 0.50 / spectral_radius
        try:
            katz = nx.katz_centrality(
                G, alpha=alpha, beta=1.0, weight=weight_attr,
                max_iter=max_iter * 2, normalized=normalized,
            )
        except nx.PowerIterationFailedConvergence:
            # Last-resort fallback: weighted degree (same units, no path penalty)
            deg = dict(G.degree(weight=weight_attr))
            max_d = max(deg.values()) or 1.0
            katz = {n: v / max_d for n, v in deg.items()}

    return katz


def compute_centralities(G):
    return {
        "degree":      nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G, weight="weight"),
        "closeness":   nx.closeness_centrality(G, distance="weight"),
        "eigenvector": nx.eigenvector_centrality(G, weight="weight",
                                                  max_iter=1000),
    }


def print_top_centralities(centralities, k=10):
    for name, values in centralities.items():
        print(f"\nTop {k} by {name.upper()}:")
        for node, val in sorted(values.items(), key=lambda x: x[1],
                                 reverse=True)[:k]:
            print(f"  {node}: {val:.4f}")


# -- Tier 2 helpers --

def _community_map(partition):
    """Return {community_id: [node, ...]} from {node: community_id}."""
    comm_map = {}
    for node, cid in partition.items():
        comm_map.setdefault(cid, []).append(node)
    return comm_map


def compute_within_community_eigenvector(G, partition, weight_attr="weight"):
    """Eigenvector centrality on each community subgraph; falls back to weighted degree for small or non-convergent communities."""
    comm_map = _community_map(partition)
    ec = {}
    for cid, members in comm_map.items():
        sub = G.subgraph(members)
        if len(members) < 3:
            # Eigenvector is ill-defined for very small subgraphs.
            deg = nx.degree_centrality(sub)
            ec.update(deg)
        else:
            try:
                ev = nx.eigenvector_centrality(sub, weight=weight_attr,
                                               max_iter=2000)
                ec.update(ev)
            except nx.PowerIterationFailedConvergence:
                # Fall back to weighted degree for this community.
                total_w = {n: sub.degree(n, weight=weight_attr) for n in sub}
                max_w = max(total_w.values()) or 1
                ec.update({n: v / max_w for n, v in total_w.items()})
    return ec


def compute_within_community_zscore(G, partition, weight_attr="weight"):
    """Guimerà-Amaral z-score: z_i = (κ_is − mean(κ_s)) / std(κ_s), where κ_is is within-community weighted degree."""
    comm_map = _community_map(partition)

    # Compute κ_ic (within-community weighted degree) for every node.
    kic = {}
    for node in G.nodes():
        my_comm = partition.get(node)
        if my_comm is None:
            kic[node] = 0.0
            continue
        kic[node] = sum(
            d.get(weight_attr, 1.0)
            for nbr, d in G[node].items()
            if partition.get(nbr) == my_comm
        )

    # Compute z per community.
    z = {}
    for cid, members in comm_map.items():
        vals = [kic[m] for m in members if m in kic]
        mean_k = np.mean(vals) if vals else 0.0
        std_k  = np.std(vals,  ddof=0) if len(vals) > 1 else 0.0
        for m in members:
            if std_k > 0:
                z[m] = (kic.get(m, 0.0) - mean_k) / std_k
            else:
                z[m] = 0.0
    return z


def compute_participation_coefficient(G, partition, weight_attr="weight"):
    """Guimerà-Amaral participation coefficient: P_i = 1 − Σ_c (κ_ic / κ_i)². P=0 is within-community; P→1 is cross-community."""
    P = {}
    for node in G.nodes():
        if node not in partition:
            P[node] = None
            continue
        ki = G.degree(node, weight=weight_attr)
        if ki == 0:
            P[node] = 0.0
            continue
        comm_weights = {}
        for nbr, data in G[node].items():
            c = partition.get(nbr)
            if c is None:
                continue
            comm_weights[c] = comm_weights.get(c, 0.0) + data.get(weight_attr, 1.0)
        P[node] = 1.0 - sum((w / ki) ** 2 for w in comm_weights.values())
    return P


def classify_ga_role(z, P):
    """
    Guimerà-Amaral (2005) node role based on z-score and participation coeff.
    The connector_hub and non_hub_connector roles identify the cross-industry
    political entrepreneurs of greatest theoretical interest.
    """
    if P is None:
        return "unknown"
    if z >= 2.5:
        if P < 0.30:  return "provincial_hub"
        if P < 0.75:  return "connector_hub"
        return "kinless_hub"
    else:
        if P < 0.05:  return "ultra_peripheral"
        if P < 0.625: return "peripheral"
        if P < 0.80:  return "non_hub_connector"
        return "kinless"


# -- Tier 2: combined community centrality --

def compute_community_centralities(G, partition, weight_attr="weight"):
    """Compute all centrality tiers; returns DataFrame with firm, community, eigenvector, z_score, P, PageRank, Katz, ga_role."""
    ec   = compute_within_community_eigenvector(G, partition, weight_attr)
    z    = compute_within_community_zscore(G, partition, weight_attr)
    P    = compute_participation_coefficient(G, partition, weight_attr)
    pr   = nx.pagerank(G, weight=weight_attr)
    katz = compute_katz_centrality(G, weight_attr=weight_attr)

    rows = []
    for node in sorted(G.nodes()):
        p_val = P.get(node)
        z_val = z.get(node, 0.0)
        rows.append({
            "firm":                    node,
            "community":               partition.get(node),
            "within_comm_eigenvector": round(ec.get(node, 0.0), 6),
            "z_score":                 round(z_val, 4),
            "participation_coeff":     round(p_val, 4) if p_val is not None else None,
            "global_pagerank":         round(pr.get(node, 0.0), 6),
            "katz_centrality":         round(katz.get(node, 0.0), 6),
            "ga_role":                 classify_ga_role(z_val, p_val),
        })

    return (pd.DataFrame(rows)
              .sort_values("within_comm_eigenvector", ascending=False)
              .reset_index(drop=True))


def print_community_centralities(cent_df, k=10):
    """Print top-k firms by within-community eigenvector, participation coeff, PageRank, Katz, and GA role distribution."""
    print("\n-- Community Centrality: Top Industry Leaders "
          "(within-community eigenvector) --")
    top_ec = cent_df.nlargest(k, "within_comm_eigenvector")
    print(f"  {'Firm':<42} {'Comm':>5}  {'EC':>8}  {'GA role'}")
    for _, r in top_ec.iterrows():
        print(f"  {r['firm']:<42} {str(r['community']):>5}  "
              f"{r['within_comm_eigenvector']:>8.4f}  {r['ga_role']}")

    print(f"\n-- Community Centrality: Top Cross-Industry Connectors "
          "(participation coeff) --")
    top_P = cent_df.dropna(subset=["participation_coeff"]).nlargest(
        k, "participation_coeff"
    )
    print(f"  {'Firm':<42} {'Comm':>5}  {'P':>8}  {'GA role'}")
    for _, r in top_P.iterrows():
        print(f"  {r['firm']:<42} {str(r['community']):>5}  "
              f"{r['participation_coeff']:>8.4f}  {r['ga_role']}")

    print(f"\n-- Community Centrality: Top Global Hubs (PageRank) --")
    top_pr = cent_df.nlargest(k, "global_pagerank")
    print(f"  {'Firm':<42} {'Comm':>5}  {'PR':>8}  {'GA role'}")
    for _, r in top_pr.iterrows():
        print(f"  {r['firm']:<42} {str(r['community']):>5}  "
              f"{r['global_pagerank']:>8.5f}  {r['ga_role']}")

    print(f"\n-- Community Centrality: Top Katz-Bonacich Hubs --")
    if "katz_centrality" in cent_df.columns:
        top_katz = cent_df.nlargest(k, "katz_centrality")
        print(f"  {'Firm':<42} {'Comm':>5}  {'Katz':>8}  {'GA role'}")
        for _, r in top_katz.iterrows():
            print(f"  {r['firm']:<42} {str(r['community']):>5}  "
                  f"{r['katz_centrality']:>8.5f}  {r['ga_role']}")
    else:
        print("  (katz_centrality not computed for this network)")

    print(f"\n-- Guimerà-Amaral Role Distribution --")
    role_counts = cent_df["ga_role"].value_counts()
    for role, count in role_counts.items():
        print(f"  {role:<22}: {count:>4}")
