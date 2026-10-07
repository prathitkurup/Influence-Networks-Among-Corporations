"""
Enrich rbo_directed_influence.gml with four node attributes: num_bills, bill_aff_community,
within_comm_net_str, and within_comm_net_inf. Overwrites the GML in-place. See §22.
"""

import sys
import pandas as pd
import networkx as nx
from pathlib import Path

sys.path.insert(0, ".")
from config import DATA_DIR, ROOT, MAX_BILL_DF
from utils.data_loading import load_bills_data
from utils.filtering import filter_bills_by_prevalence

GML_PATH  = ROOT / "visualizations" / "gml" / "rbo_directed_influence.gml"
COMM_PATH = DATA_DIR / "communities" / "communities_affiliation.csv"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_graph(path):
    """Load directed GML; nodes keyed by firm name string (label attr)."""
    G = nx.read_gml(str(path), label="label")
    print(f"Loaded GML: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_num_bills(csv_path):
    """Count unique bills per firm after MAX_BILL_DF prevalence filter; returns {firm_name: int}."""
    df = load_bills_data(csv_path)
    df_dedup = df.drop_duplicates(subset=["fortune_name", "bill_number"])
    if MAX_BILL_DF is not None:
        df_dedup = filter_bills_by_prevalence(
            df_dedup, MAX_BILL_DF, unit_col="bill_number"
        )
    counts = df_dedup.groupby("fortune_name")["bill_number"].nunique().to_dict()
    print(
        f"  num_bills: {len(counts)} firms  |  "
        f"range [{min(counts.values())}, {max(counts.values())}]"
    )
    return counts


def load_bill_aff_communities(path):
    """Load bill affiliation Leiden community assignments; returns {firm_name: community_id}."""
    df = pd.read_csv(path)
    mapping = dict(zip(df["fortune_name"], df["community_aff"].astype(int)))
    print(
        f"  bill_aff_community: {len(mapping)} firms  |  "
        f"{df['community_aff'].nunique()} communities"
    )
    return mapping


# ---------------------------------------------------------------------------
# Within-community metric computation
# ---------------------------------------------------------------------------

def compute_within_community_metrics(G, community_map):
    """Compute within-community net_strength and net_influence for each node, restricted to same-community out-edges.

    Returns (wc_net_str: {node: float}, wc_net_inf: {node: int}).
    """
    wc_net_str = {}
    wc_net_inf = {}

    for node in G.nodes():
        c_v = community_map.get(node)

        net_str_wc = 0.0
        out_sf_wc  = out_tf_wc = 0

        # Sum over out-edges only; both directions exist, so out-edges cover all pairs
        for _, u, d in G.out_edges(node, data=True):
            if community_map.get(u) != c_v:
                continue                            # skip cross-community edges
            net_str_wc += d["rbo"] * d["net_temporal"]
            out_sf_wc  += d["source_firsts"]
            out_tf_wc  += d["target_firsts"]

        wc_net_str[node] = round(net_str_wc, 4)
        wc_net_inf[node] = int(out_sf_wc - out_tf_wc)

    return wc_net_str, wc_net_inf


# ---------------------------------------------------------------------------
# Node annotation
# ---------------------------------------------------------------------------

def annotate_nodes(G, num_bills, comm_map, wc_net_str, wc_net_inf):
    """Write four enrichment attributes onto G nodes in-place; uses -1 sentinel for missing firms."""
    missing_bills = []
    missing_comm  = []

    for node in G.nodes():
        nb = num_bills.get(node)
        if nb is None:
            missing_bills.append(node)
            nb = -1
        G.nodes[node]["num_bills"] = int(nb)

        ca = comm_map.get(node)
        if ca is None:
            missing_comm.append(node)
            ca = -1
        G.nodes[node]["bill_aff_community"] = int(ca)

        G.nodes[node]["within_comm_net_str"] = float(wc_net_str.get(node, 0.0))
        G.nodes[node]["within_comm_net_inf"] = int(wc_net_inf.get(node, 0))

    if missing_bills:
        print(
            f"  Warning: {len(missing_bills)} node(s) missing num_bills → -1 sentinel: "
            f"{missing_bills[:5]}{'...' if len(missing_bills) > 5 else ''}"
        )
    if missing_comm:
        print(
            f"  Warning: {len(missing_comm)} node(s) missing bill_aff_community → -1 sentinel: "
            f"{missing_comm[:5]}{'...' if len(missing_comm) > 5 else ''}"
        )


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(G):
    """Print top/bottom nodes by within_comm_net_inf as a sanity check."""
    nodes_sorted = sorted(
        G.nodes(),
        key=lambda n: G.nodes[n]["within_comm_net_inf"],
        reverse=True,
    )
    header = f"  {'Firm':<42} {'Comm':>5} {'Bills':>6} {'WC Net Str':>12} {'WC Net Inf':>12}"
    print(f"\n  Top 8 by within_comm_net_inf:")
    print(header)
    for n in nodes_sorted[:8]:
        a = G.nodes[n]
        print(
            f"  {n:<42} {a['bill_aff_community']:>5} {a['num_bills']:>6} "
            f"{a['within_comm_net_str']:>12.4f} {a['within_comm_net_inf']:>12}"
        )
    print(f"\n  Bottom 5 by within_comm_net_inf:")
    print(header)
    for n in nodes_sorted[-5:]:
        a = G.nodes[n]
        print(
            f"  {n:<42} {a['bill_aff_community']:>5} {a['num_bills']:>6} "
            f"{a['within_comm_net_str']:>12.4f} {a['within_comm_net_inf']:>12}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading graph...")
    G = load_graph(GML_PATH)

    print("Loading num_bills (post prevalence filter)...")
    num_bills = load_num_bills(DATA_DIR / "opensecrets_lda_reports.csv")

    print("Loading bill affiliation communities...")
    comm_map = load_bill_aff_communities(COMM_PATH)

    print("Computing within-community metrics...")
    wc_net_str, wc_net_inf = compute_within_community_metrics(G, comm_map)

    print("Annotating nodes...")
    annotate_nodes(G, num_bills, comm_map, wc_net_str, wc_net_inf)

    print(f"\nWriting enriched GML -> {GML_PATH}")
    nx.write_gml(G, str(GML_PATH))
    print(f"  Done: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print_summary(G)


if __name__ == "__main__":
    main()
