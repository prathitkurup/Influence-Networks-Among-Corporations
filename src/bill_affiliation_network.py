"""
Supporting network: shared-bill affiliation between Fortune 500 firms.

Outputs: data/network_edges/affiliation_edges.csv,
data/communities/communities_affiliation.csv,
data/centralities/centrality_affiliation.csv,
visualizations/gml/bill_affiliation_network.gml
"""

import sys
import pandas as pd
import networkx as nx

sys.path.insert(0, ".")
from config import DATA_DIR, ROOT, MAX_BILL_DF
from utils.data_loading import load_bills_data
from utils.filtering import filter_bills_by_prevalence, prevalence_summary
from utils.network_building import (build_graph, write_gml_with_communities,
                                    _cent_df_to_attrs, top_k_subgraph, edge_weight_stats)
from utils.visualization import plot_circular
from utils.centrality import compute_community_centralities, print_community_centralities
from utils.community import detect_communities, print_community_summary, sweep_resolution

GML_PATH       = str(ROOT / "visualizations" / "gml" / "bill_affiliation_network.gml")
PNG_PATH       = str(ROOT / "visualizations" / "png" / "bill_affiliation_network.png")

TOP_K             = 20
CENTRALITY_K      = 10
WRITE_GML         = True
RUN_SWEEP         = False
LEIDEN_RESOLUTION = 1.0


def company_bill_edges(df, max_bill_df=MAX_BILL_DF):
    """Build shared-bill edge list with affil_norm weights."""
    df = df.drop_duplicates(subset=["fortune_name", "bill_number"])
    if max_bill_df is not None:
        prevalence_summary(df)
        df = filter_bills_by_prevalence(df, max_bill_df, unit_col="bill_number")
    N_total = df["bill_number"].nunique()
    bill_companies = df.groupby("bill_number")["fortune_name"].apply(list)
    records = []
    for _, companies in bill_companies.items():
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                if companies[i] != companies[j]:
                    src, tgt = ((companies[i], companies[j])
                                if companies[i] < companies[j]
                                else (companies[j], companies[i]))
                    records.append({"source": src, "target": tgt})
    if not records:
        return pd.DataFrame(columns=["source", "target", "weight", "affil_norm"])
    edges = (pd.DataFrame(records)
               .groupby(["source", "target"])
               .size()
               .reset_index(name="weight"))
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")
    edges = edges[edges["weight"] > 0].copy()
    edges["affil_norm"] = edges["weight"] / N_total
    return edges


def main():
    df    = load_bills_data(DATA_DIR / "opensecrets_lda_reports.csv")
    edges = company_bill_edges(df)

    print(f"Bills: {df['bill_number'].nunique():,}  |  Companies: {df['fortune_name'].nunique():,}")
    edge_weight_stats(edges, "shared bills (filtered)")

    edges_path = DATA_DIR / "network_edges" / "affiliation_edges.csv"
    edges.to_csv(edges_path, index=False)
    print(f"Edges -> {edges_path}")

    G = build_graph(edges[["source", "target", "weight"]])

    if RUN_SWEEP:
        sweep_resolution(G, resolutions=[0.5, 0.75, 1.0, 1.15, 1.25], seed=42)

    partition, Q, comm_summary = detect_communities(G, resolution=LEIDEN_RESOLUTION, seed=42)
    print(f"\nLeiden (gamma={LEIDEN_RESOLUTION})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="bill affiliation")

    comm_df = (
        pd.DataFrame([{"fortune_name": n, "community_aff": cid} for n, cid in partition.items()])
        .sort_values(["community_aff", "fortune_name"])
        .reset_index(drop=True)
    )
    comm_path = DATA_DIR / "communities" / "communities_affiliation.csv"
    comm_df.to_csv(comm_path, index=False)
    print(f"Communities -> {comm_path}")

    cent_df = None
    if WRITE_GML or CENTRALITY_K > 0:
        cent_df = compute_community_centralities(G, partition)
        if CENTRALITY_K > 0:
            print_community_centralities(cent_df, k=CENTRALITY_K)
            cent_path = DATA_DIR / "centralities" / "centrality_affiliation.csv"
            cent_df.to_csv(cent_path, index=False)
            print(f"Centrality -> {cent_path}")

    if WRITE_GML:
        node_attrs = _cent_df_to_attrs(cent_df) or {}
        node_attrs["kcore"] = nx.core_number(G)
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)

    if TOP_K > 0:
        H = top_k_subgraph(G, k=TOP_K)
        plot_circular(H, title=f"Top {TOP_K} Fortune 500 Bill Affiliation Network", path=PNG_PATH)


if __name__ == "__main__":
    main()
