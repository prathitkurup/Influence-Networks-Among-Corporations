"""
Congress-wide directed RBO influence network — 116th Congress.

Builds a single aggregate network from all 8 quarters of lobbying data where:
  - Each firm pair (i,j) with shared top-30 bills produces TWO directed edges
  - Edge weight   = proportional RBO: [(source_firsts + ties/2) / shared_bills] x RBO
  - Edge rbo      = full RBO similarity (same for both edge directions of a pair)
  - Edge net_temporal = source_firsts - target_firsts (signed; from source's perspective)
  - Node net_strength  = Σ_j [RBO(i,j) x net_temporal(i,j)] — RBO-weighted temporal dominance
  - Node net_influence = Σ_j (i_firsts_j - j_firsts_j) — unweighted first-mover count
  - Node color = #2ECC71 green (net_strength > 0), #E74C3C red (net_strength < 0),
                 #95A5A6 gray (net_strength == 0 or isolated)

Output files:
  data/rbo_directed_influence.csv
  data/ranked_bill_lists.csv
  visualizations/gml/rbo_directed_influence.gml
  visualizations/png/rbo_directed_influence.png
"""

import sys
import itertools
import pandas as pd
import networkx as nx
from pathlib import Path

sys.path.insert(0, ".")
from config import DATA_DIR, ROOT, MAX_BILL_DF
from utils.data_loading import load_bills_data, assign_quarters
from utils.filtering import filter_bills_by_prevalence
from utils.similarity import (
    aggregate_per_firm_bill, compute_zero_budget_fracs,
    build_ranked_lists, rbo_score,
)
from utils.visualization import plot_directed_circular

VIZ_GML_DIR = ROOT / "visualizations" / "gml"
VIZ_PNG_DIR = ROOT / "visualizations" / "png"

RBO_P     = 0.85   # Fortune 500-calibrated; see §18
TOP_BILLS = 30     # top bills per firm for ranking and RBO
TOP_K     = 20     # nodes included in PNG visualization
WRITE_CSV         = True
WRITE_RANKED_CSV  = True   # data/ranked_bill_lists.csv
WRITE_GML         = True
WRITE_PNG         = True


# -- Global first-quarter lookup ----------------------------------------------

def compute_total_spend(df):
    """Sum lobbying spend per firm from bill-linked reports; deduplicates by uniq_id."""
    per_report = df.drop_duplicates(subset=["uniq_id"])[["fortune_name", "amount"]]
    return per_report.groupby("fortune_name")["amount"].sum().to_dict()


def build_global_first_quarters(df):
    """Return {(firm, bill): min_quarter} — first quarter each firm lobbied each bill across the full congress."""
    return (
        df.groupby(["fortune_name", "bill_number"])["quarter"]
        .min()
        .to_dict()
    )


# -- Pairwise scoring ---------------------------------------------------------

def score_pair(firm_a, firm_b, shared_bills, bill_first):
    """Tally first-mover wins for a firm pair over shared top-30 bills; returns dict with a_firsts, b_firsts, tie_count, shared_bills."""
    a_firsts = b_firsts = ties = 0
    for bill in shared_bills:
        qa = bill_first.get((firm_a, bill))
        qb = bill_first.get((firm_b, bill))
        if qa is None or qb is None:
            continue   # defensive: bill missing from filing records
        if qa < qb:
            a_firsts += 1
        elif qb < qa:
            b_firsts += 1
        else:
            ties += 1
    return {
        "a_firsts":    a_firsts,
        "b_firsts":    b_firsts,
        "tie_count":   ties,
        "shared_bills": len(shared_bills),
    }


# -- Edge construction --------------------------------------------------------

def build_edges(ranked, bill_first, p=RBO_P):
    """Compute bidirectional RBO-weighted edges for all firm pairs sharing ≥1 top-30 bill.

    Each pair always produces two directed edges; weights partition RBO by first-mover contribution.
    """
    firms   = sorted(ranked.keys())
    records = []

    for firm_a, firm_b in itertools.combinations(firms, 2):
        list_a = ranked[firm_a]
        list_b = ranked[firm_b]

        rbo_w = rbo_score(list_a, list_b, p=p)
        if rbo_w == 0.0:
            continue   # no shared top-30 bills; skip pair

        shared = set(list_a) & set(list_b)
        sc     = score_pair(firm_a, firm_b, shared, bill_first)
        a_f, b_f = sc["a_firsts"], sc["b_firsts"]
        ties     = sc["tie_count"]
        n_shared = sc["shared_bills"]
        net_t    = a_f - b_f

        # Partition RBO proportionally; ties split 50/50 between both directions
        w_a_to_b = round(((a_f + ties / 2) / n_shared) * rbo_w, 6)
        w_b_to_a = round(((b_f + ties / 2) / n_shared) * rbo_w, 6)

        # Edge A→B
        records.append({
            "source":        firm_a,
            "target":        firm_b,
            "weight":        w_a_to_b,
            "rbo":           round(rbo_w, 6),
            "source_firsts": a_f,
            "target_firsts": b_f,
            "tie_count":     ties,
            "shared_bills":  n_shared,
            "net_temporal":  net_t,
        })
        # Edge B→A
        records.append({
            "source":        firm_b,
            "target":        firm_a,
            "weight":        w_b_to_a,
            "rbo":           round(rbo_w, 6),
            "source_firsts": b_f,
            "target_firsts": a_f,
            "tie_count":     ties,
            "shared_bills":  n_shared,
            "net_temporal":  -net_t,
        })

    cols = [
        "source", "target", "weight", "rbo", "source_firsts", "target_firsts",
        "tie_count", "shared_bills", "net_temporal",
    ]
    return pd.DataFrame(records) if records else pd.DataFrame(columns=cols)


# -- Graph construction -------------------------------------------------------

def build_graph(edges_df, ranked_firms=None, spend_map=None):
    """Build DiGraph from edge records, adding node attrs: net_influence, total_firsts/losses, out/in/net_strength, color, label.

    Isolated firms added as nodes when ranked_firms is provided.
    """
    G = nx.DiGraph()

    # Add all ranked firms upfront so isolated firms appear as nodes
    if ranked_firms:
        for firm in ranked_firms:
            G.add_node(firm)

    for _, row in edges_df.iterrows():
        G.add_edge(
            row["source"], row["target"],
            weight=float(row["weight"]),
            rbo=float(row["rbo"]),
            source_firsts=int(row["source_firsts"]),
            target_firsts=int(row["target_firsts"]),
            tie_count=int(row["tie_count"]),
            shared_bills=int(row["shared_bills"]),
            net_temporal=int(row["net_temporal"]),
        )

    for node in G.nodes():
        out_str = sum((d["weight"] for _, _, d in G.out_edges(node, data=True)), 0.0)
        in_str  = sum((d["weight"] for _, _, d in G.in_edges(node, data=True)), 0.0)

        # net_strength = Σ_j [RBO(i,j) × net_temporal(i,j)]; summed over out-edges only
        net_str = sum(
            (d["rbo"] * d["net_temporal"] for _, _, d in G.out_edges(node, data=True)),
            0.0,
        )

        # First-mover counts: out-edges only (each pair appears once as source, once as target;
        # using only out-edges avoids double-counting with in-edges)
        total_firsts = sum(d["source_firsts"] for _, _, d in G.out_edges(node, data=True))
        total_losses = sum(d["target_firsts"] for _, _, d in G.out_edges(node, data=True))
        net          = total_firsts - total_losses

        G.nodes[node]["out_strength"]  = round(out_str, 4)
        G.nodes[node]["in_strength"]   = round(in_str,  4)
        G.nodes[node]["net_strength"]  = round(net_str, 4)
        G.nodes[node]["total_firsts"]  = int(total_firsts)
        G.nodes[node]["total_losses"]  = int(total_losses)
        G.nodes[node]["net_influence"] = int(net)
        G.nodes[node]["label"]         = str(node)
        G.nodes[node]["total_spend"]   = round(float(spend_map.get(node, -1.0)), 2) if spend_map else -1.0
        # Color by net_strength: green=agenda-setter, red=follower, gray=neutral/isolated
        G.nodes[node]["color"] = (
            "#2ECC71" if net_str > 0 else ("#E74C3C" if net_str < 0 else "#95A5A6")
        )

    return G


# -- I/O helpers --------------------------------------------------------------

def export_ranked_lists(ranked, df_agg, output_path):
    """Export per-firm top-30 ranked bill lists as long-format CSV with columns: company, rank, bill_number, total_amount, budget_fraction."""
    lookup = (
        df_agg.set_index(["fortune_name", "bill_number"])
        [["amount_allocated", "frac"]]
        .to_dict("index")
    )
    rows = []
    for firm, bills in sorted(ranked.items()):
        for rank, bill in enumerate(bills, start=1):
            key    = (firm, bill)
            amount = lookup[key]["amount_allocated"] if key in lookup else 0.0
            frac   = lookup[key]["frac"]             if key in lookup else 0.0
            rows.append({
                "company":         firm,
                "rank":            rank,
                "bill_number":     bill,
                "total_amount":    round(amount, 2),
                "budget_fraction": round(frac, 6),
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print(f"  Ranked lists CSV -> {Path(output_path).name}  "
          f"({len(ranked)} firms, {len(df_out)} rows)")
    return df_out


def write_gml(G, path):
    """Write directed DiGraph as GML for Gephi; clamps zero-weight edges to 1e-5.

    Gephi silently drops edges with weight=0. Pure-follower edges (one firm leads
    on every shared bill, no ties) have weight=0 by construction. We clamp only
    those edges on a copy so the in-memory graph and CSVs are unchanged.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    G_out = G.copy()
    clamped = 0
    for u, v, d in G_out.edges(data=True):
        if d.get("weight", 1) == 0:
            d["weight"] = 1e-5
            clamped += 1
    nx.write_gml(G_out, path)
    print(f"  GML written ({G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges, {clamped} zero-weight edges clamped to 1e-5)"
          f" -> {path}")


def print_stats(edges_df, G):
    """Print edge coverage, node tallies, and top agenda-setters / followers."""
    n_total    = len(edges_df)
    n_pairs    = n_total // 2
    n_decisive = (edges_df["net_temporal"] > 0).sum()   # one per decisive pair
    n_balanced = n_pairs - n_decisive
    print(f"\n  Total edges in DiGraph: {n_total:,}  "
          f"({n_pairs:,} unique pairs: decisive={n_decisive:,}  balanced={n_balanced:,})")
    print(f"  Mean RBO:               {edges_df['rbo'].mean():.4f}")
    decisive = edges_df[edges_df["net_temporal"] > 0]
    if len(decisive) > 0:
        print(f"  Mean net_temporal (decisive pairs): {decisive['net_temporal'].mean():.2f}")
        print(f"  Mean shared_bills (decisive pairs): {decisive['shared_bills'].mean():.1f}")

    nodes_sorted = sorted(
        G.nodes(), key=lambda n: G.nodes[n]["net_strength"], reverse=True
    )
    print(f"\n  Top 8 agenda-setters (net_strength):")
    print(f"    {'Firm':<42} {'net_str':>10}  {'net_inf':>8}  {'Firsts':>7}  {'Losses':>7}")
    for n in nodes_sorted[:8]:
        a = G.nodes[n]
        print(f"    {n:<42} {a['net_strength']:>10.4f}  {a['net_influence']:>+8}  "
              f"{a['total_firsts']:>7}  {a['total_losses']:>7}")

    print(f"\n  Top 5 followers (lowest net_strength):")
    for n in nodes_sorted[-5:]:
        a = G.nodes[n]
        print(f"    {n:<42} {a['net_strength']:>10.4f}  {a['net_influence']:>+8}  "
              f"{a['total_firsts']:>7}  {a['total_losses']:>7}")


# -- Main ---------------------------------------------------------------------

def main():
    # Load and assign quarters
    df_raw = load_bills_data(DATA_DIR / "opensecrets_lda_reports.csv")
    df_raw = assign_quarters(df_raw)
    print(f"Loaded {len(df_raw):,} rows  |  "
          f"{df_raw['fortune_name'].nunique()} firms  |  "
          f"{df_raw['bill_number'].nunique()} bills")

    # Step 1: aggregate all quarters → fracs → prevalence filter
    df_agg = aggregate_per_firm_bill(df_raw)
    df_agg = compute_zero_budget_fracs(df_agg)
    if MAX_BILL_DF is not None:
        df_agg = filter_bills_by_prevalence(df_agg, MAX_BILL_DF, unit_col="bill_number")
    print(f"  After prevalence filter: {df_agg['bill_number'].nunique():,} bills")

    # Step 2: build top-30 ranked lists (by total spend across full congress)
    ranked = build_ranked_lists(df_agg, top_bills=TOP_BILLS)
    print(f"  Firms with ranked lists: {len(ranked):,}")

    if WRITE_RANKED_CSV:
        export_ranked_lists(ranked, df_agg, DATA_DIR / "ranked_bill_lists.csv")

    # Step 3: global first-quarter lookup (raw data; all quarters, all bills)
    bill_first = build_global_first_quarters(df_raw)
    print(f"  (firm, bill) first-quarter entries: {len(bill_first):,}")

    # Step 4: build edge list (RBO weight + temporal direction)
    print(f"\nBuilding edges (RBO p={RBO_P}, top_bills={TOP_BILLS}, "
          f"MAX_BILL_DF={MAX_BILL_DF})...")
    edges_df = build_edges(ranked, bill_first, p=RBO_P)
    print(f"  Done.")

    # Step 5: build graph and print stats
    spend_map = compute_total_spend(df_raw)
    G = build_graph(edges_df, ranked_firms=set(ranked.keys()), spend_map=spend_map)
    print_stats(edges_df, G)

    # Step 6: outputs
    if WRITE_CSV:
        csv_path = DATA_DIR / "rbo_directed_influence.csv"
        edges_df.to_csv(csv_path, index=False)
        print(f"\n  Edges CSV -> {csv_path.name}")

    if WRITE_GML:
        write_gml(G, str(VIZ_GML_DIR / "rbo_directed_influence.gml"))

    if WRITE_PNG:
        plot_directed_circular(
            G,
            title=f"Top {TOP_K} Congress-wide RBO Directed Influence Network",
            path=str(VIZ_PNG_DIR / "rbo_directed_influence.png"),
        )


if __name__ == "__main__":
    main()
