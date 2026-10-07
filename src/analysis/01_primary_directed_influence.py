"""
Primary directed influence network analysis — 116th Congress.

Answers:
  1. Who are the top global agenda-setters by net_strength?
  2. How does that list compare to top spenders and broadest bill portfolios?
  3. Who are the top within-community agenda-setters per sector?
  4. What are illustrative influencer→follower case studies?

Outputs (all to outputs/analysis/):
  01_global_agenda_setters.csv
  01_global_followers.csv
  01_within_community_agenda_setters.csv
  01_case_studies.csv
  01_primary_directed_influence.txt
  01_ns_vs_spend_scatter.png
  01_top30_bar.png
  01_top20_bar.png        — top-20 agenda-setters; black outline = low wc_net_strength
  01_bottom20_bar.png     — top-20 agenda-followers (most negative net_strength)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

CONGRESS    = 116
SECTOR_COLORS = {
    "Finance/Insurance":      "#4C72B0",
    "Tech/Telecom":           "#DD8452",
    "Energy/Utilities":       "#C44E52",
    "Defense/Industrial":     "#55A868",
    "Health/Pharma":          "#8172B2",
    "Consumer/Manufacturing": "#CCB974",
}
TOP_N       = 30   # global agenda-setter list length (CSV)
TOP_N_COMM  = 10   # within-community agenda-setter list length
TOP_CASES   = 20   # case study pairs to surface
PAPER_N     = 20   # figure size for paper bar charts (top/bottom 20)

COMMUNITY_LABELS = {
    0: "Finance/Insurance",
    1: "Tech/Telecom",
    2: "Energy/Utilities",
    3: "Defense/Industrial",
    4: "Health/Pharma",
    5: "Consumer/Manufacturing",
}

OUT_DIR = ROOT / "outputs" / "analysis"

# ---------------------------------------------------------------------------
# Tee helper
# ---------------------------------------------------------------------------

class _Tee:
    """Write to stdout and file simultaneously."""
    def __init__(self, *streams): self.streams = streams
    def write(self, t):
        for s in self.streams: s.write(t)
    def flush(self):
        for s in self.streams: s.flush()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_node_attrs():
    return pd.read_csv(DATA_DIR / f"congress/{CONGRESS}/node_attributes.csv")

def load_edges():
    return pd.read_csv(DATA_DIR / f"congress/{CONGRESS}/rbo_directed_influence.csv")

def load_reports():
    return pd.read_csv(DATA_DIR / f"congress/{CONGRESS}/opensecrets_lda_reports.csv")

def load_communities():
    df = pd.read_csv(DATA_DIR / "communities/communities_affiliation.csv")
    return dict(zip(df["fortune_name"], df["community_aff"]))

# ---------------------------------------------------------------------------
# Spend and portfolio metrics
# ---------------------------------------------------------------------------

def build_spend_portfolio(reports):
    """Total spend and unique bill count per firm."""
    uniq = reports.drop_duplicates(subset=["uniq_id", "fortune_name"])
    spend = uniq.groupby("fortune_name")["amount_allocated"].sum().rename("total_spend")
    bills = reports.groupby("fortune_name")["bill_number"].nunique().rename("num_bills")
    return pd.concat([spend, bills], axis=1).reset_index().rename(
        columns={"fortune_name": "firm"})

# ---------------------------------------------------------------------------
# Within-community net_strength
# ---------------------------------------------------------------------------

def compute_wc_net_strength(edges, partition):
    """Σ_j∈same_community [rbo(i,j) × net_temporal(i,j)] for each firm."""
    edges = edges.copy()
    edges["src_comm"] = edges["source"].map(partition)
    edges["tgt_comm"] = edges["target"].map(partition)
    intra = edges[
        edges["src_comm"].notna() &
        edges["tgt_comm"].notna() &
        (edges["src_comm"] == edges["tgt_comm"])
    ]
    result = (
        intra.groupby("source")
        .apply(lambda g: float((g["rbo"] * g["net_temporal"]).sum()),
               include_groups=False)
        .rename("wc_net_strength")
        .rename_axis("firm")
        .reset_index()
    )
    return result

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_figures(nodes, top_global, out_dir):
    """Scatter (net_strength vs spend) and top-30 horizontal bar chart."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    # -- Figure 1: net_strength vs total_spend scatter ---------------------
    plot_df = nodes.dropna(subset=["net_strength", "total_spend"]).copy()
    plot_df["spend_M"] = plot_df["total_spend"] / 1e6
    plot_df["color"]   = plot_df["sector"].map(SECTOR_COLORS).fillna("#AAAAAA")

    top10_firms = set(top_global.head(10)["firm"])

    fig, ax = plt.subplots(figsize=(8, 6))
    for sector, color in SECTOR_COLORS.items():
        sub = plot_df[plot_df["sector"] == sector]
        ax.scatter(sub["spend_M"], sub["net_strength"],
                   c=color, label=sector, alpha=0.65, s=36, linewidths=0.4,
                   edgecolors="white", zorder=3)
    other = plot_df[plot_df["sector"].isna()]
    if len(other):
        ax.scatter(other["spend_M"], other["net_strength"],
                   c="#AAAAAA", label="Other/Unknown", alpha=0.5, s=28, zorder=2)

    # Label top-10 agenda-setters
    for _, row in plot_df[plot_df["firm"].isin(top10_firms)].iterrows():
        ax.annotate(row["firm"].title()[:22],
                    xy=(row["spend_M"], row["net_strength"]),
                    xytext=(6, 2), textcoords="offset points",
                    fontsize=7, color="#222222",
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.5))

    ax.set_xscale("log")
    ax.set_xlabel("Total Lobbying Spend ($M, log scale)", fontsize=11)
    ax.set_ylabel("Net Strength (agenda-setting influence)", fontsize=11)
    ax.set_title("Agenda-Setting vs. Lobbying Spend — 116th Congress\n"
                 r"$\rho$ = −0.14, $p$ = 0.020 (Spearman)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "01_ns_vs_spend_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -- Figure 2: top-30 horizontal bar chart, colored by sector ----------
    bar_df  = top_global.reset_index(drop=False).copy()
    bar_df  = bar_df.iloc[::-1]                          # highest at top
    colors  = bar_df["sector"].map(SECTOR_COLORS).fillna("#AAAAAA").tolist()
    y_pos   = range(len(bar_df))

    fig, ax = plt.subplots(figsize=(8, 9))
    bars = ax.barh(list(y_pos), bar_df["net_strength"].tolist(),
                   color=colors, edgecolor="white", linewidth=0.5, height=0.72)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(bar_df["firm"].str.title().tolist(), fontsize=8.5)
    ax.set_xlabel("Net Strength", fontsize=11)
    ax.set_title("Top-30 Global Agenda-Setters by Net Strength\n116th Congress",
                 fontsize=12)

    # Sector legend patches
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=s)
                      for s, c in SECTOR_COLORS.items()]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right", framealpha=0.85)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "01_top30_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Paper figures (top-20 / bottom-20 bar charts)
# ---------------------------------------------------------------------------

def plot_paper_figures(nodes, out_dir):
    """Generate 01_top20_bar.png (agenda-setters) and 01_bottom20_bar.png (followers)."""
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    # -- Top-20 agenda-setters; highlight firms with low wc_net_strength ----
    top20 = nodes.nlargest(PAPER_N, "net_strength").copy()

    # Flag firms in bottom 50% of wc_net_strength within their own community
    valid = nodes.dropna(subset=["community", "wc_net_strength"])
    low_wc = set()
    for cid in valid["community"].unique():
        comm_vals = valid[valid["community"] == cid]
        median_wc = comm_vals["wc_net_strength"].median()
        low_wc |= set(comm_vals[comm_vals["wc_net_strength"] < median_wc]["firm"])

    top20["highlight"] = top20["firm"].isin(low_wc)
    top20 = top20.iloc[::-1]   # highest net_strength at top when plotted
    colors  = top20["sector"].map(SECTOR_COLORS).fillna("#AAAAAA").tolist()
    edges_c = ["black" if h else "white" for h in top20["highlight"]]
    lws     = [1.6 if h else 0.5 for h in top20["highlight"]]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(top20)), top20["net_strength"].tolist(),
            color=colors, edgecolor=edges_c, linewidth=lws, height=0.72)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["firm"].str.title().tolist(), fontsize=8.5)
    ax.set_xlabel("Net Strength", fontsize=11)
    ax.set_title("Top-20 Global Agenda-Setters by Net Strength\n116th Congress\n"
                 "(black outline = low within-community wc_net_strength)", fontsize=11)
    legend_handles = [Patch(facecolor=c, label=s) for s, c in SECTOR_COLORS.items()]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right", framealpha=0.85)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "01_top20_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -- Bottom-20 agenda-followers ----------------------------------------
    bottom20 = nodes.nsmallest(PAPER_N, "net_strength").copy()
    bottom20 = bottom20.iloc[::-1]   # least-negative at top when plotted
    colors_b = bottom20["sector"].map(SECTOR_COLORS).fillna("#AAAAAA").tolist()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(bottom20)), bottom20["net_strength"].tolist(),
            color=colors_b, edgecolor="white", linewidth=0.5, height=0.72)
    ax.set_yticks(range(len(bottom20)))
    ax.set_yticklabels(bottom20["firm"].str.title().tolist(), fontsize=8.5)
    ax.set_xlabel("Net Strength", fontsize=11)
    ax.set_title("Top-20 Global Agenda-Followers by Net Strength\n116th Congress", fontsize=11)
    legend_handles = [Patch(facecolor=c, label=s) for s, c in SECTOR_COLORS.items()]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower left", framealpha=0.85)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "01_bottom20_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save followers CSV
    bottom20_csv = nodes.nsmallest(PAPER_N, "net_strength")[
        ["firm", "net_strength", "net_influence", "total_spend", "num_bills", "sector"]
    ].reset_index(drop=True)
    bottom20_csv.index += 1
    bottom20_csv.to_csv(out_dir / "01_global_followers.csv", index_label="ns_rank")


# ---------------------------------------------------------------------------
# Case study builder
# ---------------------------------------------------------------------------

def build_case_studies(edges, nodes, partition, n=TOP_CASES):
    """
    Surface the most decisive influencer→follower pairs.
    A decisive pair has the largest positive net_temporal weighted by rbo.
    Returns a DataFrame with context columns attached.
    """
    # Canonical decisive pairs: source leads target (net_temporal > 0); one row per pair
    decisive = edges[
        (edges["source"] < edges["target"]) & (edges["net_temporal"] > 0)
    ].copy()
    decisive["influence_score"] = decisive["rbo"] * decisive["net_temporal"]
    decisive = decisive.nlargest(n, "influence_score")

    # Attach community labels
    decisive["sector_influencer"] = decisive["source"].map(partition).map(COMMUNITY_LABELS)
    decisive["sector_follower"]   = decisive["target"].map(partition).map(COMMUNITY_LABELS)

    # Attach global net_strength of influencer
    ns_map = nodes.set_index("firm")["net_strength"].to_dict()
    decisive["influencer_net_strength"] = decisive["source"].map(ns_map)
    decisive["follower_net_strength"]   = decisive["target"].map(ns_map)

    return decisive[[
        "source", "target", "rbo", "net_temporal", "influence_score",
        "source_firsts", "target_firsts", "shared_bills",
        "sector_influencer", "sector_follower",
        "influencer_net_strength", "follower_net_strength",
    ]].rename(columns={"source": "influencer", "target": "follower"})

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(OUT_DIR / "01_primary_directed_influence.txt", "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    SEP = "=" * 70
    print(SEP)
    print("ANALYSIS 01: PRIMARY DIRECTED INFLUENCE NETWORK — 116th CONGRESS")
    print(SEP)

    nodes     = load_node_attrs()
    edges     = load_edges()
    reports   = load_reports()
    partition = load_communities()

    print(f"\nFirms in network:   {len(nodes)}")
    print(f"Directed edge rows: {len(edges)}")
    print(f"LDA report rows:    {len(reports)}")

    # -- Spend / portfolio -----------------------------------------------
    sb = build_spend_portfolio(reports)
    nodes = nodes.merge(sb, on="firm", how="left")
    nodes["spend_rank"]  = nodes["total_spend"].rank(ascending=False, method="min")
    nodes["bills_rank"]  = nodes["num_bills"].rank(ascending=False, method="min")
    nodes["ns_rank"]     = nodes["net_strength"].rank(ascending=False, method="min")

    # -- Community membership --------------------------------------------
    nodes["community"]   = nodes["firm"].map(partition)
    nodes["sector"]      = nodes["community"].map(COMMUNITY_LABELS)

    # -- Global top-N by net_strength ------------------------------------
    top_global = nodes.nlargest(TOP_N, "net_strength")[[
        "firm", "net_strength", "net_influence", "total_spend",
        "num_bills", "ns_rank", "spend_rank", "bills_rank", "sector"
    ]].reset_index(drop=True)
    top_global.index += 1

    print(f"\n{'─'*70}")
    print(f"TOP {TOP_N} GLOBAL AGENDA-SETTERS BY NET_STRENGTH — 116th Congress")
    print(f"{'─'*70}")
    print(f"\n{'Rank':<5} {'Firm':<42} {'ns':>7} {'ni':>5} {'$M':>7} {'#bills':>7} "
          f"{'$rank':>7} {'brank':>7} {'Sector'}")
    print(f"{'─'*70}")
    for rank, row in top_global.iterrows():
        spend_m = row["total_spend"] / 1e6 if pd.notna(row["total_spend"]) else float("nan")
        print(f"{rank:<5} {row['firm']:<42} {row['net_strength']:>7.3f} "
              f"{int(row['net_influence']):>5} {spend_m:>7.2f} {int(row['num_bills'] or 0):>7} "
              f"{int(row['spend_rank']):>7} {int(row['bills_rank']):>7}  {row['sector'] or '—'}")

    top_global.to_csv(OUT_DIR / "01_global_agenda_setters.csv", index_label="ns_rank")

    # -- Rank correlation: net_strength vs spend vs bills ----------------
    from scipy.stats import spearmanr
    common = nodes.dropna(subset=["net_strength", "total_spend", "num_bills"])
    rho_spend, p_spend = spearmanr(common["net_strength"], common["total_spend"])
    rho_bills, p_bills = spearmanr(common["net_strength"], common["num_bills"])
    print(f"\n  Spearman ρ (net_strength vs total_spend):  {rho_spend:.4f}  p={p_spend:.4e}")
    print(f"  Spearman ρ (net_strength vs num_bills):    {rho_bills:.4f}  p={p_bills:.4e}")
    print(f"  → Interpretation: agenda-setting is {'moderately' if abs(rho_spend) > 0.4 else 'weakly'} "
          f"correlated with spend, {'moderately' if abs(rho_bills) > 0.4 else 'weakly'} "
          f"with portfolio breadth.")

    # Firms in top-{TOP_N} by net_strength but NOT in top-{TOP_N} by spend
    top_ns_set    = set(top_global["firm"])
    top_spend_set = set(nodes.nlargest(TOP_N, "total_spend")["firm"])
    top_bills_set = set(nodes.nlargest(TOP_N, "num_bills")["firm"])
    print(f"\n  Top-{TOP_N} overlap:")
    print(f"    net_strength ∩ top_spend:  {len(top_ns_set & top_spend_set)}/{TOP_N}")
    print(f"    net_strength ∩ top_bills:  {len(top_ns_set & top_bills_set)}/{TOP_N}")
    print(f"    In top_ns but NOT top_spend: "
          f"{sorted(top_ns_set - top_spend_set)[:8]}")
    print(f"    In top_ns but NOT top_bills: "
          f"{sorted(top_ns_set - top_bills_set)[:8]}")

    # -- Within-community agenda-setters ---------------------------------
    wc_ns = compute_wc_net_strength(edges, partition)
    nodes = nodes.merge(wc_ns, on="firm", how="left")

    print(f"\n{'─'*70}")
    print(f"WITHIN-COMMUNITY TOP-{TOP_N_COMM} AGENDA-SETTERS BY SECTOR — 116th")
    print(f"{'─'*70}")

    wc_rows = []
    for cid, label in COMMUNITY_LABELS.items():
        members = nodes[nodes["community"] == cid].copy()
        top_wc  = members.nlargest(TOP_N_COMM, "wc_net_strength")
        print(f"\n  [{label}]  (n={len(members)} firms)")
        print(f"  {'Rank':<5} {'Firm':<42} {'wc_ns':>8} {'global_ns':>10}")
        for i, (_, row) in enumerate(top_wc.iterrows(), 1):
            print(f"  {i:<5} {row['firm']:<42} {row['wc_net_strength']:>8.3f} "
                  f"{row['net_strength']:>10.3f}")
            wc_rows.append({
                "sector": label, "rank": i, "firm": row["firm"],
                "wc_net_strength": row["wc_net_strength"],
                "net_strength":    row["net_strength"],
            })

    pd.DataFrame(wc_rows).to_csv(OUT_DIR / "01_within_community_agenda_setters.csv", index=False)

    # -- Case studies ----------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"TOP {TOP_CASES} INFLUENCER→FOLLOWER PAIRS (by rbo × net_temporal)")
    print(f"{'─'*70}")
    cases = build_case_studies(edges, nodes, partition)
    cases.to_csv(OUT_DIR / "01_case_studies.csv", index=False)

    print(f"\n  {'Influencer':<38} {'Follower':<38} {'score':>7} "
          f"{'shared':>7} {'sec_I':>22} {'sec_F':>22}")
    print(f"  {'─'*140}")
    for _, row in cases.iterrows():
        print(f"  {row['influencer']:<38} {row['follower']:<38} "
              f"{row['influence_score']:>7.4f} {int(row['shared_bills']):>7} "
              f"{str(row['sector_influencer'] or '—'):>22} "
              f"{str(row['sector_follower'] or '—'):>22}")

    # -- Figures ---------------------------------------------------------
    plot_figures(nodes, top_global, OUT_DIR)
    plot_paper_figures(nodes, OUT_DIR)

    print(f"\n  Outputs:")
    print(f"    01_global_agenda_setters.csv")
    print(f"    01_global_followers.csv")
    print(f"    01_within_community_agenda_setters.csv")
    print(f"    01_case_studies.csv")
    print(f"    01_primary_directed_influence.txt")
    print(f"    01_ns_vs_spend_scatter.png")
    print(f"    01_top30_bar.png")
    print(f"    01_top20_bar.png")
    print(f"    01_bottom20_bar.png")
    print(f"\n{SEP}")
    print("Analysis complete.")

    log_f.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
