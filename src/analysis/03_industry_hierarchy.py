"""
Industry-sector influencer hierarchy — 111th–117th Congress.

Answers:
  1. Who are the top within-community agenda-setters per sector, per congress?
  2. Have those rankings stayed stable across sessions?
  3. Which firms are persistent sectoral leaders (≥4 of 7 congresses)?

Primary metric: wc_net_strength = Σ_j∈same_comm [rbo(i,j) × net_temporal(i,j)].
Community partition: fixed 116th-Congress Leiden affiliation labels.

Outputs (outputs/analysis/):
  03_within_community_leaderboards.csv   — firm × congress × sector scores
  03_rank_stability.csv                  — Kendall's W + adjacent Spearman per sector
  03_persistent_leaders.csv             — firms in top-5 in ≥4 of 7 congresses
  03_industry_hierarchy.txt
  03_kendalls_w_bar.png
  03_energy_rank_flow.png
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, chi2 as chi2_dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

CONGRESSES = [111, 112, 113, 114, 115, 116, 117]
TOP_N      = 5     # leaderboard size per sector per congress
MIN_SESSIONS_W = 3 # min sessions for Kendall's W

COMMUNITY_LABELS = {
    0: "Finance/Insurance",
    1: "Tech/Telecom",
    2: "Energy/Utilities",
    3: "Defense/Industrial",
    4: "Health/Pharma",
    5: "Consumer/Manufacturing",
}
SECTOR_COLORS = {
    "Finance/Insurance":      "#4C72B0",
    "Tech/Telecom":           "#DD8452",
    "Energy/Utilities":       "#C44E52",
    "Defense/Industrial":     "#55A868",
    "Health/Pharma":          "#8172B2",
    "Consumer/Manufacturing": "#CCB974",
}

OUT_DIR = ROOT / "outputs" / "analysis"

# ---------------------------------------------------------------------------
# Tee helper
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, t):
        for s in self.streams: s.write(t)
    def flush(self):
        for s in self.streams: s.flush()

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_kendalls_w(stability_rows, out_dir):
    """Bar chart of Kendall's W by sector with p-value annotations."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    sectors = [r["sector"] for r in stability_rows]
    w_vals  = [r["kendalls_W"] for r in stability_rows]
    p_vals  = [r["kendalls_W_p"] for r in stability_rows]
    colors  = [SECTOR_COLORS.get(s, "#999999") for s in sectors]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(sectors, w_vals, color=colors, edgecolor="white",
                  linewidth=0.6, width=0.6)

    # p-value annotation above each bar
    for bar, p in zip(bars, p_vals):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height() + 0.008
        ax.text(x, y, stars, ha="center", va="bottom", fontsize=11,
                color="#333333", fontweight="bold")

    ax.axhline(0, color="#AAAAAA", linewidth=0.8)
    ax.set_ylim(0, max(w_vals) * 1.25)
    ax.set_ylabel("Kendall's W (rank concordance)", fontsize=11)
    ax.set_title("Within-Sector Rank Stability — 111th–117th Congress\n"
                 "(*** p<0.001, ** p<0.01, * p<0.05, ns = not significant)", fontsize=11)
    ax.set_xticklabels(sectors, rotation=18, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "03_kendalls_w_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_energy_rank_flow(leaderboard_rows, out_dir, top_n=5):
    """Bump chart of top-5 Energy/Utilities firms across 7 congresses."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    energy = [r for r in leaderboard_rows
              if r["sector"] == "Energy/Utilities" and r["rank"] <= top_n]
    energy_df = pd.DataFrame(energy)

    # Collect all firms that ever appear in top-5 for Energy
    firms = sorted(energy_df["firm"].unique())
    # Assign each firm a consistent color
    palette = plt.cm.tab10.colors
    firm_colors = {f: palette[i % len(palette)] for i, f in enumerate(firms)}

    fig, ax = plt.subplots(figsize=(9, 5))
    x_vals = list(range(len(CONGRESSES)))

    for firm in firms:
        sub = energy_df[energy_df["firm"] == firm].sort_values("congress")
        xs  = [CONGRESSES.index(c) for c in sub["congress"]]
        ys  = [top_n + 1 - r for r in sub["rank"]]  # invert: rank 1 → top
        color = firm_colors[firm]
        ax.plot(xs, ys, "-o", color=color, linewidth=2.0, markersize=7,
                label=firm.title()[:28], zorder=3)
        # Label at last known position
        if xs:
            ax.annotate(firm.title()[:20],
                        xy=(xs[-1], ys[-1]),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=7.5, color=color, va="center")

    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(c) for c in CONGRESSES], fontsize=10)
    ax.set_yticks(range(1, top_n + 1))
    ax.set_yticklabels([f"#{top_n + 1 - i}" for i in range(1, top_n + 1)], fontsize=9)
    ax.set_xlabel("Congress", fontsize=11)
    ax.set_ylabel("Rank (within Energy/Utilities)", fontsize=11)
    ax.set_title("Energy/Utilities Within-Sector Rank Flow — 111th–117th Congress\n"
                 "(top-5 by wc_net_strength; Kendall's W = 0.446, p < 0.001)", fontsize=11)
    ax.invert_yaxis()   # rank 1 at top
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend outside right
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1),
              borderaxespad=0, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_dir / "03_energy_rank_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# wc_net_strength computation
# ---------------------------------------------------------------------------

def compute_wc_net_strength(congress, comm_map):
    """Σ_j∈same_community [rbo(i,j) × net_temporal(i,j)], out-edges only."""
    path = DATA_DIR / f"congress/{congress}/rbo_directed_influence.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df["src_comm"] = df["source"].map(comm_map)
    df["tgt_comm"] = df["target"].map(comm_map)
    intra = df[
        df["src_comm"].notna() & df["tgt_comm"].notna() &
        (df["src_comm"] == df["tgt_comm"])
    ]
    if "rbo" not in df.columns:
        return {}
    result = (
        intra.groupby("source")
        .apply(lambda g: float((g["rbo"] * g["net_temporal"]).sum()),
               include_groups=False)
        .rename_axis("firm")
        .to_dict()
    )
    return result

# ---------------------------------------------------------------------------
# Kendall's W
# ---------------------------------------------------------------------------

def kendalls_w(rank_matrix):
    """Kendall's W from an (n_raters × n_subjects) matrix; NaN columns excluded."""
    m_full = rank_matrix.copy()
    m_full = m_full.loc[:, m_full.notna().sum(axis=0) >= MIN_SESSIONS_W]
    if m_full.shape[1] < 2:
        return np.nan, np.nan, np.nan
    ranked = m_full.apply(lambda row: row.rank(method="average", na_option="keep"), axis=1)
    n = ranked.shape[1]
    k = ranked.shape[0]
    R = ranked.sum(axis=0, skipna=True)
    R_bar = R.mean()
    S = ((R - R_bar) ** 2).sum()
    W = float(np.clip(12 * S / (k ** 2 * (n ** 3 - n)), 0, 1))
    chi2_val = k * (n - 1) * W
    p = float(1 - chi2_dist.cdf(chi2_val, df=n - 1))
    return round(W, 4), round(chi2_val, 2), round(p, 5)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(OUT_DIR / "03_industry_hierarchy.txt", "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    SEP = "=" * 70
    print(SEP)
    print("ANALYSIS 03: INDUSTRY-SECTOR INFLUENCER HIERARCHY (111th–117th)")
    print(SEP)

    comm_df  = pd.read_csv(DATA_DIR / "communities/communities_affiliation.csv")
    comm_map = dict(zip(comm_df["fortune_name"], comm_df["community_aff"]))

    # Build long-form table: firm × congress × wc_net_strength
    print("\n[1/3] Computing wc_net_strength for all congresses ...")
    records = []
    for c in CONGRESSES:
        wc = compute_wc_net_strength(c, comm_map)
        for firm, val in wc.items():
            records.append({"firm": firm, "congress": c, "wc_net_strength": val})
        print(f"  {c}th: {len(wc)} firms")

    long_df = pd.DataFrame(records)
    long_df["community"] = long_df["firm"].map(comm_map)
    long_df = long_df.dropna(subset=["community"])
    long_df["community"] = long_df["community"].astype(int)
    long_df["sector"]    = long_df["community"].map(COMMUNITY_LABELS)

    # Wide form: firm × congress
    wide_df = long_df.pivot(index="firm", columns="congress", values="wc_net_strength")
    wide_df["community"] = wide_df.index.map(comm_map).astype(float).astype("Int64")

    # -- Per-community leaderboards and stability -------------------------
    print("\n[2/3] Per-community leaderboards and rank stability ...")

    leaderboard_rows = []
    stability_rows   = []
    persistent_rows  = []

    for c_id, c_label in COMMUNITY_LABELS.items():
        members     = comm_df[comm_df["community_aff"] == c_id]["fortune_name"].tolist()
        comm_long   = long_df[long_df["community"] == c_id]
        comm_wide   = wide_df[wide_df["community"] == c_id].drop(columns="community")
        stable_set  = comm_wide.dropna()
        n_stable    = len(stable_set)

        print(f"\n{'='*68}")
        print(f"  {c_label}  (members={len(members)}, stable across all 7 congresses={n_stable})")
        print(f"{'='*68}")

        # Collect top-N per congress
        tops = {}
        for congress in CONGRESSES:
            sub = comm_long[comm_long["congress"] == congress].nlargest(TOP_N, "wc_net_strength")
            tops[congress] = list(zip(sub["firm"].tolist(), sub["wc_net_strength"].tolist()))

        # Print leaderboard table
        print(f"\n  Top-{TOP_N} within-community agenda-setters by wc_net_strength:\n")
        col_w = 32
        header = f"  {'Rank':<5}" + "".join(f"  {c}th{'':<{col_w-4}}" for c in CONGRESSES)
        print(header)
        print(f"  {'-' * (5 + len(CONGRESSES) * (col_w + 2))}")

        for rank_i in range(TOP_N):
            row_str = f"  {rank_i+1:<5}"
            for congress in CONGRESSES:
                entry = tops[congress][rank_i] if rank_i < len(tops[congress]) else ("—", 0.0)
                cell  = f"{entry[0][:18]} ({entry[1]:+.2f})"
                row_str += f"  {cell:<{col_w}}"
            print(row_str)

        # Save leaderboard rows
        for congress in CONGRESSES:
            for rank_i, (firm, val) in enumerate(tops[congress], 1):
                leaderboard_rows.append({
                    "sector": c_label, "congress": congress,
                    "rank": rank_i, "firm": firm, "wc_net_strength": val,
                })

        # Rank stability
        if n_stable >= 5:
            adj_rhos = []
            for i in range(len(CONGRESSES) - 1):
                c1, c2 = CONGRESSES[i], CONGRESSES[i + 1]
                if c1 in stable_set.columns and c2 in stable_set.columns:
                    v1 = stable_set[c1].dropna()
                    v2 = stable_set[c2].dropna()
                    common = v1.index.intersection(v2.index)
                    if len(common) >= 5:
                        rho, pval = spearmanr(v1[common], v2[common])
                        adj_rhos.append((c1, c2, round(rho, 3), round(pval, 4)))

            rank_mat = stable_set[CONGRESSES].T
            W, chi2_val, p_W = kendalls_w(rank_mat)

            print(f"\n  Rank stability (n={n_stable} stable firms):")
            print(f"    Kendall's W = {W}  chi2={chi2_val}  p={p_W}")
            print(f"    Adjacent-congress Spearman rho:")
            for c1, c2, rho, pval in adj_rhos:
                stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                print(f"      {c1}→{c2}: rho={rho:+.3f}  p={pval:.4f} {stars}")

            stability_rows.append({
                "sector": c_label, "n_stable": n_stable,
                "kendalls_W": W, "kendalls_W_chi2": chi2_val, "kendalls_W_p": p_W,
                **{f"rho_{r[0]}_{r[1]}": r[2] for r in adj_rhos},
                **{f"p_{r[0]}_{r[1]}": r[3] for r in adj_rhos},
            })
        else:
            print(f"\n  Rank stability: insufficient stable firms (n={n_stable} < 5).")

        # Persistent leaders: in top-5 in ≥4 of 7 congresses
        appearance_counts = {}
        for congress in CONGRESSES:
            for firm, _ in tops[congress]:
                appearance_counts[firm] = appearance_counts.get(firm, 0) + 1

        persistent = [(f, cnt) for f, cnt in appearance_counts.items() if cnt >= 4]
        persistent.sort(key=lambda x: -x[1])

        print(f"\n  Persistent leaders (top-{TOP_N} by wc_net_strength in ≥4/7 congresses):")
        if persistent:
            for firm, cnt in persistent:
                in_c = [str(c) for c in CONGRESSES if firm in [x[0] for x in tops.get(c, [])]]
                print(f"    {firm:<40} {cnt}/7  ({', '.join(in_c)}th)")
                persistent_rows.append({
                    "sector": c_label, "firm": firm, "n_congresses": cnt,
                    "in_congresses": ", ".join(in_c),
                })
        else:
            print("    None — no firm reaches top-5 in ≥4 of 7 congresses.")

    # -- Summary stability table -----------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY: RANK CONCORDANCE BY SECTOR")
    print(f"{'='*70}")
    if stability_rows:
        print(f"\n  {'Sector':<22} {'n_stable':>9} {'W':>8} {'p':>10}  interpretation")
        print(f"  {'─'*65}")
        for r in stability_rows:
            W_val = r["kendalls_W"]
            p_val = r["kendalls_W_p"]
            interp = ("significant concordance" if pd.notna(p_val) and p_val < 0.05 and W_val >= 0.3
                      else "weak but significant" if pd.notna(p_val) and p_val < 0.05
                      else "not significant")
            print(f"  {r['sector']:<22} {r['n_stable']:>9} {W_val:>8.4f} "
                  f"{p_val:>10.5f}  {interp}")

    # -- Save outputs ----------------------------------------------------
    # -- Save and plot ---------------------------------------------------
    print(f"\n[3/3] Saving outputs ...")
    pd.DataFrame(leaderboard_rows).to_csv(
        OUT_DIR / "03_within_community_leaderboards.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(
        OUT_DIR / "03_rank_stability.csv", index=False)
    pd.DataFrame(persistent_rows).to_csv(
        OUT_DIR / "03_persistent_leaders.csv", index=False)

    if stability_rows:
        plot_kendalls_w(stability_rows, OUT_DIR)
    if leaderboard_rows:
        plot_energy_rank_flow(leaderboard_rows, OUT_DIR)

    print(f"    03_within_community_leaderboards.csv")
    print(f"    03_rank_stability.csv")
    print(f"    03_persistent_leaders.csv")
    print(f"    03_industry_hierarchy.txt")
    print(f"    03_kendalls_w_bar.png")
    print(f"    03_energy_rank_flow.png")
    print(f"\n{SEP}")
    print("Analysis complete.")

    log_f.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
