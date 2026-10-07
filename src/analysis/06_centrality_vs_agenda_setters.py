"""
Centrality measures vs. agenda-setter rankings — 116th Congress.

Answers: Do high-centrality firms in the bill affiliation network also emerge
as top agenda-setters in the directed influence network?

Compares three centrality measures (global PageRank, BCZ intercentrality
[precomputed], within-community eigenvector from centrality_affiliation.csv)
against net_strength and wc_net_strength using Spearman correlations and a
ranked heatmap of the top-30 firms.

Data: loads precomputed centrality from data/centralities/ and
stored correlation results from prior validation outputs where available,
otherwise recomputes from node_attributes.csv and centrality_affiliation.csv.

Outputs (outputs/analysis/):
  06_centrality_vs_ns_correlations.csv  — Spearman ρ table
  06_top30_heatmap.png                  — ranked heatmap for top-30 by net_strength
  06_centrality_vs_agenda_setters.txt
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

CONGRESS = 116
TOP_N    = 30   # top firms by net_strength for heatmap

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
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # log_f = open(OUT_DIR / "06_centrality_vs_agenda_setters.txt", "w")
    # sys.stdout = _Tee(sys.__stdout__, log_f)

    # SEP = "=" * 70
    # print(SEP)
    # print("ANALYSIS 06: CENTRALITY VS. AGENDA-SETTERS — 116th CONGRESS")
    # print(SEP)
    # print()
    # print("Question: Do structurally central firms in the bill affiliation")
    # print("network also tend to be the top agenda-setters in the directed")
    # print("influence network?")

    # -- Load data -------------------------------------------------------
    cent = pd.read_csv(DATA_DIR / "centralities/centrality_affiliation.csv")
    nodes = pd.read_csv(DATA_DIR / f"congress/{CONGRESS}/node_attributes.csv")
    comm_df = pd.read_csv(DATA_DIR / "communities/communities_affiliation.csv")
    partition = dict(zip(comm_df["fortune_name"], comm_df["community_aff"]))

    # -- Compute wc_net_strength ----------------------------------------
    edges = pd.read_csv(DATA_DIR / f"congress/{CONGRESS}/rbo_directed_influence.csv")
    edges["src_comm"] = edges["source"].map(partition)
    edges["tgt_comm"] = edges["target"].map(partition)
    intra = edges[
        edges["src_comm"].notna() & edges["tgt_comm"].notna() &
        (edges["src_comm"] == edges["tgt_comm"])
    ]
    wc_ns = (
        intra.groupby("source")
        .apply(lambda g: float((g["rbo"] * g["net_temporal"]).sum()),
               include_groups=False)
        .rename("wc_net_strength")
        .rename_axis("firm")
        .reset_index()
    )

    # -- Merge -----------------------------------------------------------
    master = (
        nodes
        .merge(cent[["firm", "global_pagerank", "katz_centrality",
                      "within_comm_eigenvector"]], on="firm", how="left")
        .merge(wc_ns, on="firm", how="left")
    )
    master["community"] = master["firm"].map(partition)

    # Centrality columns to compare
    cent_cols = {
        "global_pagerank":        "Global PageRank",
        "katz_centrality":        "Katz centrality",
        "within_comm_eigenvector": "WC eigenvector",
    }
    outcome_cols = {
        "net_strength":    "net_strength (global)",
        "wc_net_strength": "wc_net_strength (within-comm)",
    }

    # print(f"\n  Firms with all centrality measures: "
          # f"{master.dropna(subset=list(cent_cols.keys())).shape[0]}")
    # print(f"  Firms with wc_net_strength:          {master['wc_net_strength'].notna().sum()}")

    # -- Spearman correlations -------------------------------------------
    # print(f"\n[1/2] Spearman correlations ...")
    corr_rows = []
    # header = (f"  {'Centrality measure':<25} {'Outcome':<30} "
              # f"{'ρ_full':>8} {'p_full':>10} "
              # f"{'ρ_top30':>8} {'p_top30':>9} {'top30_overlap':>14}")
    # print(f"\n{header}")
    # print(f"  {'─'*105}")

    for c_col, c_label in cent_cols.items():
        for o_col, o_label in outcome_cols.items():
            merged = master[["firm", c_col, o_col]].dropna()

            # Full sample
            rho_f, p_f = (np.nan, np.nan) if len(merged) < 5 else spearmanr(
                merged[c_col], merged[o_col])

            # Top-30 by outcome
            top30 = master.nlargest(TOP_N, o_col)[[c_col, o_col]].dropna()
            rho_t, p_t = (np.nan, np.nan) if len(top30) < 5 else spearmanr(
                top30[c_col], top30[o_col])

            # Top-30 overlap
            top30_by_cent    = set(master.nlargest(TOP_N, c_col)["firm"])
            top30_by_outcome = set(master.nlargest(TOP_N, o_col)["firm"])
            overlap_frac = len(top30_by_cent & top30_by_outcome) / TOP_N

            # stars_f = "***" if not np.isnan(p_f) and p_f < 0.001 else "**" if not np.isnan(p_f) and p_f < 0.01 else "*" if not np.isnan(p_f) and p_f < 0.05 else ""
            # print(f"  {c_label:<25} {o_label:<30} "
                  # f"{float(rho_f):>8.4f}{stars_f:<1} {float(p_f):>10.4e} "
                  # f"{float(rho_t):>8.4f} {float(p_t):>10.4e} "
                  # f"{overlap_frac:>14.4f}")
            corr_rows.append({
                "centrality": c_label, "outcome": o_label,
                "rho_full": round(float(rho_f), 4), "p_full": round(float(p_f), 5),
                "n_full": len(merged),
                "rho_top30": round(float(rho_t), 4), "p_top30": round(float(p_t), 5),
                "n_top30": len(top30),
                "top30_overlap_fraction": overlap_frac,
            })

    # pd.DataFrame(corr_rows).to_csv(
        # OUT_DIR / "06_centrality_vs_ns_correlations.csv", index=False)

    # -- Top-30 heatmap --------------------------------------------------
    # print(f"\n[2/2] Generating top-{TOP_N} ranked heatmap ...")

    # top30_firms = master.nlargest(TOP_N, "net_strength")["firm"].tolist()
    # sub = master[master["firm"].isin(top30_firms)].copy()
    # sub = sub.sort_values("net_strength", ascending=False)

    # Columns to show in heatmap (rank-normalise each measure to [0,1])
    # hm_cols = ["net_strength", "net_influence", "wc_net_strength",
               # "global_pagerank", "katz_centrality",
               # "within_comm_eigenvector"]
    # hm_labels = ["net_strength", "net_influence", "wc_net_strength",
                 # "Global PageRank", "Katz centrality",
                 # "WC eigenvector"]

    # Rank-normalize: rank ascending so high rank = high value = yellow
    # hm_data = pd.DataFrame(index=sub["firm"])
    # for col in hm_cols:
        # vals = sub[col].values
        # if np.all(np.isnan(vals)):
            # hm_data[col] = np.nan
        # else:
            # Use rank percentile
            # rank = pd.Series(vals).rank(pct=True, na_option="bottom").values
            # hm_data[col] = rank

    # fig, ax = plt.subplots(figsize=(11, 9))
    # im = ax.imshow(hm_data.values.astype(float), cmap="YlOrRd", aspect="auto",
                   # vmin=0, vmax=1)
    # plt.colorbar(im, ax=ax, shrink=0.6, label="Percentile rank (all 293 firms)")

    # ax.set_xticks(range(len(hm_cols)))
    # ax.set_xticklabels(hm_labels, rotation=35, ha="right", fontsize=9)
    # ax.set_yticks(range(len(sub)))
    # ax.set_yticklabels(sub["firm"].tolist(), fontsize=8)
    # ax.set_title(
        # f"Top-{TOP_N} Agenda-Setters: Centrality vs. Influence Measures\n"
        # f"116th Congress (ranked by net_strength, percentile across 293 firms)",
        # fontsize=11, pad=12
    # )

    # Annotate cells with actual values for influence measures
    # for row_i, firm in enumerate(sub["firm"]):
        # for col_j, col in enumerate(hm_cols):
            # val = sub.loc[sub["firm"] == firm, col].values[0]
            # if not np.isnan(val):
                # txt = f"{val:.2f}" if col_j < 3 else f"{val:.3f}"
                # bg = float(hm_data.loc[firm, col])
                # color = "white" if bg > 0.65 else "black"
                # ax.text(col_j, row_i, txt, ha="center", va="center",
                        # fontsize=6, color=color)

    # fig.tight_layout()
    # fig.savefig(OUT_DIR / "06_top30_heatmap.png", dpi=150, bbox_inches="tight")
    # plt.close(fig)
    # print(f"  Heatmap saved: 06_top30_heatmap.png")

    # -- Interpretation --------------------------------------------------
    # print(f"\n  Key findings:")
    corr_df = pd.DataFrame(corr_rows)  # needed below for Figure 2
    # ns_corrs = corr_df[corr_df["outcome"] == "net_strength (global)"].sort_values(
        # "rho_full", key=abs, ascending=False)
    # best = ns_corrs.iloc[0]
    # print(f"  Highest |ρ| with net_strength: {best['centrality']}  "
          # f"ρ={best['rho_full']:.4f}  p={best['p_full']:.4e}")
    # print(f"  Top-30 overlap (best measure):  "
          # f"{best['top30_overlap_fraction']:.0%}")
    # print(f"  → Centrality and agenda-setting are weakly correlated at best,")
    # print(f"    consistent with directed influence reflecting dynamic temporal")
    # print(f"    leadership rather than static structural position.")

    # -- Figure 2: clean 4×2 Spearman ρ heatmap -------------------------
    # print(f"\n[3/3] Generating correlation heatmap and scatter ...")
    cent_labels = ["Global PageRank", "Katz centrality", "WC eigenvector"]
    out_labels  = ["net_strength (global)", "wc_net_strength (within-comm)"]

    rho_mat = np.full((3, 2), np.nan)
    p_mat   = np.full((3, 2), np.nan)
    for i, cl in enumerate(cent_labels):
        for j, ol in enumerate(out_labels):
            row = corr_df[(corr_df["centrality"] == cl) &
                          (corr_df["outcome"] == ol)]
            if len(row):
                rho_mat[i, j] = row.iloc[0]["rho_full"]
                p_mat[i, j]   = row.iloc[0]["p_full"]

    # Global configuration to match standard LaTeX document fonts
    plt.rcParams.update({
    "text.usetex": False,           # Turn off external LaTeX requirement
    "font.family": "serif",
    "font.serif": ["cmr10"],        # Use standard LaTeX Computer Modern font
    "axes.formatter.use_mathtext": True # Tells math text to style regular numbers
    })

    fig, ax = plt.subplots(figsize=(5.5, 2))
    im = ax.imshow(rho_mat, cmap="RdYlGn", vmin=-0.3, vmax=0.3, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label=r"Spearman $\rho$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Net Strength\n(Global)", "Net Strength\n(WC)"],
                       fontsize=12)
    ax.set_yticks(range(3))
    y_labels = ["Global PageRank", "Katz Centrality", "WC Eigenvector"]
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_title(r"Agenda-Setting vs. Centrality: Spearman $\rho$ (Full Sample: 116th Congress)",
                 fontsize=12, pad=10)
    for i in range(3):
        for j in range(2):
            rho = rho_mat[i, j]
            p   = p_mat[i, j]
            print(rho)
            if not np.isnan(rho):
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                txt = fr"$\rho$={rho:.3f}{stars}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=12,
                        color="white" if abs(rho) < 0.18 else "black")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_corr_heatmap.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -- Figure 3: WC eigenvector vs net_strength scatter ----------------
    # scatter_df = master.dropna(subset=["within_comm_eigenvector", "net_strength"]).copy()
    # fig, ax = plt.subplots(figsize=(6, 5))
    # ax.scatter(scatter_df["within_comm_eigenvector"], scatter_df["net_strength"],
               # alpha=0.55, s=30, color="#4C72B0", edgecolors="white", linewidths=0.4)

    # Regression line
    # from numpy.polynomial.polynomial import polyfit as npfit
    # x = scatter_df["within_comm_eigenvector"].values
    # y = scatter_df["net_strength"].values
    # mask = np.isfinite(x) & np.isfinite(y)
    # c, m = npfit(x[mask], y[mask], 1)
    # xline = np.linspace(x[mask].min(), x[mask].max(), 100)
    # ax.plot(xline, m * xline + c, color="#C44E52", linewidth=1.8,
            # label=f"OLS fit (ρ={best['rho_full']:.3f})")

    # Label top-8 by net_strength
    # top8 = scatter_df.nlargest(8, "net_strength")
    # for _, row in top8.iterrows():
        # ax.annotate(row["firm"].title()[:20],
                    # xy=(row["within_comm_eigenvector"], row["net_strength"]),
                    # xytext=(5, 2), textcoords="offset points",
                    # fontsize=6.5, color="#333333")

    # ax.set_xlabel("Within-Community Eigenvector Centrality", fontsize=11)
    # ax.set_ylabel("Net Strength (agenda-setting influence)", fontsize=11)
    # ax.set_title("WC Eigenvector Centrality vs. Net Strength\n116th Congress  "
                 # r"(Spearman $\rho$ = 0.210, $p$ < 0.001)", fontsize=11)
    # ax.legend(fontsize=9)
    # ax.grid(alpha=0.25, linestyle="--")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # fig.tight_layout()
    # fig.savefig(OUT_DIR / "06_wc_eigen_scatter.png", dpi=150, bbox_inches="tight")
    # plt.close(fig)
    # print(f"  Saved: 06_corr_heatmap.png, 06_wc_eigen_scatter.png")

    # print(f"\n  Outputs:")
    # print(f"    06_centrality_vs_ns_correlations.csv")
    # print(f"    06_top30_heatmap.png")
    # print(f"    06_corr_heatmap.png")
    # print(f"    06_wc_eigen_scatter.png")
    # print(f"    06_centrality_vs_agenda_setters.txt")
    # print(f"\n{SEP}")
    # print("Analysis complete.")

    # log_f.close()
    # sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
