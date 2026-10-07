"""
Cross-congressional stability — net_strength of top-30 agenda-setters.

Answers:
  1. Is net_strength stable across sessions? (correlation heatmap, top-30 firms)
  2. How similar are the ranked agenda-setter lists between sessions?
     (RBO similarity of ranked net_strength lists; adjacent-congress Spearman)
  3. How stable are within-community rankings? (summary table from Analysis 03)

Outputs (outputs/analysis/):
  04_net_strength_corr_matrix.csv   — Spearman ρ between all congress pairs
  04_net_strength_heatmap.png       — visual correlation matrix
  04_ranked_list_similarity.csv     — RBO + Spearman between ranked lists
  04_cross_congressional.txt
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

CONGRESSES = [111, 112, 113, 114, 115, 116, 117]
TOP_N      = 30   # top firms for correlation and RBO analyses

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
# RBO similarity between two ranked lists
# ---------------------------------------------------------------------------

def rbo_sim(list1, list2, p=0.9):
    """
    Rank-biased overlap similarity between two ranked lists.
    p: persistence parameter (0.9 weights top ranks heavily).
    Returns score in [0, 1].
    """
    if not list1 or not list2:
        return 0.0
    s, l = (list1, list2) if len(list1) <= len(list2) else (list2, list1)
    k = min(len(s), len(l))
    x_k = 0.0
    for d in range(1, k + 1):
        set_s = set(s[:d])
        set_l = set(l[:d])
        x_k += (len(set_s & set_l) / d) * (p ** (d - 1))
    return round((1 - p) * x_k, 6)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(OUT_DIR / "04_cross_congressional.txt", "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    SEP = "=" * 70
    print(SEP)
    print("ANALYSIS 04: CROSS-CONGRESSIONAL STABILITY (111th–117th)")
    print(SEP)

    # -- Load node attributes for all congresses -------------------------
    print("\n[1/4] Loading net_strength for all congresses ...")
    nodes = {}
    for c in CONGRESSES:
        p = DATA_DIR / f"congress/{c}/node_attributes.csv"
        if p.exists():
            nodes[c] = pd.read_csv(p)[["firm", "net_strength"]]
            print(f"  {c}th: {len(nodes[c])} firms")

    # Build wide matrix (all firms, all congresses)
    all_firms = sorted(set.union(*[set(df["firm"]) for df in nodes.values()]))
    wide = pd.DataFrame(index=all_firms)
    for c in CONGRESSES:
        if c in nodes:
            wide[c] = nodes[c].set_index("firm")["net_strength"]
    print(f"\n  Total unique firms: {len(wide)}")
    print(f"  Firms in all 7 congresses: {wide.dropna().shape[0]}")

    # -- Identify top-30 firms (by mean net_strength across sessions) ----
    stable = wide.dropna()
    if len(stable) < TOP_N:
        top30_firms = list(wide.index)
        print(f"  WARNING: fewer than {TOP_N} stable firms; using all {len(wide)}")
    else:
        mean_ns = stable.mean(axis=1)
        top30_firms = mean_ns.nlargest(TOP_N).index.tolist()
    top30_wide = wide.loc[top30_firms]

    print(f"\n  Top-{TOP_N} firms (by mean net_strength, stable set):")
    mean_ns_all = wide.loc[top30_firms].mean(axis=1)
    for i, firm in enumerate(top30_firms, 1):
        print(f"    {i:>3}. {firm:<42} mean_ns={mean_ns_all.get(firm, float('nan')):.4f}")

    # -- Correlation matrix on top-30 set --------------------------------
    print(f"\n[2/4] Spearman correlation matrix (net_strength, top-{TOP_N} firms) ...")
    labels = [str(c) for c in CONGRESSES]
    corr_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    pval_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    n_matrix    = pd.DataFrame(0,      index=labels, columns=labels)

    for i, ci in enumerate(CONGRESSES):
        for j, cj in enumerate(CONGRESSES):
            if i == j:
                corr_matrix.at[str(ci), str(cj)] = 1.0
                continue
            col_i = top30_wide[ci].dropna()
            col_j = top30_wide[cj].dropna()
            common = col_i.index.intersection(col_j.index)
            n = len(common)
            if n >= 5:
                rho, pval = spearmanr(col_i[common], col_j[common])
                corr_matrix.at[str(ci), str(cj)] = round(float(rho), 4)
                pval_matrix.at[str(ci), str(cj)] = round(float(pval), 6)
                n_matrix.at[str(ci), str(cj)]    = n

    corr_matrix.to_csv(OUT_DIR / "04_net_strength_corr_matrix.csv")

    print(f"\n  Spearman ρ (net_strength, top-{TOP_N} firms):")
    print(f"  {'':>8}", end="")
    for c in CONGRESSES:
        print(f"  {c:>8}", end="")
    print()
    for ci in CONGRESSES:
        print(f"  {ci:>8}", end="")
        for cj in CONGRESSES:
            v = corr_matrix.at[str(ci), str(cj)]
            p = pval_matrix.at[str(ci), str(cj)]
            if ci == cj:
                print(f"  {'1.000':>8}", end="")
            elif pd.isna(v):
                print(f"  {'—':>8}", end="")
            else:
                sig = "*" if not pd.isna(p) and p < 0.05 else " "
                print(f"  {v:>7.3f}{sig}", end="")
        print()
    print(f"  (* p < 0.05)")

    # -- Heatmap ---------------------------------------------------------
    print(f"\n[3/4] Generating heatmap ...")
    fig, ax = plt.subplots(figsize=(8, 6))
    vals = corr_matrix.astype(float).values
    im = ax.imshow(vals, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    ax.set_xticks(range(len(CONGRESSES)))
    ax.set_xticklabels([str(c) for c in CONGRESSES], fontsize=10)
    ax.set_yticks(range(len(CONGRESSES)))
    ax.set_yticklabels([str(c) for c in CONGRESSES], fontsize=10)
    ax.set_title(
        f"Cross-Congressional net_strength Correlation\n"
        f"(Top-{TOP_N} agenda-setters by mean net_strength, 111th–117th Congress)",
        fontsize=11, pad=12
    )
    for i in range(len(CONGRESSES)):
        for j in range(len(CONGRESSES)):
            v = vals[i, j]
            n = int(n_matrix.iloc[i, j]) if i != j else 0
            if not np.isnan(v):
                lbl = f"{v:.2f}" if i == j else f"{v:.2f}\n(n={n})"
                ax.text(j, i, lbl, ha="center", va="center", fontsize=7,
                        color="black" if abs(v) < 0.7 else "white")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_net_strength_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved: 04_net_strength_heatmap.png")

    # -- Ranked list similarity (RBO + Spearman adjacent) ----------------
    print(f"\n[4/4] Ranked list similarity between consecutive sessions ...")
    print(f"  (RBO p=0.9, ranked by net_strength on full firm sets per congress)\n")

    rbo_rows = []
    print(f"  {'Pair':<12} {'RBO_full':>10} {'RBO_top30':>10} {'Spearman':>10} "
          f"{'n_common':>10} {'p':>10}")
    print(f"  {'─'*64}")
    for i in range(len(CONGRESSES) - 1):
        ci, cj = CONGRESSES[i], CONGRESSES[i + 1]
        nd_i = nodes[ci].sort_values("net_strength", ascending=False)
        nd_j = nodes[cj].sort_values("net_strength", ascending=False)

        list_i_full = nd_i["firm"].tolist()
        list_j_full = nd_j["firm"].tolist()

        list_i_top  = nd_i.head(TOP_N)["firm"].tolist()
        list_j_top  = nd_j.head(TOP_N)["firm"].tolist()

        rbo_full  = rbo_sim(list_i_full, list_j_full)
        rbo_top30 = rbo_sim(list_i_top,  list_j_top)

        # Spearman on common firms
        common = set(nd_i["firm"]) & set(nd_j["firm"])
        sub_i  = nd_i[nd_i["firm"].isin(common)].set_index("firm")["net_strength"]
        sub_j  = nd_j[nd_j["firm"].isin(common)].set_index("firm")["net_strength"]
        idx    = sub_i.index.intersection(sub_j.index)
        rho, pval = (np.nan, np.nan) if len(idx) < 5 else spearmanr(sub_i[idx], sub_j[idx])

        stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {ci}→{cj:<6}   {rbo_full:>10.4f} {rbo_top30:>10.4f} "
              f"{float(rho):>10.4f} {len(idx):>10} {float(pval):>10.4f} {stars}")
        rbo_rows.append({
            "pair": f"{ci}->{cj}", "rbo_full": rbo_full, "rbo_top30": rbo_top30,
            "spearman_rho": round(float(rho), 4), "n_common": len(idx),
            "spearman_p": round(float(pval), 6),
        })

    pd.DataFrame(rbo_rows).to_csv(OUT_DIR / "04_ranked_list_similarity.csv", index=False)

    # Summary stats
    rbo_df = pd.DataFrame(rbo_rows)
    print(f"\n  Mean RBO (full ranked list):        {rbo_df['rbo_full'].mean():.4f}")
    print(f"  Mean RBO (top-{TOP_N} only):           {rbo_df['rbo_top30'].mean():.4f}")
    print(f"  Mean Spearman ρ (net_strength):     {rbo_df['spearman_rho'].mean():.4f}")
    n_sig = (rbo_df["spearman_p"] < 0.05).sum()
    print(f"  Spearman p < 0.05: {n_sig}/{len(rbo_df)} consecutive pairs")

    print(f"\n  Outputs:")
    print(f"    04_net_strength_corr_matrix.csv")
    print(f"    04_net_strength_heatmap.png")
    print(f"    04_ranked_list_similarity.csv")
    print(f"    04_cross_congressional.txt")
    print(f"\n{SEP}")
    print("Analysis complete.")

    log_f.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
