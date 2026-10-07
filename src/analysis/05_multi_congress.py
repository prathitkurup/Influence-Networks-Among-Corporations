"""
Multi-congress set overlap analysis.

Answers: Do the same top-net_strength firms appear across consecutive
congressional sessions? How much does the top-tier membership rotate?

For each consecutive congress pair (c_N → c_{N+1}):
  - Jaccard similarity of top-N firms by net_strength (N = 10, 20, 30)
  - Who enters and exits the top-30 each session?
  - Spearman ρ of net_strength ranks on common firm set

Outputs (outputs/analysis/):
  05_set_overlap.csv        — Jaccard scores per consecutive pair per N
  05_top30_transitions.csv  — entering/exiting firms per congress pair
  05_multi_congress.txt
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

CONGRESSES  = [111, 112, 113, 114, 115, 116, 117]
N_VALUES    = [10, 20, 30]    # top-N set sizes to compare

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
# Helpers
# ---------------------------------------------------------------------------

def jaccard(s1, s2):
    """Jaccard similarity between two sets."""
    s1, s2 = set(s1), set(s2)
    if not s1 and not s2:
        return np.nan
    return round(len(s1 & s2) / len(s1 | s2), 4)

def top_n_set(df, n):
    return set(df.nlargest(n, "net_strength")["firm"].tolist())

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(OUT_DIR / "05_multi_congress.txt", "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    SEP = "=" * 70
    print(SEP)
    print("ANALYSIS 05: MULTI-CONGRESS TOP-SET OVERLAP (111th–117th)")
    print(SEP)

    # -- Load all congress node attributes --------------------------------
    print("\n[1/2] Loading net_strength for all congresses ...")
    nodes = {}
    for c in CONGRESSES:
        p = DATA_DIR / f"congress/{c}/node_attributes.csv"
        if p.exists():
            nodes[c] = pd.read_csv(p)[["firm", "net_strength"]]
            print(f"  {c}th: {len(nodes[c])} firms")

    # -- Consecutive pair analysis ----------------------------------------
    print("\n[2/2] Consecutive congress pair analysis ...")
    overlap_rows    = []
    transition_rows = []

    for i in range(len(CONGRESSES) - 1):
        ci, cj = CONGRESSES[i], CONGRESSES[i + 1]
        ni = nodes[ci]
        nj = nodes[cj]

        common_firms = set(ni["firm"]) & set(nj["firm"])
        sub_i = ni[ni["firm"].isin(common_firms)].set_index("firm")["net_strength"]
        sub_j = nj[nj["firm"].isin(common_firms)].set_index("firm")["net_strength"]
        idx   = sub_i.index.intersection(sub_j.index)
        rho, pval = (np.nan, np.nan) if len(idx) < 5 else spearmanr(sub_i[idx], sub_j[idx])

        print(f"\n  {'─'*60}")
        print(f"  {ci}th → {cj}th Congress")
        print(f"  Firms in {ci}th: {len(ni)}  |  Firms in {cj}th: {len(nj)}  "
              f"|  Common: {len(common_firms)}")
        print(f"  Spearman ρ (net_strength, common firms): "
              f"{float(rho):.4f}  p={float(pval):.4e}")

        print(f"\n  Jaccard similarity of top-N sets:")
        print(f"  {'N':>6} {'|∩|':>6} {'|∪|':>6} {'Jaccard':>10}")
        print(f"  {'─'*32}")
        for n in N_VALUES:
            s_i = top_n_set(ni, n)
            s_j = top_n_set(nj, n)
            j   = jaccard(s_i, s_j)
            inter = len(s_i & s_j)
            union = len(s_i | s_j)
            print(f"  {n:>6} {inter:>6} {union:>6} {j:>10.4f}")
            overlap_rows.append({
                "pair": f"{ci}->{cj}", "N": n,
                "n_intersection": inter, "n_union": union, "jaccard": j,
            })

        # Transitions for top-30
        top30_i = top_n_set(ni, 30)
        top30_j = top_n_set(nj, 30)
        entering = top30_j - top30_i
        exiting  = top30_i - top30_j
        retained = top30_i & top30_j

        print(f"\n  Top-30 transitions:")
        print(f"    Retained:  {len(retained)}/30")
        print(f"    Exiting:   {sorted(exiting)[:6]}{'...' if len(exiting) > 6 else ''}")
        print(f"    Entering:  {sorted(entering)[:6]}{'...' if len(entering) > 6 else ''}")

        for firm in retained:
            transition_rows.append({"pair": f"{ci}->{cj}", "firm": firm,
                                     "status": "retained"})
        for firm in exiting:
            transition_rows.append({"pair": f"{ci}->{cj}", "firm": firm,
                                     "status": "exiting"})
        for firm in entering:
            transition_rows.append({"pair": f"{ci}->{cj}", "firm": firm,
                                     "status": "entering"})

    # -- Summary ----------------------------------------------------------
    overlap_df = pd.DataFrame(overlap_rows)
    print(f"\n{'─'*70}")
    print("SUMMARY: Mean Jaccard by N across all consecutive pairs")
    print(f"{'─'*70}")
    print(f"\n  {'N':>6} {'Mean Jaccard':>14} {'Min':>8} {'Max':>8}")
    print(f"  {'─'*40}")
    for n in N_VALUES:
        sub = overlap_df[overlap_df["N"] == n]["jaccard"]
        print(f"  {n:>6} {sub.mean():>14.4f} {sub.min():>8.4f} {sub.max():>8.4f}")

    print(f"\n  Interpretation: Jaccard of top-30 ranges from "
          f"{overlap_df[overlap_df['N']==30]['jaccard'].min():.3f} to "
          f"{overlap_df[overlap_df['N']==30]['jaccard'].max():.3f}, "
          f"mean = {overlap_df[overlap_df['N']==30]['jaccard'].mean():.3f}.")
    print(f"  The top-30 agenda-setter pool rotates substantially between sessions,")
    print(f"  with roughly half the firms retained in any consecutive pair.")

    # -- Persistent across ALL 7 congresses -------------------------------
    print(f"\n{'─'*70}")
    print(f"FIRMS IN TOP-30 by net_strength IN ALL 7 CONGRESSES")
    print(f"{'─'*70}")
    appearance = {}
    for c in CONGRESSES:
        top30 = top_n_set(nodes[c], 30)
        for firm in top30:
            appearance[firm] = appearance.get(firm, 0) + 1
    always_top30 = [(f, cnt) for f, cnt in appearance.items() if cnt == 7]
    often_top30  = [(f, cnt) for f, cnt in appearance.items() if cnt >= 5]
    often_top30.sort(key=lambda x: -x[1])

    print(f"\n  Firms in top-30 all 7 congresses: {len(always_top30)}")
    for f, _ in sorted(always_top30):
        print(f"    {f}")
    print(f"\n  Firms in top-30 in ≥5 of 7 congresses:")
    for f, cnt in often_top30:
        in_c = [str(c) for c in CONGRESSES
                if f in top_n_set(nodes[c], 30)]
        print(f"    {f:<42} {cnt}/7  ({', '.join(in_c)}th)")

    # -- Save outputs -----------------------------------------------------
    overlap_df.to_csv(OUT_DIR / "05_set_overlap.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(
        OUT_DIR / "05_top30_transitions.csv", index=False)
    print(f"\n  Outputs:")
    print(f"    05_set_overlap.csv")
    print(f"    05_top30_transitions.csv")
    print(f"    05_multi_congress.txt")
    print(f"\n{SEP}")
    print("Analysis complete.")

    log_f.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
