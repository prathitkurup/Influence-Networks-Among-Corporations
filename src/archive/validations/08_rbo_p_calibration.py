"""
Validation 08: RBO persistence parameter (p) calibration against empirical
spend concentration.

Computes the cumulative lobbying spend share by bill rank across all Fortune 500
firms (mega-bills INCLUDED — no MAX_BILL_DF filter) and overlays theoretical
RBO cumulative weight curves for p in {0.70, 0.80, 0.85, 0.90, 0.95, 0.98}.

The goal is to verify that RBO_P = 0.85 best tracks the empirical median firm's
spend concentration curve, grounding the geometric decay in observable data
rather than an arbitrary default.

Outputs
-------
outputs/archive/validation/rbo_p_calibration.png  - two-panel calibration figure
Printed table: cumulative spend share at key ranks (median, P25, P75)
Printed table: rank at which 50/80/95% of spend is accumulated per firm

Run: python validations/08_rbo_p_calibration.py  (from src/)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_DIR, ROOT
from utils.data_loading import load_bills_data
from utils.similarity import aggregate_per_firm_bill, compute_zero_budget_fracs

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "archive" / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

P_VALUES    = [0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
MAX_RANK    = 50   # plot up to rank 50; tail is negligible
KEY_RANKS   = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
COLORS_RBO  = ["#e41a1c", "#ff7f00", "#2ca02c", "#4daf4a", "#377eb8", "#984ea3"]


def compute_cum_spend(df_agg, max_rank):
    """
    For each rank k in 1..max_rank, compute the cumulative frac share
    (sum of fracs for bills ranked <= k) per firm, returning lists of
    per-firm values for each rank.
    """
    df_agg = df_agg.copy()
    df_agg["rank"] = (df_agg
        .groupby("fortune_name")["amount_allocated"]
        .rank(method="first", ascending=False)
        .astype(int))

    cum = {}
    for k in range(1, max_rank + 1):
        vals = []
        for _, grp in df_agg.groupby("fortune_name"):
            vals.append(grp[grp["rank"] <= k]["frac"].sum())
        cum[k] = vals
    return cum


def print_spend_table(cum, key_ranks):
    """Print cumulative spend share table at key ranks."""
    print("\n=== Cumulative spend share by bill rank (incl. mega-bills) ===")
    print(f"{'Rank':<6}  {'Median':>8}  {'P25':>8}  {'P75':>8}  {'Mean':>8}")
    print("-" * 48)
    for k in key_ranks:
        if k not in cum:
            continue
        vals = cum[k]
        print(f"{k:<6}  {np.median(vals):>7.1%}  "
              f"{np.percentile(vals,25):>7.1%}  "
              f"{np.percentile(vals,75):>7.1%}  "
              f"{np.mean(vals):>7.1%}")


def print_threshold_table(df_agg):
    """Print rank at which each firm crosses 50/80/95% cumulative spend."""
    df_agg = df_agg.copy()
    df_agg["rank"] = (df_agg
        .groupby("fortune_name")["amount_allocated"]
        .rank(method="first", ascending=False)
        .astype(int))

    print("\n=== Rank at which firm accumulates X% of lobbying spend ===")
    print(f"  (median and P25/P75 across firms)")
    print(f"{'Target':>8}  {'Median rank':>12}  {'P25 rank':>10}  {'P75 rank':>10}")
    print("-" * 48)
    for t in [0.50, 0.80, 0.95]:
        rank_at_t = []
        for _, grp in df_agg.groupby("fortune_name"):
            grp_sorted = grp.sort_values("rank")
            cum = grp_sorted["frac"].cumsum().values
            hits = np.where(cum >= t)[0]
            rank_at_t.append(hits[0] + 1 if len(hits) > 0 else len(grp_sorted))
        print(f"  {t:>6.0%}  {np.median(rank_at_t):>12.0f}  "
              f"{np.percentile(rank_at_t,25):>10.0f}  "
              f"{np.percentile(rank_at_t,75):>10.0f}")


def plot_calibration(cum, max_rank, p_values, colors, out_path):
    """Two-panel figure: empirical spend concentration + RBO weight overlay."""
    ranks   = list(range(1, max_rank + 1))
    medians = [np.median(cum[k]) for k in ranks]
    p25     = [np.percentile(cum[k], 25) for k in ranks]
    p75     = [np.percentile(cum[k], 75) for k in ranks]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # -- LEFT: empirical spend concentration --
    ax = axes[0]
    ax.fill_between(ranks, p25, p75, alpha=0.20, color="#aaaaaa", label="IQR (P25–P75)")
    ax.plot(ranks, medians, color="#222222", lw=2.2, label="Median firm")
    for pct, ls in [(0.50, "--"), (0.80, ":"), (0.95, "-.")]:
        ax.axhline(pct, color="grey", lw=0.8, ls=ls, alpha=0.7)
        ax.text(max_rank + 0.3, pct, f"{int(pct*100)}%", va="center",
                fontsize=8, color="grey")
    for k in [3, 7, 10]:
        y = medians[k - 1]
        ax.annotate(f"rank {k}\n{y:.0%}", xy=(k, y),
                    xytext=(k + 3, y - 0.07), fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.7))
    ax.set_xlabel("Bill rank (1 = highest spend)", fontsize=10)
    ax.set_ylabel("Cumulative spend share", fontsize=10)
    ax.set_title("Empirical spend concentration\n"
                 "(Fortune 500, 116th Congress, incl. mega-bills)", fontsize=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlim(1, max_rank)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # -- RIGHT: RBO cumulative weight vs empirical --
    ax2 = axes[1]
    for p, c in zip(p_values, colors):
        cum_rbo = [(1 - p ** k) for k in ranks]
        lw = 2.5 if p == 0.85 else 1.5   # highlight chosen p
        ls = "-" if p == 0.85 else "--"
        ax2.plot(ranks, cum_rbo, color=c, lw=lw, ls=ls, label=f"p={p}")
    ax2.plot(ranks, medians, color="black", lw=2, ls="-", label="Empirical median")
    ax2.fill_between(ranks, p25, p75, alpha=0.12, color="black", label="Empirical IQR")
    for pct, ls in [(0.50, "--"), (0.80, ":"), (0.95, "-.")]:
        ax2.axhline(pct, color="grey", lw=0.8, ls=ls, alpha=0.7)
    ax2.set_xlabel("Bill rank (1 = highest spend)", fontsize=10)
    ax2.set_ylabel("Cumulative weight / spend share", fontsize=10)
    ax2.set_title("RBO cumulative weight vs.\nempirical spend concentration", fontsize=10)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_xlim(1, max_rank)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nCalibration plot -> {out_path}")


def main():
    print("=" * 60)
    print("Validation 08: RBO p-parameter calibration")
    print("=" * 60)
    print("NOTE: mega-bills included (no MAX_BILL_DF filter)")

    df_raw = load_bills_data(DATA_DIR / "opensecrets_lda_reports.csv")
    print(f"\nFirms: {df_raw['fortune_name'].nunique()}  "
          f"Bills: {df_raw['bill_number'].nunique():,}")

    df_agg = aggregate_per_firm_bill(df_raw)
    df_agg = compute_zero_budget_fracs(df_agg)

    list_lengths = df_agg.groupby("fortune_name")["bill_number"].count()
    print(f"\n=== Bills lobbied per firm ===")
    print(f"  mean={list_lengths.mean():.1f}  median={list_lengths.median():.0f}  "
          f"P25={list_lengths.quantile(0.25):.0f}  "
          f"P75={list_lengths.quantile(0.75):.0f}  "
          f"max={list_lengths.max()}")

    cum = compute_cum_spend(df_agg, MAX_RANK)
    print_spend_table(cum, KEY_RANKS)
    print_threshold_table(df_agg)

    out_path = OUTPUT_DIR / "rbo_p_calibration.png"
    plot_calibration(cum, MAX_RANK, P_VALUES, COLORS_RBO, out_path)


if __name__ == "__main__":
    main()
