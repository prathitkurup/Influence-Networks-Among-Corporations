"""
RBO and cosine similarity utilities for bill-portfolio network construction.
RBO reference: Webber et al. (2010), ACM Trans. Inf. Syst., 28(4). See §18.
"""

import numpy as np
import pandas as pd


# -- Aggregation helpers -------------------------------------------------------

def aggregate_per_firm_bill(df):
    """Sum amount_allocated across multiple rows per (fortune_name, bill_number) to get true total spend."""
    return df.groupby(
        ["fortune_name", "bill_number"], as_index=False
    )["amount_allocated"].sum()


def compute_zero_budget_fracs(df):
    """Add frac column (spend / total_budget); excludes zero-budget firms and validates fracs sum to 1.0 per firm."""
    company_totals = df.groupby("fortune_name")["amount_allocated"].sum().rename("total_budget")
    df = df.merge(company_totals, on="fortune_name", how="left")

    zero_budget = company_totals[company_totals == 0].index.tolist()
    if zero_budget:
        print(f"  Warning: {len(zero_budget)} firm(s) excluded (zero total budget).")
        df = df[df["total_budget"] > 0].copy()

    df["frac"] = df["amount_allocated"] / df["total_budget"]

    frac_sums = df.groupby("fortune_name")["frac"].sum()
    bad = frac_sums[~frac_sums.between(0.999, 1.001)]
    if not bad.empty:
        raise ValueError(
            f"frac values do not sum to 1.0 for {len(bad)} firm(s): "
            f"{bad.to_dict()}\n"
            "Check opensecrets_extraction.py for changes that may have broken allocation logic."
        )
    return df


# -- RBO -----------------------------------------------------------------------

def rbo_score(l1, l2, p=0.90):
    """
    Rank-Biased Overlap (Webber et al. 2010), truncated min estimate.
    p=0.85 calibrated to Fortune 500 spend concentration (see §18).
    Returns float in [0, 1]; higher rank agreement weighted more heavily.
    """
    s = min(len(l1), len(l2))
    if s == 0:
        return 0.0
    set1: set = set()
    set2: set = set()
    total = 0.0
    for d in range(1, s + 1):
        set1.add(l1[d - 1])
        set2.add(l2[d - 1])
        total += (p ** (d - 1)) * (len(set1 & set2) / d)
    return (1.0 - p) * total


def build_ranked_lists(df, top_bills=100):
    """Build {firm: [bill, ...]} ranked by allocated spend descending, truncated to top_bills."""
    ranked = {}
    for firm, grp in df.groupby("fortune_name"):
        bills_sorted = (grp[grp["amount_allocated"] > 0]
                        .sort_values("amount_allocated", ascending=False)["bill_number"]
                        .tolist())
        if top_bills and top_bills > 0:
            bills_sorted = bills_sorted[:top_bills]
        ranked[firm] = bills_sorted
    return ranked


# -- Cosine matrix -------------------------------------------------------------

def build_frac_matrix(df):
    """Build (firms × bills) portfolio-share pivot matrix for cosine similarity; returns (pivot_df, firms_list, bills_list)."""
    pivot = df.pivot_table(
        index="fortune_name", columns="bill_number",
        values="frac", fill_value=0.0
    )
    return pivot, list(pivot.index), list(pivot.columns)
