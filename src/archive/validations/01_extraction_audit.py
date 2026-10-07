"""
01_extraction_audit.py
Prathit Kurup, Victoria Figueroa

PURPOSE
-------
Audit the raw output of opensecrets_extraction.py to understand the data
structure before any network construction. This script documents WHY there
are multiple rows per (fortune_name, bill_number) pair and quantifies the
extent of duplication.

BACKGROUND
----------
opensecrets_extraction.py's bill expansion splits each LDA filing's reported
spend equally across all bills mentioned in that report. A single report
mentioning N bills produces N rows, each carrying (total_spend / N) as
amount_allocated. A firm that files multiple reports mentioning the same bill
therefore accumulates one row per report -- not one row per (firm, bill). This
is correct and intentional for spend accounting, but must be handled before
network construction:

  - For affiliation networks: drop_duplicates() reduces to presence/absence.
  - For cosine/RBO similarity: groupby().sum() collapses to true total allocated
    spend per (firm, bill), which is the correct economic quantity for frac
    computation.

NOTE ON IND='Y' FILTER
----------------------
opensecrets_extraction.py filters to ind='y' records only (OpenSecrets validity
flag). This eliminates superseded originals (amended filings) and double-count
subsidiary records before bill expansion. The duplication documented here is
therefore only the legitimate within-dataset multi-report duplication (a firm
lobbying the same bill across multiple valid quarterly reports), not artifact
duplication from superseded records.

DESIGN DECISION DOCUMENTED
--------------------------
The raw data format (one row per bill per report) is correct for tracking how
spend was allocated across filing periods. Aggregation before network
construction is a pre-processing step, not a fix to opensecrets_extraction.py.

OUTPUT
------
Writes a human-readable audit report to outputs/archive/validation/01_extraction_audit.txt
"""

import sys
import os
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_DIR

OUTPUT_PATH = str(Path(__file__).resolve().parents[3] / "outputs" / "archive" / "validation" / "01_extraction_audit.txt")


def run_audit(df):
    lines = []
    lines.append("=" * 70)
    lines.append("EXTRACTION AUDIT: opensecrets_lda_reports.csv")
    lines.append("=" * 70)

    # Basic dimensions
    lines.append(f"\n-- Raw Data Dimensions --")
    lines.append(f"  Total rows:             {len(df):,}")
    lines.append(f"  Unique firms:           {df['fortune_name'].nunique():,}")
    lines.append(f"  Unique bill numbers:    {df['bill_number'].nunique():,}")
    lines.append(f"  Unique (firm, bill) pairs: "
                 f"{df.groupby(['fortune_name','bill_number']).ngroups:,}")

    # Column overview
    lines.append(f"\n-- Columns --")
    for col in df.columns:
        n_null = df[col].isna().sum()
        lines.append(f"  {col:<30}  dtype={str(df[col].dtype):<10}  nulls={n_null:,}")

    # Spending overview (amount_allocated = per-bill even split of report spend)
    lines.append(f"\n-- Spend Distribution (amount_allocated per row) --")
    lines.append(f"  Total allocated spend:  ${df['amount_allocated'].sum():,.0f}")
    lines.append(f"  Mean per row:           ${df['amount_allocated'].mean():,.2f}")
    lines.append(f"  Median per row:         ${df['amount_allocated'].median():,.2f}")
    lines.append(f"  Rows with $0 spend:     {(df['amount_allocated'] == 0).sum():,}  "
                 f"({100*(df['amount_allocated']==0).mean():.1f}%)")

    # Duplication: multiple rows per (firm, bill)
    pair_counts = df.groupby(["fortune_name", "bill_number"]).size()
    lines.append(f"\n-- (Firm, Bill) Pair Duplication --")
    lines.append(f"  Unique pairs:           {len(pair_counts):,}")
    lines.append(f"  Pairs with 1 row:       {(pair_counts == 1).sum():,}  "
                 f"({100*(pair_counts==1).mean():.1f}%)")
    lines.append(f"  Pairs with 2–5 rows:    {((pair_counts>=2)&(pair_counts<=5)).sum():,}  "
                 f"({100*((pair_counts>=2)&(pair_counts<=5)).mean():.1f}%)")
    lines.append(f"  Pairs with 6–10 rows:   {((pair_counts>5)&(pair_counts<=10)).sum():,}  "
                 f"({100*((pair_counts>5)&(pair_counts<=10)).mean():.1f}%)")
    lines.append(f"  Pairs with >10 rows:    {(pair_counts>10).sum():,}  "
                 f"({100*(pair_counts>10).mean():.1f}%)")
    lines.append(f"  Max rows per pair:      {pair_counts.max()}")
    lines.append(f"  Mean rows per pair (multi only): "
                 f"{pair_counts[pair_counts > 1].mean():.2f}")

    # Why duplication exists: bills per report
    lines.append(f"\n-- Why Duplication Exists: Bills per Report --")
    lines.append(f"  opensecrets_extraction.py splits report spend equally across all bills.")
    lines.append(f"  A firm mentioning the same bill in 3 different reports produces")
    lines.append(f"  3 rows for that (firm, bill) pair.")

    # Show a concrete example: firm with most duplicates
    worst_pair = pair_counts.idxmax()
    worst_count = pair_counts.max()
    lines.append(f"\n  Worst case: {worst_pair[0]} × {worst_pair[1]}")
    lines.append(f"    → {worst_count} rows for this single (firm, bill) pair")
    example_rows = df[(df["fortune_name"] == worst_pair[0]) &
                      (df["bill_number"] == worst_pair[1])][["amount_allocated"]].reset_index(drop=True)
    lines.append(f"    Row amounts: {list(example_rows['amount_allocated'].round(0).astype(int))}")
    lines.append(f"    Sum (true total allocated spend): "
                 f"${example_rows['amount_allocated'].sum():,.0f}")

    # Firms with zero total budget
    firm_totals = df.groupby("fortune_name")["amount_allocated"].sum()
    zero_budget_firms = firm_totals[firm_totals == 0].index.tolist()
    lines.append(f"\n-- Zero-Budget Firms --")
    lines.append(f"  Firms with total reported spend = $0:  {len(zero_budget_firms)}")
    if zero_budget_firms:
        for firm in zero_budget_firms:
            n_bills = df[df["fortune_name"] == firm]["bill_number"].nunique()
            lines.append(f"    {firm}  ({n_bills} unique bills filed)")
    lines.append(f"  → These firms produce NaN fracs and are excluded before")
    lines.append(f"    cosine and RBO similarity computation.")

    # Post-aggregation check
    df_agg = df.groupby(["fortune_name", "bill_number"], as_index=False)["amount_allocated"].sum()
    lines.append(f"\n-- Post-Aggregation (groupby sum) --")
    lines.append(f"  Rows after aggregation: {len(df_agg):,}")
    lines.append(f"  (was {len(df):,} rows — {len(df)/len(df_agg):.1f}x reduction)")
    lines.append(f"  Total spend preserved:  ${df_agg['amount_allocated'].sum():,.0f}  "
                 f"(should match raw = ${df['amount_allocated'].sum():,.0f})")

    lines.append(f"\n-- Summary --")
    lines.append(f"  Raw data has {len(df):,} rows for {len(pair_counts):,} unique (firm, bill)")
    lines.append(f"  pairs — an average of {len(df)/len(pair_counts):.2f} rows per pair.")
    lines.append(f"  This is correct: opensecrets_extraction.py creates one row per bill per report.")
    lines.append(f"  Aggregation before network construction collapses to true totals.")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / "opensecrets_lda_reports.csv")

    report = run_audit(df)
    print(report)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
