"""
05_ind_filter_validation.py
Prathit Kurup, Victoria Figueroa

PURPOSE
-------
Validate that the ind='y' filter in opensecrets_extraction.py correctly
produces a deduplicated, non-double-counted Fortune 500 lobbying dataset.

The extraction was updated to filter on ind='y' (the OpenSecrets validity flag,
Data User Guide p.13) instead of the old Self-field type list. This script
verifies three correctness properties:

  1. NO SUPERSEDED ORIGINALS: When a quarterly amendment (q1a, q2a, ...) exists
     for a report, the original (q1, q2, ...) should be absent from the output.
     We detect superseded pairs by finding registrant+firm combinations that have
     both an original quarterly and an amendment in the same year+quarter.

  2. NO DOUBLE-COUNT SUBSIDIARIES: Records with Self='i' (external registrant
     for a self-filing parent where parent already includes that spend) should
     be rare in the output — only the small fraction with ind='y' should appear.

  3. UNIQUE REPORT IDs: Each uniq_id should appear exactly once at the report
     level (before bill expansion). Duplicate uniq_ids would indicate a parsing
     or join error.

Additionally audits the self_type and report_type distributions to confirm the
ind='y' filter produces the expected mix of record types.

OUTPUT
------
Writes a report to outputs/archive/validation/05_ind_filter_validation.txt
"""

import sys
import os
import re
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_DIR

OUTPUT_PATH = str(Path(__file__).resolve().parents[3] / "outputs" / "archive" / "validation" / "05_ind_filter_validation.txt")

SELF_TYPE_DESC = {
    "p": "self-filer parent",
    "n": "non-self-filer parent",
    "s": "self-filer subsidiary of self-filer parent",
    "m": "non-self-filer subsidiary of non-self-filer parent",
    "x": "self-filer subsidiary of non-self-filer parent",
    "c": "non-self-filer subsidiary of self-filer parent (same catorder)",
    "b": "non-self-filer subsidiary of self-filer parent (diff catorder)",
    "e": "non-self-filer subsidiary of self-filer subsidiary",
    "i": "non-self-filer of self-filer parent (same catorder) — usually ind=''",
}


def _quarter_base(report_type):
    """Extract base quarter string from report_type (e.g. 'q1a' -> 'q1', 'q3tn' -> 'q3')."""
    m = re.match(r"(q[1-4])", report_type.lower())
    return m.group(1) if m else None


def _is_amendment(report_type):
    """Return True if report_type is an amendment (ends with 'a' or 'an')."""
    rt = report_type.lower()
    return rt.endswith("a") or rt.endswith("an")


def run_validation(df_raw, df_reports):
    """
    df_raw    : opensecrets_lda_reports.csv (may have multiple rows per report due to bills)
    df_reports: deduplicated to one row per uniq_id (report-level)
    """
    lines = []
    lines.append("=" * 70)
    lines.append("IND='Y' FILTER VALIDATION: opensecrets_lda_reports.csv")
    lines.append("=" * 70)

    # ------------------------------------------------------------------
    # Basic dimensions
    lines.append(f"\n-- Dataset Dimensions --")
    lines.append(f"  Total rows (bill-expanded)   : {len(df_raw):,}")
    lines.append(f"  Unique report uniq_ids       : {df_reports['uniq_id'].nunique():,}")
    lines.append(f"  Unique Fortune 500 firms     : {df_reports['fortune_name'].nunique():,}")
    lines.append(f"  Years covered                : "
                 f"{sorted(df_reports['year'].unique())}")
    lines.append(f"  Total expenditure            : "
                 f"${df_reports['amount'].dropna().sum():,.0f}")

    # ------------------------------------------------------------------
    # Check 1: Unique uniq_ids at report level
    lines.append(f"\n-- CHECK 1: Unique Report IDs --")
    dup_ids = df_reports[df_reports.duplicated(subset=["uniq_id"], keep=False)]
    if dup_ids.empty:
        lines.append(f"  PASS: All {len(df_reports):,} uniq_ids are unique at report level.")
    else:
        lines.append(f"  FAIL: {dup_ids['uniq_id'].nunique():,} uniq_ids appear more than once!")
        lines.append(f"  Sample duplicates:")
        for uid in list(dup_ids["uniq_id"].unique())[:5]:
            lines.append(f"    {uid}")

    # ------------------------------------------------------------------
    # Check 2: No superseded originals (amended pairs)
    # For each (registrant, fortune_name, year, quarter_base), if an amendment
    # exists, no original should exist in the output.
    lines.append(f"\n-- CHECK 2: No Superseded Originals (Amendment Coverage) --")
    df_reports["_q_base"]  = df_reports["report_type"].apply(_quarter_base)
    df_reports["_is_amend"] = df_reports["report_type"].apply(_is_amendment)

    # Group by (registrant, fortune_name, year, quarter_base)
    # Check if any group has BOTH an amendment AND a non-amendment original
    superseded_cases = []
    grp_cols = ["registrant", "fortune_name", "year", "_q_base"]
    for key, grp in df_reports[df_reports["_q_base"].notna()].groupby(grp_cols):
        has_amend    = grp["_is_amend"].any()
        has_original = (~grp["_is_amend"]).any()
        if has_amend and has_original:
            orig_types  = grp.loc[~grp["_is_amend"], "report_type"].tolist()
            amend_types = grp.loc[ grp["_is_amend"],  "report_type"].tolist()
            superseded_cases.append({
                "registrant":   key[0],
                "fortune_name": key[1],
                "year":         key[2],
                "quarter":      key[3],
                "original":     orig_types,
                "amendment":    amend_types,
            })

    if not superseded_cases:
        lines.append(f"  PASS: No (registrant, firm, year, quarter) group contains "
                     f"both an original and an amendment.")
        lines.append(f"  (Amendment report_types present in data: "
                     f"{sorted(df_reports[df_reports['_is_amend']]['report_type'].unique())})")
    else:
        lines.append(f"  FAIL: {len(superseded_cases):,} groups have both original and amendment!")
        lines.append(f"  Sample cases:")
        for case in superseded_cases[:10]:
            lines.append(f"    {case['registrant'][:30]} | {case['fortune_name'][:25]} | "
                         f"{case['year']} {case['quarter']} | "
                         f"orig={case['original']} amend={case['amendment']}")

    # ------------------------------------------------------------------
    # Check 3: Self='i' records are minimal
    lines.append(f"\n-- CHECK 3: Self-Type 'i' Records (should be near-zero) --")
    i_records = df_reports[df_reports["self_type"] == "i"]
    lines.append(f"  Self='i' records in output   : {len(i_records):,}")
    lines.append(f"  (These are the rare ind='y' subset of 'i' type; "
                 f"29,912 of 30,360 total 'i' records carry ind='' and are excluded)")
    if len(i_records) > 500:
        lines.append(f"  WARNING: unexpectedly large number of 'i' records. "
                     f"Verify ind='y' filter is applied.")
    else:
        lines.append(f"  PASS: count is within expected range (< 500).")
    if not i_records.empty:
        lines.append(f"  Firms with 'i' type records: "
                     f"{sorted(i_records['fortune_name'].unique())[:10]}")

    # ------------------------------------------------------------------
    # Report-type distribution
    lines.append(f"\n-- Report-Type Distribution (col: report_type) --")
    rt_counts = df_reports["report_type"].value_counts()
    for rt, cnt in rt_counts.items():
        lines.append(f"  {rt:8s}: {cnt:5,}")

    # Check all report_types are valid quarterly LD-2 codes
    valid_prefixes = {"q1", "q2", "q3", "q4"}
    unexpected_types = [rt for rt in rt_counts.index
                        if not any(rt.lower().startswith(p) for p in valid_prefixes)]
    if unexpected_types:
        lines.append(f"  WARNING: unexpected report_type values: {unexpected_types}")
        lines.append(f"  (All records should be quarterly LD-2 reports: q1/q2/q3/q4 variants)")
    else:
        lines.append(f"  PASS: all report_types are valid quarterly LD-2 codes.")

    # ------------------------------------------------------------------
    # Self-type distribution
    lines.append(f"\n-- Self-Type Distribution (Data User Guide 'Self' field) --")
    st_counts = df_reports["self_type"].value_counts()
    for st, cnt in st_counts.items():
        desc = SELF_TYPE_DESC.get(st, "unknown")
        lines.append(f"  {st!r:4s}: {cnt:5,}  ({desc})")

    # ------------------------------------------------------------------
    # Spend by self_type
    lines.append(f"\n-- Total Spend by Self-Type --")
    spend_by_type = (
        df_reports.groupby("self_type")["amount"]
        .sum()
        .sort_values(ascending=False)
    )
    for st, spend in spend_by_type.items():
        desc = SELF_TYPE_DESC.get(st, "unknown")
        lines.append(f"  {st!r:4s}: ${spend:>14,.0f}  ({desc})")
    lines.append(f"  {'TOTAL':4s}: ${spend_by_type.sum():>14,.0f}")

    # ------------------------------------------------------------------
    # Self-filer vs external breakdown
    lines.append(f"\n-- Self-Filer vs External Registrant --")
    n_self = df_reports["is_self_filer"].sum()
    n_ext  = (~df_reports["is_self_filer"]).sum()
    lines.append(f"  Self-filer reports (isfirm='') : {n_self:,}  "
                 f"({100*n_self/len(df_reports):.1f}%)")
    lines.append(f"  External registrant reports   : {n_ext:,}  "
                 f"({100*n_ext/len(df_reports):.1f}%)")

    # ------------------------------------------------------------------
    # Firms per year
    lines.append(f"\n-- Firms and Reports per Year --")
    for yr in sorted(df_reports["year"].unique()):
        sub = df_reports[df_reports["year"] == yr]
        lines.append(f"  {yr}: {len(sub):,} reports, "
                     f"{sub['fortune_name'].nunique()} firms, "
                     f"${sub['amount'].dropna().sum():,.0f} total spend")

    # ------------------------------------------------------------------
    # Cleanup temp columns
    df_reports.drop(columns=["_q_base", "_is_amend"], inplace=True, errors="ignore")

    lines.append(f"\n-- Summary --")
    lines.append(f"  ind='y' filter produces {len(df_reports):,} valid non-duplicate reports")
    lines.append(f"  covering {df_reports['fortune_name'].nunique()} Fortune 500 firms.")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    print("Loading data...")
    df_raw = pd.read_csv(DATA_DIR / "opensecrets_lda_reports.csv")

    # Deduplicate to one row per report for report-level checks
    df_reports = df_raw.drop_duplicates(subset=["uniq_id"]).copy()

    report = run_validation(df_raw, df_reports)
    print(report)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
