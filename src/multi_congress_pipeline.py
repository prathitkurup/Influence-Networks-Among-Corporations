"""
Multi-congress extraction + RBO directed influence pipeline (111th-117th Congress).

Runs opensecrets extraction then RBO directed influence for each congress in
CONGRESSES, writing outputs to data/congress/{num}/.

For each congress, also writes GML and PNG to:
  visualizations/gml/rbo_directed_influence_{num}.gml
  visualizations/png/rbo_directed_influence_{num}.png

NOTE: Coverage depends on manual_opensecrets_name_mapping.json including CRP
name variants for each congress era. Expand the mapping manually before running
new congresses.

Run: python multi_congress_pipeline.py
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, MANUAL_OPENSECRETS_NAME_MAPPING, MAX_BILL_DF
from opensecrets_extraction import (
    _iter_file, load_lookup, resolve_fortune_name,
    load_lobbyist_map, load_issue_map, load_issue_codes_map,
    load_bill_map, _count_ind_y_rows,
    _LOB_UNIQ_ID, _LOB_REGISTRANT_RAW, _LOB_REGISTRANT, _LOB_ISFIRM,
    _LOB_CLIENT_RAW, _LOB_CLIENT, _LOB_ULTORG, _LOB_AMOUNT, _LOB_SELF,
    _LOB_IND, _LOB_YEAR, _LOB_REPORT_TYPE, _MIN_LOB_COLS,
)
from rbo_directed_influence import build_edges, build_graph, print_stats, export_ranked_lists, write_gml, compute_total_spend
from utils.visualization import plot_directed_circular
from utils.data_loading import load_bills_data, assign_quarters, congress_year_range
from utils.filtering import filter_bills_by_prevalence
from utils.similarity import aggregate_per_firm_bill, compute_zero_budget_fracs, build_ranked_lists

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONGRESSES    = [111, 112, 113, 114, 115, 116, 117]  # congresses to run
SKIP_EXISTING = False  # skip congress if rbo_directed_influence.csv already exists
RBO_P         = 0.85   # matches 116th calibration (§18)
TOP_BILLS     = 30

OPENSECRETS_DIR = DATA_DIR / "OpenSecrets"
LOBBYING_FILE   = OPENSECRETS_DIR / "lob_lobbying.txt"
LOBBYIST_FILE   = OPENSECRETS_DIR / "lob_lobbyist.txt"
ISSUE_FILE      = OPENSECRETS_DIR / "lob_issue.txt"
BILLS_FILE      = OPENSECRETS_DIR / "lob_bills.txt"


# ---------------------------------------------------------------------------
# Congress utilities
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _load_lobbying_rows(lookup, mapping, congress_num):
    """Parse lob_lobbying.txt, filter ind='y' for congress years, map to Fortune 500.

    Local version of opensecrets_extraction.load_lobbying_f500 parameterized by congress.
    """
    start, end = congress_year_range(congress_num)
    years = {str(start), str(end)}
    rows  = []
    skipped_short = skipped_non_ind = 0

    for row in _iter_file(LOBBYING_FILE):
        if len(row) < _MIN_LOB_COLS:
            skipped_short += 1
            continue
        if row[_LOB_YEAR] not in years:
            continue
        if row[_LOB_IND] != "y":
            skipped_non_ind += 1
            continue
        fortune_name = resolve_fortune_name(row[_LOB_ULTORG], row[_LOB_CLIENT], lookup)
        if fortune_name is None:
            continue
        try:
            amount = float(row[_LOB_AMOUNT]) if row[_LOB_AMOUNT] else None
        except ValueError:
            amount = None
        rows.append({
            "uniq_id":        row[_LOB_UNIQ_ID],
            "fortune_name":   fortune_name,
            "client":         row[_LOB_CLIENT],
            "client_raw":     row[_LOB_CLIENT_RAW],
            "ultorg":         row[_LOB_ULTORG],
            "registrant":     row[_LOB_REGISTRANT],
            "registrant_raw": row[_LOB_REGISTRANT_RAW],
            "amount":         amount,
            "is_self_filer":  (row[_LOB_ISFIRM] == ""),
            "self_type":      row[_LOB_SELF],
            "year":           int(row[_LOB_YEAR]),
            "congress":       congress_num,
            "report_type":    row[_LOB_REPORT_TYPE],
        })

    if skipped_short:
        print(f"  [lobbying] Skipped {skipped_short:,} short rows")

    df       = pd.DataFrame(rows)
    n_ind_y  = _count_ind_y_rows(LOBBYING_FILE, years)
    n_match  = len(df)
    n_firms  = df["fortune_name"].nunique() if not df.empty else 0
    print(f"  ind=y reports {start}-{end}        : {n_ind_y:,}")
    print(f"  Matched to name mapping       : {n_match:,}  "
          f"({100 * n_match / max(n_ind_y, 1):.1f}%)")
    print(f"  Firms matched                 : {n_firms} / {len(mapping)}")
    missing = sorted(set(mapping) - set(df["fortune_name"].unique())) if not df.empty else list(mapping)
    if missing:
        preview = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        print(f"  Not matched ({len(missing)})            : {preview}")
    return df


def run_extraction(congress_num, out_dir, lookup, mapping):
    """Run full extraction for one congress; write reports + issues CSVs to out_dir."""
    start, end = congress_year_range(congress_num)
    years = {str(start), str(end)}

    df = _load_lobbying_rows(lookup, mapping, congress_num)
    if df.empty:
        print("  No Fortune 500 ind=y reports — check name mapping coverage.")
        return None

    # Filter to active (≥1 named lobbyist) reports
    target_ids   = set(df["uniq_id"])
    lobbyist_map = load_lobbyist_map(LOBBYIST_FILE, target_ids, years)
    n_pre        = len(df)
    df           = df[df["uniq_id"].isin(lobbyist_map)].reset_index(drop=True)
    print(f"  Active reports (≥1 lobbyist)  : {len(df):,}  (dropped {n_pre - len(df):,})")
    df["lobbyists"] = df["uniq_id"].map(lambda uid: "|".join(lobbyist_map.get(uid, [])))

    target_ids        = set(df["uniq_id"])
    target_uids_upper = {uid.upper() for uid in target_ids}

    # Bills: issue_map → bill_map → uid → sorted bill list
    uid_to_issue_ids = load_issue_map(ISSUE_FILE, target_uids_upper, years)
    all_issue_ids    = {iid for ids in uid_to_issue_ids.values() for iid in ids}
    issue_to_bills   = load_bill_map(BILLS_FILE, all_issue_ids)
    uid_to_bills = {
        uid: sorted({b for iid in iids for b in issue_to_bills.get(iid, set())})
        for uid, iids in uid_to_issue_ids.items()
        if any(issue_to_bills.get(iid) for iid in iids)
    }

    # Expand to one row per (report, bill)
    df["_uid_upper"] = df["uniq_id"].str.upper()
    bill_records = [
        {"_uid_upper": uid, "bill_number": bill, "n_bills": len(bills)}
        for uid, bills in uid_to_bills.items()
        for bill in bills
    ]
    if bill_records:
        expanded = df.merge(pd.DataFrame(bill_records), on="_uid_upper", how="left")
    else:
        df["bill_number"] = None
        df["n_bills"]     = None
        expanded = df
    expanded = expanded.drop(columns=["_uid_upper"])
    expanded["amount_allocated"] = expanded.apply(
        lambda r: (r["amount"] / r["n_bills"])
        if pd.notna(r.get("n_bills")) and r.get("n_bills", 0) > 0 else r["amount"],
        axis=1,
    )
    expanded = expanded.drop(columns=["n_bills"])
    output_cols = [
        "uniq_id", "fortune_name", "bill_number",
        "client", "client_raw", "ultorg", "registrant", "registrant_raw",
        "amount", "amount_allocated", "is_self_filer", "self_type",
        "year", "congress", "report_type", "lobbyists",
    ]
    expanded = expanded[output_cols]
    expanded.to_csv(out_dir / "opensecrets_lda_reports.csv", index=False)
    print(f"  Reports CSV written            : {len(expanded):,} rows")

    # Issue codes
    uid_to_codes = load_issue_codes_map(ISSUE_FILE, target_uids_upper, years)
    report_base  = (df[["uniq_id", "fortune_name", "amount", "year", "congress"]]
                    .drop_duplicates(subset=["uniq_id"]).copy())
    report_base["_uid_upper"] = report_base["uniq_id"].str.upper()
    issue_records = [
        {"_uid_upper": uid, "issue_code": code, "n_codes": len(codes)}
        for uid, codes in uid_to_codes.items()
        for code in codes
    ]
    if issue_records:
        issues_exp = report_base.merge(pd.DataFrame(issue_records), on="_uid_upper", how="inner")
        issues_exp["amount_allocated"] = issues_exp["amount"] / issues_exp["n_codes"]
        issues_exp = issues_exp.drop(columns=["_uid_upper", "n_codes"])
        issue_cols = ["uniq_id", "fortune_name", "issue_code",
                      "amount", "amount_allocated", "year", "congress"]
        issues_exp = issues_exp[issue_cols]
    else:
        issues_exp = pd.DataFrame(
            columns=["uniq_id", "fortune_name", "issue_code",
                     "amount", "amount_allocated", "year", "congress"]
        )
    issues_exp.to_csv(out_dir / "opensecrets_lda_issues.csv", index=False)
    print(f"  Issues CSV written             : {len(issues_exp):,} rows")
    return expanded


# ---------------------------------------------------------------------------
# RBO directed influence
# ---------------------------------------------------------------------------

def run_rbo_influence(congress_num, out_dir):
    """Run RBO directed influence for one congress; write edges + node attrs CSVs."""
    reports_path = out_dir / "opensecrets_lda_reports.csv"
    if not reports_path.exists():
        print(f"  Reports CSV missing for congress {congress_num}; skipping RBO.")
        return None

    df_raw = load_bills_data(reports_path)
    df_raw = assign_quarters(df_raw, congress_num)
    print(f"  {len(df_raw):,} rows  |  "
          f"{df_raw['fortune_name'].nunique()} firms  |  "
          f"{df_raw['bill_number'].nunique()} bills")

    df_agg = aggregate_per_firm_bill(df_raw)
    df_agg = compute_zero_budget_fracs(df_agg)
    if MAX_BILL_DF is not None:
        df_agg = filter_bills_by_prevalence(df_agg, MAX_BILL_DF, unit_col="bill_number")

    ranked = build_ranked_lists(df_agg, top_bills=TOP_BILLS)
    export_ranked_lists(ranked, df_agg, out_dir / "ranked_bill_lists.csv")
    print(f"  Firms with ranked lists        : {len(ranked):,}")

    # Global first-quarter per (firm, bill) — uses unfiltered raw data
    bill_first = (
        df_raw.groupby(["fortune_name", "bill_number"])["quarter"]
        .min().to_dict()
    )

    edges_df  = build_edges(ranked, bill_first, p=RBO_P)
    spend_map = compute_total_spend(df_raw)
    G         = build_graph(edges_df, ranked_firms=set(ranked.keys()), spend_map=spend_map)
    print_stats(edges_df, G)

    edges_df.to_csv(out_dir / "rbo_directed_influence.csv", index=False)
    print(f"  Edges CSV written              : {len(edges_df):,} edges")

    # Node attributes for cross-congressional firm-level analysis
    node_records = [
        {
            "firm":          n,
            "net_influence": G.nodes[n]["net_influence"],
            "net_strength":  G.nodes[n]["net_strength"],
            "total_firsts":  G.nodes[n]["total_firsts"],
            "total_losses":  G.nodes[n]["total_losses"],
        }
        for n in G.nodes()
    ]
    (pd.DataFrame(node_records)
     .sort_values("net_strength", ascending=False)
     .reset_index(drop=True)
     .to_csv(out_dir / "node_attributes.csv", index=False))
    print(f"  Node attrs CSV written         : {len(node_records)} nodes")

    # GML + PNG visualization
    from config import ROOT
    gml_dir = ROOT / "visualizations" / "gml"
    png_dir = ROOT / "visualizations" / "png"
    gml_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    gml_path = gml_dir / f"rbo_directed_influence_{congress_num}.gml"
    png_path = png_dir / f"rbo_directed_influence_{congress_num}.png"
    write_gml(G, str(gml_path))
    plot_directed_circular(
        G,
        title=f"RBO Directed Influence — {congress_num}th Congress",
        path=str(png_path),
        top_k=20,
    )
    print(f"  PNG written                    : {png_path.name}")
    return edges_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Multi-Congress Pipeline  (extraction + RBO directed influence)")
    print("NOTE: Early-congress coverage depends on name-mapping completeness.")
    print("=" * 70)

    mapping, lookup = load_lookup()
    print(f"Lookup entries: {len(lookup):,}  ({len(mapping)} firms in mapping)\n")

    for congress_num in CONGRESSES:
        start, end = congress_year_range(congress_num)
        out_dir    = DATA_DIR / "congress" / str(congress_num)
        out_dir.mkdir(parents=True, exist_ok=True)

        rbo_out = out_dir / "rbo_directed_influence.csv"
        if SKIP_EXISTING and rbo_out.exists():
            print(f"Congress {congress_num} ({start}-{end}): output exists — skipping "
                  f"(set SKIP_EXISTING=False to rerun)")
            continue

        print(f"\n{'='*60}")
        print(f"Congress {congress_num}  ({start}–{end})")
        print(f"{'='*60}")

        df = run_extraction(congress_num, out_dir, lookup, mapping)
        if df is None:
            continue

        print(f"\n  Running RBO directed influence...")
        run_rbo_influence(congress_num, out_dir)

    print(f"\nDone. Per-congress outputs in: data/congress/{{num}}/")


if __name__ == "__main__":
    main()
