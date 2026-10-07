"""
OpenSecrets LDA extraction pipeline for 116th Congress (2019-2020).

Filters to ind='y' (valid, countable reports per OpenSecrets Data User Guide p.13) then retains only
reports with ≥1 named lobbyist. Produces opensecrets_lda_reports.csv (one row per report×bill),
opensecrets_lda_issues.csv (one row per report×issue_code), and lobbyist_client_116_opensecrets.csv.

Run: python opensecrets_extraction.py
"""

import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR,
    MANUAL_OPENSECRETS_NAME_MAPPING,
    OPENSECRETS_OUTPUT_CSV,
    OPENSECRETS_ISSUES_CSV,
    OPENSECRETS_LOBBYIST_CLIENT_CSV,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OPENSECRETS_DIR = DATA_DIR / "OpenSecrets"
LOBBYING_FILE   = OPENSECRETS_DIR / "lob_lobbying.txt"
LOBBYIST_FILE   = OPENSECRETS_DIR / "lob_lobbyist.txt"
ISSUE_FILE      = OPENSECRETS_DIR / "lob_issue.txt"
BILLS_FILE      = OPENSECRETS_DIR / "lob_bills.txt"

CONGRESS_YEARS = {"2019", "2020"}
CONGRESS_NUM   = 116

# Self-type code descriptions (Data User Guide p.29, 'Self' field).
# These describe the organizational registrant/client relationship, NOT form type.
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

# ---------------------------------------------------------------------------
# Column positions — see OpenSecrets Data User Guide p.29 for full schema.
# lob_lobbying.txt: 0=uniq_id, 3=isfirm, 6=ultorg, 7=amount, 10=self, 13=ind, 14=year, 15=report_type
# lob_lobbyist.txt: 0=uniq_id, 2=lobbyist, 3=lobbyist_id, 4=year
# lob_issue.txt:    0=issue_id, 1=uniq_id, 2=issue_code, 5=year
# lob_bills.txt:    1=issue_id, 3=bill_number
# ---------------------------------------------------------------------------
_LOB_UNIQ_ID        = 0
_LOB_REGISTRANT_RAW = 1
_LOB_REGISTRANT     = 2
_LOB_ISFIRM         = 3   # '' = self-filer (registrant is client); 'y' = external K-street firm
_LOB_CLIENT_RAW     = 4
_LOB_CLIENT         = 5
_LOB_ULTORG         = 6
_LOB_AMOUNT         = 7
_LOB_SELF           = 10  # Organizational relationship code (see SELF_TYPE_DESC above)
_LOB_IND            = 13  # Primary validity filter: 'y' = count; '' = superseded or double-count
_LOB_YEAR           = 14
_LOB_REPORT_TYPE    = 15
_MIN_LOB_COLS       = _LOB_REPORT_TYPE + 1  # 16

_LST_UNIQ_ID        = 0
_LST_LOBBYIST       = 2
_LST_LOBBYIST_ID    = 3
_LST_YEAR           = 4
_MIN_LST_COLS       = _LST_YEAR + 1   # 5

_ISS_ISSUE_ID       = 0
_ISS_UNIQ_ID        = 1
_ISS_ISSUE_CODE     = 2
_ISS_YEAR           = 5
_MIN_ISS_COLS       = _ISS_YEAR + 1   # 6

_BILLS_ISSUE_ID     = 1
_BILLS_BILL_NUM     = 3
_MIN_BILLS_COLS     = _BILLS_BILL_NUM + 1  # 4


# ---------------------------------------------------------------------------
# File parsing helper
# ---------------------------------------------------------------------------

def _iter_file(path):
    """Yield stripped field lists from an OpenSecrets pipe-delimited file."""
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        reader = csv.reader(f, quotechar="|", delimiter=",", doublequote=False)
        for row in reader:
            yield [field.strip() for field in row]


# ---------------------------------------------------------------------------
# Name mapping (Fortune 500 -> CRP name lookup)
# ---------------------------------------------------------------------------

def _norm(s):
    """Uppercase and collapse whitespace."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.upper().strip())


def load_lookup():
    """Load manual name mapping and return (mapping_dict, flat_lookup_dict).

    Manual mapping format: {canonical_name: [variation, ...]}
    Both the canonical name and each variation are added to the lookup table.
    """
    with open(MANUAL_OPENSECRETS_NAME_MAPPING) as f:
        mapping = json.load(f)
    lookup = {}
    for canonical, variations in mapping.items():
        lookup[_norm(canonical)] = canonical
        for name in variations:
            lookup[_norm(name)] = canonical
    return mapping, lookup


def resolve_fortune_name(ultorg, client, lookup):
    """Try ultorg first (parent org) to catch subsidiaries, then fall back to client."""
    key = _norm(ultorg)
    if key and key in lookup:
        return lookup[key]
    return lookup.get(_norm(client))


# ---------------------------------------------------------------------------
# Parse lob_lobbying.txt -> Fortune 500 valid report rows (ind='y')
# ---------------------------------------------------------------------------

def load_lobbying_f500(path, lookup, mapping, years=CONGRESS_YEARS):
    """Parse lob_lobbying.txt, filter to ind='y' records in target years, and map to Fortune 500 firm names."""
    rows = []
    skipped_short   = 0
    skipped_non_ind = 0

    for row in _iter_file(path):
        if len(row) < _MIN_LOB_COLS:
            skipped_short += 1
            continue
        if row[_LOB_YEAR] not in years:
            continue
        if row[_LOB_IND] != "y":
            skipped_non_ind += 1
            continue

        fortune_name = resolve_fortune_name(
            row[_LOB_ULTORG], row[_LOB_CLIENT], lookup
        )
        if fortune_name is None:
            continue

        # isfirm: '' = registrant is the filing client (self-filer)
        #         'y' = separate K-street registrant (external)
        is_self_filer = (row[_LOB_ISFIRM] == "")

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
            "is_self_filer":  is_self_filer,
            "self_type":      row[_LOB_SELF],
            "year":           int(row[_LOB_YEAR]),
            "congress":       CONGRESS_NUM,
            "report_type":    row[_LOB_REPORT_TYPE],
        })

    if skipped_short:
        print(f"  [lobbying] Skipped {skipped_short:,} short rows")
    if skipped_non_ind:
        print(f"  [lobbying] Skipped {skipped_non_ind:,} ind!=y rows "
              f"(superseded originals and double-count subsidiary records)")

    df = pd.DataFrame(rows)

    # Coverage summary
    n_ind_y_total = _count_ind_y_rows(path, years)
    n_matched     = len(df)
    n_firms       = df["fortune_name"].nunique() if not df.empty else 0
    n_f500        = len(mapping)
    print(f"\n-- OpenSecrets name-mapping coverage (116th Congress, ind=y only) --")
    print(f"  Valid ind=y reports in 2019-2020: {n_ind_y_total:,}")
    print(f"  Matched to Fortune 500          : {n_matched:,}  "
          f"({100 * n_matched / max(n_ind_y_total, 1):.1f}%)")
    print(f"  Fortune 500 firms matched       : {n_firms} / {n_f500}")

    missing = (sorted(set(mapping) - set(df["fortune_name"].unique()))
               if not df.empty else list(mapping))
    if missing:
        preview = ", ".join(missing[:12])
        tail    = " ..." if len(missing) > 12 else ""
        print(f"  Not matched ({len(missing)}): {preview}{tail}")

    return df


def _count_ind_y_rows(path, years):
    """Count ind='y' rows in target years (valid, non-superseded, non-duplicate)."""
    n = 0
    for row in _iter_file(path):
        if (len(row) >= _MIN_LOB_COLS
                and row[_LOB_YEAR] in years
                and row[_LOB_IND] == "y"):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Parse lob_lobbyist.txt
# ---------------------------------------------------------------------------

def load_lobbyist_map(path, target_uniq_ids, years=CONGRESS_YEARS):
    """Parse lob_lobbyist.txt; return {uniq_id: [sorted lobbyist names]} for target reports."""
    lobbyist_sets = {}
    skipped = 0

    for row in _iter_file(path):
        if len(row) < _MIN_LST_COLS:
            skipped += 1
            continue
        if row[_LST_YEAR] not in years:
            continue
        uid = row[_LST_UNIQ_ID]
        if uid not in target_uniq_ids:
            continue
        name = row[_LST_LOBBYIST].strip()
        if not name:
            continue
        lobbyist_sets.setdefault(uid, set()).add(name)

    if skipped:
        print(f"  [lobbyist] Skipped {skipped:,} short rows")

    return {uid: sorted(names) for uid, names in lobbyist_sets.items()}


def load_lobbyist_rows(path, target_uniq_ids, years=CONGRESS_YEARS):
    """Parse lob_lobbyist.txt; return DataFrame with uniq_id, lobbyist_id, lobbyist for target reports."""
    rows = []
    skipped = 0

    for row in _iter_file(path):
        if len(row) < _MIN_LST_COLS:
            skipped += 1
            continue
        if row[_LST_YEAR] not in years:
            continue
        uid = row[_LST_UNIQ_ID]
        if uid not in target_uniq_ids:
            continue
        lid  = row[_LST_LOBBYIST_ID].strip()
        name = row[_LST_LOBBYIST].strip()
        if not lid or not name:
            continue
        rows.append({"uniq_id": uid, "lobbyist_id": lid, "lobbyist": name})

    if skipped:
        print(f"  [lobbyist] Skipped {skipped:,} short rows")

    return (pd.DataFrame(rows) if rows
            else pd.DataFrame(columns=["uniq_id", "lobbyist_id", "lobbyist"]))


# ---------------------------------------------------------------------------
# Parse lob_issue.txt and lob_bills.txt -> bill numbers per report
# ---------------------------------------------------------------------------

def load_issue_map(path, target_uniq_ids_upper, years=CONGRESS_YEARS):
    """Parse lob_issue.txt; return {uniq_id_upper: set(issue_ids)} for target reports."""
    uid_to_issues = {}
    for row in _iter_file(path):
        if len(row) < _MIN_ISS_COLS:
            continue
        if row[_ISS_YEAR] not in years:
            continue
        uid = row[_ISS_UNIQ_ID].strip().upper()
        if uid not in target_uniq_ids_upper:
            continue
        uid_to_issues.setdefault(uid, set()).add(row[_ISS_ISSUE_ID].strip())
    return uid_to_issues


def load_issue_codes_map(path, target_uniq_ids_upper, years=CONGRESS_YEARS):
    """Parse lob_issue.txt; return {uniq_id_upper: set(issue_codes)} for target reports."""
    uid_to_codes = {}
    for row in _iter_file(path):
        if len(row) < _MIN_ISS_COLS:
            continue
        if row[_ISS_YEAR] not in years:
            continue
        uid = row[_ISS_UNIQ_ID].strip().upper()
        if uid not in target_uniq_ids_upper:
            continue
        code = row[_ISS_ISSUE_CODE].strip()
        if code:
            uid_to_codes.setdefault(uid, set()).add(code)
    return uid_to_codes


def load_bill_map(path, target_issue_ids):
    """Parse lob_bills.txt; return {issue_id: set(bill_numbers)} for target issue entries."""
    issue_to_bills = {}
    for row in _iter_file(path):
        if len(row) < _MIN_BILLS_COLS:
            continue
        iid  = row[_BILLS_ISSUE_ID].strip()
        if iid not in target_issue_ids:
            continue
        bill = row[_BILLS_BILL_NUM].strip()
        if bill:
            issue_to_bills.setdefault(iid, set()).add(bill)
    return issue_to_bills


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -- Load name mapping -----------------------------------------------
    print("Loading manual_opensecrets_name_mapping.json...")
    mapping, lookup = load_lookup()
    print(f"  Lookup entries: {len(lookup):,}  (covers {len(mapping)} Fortune 500 firms)")

    # -- Parse lobbying file -> F500 ind=y reports ------------------------
    print(f"\nLoading {LOBBYING_FILE.name} (ind=y only, 2019-2020)...")
    df = load_lobbying_f500(LOBBYING_FILE, lookup, mapping)
    if df.empty:
        print("No Fortune 500 ind=y reports found — check name-mapping coverage.")
        return

    print(f"\n-- Report statistics (ind=y only) --")
    print(f"  Total valid F500 reports      : {len(df):,}")
    print(f"  Self-filer (isfirm=='')       : {int(df['is_self_filer'].sum()):,}  "
          f"({100 * df['is_self_filer'].mean():.1f}%)")
    print(f"  External registrant           : {int((~df['is_self_filer']).sum()):,}  "
          f"({100 * (~df['is_self_filer']).mean():.1f}%)")
    print(f"  Unique Fortune 500 firms      : {df['fortune_name'].nunique()}")
    print(f"  Unique registrants            : {df['registrant'].nunique():,}")
    amt = df["amount"].dropna()
    print(f"  Reports with amount           : {amt.count():,}")
    print(f"  Total expenditure             : ${amt.sum():,.0f}")
    print(f"  Median per-report amount      : ${amt.median():,.0f}")
    print(f"\n  Self-type breakdown (Data User Guide 'Self' field):")
    for stype, cnt in df["self_type"].value_counts().items():
        desc = SELF_TYPE_DESC.get(stype, "unknown")
        print(f"    {stype!r:4s} : {cnt:5,}  ({desc})")

    # -- Parse lobbyist file -> per-report lobbyist lists -----------------
    print(f"\nLoading {LOBBYIST_FILE.name}...")
    target_ids = set(df["uniq_id"])
    lobbyist_map = load_lobbyist_map(LOBBYIST_FILE, target_ids)
    print(f"  Reports with >= 1 lobbyist    : {len(lobbyist_map):,} / {len(df):,}")
    all_names = {n for names in lobbyist_map.values() for n in names}
    print(f"  Unique lobbyist names         : {len(all_names):,}")

    # Filter to only reports with >= 1 lobbyist.
    # Reports without lobbyists at this stage are retainer/no-activity filings
    # (e.g. report_type q1n, q2n) that have no issue codes or bills — they are
    # structurally valid ind=y records but carry no substantive lobbying activity.
    n_pre_filter = len(df)
    df = df[df["uniq_id"].isin(lobbyist_map)].reset_index(drop=True)
    print(f"  Filtered to active reports    : {len(df):,}  "
          f"(dropped {n_pre_filter - len(df):,} no-lobbyist retainer/no-activity filings)")
    target_ids = set(df["uniq_id"])  # update for downstream bill/issue loading

    # -- Build deduplicated lobbyist-client pairs -------------------------
    print(f"\nBuilding lobbyist-client pairs (with IDs)...")
    lobbyist_rows = load_lobbyist_rows(LOBBYIST_FILE, target_ids)
    pairs = (
        lobbyist_rows
        .merge(df[["uniq_id", "fortune_name"]], on="uniq_id", how="inner")
        [["lobbyist_id", "lobbyist", "fortune_name"]]
        .drop_duplicates(subset=["lobbyist_id", "fortune_name"])
        .sort_values(["fortune_name", "lobbyist"])
        .reset_index(drop=True)
    )
    pairs.to_csv(OPENSECRETS_LOBBYIST_CLIENT_CSV, index=False)
    print(f"  Unique (lobbyist, firm) pairs : {len(pairs):,}")
    print(f"  Unique lobbyist IDs           : {pairs['lobbyist_id'].nunique():,}")
    print(f"  Fortune 500 firms covered     : {pairs['fortune_name'].nunique()}")
    print(f"  Pairs -> {OPENSECRETS_LOBBYIST_CLIENT_CSV}")

    # -- Attach lobbyist column (pipe-separated) --------------------------
    df["lobbyists"] = df["uniq_id"].map(
        lambda uid: "|".join(lobbyist_map.get(uid, []))
    )

    # -- Load bill data via lob_issue.txt -> lob_bills.txt ----------------
    print(f"\nLoading bill data ({ISSUE_FILE.name} -> {BILLS_FILE.name})...")
    target_uids_upper = {uid.upper() for uid in target_ids}
    uid_to_issue_ids  = load_issue_map(ISSUE_FILE, target_uids_upper)
    print(f"  F500 reports with issue entries  : {len(uid_to_issue_ids):,} / {len(df):,}")

    all_issue_ids  = {iid for ids in uid_to_issue_ids.values() for iid in ids}
    issue_to_bills = load_bill_map(BILLS_FILE, all_issue_ids)
    print(f"  Issue entries with linked bills  : {len(issue_to_bills):,}")

    # Build uid_upper -> sorted list of bill numbers
    uid_to_bills = {}
    for uid_upper, issue_ids in uid_to_issue_ids.items():
        bills = set()
        for iid in issue_ids:
            bills.update(issue_to_bills.get(iid, set()))
        if bills:
            uid_to_bills[uid_upper] = sorted(bills)

    n_with_bills = sum(1 for uid in df["uniq_id"]
                       if uid.upper() in uid_to_bills)
    total_bill_links = sum(len(b) for b in uid_to_bills.values())
    print(f"  F500 reports with >= 1 bill      : {n_with_bills:,} / {len(df):,}")
    print(f"  Total report-bill linkages       : {total_bill_links:,}")
    print(f"  Avg bills per report (with bills): "
          f"{total_bill_links / max(n_with_bills, 1):.1f}")

    # -- Expand to one row per (report, bill) ----------------------------
    # Reports with no bills keep one row with bill_number = NaN.
    df["_uid_upper"] = df["uniq_id"].str.upper()

    bill_records = [
        {"_uid_upper": uid_upper, "bill_number": bill, "n_bills": len(bills)}
        for uid_upper, bills in uid_to_bills.items()
        for bill in bills
    ]
    if bill_records:
        bill_df  = pd.DataFrame(bill_records)
        expanded = df.merge(bill_df, on="_uid_upper", how="left")
    else:
        df["bill_number"] = None
        df["n_bills"]     = None
        expanded = df

    expanded = expanded.drop(columns=["_uid_upper"])

    # amount_allocated: even split of report spend across bills.
    # Reports with no bills keep the full report amount.
    expanded["amount_allocated"] = expanded.apply(
        lambda r: (r["amount"] / r["n_bills"])
        if pd.notna(r["n_bills"]) and r["n_bills"] > 0
        else r["amount"],
        axis=1,
    )
    expanded = expanded.drop(columns=["n_bills"])

    print(f"\n  Expanded rows (report x bill)    : {len(expanded):,}")
    print(f"  Rows with a bill_number          : {expanded['bill_number'].notna().sum():,}")
    print(f"  Rows without bill (report-only)  : {expanded['bill_number'].isna().sum():,}")

    # -- Column ordering -------------------------------------------------
    output_cols = [
        "uniq_id", "fortune_name", "bill_number",
        "client", "client_raw", "ultorg",
        "registrant", "registrant_raw",
        "amount", "amount_allocated", "is_self_filer", "self_type",
        "year", "congress", "report_type", "lobbyists",
    ]
    expanded = expanded[output_cols]

    # -- Write bill-level output -----------------------------------------
    OPENSECRETS_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(OPENSECRETS_OUTPUT_CSV, index=False)
    print(f"\n  Output -> {OPENSECRETS_OUTPUT_CSV}")
    print(f"  Rows: {len(expanded):,}  |  Columns: {len(expanded.columns)}")

    # -- Sample rows -------------------------------------------------------
    print("\n-- Sample rows (first 5) --")
    with pd.option_context("display.max_colwidth", 35, "display.width", 200):
        sample_cols = ["fortune_name", "bill_number", "registrant",
                       "amount", "amount_allocated", "is_self_filer", "self_type", "year"]
        print(expanded[sample_cols].head().to_string(index=False))

    # -- Extract issue codes -> opensecrets_lda_issues.csv ---------------
    print(f"\nExtracting issue codes ({ISSUE_FILE.name})...")
    uid_to_codes = load_issue_codes_map(ISSUE_FILE, target_uids_upper)
    print(f"  F500 reports with issue codes    : {len(uid_to_codes):,} / {len(df):,}")
    n_total_codes = sum(len(v) for v in uid_to_codes.values())
    print(f"  Total report-issue linkages      : {n_total_codes:,}")
    print(f"  Unique issue codes               : "
          f"{len({c for codes in uid_to_codes.values() for c in codes})}")

    # Build a slim version of the report-level frame for the issue join
    report_base = df[["uniq_id", "fortune_name", "amount", "year", "congress"]].copy()
    report_base["_uid_upper"] = report_base["uniq_id"].str.upper()
    report_base = report_base.drop_duplicates(subset=["uniq_id"])

    issue_records = [
        {"_uid_upper": uid_upper, "issue_code": code, "n_codes": len(codes)}
        for uid_upper, codes in uid_to_codes.items()
        for code in codes
    ]
    if issue_records:
        issue_df    = pd.DataFrame(issue_records)
        issues_exp  = report_base.merge(issue_df, on="_uid_upper", how="inner")
        issues_exp["amount_allocated"] = issues_exp["amount"] / issues_exp["n_codes"]
        issues_exp  = issues_exp.drop(columns=["_uid_upper", "n_codes"])
        issue_cols  = ["uniq_id", "fortune_name", "issue_code",
                       "amount", "amount_allocated", "year", "congress"]
        issues_exp  = issues_exp[issue_cols]
    else:
        issues_exp = pd.DataFrame(
            columns=["uniq_id", "fortune_name", "issue_code",
                     "amount", "amount_allocated", "year", "congress"]
        )

    OPENSECRETS_ISSUES_CSV.parent.mkdir(parents=True, exist_ok=True)
    issues_exp.to_csv(OPENSECRETS_ISSUES_CSV, index=False)
    print(f"  Issues output -> {OPENSECRETS_ISSUES_CSV}")
    print(f"  Rows: {len(issues_exp):,}  |  Columns: {len(issues_exp.columns)}")


if __name__ == "__main__":
    main()
