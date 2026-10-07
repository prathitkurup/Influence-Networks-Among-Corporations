import pandas as pd
from pathlib import Path


def congress_year_range(congress_num):
    """Return (start_year, end_year) ints for a given congress number."""
    start = 2009 + 2 * (congress_num - 111)
    return start, start + 1


def assign_quarters(df, congress_num=116):
    """Add 'quarter' col: year1 Q1-4 → 1-4, year2 Q1-4 → 5-8. No-op if already present."""
    if "quarter" in df.columns:
        return df
    start, end = congress_year_range(congress_num)
    df = df.copy()
    base_q   = df["report_type"].str[1].astype(int)
    year_off = df["year"].map({start: 0, end: 4})
    df["quarter"] = base_q + year_off
    return df


def load_bills_data(path):
    """Load opensecrets_lda_reports.csv and validate expected columns.

    Drops rows with a null bill_number (reports with no linked bills) so every
    row in the returned DataFrame represents a confirmed (firm, bill) pair.
    """
    df = pd.read_csv(path)
    check_columns(df, ["fortune_name", "bill_number", "amount_allocated"], path)
    return df.dropna(subset=["bill_number"]).copy()


def load_issues_data(path):
    """Load opensecrets_lda_issues.csv and validate expected columns."""
    df = pd.read_csv(path)
    check_columns(df, ["fortune_name", "issue_code", "amount_allocated"], path)
    return df


def load_lobby_firm_data(path):
    """Load opensecrets_lda_reports.csv for lobby-firm affiliation analysis.

    Keeps only rows with an external registrant (is_self_filer == False).
    is_self_filer is a boolean in the OpenSecrets CSV (True = self-filer).
    """
    df = pd.read_csv(path)
    check_columns(df, ["fortune_name", "registrant"], path)
    df = df[~df["is_self_filer"].fillna(False)].dropna(subset=["registrant", "fortune_name"])
    return df


def check_columns(df, required, path):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{Path(path).name} is missing columns: {missing}")
