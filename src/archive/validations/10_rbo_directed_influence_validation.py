"""
Validation suite for rbo_directed_influence.py outputs.

Checks:
  1.  Edge CSV schema — all required columns present, correct dtypes
  2.  RBO weight range — all weights in (0, 1]
  3.  Bidirectional structure — every edge (A→B) has antiparallel (B→A); weights sum to rbo
  4.  net_temporal arithmetic — net_temporal == source_firsts - target_firsts
  5.  Weight formula — weight ≈ ((source_firsts + tie_count/2) / shared_bills) × rbo
  6.  Balanced edge weights — net_temporal=0 edges have weight ≈ rbo/2
  7.  Node net_influence arithmetic — matches Σ(source_firsts − target_firsts) over out-edges
  8.  Node color consistency — hex color matches sign of net_strength
  9.  Ranked-list CSV integrity — contiguous ranks, no gaps or duplicates per firm
  10. RBO spot-check — recompute RBO for 20 random pairs; rbo column values match
  11. Node net_strength arithmetic — matches Σ_j [rbo × net_temporal] over out-edges

Run from src/ directory:
  python validations/10_rbo_directed_influence_validation.py
"""

import sys
import os
import pandas as pd
import networkx as nx
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_DIR, ROOT
from utils.similarity import rbo_score

OUTPUT_PATH = Path(__file__).resolve().parents[3] / "outputs" / "archive" / "validation" / "10_rbo_directed_influence_validation.txt"

class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for s in self.streams: s.write(text)
    def flush(self):
        for s in self.streams: s.flush()

EDGES_CSV  = DATA_DIR / "rbo_directed_influence.csv"
RANKED_CSV = DATA_DIR / "ranked_bill_lists.csv"
GML_PATH   = ROOT / "visualizations" / "gml" / "rbo_directed_influence.gml"

PASS = "  PASS"
FAIL = "  FAIL"

EDGE_COLS = [
    "source", "target", "weight", "rbo", "source_firsts", "target_firsts",
    "tie_count", "shared_bills", "net_temporal",
]
RANKED_COLS = ["company", "rank", "bill_number", "total_amount", "budget_fraction"]

# Hex color constants matching rbo_directed_influence.py build_graph()
COLOR_GREEN = "#2ECC71"
COLOR_RED   = "#E74C3C"
COLOR_GRAY  = "#95A5A6"


def _load():
    """Load edge CSV, ranked CSV, and GML; raise clearly if any are missing."""
    edges  = pd.read_csv(EDGES_CSV)
    ranked = pd.read_csv(RANKED_CSV)
    G      = nx.read_gml(str(GML_PATH))
    return edges, ranked, G


def check_edge_schema(edges):
    """Check 1: edge CSV has all required columns with expected dtypes."""
    missing = [c for c in EDGE_COLS if c not in edges.columns]
    if missing:
        return False, f"missing columns: {missing}"
    if not pd.api.types.is_float_dtype(edges["weight"]):
        return False, "weight column is not float"
    if not pd.api.types.is_float_dtype(edges["rbo"]):
        return False, "rbo column is not float"
    for col in ("source_firsts", "target_firsts", "tie_count", "shared_bills", "net_temporal"):
        if not pd.api.types.is_integer_dtype(edges[col]):
            return False, f"{col} is not integer dtype"
    if "balanced" in edges.columns:
        return False, "stale 'balanced' column present — regenerate from rbo_directed_influence.py"
    return True, f"{len(edges):,} rows, {len(edges.columns)} columns"


def check_weight_range(edges):
    """Check 2: all RBO weights in (0, 1]."""
    bad = edges[(edges["weight"] <= 0) | (edges["weight"] > 1)]
    if not bad.empty:
        return False, f"{len(bad)} weights outside (0,1]: {bad['weight'].describe()}"
    return True, f"min={edges['weight'].min():.6f}  max={edges['weight'].max():.6f}"


def check_bidirectional_structure(edges):
    """Check 3: every edge (A→B) has antiparallel (B→A); within each pair, weights sum to rbo."""
    edge_set = set(zip(edges["source"], edges["target"]))
    missing_antiparallel = [
        (u, v) for u, v in edge_set if (v, u) not in edge_set
    ]
    if missing_antiparallel:
        return False, f"{len(missing_antiparallel)} edges missing antiparallel: {missing_antiparallel[:3]}"

    # For each canonical pair (source < target), check weight_ab + weight_ba ≈ rbo
    canon = edges[edges["source"] < edges["target"]].copy()
    anti  = edges[edges["source"] > edges["target"]].copy()
    anti["_src"] = anti["target"]; anti["_tgt"] = anti["source"]
    anti = anti.rename(columns={"weight": "weight_ba"})[["_src", "_tgt", "weight_ba"]]
    merged = canon.merge(anti, left_on=["source","target"], right_on=["_src","_tgt"])
    merged["weight_sum"] = merged["weight"] + merged["weight_ba"]
    bad = merged[abs(merged["weight_sum"] - merged["rbo"]) > 1e-4]
    if not bad.empty:
        return False, f"{len(bad)} pairs where weight_ab + weight_ba ≠ rbo (tolerance 1e-4)"
    return True, f"{len(canon):,} pairs; all bidirectional, weights sum to rbo"


def check_net_temporal(edges):
    """Check 4: net_temporal == source_firsts - target_firsts for every row."""
    expected = edges["source_firsts"] - edges["target_firsts"]
    mismatch = (edges["net_temporal"] != expected).sum()
    if mismatch:
        return False, f"{mismatch} rows where net_temporal != source_firsts - target_firsts"
    return True, "all rows consistent"


def check_weight_formula(edges):
    """Check 5: weight ≈ ((source_firsts + tie_count/2) / shared_bills) × rbo for all rows."""
    expected = ((edges["source_firsts"] + edges["tie_count"] / 2) / edges["shared_bills"]) * edges["rbo"]
    delta    = (edges["weight"] - expected).abs()
    bad      = edges[delta > 1e-4]
    if not bad.empty:
        return False, f"{len(bad)} rows where weight deviates from formula by > 1e-4"
    return True, f"all {len(edges):,} rows satisfy weight formula within 1e-4"


def check_balanced_weight_half(edges):
    """Check 6: balanced edges (net_temporal == 0) have weight ≈ rbo / 2."""
    bal = edges[edges["net_temporal"] == 0]
    if bal.empty:
        return True, "no balanced edges (vacuously true)"
    expected = bal["rbo"] / 2
    bad = bal[(bal["weight"] - expected).abs() > 1e-4]
    if not bad.empty:
        return False, f"{len(bad)} balanced edges where weight ≠ rbo/2 (tolerance 1e-4)"
    return True, f"{len(bal):,} balanced edges all have weight ≈ rbo/2"


def check_node_net_influence(G):
    """Check 7: stored net_influence == Σ(source_firsts − target_firsts) over out-edges."""
    errors = []
    for node in G.nodes():
        # Out-edges only: each pair appears once as source, so no double-counting
        out_sf = sum(d["source_firsts"] for _, _, d in G.out_edges(node, data=True))
        out_tf = sum(d["target_firsts"] for _, _, d in G.out_edges(node, data=True))
        expected = int(out_sf - out_tf)
        stored   = G.nodes[node].get("net_influence")
        if stored != expected:
            errors.append((node, stored, expected))
    if errors:
        return False, f"{len(errors)} nodes with wrong net_influence: {errors[:3]}"
    return True, f"all {G.number_of_nodes()} nodes consistent"


def check_node_colors(G):
    """Check 8: node hex color matches sign of net_strength (green/red/gray)."""
    errors = []
    for node in G.nodes():
        ns    = G.nodes[node].get("net_strength", 0)
        color = G.nodes[node].get("color", "")
        expected = COLOR_GREEN if ns > 0 else (COLOR_RED if ns < 0 else COLOR_GRAY)
        if color != expected:
            errors.append((node, color, expected, ns))
    if errors:
        return False, f"{len(errors)} nodes with wrong color: {errors[:3]}"
    return True, f"all {G.number_of_nodes()} nodes correctly colored"


def check_ranked_csv(ranked):
    """Check 9: ranked CSV schema OK; each firm has contiguous ranks starting at 1."""
    missing = [c for c in RANKED_COLS if c not in ranked.columns]
    if missing:
        return False, f"missing columns: {missing}"
    errors = []
    for firm, grp in ranked.groupby("company"):
        ranks    = sorted(grp["rank"].tolist())
        expected = list(range(1, len(ranks) + 1))
        if ranks != expected:
            errors.append((firm, ranks[:5]))
    if errors:
        return False, f"{len(errors)} firms with non-contiguous ranks: {errors[:2]}"
    n_firms = ranked["company"].nunique()
    return True, (f"{n_firms} firms, {len(ranked):,} rows; "
                  f"max rank per firm: {ranked.groupby('company')['rank'].max().max()}")


def check_rbo_spot(edges, ranked_csv, n_samples=20, seed=42):
    """Check 10: recompute RBO for n_samples random pairs; rbo column values match."""
    ranked_dict = (
        ranked_csv.sort_values(["company", "rank"])
        .groupby("company")["bill_number"]
        .apply(list)
        .to_dict()
    )
    # Sample from decisive edges (one direction per pair: net_temporal > 0)
    decisive = edges[edges["net_temporal"] > 0].copy()
    if len(decisive) == 0:
        return False, "no decisive edges to sample"
    sample = decisive.sample(min(n_samples, len(decisive)), random_state=seed)
    rbo_errors = []
    w_errors   = []
    for _, row in sample.iterrows():
        src, tgt = row["source"], row["target"]
        if src not in ranked_dict or tgt not in ranked_dict:
            continue
        recomputed = rbo_score(ranked_dict[src], ranked_dict[tgt], p=0.85)
        if abs(recomputed - row["rbo"]) > 1e-4:
            rbo_errors.append((src, tgt, row["rbo"], recomputed))
        # Verify weight formula
        n_shared = row["shared_bills"]
        expected_w = ((row["source_firsts"] + row["tie_count"] / 2) / n_shared) * row["rbo"]
        if abs(row["weight"] - expected_w) > 1e-4:
            w_errors.append((src, tgt, row["weight"], round(expected_w, 6)))
    if rbo_errors:
        return False, f"{len(rbo_errors)} RBO column mismatches: {rbo_errors[:3]}"
    if w_errors:
        return False, f"{len(w_errors)} weight formula mismatches in spot-check: {w_errors[:3]}"
    return True, f"{len(sample)} random decisive pairs verified (rbo column + weight formula)"


def check_net_strength(G):
    """Check 11: stored net_strength == Σ_j [rbo × net_temporal] over out-edges."""
    errors = []
    for node in G.nodes():
        expected = round(
            sum(d["rbo"] * d["net_temporal"] for _, _, d in G.out_edges(node, data=True)),
            4,
        )
        stored = G.nodes[node].get("net_strength")
        if stored is None:
            errors.append((node, "missing", expected))
        elif abs(stored - expected) > 1e-3:
            errors.append((node, stored, expected))
    if errors:
        return False, f"{len(errors)} nodes with wrong net_strength: {errors[:3]}"
    return True, f"all {G.number_of_nodes()} nodes consistent"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _orig_stdout = sys.stdout
    _f = open(OUTPUT_PATH, "w")
    sys.stdout = _Tee(_orig_stdout, _f)

    try:
        print("=" * 62)
        print("  10 — Congress Influence Network Validation")
        print("=" * 62)

        try:
            edges, ranked, G = _load()
        except FileNotFoundError as e:
            print(f"\n  ERROR: {e}")
            print("  Run rbo_directed_influence.py first.")
            sys.exit(1)

        checks = [
            ("Edge CSV schema",              lambda: check_edge_schema(edges)),
            ("RBO weight range",             lambda: check_weight_range(edges)),
            ("Bidirectional structure",      lambda: check_bidirectional_structure(edges)),
            ("net_temporal arithmetic",      lambda: check_net_temporal(edges)),
            ("Weight formula",               lambda: check_weight_formula(edges)),
            ("Balanced edge weights",        lambda: check_balanced_weight_half(edges)),
            ("Node net_influence math",      lambda: check_node_net_influence(G)),
            ("Node color consistency",       lambda: check_node_colors(G)),
            ("Ranked-list CSV integrity",    lambda: check_ranked_csv(ranked)),
            ("RBO spot-check (n=20)",        lambda: check_rbo_spot(edges, ranked)),
            ("Node net_strength math",       lambda: check_net_strength(G)),
        ]

        passed = 0
        for i, (name, fn) in enumerate(checks, start=1):
            ok, detail = fn()
            status = PASS if ok else FAIL
            print(f"\n  Check {i:2d}: {name}")
            print(f"  {status}  —  {detail}")
            if ok:
                passed += 1

        print(f"\n{'='*62}")
        print(f"  {passed} / {len(checks)} checks passed")
        if passed < len(checks):
            print("  *** Some checks FAILED — review output above ***")
        print("=" * 62)

    finally:
        sys.stdout = _orig_stdout
        _f.close()


if __name__ == "__main__":
    main()
