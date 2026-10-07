"""
Congress statistics and validation for 116th and 117th sessions.

Checks:
  1.  117th Congress quarter coverage — all 8 quarters present
  2.  116th Congress quarter coverage — all 8 quarters present

Statistics & Visualizations:
  3.  Agenda-setters per community (116th and 117th) — top firms by net_strength
      per community, with all centrality metrics from the bill affiliation GML
  4.  Bill affiliation GML centrality vs RBO agenda-setter alignment
  5.  Power-law evidence — in-degree, out-degree, and PageRank distributions;
      Kolmogorov-Smirnov test against log-normal and power-law fits
  6.  Degree distribution histograms (bill affiliation + RBO)
  7.  Centrality comparison bar charts (top 10 by each metric)
  8.  Community size distribution
  9.  Net influence distribution (116th vs 117th)
 10.  Quarter-level activity (117th) — firms and reports per quarter

Output:
  outputs/archive/validation/12_congress_statistics.txt  (text report, tee'd to stdout)
  outputs/archive/validation/figures/                    (PNG charts)

Run from project root or src/:
  python src/validations/12_congress_statistics.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from collections import Counter
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parents[1]
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))

from config import DATA_DIR, ROOT

CONGRESS_DIR  = DATA_DIR / "congress"
FIGURES_DIR   = ROOT / "outputs" / "archive" / "validation" / "figures"
OUTPUT_PATH   = ROOT / "outputs" / "archive" / "validation" / "12_congress_statistics.txt"
BILL_AFF_GML  = ROOT / "visualizations" / "gml" / "bill_affiliation_network.gml"
RBO_GML       = ROOT / "visualizations" / "gml" / "rbo_directed_influence.gml"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Congress session parameters
# 116th: 2019-2020 (quarters 1-8)
# 117th: 2021-2022 (quarters 1-8)
CONGRESS_PARAMS = {
    116: {"years": (2019, 2020)},
    117: {"years": (2021, 2022)},
}

TOP_K = 10       # firms per table
PASS  = "  PASS"
FAIL  = "  FAIL"

# ---------------------------------------------------------------------------
# Tee stdout to file
# ---------------------------------------------------------------------------

class _Tee:
    """Write to stdout and file simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for s in self.streams:
            s.write(text)
    def flush(self):
        for s in self.streams:
            s.flush()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_congress_reports(congress_num):
    """Load opensecrets_lda_reports.csv for a given congress."""
    path = CONGRESS_DIR / str(congress_num) / "opensecrets_lda_reports.csv"
    return pd.read_csv(path)


def load_node_attributes(congress_num):
    """Load node_attributes.csv (net_influence, net_strength) for a given congress."""
    path = CONGRESS_DIR / str(congress_num) / "node_attributes.csv"
    return pd.read_csv(path)


def load_rbo_edges(congress_num):
    """Load rbo_directed_influence.csv for a given congress."""
    path = CONGRESS_DIR / str(congress_num) / "rbo_directed_influence.csv"
    return pd.read_csv(path)


def load_bill_affiliation_gml():
    """Load bill affiliation GML (contains all centrality metrics)."""
    return nx.read_gml(str(BILL_AFF_GML), label="label")


def load_rbo_gml():
    """Load combined RBO directed influence GML."""
    return nx.read_gml(str(RBO_GML), label="label")


def assign_quarters(df, congress_num):
    """Add quarter_num col: year1 Q1-4 -> 1-4, year2 Q1-4 -> 5-8."""
    start, end = CONGRESS_PARAMS[congress_num]["years"]
    df = df.copy()
    df["base_q"]     = df["report_type"].str.extract(r"q(\d)").astype(float)
    df               = df.dropna(subset=["base_q"])
    year_off         = df["year"].map({start: 0, end: 4})
    df["quarter_num"] = df["base_q"] + year_off
    return df


# ---------------------------------------------------------------------------
# Checks 1-2: Quarter coverage
# ---------------------------------------------------------------------------

def check_quarter_coverage(congress_num):
    """Verify all 8 quarters present; return (pass, detail_str, per_quarter_df)."""
    df = load_congress_reports(congress_num)
    df = assign_quarters(df, congress_num)

    present = set(df["quarter_num"].dropna().unique())
    missing = set(range(1, 9)) - present

    per_q = (df.groupby("quarter_num")
               .agg(rows=("fortune_name", "count"),
                    firms=("fortune_name", "nunique"))
               .reset_index()
               .sort_values("quarter_num"))

    if missing:
        return False, f"missing quarters: {sorted(missing)}", per_q
    return True, "all 8 quarters present", per_q


def print_quarter_check(congress_num, per_q_df):
    """Print per-quarter summary table."""
    start, end = CONGRESS_PARAMS[congress_num]["years"]
    print(f"\n  {'Q#':<4} {'Year':>4}  {'Cal-Q':>5}  {'Rows':>6}  {'Firms':>5}")
    for _, row in per_q_df.iterrows():
        q   = int(row["quarter_num"])
        yr  = start if q <= 4 else end
        cq  = q if q <= 4 else q - 4
        print(f"  {q:<4} {yr:>4}  {'Q'+str(cq):>5}  "
              f"{int(row['rows']):>6}  {int(row['firms']):>5}")


# ---------------------------------------------------------------------------
# Check 3: Agenda-setters per community
# ---------------------------------------------------------------------------

def build_rbo_graph_from_edges(congress_num):
    """Build directed graph from per-congress RBO edge CSV; uses only columns present in file."""
    edges = load_rbo_edges(congress_num)
    na    = load_node_attributes(congress_num)
    # Use only edge attribute columns that actually exist in this CSV
    wanted = ["weight", "rbo", "source_firsts", "target_firsts", "net_temporal"]
    edge_attr = [c for c in wanted if c in edges.columns]
    G     = nx.from_pandas_edgelist(
        edges, source="source", target="target",
        edge_attr=edge_attr,
        create_using=nx.DiGraph(),
    )
    # Attach net_influence and net_strength from node_attributes
    ni_map = dict(zip(na["firm"], na["net_influence"]))
    ns_map = dict(zip(na["firm"], na["net_strength"]))
    for node in G.nodes():
        G.nodes[node]["net_influence"] = ni_map.get(node, 0)
        G.nodes[node]["net_strength"]  = ns_map.get(node, 0.0)
    return G, na


def detect_communities(G):
    """Detect communities via Louvain on undirected projection; return {node: community_id}."""
    try:
        from community import best_partition  # python-louvain
        UG = G.to_undirected()
        partition = best_partition(UG, weight="weight", random_state=42)
    except ImportError:
        # Fall back to greedy modularity if python-louvain not installed
        UG = G.to_undirected()
        comms = list(nx.community.greedy_modularity_communities(UG, weight="weight"))
        partition = {}
        for cid, members in enumerate(comms):
            for m in members:
                partition[m] = cid
    return partition


def top_agenda_setters_per_community(G, partition, na_df, k=TOP_K):
    """Return DataFrame: top-k firms per community by net_strength."""
    rows = []
    for node in G.nodes():
        rows.append({
            "firm":          node,
            "community":     partition.get(node, -1),
            "net_influence": G.nodes[node].get("net_influence", 0),
            "net_strength":  G.nodes[node].get("net_strength", 0.0),
        })
    df = pd.DataFrame(rows)
    top = (df.sort_values("net_strength", ascending=False)
             .groupby("community")
             .head(k)
             .reset_index(drop=True))
    return top


# ---------------------------------------------------------------------------
# Check 4: Bill affiliation centrality vs RBO agenda-setters
# ---------------------------------------------------------------------------

def align_bill_aff_with_rbo(bill_aff_G, rbo_na_df, top_k=TOP_K):
    """
    For each bill affiliation community, find top-k firms by PageRank and
    cross-reference with top RBO agenda-setters (by net_influence).
    Returns a merged DataFrame.
    """
    # Build bill-aff centrality DataFrame
    rows = []
    for n, d in bill_aff_G.nodes(data=True):
        rows.append({
            "firm":                    n,
            "community":               d.get("community"),
            "within_comm_eigenvector": d.get("within_comm_eigenvector", 0.0),
            "z_score":                 d.get("z_score", 0.0),
            "participation_coeff":     d.get("participation_coeff"),
            "global_pagerank":         d.get("global_pagerank", 0.0),
            "katz_centrality":         d.get("katz_centrality", 0.0),
            "ga_role":                 d.get("ga_role", ""),
            "kcore":                   d.get("kcore"),
        })
    ba_df = pd.DataFrame(rows)

    # Merge with RBO net_influence
    merged = ba_df.merge(
        rbo_na_df[["firm", "net_influence", "net_strength"]],
        on="firm", how="left",
    )
    merged["net_influence"] = merged["net_influence"].fillna(0)
    return merged, ba_df


# ---------------------------------------------------------------------------
# Check 5: Power-law testing
# ---------------------------------------------------------------------------

def test_power_law(values, label):
    """
    Test whether `values` follows a power law by fitting log-log regression
    on the CCDF. Also runs KS test against log-normal.
    Returns dict of results.
    """
    vals = np.array([v for v in values if v > 0])
    if len(vals) < 10:
        return {"label": label, "n": len(vals), "note": "too few nonzero values"}

    # Log-normal KS test
    ln_params  = stats.lognorm.fit(vals, floc=0)
    ks_stat, ks_p = stats.kstest(vals, "lognorm", args=ln_params)

    # Power-law fit via log-log OLS on complementary CDF
    vals_sorted = np.sort(vals)[::-1]
    ranks       = np.arange(1, len(vals_sorted) + 1)
    ccdf        = ranks / len(vals_sorted)
    # Fit log(CCDF) ~ alpha * log(x) in top 80% (trim bottom noise)
    trim    = max(1, int(0.2 * len(vals_sorted)))
    x_fit   = np.log(vals_sorted[:-trim] + 1e-12)
    y_fit   = np.log(ccdf[:-trim] + 1e-12)
    slope, intercept, r, p_val, _ = stats.linregress(x_fit, y_fit)

    # Herfindahl-Hirschman Index as concentration measure
    total = vals.sum()
    hhi   = np.sum((vals / total) ** 2) if total > 0 else 0.0

    return {
        "label":      label,
        "n":          len(vals),
        "mean":       float(np.mean(vals)),
        "std":        float(np.std(vals)),
        "max":        float(np.max(vals)),
        "gini":       float(_gini(vals)),
        "hhi":        float(hhi),
        "pl_slope":   float(slope),
        "pl_r2":      float(r ** 2),
        "ln_ks_stat": float(ks_stat),
        "ln_ks_p":    float(ks_p),
    }


def _gini(vals):
    """Gini coefficient for a 1-D array of non-negative values."""
    vals = np.sort(np.abs(vals))
    n    = len(vals)
    if n == 0 or vals.sum() == 0:
        return 0.0
    cumvals = np.cumsum(vals)
    return float((2 * np.dot(np.arange(1, n + 1), vals) - (n + 1) * cumvals[-1])
                 / (n * cumvals[-1]))


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _save(fig, name):
    """Save figure and close."""
    path = FIGURES_DIR / name
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.name}")
    return path.name


def plot_degree_histogram(G, title, fname, directed=True):
    """Histogram of in/out degree (directed) or degree (undirected)."""
    fig, axes = plt.subplots(1, 2 if directed else 1, figsize=(10, 4))

    if directed:
        in_deg  = [d for _, d in G.in_degree()]
        out_deg = [d for _, d in G.out_degree()]
        axes[0].hist(in_deg,  bins=30, color="#2196F3", edgecolor="white", alpha=0.85)
        axes[0].set_title("In-Degree")
        axes[0].set_xlabel("Degree")
        axes[0].set_ylabel("Count")
        axes[1].hist(out_deg, bins=30, color="#FF5722", edgecolor="white", alpha=0.85)
        axes[1].set_title("Out-Degree")
        axes[1].set_xlabel("Degree")
    else:
        deg = [d for _, d in G.degree()]
        axes.hist(deg, bins=30, color="#4CAF50", edgecolor="white", alpha=0.85)
        axes.set_title("Degree")
        axes.set_xlabel("Degree")
        axes.set_ylabel("Count")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, fname)


def plot_loglog_ccdf(values_dict, title, fname):
    """Log-log CCDF for multiple series — power-law diagnostic."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1976D2", "#D32F2F", "#388E3C", "#7B1FA2", "#F57C00"]
    for (label, vals), color in zip(values_dict.items(), colors):
        vals_pos = np.array([v for v in vals if v > 0])
        if len(vals_pos) == 0:
            continue
        sorted_v = np.sort(vals_pos)[::-1]
        ccdf     = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.loglog(sorted_v, ccdf, ".", markersize=4, label=label, color=color, alpha=0.75)

    ax.set_xlabel("Value (log scale)")
    ax.set_ylabel("P(X ≥ x) — CCDF (log scale)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return _save(fig, fname)


def plot_top_k_bar(df, x_col, y_col, title, fname, color="#1565C0", rotate=45):
    """Horizontal bar chart of top-k firms."""
    df = df.sort_values(y_col, ascending=True).tail(TOP_K)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df[x_col], df[y_col], color=color, edgecolor="white", height=0.65)
    ax.set_xlabel(y_col.replace("_", " ").title())
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    return _save(fig, fname)


def plot_community_sizes(sizes_dict, title, fname):
    """Bar chart of community sizes."""
    comms = sorted(sizes_dict.keys())
    sizes = [sizes_dict[c] for c in comms]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([f"Comm {c}" for c in comms], sizes,
                  color="#1976D2", edgecolor="white", alpha=0.85)
    ax.set_ylabel("Number of Firms")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.bar_label(bars, padding=2)
    plt.tight_layout()
    return _save(fig, fname)


def plot_net_influence_comparison(ni_116, ni_117, fname):
    """Overlaid histograms of net_influence for 116 vs 117."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(ni_116.min(), ni_117.min()),
        max(ni_116.max(), ni_117.max()),
        35,
    )
    ax.hist(ni_116, bins=bins, alpha=0.6, label="116th Congress", color="#1976D2", edgecolor="white")
    ax.hist(ni_117, bins=bins, alpha=0.6, label="117th Congress", color="#D32F2F", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Net Influence Score")
    ax.set_ylabel("Count")
    ax.set_title("Net Influence Distribution: 116th vs 117th Congress",
                 fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return _save(fig, fname)


def plot_quarter_activity(per_q_df, congress_num, fname):
    """Bar chart of report rows per quarter for a given congress."""
    start, end = CONGRESS_PARAMS[congress_num]["years"]
    labels = []
    for _, row in per_q_df.iterrows():
        q  = int(row["quarter_num"])
        yr = start if q <= 4 else end
        cq = q if q <= 4 else q - 4
        labels.append(f"Q{cq}\n{yr}")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    color1 = "#1565C0"
    ax1.bar(labels, per_q_df["rows"], color=color1, alpha=0.75,
            label="Reports (rows)", edgecolor="white")
    ax1.set_ylabel("Report Rows", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#C62828"
    ax2.plot(labels, per_q_df["firms"], "o-", color=color2,
             linewidth=2, markersize=6, label="Unique Firms")
    ax2.set_ylabel("Unique Firms", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(f"{congress_num}th Congress — Quarterly Activity",
                  fontsize=12, fontweight="bold")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88), fontsize=9)
    plt.tight_layout()
    return _save(fig, fname)


def plot_centrality_scatter(merged_df, x_col, y_col, title, fname):
    """Scatter: bill-affil centrality vs RBO net_influence."""
    sub = merged_df.dropna(subset=[x_col, y_col])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub[x_col], sub[y_col], alpha=0.55, s=20, color="#1976D2", edgecolors="none")
    # Regression line
    if len(sub) > 5:
        slope, intercept, r, p_val, _ = stats.linregress(sub[x_col], sub[y_col])
        x_line = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "r--", linewidth=1.5,
                label=f"r={r:.2f}, p={p_val:.3f}")
        ax.legend(fontsize=9)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    return _save(fig, fname)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def sep(n=64):
    print("=" * n)

def section(title):
    sep()
    print(f"  {title}")
    sep()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _orig_stdout = sys.stdout
    _f = open(OUTPUT_PATH, "w")
    sys.stdout = _Tee(_orig_stdout, _f)

    try:
        # =================================================================
        # Checks 1-2: Quarter coverage
        # =================================================================
        section("12 — Congress Statistics & Validation")

        print("\n-- CHECK 1: 117th Congress Quarter Coverage --")
        ok117, detail117, pq117 = check_quarter_coverage(117)
        status = PASS if ok117 else FAIL
        print(f"  {status}  —  {detail117}")
        print_quarter_check(117, pq117)
        plot_quarter_activity(pq117, 117, "12a_117_quarterly_activity.png")

        print("\n-- CHECK 2: 116th Congress Quarter Coverage --")
        ok116, detail116, pq116 = check_quarter_coverage(116)
        status = PASS if ok116 else FAIL
        print(f"  {status}  —  {detail116}")
        print_quarter_check(116, pq116)
        plot_quarter_activity(pq116, 116, "12b_116_quarterly_activity.png")

        # =================================================================
        # Load core data
        # =================================================================
        print("\n-- Loading networks --")
        bill_aff_G = load_bill_affiliation_gml()
        rbo_G      = load_rbo_gml()
        na116      = load_node_attributes(116)
        na117      = load_node_attributes(117)
        G116, _    = build_rbo_graph_from_edges(116)
        G117, _    = build_rbo_graph_from_edges(117)

        print(f"  Bill affiliation GML: {bill_aff_G.number_of_nodes()} nodes, "
              f"{bill_aff_G.number_of_edges()} edges")
        print(f"  RBO combined GML:     {rbo_G.number_of_nodes()} nodes, "
              f"{rbo_G.number_of_edges()} edges")
        print(f"  116th graph:          {G116.number_of_nodes()} nodes, "
              f"{G116.number_of_edges()} edges")
        print(f"  117th graph:          {G117.number_of_nodes()} nodes, "
              f"{G117.number_of_edges()} edges")

        # =================================================================
        # Detect communities for 116th and 117th
        # =================================================================
        print("\n-- Community detection (Louvain / greedy-modularity fallback) --")
        try:
            import community as cm_pkg
            p116 = cm_pkg.best_partition(G116.to_undirected(), weight="weight", random_state=42)
            p117 = cm_pkg.best_partition(G117.to_undirected(), weight="weight", random_state=42)
            print("  Using python-louvain (Blondel et al. 2008)")
        except ImportError:
            UG116 = G116.to_undirected()
            UG117 = G117.to_undirected()
            comms116 = list(nx.community.greedy_modularity_communities(UG116, weight="weight"))
            comms117 = list(nx.community.greedy_modularity_communities(UG117, weight="weight"))
            p116 = {n: cid for cid, c in enumerate(comms116) for n in c}
            p117 = {n: cid for cid, c in enumerate(comms117) for n in c}
            print("  python-louvain not found; used greedy modularity (NetworkX)")

        n_comms116 = len(set(p116.values()))
        n_comms117 = len(set(p117.values()))
        print(f"  116th: {n_comms116} communities detected")
        print(f"  117th: {n_comms117} communities detected")

        # Community sizes
        sizes116 = Counter(p116.values())
        sizes117 = Counter(p117.values())
        plot_community_sizes(dict(sizes116), "116th Congress RBO Community Sizes",
                             "12c_community_sizes_116.png")
        plot_community_sizes(dict(sizes117), "117th Congress RBO Community Sizes",
                             "12d_community_sizes_117.png")

        # =================================================================
        # CHECK 3: Agenda-setters per community
        # =================================================================
        section("CHECK 3: Agenda-Setters per Community")

        for congress_num, G_c, partition, na in [
            (116, G116, p116, na116),
            (117, G117, p117, na117),
        ]:
            print(f"\n  -- {congress_num}th Congress --")
            top = top_agenda_setters_per_community(G_c, partition, na, k=5)

            for cid in sorted(top["community"].unique()):
                sub = top[top["community"] == cid]
                n_in_comm = list(partition.values()).count(cid)
                print(f"\n  Community {cid}  ({n_in_comm} firms):")
                print(f"  {'Firm':<40} {'Net-Str':>10}  {'Net-Inf':>8}")
                for _, r in sub.iterrows():
                    print(f"  {r['firm']:<40} {r['net_strength']:>10.4f}  "
                          f"{int(r['net_influence']):>8}")

        # =================================================================
        # CHECK 4: Bill affiliation GML centrality vs RBO agenda-setters
        # =================================================================
        section("CHECK 4: Bill-Affiliation Centrality vs RBO Agenda-Setters")

        # Use combined na (116+117 union from node_attributes)
        na_all = pd.concat([na116, na117]).drop_duplicates("firm").reset_index(drop=True)
        merged, ba_df = align_bill_aff_with_rbo(bill_aff_G, na_all, top_k=TOP_K)

        CENT_METRICS = [
            ("global_pagerank",         "Global PageRank"),
            ("within_comm_eigenvector", "Within-Comm Eigenvector"),
            ("katz_centrality",         "Katz-Bonacich"),
            ("participation_coeff",     "Participation Coeff (P)"),
            ("z_score",                 "GA z-Score"),
        ]

        for metric, label in CENT_METRICS:
            top_cent = (merged.dropna(subset=[metric])
                              .nlargest(TOP_K, metric)[["firm", metric, "net_influence", "community"]])
            print(f"\n  Top {TOP_K} by {label}:")
            print(f"  {'Firm':<38} {metric[:12]:>12}  {'Net-Inf':>8}  {'Comm':>5}")
            for _, r in top_cent.iterrows():
                comm_str = str(int(r["community"])) if not pd.isna(r["community"]) else "—"
                print(f"  {r['firm']:<38} {r[metric]:>12.5f}  "
                      f"{int(r['net_influence']):>8}  {comm_str:>5}")

        # Correlation table — net_strength (primary) and net_influence (reference)
        print("\n  Pearson correlations with RBO agenda-setter metrics:")
        print(f"  {'Metric':<30} {'r(net_str)':>10}  {'p':>7} | {'r(net_inf)':>10}  {'p':>7}  {'n':>5}")
        for metric, label in CENT_METRICS:
            sub_ns = merged.dropna(subset=[metric, "net_strength"])
            sub_ni = merged.dropna(subset=[metric, "net_influence"])
            if len(sub_ns) < 5 or len(sub_ni) < 5:
                continue
            r_ns, p_ns = stats.pearsonr(sub_ns[metric], sub_ns["net_strength"])
            r_ni, p_ni = stats.pearsonr(sub_ni[metric], sub_ni["net_influence"])
            print(f"  {label:<30} {r_ns:>10.3f}  {p_ns:>7.4f} | {r_ni:>10.3f}  {p_ni:>7.4f}  {len(sub_ns):>5}")

        # Scatter plots — y-axis: net_strength (primary)
        plot_centrality_scatter(merged, "global_pagerank", "net_strength",
            "Bill-Affil PageRank vs RBO Net Strength",
            "12e_scatter_pagerank_vs_netstr.png")
        plot_centrality_scatter(merged, "katz_centrality", "net_strength",
            "Bill-Affil Katz-Bonacich vs RBO Net Strength",
            "12f_scatter_katz_vs_netstr.png")

        # Bar charts by centrality metric
        for metric, label in CENT_METRICS[:3]:
            top_df = (merged.dropna(subset=[metric])
                            .nlargest(TOP_K, metric)[["firm", metric]])
            fname = f"12g_top_{metric[:10]}.png"
            plot_top_k_bar(top_df, "firm", metric,
                           f"Top {TOP_K} Firms — {label}", fname)

        # GA role distribution in bill affiliation
        print("\n  Guimerà-Amaral role distribution (bill affiliation GML):")
        role_counts = merged["ga_role"].value_counts()
        for role, cnt in role_counts.items():
            pct = 100 * cnt / len(merged)
            print(f"  {role:<22}: {cnt:>4}  ({pct:.1f}%)")

        # =================================================================
        # CHECK 5: Power-law evidence
        # =================================================================
        section("CHECK 5: Power-Law Evidence")

        # Degree distributions: bill affiliation and RBO
        ba_deg      = [d for _, d in bill_aff_G.degree()]
        rbo_out_deg = [d for _, d in rbo_G.out_degree()]
        rbo_in_deg  = [d for _, d in rbo_G.in_degree()]

        # Centrality distributions
        pr_vals   = [d.get("global_pagerank", 0)         for _, d in bill_aff_G.nodes(data=True)]
        katz_vals = [d.get("katz_centrality", 0)         for _, d in bill_aff_G.nodes(data=True)]
        ni_vals   = [d.get("net_influence", 0)           for _, d in rbo_G.nodes(data=True)]
        ns_vals   = [d.get("net_strength", 0)            for _, d in rbo_G.nodes(data=True)]

        power_law_tests = [
            (ba_deg,      "Bill-Aff Degree"),
            (rbo_out_deg, "RBO Out-Degree"),
            (rbo_in_deg,  "RBO In-Degree"),
            (pr_vals,     "Bill-Aff PageRank"),
            (katz_vals,   "Bill-Aff Katz"),
            ([abs(v) for v in ni_vals if v != 0], "RBO |Net Influence|"),
        ]

        print(f"\n  {'Metric':<28} {'n':>5}  {'Gini':>6}  {'HHI':>6}  "
              f"{'PL slope':>9}  {'PL R²':>7}  {'LN KS-p':>8}")
        for vals, label in power_law_tests:
            res = test_power_law(vals, label)
            if "note" in res:
                print(f"  {label:<28}  {res['note']}")
                continue
            print(f"  {label:<28} {res['n']:>5}  {res['gini']:>6.3f}  "
                  f"{res['hhi']:>6.4f}  {res['pl_slope']:>9.3f}  "
                  f"{res['pl_r2']:>7.3f}  {res['ln_ks_p']:>8.4f}")

        print("\n  Interpretation guide:")
        print("  Gini > 0.6 and PL slope ≈ -1 to -2 suggests power-law-like concentration.")
        print("  LN KS-p < 0.05 means log-normal fit is rejected (consistent with heavier tail).")
        print("  PL R² in log-log CCDF: closer to 1.0 = straighter power-law tail.")

        # Log-log CCDF plots
        plot_loglog_ccdf(
            {"Bill-Aff Degree": ba_deg,
             "RBO Out-Degree":  rbo_out_deg,
             "RBO In-Degree":   rbo_in_deg},
            "Degree Distribution CCDF (Log-Log) — Power-Law Diagnostic",
            "12h_loglog_ccdf_degree.png",
        )
        plot_loglog_ccdf(
            {"PageRank":     pr_vals,
             "Katz-Bonacich": katz_vals},
            "Centrality CCDF (Log-Log) — Power-Law Diagnostic",
            "12i_loglog_ccdf_centrality.png",
        )
        plot_loglog_ccdf(
            {"|Net Influence|": [abs(v) for v in ni_vals if v != 0],
             "|Net Strength|":  [abs(v) for v in ns_vals if v != 0]},
            "RBO Influence CCDF (Log-Log) — Power-Law Diagnostic",
            "12j_loglog_ccdf_rbo_influence.png",
        )

        # =================================================================
        # CHECK 6-7: Degree histograms and centrality bar charts
        # =================================================================
        section("CHECK 6-7: Degree Histograms & Additional Charts")

        plot_degree_histogram(
            rbo_G,
            "RBO Directed Influence — Degree Distribution",
            "12k_rbo_degree_hist.png",
            directed=True,
        )
        plot_degree_histogram(
            bill_aff_G,
            "Bill Affiliation — Degree Distribution",
            "12l_billaff_degree_hist.png",
            directed=False,
        )

        # =================================================================
        # CHECK 8: Community size distribution
        # =================================================================
        section("CHECK 8: Community Sizes")

        ba_comm_sizes = Counter(d.get("community") for _, d in bill_aff_G.nodes(data=True))
        plot_community_sizes(
            dict(ba_comm_sizes),
            "Bill Affiliation Network — Community Sizes (Leiden)",
            "12m_billaff_comm_sizes.png",
        )

        print(f"\n  Bill affiliation communities (Leiden):")
        for cid, cnt in sorted(ba_comm_sizes.items()):
            print(f"  Comm {cid}: {cnt} firms")

        # Top 5 firms per community by PageRank (bill affiliation)
        print("\n  Top 3 firms per bill-affiliation community (PageRank):")
        for cid in sorted(ba_comm_sizes.keys()):
            sub = merged[merged["community"] == cid].nlargest(3, "global_pagerank")
            label = f"  Comm {cid}:"
            firms = ", ".join(f"{r['firm']} ({r['global_pagerank']:.5f})"
                              for _, r in sub.iterrows())
            print(f"  {label} {firms}")

        # =================================================================
        # CHECK 9: Net influence distribution comparison
        # =================================================================
        section("CHECK 9: Net Influence Distribution (116th vs 117th)")

        ni116_vals = na116.set_index("firm")["net_influence"]
        ni117_vals = na117.set_index("firm")["net_influence"]

        print(f"\n  116th Congress net_influence:")
        print(f"    n={len(ni116_vals)}  mean={ni116_vals.mean():.2f}  "
              f"std={ni116_vals.std():.2f}  "
              f"min={ni116_vals.min()}  max={ni116_vals.max()}")
        print(f"    positive (agenda-setters): {(ni116_vals > 0).sum()}")
        print(f"    negative (net followers):  {(ni116_vals < 0).sum()}")
        print(f"    neutral (net_inf=0):        {(ni116_vals == 0).sum()}")

        print(f"\n  117th Congress net_influence:")
        print(f"    n={len(ni117_vals)}  mean={ni117_vals.mean():.2f}  "
              f"std={ni117_vals.std():.2f}  "
              f"min={ni117_vals.min()}  max={ni117_vals.max()}")
        print(f"    positive (agenda-setters): {(ni117_vals > 0).sum()}")
        print(f"    negative (net followers):  {(ni117_vals < 0).sum()}")
        print(f"    neutral (net_inf=0):        {(ni117_vals == 0).sum()}")

        ks_stat, ks_p = stats.ks_2samp(ni116_vals, ni117_vals)
        print(f"\n  KS test (116 vs 117 net_influence):  D={ks_stat:.4f}, p={ks_p:.4f}")
        if ks_p < 0.05:
            print("  -> Distributions differ significantly (p < 0.05)")
        else:
            print("  -> No significant distributional difference (p >= 0.05)")

        plot_net_influence_comparison(ni116_vals, ni117_vals,
                                      "12n_netinf_116_vs_117.png")

        # Top 10 agenda-setters per congress — sorted by net_strength (primary)
        print(f"\n  Top {TOP_K} Agenda-Setters — 116th Congress (by net_strength):")
        print(f"  {'Firm':<40} {'Net-Str':>10}  {'Net-Inf':>8}")
        top116 = na116.nlargest(TOP_K, "net_strength")
        for _, r in top116.iterrows():
            print(f"  {r['firm']:<40} {r['net_strength']:>10.4f}  "
                  f"{int(r['net_influence']):>8}")

        print(f"\n  Top {TOP_K} Agenda-Setters — 117th Congress (by net_strength):")
        print(f"  {'Firm':<40} {'Net-Str':>10}  {'Net-Inf':>8}")
        top117 = na117.nlargest(TOP_K, "net_strength")
        for _, r in top117.iterrows():
            print(f"  {r['firm']:<40} {r['net_strength']:>10.4f}  "
                  f"{int(r['net_influence']):>8}")

        # =================================================================
        # CHECK 10: Quarter-level summary (117th)
        # =================================================================
        section("CHECK 10: Quarter-Level Activity — 117th Congress")

        df117 = load_congress_reports(117)
        df117 = assign_quarters(df117, 117)
        q_agg = (df117.groupby("quarter_num")
                      .agg(rows=("fortune_name", "count"),
                           firms=("fortune_name", "nunique"),
                           unique_bills=("bill_number",
                                         lambda x: x.dropna().nunique()))
                      .reset_index())
        print(f"\n  {'Q#':<4}  {'Rows':>6}  {'Firms':>5}  {'Bills (w/ num)':>14}")
        for _, row in q_agg.iterrows():
            print(f"  {int(row['quarter_num']):<4}  {int(row['rows']):>6}  "
                  f"{int(row['firms']):>5}  {int(row['unique_bills']):>14}")

        # =================================================================
        # Summary
        # =================================================================
        sep()
        print("  SUMMARY OF CHECKS")
        sep()
        checks = [
            ("117th quarter coverage", ok117),
            ("116th quarter coverage", ok116),
        ]
        passed = sum(1 for _, ok in checks if ok)
        for name, ok in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print(f"\n  {passed}/{len(checks)} checks passed")
        print(f"  {len(list(FIGURES_DIR.glob('12*.png')))} figures written to {FIGURES_DIR.relative_to(ROOT)}")
        sep()

    finally:
        sys.stdout = _orig_stdout
        _f.close()
        print(f"\nOutput saved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
