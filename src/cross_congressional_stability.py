"""
Cross-congressional stability analysis (111th-117th Congress).

Tests whether directed RBO influence edges are consistent in direction and
magnitude across seven consecutive congresses for the set of Fortune 500 firms
that lobbied in all seven sessions.

Four analyses:
  1. Direction consistency — does the same firm lead across sessions?
  2. Magnitude stability — are net_temporal values correlated across sessions?
  3. Firm rank stability — do firm net_influence ranks agree across sessions?
  4. Firm net_strength rank stability — mirrors Analysis 3 for net_strength.

Requires multi_congress_pipeline.py to have been run for all seven congresses.
Outputs:
  outputs/cross_congressional/cross_congressional_stability.txt
  outputs/cross_congressional/cross_congressional_stability.png

Run: python cross_congressional_stability.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, ROOT

CONGRESSES     = [111, 112, 113, 114, 115, 116, 117]
CONGRESS_LABELS = {
    111: "111th\n(2009-10)",
    112: "112th\n(2011-12)",
    113: "113th\n(2013-14)",
    114: "114th\n(2015-16)",
    115: "115th\n(2017-18)",
    116: "116th\n(2019-20)",
    117: "117th\n(2021-22)",
}
MIN_SESSIONS_DIRECTION = 2   # min directed sessions for a pair to enter direction analysis
MIN_SESSIONS_MAGNITUDE = 2   # min sessions for magnitude analysis (per pair)
MIN_SESSIONS_W         = 3   # min sessions for Kendall's W pair-level estimate

OUT_DIR  = ROOT / "outputs" / "cross_congressional"
TXT_PATH = OUT_DIR / "cross_congressional_stability.txt"
PNG_PATH = OUT_DIR / "cross_congressional_stability.png"


# ---------------------------------------------------------------------------
# Output tee
# ---------------------------------------------------------------------------

class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w")
    def write(self, msg):
        import builtins
        builtins.print.__self__.write(msg) if hasattr(builtins.print, "__self__") else None
        self._f.write(msg)
        self._f.flush()
    def flush(self):
        self._f.flush()
    def close(self):
        self._f.close()

import io as _io

def _print(*args, **kwargs):
    """Print to stdout and the tee file."""
    print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_edges():
    """Load rbo_directed_influence.csv for each congress; return dict congress → DataFrame."""
    edges = {}
    nodes = {}
    for c in CONGRESSES:
        edge_path = DATA_DIR / "congress" / str(c) / "rbo_directed_influence.csv"
        node_path = DATA_DIR / "congress" / str(c) / "node_attributes.csv"
        if not edge_path.exists():
            raise FileNotFoundError(
                f"data/congress/{c}/rbo_directed_influence.csv not found. "
                f"Run multi_congress_pipeline.py first."
            )
        edges[c] = pd.read_csv(edge_path)
        if node_path.exists():
            nodes[c] = pd.read_csv(node_path)
        else:
            nodes[c] = None
    return edges, nodes


def get_firm_set(edges_df):
    """Return set of all firms (nodes) in an edge DataFrame."""
    return set(edges_df["source"]) | set(edges_df["target"])


# ---------------------------------------------------------------------------
# Pair records — canonical direction tracking
# ---------------------------------------------------------------------------

def build_pair_matrix(edges_by_congress, stable_firms):
    """Build per-session records for each canonical pair in the stable set.

    Returns a dict: (firm_a, firm_b) → list of dicts, one per session present.
    firm_a < firm_b alphabetically (canonical).
    Each dict: {congress, direction, net_temporal, rbo}
      direction: 1 = firm_a leads (net_temporal > 0), 0 = firm_b leads (net_temporal < 0),
                 None = balanced (net_temporal == 0).
    """
    pairs = {}
    for c, df in edges_by_congress.items():
        # Restrict to stable set and canonical direction (source < target) to get one row per pair
        mask = (
            df["source"].isin(stable_firms) &
            df["target"].isin(stable_firms) &
            (df["source"] < df["target"])
        )
        sub = df[mask].copy()
        for _, row in sub.iterrows():
            a, b = row["source"], row["target"]   # already canonical (a < b)
            key  = (a, b)
            nt   = int(row["net_temporal"])
            if nt > 0:
                direction = 1    # firm_a (source) is the net first-mover
            elif nt < 0:
                direction = 0    # firm_b (target) is the net first-mover
            else:
                direction = None # balanced: equal first-mover counts
            pairs.setdefault(key, []).append({
                "congress":     c,
                "direction":    direction,
                "net_temporal": nt,
                "rbo":          float(row["rbo"]) if "rbo" in row.index else float("nan"),
            })
    return pairs


# ---------------------------------------------------------------------------
# Analysis 1: Direction consistency
# ---------------------------------------------------------------------------

def run_direction_analysis(pairs):
    """Compute direction consistency scores and run binomial tests.

    Returns (summary_df, overall_stats_dict).
    """
    records = []
    for (a, b), sessions in pairs.items():
        directed = [s for s in sessions if s["direction"] is not None]
        n_total  = len(sessions)
        n_dir    = len(directed)
        if n_dir < MIN_SESSIONS_DIRECTION:
            continue
        n_a  = sum(1 for s in directed if s["direction"] == 1)
        n_b  = n_dir - n_a
        cons = max(n_a, n_b) / n_dir
        # Binomial test: H0 = p(consistent direction) = 0.5
        result   = stats.binomtest(max(n_a, n_b), n=n_dir, p=0.5, alternative="greater")
        records.append({
            "firm_a":        a,
            "firm_b":        b,
            "n_sessions":    n_total,
            "n_directed":    n_dir,
            "n_a_leads":     n_a,
            "n_b_leads":     n_b,
            "consistency":   round(cons, 4),
            "binom_p":       round(result.pvalue, 6),
        })
    df = pd.DataFrame(records)
    if df.empty:
        return df, {}

    n_sig = (df["binom_p"] < 0.05).sum()
    stats_out = {
        "n_pairs":           len(df),
        "mean_consistency":  df["consistency"].mean(),
        "median_consistency": df["consistency"].median(),
        "n_geq_80":          (df["consistency"] >= 0.80).sum(),
        "n_perfect":         (df["consistency"] == 1.00).sum(),
        "n_sig_binom":       n_sig,
        "pct_sig_binom":     100 * n_sig / len(df),
        "pct_majority":      100 * (df["consistency"] > 0.5).mean(),
    }
    return df, stats_out


# ---------------------------------------------------------------------------
# Analysis 2: Magnitude stability (net_temporal and weight)
# ---------------------------------------------------------------------------

def run_magnitude_analysis(pairs):
    """Spearman correlations of net_temporal across congress pairs; Kendall's W.

    Returns (corr_matrix_df, w_stat, w_pval, n_w_pairs).
    """
    # Build wide matrix: rows = pairs, cols = congresses, values = net_temporal
    rows = []
    for (a, b), sessions in pairs.items():
        row = {"pair": f"{a}|{b}"}
        for s in sessions:
            row[s["congress"]] = s["net_temporal"]
        rows.append(row)
    wide = pd.DataFrame(rows).set_index("pair")
    for c in CONGRESSES:
        if c not in wide.columns:
            wide[c] = np.nan
    wide = wide[CONGRESSES]

    # Pairwise Spearman correlations
    labels = [str(c) for c in CONGRESSES]
    corr_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    n_matrix    = pd.DataFrame(0, index=labels, columns=labels)
    p_matrix    = pd.DataFrame(np.nan, index=labels, columns=labels)

    for i, ci in enumerate(CONGRESSES):
        for j, cj in enumerate(CONGRESSES):
            if i == j:
                corr_matrix.at[str(ci), str(cj)] = 1.0
                continue
            mask = wide[ci].notna() & wide[cj].notna()
            n    = mask.sum()
            if n >= 5:
                rho, p = stats.spearmanr(wide.loc[mask, ci], wide.loc[mask, cj])
                corr_matrix.at[str(ci), str(cj)] = round(rho, 4)
                p_matrix.at[str(ci), str(cj)]    = round(p, 6)
                n_matrix.at[str(ci), str(cj)]    = n

    # Kendall's W across all congresses (pairs with ≥ MIN_SESSIONS_W non-NaN)
    eligible = wide.dropna(thresh=MIN_SESSIONS_W)
    w_stat = w_pval = n_w = None
    if len(eligible) >= 3:
        # Fill remaining NaN with column median (best available estimate)
        filled = eligible.apply(lambda col: col.fillna(col.median()))
        w_stat, w_pval, n_w = _kendalls_w(filled.values.T)

    return corr_matrix, p_matrix, n_matrix, w_stat, w_pval, n_w


def _kendalls_w(ratings_matrix):
    """Kendall's W concordance coefficient.

    ratings_matrix: shape (n_raters, n_subjects); computes chi2 p-value.
    Returns (W, p_value, n_subjects).
    """
    m, n = ratings_matrix.shape   # m raters, n subjects
    ranked = np.apply_along_axis(rankdata, 1, ratings_matrix)
    rank_sums = ranked.sum(axis=0)
    S = np.sum((rank_sums - rank_sums.mean()) ** 2)
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    chi2_stat = m * (n - 1) * W
    p = stats.chi2.sf(chi2_stat, df=n - 1)
    return round(float(W), 6), round(float(p), 8), n


# ---------------------------------------------------------------------------
# Analysis 3 / 4: Firm node-metric rank stability (generic)
# ---------------------------------------------------------------------------

def run_firm_stability(nodes_by_congress, stable_firms, metric="net_influence"):
    """Spearman + Kendall's W for firm node-metric ranks across congresses.

    metric: column name in node_attributes.csv (e.g. 'net_influence', 'net_strength').
    Returns (spearman_df, w_stat, w_pval, n_firms, firm_wide_df).
    """
    rows = []
    for firm in sorted(stable_firms):
        row = {"firm": firm}
        for c in CONGRESSES:
            nd = nodes_by_congress.get(c)
            if nd is not None and metric in nd.columns:
                match = nd.loc[nd["firm"] == firm, metric]
                row[c] = float(match.values[0]) if len(match) else np.nan
            else:
                row[c] = np.nan
        rows.append(row)
    wide = pd.DataFrame(rows).set_index("firm")[CONGRESSES]

    # Adjacent-congress Spearman
    spearman_rows = []
    for i in range(len(CONGRESSES) - 1):
        ci, cj = CONGRESSES[i], CONGRESSES[i + 1]
        mask   = wide[ci].notna() & wide[cj].notna()
        n      = mask.sum()
        if n >= 5:
            rho, p = stats.spearmanr(wide.loc[mask, ci], wide.loc[mask, cj])
            spearman_rows.append({"pair": f"{ci}-{cj}", "rho": round(rho, 4),
                                   "p": round(p, 6), "n": n})
    spearman_df = pd.DataFrame(spearman_rows)

    # Kendall's W across all 5
    eligible = wide.dropna()
    w_stat = w_pval = n_w = None
    if len(eligible) >= 3:
        w_stat, w_pval, n_w = _kendalls_w(eligible.values.T)

    return spearman_df, w_stat, w_pval, len(eligible), wide


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _spearman_bar_panel(ax, spearman_df, w_stat, w_pval, title, color):
    """Shared helper: bar chart of adjacent-congress Spearman ρ values."""
    if spearman_df.empty:
        ax.set_title(title, fontsize=10, fontweight="bold")
        return
    pairs_labels = spearman_df["pair"].tolist()
    rhos         = spearman_df["rho"].tolist()
    sig_flags    = (spearman_df["p"] < 0.05).tolist()
    x = range(len(pairs_labels))
    bars = ax.bar(x, rhos, color=[color if s else "#AAAAAA" for s in sig_flags],
                  edgecolor="white", zorder=2, width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(pairs_labels, fontsize=8)
    ax.set_ylabel("Spearman ρ", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.4, zorder=1)
    # Annotate Kendall's W
    if w_stat is not None:
        p_str = f"{w_pval:.2e}" if w_pval < 0.001 else f"{w_pval:.3f}"
        ax.annotate(f"Kendall's W = {w_stat:.3f}\np = {p_str}",
                    xy=(0.97, 0.96), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    # Star marks on significant bars
    for xi, (rho, sig) in enumerate(zip(rhos, sig_flags)):
        if sig:
            ax.text(xi, rho + 0.004, "*", ha="center", va="bottom", fontsize=10, color="black")


def plot_results(dir_df, dir_stats, corr_matrix, n_matrix, pairs,
                 str_spearman, w_stat_str, w_str_p,
                 firm_spearman, w_stat_firm, w_firm_p,
                 stable_n, png_path):
    """2×2 stability figure: direction | temporal heatmap | net_strength bars (primary) | net_influence bars (reference)."""
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    # ── Panel 1 (top-left): Direction consistency histogram ───────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if not dir_df.empty:
        bins = np.arange(0.5, 1.05, 0.05)
        ax1.hist(dir_df["consistency"], bins=bins, edgecolor="white",
                 color="#5D8AA8", zorder=2)
        ax1.axvline(0.80, color="#E74C3C", linestyle="--", linewidth=1.2,
                    label="80% threshold")
        ax1.set_xlabel("Direction consistency score", fontsize=9)
        ax1.set_ylabel("Number of firm pairs", fontsize=9)
        ax1.set_title("Edge direction consistency\nacross congresses",
                      fontsize=10, fontweight="bold")
        n80  = dir_stats.get("n_geq_80", 0)
        ntot = dir_stats.get("n_pairs", 1)
        ax1.annotate(f"{n80}/{ntot} pairs ≥ 0.80\n({100*n80/max(ntot,1):.0f}%)",
                     xy=(0.97, 0.96), xycoords="axes fraction",
                     ha="right", va="top", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
        ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(axis="y", alpha=0.4, zorder=1)
    ax1.set_xlim(0.45, 1.02)

    # ── Panel 2 (top-right): net_temporal Spearman heatmap ───────────────
    ax2 = fig.add_subplot(gs[0, 1])
    corr_vals = corr_matrix.astype(float).values
    im = ax2.imshow(corr_vals, cmap="RdYlGn", vmin=-0.3, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax2, shrink=0.75, label="Spearman ρ")
    short_labels = [str(c) for c in CONGRESSES]
    ax2.set_xticks(range(len(CONGRESSES)))
    ax2.set_xticklabels(short_labels, fontsize=8)
    ax2.set_yticks(range(len(CONGRESSES)))
    ax2.set_yticklabels(short_labels, fontsize=8)
    ax2.set_title("Net temporal magnitude\ncorrelation (Spearman ρ)",
                  fontsize=10, fontweight="bold")
    for i in range(len(CONGRESSES)):
        for j in range(len(CONGRESSES)):
            v = corr_vals[i, j]
            if not np.isnan(v):
                n_ij = int(n_matrix.iloc[i, j]) if i != j else 0
                label = f"{v:.2f}" if i == j else f"{v:.2f}\n(n={n_ij})"
                ax2.text(j, i, label, ha="center", va="center", fontsize=6.5,
                         color="black" if abs(v) < 0.7 else "white")

    # ── Panel 3 (bottom-left): net_strength adjacent Spearman bars [primary] ──
    ax3 = fig.add_subplot(gs[1, 0])
    _spearman_bar_panel(ax3, str_spearman, w_stat_str, w_str_p,
                        "Firm net_strength rank stability [primary]\n(adjacent-congress Spearman ρ)",
                        "#3498DB")

    # ── Panel 4 (bottom-right): net_influence adjacent Spearman bars [reference]
    ax4 = fig.add_subplot(gs[1, 1])
    _spearman_bar_panel(ax4, firm_spearman, w_stat_firm, w_firm_p,
                        "Firm net_influence rank stability [reference]\n(adjacent-congress Spearman ρ)",
                        "#2ECC71")

    c_range   = f"{CONGRESSES[0]}th–{CONGRESSES[-1]}th Congress"
    title_str = (f"Cross-Congressional Stability  |  {stable_n} firms in all {len(CONGRESSES)} sessions  "
                 f"({c_range})")
    fig.suptitle(title_str, fontsize=12, fontweight="bold", y=1.01)

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {png_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import builtins
    original_print = builtins.print
    log_file = open(TXT_PATH, "w")

    def tee_print(*args, **kwargs):
        original_print(*args, **kwargs)
        kwargs.pop("file", None)
        original_print(*args, file=log_file, **kwargs)
        log_file.flush()

    builtins.print = tee_print

    try:
        _run()
    finally:
        builtins.print = original_print
        log_file.close()


def _run():
    SEP = "=" * 72

    print(SEP)
    _c_range = f"{CONGRESSES[0]}th–{CONGRESSES[-1]}th Congress"
    print(f"CROSS-CONGRESSIONAL STABILITY  ({_c_range})")
    print(f"Fortune 500 firms active in all {len(CONGRESSES)} sessions")
    print(SEP)

    # -- Load data -------------------------------------------------------
    try:
        edges_by_congress, nodes_by_congress = load_all_edges()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    # -- Coverage --------------------------------------------------------
    print("\nCOVERAGE")
    print("-" * 40)
    firm_sets = {}
    for c in CONGRESSES:
        df = edges_by_congress[c]
        fs = get_firm_set(df)
        firm_sets[c] = fs
        # Each pair has two edges; decisive pairs have |net_temporal| > 0
        n_pairs    = len(df) // 2
        n_decisive = (df["net_temporal"] > 0).sum()   # one canonical edge per decisive pair
        n_balanced = n_pairs - n_decisive
        print(f"  Congress {c}: {len(fs):>4} firms  |  "
              f"{len(df):>5} total edges  "
              f"({n_pairs:,} pairs: decisive={n_decisive:,}  balanced={n_balanced:,})")

    stable_firms = firm_sets[CONGRESSES[0]]
    for c in CONGRESSES[1:]:
        stable_firms &= firm_sets[c]
    stable_firms = sorted(stable_firms)
    n_cong = len(CONGRESSES)
    print(f"\n  Stable set (all {n_cong} congresses): {len(stable_firms)} firms")
    if len(stable_firms) == 0:
        print(f"\nWARNING: No firms common to all {n_cong} congresses. Check name mapping coverage.")
        return

    # -- Pair matrix -----------------------------------------------------
    pairs = build_pair_matrix(edges_by_congress, set(stable_firms))
    n_stable_pairs = len(pairs)
    _n_cong      = len(CONGRESSES)
    n_appear_all = sum(1 for v in pairs.values() if len(v) == _n_cong)
    print(f"\n  Canonical pairs (any edge in stable set): {n_stable_pairs:,}")
    print(f"  Pairs present in all {_n_cong} sessions: {n_appear_all:,}")
    session_dist = pd.Series({k: sum(1 for v in pairs.values() if len(v) == k)
                               for k in range(1, _n_cong + 1)})
    print(f"  Session-count distribution: "
          + "  ".join(f"{k}:{v}" for k, v in session_dist.items() if v > 0))

    # -- Analysis 1: Direction consistency --------------------------------
    print(f"\n{'='*72}")
    print("ANALYSIS 1: DIRECTION CONSISTENCY")
    print("-" * 40)
    dir_df, dir_stats = run_direction_analysis(pairs)

    if dir_df.empty:
        print("  Insufficient directed pair data for direction analysis.")
    else:
        print(f"  Pairs with ≥{MIN_SESSIONS_DIRECTION} directed sessions: "
              f"{dir_stats['n_pairs']:,}")
        print(f"  Mean consistency:   {dir_stats['mean_consistency']:.4f}")
        print(f"  Median consistency: {dir_stats['median_consistency']:.4f}")
        print(f"  Pairs ≥ 0.80 consistent: "
              f"{dir_stats['n_geq_80']} / {dir_stats['n_pairs']}  "
              f"({dir_stats['n_geq_80']*100/max(dir_stats['n_pairs'],1):.1f}%)")
        print(f"  Perfectly consistent (1.00): "
              f"{dir_stats['n_perfect']} / {dir_stats['n_pairs']}  "
              f"({dir_stats['n_perfect']*100/max(dir_stats['n_pairs'],1):.1f}%)")
        print(f"\n  Binomial test (H0: each session is 50/50 coin flip):")
        print(f"  Pairs with p < 0.05: "
              f"{dir_stats['n_sig_binom']} / {dir_stats['n_pairs']}  "
              f"({dir_stats['pct_sig_binom']:.1f}%)  [expected under null: 5.0%]")
        print(f"  Pairs with majority-direction: "
              f"{dir_stats['pct_majority']:.1f}%  [expected under null: 50.0%]")

        # Top consistently directed pairs
        top_cons = (dir_df[dir_df["n_directed"] >= 3]
                    .sort_values("consistency", ascending=False)
                    .head(10))
        if not top_cons.empty:
            print(f"\n  Top 10 most direction-consistent pairs (n_directed ≥ 3):")
            print(f"    {'Firm A':<40} {'Firm B':<40} {'Cons.':>6}  "
                  f"{'n_dir':>5}  {'p(binom)':>10}")
            for _, row in top_cons.iterrows():
                print(f"    {row['firm_a']:<40} {row['firm_b']:<40} "
                      f"{row['consistency']:>6.3f}  {int(row['n_directed']):>5}  "
                      f"{row['binom_p']:>10.4f}")

    # -- Analysis 2: Magnitude stability ----------------------------------
    print(f"\n{'='*72}")
    print("ANALYSIS 2: MAGNITUDE STABILITY (net_temporal)")
    print("-" * 40)
    corr_matrix, p_matrix, n_matrix, w_edge, w_edge_p, n_w_edge = run_magnitude_analysis(pairs)

    print(f"\n  Spearman ρ between congresses (net_temporal, stable-set pairs):")
    print(f"  {'':>10}", end="")
    for c in CONGRESSES:
        print(f"  {c:>8}", end="")
    print()
    for ci in CONGRESSES:
        print(f"  {ci:>10}", end="")
        for cj in CONGRESSES:
            v = corr_matrix.at[str(ci), str(cj)]
            if pd.isna(v):
                print(f"  {'—':>8}", end="")
            elif ci == cj:
                print(f"  {'1.000':>8}", end="")
            else:
                p = p_matrix.at[str(ci), str(cj)]
                sig = "*" if (not pd.isna(p) and p < 0.05) else " "
                print(f"  {v:>7.3f}{sig}", end="")
        print()
    print(f"  (* p < 0.05)")

    print(f"\n  Pair counts per congress pair:")
    print(f"  {'':>10}", end="")
    for c in CONGRESSES:
        print(f"  {c:>8}", end="")
    print()
    for ci in CONGRESSES:
        print(f"  {ci:>10}", end="")
        for cj in CONGRESSES:
            if ci == cj:
                print(f"  {'—':>8}", end="")
            else:
                n = int(n_matrix.at[str(ci), str(cj)])
                print(f"  {n:>8}", end="")
        print()

    if w_edge is not None:
        print(f"\n  Kendall's W (net_temporal, pairs with ≥{MIN_SESSIONS_W} sessions): "
              f"W = {w_edge:.4f}  p = {w_edge_p:.4e}  n = {n_w_edge:,} pairs")
    else:
        print(f"\n  Kendall's W: insufficient pairs with ≥{MIN_SESSIONS_W} sessions.")

    # -- Analysis 3: Firm net_strength rank stability [primary] ──────────
    print(f"\n{'='*72}")
    print("ANALYSIS 3: FIRM NET_STRENGTH RANK STABILITY [PRIMARY]")
    print("-" * 40)
    str_spearman, w_str, w_str_p, n_str_w, str_wide = run_firm_stability(
        nodes_by_congress, set(stable_firms), metric="net_strength"
    )

    if not str_spearman.empty:
        print(f"\n  Adjacent-congress Spearman ρ (net_strength, {len(stable_firms)} stable firms):")
        for _, row in str_spearman.iterrows():
            sig = "  *" if row["p"] < 0.05 else "   "
            print(f"    {row['pair']:>8}:  ρ = {row['rho']:.4f}  "
                  f"p = {row['p']:.4e}  n = {row['n']}{sig}")
        print(f"  (* p < 0.05)")
    else:
        print("  Insufficient node-attribute data for net_strength Spearman analysis.")

    if w_str is not None:
        print(f"\n  Kendall's W across all {len(CONGRESSES)} congresses (firms with data in all {len(CONGRESSES)}): "
              f"W = {w_str:.4f}  p = {w_str_p:.4e}  n = {n_str_w} firms")
    else:
        print(f"\n  Kendall's W: insufficient firms with data across all {len(CONGRESSES)} congresses.")

    # Top/bottom firms by mean net_strength
    if nodes_by_congress.get(CONGRESSES[0]) is not None:
        means_str = str_wide[str_wide.notna().all(axis=1)].mean(axis=1)
        if not means_str.empty:
            top_str = means_str.nlargest(10)
            bot_str = means_str.nsmallest(5)
            print(f"\n  Top 10 highest mean net_strength firms (agenda-setters):")
            print(f"    {'Firm':<45}", end="")
            for c in CONGRESSES:
                print(f"  {c:>7}", end="")
            print(f"  {'Mean':>8}")
            for firm in top_str.index:
                print(f"    {firm:<45}", end="")
                for c in CONGRESSES:
                    v = str_wide.at[firm, c]
                    print(f"  {v:>7.3f}" if pd.notna(v) else f"  {'—':>7}", end="")
                print(f"  {top_str[firm]:>8.3f}")
            print(f"\n  Top 5 lowest mean net_strength firms (followers):")
            for firm in bot_str.index:
                print(f"    {firm:<45}", end="")
                for c in CONGRESSES:
                    v = str_wide.at[firm, c]
                    print(f"  {v:>7.3f}" if pd.notna(v) else f"  {'—':>7}", end="")
                print(f"  {bot_str[firm]:>8.3f}")

    # -- Analysis 4: Firm net_influence rank stability [reference] ────────
    print(f"\n{'='*72}")
    print("ANALYSIS 4: FIRM NET_INFLUENCE RANK STABILITY [REFERENCE]")
    print("-" * 40)
    firm_spearman, w_firm, w_firm_p, n_firm_w, firm_wide = run_firm_stability(
        nodes_by_congress, set(stable_firms), metric="net_influence"
    )

    if not firm_spearman.empty:
        print(f"\n  Adjacent-congress Spearman ρ (net_influence, {len(stable_firms)} stable firms):")
        for _, row in firm_spearman.iterrows():
            sig = "  *" if row["p"] < 0.05 else "   "
            print(f"    {row['pair']:>8}:  ρ = {row['rho']:.4f}  "
                  f"p = {row['p']:.4e}  n = {row['n']}{sig}")
        print(f"  (* p < 0.05)")
    else:
        print("  Insufficient node-attribute data for firm Spearman analysis.")

    if w_firm is not None:
        print(f"\n  Kendall's W across all {len(CONGRESSES)} congresses (firms with data in all {len(CONGRESSES)}): "
              f"W = {w_firm:.4f}  p = {w_firm_p:.4e}  n = {n_firm_w} firms")
    else:
        print(f"\n  Kendall's W: insufficient firms with data across all {len(CONGRESSES)} congresses.")

    # Top/bottom firms by mean net_influence (reference)
    if nodes_by_congress.get(CONGRESSES[0]) is not None:
        means_inf = firm_wide[firm_wide.notna().all(axis=1)].mean(axis=1)
        if not means_inf.empty:
            top_inf = means_inf.nlargest(10)
            bot_inf = means_inf.nsmallest(5)
            print(f"\n  Top 10 firms by mean net_influence (reference):")
            print(f"    {'Firm':<45}", end="")
            for c in CONGRESSES:
                print(f"  {c:>5}", end="")
            print(f"  {'Mean':>7}")
            for firm in top_inf.index:
                print(f"    {firm:<45}", end="")
                for c in CONGRESSES:
                    v = firm_wide.at[firm, c]
                    print(f"  {int(v) if pd.notna(v) else '—':>5}", end="")
                print(f"  {top_inf[firm]:>7.1f}")
            print(f"\n  Top 5 lowest mean net_influence firms (reference):")
            for firm in bot_inf.index:
                print(f"    {firm:<45}", end="")
                for c in CONGRESSES:
                    v = firm_wide.at[firm, c]
                    print(f"  {int(v) if pd.notna(v) else '—':>5}", end="")
                print(f"  {bot_inf[firm]:>7.1f}")

    # -- Plot ------------------------------------------------------------
    print(f"\n{'='*72}")
    print("OUTPUT")
    print("-" * 40)
    print(f"  Text → {TXT_PATH.name}")
    plot_results(
        dir_df, dir_stats, corr_matrix, n_matrix, pairs,
        str_spearman, w_str, w_str_p,           # panel 3: net_strength (primary)
        firm_spearman, w_firm, w_firm_p,         # panel 4: net_influence (reference)
        len(stable_firms), PNG_PATH
    )

    print(SEP)
    print("Done.")


if __name__ == "__main__":
    main()
