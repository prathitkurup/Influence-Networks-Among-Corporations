"""
Strategic complementarity regression (116th Congress).

  Δlog L_{ib,t+1} = β_E·entry_j + β_S·rbo_ij + β_ES·(entry_j × rbo_ij) + α_ib + γ_t + ε

Unit: (focal firm i, peer j, bill, quarter). Firm-bill and quarter FE.
SE: HC3, no clustering (replicates the professor's calculation: dedup only, original row-count filter).
β_ES > 0: the spending response to a peer's entry rises with RBO similarity (complementarity).
Specs A–C use 116th RBO (full, ≥p75, <p25); D–F repeat them with lagged 115th RBO.

Outputs (outputs/analysis/): 07_complementarity_regression_HC3.csv, 07_complementarity_regression_partd_HC3.csv,
                             07_robustness_coef_plot_HC3.png, 07_strategic_complementarity_HC3.txt
The unrelated direction persistence/consistency analyses are kept above main(), disabled.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

CONGRESS     = 116   # spending data and contemporaneous RBO
CONGRESS_LAG = 115   # lagged RBO for specs D–F
HIGH_RBO_Q   = 75    # percentile cutoff, high-RBO spec
LOW_RBO_Q    = 25    # percentile cutoff, low-RBO spec
MIN_OBS      = 50    # skip specs with fewer rows

LABELS     = ["A — full RBO-linked", "B — high-RBO (≥p75)", "C — low-RBO (<p25)"]
LABELS_LAG = ["D — full (lagged RBO)", "E — high-RBO lag (≥p75)", "F — low-RBO lag (<p25)"]

OUT_DIR = ROOT / "outputs" / "analysis"

# ---------------------------------------------------------------------------
# Tee helper
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, t):
        for s in self.streams: s.write(t)
    def flush(self):
        for s in self.streams: s.flush()

# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def assign_quarters(df):
    """Add 'quarter' (1–8): 2019 Q1–Q4 → 1–4, 2020 Q1–Q4 → 5–8."""
    df = df.copy()
    df["quarter"] = df["report_type"].str[1].astype(int) + df["year"].map({2019: 0, 2020: 4})
    return df

def build_spend_panel(df_raw):
    """Aggregate to total spend per (firm, bill, quarter)."""
    return (
        df_raw.groupby(["fortune_name", "bill_number", "quarter"])["amount_allocated"]
        .sum().reset_index()
        .rename(columns={"fortune_name": "firm", "bill_number": "bill", "amount_allocated": "spend"})
    )

def tag_entry_events(panel):
    """Add entry_j = 1 in the first quarter a firm lobbies a bill."""
    first_q = panel.groupby(["firm", "bill"])["quarter"].transform("min")
    return panel.assign(entry_j=(panel["quarter"] == first_q).astype(int))

def build_delta_log_spend(panel):
    """Δlog spend from t to t+1 for consecutive quarters with positive spend in both."""
    srt = panel.sort_values(["firm", "bill", "quarter"]).copy()
    srt["spend_next"]   = srt.groupby(["firm", "bill"])["spend"].shift(-1)
    srt["quarter_next"] = srt.groupby(["firm", "bill"])["quarter"].shift(-1)
    ok = (srt["quarter_next"] == srt["quarter"] + 1) & (srt["spend"] > 0) & (srt["spend_next"] > 0)
    srt = srt[ok].copy()
    srt["delta_log_spend"] = np.log(srt["spend_next"]) - np.log(srt["spend"])
    return srt[["firm", "bill", "quarter", "delta_log_spend"]].rename(columns={"firm": "firm_i"})

def build_rbo_lookup(rbo_df):
    """Ordered (i,j) RBO lookup; the edge CSV already holds both directions, so drop repeats."""
    cols = ["source", "target", "rbo"]
    fwd = rbo_df[cols].rename(columns={"source": "firm_i", "target": "firm_j"})
    rev = rbo_df[cols].rename(columns={"source": "firm_j", "target": "firm_i"})
    return pd.concat([fwd, rev]).drop_duplicates(["firm_i", "firm_j"]).reset_index(drop=True)

def build_regression_panel(delta_df, entry_df, rbo_lookup):
    """Cross each (firm_i, bill, quarter) with the peers j active on that bill; attach RBO."""
    peers = entry_df[["firm", "bill", "quarter", "entry_j"]].rename(columns={"firm": "firm_j"})
    panel = delta_df.merge(peers, on=["bill", "quarter"])
    panel = panel[panel["firm_i"] != panel["firm_j"]].merge(rbo_lookup, on=["firm_i", "firm_j"])
    panel = panel.rename(columns={"rbo": "rbo_ij"})
    panel["entry_x_rbo"] = panel["entry_j"] * panel["rbo_ij"]
    panel["firm_bill"]   = panel["firm_i"] + "||" + panel["bill"]
    assert not panel.duplicated(["firm_i", "firm_j", "bill", "quarter"]).any(), "duplicate dyad rows"
    return panel.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def run_ols_spec(df):
    """Within-transformed OLS (firm-bill + quarter FE); HC3 SE, rows treated as independent."""
    df = df[df.groupby("firm_bill")["firm_bill"].transform("count") > 1]   # drop single-row groups
    quarters = pd.get_dummies(df["quarter"], prefix="q", drop_first=True).astype(float)
    X = pd.concat([df[["entry_j", "rbo_ij", "entry_x_rbo"]], quarters], axis=1)
    groups = df["firm_bill"]
    X_dm = X - X.groupby(groups).transform("mean")
    y_dm = df["delta_log_spend"] - df.groupby("firm_bill")["delta_log_spend"].transform("mean")
    res = sm.OLS(y_dm, X_dm).fit(cov_type="HC3")
    return res, len(df), groups.nunique()

def fit_spec(df, label):
    """Fit one spec and return its results row."""
    res, n, n_groups = run_ols_spec(df)
    r = lambda x: round(float(x), 5)
    return {
        "spec": label, "n": n, "n_groups": n_groups,
        "coef_entry_j":     r(res.params["entry_j"]),
        "coef_rbo_ij":      r(res.params["rbo_ij"]),
        "se_rbo_ij":        r(res.bse["rbo_ij"]),
        "p_rbo_ij":         r(res.pvalues["rbo_ij"]),
        "coef_entry_x_rbo": r(res.params["entry_x_rbo"]),
        "se_entry_x_rbo":   r(res.bse["entry_x_rbo"]),
        "p_entry_x_rbo":    r(res.pvalues["entry_x_rbo"]),
        "r2":               r(res.rsquared),
    }

def run_regressions(delta_df, entry_df, rbo_congress, labels):
    """Build the dyad panel with RBO from rbo_congress; fit the full, high- and low-RBO specs."""
    rbo   = pd.read_csv(DATA_DIR / f"congress/{rbo_congress}/rbo_directed_influence.csv")
    panel = build_regression_panel(delta_df, entry_df, build_rbo_lookup(rbo))
    p25, p75 = np.percentile(panel["rbo_ij"], [LOW_RBO_Q, HIGH_RBO_Q])
    subsets = [panel, panel[panel["rbo_ij"] >= p75], panel[panel["rbo_ij"] < p25]]
    rows = [fit_spec(df, lab) for df, lab in zip(subsets, labels) if len(df) >= MIN_OBS]
    info = {"rows": len(panel), "firms": panel["firm_i"].nunique(),
            "groups": panel["firm_bill"].nunique(), "p25": p25, "p75": p75}
    return pd.DataFrame(rows), info

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

def print_results(reg, reg_lag, info, info_lag):
    """Print one results table covering contemporaneous and lagged RBO."""
    print("=" * 88)
    print(f"STRATEGIC COMPLEMENTARITY — {CONGRESS}th Congress")
    print("Δlog L_ib,t+1 ~ entry_j + rbo_ij + entry_j×rbo_ij | firm-bill + quarter FE | HC3 SE (no clustering)")
    print("β_ES > 0: spending response to peer entry rises with RBO similarity (complementarity)")
    print("=" * 88)
    for tag, i in [(f"RBO {CONGRESS}th", info), (f"RBO {CONGRESS_LAG}th (lagged)", info_lag)]:
        print(f"{tag:<18}{i['rows']:>7,} dyad rows | {i['firms']} focal firms | "
              f"{i['groups']:,} firm-bill groups | RBO p25={i['p25']:.4f}, p75={i['p75']:.4f}")
    print(f"\n{'Spec':<26}{'N':>8}{'Groups':>10}{'β_E':>9}{'β_S':>9}{'β_ES':>9}{'SE':>9}{'p':>8}")
    for df in (reg, reg_lag):
        print("-" * 88)
        for r in df.itertuples():
            print(f"{r.spec:<26}{r.n:>8,}{r.n_groups:>10,}{r.coef_entry_j:>9.3f}{r.coef_rbo_ij:>9.3f}"
                  f"{r.coef_entry_x_rbo:>9.3f}{r.se_entry_x_rbo:>9.3f}{r.p_entry_x_rbo:>8.3f} {stars(r.p_entry_x_rbo)}")
    print("-" * 88)
    print("N: rows in firm-bill groups with ≥2 rows. SE and p refer to β_ES. "
          "* p<0.05  ** p<0.01  *** p<0.001")

def plot_robustness(reg, reg_lag, out_dir):
    """Forest plot of β_ES, contemporaneous vs lagged RBO (low-RBO on its own axis)."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    series = [(reg, "#C44E52", "o", -0.1, f"Contemporaneous RBO ({CONGRESS}th)"),
              (reg_lag, "#4C72B0", "s", 0.1, f"Lagged RBO ({CONGRESS_LAG}th)")]
    for ax, rows, names in [(axes[0], [0, 1], ["Full sample", "High-RBO (≥p75)"]),
                            (axes[1], [2], ["Low-RBO (<p25)"])]:
        for df, color, marker, off, lab in series:
            sub = df.reindex(rows)
            ax.errorbar(np.arange(len(rows)) + off, sub["coef_entry_x_rbo"], yerr=1.96 * sub["se_entry_x_rbo"],
                        fmt=marker, color=color, capsize=5, markersize=7, label=lab)
        ax.axhline(0, color="#888888", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names)
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("β_ES  (entry_j × RBO)")
    axes[0].legend(fontsize=9, framealpha=0.85)
    fig.suptitle("Strategic complementarity: β_ES across specifications\n"
                 "95% CI, HC3 SE", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out_dir / "07_robustness_coef_plot_HC3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Unrelated to the regression: direction persistence and consistency (111th–117th)
# Disabled. To re-run, uncomment this block and the run_direction_analyses() call in main().
# ---------------------------------------------------------------------------
# from scipy.stats import fisher_exact
#
# CONGRESSES = [111, 112, 113, 114, 115, 116, 117]
#
# def plot_persistence(pers, out_dir):
#     """Bar chart: high- vs low-RBO direction persistence per consecutive congress pair."""
#     plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
#     x, w = np.arange(len(pers)), 0.35
#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.bar(x - w / 2, pers["high_rbo_persist_rate"], w, color="#C44E52", edgecolor="white", label="High-RBO (Q4)")
#     ax.bar(x + w / 2, pers["low_rbo_persist_rate"], w, color="#4C72B0", edgecolor="white", label="Low-RBO (Q1)")
#     for xi, r in zip(x, pers.itertuples()):
#         if r.fisher_p < 0.05:
#             ax.text(xi, max(r.high_rbo_persist_rate, r.low_rbo_persist_rate) + 0.012, stars(r.fisher_p),
#                     ha="center", fontsize=11, fontweight="bold")
#     mean = pers["persist_rate"].mean()
#     ax.axhline(mean, color="#888888", linestyle="--", linewidth=1.2, label=f"Overall mean ({mean:.2%})")
#     ax.set_xticks(x)
#     ax.set_xticklabels(pers["pair"], rotation=20, ha="right", fontsize=9)
#     ax.set_ylabel("Direction persistence rate")
#     ax.set_title("Direction persistence: high- vs low-RBO pairs (111th–117th)\n"
#                  "Pairs decisive in both sessions; Fisher exact, * p<0.05 ** p<0.01 *** p<0.001", fontsize=10)
#     ax.legend(fontsize=9, framealpha=0.85)
#     ax.grid(axis="y", alpha=0.25, linestyle="--")
#     ax.spines[["top", "right"]].set_visible(False)
#     fig.tight_layout()
#     fig.savefig(out_dir / "07_persistence_bar.png", dpi=150, bbox_inches="tight")
#     plt.close(fig)
#
# def plot_consistency(cons, out_dir):
#     """Histogram of per-pair direction consistency scores."""
#     plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
#     s = cons["consistency"]
#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     ax.hist(s, bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
#     ax.axvline(s.median(), color="#C44E52", linewidth=2, label=f"Median = {s.median():.2f}")
#     ax.axvline(s.mean(), color="#DD8452", linewidth=1.8, linestyle="--", label=f"Mean = {s.mean():.2f}")
#     ax.axvline(0.5, color="#AAAAAA", linewidth=1.2, linestyle=":", label="Chance (0.5)")
#     ax.set_xlabel("Direction consistency (share of sessions led by the majority-direction firm)")
#     ax.set_ylabel("Number of pairs")
#     ax.set_title(f"Direction consistency, 111th–117th (n={len(s):,} pairs with ≥2 decisive sessions; "
#                  f"{(s >= 0.8).mean():.0%} ≥ 0.80)", fontsize=10)
#     ax.legend(fontsize=9, framealpha=0.85)
#     ax.grid(axis="y", alpha=0.25, linestyle="--")
#     ax.spines[["top", "right"]].set_visible(False)
#     fig.tight_layout()
#     fig.savefig(out_dir / "07_consistency_hist.png", dpi=150, bbox_inches="tight")
#     plt.close(fig)
#
# def run_persistence():
#     """Fisher test per consecutive congress pair: do high-RBO (Q4) pairs keep their leader more than low-RBO (Q1)?
#     Pairs are oriented source < target and led by source in the first congress; the test uses pairs decisive in both."""
#     rows = []
#     for ci, cj in zip(CONGRESSES[:-1], CONGRESSES[1:]):
#         paths = [DATA_DIR / f"congress/{c}/rbo_directed_influence.csv" for c in (ci, cj)]
#         if not all(p.exists() for p in paths):
#             continue
#         ei, ej = (pd.read_csv(p) for p in paths)
#         if "rbo" not in ei.columns or "rbo" not in ej.columns:
#             continue
#         lead = ei[(ei["source"] < ei["target"]) & (ei["net_temporal"] > 0)][["source", "target", "rbo"]]
#         nxt  = ej[ej["source"] < ej["target"]].set_index(["source", "target"])["net_temporal"].rename("nt_next")
#         m = lead.join(nxt, on=["source", "target"])
#         m["quartile"] = pd.qcut(m["rbo"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
#         m["persists"] = m["nt_next"] > 0
#         both = m["nt_next"].notna() & (m["nt_next"] != 0)               # decisive in both sessions
#         hi = m.loc[(m["quartile"] == "Q4") & both, "persists"]
#         lo = m.loc[(m["quartile"] == "Q1") & both, "persists"]
#         if len(hi) < 3 or len(lo) < 3:
#             continue
#         _, p = fisher_exact([[hi.sum(), (~hi).sum()], [lo.sum(), (~lo).sum()]], alternative="greater")
#         n_persist, n_both = int(m["persists"].sum()), int(both.sum())
#         rows.append({
#             "pair": f"{ci}->{cj}", "n_decisive": len(m), "n_reappear": int(m["nt_next"].notna().sum()),
#             "n_decisive_both": n_both, "n_persist": n_persist,
#             "persist_rate": round(n_persist / max(n_both, 1), 4),
#             "high_rbo_persist_rate": round(float(hi.mean()), 4),
#             "low_rbo_persist_rate": round(float(lo.mean()), 4),
#             "fisher_p": round(p, 5),
#         })
#         print(f"  {ci}→{cj}: persist {n_persist:,}/{n_both:,} ({n_persist / max(n_both, 1):.1%}) | "
#               f"high-RBO {hi.mean():.3f} vs low-RBO {lo.mean():.3f} | Fisher p={p:.4f}")
#     return pd.DataFrame(rows)
#
# def run_consistency():
#     """Per-pair direction consistency: share of decisive sessions led by the pair's majority-direction firm (≥2 sessions)."""
#     recs = []
#     for c in CONGRESSES:
#         path = DATA_DIR / f"congress/{c}/rbo_directed_influence.csv"
#         if not path.exists():
#             continue
#         df = pd.read_csv(path)
#         if "net_temporal" not in df.columns:
#             continue
#         dec = df[(df["source"] < df["target"]) & (df["net_temporal"] != 0)][["source", "target", "net_temporal"]]
#         recs.append(dec.assign(congress=c))
#     if not recs:
#         return pd.DataFrame()
#     recs = pd.concat(recs, ignore_index=True)
#     recs["a_leads"] = (recs["net_temporal"] > 0).astype(int)
#     agg = (recs.groupby(["source", "target"])
#            .agg(n_sessions=("congress", "count"), n_leads=("a_leads", "sum")).reset_index())
#     agg = agg[agg["n_sessions"] >= 2].copy()
#     agg["n_follows"]   = agg["n_sessions"] - agg["n_leads"]
#     agg["consistency"] = agg[["n_leads", "n_follows"]].max(axis=1) / agg["n_sessions"]   # in [0.5, 1.0]
#     agg = agg.sort_values("consistency", ascending=False).reset_index(drop=True)
#     s = agg["consistency"]
#     print(f"  {len(s):,} pairs with ≥2 decisive sessions | mean {s.mean():.3f}, median {s.median():.3f} | "
#           f"≥0.80: {(s >= 0.8).mean():.1%} | =1.00: {(s == 1).mean():.1%}")
#     return agg
#
# def run_direction_analyses(out_dir):
#     """Run the persistence and consistency analyses; save CSVs and figures."""
#     print("\nDIRECTION PERSISTENCE (consecutive congresses)")
#     pers = run_persistence()
#     if not pers.empty:
#         pers.to_csv(out_dir / "07_persistence_summary.csv", index=False)
#         plot_persistence(pers, out_dir)
#     print("\nDIRECTION CONSISTENCY (all congresses)")
#     cons = run_consistency()
#     if not cons.empty:
#         cons.to_csv(out_dir / "07_direction_consistency.csv", index=False)
#         plot_consistency(cons, out_dir)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(OUT_DIR / "07_strategic_complementarity_HC3.txt", "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    raw = assign_quarters(pd.read_csv(DATA_DIR / f"congress/{CONGRESS}/opensecrets_lda_reports.csv"))
    spend    = build_spend_panel(raw)
    delta_df = build_delta_log_spend(spend)
    entry_df = tag_entry_events(spend)

    reg,     info     = run_regressions(delta_df, entry_df, CONGRESS,     LABELS)
    reg_lag, info_lag = run_regressions(delta_df, entry_df, CONGRESS_LAG, LABELS_LAG)
    reg.to_csv(OUT_DIR / "07_complementarity_regression_HC3.csv", index=False)
    reg_lag.to_csv(OUT_DIR / "07_complementarity_regression_partd_HC3.csv", index=False)

    print_results(reg, reg_lag, info, info_lag)
    plot_robustness(reg, reg_lag, OUT_DIR)
    # run_direction_analyses(OUT_DIR)  # unrelated analyses; disabled block above

    print(f"\nSaved to {OUT_DIR.relative_to(ROOT)}")
    log_f.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
