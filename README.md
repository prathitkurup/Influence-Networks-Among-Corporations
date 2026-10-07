# Corporate Lobbying Networks

Analysis of corporate lobbying behavior among Fortune 500 firms spanning the 111th through 117th Congresses (2009–2022), with the 116th Congress (2019–2020) as the primary analytical window. Uses OpenSecrets CRP bulk data. The primary analytical instrument is the **RBO directed influence network**.

---

## Core Idea

Two firms are "similar" if they lobby the same bills and rank those bills similarly by spend. A firm "influences" another if it consistently lobbied their shared bills *first* — setting the other firm's legislative agenda before the follower adopted those bills.

This agenda-setting signal is operationalized via temporal bill-adoption precedence. For each firm, bills are ranked in descending order of lobbying spend. For every pair of firms sharing at least one top-30 bill, we compute the Rank-Biased Overlap (RBO) of their ranked bill portfolios — capturing both which bills two firms co-lobby and how similarly they prioritize them, weighting agreement at the top of the ranking more heavily. If Firm A lobbied a bill in Q1 and Firm B adopted it in Q3, A receives a first-mover credit. Aggregating across all shared bills, RBO-weighted by portfolio alignment, yields the directed influence network.

A firm's **net\_strength** — the RBO-weighted sum of first-mover advantages over all pairings — is the primary measure of influence. Positive net\_strength identifies a net agenda-setter; negative identifies a net follower.


---

## Data

All data comes from **OpenSecrets CRP bulk lobbying tables**.

| File | Contents |
|---|---|
| `lob_lobbying.txt` | Report-level filings: registrant, client, amount |
| `lob_lobbyist.txt` | Named lobbyists per report |
| `lob_issue.txt` | Issue codes per report |
| `lob_bills.txt` | Bills per issue entry |

**Name mapping:** `data/manual_opensecrets_name_mapping.json` — manually curated JSON mapping Fortune 500 firm names to their OpenSecrets CRP name variants.

## Data Source

The Lobbying Disclosure Act (LDA) requires all organizations that lobby for federal legislation to publicly disclose these activities every fiscal quarter. We use OpenSecrets bulk lobbying data to access LDA reports. We focus on the 111th through 117th Congresses, spanning from January 2009 through January 2023. We choose this period because the data is available through OpenSecrets' free bulk disclosure files and also because the quarterly reporting standards introduced by the Honest Leadership and Open Government Act of 2007 remains consistent throughout. This facilitates methodological comparability across different Congress sessions. 

We particularly focus on the 116th Congress, which ran from January 2019 through January 2021. We select the 116th Congress as our primary analytical window because it spans both a stable pre-pandemic period and the onset of COVID-19, producing substantive variation in legislative priorities useful for testing the scope of our influence measure. Such changes in legislative priorities have been studied in the literature.

---

## Data Processing

Using the OpenSecrets bulk lobbying data, we extract valid LDA reports following the criteria in the OpenSecrets Data User Guide. We retain only reports flagged with the validity indicator *ind = y*, which identifies the final, correct filing for each report and removes superseded amendments. In addition, we retain only reports listing at least one named lobbyist.

We filter the sample to Fortune 500 companies using the 2019 Fortune 500 ranking (from the beginning of the 116th Congress). Kowal (2018) similarly restricts analysis to Fortune 500 firms, arguing that they constitute a representative cross-industry sample of the largest U.S. corporations with the most consistently disclosed lobbying activity and that data for smaller firms are not publicly available in sufficient detail. Matching LDA records with Fortune 500 firms requires resolving the many variants of client names, subsidiary names, and registrant names under which these large corporations file. We accomplish this through a hand-crafted mapping applied to the *ultorg* and *client* fields in the OpenSecrets data. This helps us identify all valid LDA activities by Fortune 500 companies and subsidiaries across different congressional sessions.

The result is a curated dataset with company, bill, and expenditure amounts as the primary dimensions. For each LDA report, we evenly distribute the reported dollar amount across all bills named in that report.

---

## Extraction Pipeline

`src/opensecrets_extraction.py` is the single entry point for all data. It filters to `ind='y'` records (OpenSecrets' validity flag excluding superseded filings and double-counted subsidiaries), retains only reports with at least one named lobbyist, and disaggregates reported dollar amounts evenly across all bills named in each report.

**Produces:**
- `data/opensecrets_lda_reports.csv` — one row per (report, bill); `amount_allocated = amount / n_bills`
- `data/opensecrets_lda_issues.csv` — one row per (report, issue_code)

---

## Networks

### RBO Directed Influence Network *(primary)*

**Script:** `src/rbo_directed_influence.py`

For every pair of Fortune 500 firms sharing at least one top-30 bill, two directed edges are produced:

- **Edge weight** = `[(source_firsts + ties/2) / shared_bills] × RBO` — source's fractional first-mover contribution weighted by alignment
- **Edge rbo** = full RBO (p=0.85) between the two firms' bill-priority rankings; same value for both edges of a pair; both edge weights sum to `rbo`
- **Edge net\_temporal** = `source_firsts − target_firsts` (signed; positive when source is the net first-mover)

Key node attributes:
- `net_strength` **(primary)**: `Σ_j [RBO(i,j) × net_temporal(i,j)]` — RBO-weighted temporal dominance
- `net_influence` (reference): total first-mover wins minus losses (unweighted)
- `wc_net_strength`: within-community variant, restricted to same-community neighbors

**Mega-bill filtering** (`MAX_BILL_DF = 50` in `config.py`): bills lobbied by more than 50 firms are excluded before computing similarity. These high-participation omnibus bills inflate similarity regardless of strategic alignment.

**Outputs:** `data/rbo_directed_influence.csv`, `data/ranked_bill_lists.csv`, `visualizations/gml/rbo_directed_influence.gml`, `visualizations/png/rbo_directed_influence.png`

**Enrichment:** `src/enrich_directed_gml.py` adds `num_bills`, `bill_aff_community`, `within_comm_net_str`, and `within_comm_net_inf` to the GML node attributes. Community labels (six Leiden communities: Finance/Insurance, Tech/Telecom, Energy/Utilities, Defense/Industrial, Health/Pharma, Consumer/Manufacturing) come from `src/bill_affiliation_network.py`.

**Gephi export:** `src/gephi_style_export.py` reads the enriched GML and writes a filtered, colored GEXF to `visualizations/gexf/rbo_directed_influence.gexf`.

---

### Bill Affiliation Network

**Script:** `src/bill_affiliation_network.py`

Undirected weighted network where edge weight = number of distinct co-lobbied bills. Used for Leiden community detection (six communities, γ=1.0) and as the structural basis for centrality comparisons. Outputs to `data/network_edges/`, `data/communities/`, `data/centralities/`, and `visualizations/gml/bill_affiliation_network.gml`.

---

## How to Run

The full pipeline is managed by `run_pipeline.sh` at the repository root:

```bash
bash run_pipeline.sh
```

This runs four phases in sequence: (1) core extraction and 116th Congress networks, (2) cross-congressional stability, (3) the focused analyses, and (4) the archived validation scripts (`src/archive/validations/`). Phase 1 requires the raw OpenSecrets files in `data/OpenSecrets/`, which are not included in this repository; Phase 3 runs from the processed data in `data/`. Each script is logged individually to `outputs/run_logs/`. The pipeline continues on failure and prints a pass/fail summary at the end.

**Notes:**
- `multi_congress_pipeline.py` must be run manually before the first full pipeline run — it takes 10–20 minutes and is commented out of the default script: `cd src && python multi_congress_pipeline.py`
- All parameters (resolution, top-K, RBO p, etc.) are constants at the top of each script; no CLI arguments

---

## Analyses

```bash
cd src/analysis
python 01_primary_directed_influence.py        # Top global and within-community agenda-setters, spend comparison, case studies
python 03_industry_hierarchy.py                # Kendall's W rank stability by sector, 111th–117th Congress
python 04_cross_congressional.py               # Spearman ρ heatmap, RBO list similarity across sessions
python 05_multi_congress.py                    # Jaccard top-set overlap, entry/exit transitions
python 06_centrality_vs_agenda_setters.py      # Centrality vs. net_strength Spearman ρ
python 07_strategic_complementarity_HC3.py     # Complementarity regression, HC3 standard errors
python 07_strategic_complementarity_cluster.py # Complementarity regression, (i, b, t)-clustered standard errors
```

All analysis outputs are written to `outputs/analysis/` (CSVs, TXT summaries, PNGs). Supporting validation scripts (extraction audit, mega-bill cutoff, RBO parameter calibration, congress statistics) are in `src/archive/validations/`, with outputs in `outputs/archive/validation/`.

Together, these analyses address the paper's core empirical questions: who sets the legislative agenda among Fortune 500 lobbying firms, whether that leadership is stable across industries and congressional sessions, how net strength differs from structural centrality, and whether the agenda-setting structure is consistent with strategic complementarity.

---

## Dependencies

```
pandas numpy networkx python-igraph leidenalg scikit-learn scipy matplotlib statsmodels
```

Install: `pip install pandas numpy networkx python-igraph leidenalg scikit-learn scipy matplotlib statsmodels`
