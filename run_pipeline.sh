#!/usr/bin/env bash
# run_pipeline.sh — Full pipeline runner for Lobbying Networks Research
# Run from the project root: bash run_pipeline.sh
# Logs per-script pass/fail; continues on failure.
# Raw OpenSecrets bulk files (data/OpenSecrets/) are not included in this repository;
# see README.md ("Data") for download instructions. Without them, skip Phase 1 and
# run Phase 3, which reads the processed files shipped under data/.
# Validation scripts are archived under src/archive/validations/.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$REPO_ROOT/src"
LOG_DIR="$REPO_ROOT/outputs/run_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_LOG="$LOG_DIR/pipeline_run_$TIMESTAMP.log"

PASS=0
FAIL=0
SKIPPED=0
FAILED_SCRIPTS=()

run_script() {
    local label="$1"
    local script="$2"
    local log="$LOG_DIR/${label}_$TIMESTAMP.log"

    echo -n "  [$label] ... "
    if python "$script" > "$log" 2>&1; then
        echo "PASS"
        ((PASS++))
        echo "PASS: $label" >> "$SUMMARY_LOG"
    else
        echo "FAIL  (see $log)"
        ((FAIL++))
        FAILED_SCRIPTS+=("$label")
        echo "FAIL: $label  →  $log" >> "$SUMMARY_LOG"
    fi
}

hr() { printf '%.0s─' {1..70}; echo; }

echo "" | tee -a "$SUMMARY_LOG"
hr
echo "  Lobbying Networks Pipeline  |  $(date)" | tee -a "$SUMMARY_LOG"
echo "  Logs → $LOG_DIR" | tee -a "$SUMMARY_LOG"
hr

# ── Phase 1: Core extraction and 116th Congress network ──────────────────────
echo ""
echo "Phase 1: Core extraction + directed influence network"
hr

cd "$SRC"
run_script "01_extraction"         "opensecrets_extraction.py"
run_script "02_bill_affiliation"   "bill_affiliation_network.py"
run_script "03_rbo_directed"       "rbo_directed_influence.py"
run_script "04_enrich_gml"         "enrich_directed_gml.py"
run_script "05_gephi_export"       "gephi_style_export.py"

# ── Phase 2: Multi-congress pipeline (111th–117th) ───────────────────────────
echo ""
echo "Phase 2: Multi-congress pipeline (111th–117th, ~10–20 min)"
hr

# Uncomment to regenerate data/congress/{111..117}/ from the raw OpenSecrets files:
# run_script "06_multi_congress"     "multi_congress_pipeline.py"
run_script "07_xc_stability"       "cross_congressional_stability.py"

# ── Phase 3: Focused analyses (paper results) ────────────────────────────────
echo ""
echo "Phase 3: Focused analyses (src/analysis/)"
hr

cd "$SRC/analysis"
run_script "A01_primary_influence"    "01_primary_directed_influence.py"
run_script "A03_industry_hierarchy"   "03_industry_hierarchy.py"
run_script "A04_cross_congressional"  "04_cross_congressional.py"
run_script "A05_multi_congress"       "05_multi_congress.py"
run_script "A06_centrality"           "06_centrality_vs_agenda_setters.py"
run_script "A07_complementarity_HC3"  "07_strategic_complementarity_HC3.py"
run_script "A07_complementarity_clus" "07_strategic_complementarity_cluster.py"

# ── Phase 4: Supporting validations (archived) ───────────────────────────────
echo ""
echo "Phase 4: Validation scripts (archived — src/archive/validations/)"
hr

cd "$SRC"
run_script "V01_extraction_audit"      "archive/validations/01_extraction_audit.py"
run_script "V04_mega_bill_diagnosis"   "archive/validations/04_mega_bill_diagnosis.py"
run_script "V05_ind_filter"            "archive/validations/05_ind_filter_validation.py"
run_script "V06_rbo_cosine_unit_tests" "archive/validations/06_rbo_cosine_unit_tests.py"
run_script "V08_rbo_p_calibration"     "archive/validations/08_rbo_p_calibration.py"
run_script "V10_directed_validation"   "archive/validations/10_rbo_directed_influence_validation.py"
run_script "V12_congress_statistics"   "archive/validations/12_congress_statistics.py"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
hr
echo "  Results: $PASS passed  |  $FAIL failed  |  $SKIPPED skipped"
if [ ${#FAILED_SCRIPTS[@]} -gt 0 ]; then
    echo "  Failed:"
    for s in "${FAILED_SCRIPTS[@]}"; do
        echo "    ✗ $s"
    done
fi
echo "  Full summary → $SUMMARY_LOG"
hr
echo ""
