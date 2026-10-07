# Validation scripts for the Fortune 500 lobbying network project.
# Each script is self-contained and can be run from the src/ directory:
#   python validations/01_extraction_audit.py
#
# Run order:
#   01 → extraction audit (understand raw data structure)
#   02 → inflation diagnosis (quantify and fix cartesian product bug)
#   03 → sparsity analysis (null model, 27x above expectation finding)
#   04 → mega_bill diagnosis (prevalence filtering rationale)
#   05 → issue_score_range (weight bounds, sqrt normalisation proof)
#   06 → rbo_cosine_unit_tests (unit tests for RBO and cosine helpers)
#   07 → composite_network_validation (24 checks: composite formula,
#         triple-filter sparsity, Katz centrality convergence)
#
# Outputs are written to outputs/archive/validation/ as human-readable .txt files.
