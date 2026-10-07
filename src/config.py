from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# OpenSecrets name mapping (manually curated; used by opensecrets_extraction.py)
MANUAL_OPENSECRETS_NAME_MAPPING = DATA_DIR / "manual_opensecrets_name_mapping.json"

OPENSECRETS_OUTPUT_CSV          = DATA_DIR / "opensecrets_lda_reports.csv"
OPENSECRETS_ISSUES_CSV          = DATA_DIR / "opensecrets_lda_issues.csv"
OPENSECRETS_LOBBYIST_CLIENT_CSV = DATA_DIR / "lobbyist_client_116_opensecrets.csv"

TARGET_CONGRESS       = 116
CONGRESS_FILING_YEARS = [2019, 2020]  # 116th Congress

# Exclude bills lobbied by more than MAX_BILL_DF firms (omnibus/appropriations
# bills create spurious edges with no strategic alignment signal; df>50 accounts
# for 97.5% of affiliation edges and collapses modularity from Q=0.18 to Q=0.02).
MAX_BILL_DF  = 50

# No frequency filter for issue codes (75 codes, broadly shared by design).
MAX_ISSUE_DF = None
