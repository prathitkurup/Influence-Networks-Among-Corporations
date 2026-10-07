"""
Validation 06: Unit tests for rbo_score() and cosine similarity helpers.

Verifies:
  1. RBO(L, L, p) = correct theoretical value for perfect agreement.
  2. RBO(L1, L2) = 0 when lists have no common elements.
  3. RBO respects top-weight: swapping rank-1 hurts more than swapping rank-5.
  4. build_frac_matrix produces correct shapes and no negative values.
  5. Cosine similarity is symmetric and self-similarity = 1.0.
  6. Both metrics are insensitive to firm ordering (canonical pair direction).

Design decision:
  RBO is preferred over plain Jaccard or Kendall-tau for bill priority rankings
  because its top-weighting matches the economic reality that a firm's top-3
  lobbying bills (where most spend is concentrated) are more informative about
  strategy than the long tail of low-spend bills. See utils/similarity.py
  for the full mathematical definition.
  Validation script: 06_rbo_cosine_unit_tests.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

from utils.similarity import (rbo_score, build_ranked_lists,
                               build_frac_matrix, compute_zero_budget_fracs,
                               aggregate_per_firm_bill)
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_PATH = Path(__file__).resolve().parents[3] / "outputs" / "archive" / "validation" / "06_rbo_cosine_unit_tests.txt"

class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for s in self.streams: s.write(text)
    def flush(self):
        for s in self.streams: s.flush()

PASS, FAIL = 0, 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        FAIL += 1

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
_orig_stdout = sys.stdout
_f = open(OUTPUT_PATH, "w")
sys.stdout = _Tee(_orig_stdout, _f)

try:
    print("=" * 55)
    print("Validation 06: RBO and cosine similarity unit tests")
    print("=" * 55)

    # -- Test 1: RBO perfect agreement (3 items, p=0.9)
    # Expected = (1-0.9)*(1 + 0.9 + 0.81) = 0.271
    expected = round((1 - 0.9) * (1 + 0.9 + 0.81), 4)
    got = round(rbo_score(["A","B","C"], ["A","B","C"], p=0.90), 4)
    check("RBO perfect 3-item agreement", got == expected, f"expected {expected}, got {got}")

    # -- Test 2: RBO disjoint lists = 0
    check("RBO disjoint lists = 0", rbo_score(["A","B"], ["C","D"], p=0.90) == 0.0)

    # -- Test 3: RBO empty list = 0
    check("RBO empty list = 0", rbo_score([], ["A","B"], p=0.90) == 0.0)

    # -- Test 4: Top-weight property: swapping rank-1 hurts more than swapping rank-5
    base  = ["A","B","C","D","E"]
    swap1 = ["X","B","C","D","E"]   # rank-1 differs
    swap5 = ["A","B","C","D","X"]   # rank-5 differs
    rbo_base  = rbo_score(base, base,  p=0.90)
    rbo_swap1 = rbo_score(base, swap1, p=0.90)
    rbo_swap5 = rbo_score(base, swap5, p=0.90)
    check("RBO top-weight: rank-1 swap hurts more than rank-5",
          rbo_swap1 < rbo_swap5,
          f"swap1={rbo_swap1:.4f} swap5={rbo_swap5:.4f}")

    # -- Test 5: RBO symmetry
    check("RBO symmetry",
          abs(rbo_score(["A","B","C"], ["A","C","B"], p=0.90) -
              rbo_score(["A","C","B"], ["A","B","C"], p=0.90)) < 1e-9)

    # -- Test 6: build_frac_matrix shape and non-negativity
    rows = [("F1","b1",0.6),("F1","b2",0.4),("F2","b1",1.0),("F3","b3",1.0)]
    df = pd.DataFrame(rows, columns=["fortune_name","bill_number","frac"])
    pivot, firms, bills = build_frac_matrix(df)
    check("build_frac_matrix shape (3 firms x 3 bills)", pivot.shape == (3, 3))
    check("build_frac_matrix no negatives", (pivot.values >= 0).all())
    check("build_frac_matrix zero-fill missing pairs",
          pivot.loc["F2","b3"] == 0.0 and pivot.loc["F3","b1"] == 0.0)

    # -- Test 7: Cosine self-similarity = 1.0
    mat = pivot.values.astype(np.float64)
    sim = cosine_similarity(mat)
    check("Cosine self-similarity = 1.0", np.allclose(np.diag(sim), 1.0))

    # -- Test 8: Cosine symmetry
    check("Cosine symmetry (sim[i,j] == sim[j,i])", np.allclose(sim, sim.T))

    # -- Test 9: Cosine disjoint firms = 0
    # F2 has only b1; F3 has only b3 -- no overlap
    f2_idx, f3_idx = firms.index("F2"), firms.index("F3")
    check("Cosine disjoint firms = 0", sim[f2_idx, f3_idx] == 0.0)

    # -- Test 10: build_ranked_lists ordering
    df_r = pd.DataFrame([("X","b1",500),("X","b2",200),("X","b3",1000)],
                        columns=["fortune_name","bill_number","amount_allocated"])
    ranked = build_ranked_lists(df_r, top_bills=100)
    check("build_ranked_lists sorted by spend descending",
          ranked["X"] == ["b3","b1","b2"])

    print()
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 55)

finally:
    sys.stdout = _orig_stdout
    _f.close()

if FAIL > 0:
    sys.exit(1)
