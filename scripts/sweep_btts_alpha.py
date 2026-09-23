"""Leak-free sample_weight_alpha sweep for btts:no, all 5 leagues (US#210).

Every btts model has only ever trained at ModelManager's default
(sample_weight_alpha=1.0, full sklearn 'balanced' class weighting) --
result_3way's identical full-balancing default was found (US#172) to
overcorrect its minority class (draw) into a new over-prediction bug,
requiring a per-league dampened alpha (US#172's RESULT_3WAY_ALPHA table).
btts:no is the minority class in every league (yes base rate 50-67%) and
never got the equivalent check. Calibration (US#207) and recency
(time_decay_half_life_days, US#207/US#210) are both already ruled out with
real held-out evidence -- this is the one training-time lever neither of
those investigations touched.

Reuses src.utils.real_edge_gate.check_real_edge directly (train-only fit,
genuinely-unseen X_test, real oddspapi btts odds) rather than duplicating
its retrain/join logic -- same non-leaky pattern as
scripts/sweep_total_goals_recency.py, one call per (league, alpha).
"""
import sys
sys.path.insert(0, "/Users/tianqihuang/Documents/GitHub/FPAI")

import yaml

from src.utils.real_edge_gate import check_real_edge

ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
# Real promoted config per US#210's own numbers: E0/SP1/D1/F1 hand-promoted
# with half_life=180 (US#203); I1 reverted to the undecayed default (US#207).
HALF_LIFE = {"E0": 180, "SP1": 180, "D1": 180, "F1": 180, "I1": None}

with open("/Users/tianqihuang/Documents/GitHub/FPAI/config/model_selection.yaml") as f:
    selection = yaml.safe_load(f)

for league in ["E0", "SP1", "D1", "I1", "F1"]:
    entry = selection["contexts"][league]["btts"]
    print(f"\n=== {league} (model_type={entry['model_type']}, half_life={HALF_LIFE[league]}) ===")
    for alpha in ALPHAS:
        result = check_real_edge(
            "btts", league, entry["feature_subset"], entry["model_type"],
            time_decay_half_life_days=HALF_LIFE[league], sample_weight_alpha=alpha,
        )
        no = result.by_selection.get("no", {}) if result.by_selection else {}
        yes = result.by_selection.get("yes", {}) if result.by_selection else {}
        print(
            f"  alpha={alpha:.1f} | pooled={result.status:<11} "
            f"no: edge={no.get('real_edge', float('nan')):+.1%} n={no.get('n_qualifying', 0):<3} "
            f"actual_rate={no.get('actual_rate', float('nan')):.2f} | "
            f"yes: edge={yes.get('real_edge', float('nan')):+.1%} n={yes.get('n_qualifying', 0)}"
        )
