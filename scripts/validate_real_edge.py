"""Real-edge-on-qualifying-bets validation, extracted from the ad-hoc checks
run against US#173/US#192/US#193 into one reusable script. A model claims
an edge whenever predicted_prob - implied_prob >= edge_threshold; this
reports whether that claim survives contact with the real outcome rate,
not whether an in-sample/aggregate metric looks good. See
documents/user_stories.md's Phase 38 for the methodology this formalizes.

DATA LEAKAGE WARNING (found live, US#203, 2026-09-16): when validating a
refit_on_full_data candidate, do NOT feed this function predictions from
the SAVED ARTIFACT -- that artifact is fit on 100% of available rows
(train+val+test), so any evaluation window built from real market odds
(e.g. data/oddspapi_btts_corners_odds.json, which starts 2026-01-01) is
almost certainly INSIDE that fit if it's anywhere near the data's own
full_data_cutoff, i.e. in-sample evaluation dressed up as a held-out
check. Confirmed concretely for SP1 btts: X_test's own date range
(2025-01 to 2026-05) fully contains the 2026-01+ real-odds window.
Correct approach: call ModelManager.train() directly (train-only fit on
X_train, predictions on X_test) instead of run_pipeline() -- X_test is
genuinely unseen by that fit. This one mistake inflated a reported
+14.5pp->+25.0pp SP1 btts win down to a real, but smaller, +5.3pp->+12.2pp,
and had wrongly justified promoting I1's btts candidate (reverted).
"""

from __future__ import annotations

import pandas as pd


def compute_real_edge(
    df: pd.DataFrame, predicted_col: str, implied_col: str, actual_col: str, edge_threshold: float = 0.05,
) -> dict[str, float | int]:
    """actual_col must be 0/1 (1 = the predicted selection actually happened)."""
    edge = df[predicted_col] - df[implied_col]
    qualifying = df[edge >= edge_threshold]
    n = len(qualifying)
    if n == 0:
        return {"n_qualifying": 0, "claimed_edge": float("nan"), "actual_rate": float("nan"), "real_edge": float("nan")}
    claimed_edge = (qualifying[predicted_col] - qualifying[implied_col]).mean()
    actual_rate = qualifying[actual_col].mean()
    real_edge = actual_rate - qualifying[implied_col].mean()
    return {"n_qualifying": n, "claimed_edge": float(claimed_edge), "actual_rate": float(actual_rate), "real_edge": float(real_edge)}
