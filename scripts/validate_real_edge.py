"""Real-edge-on-qualifying-bets validation, extracted from the ad-hoc checks
run against US#173/US#192/US#193 into one reusable script. A model claims
an edge whenever predicted_prob - implied_prob >= edge_threshold; this
reports whether that claim survives contact with the real outcome rate,
not whether an in-sample/aggregate metric looks good. See
documents/user_stories.md's Phase 38 for the methodology this formalizes.
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
