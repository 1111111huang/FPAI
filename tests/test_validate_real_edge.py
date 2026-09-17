from __future__ import annotations

import pandas as pd
import pytest

from scripts.validate_real_edge import compute_real_edge


def test_real_edge_is_zero_when_calibrated_perfectly():
    df = pd.DataFrame({
        "predicted_prob": [0.30, 0.30, 0.30],
        "implied_prob": [0.30, 0.30, 0.30],
        "actual": [1, 0, 0],
    })
    result = compute_real_edge(df, "predicted_prob", "implied_prob", "actual", edge_threshold=0.05)
    assert result["n_qualifying"] == 0


def test_real_edge_flags_a_qualifying_bet_and_grades_it_correctly():
    df = pd.DataFrame({
        "predicted_prob": [0.50, 0.20],
        "implied_prob": [0.30, 0.20],
        "actual": [1, 0],
    })
    result = compute_real_edge(df, "predicted_prob", "implied_prob", "actual", edge_threshold=0.05)
    assert result["n_qualifying"] == 1
    assert result["claimed_edge"] == pytest.approx(0.20)
    assert result["actual_rate"] == pytest.approx(1.0)
    assert result["real_edge"] == pytest.approx(0.70)
