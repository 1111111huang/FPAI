from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logic.target_resolver import TargetResolver


@pytest.fixture
def match_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fthg": [2, 1, 0, 3],
            "ftag": [1, 1, 2, 0],
            "hc": [7, 4, 2, 5],
            "ac": [3, 4, 8, 1],
            "FTR": ["H", "D", "A", "H"],
        }
    )


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("home_win", [1, 0, 0, 1]),
        ("result_3way", ["home", "draw", "away", "home"]),
        ("btts", [1, 1, 0, 0]),
        ("home_goals", [2, 1, 0, 3]),
        ("away_goals", [1, 1, 2, 0]),
        ("total_goals", [3, 2, 2, 3]),
        ("home_corners", [7, 4, 2, 5]),
        ("away_corners", [3, 4, 8, 1]),
        ("total_corners", [10, 8, 10, 6]),
    ],
)
def test_target_resolver_generates_registry_labels(
    match_results: pd.DataFrame,
    target: str,
    expected: list[int | str],
) -> None:
    labels = TargetResolver.get_label(match_results, {"target_type": target})

    assert labels.tolist() == expected


def test_target_resolver_accepts_target_key_alias(match_results: pd.DataFrame) -> None:
    labels = TargetResolver.get_label(match_results, {"target": "both_teams_to_score"})

    assert labels.tolist() == [1, 1, 0, 0]


def test_target_resolver_reports_missing_label_columns() -> None:
    with pytest.raises(ValueError, match="Missing columns for total_corners label: ac"):
        TargetResolver.get_label(pd.DataFrame({"hc": [5]}), {"target_type": "total_corners"})
