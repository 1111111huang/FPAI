"""Tests for the US#205 real-edge promotion gate.

_pool_and_grade is the actual pass/fail/inconclusive decision logic and is
tested directly (pure function, no DB/model training involved). check_real_edge's
fast-path (non-gated targets) is tested directly too. The DB-backed per-target
retrain-and-join path (_result_3way_edge/_btts_edge/_total_goals_edge) mirrors
scripts/sweep_total_goals_recency.py's already-manually-verified pattern and
is exercised for real via `select-best-models`, not re-fixtured here -- see
.claude/skills/promoting-a-model's own note on this being a deliberate scope
choice, not an oversight.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.real_edge_gate import (
    MIN_QUALIFYING_BETS,
    MIN_REAL_EDGE,
    _pool_and_grade,
    _resolve_model_type,
    _train_only_predictions,
    check_real_edge,
)


def _edge(n_qualifying: int, real_edge: float) -> dict:
    return {"n_qualifying": n_qualifying, "real_edge": real_edge, "claimed_edge": 0.0, "actual_rate": 0.0}


def test_pool_and_grade_inconclusive_below_min_qualifying_bets() -> None:
    edges = [_edge(5, -0.5), _edge(5, -0.5)]
    result = _pool_and_grade(edges, ["a", "b"])
    assert result.status == "inconclusive"
    assert result.n_qualifying == 10
    assert result.pooled_real_edge is None


def test_pool_and_grade_fails_below_the_real_edge_floor() -> None:
    n = MIN_QUALIFYING_BETS + 5
    edges = [_edge(n, MIN_REAL_EDGE - 0.10)]
    result = _pool_and_grade(edges, ["a"])
    assert result.status == "fail"
    assert result.pooled_real_edge == MIN_REAL_EDGE - 0.10


def test_pool_and_grade_passes_at_or_above_the_floor() -> None:
    n = MIN_QUALIFYING_BETS + 5
    edges = [_edge(n, 0.10)]
    result = _pool_and_grade(edges, ["a"])
    assert result.status == "pass"
    assert result.pooled_real_edge == 0.10


def test_pool_and_grade_weights_pooled_edge_by_qualifying_count() -> None:
    # 30 bets at +0.20 real edge, 10 bets at -0.20 -- pooled should skew toward
    # the larger, better-performing bucket, not a plain unweighted mean.
    edges = [_edge(30, 0.20), _edge(10, -0.20)]
    result = _pool_and_grade(edges, ["over_2.5", "under_2.5"])
    expected = (30 * 0.20 + 10 * -0.20) / 40
    assert result.pooled_real_edge == expected
    assert result.status == "pass"


def test_pool_and_grade_exposes_per_selection_detail_keyed_by_label() -> None:
    # The pooled verdict can hide a real problem in one named selection --
    # a caller re-validating one specific cell (e.g. total_goals/under_2.5)
    # needs the unpooled breakdown, not just the aggregate pass/fail.
    edges = [_edge(30, 0.20), _edge(10, -0.20)]
    result = _pool_and_grade(edges, ["over_2.5", "under_2.5"])
    assert result.by_selection["over_2.5"]["real_edge"] == 0.20
    assert result.by_selection["under_2.5"]["real_edge"] == -0.20
    assert result.by_selection["under_2.5"]["n_qualifying"] == 10


def test_check_real_edge_passes_through_non_gated_targets_without_training() -> None:
    result = check_real_edge("home_goals", "E0", ["OFF_HOME_FTHG_R5"], "xgb_regressor")
    assert result.status == "pass"
    assert result.n_qualifying == 0


def test_check_real_edge_is_inconclusive_for_an_unrebuildable_model_type() -> None:
    result = check_real_edge("btts", "E0", ["OFF_HOME_FTHG_R5"], "not_a_real_model_type")
    assert result.status == "inconclusive"
    assert "not_a_real_model_type" in result.detail


def test_resolve_model_type_normalizes_legacy_no_underscore_spellings() -> None:
    # Some already-recorded model_selection.yaml entries spell these without
    # underscores (e.g. tags.model_family predating a naming convention
    # change) -- the gate must still be able to rebuild them to re-validate
    # an already-live champion, not just freshly-trained candidates.
    assert _resolve_model_type("xgboostregressor") == "xgboost_regressor"
    assert _resolve_model_type("randomforestregressor") == "random_forest_regressor"
    assert _resolve_model_type("xgb_regressor") == "xgb_regressor"  # already exact, unchanged


def test_resolve_model_type_leaves_genuinely_unknown_strings_unchanged() -> None:
    assert _resolve_model_type("not_a_real_model_type") == "not_a_real_model_type"


def test_train_only_predictions_applies_time_decay_to_sample_weight() -> None:
    """Regression test: found live (2026-09-22) while sweeping half-lives for
    I1 btts that every half-life produced byte-identical predictions --
    _train_only_predictions built sample_weight via _compute_sample_weight
    but never threaded it through mgr._combine_time_decay(), unlike
    ModelManager.train()'s own equivalent line. A time_decay_half_life_days
    passed to ModelManager must actually reach model.train()."""
    mgr = MagicMock()
    mgr.prepare_training_data.return_value = (
        pd.DataFrame({"f": [1, 2, 3]}), pd.DataFrame({"f": [4]}), pd.DataFrame({"f": [5]}),
        pd.Series([0, 1, 0]), pd.Series([1]), pd.Series([0]),
        pd.DataFrame({"match_id": ["m1"]}),
    )
    mgr.target_definition.task_type = "binary_classification"
    mgr.sample_weight_alpha = 1.0
    mgr._combine_time_decay.return_value = "DECAYED_WEIGHTS"
    model = MagicMock()

    _train_only_predictions(mgr, model)

    mgr._combine_time_decay.assert_called_once()
    assert model.train.call_args.kwargs["sample_weight"] == "DECAYED_WEIGHTS"
