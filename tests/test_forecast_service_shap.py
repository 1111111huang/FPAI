"""Tests for per-prediction SHAP attribution in _predict_target.

Computed inline, in the same call that produces the prediction, so it can
never explain a different number than the one actually served -- an
after-the-fact reconstruction isn't guaranteed to match (confirmed live
reverse-engineering a BTTS prediction whose feature snapshot had already
moved on by the time it was investigated).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.forecast.forecast_service import ForecastService, _compute_shap_contributions
from src.logic.target_registry import get_target_definition


def _toy_binary_model() -> XGBClassifier:
    rng = np.random.RandomState(0)
    X = pd.DataFrame({
        "MKT_IMPLIED_HOME": rng.uniform(0.2, 0.6, 60),
        "OFF_HOME_XG_R5": rng.uniform(0.5, 2.5, 60),
        "DEF_AWAY_XGA_R5": rng.uniform(0.5, 2.5, 60),
    })
    y = (X["OFF_HOME_XG_R5"] - X["DEF_AWAY_XGA_R5"] + rng.normal(0, 0.1, 60) > 0).astype(int)
    clf = XGBClassifier(n_estimators=10, max_depth=2)
    clf.fit(X, y)
    return clf


def _toy_regressor_model() -> XGBRegressor:
    rng = np.random.RandomState(0)
    X = pd.DataFrame({
        "MKT_LAMBDA_TOTAL": rng.uniform(1.5, 3.5, 60),
        "OFF_HOME_XG_R5": rng.uniform(0.5, 2.5, 60),
    })
    y = X["MKT_LAMBDA_TOTAL"] + rng.normal(0, 0.1, 60)
    reg = XGBRegressor(n_estimators=10, max_depth=2)
    reg.fit(X, y)
    return reg


def test_compute_shap_contributions_ranks_by_absolute_value():
    clf = _toy_binary_model()
    row = pd.DataFrame({"MKT_IMPLIED_HOME": [0.4], "OFF_HOME_XG_R5": [2.4], "DEF_AWAY_XGA_R5": [0.6]})

    contributions = _compute_shap_contributions(clf, row, top_n=3)

    assert contributions is not None
    assert len(contributions) == 3
    magnitudes = [abs(c["shap_value"]) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert {c["feature"] for c in contributions} == {"MKT_IMPLIED_HOME", "OFF_HOME_XG_R5", "DEF_AWAY_XGA_R5"}


def test_compute_shap_contributions_reports_missing_value_as_none():
    clf = _toy_binary_model()
    row = pd.DataFrame({"MKT_IMPLIED_HOME": [np.nan], "OFF_HOME_XG_R5": [2.4], "DEF_AWAY_XGA_R5": [0.6]})

    contributions = _compute_shap_contributions(clf, row, top_n=3)

    assert contributions is not None
    by_feature = {c["feature"]: c for c in contributions}
    assert by_feature["MKT_IMPLIED_HOME"]["value"] is None
    # A missing feature can still carry a real (nonzero) SHAP contribution
    # via XGBoost's learned default-direction routing -- must not be
    # dropped or zeroed out just because the input was NaN.
    assert isinstance(by_feature["MKT_IMPLIED_HOME"]["shap_value"], float)


def test_compute_shap_contributions_returns_none_for_non_tree_model():
    """A composite/custom model (Skellam, TwoStage, ...) has no single tree
    estimator for TreeExplainer to introspect -- must degrade to None
    rather than raise, matching _extract_feature_importance's own gap for
    these model types."""
    class _FakeCompositeModel:
        def predict_proba(self, X):
            return np.array([[0.5, 0.5]] * len(X))

    row = pd.DataFrame({"a": [1.0]})

    assert _compute_shap_contributions(_FakeCompositeModel(), row) is None


def test_predict_target_classification_includes_shap_contributions():
    clf = _toy_binary_model()  # already has classes_ set from fit([0, 1])

    service = ForecastService.__new__(ForecastService)
    definition = get_target_definition("btts")
    row = pd.DataFrame({"MKT_IMPLIED_HOME": [0.4], "OFF_HOME_XG_R5": [2.4], "DEF_AWAY_XGA_R5": [0.6]})

    result = service._predict_target(definition, clf, {}, row)

    assert "shap_contributions" in result
    assert len(result["shap_contributions"]) > 0


def test_predict_target_regression_includes_shap_contributions():
    reg = _toy_regressor_model()

    service = ForecastService.__new__(ForecastService)
    definition = get_target_definition("total_goals")
    row = pd.DataFrame({"MKT_LAMBDA_TOTAL": [2.6], "OFF_HOME_XG_R5": [1.8]})

    result = service._predict_target(definition, reg, {}, row)

    assert "shap_contributions" in result
    assert len(result["shap_contributions"]) > 0


def test_predict_target_still_predicts_when_model_is_not_shap_explainable():
    """SHAP is a nice-to-have on top of the prediction, never a
    precondition for serving it -- a model TreeExplainer can't introspect
    (composite/custom, like the fake below) must still produce a normal
    prediction, just without shap_contributions."""
    class _FakeBinaryModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[0.3, 0.7]] * len(X))

    service = ForecastService.__new__(ForecastService)
    definition = get_target_definition("btts")
    row = pd.DataFrame({"a": [1.0]})

    result = service._predict_target(definition, _FakeBinaryModel(), {}, row)

    assert result["probabilities"]
    assert "shap_contributions" not in result
