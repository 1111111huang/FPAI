"""Per-match, heteroscedastic prediction intervals for regression targets
(US#184).

Direct response to a real gap surfaced by the user: `residual_prediction_
interval()` (src/forecast/uncertainty.py) gives every match the SAME
interval width -- one lower/upper residual pair computed once from overall
validation residual quantiles at training time, stored in the artifact's
own metadata.json. It doesn't matter whether the model is actually more or
less certain about a *specific* fixture; the band is identical either way.
Given the agent (per documents/agent_user_stories.md Phase 32) concentrates
most of its direct bets on `total_goals`/corners -- exactly the regression
targets this fixed-width mechanism serves -- this is the more consequential
half of "does the model give the agent something it can actually reason
about per-match" than the classifiers' entropy-based uncertainty already
provides.

Fits three XGBRegressor sub-models sharing the same features:
  - point: the usual point estimate (same objective/defaults as
    XGBoostRegressorModel).
  - lower/upper: XGBoost's native quantile regression objective
    (reg:quantileerror) at the requested coverage's lower/upper quantiles
    -- genuinely per-match bounds that vary with input features the same
    way the point estimate does, not a single global number.
"""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.models.base_model import FPAIBaseModel
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_POINT_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

# ponytail: no early_stopping_rounds on the quantile sub-models -- unlike
# the point model (which gets a real eval_set from ModelManager, see
# model_manager.py's eval_set isinstance list), fixed n_estimators keeps
# these two robust to being trained standalone/without one too.
_QUANTILE_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


class QuantileIntervalModel(FPAIBaseModel):
    """Point regression + per-match prediction interval via quantile regression."""

    def __init__(
        self,
        coverage: float = 0.8,
        point_params: dict[str, Any] | None = None,
        quantile_params: dict[str, Any] | None = None,
    ) -> None:
        self.coverage = coverage
        lower_q = (1.0 - coverage) / 2.0
        upper_q = 1.0 - lower_q
        self._point_params = {**_POINT_DEFAULTS, **(point_params or {})}
        self._quantile_params = {**_QUANTILE_DEFAULTS, **(quantile_params or {})}
        self.point_model = XGBRegressor(**self._point_params)
        self.lower_model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=lower_q, **self._quantile_params)
        self.upper_model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=upper_q, **self._quantile_params)
        self._feature_columns: list[str] = []

    def train(self, X: Any, y: Any, eval_set: Any | None = None, sample_weight: Any | None = None) -> None:
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self._feature_columns = list(X_df.columns)
        if eval_set is not None:
            self.point_model.fit(X_df, y, sample_weight=sample_weight, eval_set=eval_set, verbose=False)
        else:
            self.point_model.fit(X_df, y, sample_weight=sample_weight, verbose=False)
        self.lower_model.fit(X_df, y, sample_weight=sample_weight, verbose=False)
        self.upper_model.fit(X_df, y, sample_weight=sample_weight, verbose=False)

    def predict(self, X: Any) -> np.ndarray:
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.point_model.predict(X_df[self._feature_columns])

    def predict_proba(self, X: Any) -> np.ndarray:
        raise TypeError("QuantileIntervalModel is a regression model; use predict()/predict_interval().")

    def predict_interval(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """Per-match (lower, upper) bounds at this model's configured coverage."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_sel = X_df[self._feature_columns]
        raw_lower = self.lower_model.predict(X_sel)
        raw_upper = self.upper_model.predict(X_sel)
        # Two independently-fit quantile models can cross (lower > upper)
        # for a minority of rows -- swap rather than report a nonsensical
        # negative-width interval.
        lower = np.minimum(raw_lower, raw_upper)
        upper = np.maximum(raw_lower, raw_upper)
        # Goal/corner counts can't be negative.
        return np.clip(lower, 0.0, None), np.clip(upper, 0.0, None)

    def save(self, path: str) -> None:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "coverage": self.coverage,
                "point_params": self._point_params,
                "quantile_params": self._quantile_params,
                "point_model": self.point_model,
                "lower_model": self.lower_model,
                "upper_model": self.upper_model,
                "feature_columns": self._feature_columns,
            },
            str(target_path),
        )
        LOGGER.info("QuantileIntervalModel saved to %s", target_path)

    @classmethod
    def load(cls, path: str) -> "QuantileIntervalModel":
        payload = joblib.load(str(path))
        instance = cls(
            coverage=payload["coverage"],
            point_params=payload["point_params"],
            quantile_params=payload["quantile_params"],
        )
        instance.point_model = payload["point_model"]
        instance.lower_model = payload["lower_model"]
        instance.upper_model = payload["upper_model"]
        instance._feature_columns = payload["feature_columns"]
        return instance
