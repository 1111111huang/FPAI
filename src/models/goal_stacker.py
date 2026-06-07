"""Two-stage stacking ensemble for goals regression targets (US#71).

Architecture:
  Level-0: XGBoost Poisson regressor + sklearn PoissonRegressor (GLM)
  Level-1: Ridge meta-learner trained on out-of-fold level-0 predictions
  OOF strategy: chronological 50/50 split of training data to avoid leakage
"""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge
from xgboost import XGBRegressor

from src.models.base_model import FPAIBaseModel
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_XGB_POISSON_DEFAULTS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "count:poisson",
    "eval_metric": "mae",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

_GLM_DEFAULTS: dict[str, Any] = {
    "alpha": 0.1,
    "max_iter": 300,
}

_RIDGE_DEFAULTS: dict[str, Any] = {
    "alpha": 1.0,
}


class GoalStackerModel(FPAIBaseModel):
    """Two-stage stacking ensemble for goal count prediction.

    Level-0: XGBoost Poisson + Poisson GLM
    Level-1: Ridge meta-learner on OOF level-0 predictions

    The OOF split is chronological (first 50% → train base, last 50% → OOF
    meta-features). All base models are then retrained on full training data.
    """

    def __init__(
        self,
        xgb_params: dict[str, Any] | None = None,
        glm_params: dict[str, Any] | None = None,
        ridge_params: dict[str, Any] | None = None,
    ) -> None:
        xgb_cfg = {**_XGB_POISSON_DEFAULTS, **(xgb_params or {})}
        glm_cfg = {**_GLM_DEFAULTS, **(glm_params or {})}
        ridge_cfg = {**_RIDGE_DEFAULTS, **(ridge_params or {})}

        self._xgb_params = xgb_cfg
        self._glm_params = glm_cfg
        self._ridge_params = ridge_cfg

        self.xgb: XGBRegressor = XGBRegressor(**xgb_cfg)
        self.glm: PoissonRegressor = PoissonRegressor(**glm_cfg)
        self.meta: Ridge = Ridge(**ridge_cfg)
        self._feature_columns: list[str] = []

    # ------------------------------------------------------------------
    # FPAIBaseModel interface
    # ------------------------------------------------------------------

    def train(self, X: Any, y: Any, eval_set: Any | None = None) -> None:
        """Train the two-stage stack using an internal chronological OOF split."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_arr = np.asarray(y, dtype=float)
        self._feature_columns = list(X_df.columns)

        split = max(10, int(len(X_df) * 0.5))
        X_base = X_df.iloc[:split]
        y_base = y_arr[:split]
        X_oof = X_df.iloc[split:]
        y_oof = y_arr[split:]

        LOGGER.info(
            "GoalStacker OOF split: base=%d rows, oof=%d rows", len(X_base), len(X_oof)
        )

        # Level-0 OOF training
        xgb_oof = XGBRegressor(**self._xgb_params)
        oof_eval = [(X_oof, y_oof)] if eval_set is None else eval_set
        xgb_oof.fit(X_base, y_base, eval_set=oof_eval, verbose=False)
        xgb_oof_preds = xgb_oof.predict(X_oof)

        # GLM requires non-negative targets
        y_base_clipped = np.clip(y_base, 0, None)
        glm_oof = PoissonRegressor(**self._glm_params)
        glm_oof.fit(X_base, y_base_clipped)
        glm_oof_preds = glm_oof.predict(X_oof)

        # Level-1 meta-learner on OOF predictions
        X_meta = np.column_stack([xgb_oof_preds, glm_oof_preds])
        self.meta.fit(X_meta, y_oof)

        LOGGER.info(
            "Meta-learner coefficients: XGB=%.4f GLM=%.4f intercept=%.4f",
            self.meta.coef_[0],
            self.meta.coef_[1],
            self.meta.intercept_,
        )

        # Retrain base models on full training data
        # If no external eval_set, use the OOF half so early stopping still works
        val_set = eval_set if eval_set is not None else [(X_oof, y_oof)]
        self.xgb.fit(X_df, y_arr, eval_set=val_set, verbose=False)
        y_clipped = np.clip(y_arr, 0, None)
        self.glm.fit(X_df, y_clipped)

        LOGGER.info("GoalStacker training complete.")

    def predict(self, X: Any) -> np.ndarray:
        """Return ensemble predictions (clipped to >= 0)."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_df = X_df[self._feature_columns]
        xgb_pred = self.xgb.predict(X_df)
        glm_pred = self.glm.predict(X_df)
        X_meta = np.column_stack([xgb_pred, glm_pred])
        stack_pred = self.meta.predict(X_meta)
        return np.clip(stack_pred, 0, None)

    def predict_proba(self, X: Any) -> np.ndarray:
        raise TypeError("GoalStackerModel does not support predict_proba.")

    def save(self, path: str) -> None:
        """Save all sub-models and feature columns via joblib."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "xgb_params": self._xgb_params,
            "glm_params": self._glm_params,
            "ridge_params": self._ridge_params,
            "xgb": self.xgb,
            "glm": self.glm,
            "meta": self.meta,
            "feature_columns": self._feature_columns,
        }
        joblib.dump(payload, str(target_path))
        LOGGER.info("GoalStacker saved to %s", target_path)

    @classmethod
    def load(cls, path: str) -> "GoalStackerModel":
        """Load a persisted GoalStackerModel."""
        payload = joblib.load(str(path))
        instance = cls(
            xgb_params=payload["xgb_params"],
            glm_params=payload["glm_params"],
            ridge_params=payload["ridge_params"],
        )
        instance.xgb = payload["xgb"]
        instance.glm = payload["glm"]
        instance.meta = payload["meta"]
        instance._feature_columns = payload["feature_columns"]
        return instance
