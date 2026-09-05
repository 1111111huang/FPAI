"""Two-stage draw/decisive decomposition for result_3way (US#181).

Motivation: US#172/173 found repeated sample-weight retuning of a single
joint multiclass softmax model (result_3way's XGBoostModel) couldn't stop
E0/SP1 from over-predicting draw on lopsided matchups, even after several
rounds of alpha-dampening the class-balancing. Rather than a further
balancing-weight retune, this decomposes the problem structurally --
literature-standard for football outcome modeling, since a draw isn't
"a third independent class" so much as "how close is this matchup",
which is a different question from "who wins if it isn't close":

  Stage 1 (draw_model):     P(draw)          -- binary, every row
  Stage 2 (decisive_model): P(home | ~draw)  -- binary, non-draw rows ONLY

Combining: P(home) = (1-P(draw)) * P(home|~draw)
           P(away) = (1-P(draw)) * (1 - P(home|~draw))
           P(draw) = P(draw)

The decisive model's home-vs-away signal is trained on a clean subset --
never diluted by draw-balancing weights the way the single joint model's
was -- which is the actual mechanism this is betting on, not just "try
another architecture and see."
"""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.models.base_model import FPAIBaseModel
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_DRAW_MODEL_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    # ponytail: no early_stopping_rounds -- XGBoost requires an eval_set on
    # every single .fit() call once that's set at construction time, and
    # this class doesn't guarantee one (tests train without eval_set; a
    # future caller might too). Fixed n_estimators for v1; add early
    # stopping back (with an internal OOF-style fallback split, mirroring
    # GoalStackerModel) if a real hyperparameter search needs it.
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

_DECISIVE_MODEL_DEFAULTS: dict[str, Any] = dict(_DRAW_MODEL_DEFAULTS)


class TwoStageResultModel(FPAIBaseModel):
    """result_3way as draw-vs-not + home-vs-away|not-draw, not one joint softmax."""

    def __init__(
        self, draw_params: dict[str, Any] | None = None, decisive_params: dict[str, Any] | None = None,
    ) -> None:
        self._draw_params = {**_DRAW_MODEL_DEFAULTS, **(draw_params or {})}
        self._decisive_params = {**_DECISIVE_MODEL_DEFAULTS, **(decisive_params or {})}
        self.draw_model = XGBClassifier(**self._draw_params)
        self.decisive_model = XGBClassifier(**self._decisive_params)
        # Fixed, alphabetical -- matches XGBoostModel/LabelEncoder convention
        # (see model_manager.py's _classification_loss/_classes_for_calibration).
        self.classes_ = np.array(["away", "draw", "home"])
        self._feature_columns: list[str] = []

    # ------------------------------------------------------------------
    # FPAIBaseModel interface
    # ------------------------------------------------------------------

    def train(self, X: Any, y: Any, eval_set: Any | None = None, sample_weight: Any | None = None) -> None:
        """Fit both sub-models.

        `sample_weight` (the caller's 3-class-balanced array, if any) is
        deliberately NOT used here -- each sub-model computes its own
        balanced weight for its own binary problem instead. That's the
        actual point of this architecture: the decisive model's weighting
        reflects home-vs-away balance only, never contaminated by how many
        draws happen to be in the batch.
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_arr = np.asarray(y)
        self._feature_columns = list(X_df.columns)

        is_draw = (y_arr == "draw").astype(int)
        draw_weight = compute_sample_weight("balanced", is_draw)

        decisive_mask = y_arr != "draw"
        X_decisive = X_df.loc[decisive_mask] if hasattr(X_df, "loc") else X_df[decisive_mask]
        y_decisive = (y_arr[decisive_mask] == "home").astype(int)
        decisive_weight = compute_sample_weight("balanced", y_decisive) if len(y_decisive) else None

        draw_eval_set = None
        decisive_eval_set = None
        if eval_set:
            X_val, y_val = eval_set[0]
            X_val_df = pd.DataFrame(X_val) if not isinstance(X_val, pd.DataFrame) else X_val
            y_val_arr = np.asarray(y_val)
            draw_eval_set = [(X_val_df, (y_val_arr == "draw").astype(int))]
            val_decisive_mask = y_val_arr != "draw"
            if val_decisive_mask.any():
                X_val_decisive = X_val_df.loc[val_decisive_mask] if hasattr(X_val_df, "loc") else X_val_df[val_decisive_mask]
                decisive_eval_set = [(X_val_decisive, (y_val_arr[val_decisive_mask] == "home").astype(int))]

        self.draw_model.fit(X_df, is_draw, sample_weight=draw_weight, eval_set=draw_eval_set, verbose=False)
        if len(y_decisive) and len(set(y_decisive)) > 1:
            self.decisive_model.fit(
                X_decisive, y_decisive, sample_weight=decisive_weight, eval_set=decisive_eval_set, verbose=False,
            )
        else:
            LOGGER.warning(
                "TwoStageResultModel: decisive sub-model got <2 classes (%d decisive rows) -- "
                "leaving it unfit; predict_proba will fall back to a constant 0.5.", len(y_decisive),
            )

    def predict_proba(self, X: Any) -> np.ndarray:
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_df = X_df[self._feature_columns]
        p_draw = self.draw_model.predict_proba(X_df)[:, 1]
        try:
            p_home_given_decisive = self.decisive_model.predict_proba(X_df)[:, 1]
        except Exception:  # noqa: BLE001 -- decisive_model never fit (all-draw training data)
            p_home_given_decisive = np.full(len(X_df), 0.5)
        p_home = (1.0 - p_draw) * p_home_given_decisive
        p_away = (1.0 - p_draw) * (1.0 - p_home_given_decisive)
        # Column order MUST match self.classes_ = ["away", "draw", "home"].
        return np.column_stack([p_away, p_draw, p_home])

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def save(self, path: str) -> None:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "draw_params": self._draw_params,
            "decisive_params": self._decisive_params,
            "draw_model": self.draw_model,
            "decisive_model": self.decisive_model,
            "classes_": self.classes_,
            "feature_columns": self._feature_columns,
        }
        joblib.dump(payload, str(target_path))
        LOGGER.info("TwoStageResultModel saved to %s", target_path)

    @classmethod
    def load(cls, path: str) -> "TwoStageResultModel":
        payload = joblib.load(str(path))
        instance = cls(draw_params=payload["draw_params"], decisive_params=payload["decisive_params"])
        instance.draw_model = payload["draw_model"]
        instance.decisive_model = payload["decisive_model"]
        instance.classes_ = payload["classes_"]
        instance._feature_columns = payload["feature_columns"]
        return instance
