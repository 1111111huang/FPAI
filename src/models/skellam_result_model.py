"""Skellam-distribution result_3way model (US#183).

Not a from-scratch classifier: a distributional STACK over the
already-promoted home_goals/away_goals models for a competition, rather
than a separately-trained sub-model pair. Converts their goal-count
predictions into P(home/draw/away) via Skellam(mu_home, mu_away) -- the
actual discrete distribution for "difference of two independent
Poisson-like counts" -- instead of a continuous Gaussian approximation to
the margin (which was found, empirically, to structurally under-produce
draw probability mass: even at its peak a symmetric Normal rarely gives
draw enough mass to matter, while Skellam is naturally more concentrated
near zero at typical football scoring rates).

Found live (2026-09-03): this genuinely beats the current result_3way
champion on log_loss using models that already exist -- zero new goal-model
training required. Trade-off, reported honestly: like the margin-regression
approach, it essentially never gives draw the single highest probability
(0% recall on the argmax pick), even though its probabilities are, by the
log_loss measure, more calibrated than the champion's. Whether that matters
depends on whether the consumer acts on the argmax pick or the full
probability distribution (e.g. value-edge calculations against market odds)
-- not resolved by this class, a downstream decision.

Always resolves the CURRENT model_path for home_goals/away_goals from
config/model_selection.yaml at train()/load() time -- never a hardcoded
artifact filename -- so a future re-promotion of either sub-model is picked
up automatically, not silently left stale.
"""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import skellam

from src.models.base_model import FPAIBaseModel, XGBoostRegressorModel
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_MIN_EXPECTED_GOALS = 0.02  # skellam.pmf degenerates at mu=0; floor avoids that


class SkellamResultModel(FPAIBaseModel):
    """result_3way via Skellam(home_goals_pred, away_goals_pred)."""

    def __init__(
        self, competition_id: str = "E0", model_selection_path: str = "config/model_selection.yaml",
    ) -> None:
        self.competition_id = competition_id
        self.model_selection_path = model_selection_path
        self.home_model: XGBoostRegressorModel | None = None
        self.away_model: XGBoostRegressorModel | None = None
        self._home_feature_columns: list[str] | None = None
        self._away_feature_columns: list[str] | None = None
        # Fixed, alphabetical -- matches XGBoostModel/LabelEncoder convention.
        self.classes_ = np.array(["away", "draw", "home"])

    def _load_submodel(self, target_name: str) -> tuple[XGBoostRegressorModel, list[str] | None]:
        selection_file = Path(self.model_selection_path)
        if not selection_file.exists():
            raise ValueError(
                f"SkellamResultModel needs a promoted '{target_name}' model for "
                f"competition '{self.competition_id}', but {selection_file} doesn't exist."
            )
        with selection_file.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        entry = config.get("contexts", {}).get(self.competition_id, {}).get(target_name)
        if entry is None:
            raise ValueError(
                f"SkellamResultModel needs a promoted '{target_name}' model for competition "
                f"'{self.competition_id}', but config/model_selection.yaml has no such entry yet -- "
                f"run train-target/select-best-models for it first."
            )
        model_path = entry["model_path"]
        if not Path(model_path).is_absolute():
            model_path = str(selection_file.parent.parent / model_path)
        model = XGBoostRegressorModel.load(model_path)

        feature_subset = entry.get("feature_subset")
        if not feature_subset:
            # No per-target override recorded -- resolve the same way
            # ModelManager itself would have at training time (schema.yaml's
            # full list, minus this competition's own group gating). Doesn't
            # depend on X, so train()/load() behave identically regardless
            # of what (if anything) X looks like at call time.
            from src.models.model_manager import ModelManager

            resolver = ModelManager(
                model=XGBoostRegressorModel(),
                target_config={"target": target_name},
                competition_id=self.competition_id,
                context=self.competition_id,
            )
            feature_subset = resolver._load_selected_features()
        return model, feature_subset

    def train(self, X: Any, y: Any, eval_set: Any | None = None, sample_weight: Any | None = None) -> None:
        """Doesn't fit anything new -- loads the two already-promoted goal
        models this competition currently has selected. `X`/`y`/`eval_set`/
        `sample_weight` are accepted only for FPAIBaseModel interface
        consistency and are otherwise unused."""
        self.home_model, self._home_feature_columns = self._load_submodel("home_goals")
        self.away_model, self._away_feature_columns = self._load_submodel("away_goals")
        LOGGER.info(
            "SkellamResultModel: loaded promoted home_goals/away_goals for competition=%s "
            "(no fresh sub-model fitting)", self.competition_id,
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.home_model is None or self.away_model is None:
            raise RuntimeError("Model must be trained (loaded) before predicting.")
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        mu_home = np.clip(self.home_model.predict(X_df[self._home_feature_columns]), _MIN_EXPECTED_GOALS, None)
        mu_away = np.clip(self.away_model.predict(X_df[self._away_feature_columns]), _MIN_EXPECTED_GOALS, None)

        p_draw = skellam.pmf(0, mu_home, mu_away)
        p_home = skellam.sf(0, mu_home, mu_away)   # P(margin > 0)
        p_away = skellam.cdf(-1, mu_home, mu_away)  # P(margin < 0)
        proba = np.column_stack([p_away, p_draw, p_home])
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def save(self, path: str) -> None:
        """Persists only this model's own config, not the sub-models'
        weights -- they're re-loaded from model_selection.yaml on load(),
        so this artifact always reflects whichever home_goals/away_goals
        are CURRENTLY promoted, not a frozen copy from training time."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"competition_id": self.competition_id, "model_selection_path": self.model_selection_path},
            str(target_path),
        )
        LOGGER.info("SkellamResultModel saved to %s (config only; sub-models re-resolved on load)", target_path)

    @classmethod
    def load(cls, path: str) -> "SkellamResultModel":
        payload = joblib.load(str(path))
        instance = cls(
            competition_id=payload["competition_id"], model_selection_path=payload["model_selection_path"],
        )
        instance.train(pd.DataFrame(), pd.Series(dtype=object))
        return instance
