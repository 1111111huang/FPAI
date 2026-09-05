"""Ensemble of result_3way architectures (US#188).

Per direct user prioritization ("Do 8, 1, 3, 5, 6 in that order"), item #5:
"Ensemble the multiple result_3way architectures now built (XGBoost
classifier, TwoStageResultModel, SkellamResultModel, Dixon-Coles) instead
of picking one champion."

Equal-weight average of predict_proba across three members that all
already conform to FPAIBaseModel's numeric X -> proba(3) contract and
share the same alphabetical ["away", "draw", "home"] class order:

- "xgboost": a fresh XGBoostModel, fit directly on whatever X/y this
  ensemble receives (the same joint softmax the plain result_3way
  champion already is).
- "twostage": a fresh TwoStageResultModel (US#181), fit the same way.
- "skellam": a SkellamResultModel (US#183) -- doesn't fit anything new,
  loads whichever home_goals/away_goals are CURRENTLY promoted for this
  competition from model_selection.yaml.

Dixon-Coles is deliberately NOT a fourth member -- see this module's own
test file docstring for why (team-identity input, not a numeric feature
row; folding it in would mean changing the shared numeric-only serving
contract every other target relies on). Evaluated separately, offline;
see documents/user_stories.md US#188 for the real result.
"""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base_model import FPAIBaseModel, XGBoostModel
from src.models.skellam_result_model import SkellamResultModel
from src.models.two_stage_result_model import TwoStageResultModel
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class EnsembleResultModel(FPAIBaseModel):
    """result_3way via equal-weight average of xgboost + twostage + skellam."""

    def __init__(
        self, competition_id: str = "E0", model_selection_path: str = "config/model_selection.yaml",
    ) -> None:
        self.competition_id = competition_id
        self.model_selection_path = model_selection_path
        self.members: dict[str, FPAIBaseModel] = {}
        self.classes_ = np.array(["away", "draw", "home"])

    def train(self, X: Any, y: Any, eval_set: Any | None = None, sample_weight: Any | None = None) -> None:
        # ponytail: no early_stopping_rounds -- XGBoostModel's own default
        # requires an eval_set on every .fit() call once set at
        # construction, which this ensemble doesn't guarantee (same
        # posture TwoStageResultModel already established for its own
        # sub-models). Fixed n_estimators for v1.
        #
        # XGBoostModel's own default objective (binary:logistic) is wrong
        # for result_3way's 3 classes -- unlike the plain "--model xgboost"
        # CLI path, main.py's _xgb_params_for_target multiclass override
        # never reaches a ModelFactory-dispatched model like this one, so
        # this member must derive it itself from the real label set rather
        # than assume main.py already handled it (found live: XGBoost
        # crashed mid-fit, "label and prediction size not match" -- logloss
        # metric expects one probability column, softprob was producing 3).
        xgb_kwargs: dict[str, Any] = {"early_stopping_rounds": None}
        num_classes = len(np.unique(np.asarray(y)))
        if num_classes > 2:
            xgb_kwargs.update(
                {"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": num_classes}
            )
        xgb_member = XGBoostModel(**xgb_kwargs)
        # Only the plain XGBoost member gets the caller's class-balancing
        # weight -- twostage/skellam already manage their own weighting
        # internally (see each class's own train(), same posture US#181
        # established: a shared 3-class weight would contaminate twostage's
        # deliberately-clean decisive-model split).
        xgb_member.train(X, y, eval_set=eval_set, sample_weight=sample_weight)

        twostage_member = TwoStageResultModel()
        twostage_member.train(X, y, eval_set=eval_set)

        skellam_member = SkellamResultModel(
            competition_id=self.competition_id, model_selection_path=self.model_selection_path,
        )
        skellam_member.train(X, y)

        self.members = {"xgboost": xgb_member, "twostage": twostage_member, "skellam": skellam_member}

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.members:
            raise RuntimeError("Model must be trained (loaded) before predicting.")
        probas = [member.predict_proba(X) for member in self.members.values()]
        avg = np.mean(probas, axis=0)
        return avg / avg.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def save(self, path: str) -> None:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        xgb_path = target_path.with_suffix(".xgb_member.joblib")
        twostage_path = target_path.with_suffix(".twostage_member.joblib")
        self.members["xgboost"].save(str(xgb_path))
        self.members["twostage"].save(str(twostage_path))
        joblib.dump(
            {
                "competition_id": self.competition_id,
                "model_selection_path": self.model_selection_path,
                "xgb_member_path": str(xgb_path),
                "twostage_member_path": str(twostage_path),
            },
            str(target_path),
        )
        LOGGER.info("EnsembleResultModel saved to %s (skellam member re-resolved on load)", target_path)

    @classmethod
    def load(cls, path: str) -> "EnsembleResultModel":
        payload = joblib.load(str(path))
        instance = cls(
            competition_id=payload["competition_id"], model_selection_path=payload["model_selection_path"],
        )
        xgb_member = XGBoostModel.load(payload["xgb_member_path"])
        twostage_member = TwoStageResultModel.load(payload["twostage_member_path"])
        skellam_member = SkellamResultModel(
            competition_id=instance.competition_id, model_selection_path=instance.model_selection_path,
        )
        skellam_member.train(pd.DataFrame(), pd.Series(dtype=object))
        instance.members = {"xgboost": xgb_member, "twostage": twostage_member, "skellam": skellam_member}
        return instance
