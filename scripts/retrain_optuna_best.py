"""US#180: retrain a target with explicit hyperparameters (typically an
Optuna sweep's winning trial) via ModelManager.run_pipeline(), producing a
real, promotable artifact.

The gap this bridges: `main.py optuna-sweep` (src/utils/sweep_runner.py)
never logs an `artifact_filename` MLflow param -- it's pure hyperparameter
search, no save path. `select-best-models`'s BUG-014 guard
(`ModelSelector._run_artifact_resolves`) requires that param to point at a
real file on disk, so every Optuna trial is permanently ineligible to be
selected, no matter how good its metric. This script is the missing
"retrain the winner for real" step: same hyperparameters, run through
`ModelManager.run_pipeline()` (which does save + log `artifact_filename`),
so the result becomes something `select-best-models` can actually promote.

Usage:
    python scripts/retrain_optuna_best.py --target home_goals --model xgb_regressor \\
        --context E0 --params '{"n_estimators": 463, "max_depth": 2, "learning_rate": 0.05229, ...}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logic.competition_registry import get_competition_definition, resolve_feature_subset_for_tier
from src.logic.target_registry import get_target_definition
from src.models import ModelManager
from src.models.base_model import LRModel, RandomForestRegressorModel, XGBoostModel, XGBoostRegressorModel
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Same shape as main.py's MODEL_REGISTRY -- deliberately not importing that
# dict directly (main.py is a CLI entrypoint, not a stable import surface;
# these 4 names are the only ones this script's own callers have needed).
_MODEL_CLASSES = {
    "xgb": XGBoostModel,
    "xgboost": XGBoostModel,
    "xgb_regressor": XGBoostRegressorModel,
    "xgboost_regressor": XGBoostRegressorModel,
    "lr": LRModel,
    "rf_regressor": RandomForestRegressorModel,
}


def retrain_with_params(
    target: str, model_key: str, context: str, params: dict, sample_weight_alpha: float = 1.0,
) -> Path:
    """Train `target` with an explicit hyperparameter dict and save a real,
    select-best-models-eligible artifact. Returns the saved model path."""
    definition = get_target_definition(target)
    model_cls = _MODEL_CLASSES.get(model_key.strip().lower())
    if model_cls is None:
        raise ValueError(f"Unsupported model '{model_key}'. Available: {sorted(_MODEL_CLASSES)}")

    fixed_objective: dict = {}
    if model_cls in (XGBoostModel,):
        if definition.task_type == "multiclass_classification":
            fixed_objective = {"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": len(definition.classes)}
        else:
            fixed_objective = {"objective": "binary:logistic", "eval_metric": "logloss"}

    model = model_cls(**{**fixed_objective, **params})

    competition_def = get_competition_definition(context)
    feature_subset = resolve_feature_subset_for_tier(competition_def.tier)
    manager = ModelManager(
        model=model,
        target_config={"target": definition.name},
        feature_subset=feature_subset,
        context=context,
        competition_id=context,
        sample_weight_alpha=sample_weight_alpha,
    )
    model_path = manager.run_pipeline()
    LOGGER.info("Retrained %s (%s) with explicit params -> %s", definition.name, model_key, model_path)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True, choices=sorted(_MODEL_CLASSES))
    parser.add_argument("--context", default="E0")
    parser.add_argument("--params", required=True, help="JSON dict of hyperparameters, e.g. Optuna's winning trial.")
    parser.add_argument("--sample_weight_alpha", type=float, default=1.0)
    args = parser.parse_args()

    params = json.loads(args.params)
    path = retrain_with_params(args.target, args.model, args.context, params, args.sample_weight_alpha)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
