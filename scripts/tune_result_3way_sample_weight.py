"""Throwaway sweep script (2026-08-20): finds a per-league alpha for
result_3way's sample-weight dampening that fixes the draw-overprediction
bug (docs/superpowers/specs/2026-08-20-result-3way-sample-weight-retune-design.md)
without regressing to the pre-08-14 draw-blindness bug.

Deliberately does NOT call ModelManager.run_pipeline() -- that starts an
MLflow run tagged sweep_stage="final" and saves a real model artifact,
which would make every candidate alpha (including the ones we reject)
eligible for `select-best-models` to accidentally promote by raw log_loss,
the same metric-only mistake that let the original 08-14 fix overshoot
unnoticed. This script only trains in-memory and prints metrics.

Usage: python scripts/tune_result_3way_sample_weight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_score, recall_score

from src.logic.competition_registry import get_competition_definition, resolve_feature_subset_for_tier
from src.models import ModelManager, XGBoostModel
from src.models.model_manager import _compute_sample_weight

LEAGUES = ["E0", "SP1", "D1", "I1", "F1"]
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
XGB_PARAMS = {"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": 3}
LABELS = ["home", "draw", "away"]


def sweep_one_league(context: str) -> pd.DataFrame:
    competition_def = get_competition_definition(context)
    feature_subset = resolve_feature_subset_for_tier(competition_def.tier)

    rows = []
    for alpha in ALPHAS:
        model = XGBoostModel(**XGB_PARAMS)
        manager = ModelManager(
            model=model,
            target_config={"target": "result_3way"},
            feature_subset=feature_subset,
            context=context,
            competition_id=context,
            sample_weight_alpha=alpha,
        )
        selected_features = manager._load_selected_features()
        manager._log_selected_features(selected_features)
        X_train, X_val, X_test, y_train, y_val, y_test, _ = manager.prepare_training_data()

        weights = _compute_sample_weight(y_train, manager.target_definition.task_type, alpha=alpha)
        model.train(X_train, y_train, eval_set=[(X_val, y_val)], sample_weight=weights)

        proba = model.predict_proba(X_test)  # columns ordered per model.label_encoder.classes_
        class_order = list(model.label_encoder.classes_)  # alphabetical: away, draw, home
        preds = model.predict(X_test)

        draw_col = class_order.index("draw")
        mean_draw_proba = float(proba[:, draw_col].mean())

        y_test_arr = np.asarray(y_test)
        recall = recall_score(y_test_arr, preds, labels=LABELS, average=None, zero_division=0)
        precision = precision_score(y_test_arr, preds, labels=LABELS, average=None, zero_division=0)
        true_draw_rate = float((y_test_arr == "draw").mean())

        # log_loss needs probabilities in the same column order as its own
        # labels= argument, not necessarily class_order -- pass class_order
        # explicitly so this is correct regardless of alphabetical assumptions.
        loss = log_loss(y_test_arr, proba, labels=class_order)

        rows.append({
            "league": context,
            "alpha": alpha,
            "log_loss": round(loss, 4),
            "draw_recall": round(float(recall[LABELS.index("draw")]), 4),
            "draw_precision": round(float(precision[LABELS.index("draw")]), 4),
            "home_recall": round(float(recall[LABELS.index("home")]), 4),
            "away_recall": round(float(recall[LABELS.index("away")]), 4),
            "mean_predicted_draw_proba": round(mean_draw_proba, 4),
            "true_draw_rate": round(true_draw_rate, 4),
        })

    return pd.DataFrame(rows)


def main() -> None:
    all_results = pd.concat([sweep_one_league(league) for league in LEAGUES], ignore_index=True)
    out_path = Path("reports") / "result_3way_sample_weight_sweep.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(out_path, index=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(all_results.to_string(index=False))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
