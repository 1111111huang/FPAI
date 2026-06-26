from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models import LRModel, ModelFactory, XGBoostModel, XGBoostRegressorModel
from src.utils.sweep_runner import forecast_experiment_name, iter_grid_params
from src.utils.model_comparison import ModelComparison


def test_iter_grid_params_expands_in_stable_order() -> None:
    params = iter_grid_params({"C": [0.1, 1.0], "class_weight": [None, "balanced"]})

    assert params == [
        {"C": 0.1, "class_weight": None},
        {"C": 0.1, "class_weight": "balanced"},
        {"C": 1.0, "class_weight": None},
        {"C": 1.0, "class_weight": "balanced"},
    ]


def test_forecast_experiment_name_uses_target_model_stage_version() -> None:
    assert forecast_experiment_name("btts", "lr", "broad", "v1") == "FPAI_btts_lr_broad_v1"


def test_model_factory_accepts_cli_aliases() -> None:
    assert isinstance(ModelFactory.get_model("lr"), LRModel)
    assert isinstance(ModelFactory.get_model("xgb"), XGBoostModel)
    assert isinstance(ModelFactory.get_model("xgb_regressor"), XGBoostRegressorModel)


def test_xgboost_classifier_handles_string_multiclass_labels() -> None:
    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = ["home", "draw", "away", "home", "draw", "away"]
    model = ModelFactory.get_model(
        "xgb",
        {
            "n_estimators": 2,
            "max_depth": 1,
            "learning_rate": 0.1,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": 3,
            "early_stopping_rounds": None,
        },
    )

    model.train(X, y)

    assert set(model.predict(X)) <= {"home", "draw", "away"}
    assert model.predict_proba(X).shape == (6, 3)


def test_model_comparison_normalizes_split_metric_names() -> None:
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run-1", experiment_id="exp-1"),
        data=SimpleNamespace(
            metrics={"log_loss_test": 0.5, "accuracy_val": 0.6},
            params={"C": "1.0"},
            tags={"target": "btts", "model_family": "lr"},
        ),
    )

    row = ModelComparison().extract_metrics_for_run(run)

    assert row["log_loss_test"] == 0.5
    assert row["test_log_loss"] == 0.5
    assert row["accuracy_val"] == 0.6
    assert row["val_accuracy"] == 0.6
