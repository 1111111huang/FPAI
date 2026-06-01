from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import EvaluationDiagnostics


def test_regression_diagnostics_include_residual_bins() -> None:
    diagnostics = EvaluationDiagnostics("home_goals")
    y_true = pd.Series([1.0, 2.0, 0.0, 3.0, 1.0, 2.0])
    predictions = np.asarray([1.2, 1.7, 0.4, 2.5, 0.9, 2.3])
    meta = pd.DataFrame({"odds_h": [1.5, 1.8, 2.1, 2.8, 3.2, 4.0]})

    report = diagnostics.build_report(y_true, predictions, meta)

    assert report["target"] == "home_goals"
    assert report["task_type"] == "regression"
    assert "mae" in report["residuals"]
    assert "rmse" in report["residuals"]
    assert "by_home_odds_bin" in report["residuals"]
    assert report["prediction_interval_coverage"]["available"] is False


def test_binary_diagnostics_include_calibration_points() -> None:
    diagnostics = EvaluationDiagnostics("btts")
    y_true = pd.Series([0, 0, 1, 1, 1, 0])
    probabilities = np.asarray(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.55, 0.45],
        ]
    )

    report = diagnostics.build_report(y_true, probabilities)

    assert report["target"] == "btts"
    assert report["calibration"]["available"] is True
    assert report["calibration"]["bins"]
