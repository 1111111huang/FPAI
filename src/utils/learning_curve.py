"""Learning curve analysis: train on growing chronological subsets to diagnose
feature ceiling vs. data ceiling.

For each target, training data is split into growing fractions (20%–100%) of the
chronological train split. The validation set remains fixed so metric changes are
attributable solely to training set size.

A plateau near 100% training data means the model has extracted most available
signal — adding more data won't help and the bottleneck is features or irreducible
noise. A still-descending curve at 100% means we're data-limited.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

from src.logic.target_registry import TargetDefinition, get_target_definition
from src.models.base_model import FPAIBaseModel, XGBoostModel, XGBoostRegressorModel
from src.models.model_manager import ModelManager
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

DEFAULT_FRACTIONS = [0.20, 0.40, 0.60, 0.80, 1.00]

_XGB_CLF_DEFAULTS: dict = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
    random_state=42,
)
_XGB_REG_DEFAULTS: dict = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    early_stopping_rounds=20,
    random_state=42,
)


def _make_model(definition: TargetDefinition) -> FPAIBaseModel:
    if definition.task_type == "regression":
        return XGBoostRegressorModel(**_XGB_REG_DEFAULTS)
    params = dict(_XGB_CLF_DEFAULTS)
    if definition.task_type == "multiclass_classification":
        params.update(
            objective="multi:softprob",
            num_class=len(definition.classes),
            eval_metric="mlogloss",
        )
    else:
        params.update(objective="binary:logistic", eval_metric="logloss")
    return XGBoostModel(**params)


def _compute_val_metrics(
    model: FPAIBaseModel,
    definition: TargetDefinition,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, float]:
    """Compute primary (and secondary) metrics on the fixed validation split."""
    if definition.task_type == "regression":
        preds = np.asarray(model.predict(X_val), dtype=float)
        mae = float(mean_absolute_error(y_val, preds))
        return {"mae": mae, "primary": mae}

    proba = np.asarray(model.predict_proba(X_val))
    preds = np.asarray(model.predict(X_val))
    accuracy = float(accuracy_score(y_val, preds))

    labels: list | None = None
    if hasattr(model, "classes_") and model.classes_ is not None:
        labels = list(model.classes_)
    elif hasattr(model, "label_encoder") and model.label_encoder is not None:
        labels = list(model.label_encoder.classes_)

    ll = float(log_loss(y_val, proba, labels=labels))
    return {"log_loss": ll, "accuracy": accuracy, "primary": ll}


class LearningCurveAnalyzer:
    """Run training on growing chronological training subsets, record val metrics."""

    def __init__(
        self,
        target_name: str,
        config_path: str = "config.yaml",
        output_dir: str = "reports/learning_curves",
    ) -> None:
        self.target_name = target_name
        self.definition = get_target_definition(target_name)
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_splits(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        """Load chronological train/val/test splits via ModelManager."""
        model = _make_model(self.definition)
        manager = ModelManager(
            model=model,
            config_path=self.config_path,
            target_config={"target": self.target_name},
        )
        return manager.prepare_training_data()

    def run(self, fractions: list[float] | None = None) -> dict:
        """Train on growing subsets and return per-fraction val metrics.

        Returns a dict with keys: target, metric, total_train_n, total_val_n, results.
        `results` is a list of dicts, one per fraction.
        """
        fractions = sorted(set(fractions or DEFAULT_FRACTIONS))
        X_train, X_val, X_test, y_train, y_val, y_test, _ = self._load_splits()

        LOGGER.info(
            "Learning curve | target=%s | train=%d val=%d test=%d",
            self.target_name,
            len(X_train),
            len(X_val),
            len(X_test),
        )

        primary_metric = self.definition.primary_metric
        results: list[dict] = []

        for frac in fractions:
            n = max(20, int(len(X_train) * frac))
            X_sub = X_train.iloc[:n]
            y_sub = y_train.iloc[:n]

            if self.definition.task_type != "regression" and y_sub.nunique() < 2:
                LOGGER.warning("Skipping frac=%.0f%% — only one class in subset", frac * 100)
                continue

            model = _make_model(self.definition)
            model.train(X_sub, y_sub, eval_set=[(X_val, y_val)])
            metrics = _compute_val_metrics(model, self.definition, X_val, y_val)

            LOGGER.info(
                "  %.0f%% n=%d | %s=%.4f",
                frac * 100,
                n,
                primary_metric,
                metrics["primary"],
            )
            row: dict = {"fraction": round(frac, 2), "train_n": n}
            row[primary_metric] = metrics["primary"]
            for key, val in metrics.items():
                if key not in ("primary", primary_metric):
                    row[key] = val
            results.append(row)

        return {
            "target": self.target_name,
            "metric": primary_metric,
            "total_train_n": len(X_train),
            "total_val_n": len(X_val),
            "results": results,
        }

    def save_results(self, run_output: dict) -> Path:
        """Save per-fraction metrics to a CSV file."""
        df = pd.DataFrame(run_output["results"])
        path = self.output_dir / f"learning_curve_{self.target_name}.csv"
        df.to_csv(path, index=False)
        LOGGER.info("Saved learning curve CSV: %s", path)
        return path

    def save_chart(self, run_output: dict, ax=None) -> Path | None:
        """Plot val metric vs. training set size. Pass ax to embed in a grid chart."""
        results = run_output["results"]
        if not results:
            return None
        df = pd.DataFrame(results)
        metric = run_output["metric"]
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(df["train_n"], df[metric], marker="o", linewidth=2, markersize=5)
        # Highlight relative change from first to last fraction
        if len(df) >= 2:
            delta = df[metric].iloc[-1] - df[metric].iloc[0]
            direction = "lower" if metric in ("mae", "log_loss") else "higher"
            sign = "+" if delta > 0 else ""
            ax.set_title(
                f"{self.target_name}\n({metric}, {sign}{delta:.4f} full vs 20%)",
                fontsize=9,
            )
        else:
            ax.set_title(self.target_name, fontsize=9)

        ax.set_xlabel("Training samples", fontsize=8)
        ax.set_ylabel(metric.upper(), fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

        if standalone:
            plt.tight_layout()
            out = self.output_dir / f"learning_curve_{self.target_name}.png"
            plt.savefig(out, dpi=120)
            plt.close()
            LOGGER.info("Saved chart: %s", out)
            return out
        return None


def run_all_targets(
    config_path: str = "config.yaml",
    output_dir: str = "reports/learning_curves",
    fractions: list[float] | None = None,
) -> dict[str, dict]:
    """Run learning curve analysis for all 8 forecast targets and save a combined chart."""
    targets = [
        "result_3way", "btts",
        "home_goals", "away_goals", "total_goals",
        "home_corners", "away_corners", "total_corners",
    ]
    all_results: dict[str, dict] = {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()

    for idx, target_name in enumerate(targets):
        LOGGER.info("=== %s ===", target_name)
        analyzer = LearningCurveAnalyzer(target_name, config_path=config_path, output_dir=output_dir)
        result = analyzer.run(fractions=fractions)
        analyzer.save_results(result)
        analyzer.save_chart(result, ax=axes_flat[idx])
        all_results[target_name] = result

    plt.suptitle("Learning Curves — All Targets (XGBoost, fixed val set)", fontsize=11, y=1.01)
    plt.tight_layout()
    combined_path = out_dir / "learning_curves_all_targets.png"
    plt.savefig(combined_path, dpi=120, bbox_inches="tight")
    plt.close()
    LOGGER.info("Combined chart saved: %s", combined_path)

    return all_results


def summarise_findings(all_results: dict[str, dict]) -> str:
    """Return a text summary of plateau behaviour per target."""
    lines = ["Learning Curve Findings", "=" * 40]
    for target, result in all_results.items():
        rows = result["results"]
        if len(rows) < 2:
            lines.append(f"{target}: insufficient data points")
            continue
        metric = result["metric"]
        first = rows[0][metric]
        last = rows[-1][metric]
        delta = last - first
        pct = abs(delta / first) * 100 if first else 0.0
        ascending = metric not in ("mae", "log_loss")
        improving = (delta < 0) if metric in ("mae", "log_loss") else (delta > 0)
        # Check if gain between last two fractions is tiny (plateau signal)
        second_last = rows[-2][metric]
        tail_delta = abs(last - second_last)
        tail_pct = abs(tail_delta / second_last) * 100 if second_last else 0.0
        plateau = tail_pct < 0.5
        lines.append(
            f"{target}: {metric} {first:.4f}→{last:.4f} "
            f"({'−' if delta < 0 else '+'}{pct:.1f}%) | "
            f"{'PLATEAU' if plateau else 'STILL IMPROVING'} at 100%"
        )
    return "\n".join(lines)
