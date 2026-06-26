"""Model selection utility: select best MLflow runs per target/context and persist to YAML (US#78)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import yaml

from src.logic.target_registry import list_target_definitions, get_target_definition
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

SELECTION_CONFIG_PATH = Path("config/model_selection.yaml")

# Metrics where lower = better
_LOWER_IS_BETTER = {"log_loss", "mae", "rmse", "test_log_loss", "test_mae", "test_rmse"}

# Classification targets use test_log_loss; regression targets use test_mae
_CLASSIFICATION_TARGETS = {"result_3way", "btts"}


def _primary_metric_for_target(target_name: str) -> str:
    try:
        defn = get_target_definition(target_name)
        metric = defn.primary_metric
    except Exception:
        metric = "log_loss"
    return f"test_{metric}" if not metric.startswith("test_") else metric


def _is_better(new_val: float, old_val: float, metric: str, min_improvement: float) -> bool:
    if metric in _LOWER_IS_BETTER:
        return new_val < old_val - min_improvement
    return new_val > old_val + min_improvement


class ModelSelector:
    """Select and persist the best-performing model per target and context."""

    def __init__(self, config_path: str | Path = SELECTION_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        self.client = mlflow.tracking.MlflowClient()

    def load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {"contexts": {"league": {}, "international": {}}}
        with self.config_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {"contexts": {"league": {}, "international": {}}}

    def _save_config(self, config: dict[str, Any]) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=True, allow_unicode=True)
        LOGGER.info("model_selection.yaml written to %s", self.config_path)

    def _fetch_eligible_runs(self, target_name: str, context: str) -> list[dict[str, Any]]:
        """Fetch MLflow runs tagged sweep_stage in {optuna, final} for a target/context."""
        experiment_ids = [e.experiment_id for e in self.client.search_experiments()]
        if not experiment_ids:
            return []

        context_filter = f" AND tags.context = '{context}'" if context else ""
        # Eligible sweep stages: optuna or final
        runs_optuna = self.client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=f"tags.target = '{target_name}' AND tags.sweep_stage = 'optuna'{context_filter}",
        )
        runs_final = self.client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=f"tags.target = '{target_name}' AND tags.sweep_stage = 'final'{context_filter}",
        )
        results = []
        for run in list(runs_optuna) + list(runs_final):
            metrics = run.data.metrics or {}
            tags = run.data.tags or {}
            results.append({
                "run_id": run.info.run_id,
                "model_type": tags.get("model_family", tags.get("model_type", "unknown")),
                "artifact_uri": run.info.artifact_uri,
                "metrics": metrics,
                "feature_subset": (run.data.params or {}).get("feature_subset"),
            })
        return results

    def _best_run(
        self,
        runs: list[dict[str, Any]],
        metric: str,
    ) -> dict[str, Any] | None:
        eligible = [r for r in runs if metric in r["metrics"]]
        if not eligible:
            # Fallback: try without test_ prefix
            fallback = metric.replace("test_", "")
            eligible = [r for r in runs if fallback in r["metrics"]]
            if eligible:
                metric = fallback
        if not eligible:
            return None
        lower = metric in _LOWER_IS_BETTER
        return min(eligible, key=lambda r: r["metrics"][metric]) if lower else max(eligible, key=lambda r: r["metrics"][metric])

    def _select_for_target_context(
        self,
        target_name: str,
        context: str,
        current_entry: dict[str, Any],
        min_improvement: float,
        dry_run: bool,
    ) -> dict[str, Any] | None:
        metric = _primary_metric_for_target(target_name)
        runs = self._fetch_eligible_runs(target_name, context)
        if not runs:
            LOGGER.info("No eligible runs for target=%s context=%s", target_name, context)
            return None

        best = self._best_run(runs, metric)
        if best is None:
            LOGGER.info("No runs with metric '%s' for target=%s context=%s", metric, target_name, context)
            return None

        best_metric_val = best["metrics"].get(metric, best["metrics"].get(metric.replace("test_", "")))
        current_metric_val = current_entry.get("metric_value")

        if current_metric_val is not None:
            if not _is_better(float(best_metric_val), float(current_metric_val), metric, min_improvement):
                LOGGER.info(
                    "No improvement for target=%s context=%s: current=%.4f best=%.4f (min_improvement=%.4f)",
                    target_name, context, current_metric_val, best_metric_val, min_improvement,
                )
                return None

        new_entry: dict[str, Any] = {
            "mlflow_run_id": best["run_id"],
            "model_type": best["model_type"],
            "model_path": f"{best['artifact_uri']}/model",
            "metric_name": metric,
            "metric_value": round(float(best_metric_val), 6),
            "selected_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        }
        if current_entry.get("model_path"):
            new_entry["previous_model_path"] = current_entry["model_path"]
        if best.get("feature_subset"):
            new_entry["feature_subset"] = best["feature_subset"]

        action = "[DRY RUN] Would select" if dry_run else "Selected"
        LOGGER.info(
            "%s target=%s context=%s model_type=%s %s=%.4f run_id=%s",
            action, target_name, context, new_entry["model_type"],
            metric, best_metric_val, best["run_id"],
        )
        print(
            f"  {action}: {target_name} [{context}] | {new_entry['model_type']} | "
            f"{metric}={best_metric_val:.4f} | run={best['run_id'][:8]}"
        )
        return new_entry

    def run(
        self,
        target: str | None = None,
        context: str | None = None,
        dry_run: bool = False,
        min_improvement: float = 0.005,
    ) -> None:
        targets = (
            [target]
            if target
            else [d.name for d in list_target_definitions() if d.name != "home_win"]
        )
        contexts = [context] if context else ["league", "international"]

        config = self.load_config()
        config.setdefault("contexts", {})
        for ctx in ["league", "international"]:
            config["contexts"].setdefault(ctx, {})

        changed = False
        print(f"\nModel selection{'  [DRY RUN]' if dry_run else ''}:")
        for ctx in contexts:
            for tgt in targets:
                current = config["contexts"].get(ctx, {}).get(tgt, {})
                new_entry = self._select_for_target_context(tgt, ctx, current, min_improvement, dry_run)
                if new_entry is not None and not dry_run:
                    config["contexts"][ctx][tgt] = new_entry
                    changed = True

        if changed and not dry_run:
            self._save_config(config)
            print(f"\nmodel_selection.yaml updated at {self.config_path}")
        elif dry_run:
            print("\n[DRY RUN] No changes written.")
        else:
            print("\nNo changes — current selections are still optimal.")
