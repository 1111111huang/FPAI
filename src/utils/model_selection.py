"""Model selection utility: select best MLflow runs per target/context and persist to YAML (US#78)."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import mlflow
import yaml

from src.logic.target_registry import list_target_definitions, get_target_definition
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

SELECTION_CONFIG_PATH = Path("config/model_selection.yaml")
DEFAULT_MODEL_DIR = Path("models")

# Metrics where lower = better
_LOWER_IS_BETTER = {"log_loss", "mae", "rmse", "test_log_loss", "test_mae", "test_rmse"}

# Classification targets use test_log_loss; regression targets use test_mae
_CLASSIFICATION_TARGETS = {"result_3way", "btts"}


def missing_features(required: list[str], available: set[str]) -> list[str]:
    """Return the subset of `required` not present in `available`.

    Used at promotion time (BUG-012 layer 3c) to check whether a candidate
    model's declared feature list can actually be produced by the live
    feature pipeline (FeatureFactory.build_for_match) before writing it to
    model_selection.yaml — catching a training/serving feature-set mismatch
    before it reaches live inference, not after.
    """
    return [f for f in required if f not in available]


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

    def __init__(
        self,
        config_path: str | Path = SELECTION_CONFIG_PATH,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        computable_features: set[str] | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.model_dir = Path(model_dir)
        # BUG-012 layer 3c: the set of feature names the live feature pipeline
        # can currently produce (e.g. FeatureFactory.build_for_match(...).columns
        # on a sample match). None disables the promotion-time coverage check
        # (e.g. in contexts with no live DB, or existing callers/tests that
        # predate this guard) — see run_select_best_models in main.py for how
        # this is populated in the real CLI path.
        self.computable_features = computable_features
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
            params = run.data.params or {}
            results.append({
                "run_id": run.info.run_id,
                "model_type": tags.get("model_family", tags.get("model_type", "unknown")),
                "artifact_uri": run.info.artifact_uri,
                "artifact_filename": params.get("artifact_filename"),
                "metrics": metrics,
                "feature_subset": params.get("feature_subset"),
            })
        return results

    def _run_artifact_resolves(self, run: dict[str, Any]) -> bool:
        """Whether this run's own model artifact actually exists on disk.

        BUG-014 layer 2: an MLflow run can carry a real, competitive metric
        while its artifact_filename points to a file that was never saved or
        has since been deleted -- such a run can never actually be loaded by
        ForecastService, so it must not be eligible to be "best" at all, not
        merely deprioritized against whatever the current champion is.
        """
        artifact_filename = run.get("artifact_filename")
        if not artifact_filename:
            # No artifact_filename param logged -- the pre-BUG-010-fix
            # MLflow-flavor path convention, never loadable either.
            return False
        return (self.model_dir / artifact_filename).exists()

    def _best_run(
        self,
        runs: list[dict[str, Any]],
        metric: str,
    ) -> dict[str, Any] | None:
        runs = [r for r in runs if self._run_artifact_resolves(r)]
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

        # BUG-014: a tied/no-improvement metric only justifies skipping
        # promotion if the current champion's model_path is actually usable.
        # select-best-models doesn't train anything itself, so re-running it
        # against an already-recorded champion finds the identical run every
        # time -- "no improvement" forever -- which would otherwise leave a
        # stale, broken model_path (e.g. a pre-BUG-010-fix MLflow artifact
        # URI) permanently un-refreshed, with no way to self-correct.
        current_model_path = current_entry.get("model_path")
        current_path_resolves = False
        if current_model_path:
            resolved = Path(current_model_path)
            if not resolved.is_absolute():
                resolved = self.model_dir.parent / resolved
            current_path_resolves = resolved.exists()

        if current_metric_val is not None and current_path_resolves:
            if not _is_better(float(best_metric_val), float(current_metric_val), metric, min_improvement):
                LOGGER.info(
                    "No improvement for target=%s context=%s: current=%.4f best=%.4f (min_improvement=%.4f)",
                    target_name, context, current_metric_val, best_metric_val, min_improvement,
                )
                return None
        elif current_metric_val is not None and not current_path_resolves:
            LOGGER.info(
                "Refreshing target=%s context=%s despite tied/no-improvement metric "
                "(current=%.4f best=%.4f) -- current model_path %r does not resolve "
                "to an existing file (BUG-014).",
                target_name, context, current_metric_val, best_metric_val, current_model_path,
            )

        # Prefer the plain joblib path (ForecastService.joblib.load-compatible) logged
        # via the "artifact_filename" param. Fall back to the MLflow-flavor autolog
        # path for older runs that predate that param — not loadable by ForecastService,
        # but keeps this function from crashing on legacy run data.
        model_path = (
            f"models/{best['artifact_filename']}"
            if best.get("artifact_filename")
            else f"{best['artifact_uri']}/model"
        )
        new_entry: dict[str, Any] = {
            "mlflow_run_id": best["run_id"],
            "model_type": best["model_type"],
            "model_path": model_path,
            "metric_name": metric,
            "metric_value": round(float(best_metric_val), 6),
            "selected_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        }
        if current_entry.get("model_path"):
            new_entry["previous_model_path"] = current_entry["model_path"]

        # BUG-012 layer 3d: prefer the feature_subset MLflow param when the
        # training run logged one (as international-context runs do); when it
        # didn't (as league-context runs previously never did), fall back to
        # the model's own .metadata.json feature_names — the training
        # pipeline already writes this file, so this is free, and it keeps
        # model_selection.yaml self-documenting for every context.
        feature_subset = best.get("feature_subset")
        artifact_feature_names: list[str] | None = None
        metadata_path = self.model_dir / f"{best.get('artifact_filename', '')}.metadata.json"
        if best.get("artifact_filename") and metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as meta_fh:
                artifact_feature_names = json.load(meta_fh).get("feature_names")
        if not feature_subset and artifact_feature_names:
            feature_subset = artifact_feature_names
        if feature_subset:
            new_entry["feature_subset"] = feature_subset

        # BUG-012 layer 3c: refuse to promote a model whose required features
        # the live feature pipeline can't currently produce — fail loudly here
        # instead of at live-inference time (see BUG-012 in documents/bugs.md).
        if self.computable_features is not None and feature_subset:
            gaps = missing_features(feature_subset, self.computable_features)
            if gaps:
                LOGGER.error(
                    "Refusing to promote target=%s context=%s run_id=%s: "
                    "%d required feature(s) not computable by the live feature "
                    "pipeline: %s",
                    target_name, context, best["run_id"], len(gaps), gaps,
                )
                print(
                    f"  REFUSED: {target_name} [{context}] | run={best['run_id'][:8]} | "
                    f"missing from live feature pipeline: {gaps}"
                )
                return None

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
