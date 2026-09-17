"""Model management utilities for training, evaluation, and versioned saving."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, precision_score

from src.logic.target_resolver import TargetResolver
from src.logic.target_registry import TargetDefinition, get_target_definition
from src.models.base_model import FPAIBaseModel, XGBoostModel, XGBoostRegressorModel
from src.models.ensemble_result_model import EnsembleResultModel
from src.models.goal_stacker import GoalStackerModel
from src.models.quantile_interval_model import QuantileIntervalModel
from src.models.skellam_result_model import SkellamResultModel
from src.models.two_stage_result_model import TwoStageResultModel
from src.utils.config_loader import AppSettings, load_settings
from src.utils.db_manager import DuckDBManager
from src.utils.fingerprint import data_fingerprint, file_fingerprint
from src.utils.mlflow_config import configure_mlflow_tracking
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# BUG-066: isotonic regression's monotonic step-function fit produces wide
# flat plateaus when fit on too few validation samples -- confirmed live,
# E0 result_3way's promoted calibrator mapped every raw away-probability in
# 0.4708-0.5139 (a genuinely differentiated range across real, distinct
# matches) to one constant 0.428571, because only ~14 distinct validation
# values informed that plateau. sklearn's own calibration docs recommend
# sigmoid (Platt) scaling below ~1000 calibration samples for exactly this
# overfitting failure mode; isotonic above that threshold still wins on
# log_loss (US#61's own original finding).
_MIN_ISOTONIC_SAMPLES = 1000


def _make_calibrator(n_samples: int) -> IsotonicRegression | _SigmoidCalibration:
    """Isotonic above _MIN_ISOTONIC_SAMPLES, else Platt/sigmoid scaling --
    sklearn's own internal (the same one CalibratedClassifierCV uses for
    method="sigmoid"), so no new dependency and identical .fit()/.predict()
    interface as IsotonicRegression (nothing downstream needs to branch)."""
    if n_samples >= _MIN_ISOTONIC_SAMPLES:
        return IsotonicRegression(out_of_bounds="clip")
    return _SigmoidCalibration()


def _compute_sample_weight(y: pd.Series, task_type: str, alpha: float = 1.0) -> np.ndarray | None:
    """Inverse-class-frequency sample weights for classification targets.

    Found live: result_3way's XGBoost classifier had 2.1% recall on 'draw'
    (SP1 test split) despite draws being ~25% of real outcomes -- trained
    with plain unweighted multiclass log-loss, so the model could minimize
    loss by mostly ignoring the harder-to-separate minority class. Reuses
    sklearn's own compute_sample_weight('balanced', ...) rather than a
    hand-rolled formula -- already a project dependency, standard technique.
    Regression targets have no notion of class balance; returns None so
    every model's .train(sample_weight=None) call is a byte-identical no-op
    for them.

    alpha dampens the balanced weighting: weights ** alpha. alpha=1.0 (default)
    is full balancing (today's behavior, byte-identical to the pre-alpha
    function). alpha=0.0 collapses every weight to 1.0 -- the pre-08-14
    unweighted behavior. Added 2026-08-20 after full 'balanced' weighting
    (alpha=1.0) was found to overcorrect result_3way's draw-blindness bug
    into a draw-overprediction bug on E0/SP1 -- see
    docs/superpowers/specs/2026-08-20-result-3way-sample-weight-retune-design.md."""
    if task_type == "regression":
        return None
    from sklearn.utils.class_weight import compute_sample_weight

    weights = compute_sample_weight("balanced", y)
    if alpha != 1.0:
        weights = weights ** alpha
    return weights


def _compute_time_decay_weight(dates: pd.Series, half_life_days: float) -> np.ndarray:
    """Recency sample weights: exponential decay relative to the most recent
    date in the training set (day 0 == that match, weight 1.0), halving every
    half_life_days.

    Per direct user prioritization ("Do 8, 1, 3, 5, 6"), item #6: every model
    in this project has always weighted an 8-10-year-old match the same as a
    recent one -- this is the first mechanism giving recency any weight at
    all. Independent of and combined multiplicatively with
    _compute_sample_weight's class-balance weight (see train()/run_pipeline())
    rather than folded into one function, since they're orthogonal concerns:
    a match's class-balance weight depends only on its label, its recency
    weight only on its date.

    Relative to the training set's own most recent match (not real wall-clock
    "today") -- this project trains on archived historical data (some
    contexts, e.g. E0's raw_matches, haven't been refreshed past the end of
    a season), so "today" would apply an arbitrary, source-dependent extra
    discount having nothing to do with actual recency within the data."""
    dates = pd.to_datetime(dates)
    reference = dates.max()
    days_ago = (reference - dates).dt.total_seconds().to_numpy() / 86400.0
    return np.asarray(0.5 ** (days_ago / half_life_days))


def _classes_for_calibration(model: FPAIBaseModel) -> np.ndarray | None:
    """Ordered class labels matching predict_proba's column order.

    Tries the wrapper's own .classes_ first (XGBoostModel sets this from its
    internal LabelEncoder after training), then the underlying sklearn
    estimator's .classes_ (LRModel/RandomForestModel never set a wrapper-level
    one). sklearn/XGBoostModel's predict_proba column order always matches
    classes_ in sorted order -- same assumption XGBoostModel.predict()'s own
    inverse_transform already relies on."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = getattr(getattr(model, "model", None), "classes_", None)
    return np.asarray(classes) if classes is not None else None


def build_artifact_filename(target_name: str, competition_id: str | None, model_prefix: str, date_tag: str) -> str:
    """Construct a model artifact filename, disambiguated by competition (US#139 follow-up).

    Confirmed live: before this fix, filenames had no competition_id component
    at all (just f"{target}_{model_prefix}_v1_{date}.joblib"), so training two
    different competitions' models for the same target+model-type+date
    silently collided on disk -- training international's result_3way on the
    same day as SWE's result_3way overwrote SWE's already-committed artifact
    with international's content, while config/model_selection.yaml's
    contexts.SWE entry kept pointing at the now-wrong file.

    E0 keeps its pre-existing unsuffixed filename shape (competition_id "E0"
    or None collapses to no suffix) so none of E0's already-recorded
    model_selection.yaml entries needed to be re-pointed by this fix.
    """
    competition_tag = "" if competition_id in ("E0", None) else f"_{competition_id.lower()}"
    return f"{target_name}{competition_tag}_{model_prefix}_v1_{date_tag}.joblib"


class ModelManager:
    """Handle training data preparation, model evaluation, and model versioning."""

    def __init__(
        self,
        model: FPAIBaseModel,
        config_path: str = "config.yaml",
        league_tier: str = "all",
        test_season: str = "time_split",
        feature_version: str = "v1",
        target_config: dict[str, str | float | int] | None = None,
        feature_subset: list[str] | None = None,
        context: str = "E0",
        competition_id: str = "E0",
        sample_weight_alpha: float = 1.0,
        time_decay_half_life_days: float | None = None,
        refit_on_full_data: bool = False,
    ) -> None:
        """Initialize manager with a model instance and YAML config path."""
        self.model = model
        self.config_path = Path(config_path)
        self.config: AppSettings = load_settings(str(self.config_path))
        self.db_manager = DuckDBManager(config_path=str(self.config_path))
        self.model_dir = Path(self.config.paths.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = float(self.config.settings.test_size)
        # context/sweep_stage tags make this run_pipeline() artifact eligible for
        # ModelSelector._fetch_eligible_runs() (requires both tags). Sweep-based
        # training (src/utils/sweep_runner.py) tags sweep_stage itself and never
        # calls run_pipeline(), so this default doesn't affect sweep runs.
        self.mlflow_tags = {
            "league_tier": league_tier,
            "test_season": test_season,
            "feature_version": feature_version,
            "target": (target_config or {}).get("target") or (target_config or {}).get("target_type", "home_win"),
            "context": context,
            "sweep_stage": "final",
        }
        self.target_config = target_config or {"target_type": "home_win"}
        self.target_definition = get_target_definition(
            str(self.target_config.get("target") or self.target_config.get("target_type", "home_win"))
        )
        # US#62: optional override to train on a subset of selected_features
        self.feature_subset: list[str] | None = feature_subset
        self.competition_id: str = competition_id
        # 2026-08-20: dampens compute_sample_weight('balanced', ...) -- see
        # _compute_sample_weight's own docstring. 1.0 (default) preserves
        # every existing caller's exact current behavior.
        self.sample_weight_alpha: float = sample_weight_alpha
        # US#189: None (default) preserves every existing caller's exact
        # current behavior -- no recency weighting at all unless a caller
        # opts in with a real half-life. Populated by prepare_training_data()
        # (same side-effect-attribute pattern as self.training_cutoff below).
        self.time_decay_half_life_days: float | None = time_decay_half_life_days
        self.train_dates: pd.Series | None = None
        self.full_data_dates: pd.Series | None = None
        # Direct user decision (2026-09-16): the deployed artifact should be
        # a static, season-frozen evidence provider -- train/val/test exists
        # to SELECT the architecture/hyperparameters honestly, not to starve
        # the actually-served model of ~30% of its own available history.
        # True refits self.model on train+val+test combined (chronologically
        # everything up to the real data ceiling) as the LAST step of
        # run_pipeline(), after the honest held-out metrics are already
        # computed and logged from the train-only fit -- so promotion
        # decisions still compare genuinely out-of-sample numbers, only the
        # artifact that actually gets saved/served is the full-data one.
        # Calibration fitting is skipped in this mode (see run_pipeline):
        # X_val is no longer held-out once folded into the final fit, so
        # calibrating against it there would be in-sample, not validation.
        self.refit_on_full_data: bool = refit_on_full_data
        self.full_data_cutoff: str | None = None
        # US#185: defense-in-depth for direct-Python usage that bypasses
        # main.py's own call -- cheap/idempotent, see mlflow_config.py.
        configure_mlflow_tracking(config_path)
        mlflow.set_experiment("FPAI_Evolution")

    def _load_selected_features(self) -> list[str]:
        schema_path = self.config_path.parent / "config" / "schema.yaml"
        if not schema_path.exists():
            raise FileNotFoundError(f"Missing schema file: {schema_path}")
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = yaml.safe_load(handle) or {}
        training_setup = schema.get("training_setup", {})
        selected = training_setup.get("selected_features")
        if not isinstance(selected, list) or not selected:
            raise ValueError("training_setup.selected_features must be a non-empty list in config/schema.yaml.")
        if not all(isinstance(item, str) and item.strip() for item in selected):
            raise ValueError("training_setup.selected_features must contain only non-empty strings.")
        all_features = [item.strip() for item in selected]
        # US#62: if a feature_subset is specified, filter to intersection with schema list
        if self.feature_subset:
            schema_set = set(all_features)
            subset = [f for f in self.feature_subset if f in schema_set]
            if not subset:
                raise ValueError(
                    f"feature_subset contains no features present in schema.yaml: {self.feature_subset[:5]}"
                )
            LOGGER.info("Feature subset active: %d/%d features", len(subset), len(all_features))
            if mlflow.active_run() is not None:
                mlflow.log_param("feature_subset_size", len(subset))
            return subset
        # US#66: check for per-target feature lists in schema.yaml
        target_name = self.target_definition.name
        target_features_map = schema.get("target_features", {})
        if target_name in target_features_map:
            target_list = target_features_map[target_name]
            if isinstance(target_list, list) and target_list:
                all_set = set(all_features)
                subset = [f for f in target_list if f in all_set]
                if subset:
                    LOGGER.info(
                        "Using per-target feature list for '%s': %d features", target_name, len(subset)
                    )
                    if mlflow.active_run() is not None:
                        mlflow.log_param("target_feature_count", len(subset))
                    return subset
        # US#97: filter SQUAD_* features for competitions whose registry entry
        # does not include "SQUAD" in enabled_feature_groups.
        # US#133: additionally gate OFF/DEF/OPP_ADJ/STRENGTH/INTERACTION
        # sub-features by their raw-data dependency (goals vs. shots/SOT vs.
        # corners), so a competition whose data source lacks shots/corners
        # entirely (e.g. Sweden's football-data.co.uk "New Leagues" CSVs) can
        # opt out of those specific sub-features via the registry instead of
        # silently cold-start-imputing a wholesale-missing column with
        # another competition's column mean. See src/logic/feature_groups.py.
        try:
            from src.logic.competition_registry import get_competition_definition
            from src.logic.feature_groups import resolve_feature_group_tag
            comp_def = get_competition_definition(self.competition_id)
            enabled_groups = set(comp_def.enabled_feature_groups)
            if "SQUAD" not in enabled_groups:
                all_features = [f for f in all_features if not f.startswith("SQUAD_")]
                all_features = [f for f in all_features if not f.startswith("LUCK_")]
                all_features = [f for f in all_features if not f.startswith("XOC_")]
                all_features = [f for f in all_features if not f.startswith("FRDS_")]
                all_features = [f for f in all_features if not f.startswith("DEF_ANCHOR_")]
                # US#175: same lineup/raw_player_match_stats dependency as the
                # four prefixes above.
                all_features = [f for f in all_features if not f.startswith("LINEUP_")]

            def _passes_group_gate(feature: str) -> bool:
                tag = resolve_feature_group_tag(feature)
                # None means "not governed by this mechanism" (e.g. DIS/CTX/MKT/
                # EFFICIENCY/H2H/SQUAD-managed prefixes) -- always pass through,
                # unchanged from behavior before US#133.
                return tag is None or tag in enabled_groups

            all_features = [f for f in all_features if _passes_group_gate(f)]
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Feature-group gating skipped for competition_id=%r — registry unavailable or unknown: %s",
                self.competition_id,
                exc,
            )
        return all_features

    @staticmethod
    def _fit_and_save_calibrator(
        model: FPAIBaseModel,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_path: Path,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
    ) -> dict[str, float] | None:
        """Fit a probability calibrator on val-set probabilities and save as sidecar.

        X_test/y_test (optional, but always passed by run_pipeline): the
        REAL promotion gate, added after BUG-066 found the val-set-only
        ll_before/ll_after check can't distinguish a genuinely better
        calibrator from one that just overfits a plateau to the exact data
        it was fit on -- confirmed live, 7 of 8 real production calibrators
        tested showed a clean in-sample gain and a real out-of-sample loss.
        A calibrator that doesn't ALSO improve log_loss on X_test (never
        touched during fitting) is not saved at all. Omitting X_test/y_test
        preserves the old val-only behavior for any caller that hasn't been
        updated yet -- but every current caller (run_pipeline) now passes
        both, so this only matters for tests exercising the pre-gate shape.

        Returns a dict with log_loss before/after calibration (both
        measured on X_val, for continuity with existing diagnostics/mlflow
        logging), or None if calibration was skipped or failed the gate.
        """
        if not hasattr(model, "predict_proba"):
            return None
        try:
            raw_proba = np.asarray(model.predict_proba(X_val))

            if raw_proba.ndim == 2 and raw_proba.shape[1] == 2:
                # Binary classifier: labels are already numeric (0/1) here --
                # see TargetResolver.get_label, home_win/btts .astype(int)/(Int64).
                y_val_arr = pd.to_numeric(y_val, errors="coerce").astype(float).to_numpy()
                pos_proba = raw_proba[:, 1]
                calibrator = _make_calibrator(len(pos_proba))
                calibrator.fit(pos_proba, y_val_arr)
                cal_pos = calibrator.predict(pos_proba)
                cal_proba = np.stack([1 - cal_pos, cal_pos], axis=1)
                ll_before = float(log_loss(y_val_arr, raw_proba))
                ll_after = float(log_loss(y_val_arr, cal_proba))
                sidecar = {"type": "binary", "calibrator": calibrator}
            elif raw_proba.ndim == 2 and raw_proba.shape[1] > 2:
                # Multi-class: labels may be strings (result_3way's
                # 'home'/'draw'/'away') that don't survive pd.to_numeric --
                # found live, this previously produced an all-NaN y_val_arr
                # and silently no-op'd calibration for every such target.
                # Match against the model's own class order instead of coercing.
                class_labels = _classes_for_calibration(model)
                if class_labels is None or len(class_labels) != raw_proba.shape[1]:
                    return None
                y_val_raw = y_val.to_numpy()
                n_classes = raw_proba.shape[1]
                calibrators = []
                cal_proba = np.zeros_like(raw_proba)
                for c in range(n_classes):
                    y_bin = (y_val_raw == class_labels[c]).astype(float)
                    cal = _make_calibrator(len(y_bin))
                    cal.fit(raw_proba[:, c], y_bin)
                    cal_proba[:, c] = cal.predict(raw_proba[:, c])
                    calibrators.append(cal)
                # Re-normalise rows so they sum to 1
                row_sums = cal_proba.sum(axis=1, keepdims=True).clip(min=1e-9)
                cal_proba /= row_sums
                ll_before = float(log_loss(y_val_raw, raw_proba, labels=class_labels))
                ll_after = float(log_loss(y_val_raw, cal_proba, labels=class_labels))
                sidecar = {"type": "multiclass", "calibrator": calibrators, "classes": class_labels}
            else:
                return None

            if X_test is not None and y_test is not None:
                test_raw_proba = np.asarray(model.predict_proba(X_test))
                if sidecar["type"] == "binary":
                    y_test_arr = pd.to_numeric(y_test, errors="coerce").astype(float).to_numpy()
                    test_cal_pos = sidecar["calibrator"].predict(test_raw_proba[:, 1])
                    test_cal_proba = np.stack([1 - test_cal_pos, test_cal_pos], axis=1)
                    test_ll_before = float(log_loss(y_test_arr, test_raw_proba))
                    test_ll_after = float(log_loss(y_test_arr, test_cal_proba))
                else:
                    y_test_raw = y_test.to_numpy()
                    test_cal_proba = np.zeros_like(test_raw_proba)
                    for c, cal in enumerate(sidecar["calibrator"]):
                        test_cal_proba[:, c] = cal.predict(test_raw_proba[:, c])
                    test_cal_proba /= test_cal_proba.sum(axis=1, keepdims=True).clip(min=1e-9)
                    test_ll_before = float(log_loss(y_test_raw, test_raw_proba, labels=sidecar.get("classes")))
                    test_ll_after = float(log_loss(y_test_raw, test_cal_proba, labels=sidecar.get("classes")))
                if test_ll_after >= test_ll_before:
                    LOGGER.warning(
                        "Calibration REJECTED (fails held-out gate) | val: before=%.4f after=%.4f | "
                        "test: before=%.4f after=%.4f -- not saved",
                        ll_before, ll_after, test_ll_before, test_ll_after,
                    )
                    return None

            # Found live while adding test coverage for this: several
            # existing tests call this function against a model_path that's
            # never actually written to disk (irrelevant to what they're
            # testing). In the real run_pipeline() call site, model.save()
            # always runs before this function, so model_path always
            # exists there -- but this function's own contract never
            # required that, and shouldn't start silently requiring it now.
            # None here means the same thing None already means for a
            # pre-fingerprint sidecar: nothing to check against.
            sidecar["model_fingerprint"] = file_fingerprint(model_path) if model_path.exists() else None
            cal_path = model_path.with_suffix(model_path.suffix + ".calibration.pkl")
            joblib.dump(sidecar, str(cal_path))
            if mlflow.active_run() is not None:
                mlflow.log_artifact(str(cal_path))
                mlflow.log_metric("val_log_loss_uncalibrated", ll_before)
                mlflow.log_metric("val_log_loss_calibrated", ll_after)
                mlflow.log_metric("calibration_improvement", ll_before - ll_after)
            LOGGER.info(
                "Calibration saved | before=%.4f after=%.4f delta=%.4f | %s",
                ll_before, ll_after, ll_before - ll_after, cal_path.name,
            )
            return {"log_loss_before": ll_before, "log_loss_after": ll_after}
        except Exception as exc:
            LOGGER.warning("Calibration skipped: %s", exc)
            return None

    @staticmethod
    def _log_selected_features(selected_features: list[str]) -> None:
        active_run = mlflow.active_run()
        if active_run is None:
            return
        mlflow.log_param("selected_features", ",".join(selected_features))

    @staticmethod
    def _extract_feature_importance(feature_names: list[str], model: FPAIBaseModel) -> pd.DataFrame:
        estimator = getattr(model, "model", None)
        if estimator is None:
            return pd.DataFrame(columns=["feature", "importance"])
        importances = None
        if hasattr(estimator, "feature_importances_"):
            importances = getattr(estimator, "feature_importances_")
        elif hasattr(estimator, "coef_"):
            coef = getattr(estimator, "coef_")
            try:
                importances = np.abs(coef).ravel()
            except Exception:
                importances = None
        if importances is None:
            return pd.DataFrame(columns=["feature", "importance"])
        values = np.asarray(importances, dtype=float)
        if values.ndim > 1:
            values = np.mean(np.abs(values), axis=0)
        if len(values) != len(feature_names):
            values = values.ravel()[: len(feature_names)]
        return pd.DataFrame(
            {"feature": list(feature_names)[: len(values)], "importance": list(values)}
        ).sort_values("importance", ascending=False)

    @staticmethod
    def _log_feature_importance(feature_names: list[str], model: FPAIBaseModel) -> None:
        active_run = mlflow.active_run()
        if active_run is None:
            return
        df = ModelManager._extract_feature_importance(feature_names, model)
        if df.empty:
            return

        with TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "feature_importance.csv"
            df.to_csv(out_path, index=False)
            mlflow.log_artifact(str(out_path))
            plot_path = Path(tmpdir) / "feature_importance.png"
            top = df.head(20)
            try:
                import matplotlib.pyplot as plt
            except Exception:
                return
            plt.figure(figsize=(8, 6))
            plt.barh(top["feature"][::-1], top["importance"][::-1])
            plt.title("Top 20 Feature Importances")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(str(plot_path))

    def _build_artifact_metadata(
        self,
        model_path: Path,
        feature_names: list[str],
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, object]:
        """Build sidecar metadata for forecast-time diagnostics and intervals."""
        created_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        metadata: dict[str, object] = {
            "target": self.target_definition.name,
            "task_type": self.target_definition.task_type,
            "classes": list(self.target_definition.classes),
            "model_type": self.model.__class__.__name__,
            "feature_schema_version": self.mlflow_tags.get("feature_version", "v1"),
            "feature_names": feature_names,
            "artifact_path": str(model_path),
            "artifact_name": model_path.name,
            "created_at": created_at,
            "training_cutoff": getattr(self, "training_cutoff", None),
            "primary_metric": self.target_definition.primary_metric,
            "secondary_metrics": list(self.target_definition.secondary_metrics),
        }
        if self.target_definition.task_type == "regression":
            validation_predictions = np.asarray(self.model.predict(X_val), dtype=float)
            residuals = pd.to_numeric(y_val, errors="coerce").astype(float).to_numpy() - validation_predictions
            residuals = residuals[~np.isnan(residuals)]
            if len(residuals):
                metadata["prediction_interval"] = {
                    "coverage": 0.8,
                    "lower_residual": float(np.quantile(residuals, 0.10)),
                    "upper_residual": float(np.quantile(residuals, 0.90)),
                    "method": "validation_residual_quantile",
                }
        feature_importance = self._extract_feature_importance(feature_names, self.model)
        metadata["feature_importance"] = feature_importance.head(50).to_dict(orient="records")
        # US#197: a content hash of X_val's shape/values, so a later "did
        # this model's training data actually change" question (e.g. after
        # a feature_store rebuild) is a direct equality check instead of
        # the forensic raw-vs-calibrated probability reconstruction this
        # session needed by hand.
        metadata["training_data_fingerprint"] = data_fingerprint({
            "n_rows": len(X_val),
            "columns": sorted(X_val.columns),
            "value_summary": {
                col: round(float(X_val[col].mean()), 6)
                for col in sorted(X_val.columns) if pd.api.types.is_numeric_dtype(X_val[col])
            },
        })
        return metadata

    @staticmethod
    def _write_artifact_metadata(model_path: Path, metadata: dict[str, object]) -> Path:
        metadata_path = model_path.with_suffix(model_path.suffix + ".metadata.json")
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if mlflow.active_run() is not None:
            mlflow.log_artifact(str(metadata_path))
        return metadata_path

    def prepare_training_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        """Build feature matrix and labels, then apply a chronological 70/15/15 split."""
        feature_columns = self._load_selected_features()
        for feature_name in feature_columns:
            if not feature_name.replace("_", "").isalnum():
                raise ValueError(f"Invalid feature name in selected_features: {feature_name}")
        feature_select = ",\n                    ".join(f"f.{name}" for name in feature_columns)
        label_columns = list(dict.fromkeys(self.target_definition.label_columns))
        label_select = ",\n                    ".join(f"r.{name}" for name in label_columns)

        # US#131 fix: this query previously had NO competition/league filter at
        # all, joining the *entire* raw_matches/feature_store tables regardless
        # of self.competition_id. That was invisible while only E0 existed, but
        # once Sweden's rows also existed in the shared tables, training with
        # context=SWE silently trained on E0 data instead: Sweden's 74-feature
        # list includes 9 MKT_AH_*/MKT_LAMBDA_*/MKT_IMPLIED_OVER25 features that
        # are permanently NaN for Sweden (no O/U-2.5 or AH odds in its source)
        # but populated for E0 -- the mandatory non-null dropna below (for
        # non-XGBoost models) then silently dropped every Sweden row and kept
        # only E0's, training an EPL model mislabeled context=SWE. Filter by
        # this competition's own league_code so training data always matches
        # the context it's tagged with. A competition with no single league_code
        # (e.g. "international"/general_purpose, league_code=None) intentionally
        # stays unfiltered -- pooling across every competition is that tier's
        # actual design (see US#138), not a gap to close here.
        league_filter_sql = ""
        params: list[str] = []
        try:
            from src.logic.competition_registry import get_competition_definition

            comp_def = get_competition_definition(self.competition_id)
            if comp_def.league_code is not None:
                league_filter_sql = "WHERE r.league = ?"
                params.append(comp_def.league_code)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "League filter skipped for competition_id=%r — registry unavailable or unknown: %s",
                self.competition_id,
                exc,
            )

        with self.db_manager.connection(read_only=True) as conn:
            df = conn.execute(
                f"""
                SELECT
                    r.match_id,
                    r.date,
                    r.odds_h,
                    {label_select},
                    {feature_select}
                FROM raw_matches r
                INNER JOIN feature_store f ON r.match_id = f.match_id
                {league_filter_sql}
                ORDER BY r.date, r.match_id
                """,
                params,
            ).fetchdf()

        if df.empty:
            raise ValueError("No joined training data found in raw_matches and feature_store.")

        df["target"] = TargetResolver.get_label(df, self.target_config)
        # XGBoost handles NaN features natively; only require a non-null target.
        # Non-XGBoost models require all feature columns to be present.
        # TwoStageResultModel (US#181) is built entirely from XGBClassifier
        # sub-models, so it tolerates NaN the same way -- unlike
        # GoalStackerModel, which mixes in sklearn's PoissonRegressor/Ridge
        # and genuinely needs the strict dropna.
        required_non_null = ["target"]
        if not isinstance(self.model, (XGBoostModel, XGBoostRegressorModel, TwoStageResultModel, SkellamResultModel, QuantileIntervalModel, EnsembleResultModel)):
            required_non_null.extend(feature_columns)
        df = df.dropna(subset=required_non_null).reset_index(drop=True)

        if df.empty:
            raise ValueError("No rows left after dropping records with missing labels or features.")

        for feature_name in feature_columns:
            if feature_name not in df.columns:
                raise ValueError(f"Missing selected feature in training data: {feature_name}")

        X = df[feature_columns]
        y = df["target"]

        total = len(df)
        train_ratio = float(self.config.settings.train_split)
        val_ratio = float(self.config.settings.val_split)
        test_ratio = float(self.config.settings.test_split)
        ratio_sum = train_ratio + val_ratio + test_ratio
        if ratio_sum <= 0:
            raise ValueError("Train/val/test split ratios must sum to a positive value.")
        train_ratio = train_ratio / ratio_sum
        val_ratio = val_ratio / ratio_sum
        test_ratio = test_ratio / ratio_sum

        train_end = max(1, int(total * train_ratio))
        val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))
        val_end = min(val_end, total - 1)
        self.training_cutoff = pd.to_datetime(df.iloc[train_end - 1]["date"]).isoformat()
        # Real data ceiling (train+val+test's own last date) -- distinct
        # from training_cutoff above, which only reflects the train-only
        # split boundary. Recorded here so a refit_on_full_data artifact's
        # metadata can honestly state what it was actually trained through,
        # rather than leaving the misleadingly-stale training_cutoff as the
        # only recorded date.
        self.full_data_cutoff = pd.to_datetime(df.iloc[-1]["date"]).isoformat()
        # US#189: train rows' own dates, aligned by position to X_train/y_train
        # (same df slice) -- consumed by _compute_time_decay_weight in
        # train()/run_pipeline() when time_decay_half_life_days is set.
        self.train_dates = pd.to_datetime(df.iloc[:train_end]["date"]).reset_index(drop=True)
        # All of train+val+test's own dates, same row order pd.concat(X_train,
        # X_val, X_test) produces -- the refit_on_full_data fit needs THESE
        # dates for its own time-decay weights, not train_dates (see
        # _combine_time_decay's dates param and the bug noted there).
        self.full_data_dates = pd.to_datetime(df["date"]).reset_index(drop=True)

        X_train = X.iloc[:train_end].copy()
        X_val = X.iloc[train_end:val_end].copy()
        X_test = X.iloc[val_end:].copy()
        y_train = y.iloc[:train_end].copy()
        y_val = y.iloc[train_end:val_end].copy()
        y_test = y.iloc[val_end:].copy()
        test_meta = df.iloc[val_end:][["match_id", "odds_h"]].copy()

        # Coerce features to numeric and ensure missing values are np.nan (XGBoost-compatible).
        X_train = X_train.apply(pd.to_numeric, errors="coerce").astype(float)
        X_val = X_val.apply(pd.to_numeric, errors="coerce").astype(float)
        X_test = X_test.apply(pd.to_numeric, errors="coerce").astype(float)
        X_train = X_train.replace({pd.NA: np.nan})
        X_val = X_val.replace({pd.NA: np.nan})
        X_test = X_test.replace({pd.NA: np.nan})

        if not isinstance(self.model, (XGBoostModel, XGBoostRegressorModel, TwoStageResultModel, SkellamResultModel, QuantileIntervalModel, EnsembleResultModel)):
            if X_train.isna().any().any() or X_val.isna().any().any() or X_test.isna().any().any():
                raise ValueError(
                    "Missing values detected in features. "
                    "Current model does not support NaNs; use XGBoost or add imputation."
                )

        if self.target_definition.task_type != "regression" and y_train.nunique() < 2:
            raise ValueError("Training split has a single class; cannot train Logistic Regression.")

        return X_train, X_val, X_test, y_train, y_val, y_test, test_meta

    @staticmethod
    def _positive_probability(probabilities: np.ndarray) -> np.ndarray:
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()

    @staticmethod
    def _classification_loss(
        y_true: pd.Series,
        probabilities: np.ndarray,
        definition: TargetDefinition,
        model: FPAIBaseModel,
    ) -> float:
        if probabilities.ndim == 1:
            classes = list(range(2)) if definition.name in {"home_win", "btts"} else list(definition.classes)
            return float(log_loss(y_true, probabilities, labels=classes))
        estimator = getattr(model, "model", None)
        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = getattr(estimator, "classes_", None)
        labels = list(classes) if classes is not None else list(definition.classes)
        return float(log_loss(y_true, probabilities, labels=labels))

    def _evaluate_target(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: pd.DataFrame | None = None,
        y_train: pd.Series | None = None,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> tuple[dict[str, float], np.ndarray]:
        """Evaluate the configured target with registry-defined metrics.
        
        Optionally evaluate on train/val/test splits with explicit split labels.
        When all splits are provided, logs metrics for all three with '_train', '_val', '_test' suffixes.
        """
        def _compute_metrics(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
            """Compute metrics for a given split."""
            if self.target_definition.task_type == "regression":
                predictions = np.asarray(self.model.predict(X), dtype=float)
                mae = float(mean_absolute_error(y, predictions))
                mse = float(mean_squared_error(y, predictions))
                rmse = float(np.sqrt(mse))
                return {"mae": mae, "rmse": rmse}

            probabilities = np.asarray(self.model.predict_proba(X))
            predictions = np.asarray(self.model.predict(X))
            accuracy = float(accuracy_score(y, predictions))
            metrics = {
                "log_loss": self._classification_loss(y, probabilities, self.target_definition, self.model),
                "accuracy": accuracy,
            }
            if self.target_definition.name in {"home_win", "btts"}:
                positive = self._positive_probability(probabilities)
                metrics["precision"] = float(precision_score(y, positive >= 0.5, zero_division=0))
            return metrics

        # Evaluate test split (required)
        test_metrics = _compute_metrics(X_test, y_test)

        # If all splits provided, evaluate train and val as well, and log all three
        if X_train is not None and y_train is not None and X_val is not None and y_val is not None:
            train_metrics = _compute_metrics(X_train, y_train)
            val_metrics = _compute_metrics(X_val, y_val)
            
            active_run = mlflow.active_run()
            if active_run is not None:
                for split_name, split_metrics in {
                    "train": train_metrics,
                    "val": val_metrics,
                    "test": test_metrics,
                }.items():
                    for metric_name, value in split_metrics.items():
                        mlflow.log_metric(f"{metric_name}_{split_name}", float(value))
                        mlflow.log_metric(f"{split_name}_{metric_name}", float(value))
        
        # Get predictions for test split
        if self.target_definition.task_type == "regression":
            prediction_output = np.asarray(self.model.predict(X_test), dtype=float)
        else:
            prediction_output = np.asarray(self.model.predict_proba(X_test))
        
        return test_metrics, prediction_output

    def _combine_time_decay(self, sample_weight: np.ndarray | None, dates: pd.Series | None = None) -> np.ndarray | None:
        """US#189: multiply in the recency weight, if configured. A no-op
        (returns sample_weight unchanged) when time_decay_half_life_days is
        None (the default) or no dates are available yet.

        `dates` defaults to self.train_dates (the train-only fit) -- pass
        self.full_data_dates explicitly for the refit_on_full_data fit.
        Bug found live (2026-09-16): before this parameter existed, the
        full-data refit always multiplied its train+val+test-length sample
        weights against train_dates' train-only length, a shape mismatch
        that had simply never been exercised until a caller combined
        refit_on_full_data with time_decay_half_life_days for the first
        time. See tests/test_model_manager_refit_full_data.py."""
        dates = self.train_dates if dates is None else dates
        if self.time_decay_half_life_days is None or dates is None:
            return sample_weight
        decay_weight = _compute_time_decay_weight(dates, self.time_decay_half_life_days)
        return decay_weight if sample_weight is None else np.asarray(sample_weight) * decay_weight

    def train(self) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
        """Train on the chronological train split, tune on val, and return test predictions."""
        selected_features = self._load_selected_features()
        self._log_selected_features(selected_features)
        X_train, X_val, X_test, y_train, y_val, y_test, test_meta = self.prepare_training_data()
        eval_set = [(X_val, y_val)] if isinstance(self.model, (XGBoostModel, XGBoostRegressorModel, GoalStackerModel, TwoStageResultModel, QuantileIntervalModel, EnsembleResultModel)) else None
        sample_weight = _compute_sample_weight(y_train, self.target_definition.task_type, alpha=self.sample_weight_alpha)
        sample_weight = self._combine_time_decay(sample_weight)
        self.model.train(X_train, y_train, eval_set=eval_set, sample_weight=sample_weight)
        self._log_feature_importance(list(X_train.columns), self.model)
        if isinstance(self.model, (XGBoostModel, XGBoostRegressorModel)):
            estimator = getattr(self.model, "model", None)
            if estimator is not None:
                best_iter = getattr(estimator, "best_iteration", None)
                if best_iter is not None and mlflow.active_run() is not None:
                    mlflow.log_metric("best_iteration", int(best_iter))
                evals = getattr(estimator, "evals_result_", None)
                if isinstance(evals, dict):
                    logloss_hist = evals.get("validation_0", {}).get("logloss", [])
                    if logloss_hist and mlflow.active_run() is not None:
                        mlflow.log_metric("val_logloss", float(logloss_hist[-1]))
        probabilities = self.model.predict_proba(X_test)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            positive_proba = pd.Series(probabilities[:, 1], index=y_test.index)
        else:
            positive_proba = pd.Series(probabilities.ravel(), index=y_test.index)
        return y_test, test_meta, positive_proba

    def run_pipeline(self, external_run: bool = False) -> Path:
        """Train model, evaluate it, and save a timestamped artifact path."""
        try:
            selected_features = self._load_selected_features()
            self._log_selected_features(selected_features)
            X_train, X_val, X_test, y_train, y_val, y_test, test_meta = self.prepare_training_data()

            if isinstance(self.model, (XGBoostModel, XGBoostRegressorModel)):
                mlflow.xgboost.autolog()
            else:
                mlflow.sklearn.autolog()

            def _run_training() -> Path:
                mlflow.set_tags(self.mlflow_tags)
                mlflow.set_tag("primary_metric", self.target_definition.primary_metric)
                mlflow.set_tag("secondary_metrics", ",".join(self.target_definition.secondary_metrics))
                mlflow.log_param("target_type", self.target_definition.name)
                eval_set = [(X_val, y_val)] if isinstance(self.model, (XGBoostModel, XGBoostRegressorModel, GoalStackerModel, TwoStageResultModel, QuantileIntervalModel, EnsembleResultModel)) else None
                sample_weight = _compute_sample_weight(y_train, self.target_definition.task_type, alpha=self.sample_weight_alpha)
                sample_weight = self._combine_time_decay(sample_weight)
                self.model.train(X_train, y_train, eval_set=eval_set, sample_weight=sample_weight)
                self._log_feature_importance(list(X_train.columns), self.model)
                if isinstance(self.model, (XGBoostModel, XGBoostRegressorModel)):
                    estimator = getattr(self.model, "model", None)
                    if estimator is not None:
                        best_iter = getattr(estimator, "best_iteration", None)
                        if best_iter is not None:
                            mlflow.log_metric("best_iteration", int(best_iter))
                        evals = getattr(estimator, "evals_result_", None)
                        if isinstance(evals, dict):
                            logloss_hist = evals.get("validation_0", {}).get("logloss", [])
                            if logloss_hist:
                                mlflow.log_metric("val_logloss", float(logloss_hist[-1]))
                target_name = self.target_definition.name
                metrics, prediction_output = self._evaluate_target(X_test, y_test, X_train, y_train, X_val, y_val)
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, float(value))
                    mlflow.log_metric(f"{target_name}_{metric_name}", float(value))
                    LOGGER.info("%s %s: %.4f", target_name, metric_name, value)

                # Direct user decision (2026-09-16): metrics above are the
                # honest, held-out promotion signal (train-only fit vs.
                # genuinely unseen val/test) -- unchanged. The ARTIFACT that
                # actually gets saved/served is a different question: once
                # this architecture/hyperparameter choice is validated, the
                # deployed model should be a static, season-frozen evidence
                # provider trained on everything available up to the real
                # data ceiling, not just the oldest 70%. Refit in place,
                # after metrics are already locked in, so nothing here can
                # leak into the promotion decision above.
                if self.refit_on_full_data:
                    X_all = pd.concat([X_train, X_val, X_test])
                    y_all = pd.concat([y_train, y_val, y_test])
                    full_sample_weight = _compute_sample_weight(y_all, self.target_definition.task_type, alpha=self.sample_weight_alpha)
                    full_sample_weight = self._combine_time_decay(full_sample_weight, dates=self.full_data_dates)
                    # Found live (2026-09-16): early_stopping_rounds is baked
                    # into the underlying XGBoost estimator's constructor for
                    # these model classes, unconditionally requiring an
                    # eval_set on every .fit() call for that instance's
                    # lifetime -- eval_set=None raised "Must have at least 1
                    # validation dataset for early stopping." X_val is
                    # already folded into X_all/y_all above, so reusing it
                    # here only means "watch this slice to decide when to
                    # stop adding trees," not a held-out evaluation -- a
                    # standard, accepted compromise for a final full-data
                    # refit, and no worse than the leakage every other
                    # X_all-inclusive model family already accepts by fitting
                    # once with no eval_set at all.
                    refit_eval_set = [(X_val, y_val)] if isinstance(
                        self.model, (XGBoostModel, XGBoostRegressorModel, GoalStackerModel, TwoStageResultModel, QuantileIntervalModel, EnsembleResultModel)
                    ) else None
                    self.model.train(X_all, y_all, eval_set=refit_eval_set, sample_weight=full_sample_weight)
                    LOGGER.info(
                        "%s: refit on full data (train+val+test, %d rows through %s) for serving",
                        target_name, len(X_all), self.full_data_cutoff,
                    )

                date_tag = datetime.now().strftime("%Y%m%d")
                model_prefix = self.model.__class__.__name__.lower().replace("model", "")
                save_path = self.model_dir / build_artifact_filename(
                    target_name, self.competition_id, model_prefix, date_tag
                )
                self.model.save(str(save_path))
                metadata = self._build_artifact_metadata(save_path, selected_features, X_val, y_val)
                metadata["metrics"] = {metric_name: float(value) for metric_name, value in metrics.items()}
                metadata["refit_on_full_data"] = self.refit_on_full_data
                if self.refit_on_full_data:
                    metadata["full_data_cutoff"] = self.full_data_cutoff
                # US#61: fit isotonic calibrator on val set for classifiers --
                # skipped when refit_on_full_data, since X_val is no longer
                # held-out once folded into the final fit above (calibrating
                # against it there would be in-sample, not validation).
                if self.target_definition.task_type != "regression" and not self.refit_on_full_data:
                    cal_metrics = self._fit_and_save_calibrator(self.model, X_val, y_val, save_path, X_test=X_test, y_test=y_test)
                    if cal_metrics:
                        metadata["calibration"] = cal_metrics
                self._write_artifact_metadata(save_path, metadata)
                mlflow.log_artifact(str(save_path))
                # ModelSelector reads this param to build a loadable model_path —
                # the autologged "model" artifact is an MLflow-flavor directory,
                # not the plain joblib file ForecastService expects.
                mlflow.log_param("artifact_filename", save_path.name)
                mlflow.set_tag("model_family", model_prefix)
                return save_path

            if external_run:
                return _run_training()

            with mlflow.start_run():
                return _run_training()
        except duckdb.Error:
            LOGGER.exception("Database failure during model pipeline.")
            raise
        except Exception:
            LOGGER.exception("Model pipeline failed.")
            raise
