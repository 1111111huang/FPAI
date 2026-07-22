"""Service for producing structured forecast JSON payloads."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier, XGBRegressor

from src.forecast.schema import validate_forecast_payload
from src.forecast.uncertainty import (
    normalized_entropy_uncertainty,
    poisson_count_distribution,
    residual_prediction_interval,
)
from src.logic.competition_registry import get_competition_definition
from src.logic.target_registry import TargetDefinition, get_target_definition, list_target_definitions
from src.utils.config_loader import AppSettings, load_settings
from src.utils.db_manager import DuckDBManager

# MKT_* features used for international-context inference (US#85/US#86)
_MKT_FEATURES = [
    "MKT_IMPLIED_HOME", "MKT_IMPLIED_DRAW", "MKT_IMPLIED_AWAY",
    "MKT_OVERROUND", "MKT_LAMBDA_TOTAL", "MKT_LAMBDA_HOME", "MKT_LAMBDA_AWAY",
    "MKT_POISSON_BTTS_PROB", "MKT_LAMBDA_AH_DIFF",
    "MKT_AH_LINE", "MKT_AH_HOME_ODDS", "MKT_AH_AWAY_ODDS", "MKT_IMPLIED_OVER25",
]


class ForecastService:
    """Load features and target artifacts to assemble forecast payloads."""

    def __init__(self, config_path: str = "config.yaml", targets: list[str] | None = None) -> None:
        self.config_path = Path(config_path)
        self.config: AppSettings = load_settings(str(self.config_path))
        self.db_manager = DuckDBManager(config_path=str(self.config_path))
        self.model_dir = Path(self.config.paths.model_dir)
        self.targets = [get_target_definition(target).name for target in targets] if targets else [
            definition.name for definition in list_target_definitions() if definition.name != "home_win"
        ]
        self.feature_names = self._load_selected_features()

    def _load_selected_features(self) -> list[str]:
        schema_path = self.config_path.parent / "config" / "schema.yaml"
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = yaml.safe_load(handle) or {}
        selected = schema.get("training_setup", {}).get("selected_features")
        if not isinstance(selected, list) or not selected:
            raise ValueError("training_setup.selected_features must be a non-empty list in config/schema.yaml.")
        return [str(feature).strip() for feature in selected if str(feature).strip()]

    def _fetch_feature_rows(
        self,
        match_ids: list[str] | None = None,
        league: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        for feature_name in self.feature_names:
            if not feature_name.replace("_", "").isalnum():
                raise ValueError(f"Invalid feature name in selected_features: {feature_name}")
        feature_select = ",\n                    ".join(f"f.{name}" for name in self.feature_names)
        filters: list[str] = []
        params: list[object] = []
        if match_ids:
            placeholders = ", ".join(["?"] * len(match_ids))
            filters.append(f"r.match_id IN ({placeholders})")
            params.extend(match_ids)
        if league:
            filters.append("UPPER(r.league) = ?")
            params.append(league.upper())
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
        query = f"""
            SELECT
                r.match_id,
                r.date,
                r.league,
                r.home_team,
                r.away_team,
                {feature_select}
            FROM raw_matches r
            INNER JOIN feature_store f ON r.match_id = f.match_id
            {where_clause}
            ORDER BY r.date, r.match_id
            {limit_clause}
        """
        with self.db_manager.connection() as conn:
            return conn.execute(query, params).fetchdf()

    def _latest_artifact(self, target: str) -> tuple[Path, dict[str, Any]] | None:
        candidates = sorted(
            self.model_dir.glob(f"{target}_*.joblib"),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            return None
        model_path = candidates[-1]
        metadata_path = model_path.with_suffix(model_path.suffix + ".metadata.json")
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        metadata.setdefault("target", target)
        metadata.setdefault("artifact_name", model_path.name)
        metadata.setdefault("model_type", "unknown")
        metadata.setdefault("feature_names", self.feature_names)
        return model_path, metadata

    @staticmethod
    def _load_model(model_path: Path, metadata: dict[str, Any]) -> Any:
        model_type = metadata.get("model_type", "")
        if model_type == "XGBoostModel":
            model = XGBClassifier()
            model.load_model(str(model_path))
            return model
        if model_type == "XGBoostRegressorModel":
            model = XGBRegressor()
            model.load_model(str(model_path))
            return model
        # Sniff file format: XGBoost native UBJSON starts with b'{'
        with open(model_path, "rb") as _f:
            _magic = _f.read(1)
        if _magic == b"{":
            # Regressor targets have "_regressor" or "_goals"/"_corners" in name
            _name = model_path.stem.lower()
            _regressor_hints = ("corner", "goal", "regressor", "xgb_regressor")
            if any(h in _name for h in _regressor_hints):
                _m = XGBRegressor()
            else:
                _m = XGBClassifier()
            _m.load_model(str(model_path))
            return _m
        return joblib.load(model_path)

    @staticmethod
    def _coerce_probability_vector(probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim == 2:
            probs = probs[0]
        if probs.ndim == 1 and len(probs) == 1:
            probs = np.asarray([1.0 - probs[0], probs[0]], dtype=float)
        total = float(probs.sum())
        if total <= 0:
            raise ValueError("Model produced probabilities with no positive mass.")
        return probs / total

    @staticmethod
    def _class_labels(definition: TargetDefinition, model: Any, probabilities: np.ndarray) -> list[str]:
        model_classes = getattr(model, "classes_", None)
        if model_classes is not None and len(model_classes) == len(probabilities):
            # If classes are integers and we have named classes, reconstruct the label encoder mapping.
            # sklearn LabelEncoder assigns integers in alphabetical order, so sort definition.classes.
            if all(isinstance(c, (int, np.integer)) for c in model_classes) and definition.classes:
                return list(sorted(definition.classes))
            if definition.name in {"btts", "home_win"} and set(model_classes) == {0, 1}:
                return list(definition.classes)
            return [str(label) for label in model_classes]
        if definition.classes:
            return list(definition.classes)
        return [str(index) for index in range(len(probabilities))]

    @staticmethod
    def _feature_completeness(row: pd.Series, feature_names: list[str]) -> float:
        if not feature_names:
            return 0.0
        present = row[feature_names].notna().sum()
        return round(float(present / len(feature_names)), 6)

    @staticmethod
    def _cold_start_risk(row: pd.Series, feature_names: list[str], feature_completeness: float) -> bool:
        rolling_features = [name for name in feature_names if "_R5" in name or name.endswith("_EMA5")]
        if rolling_features and row[rolling_features].isna().any():
            return True
        return feature_completeness < 0.85

    @staticmethod
    def _top_features(row: pd.Series, metadata_by_target: dict[str, dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
        merged: dict[str, float] = {}
        for metadata in metadata_by_target.values():
            for item in metadata.get("feature_importance", []) or []:
                name = str(item.get("feature", ""))
                if not name:
                    continue
                importance = float(item.get("importance", 0.0) or 0.0)
                merged[name] = max(merged.get(name, 0.0), importance)
        top = sorted(merged.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [
            {
                "name": name,
                "value": None if pd.isna(row.get(name)) else float(row.get(name)),
                "importance": round(float(importance), 6),
            }
            for name, importance in top
            if name in row.index
        ]

    @staticmethod
    def _format_date(value: Any) -> str:
        if pd.isna(value):
            return ""
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    def _predict_target(
        self,
        definition: TargetDefinition,
        model: Any,
        metadata: dict[str, Any],
        feature_row: pd.DataFrame,
    ) -> dict[str, Any]:
        # BUG-012 layer 3b: feature_row may carry the union of every loaded
        # target's columns (see forecast_upcoming) — slice to this target's
        # own trained feature list/order before predicting, so one target
        # needing extra columns never affects another's input shape.
        target_feature_names = metadata.get("feature_names")
        if target_feature_names:
            feature_row = feature_row[target_feature_names]

        if definition.task_type in {"binary_classification", "multiclass_classification"}:
            probabilities = self._coerce_probability_vector(model.predict_proba(feature_row))
            labels = self._class_labels(definition, model, probabilities)
            probability_map = {
                label: round(float(probability), 6)
                for label, probability in zip(labels, probabilities, strict=False)
            }
            return {
                "probabilities": probability_map,
                "uncertainty": normalized_entropy_uncertainty(probabilities),
            }

        expected = float(np.asarray(model.predict(feature_row), dtype=float).ravel()[0])
        payload: dict[str, Any] = {
            "expected": round(expected, 6),
            "distribution": poisson_count_distribution(expected),
        }
        interval_config = metadata.get("prediction_interval")
        if isinstance(interval_config, dict):
            payload["prediction_interval"] = residual_prediction_interval(
                expected=expected,
                lower_residual=float(interval_config.get("lower_residual", 0.0)),
                upper_residual=float(interval_config.get("upper_residual", 0.0)),
                coverage=float(interval_config.get("coverage", 0.8)),
            )
        return payload

    def _load_context_models(self, context: str) -> dict[str, tuple[TargetDefinition, Any, dict[str, Any]]]:
        """Load models from model_selection.yaml for a given context.

        Returns an empty dict (not a glob fallback) when the context has no entries —
        this lets the caller raise FileNotFoundError cleanly rather than loading
        models from the wrong context (which causes XGBoost feature-name mismatches).
        """
        selection_path = self.config_path.parent / "config" / "model_selection.yaml"
        if selection_path.exists():
            with selection_path.open("r", encoding="utf-8") as fh:
                sel_config = yaml.safe_load(fh) or {}
            ctx_data = sel_config.get("contexts", {}).get(context, {})
            if ctx_data:
                loaded: dict[str, tuple[TargetDefinition, Any, dict[str, Any]]] = {}
                for target in self.targets:
                    entry = ctx_data.get(target)
                    if entry is None:
                        continue
                    definition = get_target_definition(target)
                    model_path = Path(entry["model_path"])
                    if not model_path.is_absolute():
                        model_path = self.config_path.parent / model_path
                    if not model_path.exists():
                        continue
                    # BUG-012 layer 3a: the model's own .metadata.json sidecar
                    # (written by the training pipeline) is ground truth for
                    # what features it was actually trained on — prefer it over
                    # schema.yaml's aspirational selected_features list, which
                    # can drift ahead of what any given artifact supports.
                    metadata_path = model_path.with_suffix(model_path.suffix + ".metadata.json")
                    artifact_feature_names: list[str] | None = None
                    if metadata_path.exists():
                        with metadata_path.open("r", encoding="utf-8") as meta_fh:
                            artifact_metadata = json.load(meta_fh)
                        artifact_feature_names = artifact_metadata.get("feature_names")
                    metadata: dict[str, Any] = {
                        "target": target,
                        "model_type": entry.get("model_type", "unknown"),
                        "artifact_name": model_path.name,
                        "feature_names": entry.get("feature_subset") or artifact_feature_names or self.feature_names,
                        "feature_subset": entry.get("feature_subset"),
                    }
                    loaded[target] = (definition, self._load_model(model_path, metadata), metadata)
                if loaded:
                    return loaded

        return {}

    def _score_targets(
        self,
        feature_frame: pd.DataFrame,
        loaded: dict[str, tuple[TargetDefinition, Any, dict[str, Any]]],
        row: pd.Series,
        feature_names: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run prediction across all loaded targets for a single feature row."""
        forecast: dict[str, Any] = {}
        metadata_by_target = {t: data[2] for t, data in loaded.items()}
        for target, (definition, model, metadata) in loaded.items():
            forecast[target] = self._predict_target(definition, model, metadata, feature_frame)
        return forecast, metadata_by_target

    def forecast_upcoming(
        self,
        home_team: str,
        away_team: str,
        date: str,
        league: str,
        odds_h: float,
        odds_d: float,
        odds_a: float,
        match_type: str = "league",
        over25_odds: float | None = None,
        ah_line: float | None = None,
        ah_home_odds: float | None = None,
        ah_away_odds: float | None = None,
        targets: list[str] | None = None,
    ) -> dict[str, Any]:
        """Produce a forecast for a named upcoming match without requiring it in the feature store.

        US#84: league path — full rolling features via FeatureFactory.build_for_match.
        US#86: international path — MKT_* features only from supplied odds.
        """
        from src.features.feature_factory import FeatureFactory

        generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        active_targets = targets or self.targets

        if match_type == "international":
            effective_context = "international"
        else:
            # US#107: consult the competition registry before committing to full
            # team-history features. A league the registry doesn't know about,
            # or one explicitly tiered 'general_purpose', must fall back to the
            # market-odds-only path automatically rather than silently
            # cold-starting a mislabeled 'team_history_and_market' result.
            #
            # US#110: the model-context bucket is now keyed off the competition's
            # own competition_id (e.g. "E0", "SWE"), not a flat tier-derived
            # "league" string — so two competition_specific competitions never
            # collide over the same config/model_selection.yaml entry.
            registry_path = self.config_path.parent / "config" / "competitions.yaml"
            try:
                competition_def = get_competition_definition(league, registry_path=registry_path)
                tier = competition_def.tier
                resolved_competition_id = competition_def.competition_id
            except ValueError:
                tier = "general_purpose"
                resolved_competition_id = None
            effective_context = "international" if tier == "general_purpose" else resolved_competition_id

        if effective_context == "international":
            # US#86: compute MKT features from odds only, no team history needed
            prediction_basis = "market_odds_only"
            mkt_row = self._compute_mkt_features_from_odds(odds_h, odds_d, odds_a, over25_odds, ah_line, ah_home_odds, ah_away_odds)
            feature_row = mkt_row
            context = "international"
            unknown_team = False  # no team-identity lookup happens on this path
        else:
            # US#84: full feature computation
            prediction_basis = "team_history_and_market"
            factory = FeatureFactory(config_path=str(self.config_path))
            feature_row = factory.build_for_match(
                home_team=home_team, away_team=away_team, match_date=date,
                league=league, odds_h=odds_h, odds_d=odds_d, odds_a=odds_a,
                over25_odds=over25_odds, ah_line=ah_line, ah_home_odds=ah_home_odds, ah_away_odds=ah_away_odds,
            )
            # US#110: effective_context is already the resolved competition_id
            # (e.g. "E0") in this branch — never None here, since a None
            # resolved_competition_id only happens together with tier ==
            # "general_purpose", which takes the "international" branch above.
            context = effective_context
            # US#108: build_for_match's own zero-history detection, distinct from
            # the feature_completeness-based cold_start_risk computed below.
            unknown_team = bool(feature_row["_unknown_team"].iloc[0])

        loaded = self._load_context_models(context)
        if not loaded:
            raise FileNotFoundError(f"No target model artifacts found for context '{context}'.")

        # Filter loaded to active_targets
        loaded = {t: v for t, v in loaded.items() if t in active_targets}

        if effective_context == "international":
            feature_names_used = [f for f in _MKT_FEATURES if f in feature_row.columns]
        else:
            # BUG-012 layer 3b: use the union of each *loaded* model's own
            # feature list (from _load_context_models, which now reads each
            # artifact's .metadata.json) rather than schema.yaml's full
            # selected_features — a model is never asked for a column it
            # wasn't trained on, even if schema.yaml has since drifted ahead
            # of what that specific artifact supports.
            feature_names_used = sorted({
                name
                for _, __, metadata in loaded.values()
                for name in metadata.get("feature_names", [])
            })

        # Align feature frame to expected columns (fill missing with NaN) —
        # this must run before any feature_row[feature_names_used] access, or
        # a column absent from feature_row (e.g. a lineup-gated feature with
        # no match_lineups data yet) KeyErrors instead of cold-start imputing.
        for col in feature_names_used:
            if col not in feature_row.columns:
                feature_row[col] = float("nan")
        feature_frame = feature_row[feature_names_used].apply(pd.to_numeric, errors="coerce").astype(float)
        feature_count = feature_frame.notna().sum().sum()

        # Impute column means for any remaining NaN (cold-start)
        for col in feature_frame.columns:
            if feature_frame[col].isna().any():
                feature_frame[col] = feature_frame[col].fillna(0.0)

        row_series = feature_frame.iloc[0]
        forecast_result, metadata_by_target = self._score_targets(feature_frame, loaded, row_series, feature_names_used)

        completeness = round(float(feature_count) / max(len(feature_names_used), 1), 6)
        caveat = (
            "Market-odds-only prediction; team history not used."
            if effective_context == "international"
            else "Full team history and market features used."
        )

        payload: dict[str, Any] = {
            "match_id": f"__spot__{home_team}__{away_team}__{date}".replace(" ", "_"),
            "date": date,
            "league": league,
            "home_team": home_team,
            "away_team": away_team,
            "forecast": forecast_result,
            "explainability": {
                "top_features": self._top_features(row_series, metadata_by_target),
            },
            "diagnostics": {
                "model_version": f"forecast_suite_{context}_v1",
                "target_versions": {
                    target: {
                        "artifact": meta.get("artifact_name"),
                        "created_at": meta.get("created_at"),
                        "model_type": meta.get("model_type"),
                    }
                    for target, (_, __, meta) in loaded.items()
                },
                "feature_completeness": completeness,
                "cold_start_risk": completeness < 0.85,
                "generated_at": generated_at,
            },
            "data_quality": {
                "prediction_basis": prediction_basis,
                "feature_count": int(feature_count),
                "unknown_team": unknown_team,
                "caveat": caveat,
            },
        }
        validate_forecast_payload(payload)
        return payload

    @staticmethod
    def _compute_mkt_features_from_odds(
        odds_h: float,
        odds_d: float,
        odds_a: float,
        over25_odds: float | None = None,
        ah_line: float | None = None,
        ah_home_odds: float | None = None,
        ah_away_odds: float | None = None,
    ) -> pd.DataFrame:
        """Compute MKT_* features from raw odds (US#86 — international inference)."""
        import numpy as np
        from scipy.optimize import brentq

        inv_h = 1.0 / odds_h
        inv_d = 1.0 / odds_d
        inv_a = 1.0 / odds_a
        total = inv_h + inv_d + inv_a
        mkt_h = inv_h / total
        mkt_d = inv_d / total
        mkt_a = inv_a / total
        overround = total

        mkt_over25 = (1.0 / over25_odds) if over25_odds and over25_odds > 1.0 else np.nan

        def _poisson_cdf2(lam: float) -> float:
            return np.exp(-lam) * (1.0 + lam + lam * lam / 2.0)

        def _solve_lambda(p_over: float) -> float:
            if np.isnan(p_over) or p_over <= 0.0 or p_over >= 1.0:
                return np.nan
            try:
                return brentq(lambda lam: _poisson_cdf2(lam) - (1.0 - p_over), 0.01, 20.0)
            except ValueError:
                return np.nan

        lam_total = _solve_lambda(mkt_over25)
        ah_abs = abs(ah_line) if ah_line is not None else np.nan
        lam_home = (lam_total + ah_abs) / 2.0 if not np.isnan(lam_total) and not np.isnan(ah_abs) else np.nan
        lam_away = max((lam_total - ah_abs) / 2.0, 0.0) if not np.isnan(lam_total) and not np.isnan(ah_abs) else np.nan
        btts_prob = (1.0 - np.exp(-lam_home)) * (1.0 - np.exp(-lam_away)) if not np.isnan(lam_home) else np.nan
        lam_ah_diff = (lam_total - ah_abs) if not np.isnan(lam_total) and not np.isnan(ah_abs) else np.nan

        return pd.DataFrame([{
            "MKT_IMPLIED_HOME": mkt_h,
            "MKT_IMPLIED_DRAW": mkt_d,
            "MKT_IMPLIED_AWAY": mkt_a,
            "MKT_OVERROUND": overround,
            "MKT_LAMBDA_TOTAL": lam_total,
            "MKT_LAMBDA_HOME": lam_home,
            "MKT_LAMBDA_AWAY": lam_away,
            "MKT_POISSON_BTTS_PROB": btts_prob,
            "MKT_LAMBDA_AH_DIFF": lam_ah_diff,
            "MKT_AH_LINE": ah_line if ah_line is not None else np.nan,
            "MKT_AH_HOME_ODDS": ah_home_odds if ah_home_odds is not None else np.nan,
            "MKT_AH_AWAY_ODDS": ah_away_odds if ah_away_odds is not None else np.nan,
            "MKT_IMPLIED_OVER25": mkt_over25,
        }])

    def forecast(
        self,
        match_ids: list[str] | None = None,
        league: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return forecast payloads for selected matches."""
        rows = self._fetch_feature_rows(match_ids=match_ids, league=league, limit=limit)
        if rows.empty:
            return []

        loaded: dict[str, tuple[TargetDefinition, Any, dict[str, Any]]] = {}
        for target in self.targets:
            definition = get_target_definition(target)
            artifact = self._latest_artifact(definition.name)
            if artifact is None:
                continue
            model_path, metadata = artifact
            loaded[definition.name] = (definition, self._load_model(model_path, metadata), metadata)
        if not loaded:
            raise FileNotFoundError(f"No target model artifacts found in {self.model_dir}")

        payloads: list[dict[str, Any]] = []
        generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        metadata_by_target = {target: data[2] for target, data in loaded.items()}
        for _, row in rows.iterrows():
            feature_values = row[self.feature_names].apply(pd.to_numeric, errors="coerce").astype(float)
            feature_frame = pd.DataFrame([feature_values.to_dict()], columns=self.feature_names)
            forecast: dict[str, Any] = {}
            for target, (definition, model, metadata) in loaded.items():
                forecast[target] = self._predict_target(definition, model, metadata, feature_frame)

            completeness = self._feature_completeness(row, self.feature_names)
            payload = {
                "match_id": str(row["match_id"]),
                "date": self._format_date(row["date"]),
                "league": str(row["league"]),
                "home_team": str(row["home_team"]),
                "away_team": str(row["away_team"]),
                "forecast": forecast,
                "explainability": {
                    "top_features": self._top_features(row, metadata_by_target),
                },
                "diagnostics": {
                    "model_version": "forecast_suite_v1",
                    "target_versions": {
                        target: {
                            "artifact": metadata.get("artifact_name"),
                            "created_at": metadata.get("created_at"),
                            "model_type": metadata.get("model_type"),
                        }
                        for target, metadata in metadata_by_target.items()
                    },
                    "feature_completeness": completeness,
                    "cold_start_risk": self._cold_start_risk(row, self.feature_names, completeness),
                    "generated_at": generated_at,
                },
            }
            validate_forecast_payload(payload)
            payloads.append(payload)
        return payloads
