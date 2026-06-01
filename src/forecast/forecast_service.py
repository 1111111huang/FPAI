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
from src.logic.target_registry import TargetDefinition, get_target_definition, list_target_definitions
from src.utils.config_loader import AppSettings, load_settings
from src.utils.db_manager import DuckDBManager


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
        if metadata.get("model_type") == "XGBoostModel":
            model = XGBClassifier()
            model.load_model(str(model_path))
            return model
        if metadata.get("model_type") == "XGBoostRegressorModel":
            model = XGBRegressor()
            model.load_model(str(model_path))
            return model
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
