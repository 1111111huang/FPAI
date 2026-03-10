"""Model factory for constructing FPAI models by type."""

from __future__ import annotations

from typing import Any

from src.models.base_model import FPAIBaseModel, LRModel, RandomForestModel, XGBoostModel


class ModelFactory:
    """Factory for FPAIBaseModel implementations."""

    _REGISTRY = {
        "xgboost": XGBoostModel,
        "logistic_regression": LRModel,
        "random_forest": RandomForestModel,
    }

    @staticmethod
    def get_model(model_type: str, params: dict[str, Any] | None = None) -> FPAIBaseModel:
        """Return a model instance given a type string and optional params."""
        normalized = model_type.strip().lower()
        model_cls = ModelFactory._REGISTRY.get(normalized)
        if model_cls is None:
            valid = ", ".join(sorted(ModelFactory._REGISTRY.keys()))
            raise ValueError(f"Unsupported model_type '{model_type}'. Supported types: {valid}")
        return model_cls(**(params or {}))
