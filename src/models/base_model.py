"""Base and concrete model implementations for FPAI prediction tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class FPAIBaseModel(ABC):
    """Abstract base class for all FPAI prediction models."""

    @abstractmethod
    def train(self, X: Any, y: Any) -> None:
        """Fit the model using feature matrix X and target y."""

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for input features X."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model state to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "FPAIBaseModel":
        """Load model state from disk and return an initialized model instance."""


class LRModel(FPAIBaseModel):
    """Logistic Regression model wrapper with joblib serialization."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize LogisticRegression with optional sklearn kwargs."""
        default_kwargs = {"max_iter": 1000}
        default_kwargs.update(kwargs)
        self.model = LogisticRegression(**default_kwargs)

    def train(self, X: Any, y: Any) -> None:
        """Train the logistic regression model."""
        self.model.fit(X, y)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return predicted class probabilities."""
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        """Serialize the model to disk using joblib."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, target_path)

    @classmethod
    def load(cls, path: str) -> "LRModel":
        """Load a serialized logistic regression model from disk."""
        instance = cls()
        instance.model = joblib.load(path)
        return instance
