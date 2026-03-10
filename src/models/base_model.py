"""Base and concrete model implementations for FPAI prediction tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


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


class XGBoostModel(FPAIBaseModel):
    """XGBoost classifier wrapper with early stopping and native serialization."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize XGBClassifier with robust defaults for binary home-win prediction."""
        default_kwargs = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        default_kwargs.update(kwargs)
        self.model = XGBClassifier(**default_kwargs)

    def train(self, X: Any, y: Any) -> None:
        """Train model with a 10% validation split and early stopping when feasible."""
        X_np = np.asarray(X)
        y_np = np.asarray(y)

        # Early stopping requires enough data and both classes in train/validation sets.
        can_split = len(y_np) >= 20 and len(np.unique(y_np)) > 1
        if can_split:
            X_train, X_val, y_train, y_val = train_test_split(
                X_np,
                y_np,
                test_size=0.1,
                random_state=42,
                stratify=y_np,
            )
            self.model.set_params(early_stopping_rounds=20)
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            return

        self.model.fit(X_np, y_np, verbose=False)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return positive-class probability (home win) as a 1D array."""
        probabilities = self.model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()

    def save(self, path: str) -> None:
        """Persist the model using native XGBoost serialization."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(target_path))

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        """Load model from native XGBoost serialized artifact."""
        instance = cls()
        instance.model.load_model(path)
        return instance


class RandomForestModel(FPAIBaseModel):
    """Random Forest classifier wrapper with joblib serialization."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize RandomForestClassifier with optional sklearn kwargs."""
        default_kwargs = {"n_estimators": 200, "random_state": 42, "n_jobs": -1}
        default_kwargs.update(kwargs)
        self.model = RandomForestClassifier(**default_kwargs)

    def train(self, X: Any, y: Any) -> None:
        """Train the random forest model."""
        self.model.fit(X, y)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return predicted class probabilities."""
        probabilities = self.model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()

    def save(self, path: str) -> None:
        """Serialize the model to disk using joblib."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, target_path)

    @classmethod
    def load(cls, path: str) -> "RandomForestModel":
        """Load a serialized random forest model from disk."""
        instance = cls()
        instance.model = joblib.load(path)
        return instance
