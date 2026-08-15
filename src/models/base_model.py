"""Base and concrete model implementations for FPAI prediction tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor


class FPAIBaseModel(ABC):
    """Abstract base class for all FPAI prediction models."""

    @abstractmethod
    def train(self, X: Any, y: Any, eval_set: Any | None = None, sample_weight: Any | None = None) -> None:
        """Fit the model using feature matrix X and target y."""

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for input features X."""

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Predict target values or classes for input features X."""

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

    def train(self, X: Any, y: Any, eval_set: Any | None = None, sample_weight: Any | None = None) -> None:
        """Train the logistic regression model."""
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return predicted class probabilities."""
        return self.model.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        """Return predicted classes."""
        return self.model.predict(X)

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
            "early_stopping_rounds": 50,
            "random_state": 42,
            "n_jobs": -1,
        }
        default_kwargs.update(kwargs)
        self.model = XGBClassifier(**default_kwargs)
        self.label_encoder: LabelEncoder | None = None
        self.classes_: np.ndarray | None = None

    def train(self, X: Any, y: Any, eval_set: Any | None = None, sample_weight: Any | None = None) -> None:
        """Train model with optional validation set for early stopping."""
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        encoded_eval_set = None
        if eval_set is not None:
            encoded_eval_set = [
                (eval_X, self.label_encoder.transform(eval_y))
                for eval_X, eval_y in eval_set
            ]
            self.model.fit(X, y_encoded, sample_weight=sample_weight, eval_set=encoded_eval_set, verbose=False)
            return
        self.model.fit(X, y_encoded, sample_weight=sample_weight, verbose=False)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return predicted class probabilities."""
        return self.model.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        """Return predicted classes."""
        predictions = self.model.predict(X)
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(predictions.astype(int))
        return predictions

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

    def train(self, X: Any, y: Any, eval_set: Any | None = None, sample_weight: Any | None = None) -> None:
        """Train the random forest model."""
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return predicted class probabilities."""
        return self.model.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        """Return predicted classes."""
        return self.model.predict(X)

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


class RandomForestRegressorModel(FPAIBaseModel):
    """Random Forest regressor wrapper for count/regression forecast targets."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize RandomForestRegressor with optional sklearn kwargs."""
        default_kwargs = {"n_estimators": 200, "random_state": 42, "n_jobs": -1}
        default_kwargs.update(kwargs)
        self.model = RandomForestRegressor(**default_kwargs)

    def train(self, X: Any, y: Any, eval_set: Any | None = None, sample_weight: Any | None = None) -> None:
        """Train the random forest regressor."""
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Regression models do not produce class probabilities."""
        raise TypeError("RandomForestRegressorModel does not support predict_proba.")

    def predict(self, X: Any) -> np.ndarray:
        """Return numeric predictions."""
        return self.model.predict(X)

    def save(self, path: str) -> None:
        """Serialize the model to disk using joblib."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, target_path)

    @classmethod
    def load(cls, path: str) -> "RandomForestRegressorModel":
        """Load a serialized random forest regressor from disk."""
        instance = cls()
        instance.model = joblib.load(path)
        return instance


class XGBoostRegressorModel(FPAIBaseModel):
    """XGBoost regressor wrapper for count/regression forecast targets."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize XGBRegressor with robust defaults for count targets."""
        default_kwargs = {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "early_stopping_rounds": 30,
            "random_state": 42,
            "n_jobs": -1,
        }
        default_kwargs.update(kwargs)
        self.model = XGBRegressor(**default_kwargs)

    def train(self, X: Any, y: Any, eval_set: Any | None = None, sample_weight: Any | None = None) -> None:
        """Train model with optional validation set for early stopping."""
        if eval_set is not None:
            self.model.fit(X, y, sample_weight=sample_weight, eval_set=eval_set, verbose=False)
            return
        self.model.fit(X, y, sample_weight=sample_weight, verbose=False)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Regression models do not produce class probabilities."""
        raise TypeError("XGBoostRegressorModel does not support predict_proba.")

    def predict(self, X: Any) -> np.ndarray:
        """Return numeric predictions."""
        return self.model.predict(X)

    def save(self, path: str) -> None:
        """Persist the model using native XGBoost serialization."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(target_path))

    @classmethod
    def load(cls, path: str) -> "XGBoostRegressorModel":
        """Load model from native XGBoost serialized artifact."""
        instance = cls()
        instance.model.load_model(path)
        return instance
