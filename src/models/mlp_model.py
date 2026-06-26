"""Fully connected network models for FPAI (US#74).

Wraps sklearn MLPClassifier/MLPRegressor with:
  - StandardScaler (fit on train only, applied to all splits)
  - Manual partial_fit epoch loop for early stopping
  - Optuna ASHA pruning support via set_optuna_trial()
  - LabelEncoder for multiclass targets

Architecture params depth+hidden_size are split for Optuna search, then
assembled into sklearn's hidden_layer_sizes tuple at train time.
"""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.models.base_model import FPAIBaseModel
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_CLASSIFIER_DEFAULTS: dict[str, Any] = {
    "depth": 3,
    "hidden_size": 128,
    "activation": "relu",
    "alpha": 1e-4,
    "learning_rate_init": 1e-3,
    "batch_size": 64,
    "max_iter": 100,
    "patience": 15,
}

_REGRESSOR_DEFAULTS: dict[str, Any] = {
    **_CLASSIFIER_DEFAULTS,
}


def _build_hidden_layer_sizes(depth: int, hidden_size: int) -> tuple[int, ...]:
    return tuple([hidden_size] * depth)


class MLPModel(FPAIBaseModel):
    """Fully connected classifier (binary and multiclass).

    Hyperparameters used in Optuna search:
      depth           int  2–5      number of hidden layers
      hidden_size     int  64–512   neurons per layer (uniform width)
      activation      cat  relu/tanh
      alpha           float         L2 weight penalty
      learning_rate_init float      initial learning rate
      batch_size      int           mini-batch size
      max_iter        int           total epochs (low in stage-1, full in stage-2)
    """

    def __init__(
        self,
        depth: int = _CLASSIFIER_DEFAULTS["depth"],
        hidden_size: int = _CLASSIFIER_DEFAULTS["hidden_size"],
        activation: str = _CLASSIFIER_DEFAULTS["activation"],
        alpha: float = _CLASSIFIER_DEFAULTS["alpha"],
        learning_rate_init: float = _CLASSIFIER_DEFAULTS["learning_rate_init"],
        batch_size: int = _CLASSIFIER_DEFAULTS["batch_size"],
        max_iter: int = _CLASSIFIER_DEFAULTS["max_iter"],
        patience: int = _CLASSIFIER_DEFAULTS["patience"],
        **kwargs: Any,
    ) -> None:
        self._depth = int(depth)
        self._hidden_size = int(hidden_size)
        self._activation = str(activation)
        self._alpha = float(alpha)
        self._lr_init = float(learning_rate_init)
        self._batch_size = int(batch_size)
        self._max_iter = int(max_iter)
        self._patience = int(patience)

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.classes_: np.ndarray | None = None
        self.model: MLPClassifier | None = None
        self._trial: Any = None  # optuna.Trial injected by StagedOptunaRunner

    def set_optuna_trial(self, trial: Any) -> None:
        """Inject an Optuna trial for ASHA pruning callbacks."""
        self._trial = trial

    def train(self, X: Any, y: Any, eval_set: Any | None = None) -> None:
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_arr = np.asarray(y)

        X_scaled = self.scaler.fit_transform(X_df)
        y_enc = self.label_encoder.fit_transform(y_arr)
        self.classes_ = self.label_encoder.classes_

        all_classes = np.unique(y_enc)
        hidden = _build_hidden_layer_sizes(self._depth, self._hidden_size)

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=self._activation,
            alpha=self._alpha,
            learning_rate_init=self._lr_init,
            batch_size=min(self._batch_size, len(X_scaled)),
            max_iter=1,         # we drive the loop manually
            warm_start=True,    # accumulate across partial_fit calls
            random_state=42,
            solver="adam",
        )

        X_val_scaled: np.ndarray | None = None
        y_val_enc: np.ndarray | None = None
        if eval_set is not None:
            X_v, y_v = eval_set[0]
            X_val_scaled = self.scaler.transform(pd.DataFrame(X_v) if not isinstance(X_v, pd.DataFrame) else X_v)
            y_val_enc = self.label_encoder.transform(np.asarray(y_v))

        best_val_loss = float("inf")
        no_improve = 0

        for epoch in range(self._max_iter):
            self.model.partial_fit(X_scaled, y_enc, classes=all_classes)

            if X_val_scaled is not None:
                val_proba = self.model.predict_proba(X_val_scaled)
                val_proba = np.clip(val_proba, 1e-12, 1.0)
                # cross-entropy
                n = len(y_val_enc)
                val_loss = float(-np.sum(np.log(val_proba[np.arange(n), y_val_enc])) / n)

                if self._trial is not None:
                    self._trial.report(val_loss, epoch)
                    if self._trial.should_prune():
                        LOGGER.debug("ASHA pruned at epoch %d", epoch)
                        raise __import__("optuna").exceptions.TrialPruned()

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self._patience:
                    LOGGER.debug("MLPModel early stopping at epoch %d", epoch)
                    break

        LOGGER.info("MLPModel trained: depth=%d width=%d epochs=%d", self._depth, self._hidden_size, epoch + 1)

    def predict_proba(self, X: Any) -> np.ndarray:
        assert self.model is not None, "Model not trained."
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(self.scaler.transform(X_df))

    def predict(self, X: Any) -> np.ndarray:
        assert self.model is not None, "Model not trained."
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        encoded = self.model.predict(self.scaler.transform(X_df))
        return self.label_encoder.inverse_transform(encoded)

    def save(self, path: str) -> None:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "scaler": self.scaler,
             "label_encoder": self.label_encoder, "classes_": self.classes_},
            str(target_path),
        )

    @classmethod
    def load(cls, path: str) -> "MLPModel":
        payload = joblib.load(str(path))
        obj = cls()
        obj.model = payload["model"]
        obj.scaler = payload["scaler"]
        obj.label_encoder = payload["label_encoder"]
        obj.classes_ = payload["classes_"]
        return obj


class MLPRegressorModel(FPAIBaseModel):
    """Fully connected regressor for count/continuous targets."""

    def __init__(
        self,
        depth: int = _REGRESSOR_DEFAULTS["depth"],
        hidden_size: int = _REGRESSOR_DEFAULTS["hidden_size"],
        activation: str = _REGRESSOR_DEFAULTS["activation"],
        alpha: float = _REGRESSOR_DEFAULTS["alpha"],
        learning_rate_init: float = _REGRESSOR_DEFAULTS["learning_rate_init"],
        batch_size: int = _REGRESSOR_DEFAULTS["batch_size"],
        max_iter: int = _REGRESSOR_DEFAULTS["max_iter"],
        patience: int = _REGRESSOR_DEFAULTS["patience"],
        **kwargs: Any,
    ) -> None:
        self._depth = int(depth)
        self._hidden_size = int(hidden_size)
        self._activation = str(activation)
        self._alpha = float(alpha)
        self._lr_init = float(learning_rate_init)
        self._batch_size = int(batch_size)
        self._max_iter = int(max_iter)
        self._patience = int(patience)

        self.scaler = StandardScaler()
        self.model: MLPRegressor | None = None
        self._trial: Any = None

    def set_optuna_trial(self, trial: Any) -> None:
        self._trial = trial

    def train(self, X: Any, y: Any, eval_set: Any | None = None) -> None:
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_arr = np.asarray(y, dtype=float)

        X_scaled = self.scaler.fit_transform(X_df)
        hidden = _build_hidden_layer_sizes(self._depth, self._hidden_size)

        self.model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation=self._activation,
            alpha=self._alpha,
            learning_rate_init=self._lr_init,
            batch_size=min(self._batch_size, len(X_scaled)),
            max_iter=1,
            warm_start=True,
            random_state=42,
            solver="adam",
        )

        X_val_scaled: np.ndarray | None = None
        y_val: np.ndarray | None = None
        if eval_set is not None:
            X_v, y_v = eval_set[0]
            X_val_scaled = self.scaler.transform(pd.DataFrame(X_v) if not isinstance(X_v, pd.DataFrame) else X_v)
            y_val = np.asarray(y_v, dtype=float)

        best_val_mae = float("inf")
        no_improve = 0

        for epoch in range(self._max_iter):
            self.model.partial_fit(X_scaled, y_arr)

            if X_val_scaled is not None:
                val_pred = self.model.predict(X_val_scaled)
                val_mae = float(np.mean(np.abs(y_val - val_pred)))

                if self._trial is not None:
                    self._trial.report(val_mae, epoch)
                    if self._trial.should_prune():
                        LOGGER.debug("ASHA pruned at epoch %d", epoch)
                        raise __import__("optuna").exceptions.TrialPruned()

                if val_mae < best_val_mae - 1e-5:
                    best_val_mae = val_mae
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self._patience:
                    LOGGER.debug("MLPRegressorModel early stopping at epoch %d", epoch)
                    break

        LOGGER.info("MLPRegressorModel trained: depth=%d width=%d epochs=%d", self._depth, self._hidden_size, epoch + 1)

    def predict_proba(self, X: Any) -> np.ndarray:
        raise TypeError("MLPRegressorModel does not support predict_proba.")

    def predict(self, X: Any) -> np.ndarray:
        assert self.model is not None, "Model not trained."
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.model.predict(self.scaler.transform(X_df))

    def save(self, path: str) -> None:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, str(target_path))

    @classmethod
    def load(cls, path: str) -> "MLPRegressorModel":
        payload = joblib.load(str(path))
        obj = cls()
        obj.model = payload["model"]
        obj.scaler = payload["scaler"]
        return obj
