"""Tests for TwoStageResultModel (US#181) -- a draw-vs-decisive decomposition
for result_3way, tried as a genuinely different mechanism after US#172/173
found repeated sample-weight retuning of a single joint multiclass softmax
model couldn't fix E0/SP1's draw over-prediction on lopsided matchups.

Architecture: P(draw) from a dedicated binary draw-vs-not model (trained on
every row), P(home|not draw) from a second binary model trained ONLY on
decisive (non-draw) rows -- so its home/away signal is never diluted by
draw-balancing. Combined: P(home) = (1-P(draw))*P(home|decisive),
P(away) = (1-P(draw))*(1-P(home|decisive)).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.models.model_manager import ModelManager
from src.models.two_stage_result_model import TwoStageResultModel


def _make_matches(n: int = 300, seed: int = 42, draw_rate: float = 0.25) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic result_3way data: a 'strength_diff' feature that genuinely
    separates home/away, plus label noise, at a controlled draw rate."""
    rng = np.random.default_rng(seed)
    strength_diff = rng.normal(0, 1, size=n)
    is_draw = rng.random(n) < draw_rate
    # Decisive rows: home wins when strength_diff > 0, with some noise.
    home_wins = (strength_diff + rng.normal(0, 0.5, size=n)) > 0
    labels = np.where(is_draw, "draw", np.where(home_wins, "home", "away"))
    X = pd.DataFrame({
        "strength_diff": strength_diff,
        "noise": rng.normal(0, 1, size=n),
    })
    return X, pd.Series(labels)


class TestTwoStageResultModelTrain:
    def test_classes_are_alphabetical_matching_xgboost_convention(self):
        model = TwoStageResultModel()
        X, y = _make_matches(200)
        model.train(X, y)
        assert list(model.classes_) == ["away", "draw", "home"]

    def test_predict_proba_shape_and_sums_to_one(self):
        model = TwoStageResultModel()
        X, y = _make_matches(200)
        model.train(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_returns_only_known_classes(self):
        model = TwoStageResultModel()
        X, y = _make_matches(200)
        model.train(X, y)
        preds = model.predict(X)
        assert set(preds) <= {"home", "draw", "away"}

    def test_decisive_submodel_never_trained_on_draws(self):
        """The decisive (home-vs-away) sub-model's own training labels must
        never include a draw row -- the whole point of the architecture."""
        model = TwoStageResultModel()
        X, y = _make_matches(200)
        model.train(X, y)
        # decisive_model is a binary XGBClassifier; its own classes_ must be
        # exactly {0, 1} (home/away), never a third value.
        assert set(model.decisive_model.classes_) == {0, 1}

    def test_strong_favourite_gets_low_draw_probability(self):
        """Core motivation check: a match with a large, clear strength
        advantage should NOT have draw as its highest-probability outcome --
        the exact failure mode US#172/173 documented for the joint model."""
        model = TwoStageResultModel()
        X, y = _make_matches(600, seed=7)
        model.train(X, y)
        extreme_favourite = pd.DataFrame({"strength_diff": [4.0], "noise": [0.0]})
        proba = model.predict_proba(extreme_favourite)[0]
        classes = list(model.classes_)
        p_draw = proba[classes.index("draw")]
        p_home = proba[classes.index("home")]
        assert p_home > p_draw, f"draw ({p_draw}) should not dominate a clear favourite (home={p_home})"


class TestTwoStageResultModelPersistence:
    def test_save_load_roundtrip_predictions_match(self):
        model = TwoStageResultModel()
        X, y = _make_matches(200)
        model.train(X, y)
        proba_before = model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.joblib")
            model.save(path)
            loaded = TwoStageResultModel.load(path)
            proba_after = loaded.predict_proba(X)

        assert np.allclose(proba_before, proba_after)
        assert list(loaded.classes_) == ["away", "draw", "home"]


class TestTwoStageResultModelPrepareTrainingData:
    """Regression test for the NaN-tolerance gap this model hit against the
    real DB: `ModelManager.prepare_training_data()` requires every selected
    feature to be non-null for any model that ISN'T an XGBoost*Model
    instance -- TwoStageResultModel is built entirely from XGBClassifier
    sub-models, so it needs the same NaN tolerance, not the strict dropna
    GoalStackerModel (mixes in sklearn estimators that can't handle NaN)
    correctly still needs."""

    def _make_manager(self, tmp_path: Path) -> ModelManager:
        db_path = tmp_path / "test.db"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
        schema_path = tmp_path / "config" / "schema.yaml"
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(
            yaml.safe_dump({"training_setup": {"selected_features": ["MKT_IMPLIED_HOME", "MKT_LAMBDA_HOME"]}}),
            encoding="utf-8",
        )
        with duckdb.connect(str(db_path)) as conn:
            conn.execute(
                "CREATE TABLE raw_matches (match_id TEXT PRIMARY KEY, date TIMESTAMP, "
                "fthg INTEGER, ftag INTEGER, odds_h FLOAT)"
            )
            conn.execute(
                "CREATE TABLE feature_store (match_id TEXT PRIMARY KEY, "
                "MKT_IMPLIED_HOME FLOAT, MKT_LAMBDA_HOME FLOAT)"
            )
            rows = [(f"m{i}", f"2024-01-{i+1:02d}", i % 3, (i + 1) % 3, 2.0) for i in range(20)]
            conn.executemany("INSERT INTO raw_matches VALUES (?, ?, ?, ?, ?)", rows)
            # MKT_LAMBDA_HOME NaN for every row (mirrors real cold-start MKT_
            # gaps) -- would wipe out every row under the strict dropna path.
            conn.executemany(
                "INSERT INTO feature_store VALUES (?, ?, NULL)",
                [(f"m{i}", 0.5) for i in range(20)],
            )
        return ModelManager(
            model=TwoStageResultModel(),
            config_path=str(config_path),
            target_config={"target": "result_3way"},
            competition_id="unregistered_test_context",
        )

    def test_two_stage_result_model_tolerates_nan_features(self, tmp_path: Path) -> None:
        manager = self._make_manager(tmp_path)
        X_train, X_val, X_test, y_train, y_val, y_test, _ = manager.prepare_training_data()
        assert len(X_train) + len(X_val) + len(X_test) == 20


class TestTwoStageResultModelEvalSet:
    def test_train_accepts_eval_set_without_error(self):
        """eval_set carries the ORIGINAL 3-class y_val -- the model must
        derive its own binary eval sets internally, not choke on 3 classes."""
        model = TwoStageResultModel()
        X, y = _make_matches(200, seed=1)
        X_val, y_val = _make_matches(60, seed=2)
        model.train(X, y, eval_set=[(X_val, y_val)])
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), 3)
