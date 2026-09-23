"""Leak-free real-edge gate for result_3way/btts/total_goals promotions (US#205).

Wires scripts.validate_real_edge.compute_real_edge into ModelSelector: before
a candidate for one of these three targets is promoted, retrain it train-only
(ModelManager.prepare_training_data + model.train(), never a
refit_on_full_data artifact -- see validate_real_edge.py's own leakage
warning, US#203) and check whether its qualifying-bet predictions show a real
edge against actual outcomes, not just a better offline metric. This is the
automated version of the manual check documents/systematic_practices.md
calls A1 -- US#172 shipped a result_3way retrain that passed the offline
gate and then measurably worsened live draw predictions (US#173); nothing in
the pipeline caught it before this.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
from scipy.stats import poisson

from src.features.feature_factory import remove_margin
from src.models.base_model import FPAIBaseModel, XGBoostModel, XGBoostRegressorModel
from src.models.ensemble_result_model import EnsembleResultModel
from src.models.goal_stacker import GoalStackerModel
from src.models.quantile_interval_model import QuantileIntervalModel
from src.models.two_stage_result_model import TwoStageResultModel
from src.models.model_factory import ModelFactory
from src.models.model_manager import ModelManager, _classes_for_calibration, _compute_sample_weight
from src.utils.db_manager import DuckDBManager
from src.utils.logger import get_logger
from scripts.validate_real_edge import compute_real_edge

LOGGER = get_logger(__name__)

REAL_EDGE_GATED_TARGETS = frozenset({"result_3way", "btts", "total_goals"})

# Below this many qualifying bets, a pooled real-edge number is noise, not
# signal -- neither pass nor fail on it.
MIN_QUALIFYING_BETS = 15
# A small negative tolerance: qualifying bets that land exactly on the
# implied rate (real_edge == 0) aren't a red flag, only a real, sustained
# shortfall is.
MIN_REAL_EDGE = -0.03

ODDSPAPI_BTTS_PATH = Path("data/oddspapi_btts_corners_odds.json")

_EVAL_SET_MODEL_TYPES = (
    XGBoostModel, XGBoostRegressorModel, GoalStackerModel,
    TwoStageResultModel, QuantileIntervalModel, EnsembleResultModel,
)

# ponytail: retrains the candidate's architecture (same model_type) with that
# type's default hyperparameters, not the exact swept params of the winning
# MLflow run -- faithfully replaying arbitrary logged params would need type
# coercion (MLflow stringifies every param) plus a per-model-type allowlist
# to avoid passing stray non-hyperparam keys into the constructor. This
# gate's job is catching systematic prediction-vs-market bias (US#172/192/193's
# failure mode), which shows up at the architecture level, not from small
# hyperparameter deltas -- US#203's actual differentiator (time_decay) is a
# ModelManager-level knob, not something sweep_runner even sweeps today.
# Upgrade path: thread run["params"] through ModelFactory.get_model() with
# type coercion if a candidate's real edge later proves param-sensitive.


@dataclass
class RealEdgeResult:
    status: str  # "pass" | "fail" | "inconclusive"
    detail: str
    pooled_real_edge: float | None
    n_qualifying: int
    # Per-selection detail (e.g. "home"/"draw"/"away" or "over_2.5"/"under_2.5"),
    # keyed exactly as passed to _pool_and_grade's `labels`. The gate itself only
    # acts on the pooled verdict above, but a specific selection can have a real
    # edge problem the pool average hides (e.g. total_goals/under_2.5 failing
    # while over_2.5 offsets it) -- this is what a re-validation for one named
    # cell (as opposed to "should this promotion go through") should read.
    by_selection: dict[str, dict[str, float | int]] | None = None


def _pool_and_grade(edges: list[dict[str, float | int]], labels: list[str]) -> RealEdgeResult:
    by_selection = dict(zip(labels, edges))
    n_total = sum(int(e["n_qualifying"]) for e in edges)
    if n_total < MIN_QUALIFYING_BETS:
        return RealEdgeResult(
            "inconclusive",
            f"only {n_total} qualifying bets in the leak-free test split (need >= {MIN_QUALIFYING_BETS})",
            None, n_total, by_selection,
        )
    weighted = sum(e["real_edge"] * e["n_qualifying"] for e in edges if e["n_qualifying"] > 0)
    pooled = weighted / n_total
    if pooled < MIN_REAL_EDGE:
        return RealEdgeResult(
            "fail",
            f"pooled real edge {pooled:+.1%} on {n_total} qualifying bets is below the {MIN_REAL_EDGE:+.1%} floor",
            pooled, n_total, by_selection,
        )
    return RealEdgeResult("pass", f"pooled real edge {pooled:+.1%} on {n_total} qualifying bets", pooled, n_total, by_selection)


def _train_only_predictions(mgr: ModelManager, model: FPAIBaseModel) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Train-only fit on X_train (tuned against X_val), evaluated on the
    genuinely-unseen X_test -- same non-leaky pattern as
    scripts/sweep_total_goals_recency.py's honest_predictions()."""
    X_train, X_val, X_test, y_train, y_val, y_test, test_meta = mgr.prepare_training_data()
    eval_set = [(X_val, y_val)] if isinstance(model, _EVAL_SET_MODEL_TYPES) else None
    sample_weight = _compute_sample_weight(y_train, mgr.target_definition.task_type, alpha=mgr.sample_weight_alpha)
    # Found live (2026-09-22) while sweeping time_decay_half_life_days for I1
    # btts: every half-life produced byte-identical results, because this was
    # missing -- ModelManager.train()'s own equivalent line
    # (model_manager.py:804) applies the decay; a caller building the fit by
    # hand (mirroring ModelManager.train() rather than calling it, so a 3-way
    # candidate's full predict_proba() stays reachable) has to opt in too.
    sample_weight = mgr._combine_time_decay(sample_weight)
    model.train(X_train, y_train, eval_set=eval_set, sample_weight=sample_weight)
    return X_test, y_test, test_meta


def _result_3way_edge(model: FPAIBaseModel, X_test: pd.DataFrame, y_test: pd.Series, test_meta: pd.DataFrame, db: DuckDBManager) -> RealEdgeResult:
    classes = _classes_for_calibration(model)
    if classes is None:
        return RealEdgeResult("inconclusive", "candidate model has no classes_ attribute", None, 0)

    match_ids = test_meta["match_id"].tolist()
    if not match_ids:
        return RealEdgeResult("inconclusive", "no rows in the leak-free test split", None, 0)
    placeholders = ",".join("?" * len(match_ids))
    with db.connection(read_only=True) as conn:
        odds = conn.execute(
            f"SELECT match_id, odds_h, odds_d, odds_a FROM raw_matches WHERE match_id IN ({placeholders})",
            match_ids,
        ).fetchdf()

    probs = model.predict_proba(X_test)
    df = pd.DataFrame(probs, columns=list(classes))
    df["match_id"] = test_meta["match_id"].to_numpy()
    df["actual_home"] = (y_test.to_numpy() == "home").astype(int)
    df["actual_draw"] = (y_test.to_numpy() == "draw").astype(int)
    df["actual_away"] = (y_test.to_numpy() == "away").astype(int)
    real = df.merge(odds, on="match_id", how="inner").dropna(subset=["odds_h", "odds_d", "odds_a"]).reset_index(drop=True)
    if real.empty:
        return RealEdgeResult("inconclusive", "no rows with real result_3way odds in the test split", None, 0)

    implied = remove_margin(real["odds_h"], real["odds_d"], real["odds_a"])
    real["implied_home"] = implied["MKT_Home_Prob_Real"]
    real["implied_draw"] = implied["MKT_Draw_Prob_Real"]
    real["implied_away"] = implied["MKT_Away_Prob_Real"]

    edges = [
        compute_real_edge(real, "home", "implied_home", "actual_home"),
        compute_real_edge(real, "draw", "implied_draw", "actual_draw"),
        compute_real_edge(real, "away", "implied_away", "actual_away"),
    ]
    return _pool_and_grade(edges, ["home", "draw", "away"])


def _btts_edge(model: FPAIBaseModel, X_test: pd.DataFrame, y_test: pd.Series, test_meta: pd.DataFrame) -> RealEdgeResult:
    if not ODDSPAPI_BTTS_PATH.exists():
        return RealEdgeResult("inconclusive", f"no real btts odds file at {ODDSPAPI_BTTS_PATH}", None, 0)
    lookup = json.loads(ODDSPAPI_BTTS_PATH.read_text(encoding="utf-8"))

    probs = model.predict_proba(X_test)
    positive = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
    df = pd.DataFrame({
        "match_id": test_meta["match_id"].to_numpy(),
        "pred_yes": positive,
        "actual_yes": y_test.to_numpy().astype(int),
    })
    df["actual_no"] = 1 - df["actual_yes"]
    df["pred_no"] = 1 - df["pred_yes"]

    odds = df["match_id"].map(lambda mid: lookup.get(mid, {}).get("btts_odds"))
    df["odds_yes"] = odds.map(lambda o: o.get("yes") if o else None)
    df["odds_no"] = odds.map(lambda o: o.get("no") if o else None)
    real = df.dropna(subset=["odds_yes", "odds_no"]).reset_index(drop=True)
    if real.empty:
        return RealEdgeResult("inconclusive", "no rows with real btts odds in the test split", None, 0)

    raw_yes = 1.0 / real["odds_yes"]
    raw_no = 1.0 / real["odds_no"]
    vig = raw_yes + raw_no
    real["implied_yes"] = raw_yes / vig
    real["implied_no"] = raw_no / vig

    edges = [
        compute_real_edge(real, "pred_yes", "implied_yes", "actual_yes"),
        compute_real_edge(real, "pred_no", "implied_no", "actual_no"),
    ]
    return _pool_and_grade(edges, ["yes", "no"])


def _total_goals_edge(model: FPAIBaseModel, X_test: pd.DataFrame, y_test: pd.Series, test_meta: pd.DataFrame, db: DuckDBManager) -> RealEdgeResult:
    match_ids = test_meta["match_id"].tolist()
    if not match_ids:
        return RealEdgeResult("inconclusive", "no rows in the leak-free test split", None, 0)
    placeholders = ",".join("?" * len(match_ids))
    with db.connection(read_only=True) as conn:
        odds = conn.execute(
            f"SELECT match_id, over25_odds, under25_odds FROM raw_matches WHERE match_id IN ({placeholders})",
            match_ids,
        ).fetchdf()

    lam = model.predict(X_test)
    df = pd.DataFrame({
        "match_id": test_meta["match_id"].to_numpy(),
        "pred_over": 1 - poisson.cdf(2, lam),
        "pred_under": poisson.cdf(2, lam),
        "actual_over": (y_test.to_numpy() > 2.5).astype(int),
    })
    df["actual_under"] = 1 - df["actual_over"]
    real = df.merge(odds, on="match_id", how="inner").dropna(subset=["over25_odds", "under25_odds"]).reset_index(drop=True)
    if real.empty:
        return RealEdgeResult("inconclusive", "no rows with real total_goals odds in the test split", None, 0)

    raw_over = 1.0 / real["over25_odds"]
    raw_under = 1.0 / real["under25_odds"]
    vig = raw_over + raw_under
    real["implied_over"] = raw_over / vig
    real["implied_under"] = raw_under / vig

    edges = [
        compute_real_edge(real, "pred_over", "implied_over", "actual_over"),
        compute_real_edge(real, "pred_under", "implied_under", "actual_under"),
    ]
    return _pool_and_grade(edges, ["over_2.5", "under_2.5"])


def _resolve_model_type(model_type: str) -> str:
    """Some already-recorded model_selection.yaml entries (older runs,
    predating a naming-convention change, or hand-promoted) use a spelling
    like 'xgboostregressor' or 'randomforestregressor' -- no underscore --
    that doesn't match ModelFactory's registry keys exactly, even though the
    same architecture IS registered under a different spelling
    ('xgboost_regressor'). Normalize both sides (lowercase, strip
    underscores) and match on that instead of refusing to re-validate an
    already-live champion over a cosmetic naming drift. Returns the
    original string unchanged when no normalized match exists, so
    ModelFactory's own error still fires with the real, unresolved name."""
    if model_type in ModelFactory._REGISTRY:
        return model_type
    normalized = model_type.lower().replace("_", "")
    for key in ModelFactory._REGISTRY:
        if key.lower().replace("_", "") == normalized:
            return key
    return model_type


def check_real_edge(
    target_name: str,
    context: str,
    feature_subset: list[str] | None,
    model_type: str,
    config_path: str = "config.yaml",
    time_decay_half_life_days: float | None = None,
) -> RealEdgeResult:
    """Leak-free real-edge check for a promotion candidate. Only gates the
    three targets with a documented real-edge history (REAL_EDGE_GATED_TARGETS);
    anything else passes through untouched -- this is deliberately not a
    general-purpose gate, see US#205's own acceptance criteria.

    time_decay_half_life_days (added 2026-09-22, A96 investigation): several
    already-promoted btts models (E0/SP1/D1/F1, US#203) were hand-promoted
    WITH a non-default half-life -- sweep_runner.py doesn't wire this knob at
    all, so a freshly-trained select-best-models candidate never has one, but
    re-validating an already-live hand-promoted model needs to reproduce its
    actual training config, not silently fall back to the undecayed default
    and report on a materially different model than what's really serving."""
    if target_name not in REAL_EDGE_GATED_TARGETS:
        return RealEdgeResult("pass", "not a real-edge-gated target", None, 0)

    try:
        model = ModelFactory.get_model(_resolve_model_type(model_type))
    except ValueError as exc:
        return RealEdgeResult("inconclusive", f"can't rebuild model_type {model_type!r} for the gate: {exc}", None, 0)

    try:
        mgr = ModelManager(
            model=model,
            config_path=config_path,
            target_config={"target": target_name},
            feature_subset=feature_subset,
            context=context,
            competition_id=context,
            time_decay_half_life_days=time_decay_half_life_days,
        )
        X_test, y_test, test_meta = _train_only_predictions(mgr, model)
    except Exception as exc:  # noqa: BLE001 -- any data/training failure here should degrade to inconclusive, not crash promotion
        LOGGER.warning("Real-edge gate could not train a leak-free candidate for %s/%s: %s", target_name, context, exc)
        return RealEdgeResult("inconclusive", f"leak-free retrain failed: {exc}", None, 0)

    db = mgr.db_manager
    if target_name == "result_3way":
        return _result_3way_edge(model, X_test, y_test, test_meta, db)
    if target_name == "btts":
        return _btts_edge(model, X_test, y_test, test_meta)
    return _total_goals_edge(model, X_test, y_test, test_meta, db)
