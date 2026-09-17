"""Leak-free recency-weighting sweep for total_goals, all 5 leagues.

Regression target -- ModelManager.train() can't be reused as-is (it calls
predict_proba, which XGBoostRegressorModel doesn't support), so this
replicates train()'s train-only-fit logic manually via prepare_training_data()
+ model.train()/predict(), same non-leaky principle as US#203's corrected
btts check: evaluate the TRAIN-ONLY fit on genuinely-unseen X_test, never a
refit_on_full_data artifact.

Real market: over25_odds/under25_odds are real columns on raw_matches
itself (football-data.co.uk), not the oddspapi-limited window -- so no
date-window bottleneck the way btts had.
"""
import sys
sys.path.insert(0, "/Users/tianqihuang/Documents/GitHub/FPAI")
import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.models.model_manager import ModelManager, _compute_sample_weight
from src.models.base_model import XGBoostRegressorModel
from src.logic.competition_registry import get_competition_definition, resolve_feature_subset_for_tier
from src.utils.db_manager import DuckDBManager
from scripts.validate_real_edge import compute_real_edge

HALF_LIVES = [None, 180, 365, 730]
db = DuckDBManager("/Users/tianqihuang/Documents/GitHub/FPAI/config.yaml")


def pooled(r_over, r_under):
    n_o, n_u = r_over["n_qualifying"], r_under["n_qualifying"]
    tot = n_o + n_u
    if tot == 0:
        return float("nan")
    e_o = 0.0 if n_o == 0 else r_over["real_edge"] * n_o
    e_u = 0.0 if n_u == 0 else r_under["real_edge"] * n_u
    return (e_o + e_u) / tot


def honest_predictions(league, feature_subset, hl):
    """Train-only fit (X_train, with time-decay by X_train's own dates),
    predict on X_test -- genuinely unseen, no refit_on_full_data."""
    model = XGBoostRegressorModel()
    mgr = ModelManager(
        model=model, target_config={"target": "total_goals"}, feature_subset=feature_subset,
        context=league, competition_id=league, time_decay_half_life_days=hl,
    )
    selected_features = mgr._load_selected_features()
    X_train, X_val, X_test, y_train, y_val, y_test, test_meta = mgr.prepare_training_data()
    sample_weight = _compute_sample_weight(y_train, mgr.target_definition.task_type, alpha=mgr.sample_weight_alpha)
    sample_weight = mgr._combine_time_decay(sample_weight)
    model.train(X_train, y_train, eval_set=[(X_val, y_val)], sample_weight=sample_weight)
    lam = model.predict(X_test)
    df = test_meta.copy()
    df["actual_total"] = y_test.values
    df["lam"] = lam
    return df


for league in ["SP1", "E0", "D1", "I1", "F1"]:
    print(f"\n{'='*20} {league} total_goals (leak-free) {'='*20}")
    comp = get_competition_definition(league)
    feature_subset = resolve_feature_subset_for_tier(comp.tier)

    preds = {}
    for hl in HALF_LIVES:
        df = honest_predictions(league, feature_subset, hl)
        match_ids = df["match_id"].tolist()
        placeholders = ",".join("?" * len(match_ids))
        with db.connection(read_only=True) as conn:
            odds = conn.execute(
                f"SELECT match_id, over25_odds, under25_odds FROM raw_matches WHERE match_id IN ({placeholders})",
                match_ids,
            ).fetchdf()
        df = df.merge(odds, on="match_id", how="left")
        real = df[df["over25_odds"].notna() & df["under25_odds"].notna()].reset_index(drop=True)
        raw_over = 1 / real["over25_odds"]
        raw_under = 1 / real["under25_odds"]
        vig_sum = raw_over + raw_under
        real["implied_over"], real["implied_under"] = raw_over / vig_sum, raw_under / vig_sum
        real["actual_over"] = (real["actual_total"] > 2.5).astype(int)
        real["actual_under"] = 1 - real["actual_over"]
        real["pred_over"] = 1 - poisson.cdf(2, real["lam"])
        real["pred_under"] = poisson.cdf(2, real["lam"])
        preds[hl] = real

    n = len(preds[None])
    print(f"  n_real_odds_in_Xtest={n}")
    if n < 20:
        print("  SKIP: too few real-odds rows")
        continue
    mid = n // 2

    best_hl, best_p = None, -999
    for hl in HALF_LIVES:
        sweep = preds[hl].iloc[:mid]
        r_over = compute_real_edge(sweep, "pred_over", "implied_over", "actual_over")
        r_under = compute_real_edge(sweep, "pred_under", "implied_under", "actual_under")
        p = pooled(r_over, r_under)
        print(f"  SWEEP hl={hl}: pooled={p:.4f}  over_n={r_over['n_qualifying']} over_edge={r_over['real_edge']}  "
              f"under_n={r_under['n_qualifying']} under_edge={r_under['real_edge']}")
        if not np.isnan(p) and p > best_p:
            best_p, best_hl = p, hl

    confirm_base = preds[None].iloc[mid:]
    confirm_chosen = preds[best_hl].iloc[mid:]
    r_over_b = compute_real_edge(confirm_base, "pred_over", "implied_over", "actual_over")
    r_under_b = compute_real_edge(confirm_base, "pred_under", "implied_under", "actual_under")
    p_base = pooled(r_over_b, r_under_b)
    r_over_c = compute_real_edge(confirm_chosen, "pred_over", "implied_over", "actual_over")
    r_under_c = compute_real_edge(confirm_chosen, "pred_under", "implied_under", "actual_under")
    p_chosen = pooled(r_over_c, r_under_c)

    print(f"  chosen on sweep: hl={best_hl}")
    print(f"  CONFIRM baseline (hl=None): pooled={p_base:.4f}  over={r_over_b}  under={r_under_b}")
    print(f"  CONFIRM chosen  (hl={best_hl}): pooled={p_chosen:.4f}  over={r_over_c}  under={r_under_c}")
    win = (not np.isnan(p_chosen)) and p_chosen > p_base
    print(f"  RESULT {league}: {'WIN' if win else 'NO IMPROVEMENT'} ({p_base:.4f} -> {p_chosen:.4f})")
