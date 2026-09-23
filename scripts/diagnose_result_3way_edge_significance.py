"""Significance check for a result_3way real-edge gate failure (US#211).

check_real_edge's fixed -3pp floor (src/utils/real_edge_gate.py) has no
notion of sample size -- a genuinely well-calibrated, skillful model can dip
below it purely from small-sample noise on a qualifying-bet subset of a few
dozen. Before treating a "fail" as a real defect, this checks whether the
model has actual discriminative skill (AUC, decile calibration against its
own predictions) and whether the qualifying-bet shortfall is even
statistically distinguishable from zero (z-test against the gate's own
implied-probability baseline) -- exactly the two checks that found E0's
"fail" (-3.2%, home/away-driven) was noise around a genuinely skillful
model (AUC 0.69-0.71, clean deciles, z in [-1.15, -0.56], p >= 0.25), unlike
US#210's btts:no (near-random AUC, flat deciles -- a real defect).

Usage:
    python scripts/diagnose_result_3way_edge_significance.py --league E0
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/Users/tianqihuang/Documents/GitHub/FPAI")

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.stats import skellam
from sklearn.metrics import roc_auc_score

from src.features.feature_factory import remove_margin
from src.models.model_factory import ModelFactory
from src.models.model_manager import ModelManager
from src.utils.real_edge_gate import _resolve_model_type, _train_only_predictions

parser = argparse.ArgumentParser()
parser.add_argument("--league", required=True)
args = parser.parse_args()

with open("config/model_selection.yaml") as f:
    selection = yaml.safe_load(f)
entry = selection["contexts"][args.league]["result_3way"]

model = ModelFactory.get_model(_resolve_model_type(entry["model_type"]))
mgr = ModelManager(
    model=model, target_config={"target": "result_3way"}, feature_subset=entry.get("feature_subset"),
    context=args.league, competition_id=args.league,
)
X_test, y_test, test_meta = _train_only_predictions(mgr, model)

if not hasattr(model, "home_model"):
    print(f"{entry['model_type']} isn't Skellam-based -- this script's mu_home/mu_away path doesn't apply.")
    sys.exit(1)

mu_home = np.clip(model.home_model.predict(X_test[model._home_feature_columns]), 0.02, None)
mu_away = np.clip(model.away_model.predict(X_test[model._away_feature_columns]), 0.02, None)
p_home, p_away, p_draw = skellam.sf(0, mu_home, mu_away), skellam.cdf(-1, mu_home, mu_away), skellam.pmf(0, mu_home, mu_away)
tot = p_home + p_away + p_draw

db = mgr.db_manager
match_ids = test_meta["match_id"].tolist()
placeholders = ",".join("?" * len(match_ids))
with db.connection(read_only=True) as conn:
    odds = conn.execute(
        f"SELECT match_id, odds_h, odds_d, odds_a FROM raw_matches WHERE match_id IN ({placeholders})", match_ids,
    ).fetchdf()

df = pd.DataFrame({
    "match_id": test_meta["match_id"].to_numpy(),
    "pred_home": p_home / tot, "pred_away": p_away / tot, "pred_draw": p_draw / tot,
    "result": y_test.to_numpy(),
}).merge(odds, on="match_id", how="inner")
implied = remove_margin(df["odds_h"], df["odds_d"], df["odds_a"])
df["implied_home"], df["implied_away"] = implied["MKT_Home_Prob_Real"], implied["MKT_Away_Prob_Real"]
df["actual_home"] = (df.result == "home").astype(int)
df["actual_away"] = (df.result == "away").astype(int)

print(f"=== {args.league} result_3way ({entry['model_type']}), n={len(df)} ===\n")
for label, pcol, acol, icol in [("home", "pred_home", "actual_home", "implied_home"), ("away", "pred_away", "actual_away", "implied_away")]:
    auc = roc_auc_score(df[acol], df[pcol])
    deciles = pd.qcut(df[pcol], 5, duplicates="drop")
    means = df.groupby(deciles, observed=True)[acol].mean().to_numpy()
    print(f"{label}: AUC={auc:.3f}  decile hit-rates={[round(m, 2) for m in means]}")

    qual = df[df[pcol] - df[icol] >= 0.05]
    n = len(qual)
    if n < 15:
        print(f"  qualifying n={n} -- too thin to test\n")
        continue
    actual_rate, implied_mean = qual[acol].mean(), qual[icol].mean()
    edge = actual_rate - implied_mean
    se = np.sqrt(implied_mean * (1 - implied_mean) / n)
    z = edge / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    verdict = "noise (not significant)" if p >= 0.05 else "SIGNIFICANT -- investigate further"
    print(f"  qualifying n={n}: real_edge={edge:+.1%}, SE={se:.1%}, z={z:.2f}, p={p:.3f} -> {verdict}\n")
