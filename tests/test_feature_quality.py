"""
Feature quality integration tests — run against the real DuckDB feature store.

NaN rate ceilings (assert < threshold):
  OFF_ / DEF_ / DIS_                    < 30 %  (rolling cold-start 3-4%; xG features add ~25%
                                                  NaN when understat data is absent — expected)
  CTX_                                  < 5  %  (context features; typically 1 %)
  MKT_                                  < 5  %  ← CURRENTLY FAILING at ~30.6 %; intentional
                                                  flag to surface missing odds data for
                                                  older seasons.
  STRENGTH_ / INTERACTION_ / EFFICIENCY_ < 15 %
  OPP_ADJ_                              < 8  %  (combined venue timeline; typically 2 %)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

DB_PATH = Path("data/fpai_core.db")
SCHEMA_PATH = Path("config/schema.yaml")


def _selected_features() -> list[str]:
    with open(SCHEMA_PATH) as fh:
        return yaml.safe_load(fh)["training_setup"]["selected_features"]


@pytest.fixture(scope="module")
def real_db():
    if not DB_PATH.exists():
        pytest.skip("Real database not available — run ingest first")
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def feature_df(real_db):
    features = _selected_features()
    return real_db.execute(
        f"SELECT {', '.join(features)} FROM feature_store"
    ).fetchdf()


@pytest.fixture(scope="module")
def labelled_df(real_db):
    features = _selected_features()
    return real_db.execute(
        f"""
        SELECT f.{', f.'.join(features)}, r.fthg, r.ftag, r.hc, r.ac
        FROM feature_store f
        JOIN raw_matches r USING (match_id)
        """
    ).fetchdf()


# ---------------------------------------------------------------------------
# 1. Schema contract
# ---------------------------------------------------------------------------

def test_schema_contract(real_db):
    """All 114 selected_features must exist as columns in feature_store."""
    features = _selected_features()
    db_cols = {
        row[1]
        for row in real_db.execute("PRAGMA table_info('feature_store')").fetchall()
    }
    missing = [f for f in features if f not in db_cols]
    assert missing == [], f"Missing from feature_store: {missing}"


# ---------------------------------------------------------------------------
# 2. No infinity values
# ---------------------------------------------------------------------------

def test_no_inf_values(feature_df):
    num = feature_df.select_dtypes(include=[float, int])
    inf_mask = np.isinf(num.values)
    assert not inf_mask.any(), (
        f"{inf_mask.sum()} inf value(s) found in "
        f"{list(num.columns[inf_mask.any(axis=0)])}"
    )


# ---------------------------------------------------------------------------
# 3. Market probability sums to 1.0
# ---------------------------------------------------------------------------

def test_mkt_probability_sum_implied(feature_df):
    """MKT_IMPLIED_HOME + DRAW + AWAY must equal 1.0 ± 1e-5 on non-null rows."""
    cols = ["MKT_IMPLIED_HOME", "MKT_IMPLIED_DRAW", "MKT_IMPLIED_AWAY"]
    sub = feature_df[cols].dropna()
    deviation = (sub.sum(axis=1) - 1.0).abs()
    assert (deviation <= 1e-5).all(), (
        f"MKT_IMPLIED_ prob sum deviates from 1.0: max={deviation.max():.2e}"
    )


def test_mkt_probability_sum_real(feature_df):
    """MKT_*_Prob_Real must sum to 1.0 ± 1e-5 on non-null rows (US#77 Tier 1: removed from selected features)."""
    cols = ["MKT_Home_Prob_Real", "MKT_Draw_Prob_Real", "MKT_Away_Prob_Real"]
    available = [c for c in cols if c in feature_df.columns]
    if not available:
        pytest.skip("MKT_*_Prob_Real removed from selected features (US#77 Tier 1)")
    sub = feature_df[available].dropna()
    deviation = (sub.sum(axis=1) - 1.0).abs()
    assert (deviation <= 1e-5).all(), (
        f"MKT_*_Prob_Real sum deviates from 1.0: max={deviation.max():.2e}"
    )


# ---------------------------------------------------------------------------
# 4. NaN rate ceilings per feature family
# ---------------------------------------------------------------------------

_NAN_THRESHOLDS = [
    # OFF_ / DEF_: ceiling raised to 30 % because xG/LUCK rolling features (US#45)
    # are legitimately ~100 % NaN when understat data hasn't been fetched.
    # Run `python main.py fetch-understat` to populate xG and restore near-0 % rate.
    ("OFF_",          0.30),
    ("DEF_",          0.30),
    ("DIS_",          0.10),
    ("CTX_",          0.05),
    # MKT_ INTENTIONALLY FAILING: currently at ~30.6 % because odds data is
    # absent for older seasons.  Threshold set to < 5 % to force investigation.
    ("MKT_",          0.05),
    ("STRENGTH_",     0.15),
    ("INTERACTION_",  0.15),
    ("EFFICIENCY_",   0.15),
    ("OPP_ADJ_",      0.08),
]


@pytest.mark.parametrize("prefix,max_nan", _NAN_THRESHOLDS)
def test_nan_rate(feature_df, prefix, max_nan):
    features = _selected_features()
    family = [c for c in features if c.startswith(prefix)]
    if not family:
        pytest.skip(f"No selected features with prefix {prefix!r}")
    rate = feature_df[family].isna().mean().mean()
    assert rate < max_nan, (
        f"{prefix}* NaN rate {rate:.1%} exceeds ceiling {max_nan:.0%} "
        f"({len(family)} features)"
    )


# ---------------------------------------------------------------------------
# 5. Raw rolling features are non-negative
# ---------------------------------------------------------------------------

def test_off_features_non_negative(feature_df):
    """Attacking rolling averages (counts) must be >= 0.

    LUCK features are goals − xG and can legitimately be negative; excluded here.
    """
    features = _selected_features()
    off_cols = [c for c in features if c.startswith("OFF_") and "LUCK" not in c]
    sub = feature_df[off_cols].dropna()
    neg = (sub < 0).any()
    assert not neg.any(), (
        f"OFF_ features with negative values: {list(neg[neg].index)}"
    )


def test_opp_adj_raw_rolling_non_negative(feature_df):
    """OPP_ADJ raw rolling averages (not matchup deltas) must be >= 0."""
    features = _selected_features()
    raw_cols = [
        c for c in features
        if c.startswith("OPP_ADJ_") and "MATCHUP" not in c
    ]
    sub = feature_df[raw_cols].dropna()
    neg = (sub < 0).any()
    assert not neg.any(), (
        f"OPP_ADJ raw rolling features with negative values: {list(neg[neg].index)}"
    )


# ---------------------------------------------------------------------------
# 6. OPP_ADJ NaN coverage better than venue-split rolling
# ---------------------------------------------------------------------------

def test_opp_adj_nan_better_than_venue_split(feature_df):
    """OPP_ADJ NaN rate must be <= OFF_ NaN rate.

    Cold-start imputation may reduce both to 0%, so equality is permitted.
    The invariant is that the combined-venue timeline never degrades NaN coverage.
    """
    features = _selected_features()
    opp_raw = [
        c for c in features
        if c.startswith("OPP_ADJ_") and "MATCHUP" not in c
    ]
    off_cols = [c for c in features if c.startswith("OFF_")]

    opp_nan = feature_df[opp_raw].isna().mean().mean()
    off_nan = feature_df[off_cols].isna().mean().mean()

    assert opp_nan <= off_nan, (
        f"OPP_ADJ NaN rate {opp_nan:.1%} is higher than "
        f"OFF_ NaN rate {off_nan:.1%}"
    )


# ---------------------------------------------------------------------------
# 7. Feature-label correlation directions
# ---------------------------------------------------------------------------

_CORRELATION_PAIRS = [
    # (feature, label, expected_sign)
    ("OFF_HOME_FTHG_R5",               "fthg", 1),  # home attack form → home goals
    ("DEF_AWAY_FTHG_R5",               "fthg", 1),  # away defence concedes → home scores
    # MKT_Home_Prob_Real removed from selected features (US#77 Tier 1)
    ("MKT_IMPLIED_HOME",               "fthg", 1),  # market-implied home prob → home goals
    ("OFF_HOME_HC_R5",                 "hc",   1),  # home corner form → home corners
    ("OPP_ADJ_HOME_GOALS_SCORED_R5",   "fthg", 1),  # opp-adj home attack → home goals
    ("OPP_ADJ_AWAY_GOALS_CONCEDED_R5", "fthg", 1),  # opp-adj away defence → home scores
]


@pytest.mark.parametrize("feat,label,sign", _CORRELATION_PAIRS)
def test_correlation_direction(labelled_df, feat, label, sign):
    sub = labelled_df[[feat, label]].dropna()
    r = sub[feat].corr(sub[label])
    assert r * sign > 0, (
        f"r({feat}, {label}) = {r:.4f}; "
        f"expected {'positive' if sign > 0 else 'negative'}"
    )


# ---------------------------------------------------------------------------
# 8. OPP_ADJ matchup delta mean ≈ 0
# ---------------------------------------------------------------------------

def test_opp_adj_matchup_delta_mean_near_zero(feature_df):
    """Matchup deltas (home_scored - away_conceded) should average near 0."""
    features = _selected_features()
    matchup_cols = [c for c in features if "MATCHUP" in c]
    for col in matchup_cols:
        mean_val = feature_df[col].dropna().mean()
        assert abs(mean_val) < 0.5, (
            f"{col} mean = {mean_val:.4f}; expected near 0 (symmetric matchup)"
        )
