"""Fallback computation for MKT_LAMBDA_HOME/MKT_LAMBDA_AWAY when the Asian
Handicap line is missing at live-forecast generation time -- see
docs/superpowers/specs/2026-11-16-goal-market-lambda-fallback-design.md.

Live-serving only: only called from FeatureFactory.build_for_match
(src/features/feature_factory.py), the single-upcoming-match path. Never
called from compute_rolling_stats()/_compute_odds_features(), the bulk
training/feature_store-build path -- training data has ~100% real AH-line
coverage (BUG-096), so this fallback is never exercised there, and it must
stay that way to avoid a train/serve mismatch.
"""

from __future__ import annotations

import math

# Per-league (slope, intercept) for
#   real_home_share ~= intercept + slope * (MKT_IMPLIED_HOME - MKT_IMPLIED_AWAY)
# fit offline by scripts/calibrate_goal_market_fallback.py against
# feature_store rows where MKT_AH_LINE is real (so the true split is known).
# INTERIM PLACEHOLDER (slope=0.0, a flat global home-advantage prior) until
# that script is run against production and its real output is pasted in
# here -- see this plan's Task 6. Do not treat these as calibrated.
HOME_SHARE_COEFFICIENTS: dict[str, tuple[float, float]] = {
    "E0": (0.0, 0.54),
    "SP1": (0.0, 0.54),
    "I1": (0.0, 0.54),
    "D1": (0.0, 0.54),
    "F1": (0.0, 0.54),
}

# Used for any league not in HOME_SHARE_COEFFICIENTS (shouldn't happen for
# the 5 registered domestic leagues, but keeps this module from raising on
# an unexpected league code rather than crashing a live forecast).
DEFAULT_HOME_SHARE_COEFFICIENTS: tuple[float, float] = (0.0, 0.54)

# A match's calibrated home-share can never fall outside this band, however
# extreme the implied-probability gap -- guards against a pathological
# extrapolation inverting or degenerating the split.
_MIN_HOME_SHARE = 0.30
_MAX_HOME_SHARE = 0.75


def fallback_home_share(
    implied_home: float,
    implied_away: float,
    league: str,
    coefficients: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Calibrated estimate of the match's home goal-share, used only when
    the real AH-derived split is unavailable.

    implied_home/implied_away: this match's own MKT_IMPLIED_HOME/
    MKT_IMPLIED_AWAY. Pass float('nan') for either (not None) when the 1X2
    market is also missing (rare) -- the slope term is dropped and the flat
    per-league intercept is used alone.

    coefficients: defaults to the module-level HOME_SHARE_COEFFICIENTS;
    callers (tests, the calibration script) may pass their own table.
    """
    table = coefficients if coefficients is not None else HOME_SHARE_COEFFICIENTS
    slope, intercept = table.get(league, DEFAULT_HOME_SHARE_COEFFICIENTS)
    gap = implied_home - implied_away
    share = intercept if math.isnan(gap) else intercept + slope * gap
    return min(max(share, _MIN_HOME_SHARE), _MAX_HOME_SHARE)


def fallback_lambda_home_away(
    lambda_total: float,
    implied_home: float,
    implied_away: float,
    league: str,
    coefficients: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """(lambda_home, lambda_away), summing exactly to lambda_total, split by
    fallback_home_share(). Caller is responsible for only invoking this when
    lambda_total is real and the real AH-derived split is NaN."""
    home_share = fallback_home_share(implied_home, implied_away, league, coefficients=coefficients)
    return lambda_total * home_share, lambda_total * (1.0 - home_share)
