"""Tests for the goal-market lambda NaN fallback -- see
docs/superpowers/specs/2026-11-16-goal-market-lambda-fallback-design.md."""
from __future__ import annotations

import math

from src.features.goal_market_fallback import (
    DEFAULT_HOME_SHARE_COEFFICIENTS,
    fallback_home_share,
    fallback_lambda_home_away,
)


def test_fallback_home_share_is_flat_intercept_when_gap_is_zero():
    coefficients = {"E0": (0.2, 0.55)}
    share = fallback_home_share(0.4, 0.4, "E0", coefficients=coefficients)
    assert share == 0.55


def test_fallback_home_share_increases_with_home_favoritism():
    coefficients = {"E0": (0.2, 0.55)}
    favored_home = fallback_home_share(0.7, 0.2, "E0", coefficients=coefficients)
    favored_away = fallback_home_share(0.2, 0.7, "E0", coefficients=coefficients)
    assert favored_home > 0.55 > favored_away


def test_fallback_home_share_clips_at_bounds():
    coefficients = {"E0": (5.0, 0.55)}  # extreme slope, forces clipping both ways
    assert fallback_home_share(0.9, 0.05, "E0", coefficients=coefficients) == 0.75
    assert fallback_home_share(0.05, 0.9, "E0", coefficients=coefficients) == 0.30


def test_fallback_home_share_falls_back_to_intercept_when_implied_probs_missing():
    coefficients = {"E0": (0.2, 0.55)}
    share = fallback_home_share(float("nan"), float("nan"), "E0", coefficients=coefficients)
    assert share == 0.55


def test_fallback_home_share_unknown_league_uses_default_coefficients():
    share = fallback_home_share(0.4, 0.4, "XX_UNKNOWN", coefficients={})
    assert share == DEFAULT_HOME_SHARE_COEFFICIENTS[1]


def test_fallback_lambda_home_away_sums_to_total():
    coefficients = {"E0": (0.0, 0.54)}
    home, away = fallback_lambda_home_away(2.6, 0.5, 0.3, "E0", coefficients=coefficients)
    assert math.isclose(home + away, 2.6, rel_tol=1e-9)
    assert math.isclose(home, 2.6 * 0.54, rel_tol=1e-9)
    assert math.isclose(away, 2.6 * 0.46, rel_tol=1e-9)
