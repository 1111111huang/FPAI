# Feature Inventory

This document lists the engineered features produced by `FeatureFactory` and stored in `feature_store`.

## Rolling Performance

- `home_avg_goals_scored`: Rolling average of home team goals scored (leakage-safe, uses prior matches only).
- `home_avg_goals_conceded`: Rolling average of home team goals conceded (leakage-safe).
- `away_avg_goals_scored`: Rolling average of away team goals scored (leakage-safe).
- `away_avg_goals_conceded`: Rolling average of away team goals conceded (leakage-safe).

## Season/Context Flags

- `is_cold_start`: `True` if either team has fewer than `window` prior matches in history.
- `relative_tier_change`: Home team tier change minus away team tier change.
  - `1` indicates promotion (moving to a higher tier).
  - `-1` indicates relegation (moving to a lower tier).
  - `0` indicates no tier change.

## Market Bias

- `market_prob_h`: Implied home win probability from odds: `1 / odds_h`.

## Form/Strength Signals

- `elo_rating_diff`: Simplified pre-match score difference.
  - Each team’s internal score updates by last result: win `+1`, loss `-1`, draw `0`.
  - Feature is `home_score_before - away_score_before`.
- `home_advantage_trend`: Home team’s average points at home (last 10 home games)
  minus their overall average points (last 10 total games).
