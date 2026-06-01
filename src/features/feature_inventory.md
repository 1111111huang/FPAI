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
- `MKT_H_Prob_Clean`: Margin-free home win probability from AvgH/AvgD/AvgA (normalized implied odds).
- `MKT_D_Prob_Clean`: Margin-free draw probability from AvgH/AvgD/AvgA.
- `MKT_A_Prob_Clean`: Margin-free away win probability from AvgH/AvgD/AvgA.

## Form/Strength Signals

- `elo_rating_diff`: Simplified pre-match score difference.
  - Each team’s internal score updates by last result: win `+1`, loss `-1`, draw `0`.
  - Feature is `home_score_before - away_score_before`.
- `home_advantage_trend`: Home team’s average points at home (last 10 home games)
  minus their overall average points (last 10 total games).

## Efficiency & Strength Additions

- `OFF_Shot_Quality_R5`: Rolling 5-match average of `HST / HS` for the home team.
- `DEF_Save_Rate_R5`: Rolling 5-match average of `(AST - FTAG) / AST` for the home team.
- `STRENGTH_Goal_Diff`: `OFF_HOME_FTHG_R5 - DEF_AWAY_FTHG_R5`.
- `STRENGTH_SoT_Diff`: `OFF_HOME_HST_R5 - DEF_AWAY_HST_R5`.
