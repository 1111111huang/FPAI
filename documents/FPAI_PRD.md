# Product Requirements Document - FPAI Forecast Engine

## 1. Product Objective
FPAI is a pre-match football forecasting engine. Its primary job is to convert historical match data, market data, and engineered team-form signals into structured probabilistic forecasts that an external AI agent can use during its decision process.

The model is no longer the final betting selector. It provides quantitative priors. The external agent is responsible for combining those priors with odds, news, injuries, lineups, tactical context, market movement, and other qualitative information before making any betting decision.

## 2. Product Positioning
### 2.1 Primary User
The primary consumer is an external AI agent that can call FPAI as a forecasting tool.

### 2.2 Secondary User
A human analyst or developer who wants readable forecast output, model diagnostics, and target-level evaluation reports.

### 2.3 Legacy Demotion
Existing value-bet and bankroll backtesting tools are legacy utilities. They may remain available for comparison, but new product work should not optimize around ROI, win rate, or direct bet recommendations.

## 3. Forecast Scope
FPAI should produce pre-match forecasts for the following target families:

1. Match result probabilities: home, draw, away.
2. Total goals distribution.
3. Home goals distribution.
4. Away goals distribution.
5. Both teams to score probability.
6. Corners distribution, where data quality permits.
7. Uncertainty estimates for every forecast family.

The project may later expose additional agent-callable tools, but those tools should be secondary to the forecast contract.

## 4. Output Requirements
Forecast output must be formatted JSON that is both agent-readable and human-readable.

Each forecast payload must include:

- Match identity: `match_id`, `date`, `league`, `home_team`, `away_team`.
- Forecast values for every available target.
- Uncertainty values for every forecast target.
- Top feature explanations, including feature name, match-level feature value, and importance or impact score where available.
- Diagnostics: model version, target version, feature completeness, cold-start risk, and generated timestamp.

Example payload shape:

```json
{
  "match_id": "2026-05-25_liverpool_arsenal",
  "league": "E0",
  "home_team": "Liverpool",
  "away_team": "Arsenal",
  "forecast": {
    "result_3way": {
      "probabilities": {
        "home": 0.42,
        "draw": 0.27,
        "away": 0.31
      },
      "uncertainty": {
        "method": "entropy",
        "score": 0.91,
        "level": "high"
      }
    },
    "home_goals": {
      "expected": 1.56,
      "distribution": {
        "0": 0.21,
        "1": 0.33,
        "2": 0.25,
        "3_plus": 0.21
      },
      "prediction_interval": {
        "lower": 0.4,
        "upper": 3.2,
        "coverage": 0.8
      }
    },
    "away_goals": {
      "expected": 1.22,
      "distribution": {
        "0": 0.29,
        "1": 0.35,
        "2": 0.22,
        "3_plus": 0.14
      },
      "prediction_interval": {
        "lower": 0.2,
        "upper": 2.8,
        "coverage": 0.8
      }
    },
    "both_teams_to_score": {
      "probabilities": {
        "yes": 0.57,
        "no": 0.43
      },
      "uncertainty": {
        "method": "entropy",
        "score": 0.99,
        "level": "high"
      }
    }
  },
  "explainability": {
    "top_features": [
      {
        "name": "OFF_HOME_XG_R5",
        "value": 1.72,
        "importance": 0.083
      },
      {
        "name": "MKT_Home_Prob_Real",
        "value": 0.46,
        "importance": 0.071
      }
    ]
  },
  "diagnostics": {
    "model_version": "forecast_suite_v1",
    "feature_completeness": 0.94,
    "cold_start_risk": false,
    "generated_at": "2026-05-25T00:00:00Z"
  }
}
```

## 5. Modeling Requirements
FPAI should use separate models per target unless a multi-output approach clearly improves validation performance and maintainability.

### 5.1 Model Tiers (Planned)
Models are organized into two tiers, declared per competition in a competition registry:

- `general_purpose`: market-odds-only features. Usable for any competition regardless of data richness, including matches with no team-history coverage.
- `competition_specific`: the full team-form feature set, extendable with player-level signals where a data source has been integrated for that competition.

A competition-specific model must never be less informed than the general-purpose model for the same target. Today this is guaranteed by a feature-superset rule (every competition-specific feature list contains the general-purpose feature list). If a future tier needs a model architecture where a literal feature superset doesn't apply, the competition-specific model may instead consume the general-purpose model's own prediction as an input feature.

Initial target candidates:

- `result_3way`: classification from full-time home and away goals.
- `home_goals`: regression or count distribution from `fthg`.
- `away_goals`: regression or count distribution from `ftag`.
- `total_goals`: regression or derived distribution from home and away goals.
- `btts`: binary classification from `fthg > 0 and ftag > 0`.
- `home_corners`: regression or count distribution from `hc`.
- `away_corners`: regression or count distribution from `ac`.
- `total_corners`: regression or derived distribution from home and away corners.

Only targets that can be generated from pre-match-safe labels and historically available data should be trained.

## 6. Evaluation Requirements
Evaluation should measure forecast quality, not betting profitability.

Classification targets:

- Primary metric: log loss.
- Secondary metric: accuracy.
- Additional recommended metric: calibration curve or Brier score.

Regression/count targets:

- Primary metric: MAE.
- Secondary metric: RMSE where useful.
- Prediction interval quality should be tracked once intervals are implemented.

All evaluation must use chronological splits. Random cross-validation is not acceptable for pre-match forecasting.

### 6.1 Performance Targets (Phase 2: Enhanced Model Performance)

The following performance targets have been established as improvement goals to maximize forecast quality:

**Classification Targets:**
- `result_3way`: Test set accuracy **≥ 60%** (baseline: 51.15%)
- `btts`: Test set log loss **≤ 0.68** (baseline: 0.6813)

**Regression Targets:**
- `home_goals`: Test set MAE **< 0.5** (baseline: 0.9428)
- `away_goals`: Test set MAE **< 0.5** (baseline: 0.8263)
- `total_goals`: Test set MAE **< 0.75** (baseline: 1.3047)
- `home_corners`: Test set MAE **< 1.5** (baseline: 2.1399)
- `away_corners`: Test set MAE **< 1.5** (baseline: 2.1133)
- `total_corners`: Test set MAE **< 1.5** (baseline: 2.6659)

These targets represent ~50% improvement over current baseline models and require advanced feature engineering, ensemble methods, or domain-specific optimization.

## 7. Data Requirements
The system must preserve the pre-match boundary. A forecast may only use information that would have been available before kickoff.

Allowed feature categories:

- Historical rolling team performance.
- Historical goals, shots, corners, cards, and xG-derived features.
- Market-implied probabilities where available.
- Rest days and schedule context.
- Team and league metadata.
- Squad-level player performance aggregates (e.g. rolling per-90 expected goals/assists across the roster), computed from data known before kickoff. These describe overall squad form, not the confirmed starting lineup.

Disallowed feature usage:

- Same-match outcomes as features.
- Same-match shots, corners, cards, or xG as direct features.
- Any data that would only be known after kickoff.

## 8. Non-Goals
FPAI does not need to:

- Recommend bets directly.
- Optimize for bankroll ROI as the primary metric.
- Require complete odds coverage for every target.
- Replace external agent judgment.

## 9. Roadmap
1. Define target registry and target-specific training contracts.
2. Add forecast JSON payload generation.
3. Train and evaluate separate models for existing labels.
4. Add entropy-based uncertainty for classification targets.
5. Add prediction intervals for regression/count targets.
6. Add top feature values and model explanation metadata.
7. Demote legacy strategy/backtest commands in CLI and documentation.
8. Add richer data sources such as Understat xG and future corner odds where available.
9. Formalize general-purpose vs competition-specific model tiers via a competition registry, guaranteeing competition-specific models retain general-purpose capability.
10. Source and ingest player-level performance data (starting with FBref) to build squad-level form features.
11. Extend competition-specific models with squad-level features and re-evaluate against current performance targets.
