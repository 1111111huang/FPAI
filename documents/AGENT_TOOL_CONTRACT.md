# FPAI Agent Tool Contract

External agents should call FPAI through the forecast CLI:

```bash
python main.py forecast --league E0 --limit 20
python main.py forecast --match_id 2026-05-25_liverpool_arsenal
python main.py forecast --league E0 --target result_3way btts total_goals
```

The command returns formatted JSON: a list of forecast payloads, one per match.

Each payload includes:

- Match identity: `match_id`, `date`, `league`, `home_team`, `away_team`.
- `forecast`: target-keyed predictions.
- `explainability.top_features`: match feature values plus global feature importance.
- `diagnostics`: model versions, target artifact metadata, feature completeness, cold-start risk, and generation time.

Classification targets expose probabilities and entropy uncertainty. Regression/count targets expose an expected value, count buckets, and validation-residual prediction intervals where metadata is available.

Agent interpretation guidance:

- Treat FPAI output as quantitative priors, not final betting advice.
- High entropy means the model sees a flatter classification probability vector.
- Low `feature_completeness` or `cold_start_risk: true` means downstream reasoning should rely more heavily on external context.
- Prediction intervals are validation-residual intervals from the current target artifact metadata.
- `top_features` are global model importances with the current match values attached; they are not local causal explanations.
