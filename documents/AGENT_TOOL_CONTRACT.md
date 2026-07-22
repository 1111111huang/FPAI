# FPAI Agent Tool Contract

External agents should call FPAI through the forecast CLI:

```bash
python main.py forecast --league E0 --limit 20
python main.py forecast --match_id 2026-05-25_liverpool_arsenal
python main.py forecast --league E0 --target result_3way btts total_goals
python main.py forecast --home "Malmo FF" --away "Djurgarden" --date 2026-07-27 --league SWE --odds_h 2.1 --odds_d 3.4 --odds_a 3.2
```

The command returns formatted JSON: a list of forecast payloads, one per match.

Each payload includes:

- Match identity: `match_id`, `date`, `league`, `home_team`, `away_team`.
- `forecast`: target-keyed predictions.
- `explainability.top_features`: match feature values plus global feature importance.
- `diagnostics`: model versions, target artifact metadata, feature completeness, cold-start risk, and generation time.

Classification targets expose probabilities and entropy uncertainty. Regression/count targets expose an expected value, count buckets, and validation-residual prediction intervals where metadata is available.

## Registered competitions (US#128, US#136)

FPAI is no longer EPL-only. `league` is resolved against a competition registry (`config/competitions.yaml`), not a hardcoded pair of values — call `model_status` (below) to see exactly which competitions and targets are live at any given time rather than assuming this list is exhaustive or permanent. As of this writing:

| `league` | Tier | Available targets | Notes |
| :--- | :--- | :--- | :--- |
| `E0` (English Premier League) | `competition_specific` | All 8: `result_3way`, `btts`, `home_goals`, `away_goals`, `total_goals`, `home_corners`, `away_corners`, `total_corners` | Full feature set (167 features): team-form rolling stats, real xG, squad/lineup features, market odds. |
| `SWE` (Sweden Allsvenskan) | `competition_specific` | 5 only: `result_3way`, `btts`, `home_goals`, `away_goals`, `total_goals` — **no corners targets** | Source (football-data.co.uk's "New Leagues" feed) has no shots/corners/cards columns at all. `data_quality.feature_count` will typically read lower than an equivalent E0 call (~53/74 vs. ~130+/167) for the same reason — this is expected, not a data-quality problem, unless it drops further after a source-format change. |
| any unregistered league / `match_type="international"` | `general_purpose` | All 8, but market-odds-only (13 `MKT_*` features, no team-history rolling stats) | Trained on pooled data across every registered `competition_specific` competition (E0 + SWE) — see US#138/139. |

Requesting a target FPAI doesn't have for a given competition (e.g. `home_corners` for `SWE`) simply won't appear in that match's `forecast` dict — there is no per-request error for an unavailable target; check which keys are actually present rather than assuming all requested targets returned.

## Agent tools (MCP server, `src/mcp_server.py`)

| Tool | Backing function | Notes |
| :--- | :--- | :--- |
| `forecast` | `forecast_tools.forecast_matches` | Historical/feature-store matches by `match_id`; `league` is a free-form filter, not restricted to a fixed enum. |
| `forecast_upcoming` | `forecast_tools.forecast_upcoming` | Spot inference for an arbitrary upcoming fixture; `match_type="league"` (default) or `"international"`. Routes through the same registry-driven resolution as the CLI. |
| `list_matches` | `data_tools.list_matches` | Historical match lookup; `league` free-form filter. |
| `model_status` | `model_tools.get_model_status` | Per-context per-target model selection status. Keys are derived from the competition registry, **not** a fixed `{"league", "international"}` pair — will include `E0`, `SWE`, `international`, and any future registered competition automatically as each is trained. |
| `data_freshness` | `data_tools.get_data_freshness` | Data currency check. Top-level fields (`latest_match_date`, `days_since_update`, `match_count`, `is_stale`) are blended across *every* competition in `raw_matches` — with two competitions on different season calendars (E0: Aug–May; SWE: Mar–Nov) and different refresh cadences, a blended `is_stale: false` can mask one specific competition going stale behind another staying fresh. Use the `by_league` dict (added US#136) for a per-competition breakdown when that distinction matters — e.g. `by_league.E0.is_stale` can be `true` even when the top-level `is_stale` reads `false`. |

Agent interpretation guidance:

- Treat FPAI output as quantitative priors, not final betting advice.
- High entropy means the model sees a flatter classification probability vector.
- Low `feature_completeness` or `cold_start_risk: true` means downstream reasoning should rely more heavily on external context. For `SWE` specifically, a moderate `feature_completeness` (roughly 70%) is the **normal, permanent** state given its narrower source data — don't treat it as equivalent to an E0 forecast reporting the same number, which would indicate a genuine data gap for that specific match rather than a structural one shared by every SWE forecast.
- Prediction intervals are validation-residual intervals from the current target artifact metadata.
- `top_features` are global model importances with the current match values attached; they are not local causal explanations.
- When checking data freshness for a specific competition (e.g. before deciding whether to trust an SWE forecast), read `data_freshness.by_league.<league>` rather than the top-level fields, which describe the most-recently-updated competition, not necessarily the one you asked about.
