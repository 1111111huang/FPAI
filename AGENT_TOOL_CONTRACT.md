# FPAI Agent Tool Contract

This document defines the formal input/output schemas for all five MCP tools exposed by `src/mcp_server.py`.

> **Important:** `list_matches` returns **historical** feature-store matches only. Upcoming matches not yet played are not included. Use `forecast_upcoming` to produce on-demand forecasts for future matches.

---

## 1. `forecast`

Produce structured forecast JSON for one or more historical feature-store matches.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `league` | string | No | League code filter (e.g. `"E0"`). |
| `match_ids` | string[] | No | Specific match IDs to forecast. |
| `targets` | string[] | No | Forecast target subset. |
| `limit` | integer | No | Maximum number of matches to return. |

### Output

Array of forecast payload objects. Each item contains:

```json
{
  "match_id": "string",
  "date": "ISO-8601",
  "league": "string",
  "home_team": "string",
  "away_team": "string",
  "forecast": { "<target>": { ... } },
  "explainability": { "top_features": [ ... ] },
  "diagnostics": {
    "model_version": "string",
    "target_versions": { "<target>": { "artifact": "...", "created_at": "...", "model_type": "..." } },
    "feature_completeness": 0.0,
    "cold_start_risk": false,
    "generated_at": "ISO-8601Z"
  }
}
```

---

## 2. `forecast_upcoming`

Produce an on-demand forecast for a named upcoming match without requiring it to exist in the feature store.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `home_team` | string | **Yes** | Home team name. Fuzzy-matched against `config/team_mapping.json`. |
| `away_team` | string | **Yes** | Away team name. |
| `date` | string | **Yes** | Match date `YYYY-MM-DD`. |
| `odds_h` | number | **Yes** | Home win decimal odds. |
| `odds_d` | number | **Yes** | Draw decimal odds. |
| `odds_a` | number | **Yes** | Away win decimal odds. |
| `league` | string | Cond. | League code. Required when `match_type="league"`. Optional for `"international"`. |
| `match_type` | string | No | `"league"` (default) or `"international"`. See below. |
| `over25_odds` | number | No | Over 2.5 goals decimal odds — unlocks Poisson lambda features. |
| `ah_line` | number | No | Asian handicap line — unlocks team-level lambda decomposition. |
| `ah_home_odds` | number | No | AH home decimal odds. |
| `ah_away_odds` | number | No | AH away decimal odds. |
| `targets` | string[] | No | Forecast target subset. |

#### `match_type` values

| Value | Description |
|---|---|
| `"league"` | Full rolling feature computation using team history from the feature store, plus MKT features from supplied odds. Requires `league`. |
| `"international"` | MKT_* features only (computed from odds). Team name lookup is skipped. `league` is optional. |

### Output

Single forecast payload with an additional `data_quality` field:

```json
{
  "match_id": "string",
  "date": "string",
  "league": "string",
  "home_team": "string",
  "away_team": "string",
  "forecast": { "<target>": { ... } },
  "explainability": { "top_features": [ ... ] },
  "diagnostics": { ... },
  "data_quality": {
    "prediction_basis": "team_history_and_market | market_odds_only | partial",
    "feature_count": 114,
    "caveat": "string"
  }
}
```

#### `data_quality.prediction_basis` values

| Value | When |
|---|---|
| `"team_history_and_market"` | `match_type="league"` — full features used. |
| `"market_odds_only"` | `match_type="international"` — only MKT_* features. |
| `"partial"` | Partial history available (cold-start or missing optional odds). |

---

## 3. `list_matches`

List historical matches from the feature store.

> **Note:** Returns historical matches only — upcoming matches not yet played are NOT included here.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `league` | string | No | League code filter. |
| `from_date` | string | No | ISO date lower bound (inclusive). |
| `to_date` | string | No | ISO date upper bound (inclusive). |
| `limit` | integer | No | Maximum number of matches. |

### Output

```json
[
  {
    "match_id": "string",
    "date": "YYYY-MM-DD",
    "home_team": "string",
    "away_team": "string",
    "league": "string"
  }
]
```

---

## 4. `model_status`

Return per-context per-target model selection status.

### Input

No parameters.

### Output

```json
{
  "league": {
    "<target>": {
      "model_type": "string",
      "primary_metric_value": 0.0,
      "metric_name": "string",
      "selected_at": "ISO-8601Z"
    }
  },
  "international": {
    "<target>": { ... }
  }
}
```

---

## 5. `data_freshness`

Return data freshness metadata.

### Input

No parameters.

### Output

```json
{
  "latest_match_date": "YYYY-MM-DD",
  "days_since_update": 3,
  "match_count": 12450,
  "is_stale": false
}
```

`is_stale` is `true` when `days_since_update > 7` or no data is present.
