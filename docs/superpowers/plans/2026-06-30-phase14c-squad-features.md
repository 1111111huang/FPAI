# Phase 14c: Squad Feature Engineering & Model Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pre-match-safe SQUAD_* rolling features derived from FotMob per-match player stats (xG, xA, rating), gate them to Premier League (E0) via the competition registry, rebuild the feature store, and retrain competition_specific models with the expanded feature set.

**Architecture:** `feature_factory.py` gains two new methods — `_compute_squad_features()` opens a separate DB read connection to query `raw_player_match_stats`, and `_squad_rolling_from_data()` (static, pure, fully testable) aggregates per-team-per-match player stats and applies shifted rolling R3/R5 windows. The result is merged into `compute_rolling_stats()`'s output exactly like every other feature sub-component (`h2h`, `temporal`, etc.). `competitions.yaml` adds "SQUAD" to E0's `enabled_feature_groups`. `ModelManager` gains an optional `competition_id` parameter; `_load_selected_features()` filters out `SQUAD_*` columns when the competition's registry entry does not include "SQUAD" — ensuring general_purpose models never accidentally pick up player features. `main.py:run_train_target` passes `competition_id` through. US#99 (explainability surfacing) is handled automatically by the existing `ForecastService._top_features()` mechanism, which is already name-agnostic — no new forecast service code is needed; the task just verifies it works.

**Tech Stack:** Python, pandas, DuckDB, pytest, PyYAML.

**Prerequisite:** `raw_player_match_stats` and `player_dim` tables must be populated in the main repo's `data/fpai_core.db` before Tasks 3-4. Run: `python main.py fetch-fotmob --league E0 --from_season 2024 --to_season 2024 --delay 1.0` (takes ~20 minutes; rerun from_season 2016 onward for a full historical backfill). This is NOT automated in tests — Task 1 and Task 2 have isolated unit tests that create their own in-memory data.

---

### Task 1: Squad feature computation in FeatureFactory (US#96)

**Files:**
- Modify: `src/features/feature_factory.py` (add `_compute_squad_features`, `_squad_rolling_from_data`, one merge in `compute_rolling_stats`)
- Modify: `config/schema.yaml` (add 12 `SQUAD_*` feature names to `selected_features`)
- Test: `tests/test_squad_features.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for SQUAD_* rolling feature computation from raw_player_match_stats."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features.feature_factory import FeatureFactory

SQUAD_FEATURE_NAMES = [
    "SQUAD_HOME_XG_MEAN_R3", "SQUAD_HOME_XG_MEAN_R5",
    "SQUAD_HOME_XA_MEAN_R3", "SQUAD_HOME_XA_MEAN_R5",
    "SQUAD_HOME_RATING_MEAN_R3", "SQUAD_HOME_RATING_MEAN_R5",
    "SQUAD_AWAY_XG_MEAN_R3", "SQUAD_AWAY_XG_MEAN_R5",
    "SQUAD_AWAY_XA_MEAN_R3", "SQUAD_AWAY_XA_MEAN_R5",
    "SQUAD_AWAY_RATING_MEAN_R3", "SQUAD_AWAY_RATING_MEAN_R5",
]


def _raw_matches_df() -> pd.DataFrame:
    """Minimal raw_matches frame: 3 matches, Arsenal (home) vs Everton (away)."""
    return pd.DataFrame([
        {"match_id": "m1", "date": "2024-08-10", "home_team": "Arsenal", "away_team": "Everton"},
        {"match_id": "m2", "date": "2024-08-17", "home_team": "Arsenal", "away_team": "Everton"},
        {"match_id": "m3", "date": "2024-08-24", "home_team": "Arsenal", "away_team": "Everton"},
    ])


def _player_df_two_matches() -> pd.DataFrame:
    """2 players per team for matches m1 and m2."""
    rows = []
    for match_id, xg, xa, rating in [("m1", 0.5, 0.2, 7.0), ("m2", 0.8, 0.4, 7.5)]:
        for team in ["Arsenal", "Everton"]:
            for i in range(2):
                rows.append({
                    "match_id": match_id,
                    "team_name": team,
                    "xg": xg + i * 0.1,
                    "xa": xa + i * 0.05,
                    "rating": rating + i * 0.1,
                })
    return pd.DataFrame(rows)


def test_squad_rolling_returns_correct_columns() -> None:
    result = FeatureFactory._squad_rolling_from_data(_player_df_two_matches(), _raw_matches_df())

    assert "match_id" in result.columns
    for col in SQUAD_FEATURE_NAMES:
        assert col in result.columns, f"Missing column: {col}"


def test_squad_rolling_first_match_is_nan_because_no_prior_data() -> None:
    result = FeatureFactory._squad_rolling_from_data(_player_df_two_matches(), _raw_matches_df())

    # m1 is each team's first match → no prior data → all rolling features are NaN
    row_m1 = result[result["match_id"] == "m1"].iloc[0]
    assert pd.isna(row_m1["SQUAD_HOME_XG_MEAN_R3"])
    assert pd.isna(row_m1["SQUAD_AWAY_RATING_MEAN_R5"])


def test_squad_rolling_second_match_uses_first_match_stats() -> None:
    result = FeatureFactory._squad_rolling_from_data(_player_df_two_matches(), _raw_matches_df())

    # m2 R3/R5 for Arsenal home: mean of m1 Arsenal players' xG
    # m1 Arsenal players: xg=[0.5, 0.6] → mean=0.55
    row_m2 = result[result["match_id"] == "m2"].iloc[0]
    assert row_m2["SQUAD_HOME_XG_MEAN_R3"] == pytest.approx(0.55, abs=1e-3)
    assert row_m2["SQUAD_HOME_XG_MEAN_R5"] == pytest.approx(0.55, abs=1e-3)  # only 1 prior → same value


def test_squad_rolling_handles_empty_player_data() -> None:
    empty = pd.DataFrame(columns=["match_id", "team_name", "xg", "xa", "rating"])
    result = FeatureFactory._squad_rolling_from_data(empty, _raw_matches_df())

    assert result.empty or (len(result.columns) == 1 and result.columns[0] == "match_id")


def test_squad_rolling_normalises_abbreviated_team_names() -> None:
    """'Man City' in player stats must match 'Manchester City' in raw_matches."""
    raw = pd.DataFrame([
        {"match_id": "m1", "date": "2024-08-10", "home_team": "Manchester City", "away_team": "Arsenal"},
        {"match_id": "m2", "date": "2024-08-17", "home_team": "Manchester City", "away_team": "Arsenal"},
    ])
    players = pd.DataFrame([
        {"match_id": "m1", "team_name": "Man City", "xg": 0.7, "xa": 0.3, "rating": 7.2},
        {"match_id": "m1", "team_name": "Arsenal", "xg": 0.4, "xa": 0.1, "rating": 6.8},
        {"match_id": "m2", "team_name": "Man City", "xg": 0.9, "xa": 0.5, "rating": 7.8},
        {"match_id": "m2", "team_name": "Arsenal", "xg": 0.3, "xa": 0.2, "rating": 6.5},
    ])

    result = FeatureFactory._squad_rolling_from_data(players, raw)

    row_m2 = result[result["match_id"] == "m2"].iloc[0]
    # m2 home (Man City) R3 should be m1's Man City mean xg = 0.7 (not NaN / not mismatched)
    assert row_m2["SQUAD_HOME_XG_MEAN_R3"] == pytest.approx(0.7, abs=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_squad_features.py -v`
Expected: FAIL with `AttributeError: type object 'FeatureFactory' has no attribute '_squad_rolling_from_data'`

- [ ] **Step 3: Implement `_squad_rolling_from_data` and `_compute_squad_features` in `feature_factory.py`**

At the bottom of the `FeatureFactory` class (after `_ensure_feature_store_schema`), add:

```python
    def _compute_squad_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Query raw_player_match_stats and delegate to the pure rolling helper.

        Returns empty DataFrame (with only match_id column) when the table does
        not exist yet — feature_factory callers must check for emptiness before
        merging.
        """
        import duckdb as _duckdb

        try:
            with self.db_manager.connection(read_only=True) as conn:
                player_df = conn.execute(
                    "SELECT match_id, team_name, xg, xa, rating FROM raw_player_match_stats"
                ).fetchdf()
        except _duckdb.CatalogException:
            return pd.DataFrame(columns=["match_id"])
        return self._squad_rolling_from_data(player_df, raw_df)

    @staticmethod
    def _squad_rolling_from_data(
        player_df: pd.DataFrame, raw_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate per-match player stats to rolling squad-level features.

        Applies shifted R3/R5 windows to ensure all SQUAD_* values reflect
        information available *before* the current match (pre-match safe).

        Args:
            player_df: Rows from raw_player_match_stats (match_id, team_name,
                xg, xa, rating). FotMob-abbreviated team names are normalised
                via standardize_team_name before joining.
            raw_df: Rows from raw_matches (match_id, date, home_team, away_team)
                with already-canonical team names.

        Returns:
            DataFrame keyed by match_id with 12 SQUAD_* columns, one row per
            match. Returns a single-column match_id DataFrame (empty) when
            player_df is empty.
        """
        from src.utils.helpers import standardize_team_name

        if player_df.empty:
            return pd.DataFrame(columns=["match_id"])

        # Normalise FotMob-abbreviated team names (e.g. "Man City" → "Manchester City")
        player_df = player_df.copy()
        player_df["team_std"] = player_df["team_name"].map(standardize_team_name)

        # Per-match per-team aggregate (NaN-safe: skip null xg/xa/rating values)
        agg = (
            player_df.groupby(["match_id", "team_std"])
            .agg(squad_xg=("xg", "mean"), squad_xa=("xa", "mean"), squad_rating=("rating", "mean"))
            .reset_index()
        )

        match_info = raw_df[["match_id", "date", "home_team", "away_team"]].copy()
        match_info["date"] = pd.to_datetime(match_info["date"])

        # Join dates for chronological ordering
        agg = agg.merge(match_info[["match_id", "date"]], on="match_id", how="inner")
        agg = agg.sort_values(["team_std", "date", "match_id"]).reset_index(drop=True)

        # Shifted rolling windows per team (shift=1 guarantees pre-match safety)
        for metric in ["squad_xg", "squad_xa", "squad_rating"]:
            for window in [3, 5]:
                col = f"_{metric}_r{window}"
                agg[col] = agg.groupby("team_std")[metric].transform(
                    lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
                )

        stat_cols = [
            "_squad_xg_r3", "_squad_xg_r5",
            "_squad_xa_r3", "_squad_xa_r5",
            "_squad_rating_r3", "_squad_rating_r5",
        ]

        def _join_side(side: str) -> pd.DataFrame:
            rename_map = {
                "_squad_xg_r3":     f"SQUAD_{side}_XG_MEAN_R3",
                "_squad_xg_r5":     f"SQUAD_{side}_XG_MEAN_R5",
                "_squad_xa_r3":     f"SQUAD_{side}_XA_MEAN_R3",
                "_squad_xa_r5":     f"SQUAD_{side}_XA_MEAN_R5",
                "_squad_rating_r3": f"SQUAD_{side}_RATING_MEAN_R3",
                "_squad_rating_r5": f"SQUAD_{side}_RATING_MEAN_R5",
            }
            team_col = "home_team" if side == "HOME" else "away_team"
            joined = match_info.merge(
                agg[["match_id", "team_std"] + stat_cols],
                left_on=["match_id", team_col],
                right_on=["match_id", "team_std"],
                how="left",
            )
            return joined.rename(columns=rename_map)[
                ["match_id"] + list(rename_map.values())
            ]

        home_feats = _join_side("HOME")
        away_feats = _join_side("AWAY")
        return home_feats.merge(away_feats, on="match_id", how="left")
```

- [ ] **Step 4: Wire `_compute_squad_features` into `compute_rolling_stats`**

In `src/features/feature_factory.py`, find the line (around line 303):
```python
        temporal = self._compute_temporal_features(raw_df)
        features = features.merge(temporal, on="match_id", how="left")

        # US#59: cold-start imputation — fill NaN rolling values with column means
        features = self._apply_cold_start_imputation(features)
```

Change it to:
```python
        temporal = self._compute_temporal_features(raw_df)
        features = features.merge(temporal, on="match_id", how="left")

        # US#96: squad-level rolling features (skipped when raw_player_match_stats absent)
        squad = self._compute_squad_features(raw_df)
        if not squad.empty:
            features = features.merge(squad, on="match_id", how="left")

        # US#59: cold-start imputation — fill NaN rolling values with column means
        features = self._apply_cold_start_imputation(features)
```

- [ ] **Step 5: Add SQUAD_* features to `config/schema.yaml`**

At the end of `training_setup.selected_features` (after the last `CTX_AWAY_CS_STREAK` entry), append:

```yaml
  # US#96: squad-level rolling features from FotMob player stats (Phase 14c)
  - SQUAD_HOME_XG_MEAN_R3
  - SQUAD_HOME_XG_MEAN_R5
  - SQUAD_HOME_XA_MEAN_R3
  - SQUAD_HOME_XA_MEAN_R5
  - SQUAD_HOME_RATING_MEAN_R3
  - SQUAD_HOME_RATING_MEAN_R5
  - SQUAD_AWAY_XG_MEAN_R3
  - SQUAD_AWAY_XG_MEAN_R5
  - SQUAD_AWAY_XA_MEAN_R3
  - SQUAD_AWAY_XA_MEAN_R5
  - SQUAD_AWAY_RATING_MEAN_R3
  - SQUAD_AWAY_RATING_MEAN_R5
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_squad_features.py -v`
Expected: 5 passed

- [ ] **Step 7: Run full suite to check for regressions**

Run: `pytest -q`
Expected: (previous pass count + 5) passed, same skipped, 0 failures

- [ ] **Step 8: Commit**

```bash
git add src/features/feature_factory.py config/schema.yaml tests/test_squad_features.py
git commit -m "feat: add SQUAD_* rolling squad features to FeatureFactory (US#96)"
```

---

### Task 2: Competition registry gating (US#97)

**Files:**
- Modify: `config/competitions.yaml` (add `SQUAD` to E0's `enabled_feature_groups`)
- Modify: `src/models/model_manager.py:36-73` (add `competition_id` param, filter SQUAD_* in `_load_selected_features`)
- Modify: `main.py` (pass `competition_id` to `ModelManager`)
- Test: `tests/test_competition_registry.py` (add one gating test)
- Test: `tests/test_model_manager_squad_gating.py` (new, tests `_load_selected_features` filtering)

- [ ] **Step 1: Write the failing tests**

Add this test at the bottom of `tests/test_competition_registry.py`:

```python
def test_squad_is_in_e0_enabled_feature_groups_after_phase14c() -> None:
    """E0 must declare SQUAD once competitions.yaml is updated for Phase 14c."""
    definition = get_competition_definition("E0")
    assert "SQUAD" in definition.enabled_feature_groups
```

Create `tests/test_model_manager_squad_gating.py`:

```python
"""Tests for SQUAD_* feature gating in ModelManager._load_selected_features."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.model_manager import ModelManager
from src.models.base_model import XGBoostRegressorModel


def _make_manager(competition_id: str, tmp_path: Path) -> ModelManager:
    db_path = tmp_path / "test.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8"
    )
    schema_path = tmp_path / "config" / "schema.yaml"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(
        yaml.safe_dump({
            "training_setup": {
                "selected_features": [
                    "MKT_IMPLIED_HOME",
                    "MKT_IMPLIED_AWAY",
                    "OFF_HOME_FTHG_R5",
                    "SQUAD_HOME_XG_MEAN_R5",
                    "SQUAD_AWAY_RATING_MEAN_R3",
                ]
            }
        }),
        encoding="utf-8",
    )
    return ModelManager(
        model=XGBoostRegressorModel(),
        config_path=str(config_path),
        target_config={"target": "home_goals"},
        competition_id=competition_id,
    )


def test_competition_specific_with_squad_includes_squad_features(tmp_path: Path) -> None:
    manager = _make_manager("E0", tmp_path)
    features = manager._load_selected_features()
    squad_features = [f for f in features if f.startswith("SQUAD_")]
    assert len(squad_features) == 2  # both SQUAD_* in the schema list are included


def test_competition_without_squad_group_excludes_squad_features(tmp_path: Path) -> None:
    manager = _make_manager("international", tmp_path)
    # international has feature_subset=MKT_FEATURES (passed explicitly), so
    # _load_selected_features hits the feature_subset branch. Test the filtering
    # separately by patching feature_subset to None to exercise the gating branch.
    manager.feature_subset = None
    features = manager._load_selected_features()
    squad_features = [f for f in features if f.startswith("SQUAD_")]
    assert len(squad_features) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_competition_registry.py::test_squad_is_in_e0_enabled_feature_groups_after_phase14c tests/test_model_manager_squad_gating.py -v`
Expected:
- `test_squad_is_in_e0_enabled_feature_groups_after_phase14c` FAIL: `AssertionError: "SQUAD" not in enabled_feature_groups`
- `test_competition_specific_with_squad_includes_squad_features` FAIL: `TypeError: __init__() got unexpected keyword argument 'competition_id'`

- [ ] **Step 3: Update `config/competitions.yaml` — add SQUAD to E0**

Change `config/competitions.yaml` E0 entry from:

```yaml
    enabled_feature_groups:
      - "OFF"
      - DEF
      - DIS
      - CTX
      - MKT
      - STRENGTH
      - INTERACTION
      - EFFICIENCY
```

to:

```yaml
    enabled_feature_groups:
      - "OFF"
      - DEF
      - DIS
      - CTX
      - MKT
      - STRENGTH
      - INTERACTION
      - EFFICIENCY
      - SQUAD
```

Also update `player_data_sources` under E0 to document the active source:

```yaml
    player_data_sources:
      - fotmob
```

- [ ] **Step 4: Add `competition_id` param to `ModelManager.__init__`**

In `src/models/model_manager.py`, change `__init__`'s signature from:

```python
    def __init__(
        self,
        model: FPAIBaseModel,
        config_path: str = "config.yaml",
        league_tier: str = "all",
        test_season: str = "time_split",
        feature_version: str = "v1",
        target_config: dict[str, str | float | int] | None = None,
        feature_subset: list[str] | None = None,
        context: str = "league",
    ) -> None:
```

to:

```python
    def __init__(
        self,
        model: FPAIBaseModel,
        config_path: str = "config.yaml",
        league_tier: str = "all",
        test_season: str = "time_split",
        feature_version: str = "v1",
        target_config: dict[str, str | float | int] | None = None,
        feature_subset: list[str] | None = None,
        context: str = "league",
        competition_id: str = "E0",
    ) -> None:
```

And inside `__init__`, after `self.feature_subset: list[str] | None = feature_subset`, add:

```python
        self.competition_id: str = competition_id
```

- [ ] **Step 5: Add SQUAD_* gating in `_load_selected_features`**

In `src/models/model_manager.py`, in `_load_selected_features`, change the final `return all_features` (currently the last line of the method, around line 115) from:

```python
        return all_features
```

to:

```python
        # US#97: filter SQUAD_* features for competitions whose registry entry
        # does not include "SQUAD" in enabled_feature_groups.
        try:
            from src.logic.competition_registry import get_competition_definition
            comp_def = get_competition_definition(self.competition_id)
            if "SQUAD" not in comp_def.enabled_feature_groups:
                all_features = [f for f in all_features if not f.startswith("SQUAD_")]
        except Exception:
            pass  # Registry unavailable or unknown competition_id — include all features
        return all_features
```

- [ ] **Step 6: Pass `competition_id` from `main.py:run_train_target`**

In `main.py`, change the `ModelManager(...)` call inside `run_train_target` from:

```python
    model_manager = ModelManager(
        model=model,
        target_config={"target": definition.name},
        feature_subset=feature_subset,
        context=context,
    )
```

to:

```python
    model_manager = ModelManager(
        model=model,
        target_config={"target": definition.name},
        feature_subset=feature_subset,
        context=context,
        competition_id=competition_id,
    )
```

(`competition_id` is already in scope two lines above: `competition_id = "international" if context == "international" else "E0"`)

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_competition_registry.py tests/test_model_manager_squad_gating.py -v`
Expected: all tests pass (including the new gating tests)

- [ ] **Step 8: Run full suite**

Run: `pytest -q`
Expected: 0 failures

- [ ] **Step 9: Commit**

```bash
git add config/competitions.yaml src/models/model_manager.py main.py \
        tests/test_competition_registry.py tests/test_model_manager_squad_gating.py
git commit -m "feat: gate SQUAD_* features via competition registry (US#97)"
```

---

### Task 3: Populate player data and retrain competition_specific models (US#98)

**Files:** No code changes. CLI-only steps.

**Prerequisite check before starting this task:**

```bash
python -c "
from src.utils.db_manager import DuckDBManager
db = DuckDBManager('config.yaml')
with db.connection(read_only=True) as conn:
    try:
        n = conn.execute('SELECT COUNT(*) FROM raw_player_match_stats').fetchone()[0]
        print(f'raw_player_match_stats rows: {n}')
    except Exception as e:
        print(f'Table absent: {e}')
"
```

If the table is absent or empty, run Steps 1-2 first. If it already has rows (e.g. from a previous fetch-fotmob run in a worktree), skip to Step 3.

- [ ] **Step 1: Populate raw_player_match_stats (skip if already done)**

Run (takes ~20 minutes for one season due to per-match API calls with 1s delay):
```bash
python main.py fetch-fotmob --league E0 --from_season 2024 --to_season 2024 --delay 1.0
```

Expected final log lines:
```
INFO | Fetched <N> FotMob player-match rows total.
INFO | FotMob upsert | matched=<M> | unmatched=<U> | players=<P> | rows=<R>
```
`matched` should be close to `fetched` (some unmatched rows are expected for matches not yet in raw_matches). Typical: ~15,000 rows for one season.

For a full multi-season backfill (recommended before retraining):
```bash
python main.py fetch-fotmob --league E0 --from_season 2016 --to_season 2024 --delay 1.0
```

- [ ] **Step 2: Rebuild the feature store (includes SQUAD_* features now)**

```bash
python main.py ingest --force
```

Expected log: `Ingest complete | raw_matches=<N> | feature_store=<N>`

Verify SQUAD_* features are present in the rebuilt store:
```bash
python -c "
from src.utils.db_manager import DuckDBManager
db = DuckDBManager('config.yaml')
with db.connection(read_only=True) as conn:
    cols = [row[1] for row in conn.execute(\"PRAGMA table_info('feature_store')\").fetchall()]
    squad = [c for c in cols if c.startswith('SQUAD_')]
    print(f'SQUAD_* columns in feature_store: {squad}')
    non_null = conn.execute(\"SELECT COUNT(*) FROM feature_store WHERE SQUAD_HOME_XG_MEAN_R5 IS NOT NULL\").fetchone()[0]
    print(f'Rows with non-null SQUAD_HOME_XG_MEAN_R5: {non_null}')
"
```

Expected: 12 SQUAD_* columns present, non_null count > 0 (for matches that have at least 1 prior match with player data in the rolling window).

- [ ] **Step 3: Retrain competition_specific models for all 8 forecast targets**

```bash
python main.py train-forecast-suite --context league
```

This trains all 8 targets (`result_3way`, `btts`, `home_goals`, `away_goals`, `total_goals`, `home_corners`, `away_corners`, `total_corners`) with the full feature set including SQUAD_*. Each uses XGBoost (via the suite's default model selection), which handles the many NaN SQUAD_* values for early-season matches natively. Takes ~5-15 minutes total.

Expected log per target: `Feature subset active: <N>/171 features` (the 171 comes from 159 previous features + 12 SQUAD_* = 171 when SQUAD_* columns exist in feature_store).

- [ ] **Step 4: Select best models**

```bash
python main.py select-best-models --context league
```

Expected: `config/model_selection.yaml` updated under `contexts.league` with new model paths dated today.

- [ ] **Step 5: Verify a feature importance metadata file includes SQUAD_* entries**

```bash
python -c "
import json, glob
metafiles = glob.glob('models/*_xgboost*_$(date +%Y%m%d)*.metadata.json')
if metafiles:
    meta = json.load(open(metafiles[0]))
    squad = [f for f in meta.get('feature_importance', []) if f['feature'].startswith('SQUAD_')]
    print(f'SQUAD_* in feature importance: {squad[:3]}')
else:
    print('No metadata file found — check models/ for today\\'s date')
"
```

Expected: at least some SQUAD_* entries with importance > 0 in the feature importance list.

---

### Task 4: Verify explainability surfaces SQUAD_* and update docs (US#99)

**Files:**
- Test: `tests/test_squad_features.py` (add one explainability smoke test)
- Modify: `documents/FRAI_TECHSPEC.md` (§27 status line)
- Modify: `documents/user_stories.md` (Phase 14c heading + US#96-99 statuses)

- [ ] **Step 1: Write and run the explainability unit test**

Add to `tests/test_squad_features.py`:

```python
def test_top_features_surfaces_squad_features_when_present() -> None:
    """ForecastService._top_features must include SQUAD_* when they have importance."""
    import sys
    from pathlib import Path as _Path
    sys.path.append(str(_Path(__file__).resolve().parents[1]))
    from src.forecast.forecast_service import ForecastService
    import pandas as pd

    row = pd.Series({
        "MKT_IMPLIED_HOME": 0.45,
        "SQUAD_HOME_XG_MEAN_R5": 0.72,
        "SQUAD_AWAY_RATING_MEAN_R5": 7.1,
        "OFF_HOME_FTHG_R5": 1.8,
    })
    metadata_by_target = {
        "home_goals": {
            "feature_importance": [
                {"feature": "SQUAD_HOME_XG_MEAN_R5", "importance": 0.15},
                {"feature": "MKT_IMPLIED_HOME", "importance": 0.10},
                {"feature": "SQUAD_AWAY_RATING_MEAN_R5", "importance": 0.08},
                {"feature": "OFF_HOME_FTHG_R5", "importance": 0.05},
            ]
        }
    }

    top = ForecastService._top_features(row, metadata_by_target, limit=4)

    names = [f["name"] for f in top]
    assert "SQUAD_HOME_XG_MEAN_R5" in names
    assert "SQUAD_AWAY_RATING_MEAN_R5" in names
    # Values must be correct from the row
    squad_entry = next(f for f in top if f["name"] == "SQUAD_HOME_XG_MEAN_R5")
    assert squad_entry["value"] == pytest.approx(0.72)
    assert squad_entry["importance"] == pytest.approx(0.15, abs=1e-5)
```

Run: `pytest tests/test_squad_features.py::test_top_features_surfaces_squad_features_when_present -v`
Expected: PASS (no code changes needed — ForecastService._top_features already handles arbitrary feature names)

- [ ] **Step 2: Run the full test suite one final time**

Run: `pytest -q`
Expected: 0 failures

- [ ] **Step 3: Manual end-to-end smoke test**

Run a spot forecast for a known Premier League fixture that is in the feature store:
```bash
python main.py forecast --home Arsenal --away Everton --date 2024-05-19 \
    --odds_h 1.70 --odds_d 3.80 --odds_a 4.50 --league E0
```

Check the `explainability.top_features` section of the JSON output. If any SQUAD_* feature ranks in the top 8 by importance from the retrained model, it should appear here automatically.

- [ ] **Step 4: Update `documents/FRAI_TECHSPEC.md` Section 27 status line**

Change:
```
**Status: Phase 14a and Phase 14b (US#87–95) implemented. Phase 14c remains planned.**
```

to:
```
**Status: Phase 14a, 14b, and 14c (US#87–99) fully implemented.** All three phases complete: competition registry (27.2), FotMob player ingestion (27.3), and SQUAD_* squad features with competition-gated model training (27.4).
```

- [ ] **Step 5: Update `documents/user_stories.md` Phase 14c section**

Change `### Phase 14c: Squad Feature Engineering & Model Integration` heading to `### Phase 14c: Squad Feature Engineering & Model Integration — Completed`

Update each US#96–99 status from `not started` to `completed`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_squad_features.py documents/FRAI_TECHSPEC.md documents/user_stories.md
git commit -m "feat: verify SQUAD_* explainability, mark Phase 14c complete (US#99)"
```
