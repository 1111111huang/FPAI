# Goal-Market Lambda Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `MKT_LAMBDA_HOME`/`MKT_LAMBDA_AWAY`'s NaN (today's behavior whenever the Asian Handicap line is missing at live-forecast time) with a calibrated, per-league, 1X2-implied-probability-based split of the already-real `MKT_LAMBDA_TOTAL` — live-serving only, no retrain.

**Architecture:** A new standalone module (`src/features/goal_market_fallback.py`) holds the pure split-math and a committed per-league coefficients table. `FeatureFactory.build_for_match` (the only call path for a genuinely upcoming, never-before-played match) calls it right after the existing AH/lambda computation, only when `MKT_LAMBDA_TOTAL` is real but `MKT_AH_LINE` is NaN, and stamps a new `MKT_LAMBDA_HOME_AWAY_IMPUTED` flag column. `forecast_service.py` surfaces that flag as `data_quality.lambda_home_away_imputed` and is corrected so its existing `goal_market_features_available` diagnostic keeps meaning "a real AH price was used" rather than being fooled by the new fallback. A one-time offline script fits the real per-league coefficients from `feature_store` history.

**Tech Stack:** Python, pandas, numpy, DuckDB (existing stack — no new dependencies).

**Reference spec:** `docs/superpowers/specs/2026-11-16-goal-market-lambda-fallback-design.md`

---

## Task 1: Pure fallback module

**Files:**
- Create: `src/features/goal_market_fallback.py`
- Test: `tests/test_goal_market_fallback.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_goal_market_fallback.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features.goal_market_fallback'`

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_goal_market_fallback.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/features/goal_market_fallback.py tests/test_goal_market_fallback.py
git commit -m "feat(features): add goal-market lambda NaN fallback (1/3)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 2: Calibration script

**Files:**
- Create: `scripts/calibrate_goal_market_fallback.py`
- Test: `tests/test_calibrate_goal_market_fallback.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for scripts/calibrate_goal_market_fallback.py -- see
docs/superpowers/specs/2026-11-16-goal-market-lambda-fallback-design.md."""
from __future__ import annotations

import contextlib
from pathlib import Path
import sys

import duckdb
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import calibrate_goal_market_fallback as script


def test_fit_home_share_coefficients_recovers_a_known_linear_relationship():
    # real_home_share = 0.54 + 0.2 * gap, exactly, for 5 synthetic rows.
    rows = []
    for gap in (-0.4, -0.2, 0.0, 0.2, 0.4):
        share = 0.54 + 0.2 * gap
        lambda_total = 2.6
        lambda_home = lambda_total * share
        implied_home = 0.4 + gap / 2.0
        implied_away = 0.4 - gap / 2.0
        rows.append((lambda_total, lambda_home, implied_home, implied_away))

    slope, intercept = script.fit_home_share_coefficients(rows)

    assert slope == pytest.approx(0.2, abs=1e-6)
    assert intercept == pytest.approx(0.54, abs=1e-6)


def test_fetch_rows_by_league_reads_only_rows_with_a_real_ah_line():
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE raw_matches (match_id VARCHAR, league VARCHAR)")
    conn.execute(
        'CREATE TABLE feature_store (match_id VARCHAR, "MKT_LAMBDA_TOTAL" FLOAT, '
        '"MKT_LAMBDA_HOME" FLOAT, "MKT_AH_LINE" FLOAT, "MKT_IMPLIED_HOME" FLOAT, '
        '"MKT_IMPLIED_AWAY" FLOAT)'
    )
    conn.execute("INSERT INTO raw_matches VALUES ('e1', 'E0'), ('e2', 'E0')")
    conn.execute(
        "INSERT INTO feature_store VALUES "
        "('e1', 2.6, 1.4, -0.5, 0.5, 0.3), "  # real AH line -> included
        "('e2', 2.6, 1.4, NULL, 0.5, 0.3)"  # missing AH line -> excluded
    )

    class _FakeDb:
        @contextlib.contextmanager
        def connection(self, read_only=False):
            yield conn

    rows_by_league = script._fetch_rows_by_league(_FakeDb())

    assert rows_by_league == {"E0": [(2.6, 1.4, 0.5, 0.3)]}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_calibrate_goal_market_fallback.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.calibrate_goal_market_fallback'`

- [ ] **Step 3: Write the implementation**

```python
"""2026-11-16: offline calibration for the goal-market lambda NaN fallback --
see docs/superpowers/specs/2026-11-16-goal-market-lambda-fallback-design.md.

Fits a per-league linear mapping from a match's own 1X2-implied-probability
gap (MKT_IMPLIED_HOME - MKT_IMPLIED_AWAY) to its real home goal-share
(MKT_LAMBDA_HOME / MKT_LAMBDA_TOTAL), using every feature_store row where a
real Asian Handicap line already exists -- i.e. where the true split is
known, so the fit can be checked against ground truth. The result is used by
FeatureFactory.build_for_match (src/features/feature_factory.py) only when
the AH line is genuinely missing, in place of propagating NaN.

Run against production's DB:

    python scripts/calibrate_goal_market_fallback.py

Paste the printed per-league (slope, intercept) pairs into
HOME_SHARE_COEFFICIENTS in src/features/goal_market_fallback.py, replacing
the interim flat-prior placeholder there.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.db_manager import DuckDBManager  # noqa: E402

# Below this many real-AH-line rows, keep the flat default rather than fit
# noise from too small a sample.
_MIN_ROWS_PER_LEAGUE = 30


def fit_home_share_coefficients(rows: list[tuple[float, float, float, float]]) -> tuple[float, float]:
    """rows: (lambda_total, lambda_home, implied_home, implied_away) tuples,
    all real (caller filters out any NaN/non-positive lambda_total first).
    Returns (slope, intercept) for
        real_home_share ~= intercept + slope * (implied_home - implied_away)
    fit by ordinary least squares (numpy.polyfit, degree 1)."""
    import numpy as np

    gaps = np.array([implied_home - implied_away for _, _, implied_home, implied_away in rows])
    shares = np.array([lambda_home / lambda_total for lambda_total, lambda_home, _, _ in rows])
    slope, intercept = np.polyfit(gaps, shares, deg=1)
    return float(slope), float(intercept)


def _fetch_rows_by_league(db_manager: DuckDBManager) -> dict[str, list[tuple[float, float, float, float]]]:
    """{league: [(lambda_total, lambda_home, implied_home, implied_away), ...]}
    -- only rows with a real MKT_AH_LINE (the true split is known for these,
    which is what makes them usable as calibration ground truth)."""
    rows_by_league: dict[str, list[tuple[float, float, float, float]]] = {}
    with db_manager.connection(read_only=True) as conn:
        query = """
            SELECT rm.league, fs."MKT_LAMBDA_TOTAL", fs."MKT_LAMBDA_HOME",
                   fs."MKT_IMPLIED_HOME", fs."MKT_IMPLIED_AWAY"
            FROM feature_store fs
            JOIN raw_matches rm ON fs.match_id = rm.match_id
            WHERE fs."MKT_AH_LINE" IS NOT NULL
              AND fs."MKT_LAMBDA_TOTAL" IS NOT NULL AND fs."MKT_LAMBDA_TOTAL" > 0
              AND fs."MKT_LAMBDA_HOME" IS NOT NULL
              AND fs."MKT_IMPLIED_HOME" IS NOT NULL AND fs."MKT_IMPLIED_AWAY" IS NOT NULL
        """
        for league, lambda_total, lambda_home, implied_home, implied_away in conn.execute(query).fetchall():
            rows_by_league.setdefault(league, []).append((lambda_total, lambda_home, implied_home, implied_away))
    return rows_by_league


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml", help="Config file that points at the DB.")
    args = parser.parse_args(argv)

    rows_by_league = _fetch_rows_by_league(DuckDBManager(config_path=args.config))
    if not rows_by_league:
        print("No feature_store rows with a real AH line found.")
        return 1

    print(f"{'league':<8} {'n_rows':>7} {'slope':>10} {'intercept':>10} {'MAE':>8} {'MAE@50/50':>10}")
    for league, rows in sorted(rows_by_league.items()):
        if len(rows) < _MIN_ROWS_PER_LEAGUE:
            print(f"{league:<8} {len(rows):>7}   -- too few rows, keep the flat default --")
            continue
        slope, intercept = fit_home_share_coefficients(rows)
        # Design doc §8: the fitted split must beat a naive flat 50/50 split
        # on this same data, or it isn't earning its complexity -- printed
        # here so Task 6 can eyeball it before accepting the fit (not a
        # fixed accept/reject threshold; a real per-league real-edge
        # check, against genuinely new live data, is the actual validation
        # and has to wait until after this ships).
        real_shares = [lambda_home / lambda_total for lambda_total, lambda_home, _, _ in rows]
        fitted_shares = [
            intercept + slope * (implied_home - implied_away)
            for _, _, implied_home, implied_away in rows
        ]
        fitted_mae = sum(abs(r - f) for r, f in zip(real_shares, fitted_shares)) / len(rows)
        baseline_mae = sum(abs(r - 0.5) for r in real_shares) / len(rows)
        print(
            f"{league:<8} {len(rows):>7} {slope:>10.4f} {intercept:>10.4f} "
            f"{fitted_mae:>8.4f} {baseline_mae:>10.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_calibrate_goal_market_fallback.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_goal_market_fallback.py tests/test_calibrate_goal_market_fallback.py
git commit -m "feat(scripts): add offline calibration script for the goal-market lambda fallback (2/3)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 3: Wire the fallback into `build_for_match`

**Files:**
- Modify: `src/features/feature_factory.py:1146-1147` (local imports), `:1416-1417` (insertion point), `:1468-1475` (dedupe `league_by_match_id`)
- Test: `tests/test_feature_factory.py` (new test appended)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_feature_factory.py`:

```python
def test_build_for_match_fills_lambda_home_away_when_ah_line_missing(tmp_path: Path) -> None:
    """2026-11-16: MKT_LAMBDA_HOME/AWAY must not stay NaN just because the
    Asian Handicap line hasn't posted yet -- see
    docs/superpowers/specs/2026-11-16-goal-market-lambda-fallback-design.md."""
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}),
        encoding="utf-8",
    )

    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE raw_matches (
                match_id TEXT PRIMARY KEY, league TEXT, tier INTEGER, date TIMESTAMP,
                home_team TEXT, away_team TEXT, fthg INTEGER, ftag INTEGER,
                hs FLOAT, "as" FLOAT, hst FLOAT, ast FLOAT, hc FLOAT, ac FLOAT,
                hy FLOAT, ay FLOAT, hr FLOAT, ar FLOAT,
                odds_h FLOAT, odds_d FLOAT, odds_a FLOAT,
                avgh FLOAT, avgd FLOAT, avga FLOAT,
                maxch FLOAT, maxcd FLOAT, maxca FLOAT,
                avgch FLOAT, avgcd FLOAT, avgca FLOAT,
                xg_h FLOAT, xg_a FLOAT, xga_h FLOAT, xga_a FLOAT,
                over25_odds FLOAT, under25_odds FLOAT,
                ah_line FLOAT, ah_home_odds FLOAT, ah_away_odds FLOAT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO raw_matches
            (match_id, league, tier, date, home_team, away_team, fthg, ftag, odds_h, odds_d, odds_a, avgh, avgd, avga)
            VALUES
            ('m1', 'E0', 1, '2025-08-10 20:00:00', 'Arsenal', 'Everton', 2, 1, 1.5, 4.0, 5.0, 1.5, 4.0, 5.0)
            """
        )
        conn.execute(
            "CREATE TABLE raw_player_match_stats (match_id TEXT, team_name TEXT, player_id BIGINT, "
            "xg FLOAT, xa FLOAT, rating FLOAT, goals INTEGER, assists INTEGER)"
        )
        conn.execute(
            "CREATE TABLE player_market_values (fotmob_player_id BIGINT, transfermarkt_player_id BIGINT, "
            "snapshot_date TEXT, market_value_eur BIGINT)"
        )

    feature_factory = FeatureFactory(config_path=str(config_path))
    row = feature_factory.build_for_match(
        home_team="Arsenal", away_team="Everton", match_date="2025-08-24",
        league="E0", odds_h=1.8, odds_d=3.6, odds_a=4.2,
        over25_odds=1.9,  # real O/U 2.5 price -> MKT_LAMBDA_TOTAL is real
        # ah_line/ah_home_odds/ah_away_odds deliberately omitted -- AH hasn't posted yet
    )

    assert not pd.isna(row["MKT_LAMBDA_TOTAL"].iloc[0])  # sanity: trigger condition is real
    assert pd.isna(row["MKT_AH_LINE"].iloc[0])  # sanity: AH is genuinely missing
    assert not pd.isna(row["MKT_LAMBDA_HOME"].iloc[0])
    assert not pd.isna(row["MKT_LAMBDA_AWAY"].iloc[0])
    assert not pd.isna(row["MKT_POISSON_BTTS_PROB"].iloc[0])
    assert bool(row["MKT_LAMBDA_HOME_AWAY_IMPUTED"].iloc[0]) is True
    lam_total = row["MKT_LAMBDA_TOTAL"].iloc[0]
    assert row["MKT_LAMBDA_HOME"].iloc[0] + row["MKT_LAMBDA_AWAY"].iloc[0] == pytest.approx(lam_total)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_feature_factory.py::test_build_for_match_fills_lambda_home_away_when_ah_line_missing -v`
Expected: FAIL — `MKT_LAMBDA_HOME` is NaN (and `MKT_LAMBDA_HOME_AWAY_IMPUTED` doesn't exist yet, `KeyError`).

- [ ] **Step 3: Write the implementation**

In `src/features/feature_factory.py`, `build_for_match`'s local-import block (around line 1146-1147), add the new import:

```python
        import json
        import numpy as np

        from src.features.goal_market_fallback import fallback_lambda_home_away
```

Then replace this block (around line 1416-1417):

```python
        odds_feats = self._compute_odds_features(combined)
        features = features.merge(odds_feats, on="match_id", how="left")
```

with:

```python
        odds_feats = self._compute_odds_features(combined)
        features = features.merge(odds_feats, on="match_id", how="left")

        # US#134 (moved up from below): computed here, not just where
        # cold-start imputation uses it, so the fallback right below can
        # also look up each row's own league.
        league_by_match_id = combined.set_index("match_id")["league"]
        match_league = features["match_id"].map(league_by_match_id)

        # 2026-11-16: MKT_LAMBDA_HOME/MKT_LAMBDA_AWAY are NaN whenever
        # MKT_AH_LINE is missing (no real AH price at generation time) --
        # see docs/superpowers/specs/2026-11-16-goal-market-lambda-fallback-design.md.
        # That NaN then reaches XGBoost-family models raw (BUG-096), whose
        # missing-value routing was never trained on real NaN examples
        # (feature_store is ~100% AH-populated) and was found to push hard
        # toward "over" regardless of real signal. Replace it with a
        # calibrated split of the already-real MKT_LAMBDA_TOTAL instead of
        # leaving it NaN. Live-serving only by construction: build_for_match
        # is never called for an already-played match (those hit the
        # feature_store cache lookup in forecast_service.py first) or from
        # the bulk training/replay path (compute_rolling_stats).
        needs_fallback = features["MKT_LAMBDA_TOTAL"].notna() & features["MKT_AH_LINE"].isna()
        features["MKT_LAMBDA_HOME_AWAY_IMPUTED"] = needs_fallback
        for idx in features.index[needs_fallback]:
            lam_home, lam_away = fallback_lambda_home_away(
                lambda_total=features.at[idx, "MKT_LAMBDA_TOTAL"],
                implied_home=features.at[idx, "MKT_IMPLIED_HOME"],
                implied_away=features.at[idx, "MKT_IMPLIED_AWAY"],
                league=match_league.at[idx],
            )
            features.at[idx, "MKT_LAMBDA_HOME"] = lam_home
            features.at[idx, "MKT_LAMBDA_AWAY"] = lam_away
            features.at[idx, "MKT_POISSON_BTTS_PROB"] = (
                (1.0 - np.exp(-lam_home)) * (1.0 - np.exp(-lam_away))
            )
```

Then, further down (around the old line 1468-1475), remove the now-duplicate `league_by_match_id` computation and reuse `match_league`:

Replace:

```python
        # US#134: group by league (see _apply_cold_start_imputation docstring).
        # combined's "league" column covers each historical row's own
        # competition plus the synthetic row's target league, so a team-name
        # collision across competitions can't cross-contaminate fill values.
        league_by_match_id = combined.set_index("match_id")["league"]
        features = self._apply_cold_start_imputation(
            features, league=features["match_id"].map(league_by_match_id)
        )
```

with:

```python
        # US#134: group by league (see _apply_cold_start_imputation
        # docstring) -- match_league was already computed above, right
        # after the odds_feats merge, for the lambda fallback to use too.
        features = self._apply_cold_start_imputation(features, league=match_league)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_feature_factory.py::test_build_for_match_fills_lambda_home_away_when_ah_line_missing -v`
Expected: PASS

- [ ] **Step 5: Run the full feature_factory test suite to check for regressions**

Run: `python -m pytest tests/test_feature_factory.py -v`
Expected: PASS (all tests, including the pre-existing `test_build_for_match_includes_squad_and_luck_columns` and `test_build_for_match_includes_dixon_coles_columns`, which don't pass `over25_odds` and so never trigger `needs_fallback`)

- [ ] **Step 6: Commit**

```bash
git add src/features/feature_factory.py tests/test_feature_factory.py
git commit -m "feat(features): wire the goal-market lambda fallback into build_for_match (3/3)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 4: Keep `goal_market_features_available` correct, add `lambda_home_away_imputed`

**Files:**
- Modify: `src/forecast/forecast_service.py:101` (`_GOAL_MARKET_FEATURES`), `:117-128` (`_goal_market_features_available` docstring), add new `_lambda_home_away_imputed` function nearby, `:811` and `:877-887` (`forecast_upcoming`'s `data_quality` dict)
- Modify test: `tests/test_bug096_goal_market_inputs.py:149-156` (existing test, update)
- Test: `tests/test_bug096_goal_market_inputs.py` (new test appended)

- [ ] **Step 1: Write the failing tests**

In `tests/test_bug096_goal_market_inputs.py`, replace the existing test (lines 149-156):

```python
def test_goal_market_features_available_needs_all_three_real_values():
    full = pd.DataFrame([{"MKT_IMPLIED_OVER25": 0.6, "MKT_LAMBDA_HOME": 1.8, "MKT_LAMBDA_AWAY": 1.0}])
    missing = full.assign(MKT_LAMBDA_HOME=np.nan)
    absent = full.drop(columns=["MKT_LAMBDA_AWAY"])

    assert _goal_market_features_available(full) is True
    assert _goal_market_features_available(missing) is False
    assert _goal_market_features_available(absent) is False
```

with:

```python
def test_goal_market_features_available_needs_over25_and_ah_line():
    """2026-11-16: checks MKT_AH_LINE directly now, not MKT_LAMBDA_HOME/AWAY
    -- those two can now be a calibrated fallback value even when AH is
    genuinely missing (src/features/goal_market_fallback.py), so checking
    them here would wrongly report True for a fallback-derived row."""
    full = pd.DataFrame([{"MKT_IMPLIED_OVER25": 0.6, "MKT_AH_LINE": -0.5}])
    missing = full.assign(MKT_AH_LINE=np.nan)
    absent = full.drop(columns=["MKT_AH_LINE"])

    assert _goal_market_features_available(full) is True
    assert _goal_market_features_available(missing) is False
    assert _goal_market_features_available(absent) is False


def test_lambda_home_away_imputed_reads_the_flag_column():
    flagged = pd.DataFrame([{"MKT_LAMBDA_HOME_AWAY_IMPUTED": True}])
    clean = pd.DataFrame([{"MKT_LAMBDA_HOME_AWAY_IMPUTED": False}])
    absent = pd.DataFrame([{"MKT_LAMBDA_HOME": 1.4}])

    assert _lambda_home_away_imputed(flagged) is True
    assert _lambda_home_away_imputed(clean) is False
    assert _lambda_home_away_imputed(absent) is False
```

And update the import block near the top of the same file:

```python
from src.forecast.forecast_service import (
    ForecastService,
    _goal_market_features_available,
    _lambda_home_away_imputed,
    _market_nan_frame,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bug096_goal_market_inputs.py -v`
Expected: FAIL — `test_goal_market_features_available_needs_over25_and_ah_line` fails (old tuple still checks `MKT_LAMBDA_HOME`/`AWAY`), `test_lambda_home_away_imputed_reads_the_flag_column` fails with `ImportError: cannot import name '_lambda_home_away_imputed'`.

- [ ] **Step 3: Write the implementation**

In `src/forecast/forecast_service.py`, replace line 101:

```python
_GOAL_MARKET_FEATURES = ("MKT_IMPLIED_OVER25", "MKT_LAMBDA_HOME", "MKT_LAMBDA_AWAY")
```

with:

```python
# 2026-11-16: checks MKT_AH_LINE directly, not MKT_LAMBDA_HOME/AWAY -- those
# two can now be a calibrated fallback value (src/features/goal_market_fallback.py)
# even when AH is genuinely missing, so checking them here would wrongly
# report "available" for a fallback-derived row. MKT_AH_LINE itself is never
# given a fallback, so it stays the correct signal for "a real AH price was
# actually used."
_GOAL_MARKET_FEATURES = ("MKT_IMPLIED_OVER25", "MKT_AH_LINE")
```

Then add a new function right after `_goal_market_features_available` (after line 128):

```python
def _lambda_home_away_imputed(feature_row: pd.DataFrame) -> bool:
    """True when MKT_LAMBDA_HOME/MKT_LAMBDA_AWAY were filled by the
    calibrated AH-missing fallback (src/features/goal_market_fallback.py)
    rather than a real AH-derived split. The flag column is absent entirely
    on the 'international'/market-odds-only path, which never runs the
    fallback (ForecastService._compute_mkt_features_from_odds) -- treated
    as not-imputed in that case, same as every other MKT_* diagnostic on
    that path."""
    if "MKT_LAMBDA_HOME_AWAY_IMPUTED" not in feature_row.columns:
        return False
    return bool(feature_row["MKT_LAMBDA_HOME_AWAY_IMPUTED"].iloc[0])
```

Then in `forecast_upcoming`, change line 811:

```python
        goal_market_features_available = _goal_market_features_available(feature_row)
```

to:

```python
        goal_market_features_available = _goal_market_features_available(feature_row)
        lambda_home_away_imputed = _lambda_home_away_imputed(feature_row)
```

And in the `data_quality` dict (around line 877-887), change:

```python
            "data_quality": {
                "prediction_basis": prediction_basis,
                "feature_count": int(feature_count),
                "unknown_team": unknown_team,
                "caveat": caveat,
                # BUG-096: False means no real over/under 2.5 or Asian
                # handicap price reached this forecast, so every goal target
                # (and result_3way, built from them) scored on placeholder
                # market features. Informational only -- nothing blocks on it.
                "goal_market_features_available": goal_market_features_available,
            },
```

to:

```python
            "data_quality": {
                "prediction_basis": prediction_basis,
                "feature_count": int(feature_count),
                "unknown_team": unknown_team,
                "caveat": caveat,
                # BUG-096: False means no real over/under 2.5 or Asian
                # handicap price reached this forecast, so every goal target
                # (and result_3way, built from them) scored on placeholder
                # market features. Informational only -- nothing blocks on it.
                "goal_market_features_available": goal_market_features_available,
                # 2026-11-16: True when MKT_LAMBDA_HOME/MKT_LAMBDA_AWAY came
                # from the calibrated AH-missing fallback rather than a real
                # AH-derived split -- see
                # docs/superpowers/specs/2026-11-16-goal-market-lambda-fallback-design.md.
                # Informational only, same as the field above.
                "lambda_home_away_imputed": lambda_home_away_imputed,
            },
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bug096_goal_market_inputs.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the full forecast_service test suite to check for regressions**

Run: `python -m pytest tests/test_forecast_service_feature_importance.py tests/test_forecast_service_shap.py tests/test_forecast_service_composite_model_loading.py tests/test_forecast_service_calibration.py tests/test_forecast_service_cached_features.py tests/test_forecast_service_quantile_interval.py tests/test_forecast_service_training_metrics.py -v`
Expected: PASS (no other test reads `_GOAL_MARKET_FEATURES` or asserts on the old 3-tuple)

- [ ] **Step 6: Commit**

```bash
git add src/forecast/forecast_service.py tests/test_bug096_goal_market_inputs.py
git commit -m "fix(forecast): keep goal_market_features_available correct under the new lambda fallback, add lambda_home_away_imputed diagnostic

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 5: Run the full regression suite

**Files:** None (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ app/backend/tests/ -q`
Expected: PASS — same pass/fail counts as `main` before this branch, plus the new tests added in Tasks 1-4. If any pre-existing failure appears that isn't already known/unrelated (check `documents/bugs.md` for a matching known-failure note before assuming this plan caused it), stop and investigate before continuing to Task 6.

---

## Task 6: Calibrate against real data (requires production DB access)

**Files:**
- Modify: `src/features/goal_market_fallback.py` (`HOME_SHARE_COEFFICIENTS`)

This task needs read access to the real `feature_store`/`raw_matches` tables (production, or a local replica with real historical data synced — not the placeholder/empty local DB) and cannot be completed from an environment without that access.

- [ ] **Step 1: Run the calibration script against the real database**

```bash
python scripts/calibrate_goal_market_fallback.py --config config.yaml
```

(Point `--config` at whatever config file resolves to the real DB in the environment this is run from — production's own `config.yaml`, or a local one pointed at a synced copy.)

- [ ] **Step 2: Check the printed MAE column before accepting any league's fit**

Expected output shape (per-league row, real numbers will differ):

```
league    n_rows      slope  intercept      MAE  MAE@50/50
D1          1200     0.1xxx     0.5xxx   0.0xxx     0.0xxx
E0          1200     0.1xxx     0.5xxx   0.0xxx     0.0xxx
F1          1200     0.1xxx     0.5xxx   0.0xxx     0.0xxx
I1          1200     0.1xxx     0.5xxx   0.0xxx     0.0xxx
SP1         1200     0.1xxx     0.5xxx   0.0xxx     0.0xxx
```

Per the design doc's §8 testing goal, a league's fitted `MAE` column should
be meaningfully lower than its `MAE@50/50` column — if a league's fit
*doesn't* clear that bar, keep `(0.0, <that league's own real_home_share
mean>)` for it instead of the fitted slope (a flat, better-than-naive prior
beats a fit that isn't earning its complexity). If any league instead
prints `-- too few rows, keep the flat default --` (fewer than 30 real-AH-line
rows), leave that league's entry in `HOME_SHARE_COEFFICIENTS` as `(0.0, 0.54)`
rather than fitting noise. Note which league(s) got which treatment in the
commit message.

- [ ] **Step 3: Replace the placeholder coefficients**

In `src/features/goal_market_fallback.py`, replace the `HOME_SHARE_COEFFICIENTS` dict's placeholder values with the script's real per-league output, and delete the "INTERIM PLACEHOLDER" comment above it (replace with a note of when/against what data it was calibrated):

```python
# Per-league (slope, intercept) for
#   real_home_share ~= intercept + slope * (MKT_IMPLIED_HOME - MKT_IMPLIED_AWAY)
# Calibrated <DATE> by scripts/calibrate_goal_market_fallback.py against
# production feature_store (<N> total rows with a real AH line across the
# 5 leagues). Re-run and update if feature_store's AH-line coverage grows
# substantially (more real rows -> a more reliable fit) -- not on any fixed
# schedule.
HOME_SHARE_COEFFICIENTS: dict[str, tuple[float, float]] = {
    "E0": (<real_slope>, <real_intercept>),
    "SP1": (<real_slope>, <real_intercept>),
    "I1": (<real_slope>, <real_intercept>),
    "D1": (<real_slope>, <real_intercept>),
    "F1": (<real_slope>, <real_intercept>),
}
```

- [ ] **Step 4: Re-run Task 1's tests to confirm they're unaffected**

Run: `python -m pytest tests/test_goal_market_fallback.py -v`
Expected: PASS — these tests all pass their own `coefficients=` argument, so they're unaffected by changing the module-level default.

- [ ] **Step 5: Commit**

```bash
git add src/features/goal_market_fallback.py
git commit -m "chore(features): calibrate the goal-market lambda fallback against real feature_store data

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Not a task here: post-ship validation

The design doc's §8 fourth check — re-running the per-league real-edge
check restricted to picks where `lambda_home_away_imputed=true`, to confirm
hit rate moves closer to fair than today's observed coin-flip-with-over-skew
— needs genuinely new live picks made *after* this ships (the whole point is
checking real outcomes against the new fallback, not the old NaN behavior).
Not a task in this plan; revisit once a week or two of live data has
accumulated post-deploy, the same way the original investigation pulled the
week of Nov 1/Nov 7 pick-review artifacts.

## Explicitly out of scope (per the design doc's non-goals)

- `ForecastService._compute_mkt_features_from_odds` (the 'international'/
  general_purpose-competition path) is untouched — it has no `league`
  parameter to key per-league coefficients off, and none of the 5 domestic
  leagues this bug was found in route through it.
- No changes to `feature_factory.py`'s bulk `compute_rolling_stats()`/
  `_compute_odds_features()` training path, or to `MKT_LAMBDA_AH_DIFF`/
  `MKT_AH_LINE`'s own NaN behavior.
- No `team_totals`-odds-based feature_store backfill/retrain, and no trained
  "this value is imputed" model feature — both logged as separate follow-up
  work in the design doc, not part of this plan.
