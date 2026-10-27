"""BUG-096 (2026-10-26): live forecasts never received over/under 2.5 or
Asian handicap odds, so every goal-market feature (MKT_IMPLIED_OVER25,
MKT_AH_*, MKT_LAMBDA_*, MKT_POISSON_BTTS_PROB) was a placeholder: the
league's feature_store mean, or a flat 0.0 in E0/SP1 where that mean was
NULL. A 0.0 lambda reads as "the market expects no goals" and pushed
team-goal models toward under 1.5 (team-goal direct bets went 5/21 live).

Covers the four parts of the fix:
1. forecast_node passes real O/U 2.5 and AH odds from match_info (live only).
2. Missing MKT_* values stay NaN for NaN-trained (XGBoost-family) models.
3. The market-only path's goal split is signed (BUG-075's fix, ported).
4. (Removed 2026-10-27, by request) a feature-based guardrail on goal
   bets. Missing live odds now raise a Slack alert instead -- see
   app/backend/tests/test_missing_odds_alert.py.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agent import tools as agent_tools
from src.agent.pipeline import forecast_node, goal_market_odds_kwargs
from src.forecast.forecast_service import (
    ForecastService,
    _goal_market_features_available,
    _market_nan_frame,
)


@pytest.fixture(autouse=True)
def reset_snapshot_store():
    agent_tools._snapshot_store.set_mode("live")
    agent_tools._snapshot_store.set_tool_mode_overrides({})
    yield
    agent_tools._snapshot_store.set_mode("live")
    agent_tools._snapshot_store.set_tool_mode_overrides({})


_MATCH_INFO = {
    "home_team": "Liverpool", "away_team": "Brighton", "date": "2026-10-25", "league": "E0",
    "odds": {"home": 1.5, "draw": 4.5, "away": 6.0},
    "total_goals_odds": {"over_2.5": 1.65, "under_2.5": 2.25},
    "asian_handicap_odds": {"line": -1.25, "home": 1.95, "away": 1.95},
}


def _state(match_info: dict) -> dict:
    return {
        "messages": [],
        "match_info": match_info,
        "competition_resolution": {"competition": "E0", "tier": "competition_specific", "recommended_tool": "forecast_league"},
        "research_evidence": None,
    }


# --- 1. forecast_node threads goal-market odds -----------------------------


def test_goal_market_odds_kwargs_reads_over25_and_asian_handicap():
    assert goal_market_odds_kwargs(_MATCH_INFO) == {
        "over25_odds": 1.65, "ah_line": -1.25, "ah_home_odds": 1.95, "ah_away_odds": 1.95,
    }


def test_goal_market_odds_kwargs_omits_missing_markets():
    """A missing market must stay missing (NaN downstream), never be sent
    as an explicit None or 0."""
    assert goal_market_odds_kwargs({"odds": {"home": 2.0, "draw": 3.0, "away": 3.5}}) == {}
    partial_ah = {"asian_handicap_odds": {"line": -0.5, "home": None, "away": 1.9}}
    assert goal_market_odds_kwargs(partial_ah) == {}


def test_goal_market_odds_kwargs_keeps_a_level_handicap_line():
    """A pick'em line is 0.0 -- a real value, not a missing one."""
    info = {"asian_handicap_odds": {"line": 0.0, "home": 1.9, "away": 1.9}}
    assert goal_market_odds_kwargs(info)["ah_line"] == 0.0


def test_forecast_node_live_passes_goal_market_odds_to_forecast_upcoming():
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = {"result_3way": {}, "data_quality": {}}

        forecast_node(_state(_MATCH_INFO))

    kwargs = instance.forecast_upcoming.call_args.kwargs
    assert kwargs["over25_odds"] == 1.65
    assert kwargs["ah_line"] == -1.25
    assert kwargs["ah_home_odds"] == 1.95
    assert kwargs["ah_away_odds"] == 1.95


def test_forecast_node_outside_live_mode_keeps_the_original_snapshot_kwargs():
    """Replay/record runs score already-played matches from feature_store,
    and the extra kwargs would change every recorded snapshot key."""
    captured = {}

    def _fake_wrap(tool, fn):
        def wrapped(**kwargs):
            captured.update(kwargs)
            return json.dumps({"result_3way": {}, "data_quality": {}})
        return wrapped

    fake_store = MagicMock()
    fake_store.effective_mode.return_value = "replay"
    fake_store.wrap.side_effect = _fake_wrap
    with patch("src.agent.tools.get_snapshot_store", return_value=fake_store):
        forecast_node(_state(_MATCH_INFO))

    assert set(captured) == {"home_team", "away_team", "date", "league", "odds_h", "odds_d", "odds_a"}


def test_forecast_league_impl_forwards_goal_market_odds():
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = {"ok": True}

        agent_tools._forecast_league_impl(
            home_team="A", away_team="B", date="2026-10-25", league="E0",
            odds_h=2.0, odds_d=3.4, odds_a=3.6, over25_odds=1.8, ah_line=-0.25,
            ah_home_odds=1.9, ah_away_odds=2.0,
        )

    kwargs = instance.forecast_upcoming.call_args.kwargs
    assert (kwargs["over25_odds"], kwargs["ah_line"]) == (1.8, -0.25)
    assert (kwargs["ah_home_odds"], kwargs["ah_away_odds"]) == (1.9, 2.0)


# --- 2. missing MKT_* values stay NaN --------------------------------------


def test_market_nan_frame_restores_only_market_columns():
    unimputed = pd.DataFrame([{"MKT_LAMBDA_HOME": np.nan, "MKT_IMPLIED_HOME": 0.6, "OFF_HOME_XG_R5": np.nan}])
    imputed = pd.DataFrame([{"MKT_LAMBDA_HOME": 0.0, "MKT_IMPLIED_HOME": 0.6, "OFF_HOME_XG_R5": 1.4}])

    frame = _market_nan_frame(imputed, unimputed)

    assert pd.isna(frame.loc[0, "MKT_LAMBDA_HOME"])
    assert frame.loc[0, "MKT_IMPLIED_HOME"] == 0.6
    assert frame.loc[0, "OFF_HOME_XG_R5"] == 1.4  # non-market imputation is unchanged


def test_goal_market_features_available_needs_all_three_real_values():
    full = pd.DataFrame([{"MKT_IMPLIED_OVER25": 0.6, "MKT_LAMBDA_HOME": 1.8, "MKT_LAMBDA_AWAY": 1.0}])
    missing = full.assign(MKT_LAMBDA_HOME=np.nan)
    absent = full.drop(columns=["MKT_LAMBDA_AWAY"])

    assert _goal_market_features_available(full) is True
    assert _goal_market_features_available(missing) is False
    assert _goal_market_features_available(absent) is False


class XGBRegressor:  # name matters: matches _NAN_TRAINED_MODEL_CLASSES
    pass


class RandomForestRegressor:
    pass


def test_score_targets_gives_nan_frame_only_to_nan_trained_models():
    service = ForecastService.__new__(ForecastService)
    imputed = pd.DataFrame([{"MKT_LAMBDA_HOME": 0.0}])
    preferred = pd.DataFrame([{"MKT_LAMBDA_HOME": np.nan}])
    seen: dict[str, pd.DataFrame] = {}

    def _fake_predict(definition, model, metadata, frame):
        seen[metadata["target"]] = frame
        return {"expected": 1.0}

    loaded = {
        "home_goals": (MagicMock(), XGBRegressor(), {"target": "home_goals"}),
        "total_goals": (MagicMock(), RandomForestRegressor(), {"target": "total_goals"}),
    }
    with patch.object(service, "_predict_target", side_effect=_fake_predict):
        service._score_targets(imputed, loaded, imputed.iloc[0], ["MKT_LAMBDA_HOME"], preferred_frame=preferred)

    assert pd.isna(seen["home_goals"].loc[0, "MKT_LAMBDA_HOME"])
    assert seen["total_goals"].loc[0, "MKT_LAMBDA_HOME"] == 0.0


def test_score_targets_falls_back_to_imputed_frame_when_a_model_rejects_nan():
    service = ForecastService.__new__(ForecastService)
    imputed = pd.DataFrame([{"MKT_LAMBDA_HOME": 0.0}])
    preferred = pd.DataFrame([{"MKT_LAMBDA_HOME": np.nan}])
    calls = []

    def _fake_predict(definition, model, metadata, frame):
        calls.append(frame)
        if frame.isna().any().any():
            raise ValueError("Input X contains NaN")
        return {"expected": 1.0}

    loaded = {"home_goals": (MagicMock(), XGBRegressor(), {"target": "home_goals"})}
    with patch.object(service, "_predict_target", side_effect=_fake_predict):
        forecast, _ = service._score_targets(
            imputed, loaded, imputed.iloc[0], ["MKT_LAMBDA_HOME"], preferred_frame=preferred,
        )

    assert forecast["home_goals"] == {"expected": 1.0}
    assert len(calls) == 2


# --- 3. signed goal split on the market-only path --------------------------


def test_market_only_goal_split_credits_the_favored_side():
    """BUG-075's fix, ported: a positive line means the away side is
    favored, so the away lambda must be the larger one."""
    away_favored = ForecastService._compute_mkt_features_from_odds(
        6.0, 4.3, 1.55, over25_odds=1.7, ah_line=1.0, ah_home_odds=1.9, ah_away_odds=1.95,
    ).iloc[0]
    home_favored = ForecastService._compute_mkt_features_from_odds(
        1.55, 4.3, 6.0, over25_odds=1.7, ah_line=-1.0, ah_home_odds=1.9, ah_away_odds=1.95,
    ).iloc[0]

    assert away_favored["MKT_LAMBDA_AWAY"] > away_favored["MKT_LAMBDA_HOME"]
    assert home_favored["MKT_LAMBDA_HOME"] > home_favored["MKT_LAMBDA_AWAY"]
    total = away_favored["MKT_LAMBDA_TOTAL"]
    assert away_favored["MKT_LAMBDA_HOME"] + away_favored["MKT_LAMBDA_AWAY"] == pytest.approx(total)
    assert away_favored["MKT_LAMBDA_AH_DIFF"] == pytest.approx(total - 1.0)


def test_market_only_goal_split_stays_nan_without_a_line():
    row = ForecastService._compute_mkt_features_from_odds(2.0, 3.4, 3.6, over25_odds=1.8).iloc[0]

    assert not pd.isna(row["MKT_LAMBDA_TOTAL"])
    assert pd.isna(row["MKT_LAMBDA_HOME"])
    assert pd.isna(row["MKT_POISSON_BTTS_PROB"])


# --- diagnostic script -----------------------------------------------------


def test_coverage_script_flags_a_league_missing_ah_lambdas(capsys):
    """Mirrors what production looked like: E0's raw ah_line is present but
    feature_store's MKT_LAMBDA_HOME is all NULL; I1 is fine."""
    import contextlib

    import duckdb

    from scripts import check_market_feature_coverage as script

    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE raw_matches (match_id VARCHAR, league VARCHAR, date DATE, over25_odds FLOAT, ah_line FLOAT, ah_home_odds FLOAT, ah_away_odds FLOAT)")
    conn.execute('CREATE TABLE feature_store (match_id VARCHAR, "MKT_IMPLIED_OVER25" FLOAT, "MKT_AH_LINE" FLOAT, "MKT_LAMBDA_HOME" FLOAT, "MKT_LAMBDA_AWAY" FLOAT)')
    conn.execute("INSERT INTO raw_matches VALUES ('e1','E0','2026-01-01',1.9,-0.5,1.9,1.95), ('i1','I1','2026-01-01',1.9,-0.5,1.9,1.95)")
    conn.execute("INSERT INTO feature_store VALUES ('e1',0.52,-0.5,NULL,NULL), ('i1',0.52,-0.5,1.6,1.2)")

    class _FakeDb:
        @contextlib.contextmanager
        def connection(self, read_only=False):
            yield conn

    coverage = script.coverage_by_league(_FakeDb())
    assert coverage["E0"]["raw"]["ah_line"] == 1.0
    assert coverage["E0"]["features"]["MKT_LAMBDA_HOME"] == 0.0
    assert coverage["I1"]["features"]["MKT_LAMBDA_HOME"] == 1.0

    with patch.object(script, "DuckDBManager", return_value=_FakeDb()):
        exit_code = script.main([])
    out = capsys.readouterr().out
    assert exit_code == 2
    assert "E0: raw ah_line is fine but feature_store MKT_LAMBDA_HOME coverage is 0%" in out
    assert "I1:" not in out.split("Problems")[1]


# --- backfill script -------------------------------------------------------


def test_backfill_fills_only_null_goal_market_columns_and_never_inserts(tmp_path, capsys):
    """Production E0/SP1 rows were stored without ah_line, and CSVLoader's
    INSERT OR IGNORE + processed-file skip means a later CSV never fills
    them. The backfill must fill NULLs only: no overwrite, no new rows."""
    import duckdb
    import yaml

    from scripts import backfill_goal_market_odds as backfill
    from src.ingestion.football_data.loader import CSVLoader

    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "prod.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path), "raw_data_dir": str(raw_dir)}}),
        encoding="utf-8",
    )

    # 1) How the production row got there: an older CSV with no AH columns.
    old_csv = tmp_path / "old_E0.csv"
    old_csv.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,Avg>2.5,Avg<2.5\n"
        "15/08/2025,Liverpool,Bournemouth,4,2,1.3,6.0,8.5,1.35,3.20\n",
        encoding="utf-8",
    )
    CSVLoader(config_path=str(config_path)).process_v1_csv(file_path=str(old_csv), league_code="E0")

    # 2) The CSV now on disk has AH odds, a different over25 price, and a
    #    match the real table doesn't have yet.
    csv_dir = raw_dir / "football_data"
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "E0_2526.csv").write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,Avg>2.5,Avg<2.5,AHh,AvgAHH,AvgAHA\n"
        "15/08/2025,Liverpool,Bournemouth,4,2,1.3,6.0,8.5,1.40,3.10,-1.25,1.92,1.98\n"
        "16/08/2025,Aston Villa,Newcastle,0,0,2.25,3.5,3.1,1.61,2.15,0.0,1.92,1.90\n",
        encoding="utf-8",
    )

    def _rows():
        with duckdb.connect(str(db_path)) as conn:
            return conn.execute(
                "SELECT home_team, over25_odds, ah_line, ah_home_odds, ah_away_odds FROM raw_matches ORDER BY home_team"
            ).fetchall()

    assert backfill.main(["--leagues", "E0", "--config", str(config_path)]) == 0
    assert "Dry run: 3 NULL value(s) would be filled" in capsys.readouterr().out
    assert _rows()[0][2] is None  # dry run wrote nothing

    assert backfill.main(["--leagues", "E0", "--config", str(config_path), "--apply"]) == 0
    rows = _rows()
    assert len(rows) == 1  # Aston Villa v Newcastle was NOT inserted
    home, over25, ah_line, ah_home, ah_away = rows[0]
    assert home == "Liverpool"
    assert over25 == pytest.approx(1.35)  # existing value kept, not overwritten by 1.40
    assert ah_line == pytest.approx(-1.25)
    assert (ah_home, ah_away) == (pytest.approx(1.92), pytest.approx(1.98))
