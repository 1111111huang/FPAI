"""Tests for main.py's run_agent_batch CLI entry point (A18)."""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from main import _next_weekend, _picked_candidate, run_agent_batch


def _rec(overall, market=None, selection=None, value_edge=0.0):
    candidates = []
    pick = None
    if market is not None:
        candidates.append({"market": market, "selection": selection, "value_edge": value_edge})
        pick = {"market": market, "selection": selection}
    return {"overall": overall, "candidates": candidates, "recommendation_pick": pick}


def test_next_weekend_from_a_weekday_returns_the_upcoming_saturday_sunday():
    # Wednesday 2026-09-23
    saturday, sunday = _next_weekend(date(2026, 9, 23))
    assert (saturday, sunday) == (date(2026, 9, 26), date(2026, 9, 27))


def test_next_weekend_from_a_saturday_stays_on_the_same_weekend():
    saturday, sunday = _next_weekend(date(2026, 9, 26))
    assert (saturday, sunday) == (date(2026, 9, 26), date(2026, 9, 27))


def test_picked_candidate_finds_the_matching_candidate():
    rec = _rec("direct_bet", market="btts", selection="yes", value_edge=0.12)
    assert _picked_candidate(rec) == {"market": "btts", "selection": "yes", "value_edge": 0.12}


def test_picked_candidate_returns_none_when_no_pick():
    assert _picked_candidate(_rec("no_bet")) is None


def _fixture(match_id, home, away):
    return SimpleNamespace(match_id=match_id, home_team=home, away_team=away, utc_date="2026-09-26T15:00:00Z")


def test_run_agent_batch_ranks_best_value_first_and_skips_errors(capsys):
    fixtures = [
        _fixture("m1", "TeamA", "TeamB"),
        _fixture("m2", "TeamC", "TeamD"),
        _fixture("m3", "TeamE", "TeamF"),
        _fixture("m4", "TeamG", "TeamH"),
    ]
    recommendations = {
        "TeamA": _rec("direct_bet", "result_3way", "home", value_edge=0.05),
        "TeamC": _rec("direct_bet", "result_3way", "away", value_edge=0.20),
        "TeamE": _rec("no_bet"),
    }

    def fake_run_agent(match_info, config):
        home = match_info["home_team"]
        if home == "TeamG":
            raise RuntimeError("boom")
        return recommendations[home]

    with patch("app.backend.football_data_client.FootballDataClient") as MockClient, \
         patch("app.backend.scheduler_wiring.build_odds_client", return_value=None), \
         patch("src.agent.graph.run_agent", side_effect=fake_run_agent), \
         patch("src.agent.agent_config.AgentConfig") as MockConfig:
        MockClient.return_value.get_fixtures.return_value = fixtures
        MockConfig.default.return_value = "cfg"
        run_agent_batch(
            league="E0", weekend=False, from_date="2026-09-26", to_date="2026-09-27",
            config_path=None, concurrency=5,
        )

    out = capsys.readouterr().out
    order = [line.split(" -- ")[0] for line in out.splitlines() if " v " in line]
    # TeamC (edge 0.20) ranks above TeamA (edge 0.05, still direct_bet) which
    # ranks above TeamE (no_bet) -- TeamG never appears, it errored and was
    # skipped rather than aborting the whole batch.
    assert "TeamC" in order[0]
    assert "TeamA" in order[1]
    assert "TeamE" in order[2]
    assert len(order) == 3
    assert "3/4 evaluated, 1 skipped" in out


def test_run_agent_batch_rejects_missing_date_range(capsys):
    with pytest.raises(SystemExit):
        run_agent_batch(league="E0", weekend=False, from_date=None, to_date=None, config_path=None, concurrency=5)
    assert "Provide --weekend" in capsys.readouterr().err


def test_run_agent_batch_rejects_unsupported_league(capsys):
    with pytest.raises(SystemExit):
        run_agent_batch(league="SWE", weekend=True, from_date=None, to_date=None, config_path=None, concurrency=5)
    assert "no football-data.org fixture source" in capsys.readouterr().err
