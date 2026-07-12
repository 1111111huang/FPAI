"""W09: EOD scheduled batch recommendation generation (D2a). Fetches
tomorrow's E0 fixtures (W05) and current odds (W07), generates
recommendations concurrently using the bounded-semaphore,
skip-and-continue-on-error pattern from main.py's _run_backtest_concurrent
(agent_techspec.md §13), and writes each into W11's cache. Also schedules
each fixture's T-30 job (W10) -- for every fixture, regardless of whether
EOD generation itself succeeded, since W10's own refresh is independently
best-effort."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.agent_config_hash import compute_agent_config_hash
from app.backend.eod_batch import run_eod_batch
from app.backend.football_data_client import NormalizedMatch
from app.backend.odds_api_client import NormalizedOdds
from app.backend.recommendation_cache import RecommendationCache
from src.agent.agent_config import AgentConfig

_RECOMMENDATION = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
    "overall": "direct_bet",
    "markets": [{"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.1}],
    "explanation": "test", "confidence": "medium", "limitations": [], "prediction_basis": "team_history_and_market",
}


def _fixture(match_id: str, home: str, away: str) -> NormalizedMatch:
    return NormalizedMatch(
        match_id=match_id, utc_date="2026-08-22T15:00:00Z", status="SCHEDULED",
        home_team=home, away_team=away, home_goals=None, away_goals=None,
    )


def test_generates_a_recommendation_and_schedules_t30_for_every_fixture(tmp_path: Path) -> None:
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [
        _fixture("m1", "Arsenal", "Everton"),
        _fixture("m2", "Chelsea", "Fulham"),
        _fixture("m3", "Liverpool", "Wolves"),
    ]
    odds_client = MagicMock()
    odds_client.get_odds.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    scheduled: list[NormalizedMatch] = []

    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION):
        result = asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=scheduled.append, date_str="2026-08-22",
            )
        )

    assert result.generated == 3
    assert result.skipped == 0
    assert len(scheduled) == 3
    agent_config_hash = compute_agent_config_hash(config)
    for match_id in ("m1", "m2", "m3"):
        assert cache.get_latest(match_id, "2026-08-22", agent_config_hash) is not None


def test_one_erroring_match_is_skipped_not_fatal(tmp_path: Path) -> None:
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [
        _fixture("m1", "Arsenal", "Everton"),
        _fixture("m2", "Chelsea", "Fulham"),
    ]
    odds_client = MagicMock()
    odds_client.get_odds.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    scheduled: list[NormalizedMatch] = []

    def _run_agent_side_effect(match_info, config):
        if match_info["home_team"] == "Chelsea":
            raise RuntimeError("LLM timeout")
        return _RECOMMENDATION

    with patch("app.backend.recommendations.run_agent", side_effect=_run_agent_side_effect):
        result = asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=scheduled.append, date_str="2026-08-22",
            )
        )

    assert result.generated == 1
    assert result.skipped == 1
    # T-30 is still scheduled for both fixtures -- best-effort, independent of EOD success
    assert len(scheduled) == 2


def test_odds_matched_to_fixture_by_team_name(tmp_path: Path) -> None:
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [_fixture("m1", "Arsenal", "Everton")]
    odds_client = MagicMock()
    odds_client.get_odds.return_value = [
        NormalizedOdds(
            home_team="Arsenal", away_team="Everton", commence_time="2026-08-22T15:00:00Z",
            home_odds=1.8, draw_odds=3.6, away_odds=4.5,
        ),
    ]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    captured_match_info = {}

    def _capture(match_info, config):
        captured_match_info.update(match_info)
        return _RECOMMENDATION

    with patch("app.backend.recommendations.run_agent", side_effect=_capture):
        asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda f: None, date_str="2026-08-22",
            )
        )

    assert captured_match_info["odds"] == {"home": 1.8, "draw": 3.6, "away": 4.5}


def test_unmatched_odds_proceeds_with_no_odds_rather_than_skipping(tmp_path: Path) -> None:
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [_fixture("m1", "Arsenal", "Everton")]
    odds_client = MagicMock()
    odds_client.get_odds.return_value = [
        NormalizedOdds(
            home_team="Some Other Team", away_team="Another Team", commence_time="2026-08-22T15:00:00Z",
            home_odds=1.8, draw_odds=3.6, away_odds=4.5,
        ),
    ]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    captured_match_info = {}

    def _capture(match_info, config):
        captured_match_info.update(match_info)
        return _RECOMMENDATION

    with patch("app.backend.recommendations.run_agent", side_effect=_capture):
        result = asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda f: None, date_str="2026-08-22",
            )
        )

    assert "odds" not in captured_match_info
    assert result.generated == 1


def test_odds_client_returning_none_gracefully_proceeds_without_odds(tmp_path: Path) -> None:
    """Odds API credit budget exhausted (W07's own None-return convention)
    -- the whole batch must still proceed, just without odds."""
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [_fixture("m1", "Arsenal", "Everton")]
    odds_client = MagicMock()
    odds_client.get_odds.return_value = None
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION):
        result = asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda f: None, date_str="2026-08-22",
            )
        )

    assert result.generated == 1


def test_no_odds_client_at_all_proceeds_without_odds(tmp_path: Path) -> None:
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [_fixture("m1", "Arsenal", "Everton")]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION):
        result = asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config,
                schedule_t30=lambda f: None, date_str="2026-08-22",
            )
        )

    assert result.generated == 1
