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


def _fixture(match_id: str, home: str, away: str, utc_date: str = "2026-08-22T15:00:00Z") -> NormalizedMatch:
    return NormalizedMatch(
        match_id=match_id, utc_date=utc_date, status="SCHEDULED",
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


def test_odds_matched_via_canonical_team_name_despite_provider_spelling_differences(tmp_path: Path) -> None:
    """BUG-015: football-data.org and The Odds API spell many clubs
    differently (confirmed live against a real key -- 'Man United' vs
    'Manchester United', 'Nottingham' vs 'Nottingham Forest', 'Tottenham'
    vs 'Tottenham Hotspur', 'Brighton Hove' vs 'Brighton and Hove Albion',
    etc.). Matching by raw string equality silently drops odds for most
    real fixtures; matching via the shared TeamNameMapper (the same
    canonical space ingestion already uses) must resolve them."""
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [_fixture("m1", "Man United", "Nottingham")]
    odds_client = MagicMock()
    odds_client.get_odds.return_value = [
        NormalizedOdds(
            home_team="Manchester United", away_team="Nottingham Forest", commence_time="2026-08-22T15:00:00Z",
            home_odds=1.4, draw_odds=4.5, away_odds=7.0,
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

    assert captured_match_info["odds"] == {"home": 1.4, "draw": 4.5, "away": 7.0}


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


def test_league_parameter_tags_match_info_and_selects_the_matching_sport_key(tmp_path: Path) -> None:
    """W62: run_eod_batch is no longer hardcoded to E0 -- a caller-supplied
    `league` both tags every generated match_info and selects the matching
    Odds-API sport_key, so a single function serves any competition_specific
    league the multi-competition scheduler orchestration loops over."""
    fixtures_client = MagicMock()
    odds_client = MagicMock()
    odds_client.get_odds.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    swedish_fixture = _fixture("sw1", "Malmo FF", "AIK")

    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION) as mock_run:
        asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda f: None, date_str="2026-08-22",
                fixtures=[swedish_fixture], league="SWE",
            )
        )

    odds_client.get_odds.assert_called_once_with(sport_key="soccer_sweden_allsvenskan")
    match_info = mock_run.call_args.kwargs["match_info"]
    assert match_info["league"] == "SWE"


def test_league_parameter_defaults_to_e0_preserving_existing_behavior(tmp_path: Path) -> None:
    fixtures_client = MagicMock()
    odds_client = MagicMock()
    odds_client.get_odds.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION) as mock_run:
        asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda f: None, date_str="2026-08-22",
                fixtures=[_fixture("m1", "Arsenal", "Everton")],
            )
        )

    odds_client.get_odds.assert_called_once_with(sport_key="soccer_epl")
    assert mock_run.call_args.kwargs["match_info"]["league"] == "E0"


def test_get_odds_is_called_with_an_explicit_epl_sport_key(tmp_path: Path) -> None:
    """W58: must not rely on get_odds()'s own "soccer_epl" default parameter
    -- an explicit sport_key from the competition-id mapping, so the same
    call shape is ready for a future per-competition loop (W62) without a
    silent EPL-only assumption baked into an omitted argument."""
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [_fixture("m1", "Arsenal", "Everton")]
    odds_client = MagicMock()
    odds_client.get_odds.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION):
        asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda f: None, date_str="2026-08-22",
            )
        )

    odds_client.get_odds.assert_called_once_with(sport_key="soccer_epl")


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


def test_supplied_fixtures_bypass_fixtures_client_get_fixtures(tmp_path: Path) -> None:
    """W50: a caller sourcing a *past* sandbox date's fixtures via
    get_results() (since get_fixtures() only ever queries status=SCHEDULED,
    and can never return anything for an already-played date) hands them in
    directly -- fixtures_client.get_fixtures() must never be called in that
    case, and generation must proceed against the supplied list."""
    fixtures_client = MagicMock()
    supplied = [
        _fixture("m1", "Arsenal", "Everton"),
        _fixture("m2", "Chelsea", "Fulham"),
    ]
    odds_client = MagicMock()
    odds_client.get_odds.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION):
        result = asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda f: None, date_str="2026-08-22", fixtures=supplied,
            )
        )

    fixtures_client.get_fixtures.assert_not_called()
    assert result.generated == 2
    assert result.fixtures == supplied
    agent_config_hash = compute_agent_config_hash(config)
    for match_id in ("m1", "m2"):
        assert cache.get_latest(match_id, "2026-08-22", agent_config_hash) is not None


def test_omitting_fixtures_preserves_default_get_fixtures_call(tmp_path: Path) -> None:
    """Default (no injection) behavior is unchanged: fixtures_client.get_fixtures()
    is still called exactly as before, with the same arguments."""
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [_fixture("m1", "Arsenal", "Everton")]
    odds_client = MagicMock()
    odds_client.get_odds.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION):
        result = asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda f: None, date_str="2026-08-22",
            )
        )

    fixtures_client.get_fixtures.assert_called_once_with(
        competition_code="PL", date_from="2026-08-22", date_to="2026-08-22",
    )
    assert result.generated == 1


def test_on_progress_called_once_per_fixture_with_outcome(tmp_path: Path) -> None:
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [
        _fixture("m1", "Arsenal", "Everton"),
        _fixture("m2", "Chelsea", "Fulham"),
    ]
    odds_client = MagicMock()
    odds_client.get_odds.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    progress: list[tuple[str, str]] = []

    def _run_agent_side_effect(match_info, config):
        if match_info["home_team"] == "Chelsea":
            raise RuntimeError("LLM timeout")
        return _RECOMMENDATION

    with patch("app.backend.recommendations.run_agent", side_effect=_run_agent_side_effect):
        asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda f: None, date_str="2026-08-22",
                on_progress=lambda fixture, outcome: progress.append((fixture.match_id, outcome)),
            )
        )

    assert sorted(progress) == [("m1", "generated"), ("m2", "skipped")]


# --- W54: sandbox fallback-window fixtures must be cached/scoped under their
# own date, not the batch's date_str (which is only ever correct for a true
# one-day batch: the live scheduler, or an exact-date sandbox precompute). ---

def test_fixture_with_a_different_date_than_date_str_is_cached_under_its_own_date(tmp_path: Path) -> None:
    """W54: SANDBOX_DATE (date_str) has no real fixtures, so the fallback
    window (W51) supplies a fixture dated well after it -- the cache row
    must land under the fixture's real date, since that's what the Dashboard
    (and W53's initial-list check) actually queries by."""
    fixtures_client = MagicMock()
    odds_client = MagicMock()
    odds_client.get_odds.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    fixture = _fixture("m1", "Sunderland", "Brighton Hove", utc_date="2026-03-14T15:00:00Z")

    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION):
        asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda f: None, date_str="2026-03-08", fixtures=[fixture],
            )
        )

    agent_config_hash = compute_agent_config_hash(config)
    assert cache.get_latest("m1", "2026-03-14", agent_config_hash) is not None
    assert cache.get_latest("m1", "2026-03-08", agent_config_hash) is None


def test_match_info_date_uses_the_fixtures_own_date_not_date_str(tmp_path: Path) -> None:
    fixtures_client = MagicMock()
    odds_client = MagicMock()
    odds_client.get_odds.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    fixture = _fixture("m1", "Sunderland", "Brighton Hove", utc_date="2026-03-14T15:00:00Z")
    captured_match_info = {}

    def _capture(match_info, config):
        captured_match_info.update(match_info)
        return _RECOMMENDATION

    with patch("app.backend.recommendations.run_agent", side_effect=_capture):
        asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda f: None, date_str="2026-03-08", fixtures=[fixture],
            )
        )

    assert captured_match_info["date"] == "2026-03-14"


def test_odds_fetched_per_distinct_fixture_date_when_fixtures_span_multiple_dates(tmp_path: Path) -> None:
    """W54: a fallback window can contain fixtures on several different
    dates (not just one date later than date_str) -- each fixture's odds
    must come from its own date, not a single date_str-scoped lookup that
    would (per HistoricalOddsClient) return odds for none of them."""
    fixtures_client = MagicMock()
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    fixture_14 = _fixture("m1", "Sunderland", "Brighton Hove", utc_date="2026-03-14T15:00:00Z")
    fixture_15 = _fixture("m2", "Arsenal", "Everton", utc_date="2026-03-15T15:00:00Z")

    def _odds_for_date(sport_key: str, date: str | None = None):
        if date == "2026-03-14":
            return [NormalizedOdds(
                home_team="Sunderland", away_team="Brighton Hove", commence_time="2026-03-14T15:00:00Z",
                home_odds=2.5, draw_odds=3.2, away_odds=2.9,
            )]
        if date == "2026-03-15":
            return [NormalizedOdds(
                home_team="Arsenal", away_team="Everton", commence_time="2026-03-15T15:00:00Z",
                home_odds=1.5, draw_odds=4.0, away_odds=6.0,
            )]
        return []

    odds_client = MagicMock()
    odds_client.get_odds.side_effect = _odds_for_date
    captured: dict[str, dict] = {}

    def _capture(match_info, config):
        captured[match_info["home_team"]] = match_info.get("odds")
        return _RECOMMENDATION

    with patch("app.backend.recommendations.run_agent", side_effect=_capture):
        asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda f: None, date_str="2026-03-08", fixtures=[fixture_14, fixture_15],
            )
        )

    assert captured["Sunderland"] == {"home": 2.5, "draw": 3.2, "away": 2.9}
    assert captured["Arsenal"] == {"home": 1.5, "draw": 4.0, "away": 6.0}
