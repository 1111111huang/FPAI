"""W10: per-match refresh 30 minutes before kickoff (D2a). Fetches fresh
odds (W07) first and compares them against the odds used for the
currently-cached recommendation (W11) -- only re-runs run_agent() if the
odds actually changed; best-effort: a fixture whose odds can't be
fetched/matched, or a run_agent() error, leaves the prior recommendation
in place rather than failing."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.agent_config_hash import compute_agent_config_hash
from app.backend.football_data_client import NormalizedMatch
from app.backend.odds_api_client import NormalizedOdds
from app.backend.recommendation_cache import RecommendationCache
from app.backend.t30_refresh import refresh_match_at_t30
from src.agent.agent_config import AgentConfig

_RECOMMENDATION = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
    "overall": "direct_bet",
    "markets": [{"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.1}],
    "explanation": "test", "confidence": "medium", "limitations": [], "prediction_basis": "team_history_and_market",
}


def _fixture(match_id: str = "m1", home: str = "Arsenal", away: str = "Everton") -> NormalizedMatch:
    return NormalizedMatch(
        match_id=match_id, utc_date="2026-08-22T15:00:00Z", status="SCHEDULED",
        home_team=home, away_team=away, home_goals=None, away_goals=None,
    )


def _seed_cache(cache: RecommendationCache, config: AgentConfig, odds: dict) -> None:
    cache.record_generation(
        match_id="m1", date="2026-08-22", agent_config_hash=compute_agent_config_hash(config),
        odds=odds, recommendation=_RECOMMENDATION, triggered_by="scheduled",
    )


def test_skips_refresh_when_odds_unchanged(tmp_path: Path) -> None:
    config = AgentConfig.default()
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    _seed_cache(cache, config, odds={"home": 1.8, "draw": 3.6, "away": 4.5})

    odds_client = MagicMock()
    odds_client.get_odds.return_value = [
        NormalizedOdds(home_team="Arsenal", away_team="Everton", commence_time="2026-08-22T15:00:00Z",
                        home_odds=1.8, draw_odds=3.6, away_odds=4.5),
    ]

    with patch("app.backend.recommendations.run_agent") as mock_run_agent:
        result = refresh_match_at_t30(_fixture(), odds_client=odds_client, cache=cache, config=config, date_str="2026-08-22")

    mock_run_agent.assert_not_called()
    assert result.outcome == "skipped_no_change"
    agent_config_hash = compute_agent_config_hash(config)
    entry = cache.get_latest("m1", "2026-08-22", agent_config_hash)
    assert entry.triggered_by == "scheduled"
    assert len(cache.get_history("m1", "2026-08-22", agent_config_hash)) == 1  # no new row written


def test_get_odds_is_called_with_an_explicit_epl_sport_key(tmp_path: Path) -> None:
    """W58: must not rely on get_odds()'s own "soccer_epl" default parameter."""
    config = AgentConfig.default()
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    odds_client = MagicMock()
    odds_client.get_odds.return_value = []

    refresh_match_at_t30(_fixture(), odds_client=odds_client, cache=cache, config=config, date_str="2026-08-22")

    odds_client.get_odds.assert_called_once_with(sport_key="soccer_epl")


def test_league_parameter_tags_match_info_and_selects_the_matching_sport_key(tmp_path: Path) -> None:
    """W62: refresh_match_at_t30 is no longer hardcoded to E0 -- a
    caller-supplied `league` both tags match_info and selects the matching
    Odds-API sport_key, so one function serves any competition_specific
    league the multi-competition scheduler orchestration loops over."""
    config = AgentConfig.default()
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    odds_client = MagicMock()
    odds_client.get_odds.return_value = [
        NormalizedOdds(home_team="Malmo FF", away_team="AIK", commence_time="2026-08-22T15:00:00Z",
                        home_odds=1.8, draw_odds=3.6, away_odds=4.5),
    ]
    swedish_fixture = _fixture(match_id="sw1", home="Malmo FF", away="AIK")

    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION) as mock_run:
        refresh_match_at_t30(
            swedish_fixture, odds_client=odds_client, cache=cache, config=config,
            date_str="2026-08-22", league="SWE",
        )

    odds_client.get_odds.assert_called_once_with(sport_key="soccer_sweden_allsvenskan")
    assert mock_run.call_args.kwargs["match_info"]["league"] == "SWE"


def test_league_parameter_defaults_to_e0_preserving_existing_behavior(tmp_path: Path) -> None:
    config = AgentConfig.default()
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    odds_client = MagicMock()
    odds_client.get_odds.return_value = [
        NormalizedOdds(home_team="Arsenal", away_team="Everton", commence_time="2026-08-22T15:00:00Z",
                        home_odds=1.8, draw_odds=3.6, away_odds=4.5),
    ]

    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION) as mock_run:
        refresh_match_at_t30(_fixture(), odds_client=odds_client, cache=cache, config=config, date_str="2026-08-22")

    odds_client.get_odds.assert_called_once_with(sport_key="soccer_epl")
    assert mock_run.call_args.kwargs["match_info"]["league"] == "E0"


def test_refreshes_when_odds_changed(tmp_path: Path) -> None:
    config = AgentConfig.default()
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    _seed_cache(cache, config, odds={"home": 1.8, "draw": 3.6, "away": 4.5})

    odds_client = MagicMock()
    odds_client.get_odds.return_value = [
        NormalizedOdds(home_team="Arsenal", away_team="Everton", commence_time="2026-08-22T15:00:00Z",
                        home_odds=1.6, draw_odds=3.8, away_odds=5.0),  # moved
    ]

    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION) as mock_run_agent:
        result = refresh_match_at_t30(_fixture(), odds_client=odds_client, cache=cache, config=config, date_str="2026-08-22")

    mock_run_agent.assert_called_once()
    assert result.outcome == "refreshed"
    agent_config_hash = compute_agent_config_hash(config)
    history = cache.get_history("m1", "2026-08-22", agent_config_hash)
    assert len(history) == 2
    assert history[-1].odds == {"home": 1.6, "draw": 3.8, "away": 5.0}


def test_generates_fresh_when_no_prior_cache_entry_exists(tmp_path: Path) -> None:
    """No baseline to compare against (e.g. EOD generation failed for this
    match) -- nothing to lose by generating fresh rather than leaving the
    match with no recommendation at all."""
    config = AgentConfig.default()
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    odds_client = MagicMock()
    odds_client.get_odds.return_value = [
        NormalizedOdds(home_team="Arsenal", away_team="Everton", commence_time="2026-08-22T15:00:00Z",
                        home_odds=1.8, draw_odds=3.6, away_odds=4.5),
    ]

    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION) as mock_run_agent:
        result = refresh_match_at_t30(_fixture(), odds_client=odds_client, cache=cache, config=config, date_str="2026-08-22")

    mock_run_agent.assert_called_once()
    assert result.outcome == "refreshed"


def test_skips_gracefully_when_credit_budget_exhausted(tmp_path: Path) -> None:
    config = AgentConfig.default()
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    _seed_cache(cache, config, odds={"home": 1.8, "draw": 3.6, "away": 4.5})

    odds_client = MagicMock()
    odds_client.get_odds.return_value = None  # W07's own budget-exhausted convention

    with patch("app.backend.recommendations.run_agent") as mock_run_agent:
        result = refresh_match_at_t30(_fixture(), odds_client=odds_client, cache=cache, config=config, date_str="2026-08-22")

    mock_run_agent.assert_not_called()
    assert result.outcome == "skipped_no_odds"
    agent_config_hash = compute_agent_config_hash(config)
    assert len(cache.get_history("m1", "2026-08-22", agent_config_hash)) == 1  # prior recommendation untouched


def test_skips_gracefully_when_fixture_odds_not_found_eg_postponed(tmp_path: Path) -> None:
    config = AgentConfig.default()
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    _seed_cache(cache, config, odds={"home": 1.8, "draw": 3.6, "away": 4.5})

    odds_client = MagicMock()
    odds_client.get_odds.return_value = []  # fixture no longer in the odds feed

    with patch("app.backend.recommendations.run_agent") as mock_run_agent:
        result = refresh_match_at_t30(_fixture(), odds_client=odds_client, cache=cache, config=config, date_str="2026-08-22")

    mock_run_agent.assert_not_called()
    assert result.outcome == "skipped_no_odds"


def test_skips_gracefully_on_run_agent_error(tmp_path: Path) -> None:
    config = AgentConfig.default()
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    _seed_cache(cache, config, odds={"home": 1.8, "draw": 3.6, "away": 4.5})

    odds_client = MagicMock()
    odds_client.get_odds.return_value = [
        NormalizedOdds(home_team="Arsenal", away_team="Everton", commence_time="2026-08-22T15:00:00Z",
                        home_odds=1.6, draw_odds=3.8, away_odds=5.0),
    ]

    with patch("app.backend.recommendations.run_agent", side_effect=RuntimeError("LLM timeout")):
        result = refresh_match_at_t30(_fixture(), odds_client=odds_client, cache=cache, config=config, date_str="2026-08-22")

    assert result.outcome == "skipped_error"
    agent_config_hash = compute_agent_config_hash(config)
    assert len(cache.get_history("m1", "2026-08-22", agent_config_hash)) == 1  # prior recommendation untouched


def test_no_odds_client_at_all_skips_gracefully(tmp_path: Path) -> None:
    config = AgentConfig.default()
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    _seed_cache(cache, config, odds={"home": 1.8, "draw": 3.6, "away": 4.5})

    with patch("app.backend.recommendations.run_agent") as mock_run_agent:
        result = refresh_match_at_t30(_fixture(), odds_client=None, cache=cache, config=config, date_str="2026-08-22")

    mock_run_agent.assert_not_called()
    assert result.outcome == "skipped_no_odds"
