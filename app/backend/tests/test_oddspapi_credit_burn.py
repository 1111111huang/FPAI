"""BUG-097 (2026-10-27): OddsPapi corners credits ran out by the 27th on 5
keys (1,250 calls a month) when a ~185-match month needs ~200. Two causes:

1. get_corners_odds() returned None both for "no 9.5 line posted yet" and
   for "this key's budget is spent", so FallbackOddsPapiClient paid every
   remaining key for the same empty answer, then fired a false "all keys
   exhausted" Slack alert.
2. add_secondary_odds() only reused a cached corners result when it held a
   real price, so an unposted line was re-fetched on every pass.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

import pytest  # noqa: E402

from app.backend import oddspapi_client as oddspapi_client_module  # noqa: E402
from app.backend import slack_alerts  # noqa: E402
from app.backend.agent_config_hash import compute_agent_config_hash  # noqa: E402
from app.backend.eod_batch import already_fresh, run_eod_batch  # noqa: E402
from app.backend.odds_api_client import CreditCounter, NormalizedSecondaryOdds  # noqa: E402
from app.backend.oddspapi_client import OddsPapiClient  # noqa: E402
from app.backend.recommendation_cache import RecommendationCache  # noqa: E402
from app.backend.scheduler_wiring import FallbackOddsPapiClient  # noqa: E402
from app.backend.tests.test_eod_batch import (  # noqa: E402
    _RECOMMENDATION,
    _fixture,
    _future_date,
    _seeded_odds_client,
    _seeded_oddspapi_client,
)
from src.agent.agent_config import AgentConfig  # noqa: E402

_NO_LINE_PAYLOAD = {"bookmakers": {"pinnacle": {"markets": {}}}}
_H2H = {"home": 1.8, "draw": 3.6, "away": 4.5}
_SECONDARY = {
    "total_goals": {"over_2.5": 1.9, "under_2.5": 1.95}, "btts": {"yes": 1.7, "no": 2.1},
    "home_goals": {"over_1.5": 1.6, "under_1.5": 2.2}, "away_goals": {"over_1.5": 2.5, "under_1.5": 1.5},
    "asian_handicap": {"line": -0.75, "home": 1.95, "away": 1.95},
}


@pytest.fixture(autouse=True)
def _no_real_throttle_sleep(monkeypatch):
    monkeypatch.setattr(oddspapi_client_module.time, "sleep", lambda seconds: None)
    oddspapi_client_module._last_request_at = 0.0


def _session(payload) -> MagicMock:
    session = MagicMock()
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    session.get.return_value = response
    return session


# --- 1. "no line yet" is an answer, not an exhausted key -------------------


def test_get_corners_odds_returns_empty_dict_when_no_line_is_posted():
    counter = CreditCounter()
    client = OddsPapiClient(api_key="k", credit_counter=counter, session=_session(_NO_LINE_PAYLOAD))

    assert client.get_corners_odds(fixture_id="555") == {}
    assert counter.credits_used == 1  # the call was made and paid for


def test_get_corners_odds_still_returns_none_when_budget_is_spent():
    session = _session(_NO_LINE_PAYLOAD)
    client = OddsPapiClient(api_key="k", credit_counter=CreditCounter(credits_used=241), session=session)

    assert client.get_corners_odds(fixture_id="555") is None
    session.get.assert_not_called()


def test_fallback_stops_at_the_first_key_that_answers_with_no_line(monkeypatch):
    monkeypatch.setenv("SLACK_ALERT_WEBHOOK_URL", "https://hooks.slack.test/fake")
    slack_alerts._last_slack_alert_at.clear()
    keys = [MagicMock() for _ in range(5)]
    for key in keys:
        key.is_available.return_value = True
        key.get_corners_odds.return_value = {}

    with patch("app.backend.slack_alerts.requests.post") as mock_post:
        result = FallbackOddsPapiClient(keys).get_corners_odds(fixture_id="555")

    assert result == {}
    keys[0].get_corners_odds.assert_called_once()
    for key in keys[1:]:
        key.get_corners_odds.assert_not_called()  # was: every key paid for the same answer
    mock_post.assert_not_called()  # was: a false "all 5 keys exhausted" alert


def test_fallback_still_moves_on_and_alerts_when_keys_are_really_spent(monkeypatch):
    monkeypatch.setenv("SLACK_ALERT_WEBHOOK_URL", "https://hooks.slack.test/fake")
    slack_alerts._last_slack_alert_at.clear()
    keys = [MagicMock() for _ in range(2)]
    for key in keys:
        key.is_available.return_value = True
        key.get_corners_odds.return_value = None

    with patch("app.backend.slack_alerts.requests.post") as mock_post:
        assert FallbackOddsPapiClient(keys).get_corners_odds(fixture_id="555") is None

    assert all(key.get_corners_odds.call_count == 1 for key in keys)
    mock_post.assert_called_once()


# --- 2. an empty answer is cached for a while ------------------------------


def _seed_cache(tmp_path: Path, corners, generated_at: datetime | None = None) -> tuple[RecommendationCache, AgentConfig]:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    odds = {**_H2H, **_SECONDARY}
    if corners is not ...:
        odds["corners"] = corners
    cache.record_generation(
        match_id="m1", date=_future_date(1), agent_config_hash=compute_agent_config_hash(config),
        odds=odds, recommendation=_RECOMMENDATION, triggered_by="scheduled",
        generated_at=(generated_at or datetime.now(timezone.utc)).replace(microsecond=0).isoformat(),
    )
    return cache, config


def _run(cache, config, oddspapi_client):
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [_fixture("m1", "Arsenal", "Everton")]
    odds_client = _seeded_odds_client()
    odds_client.get_event_odds.return_value = NormalizedSecondaryOdds(
        total_goals=_SECONDARY["total_goals"], btts=_SECONDARY["btts"],
        home_goals=_SECONDARY["home_goals"], away_goals=_SECONDARY["away_goals"],
        asian_handicap=_SECONDARY["asian_handicap"],
    )
    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION):
        return asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, oddspapi_client=oddspapi_client,
                cache=cache, config=config, schedule_t30=lambda f: None, date_str=_future_date(1),
            )
        )


def test_recent_empty_corners_answer_is_reused_not_refetched(tmp_path: Path):
    cache, config = _seed_cache(tmp_path, corners={})
    oddspapi_client = _seeded_oddspapi_client()

    result = _run(cache, config, oddspapi_client)

    oddspapi_client.get_corners_odds.assert_not_called()
    assert result.unchanged == 1 and result.generated == 0


def test_stale_empty_corners_answer_is_retried_once(tmp_path: Path):
    cache, config = _seed_cache(tmp_path, corners={}, generated_at=datetime.now(timezone.utc) - timedelta(hours=13))
    oddspapi_client = _seeded_oddspapi_client()
    oddspapi_client.get_corners_odds.return_value = {}

    result = _run(cache, config, oddspapi_client)

    oddspapi_client.get_corners_odds.assert_called_once()
    assert result.unchanged == 1  # still no line: no LLM regeneration for it


def test_new_empty_answer_is_stored_and_does_not_force_a_regeneration(tmp_path: Path):
    """A cache row from before BUG-097 has no "corners" key at all. Getting
    {} back for it must not count as an odds move."""
    cache, config = _seed_cache(tmp_path, corners=...)
    oddspapi_client = _seeded_oddspapi_client()
    oddspapi_client.get_corners_odds.return_value = {}

    result = _run(cache, config, oddspapi_client)

    oddspapi_client.get_corners_odds.assert_called_once()
    assert result.unchanged == 1


def test_already_fresh_treats_empty_corners_like_no_corners(tmp_path: Path):
    cache, config = _seed_cache(tmp_path, corners=...)
    agent_config_hash = compute_agent_config_hash(config)
    odds = {**_H2H, **_SECONDARY}

    assert already_fresh(cache, "m1", _future_date(1), agent_config_hash, {**odds, "corners": {}})
    assert not already_fresh(
        cache, "m1", _future_date(1), agent_config_hash, {**odds, "corners": {"over_9.5": 1.9, "under_9.5": 1.95}},
    )
