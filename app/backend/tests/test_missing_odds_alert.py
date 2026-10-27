"""BUG-096 follow-up (2026-10-27): missing live odds raise a Slack alert
instead of blocking bets on missing model features."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

import pytest  # noqa: E402

from app.backend import slack_alerts  # noqa: E402
from app.backend.eod_batch import run_eod_batch  # noqa: E402
from app.backend.missing_odds_alert import (  # noqa: E402
    alert_missing_live_odds_for_batch,
    alert_missing_live_odds_for_match,
    missing_live_odds,
)
from app.backend.odds_api_client import NormalizedOdds, NormalizedSecondaryOdds  # noqa: E402
from app.backend.recommendation_cache import RecommendationCache  # noqa: E402
from app.backend.tests.test_eod_batch import _RECOMMENDATION, _fixture, _future_date  # noqa: E402
from src.agent.agent_config import AgentConfig  # noqa: E402

_FULL = {
    "odds": {"home": 1.8, "draw": 3.6, "away": 4.5},
    "total_goals_odds": {"over_2.5": 1.9, "under_2.5": 1.95},
    "asian_handicap_odds": {"line": -0.75, "home": 1.95, "away": 1.95},
}


@pytest.fixture(autouse=True)
def _webhook(monkeypatch):
    monkeypatch.setenv("SLACK_ALERT_WEBHOOK_URL", "https://hooks.slack.test/fake")
    slack_alerts._last_slack_alert_at.clear()
    yield
    slack_alerts._last_slack_alert_at.clear()


def test_missing_live_odds_lists_each_missing_market():
    assert missing_live_odds(_FULL) == []
    assert missing_live_odds({}) == ["1X2", "over/under 2.5", "Asian handicap"]
    no_ah = {k: v for k, v in _FULL.items() if k != "asian_handicap_odds"}
    assert missing_live_odds(no_ah) == ["Asian handicap"]


def test_a_level_handicap_line_counts_as_present():
    level = {**_FULL, "asian_handicap_odds": {"line": 0.0, "home": 1.9, "away": 1.9}}
    assert missing_live_odds(level) == []


def test_batch_alert_posts_one_summary_and_is_debounced():
    reports = [("Arsenal", "Everton", ["Asian handicap"]), ("Chelsea", "Fulham", ["1X2", "over/under 2.5"])]
    with patch("app.backend.slack_alerts.requests.post") as mock_post:
        assert alert_missing_live_odds_for_batch("E0", "2026-10-31", reports, total_fixtures=10) is True
        assert alert_missing_live_odds_for_batch("E0", "2026-10-31", reports, total_fixtures=10) is False

    mock_post.assert_called_once()
    text = mock_post.call_args.kwargs["json"]["text"]
    assert "E0 2026-10-31: 2 of 10" in text
    assert "Arsenal v Everton: Asian handicap" in text
    assert "Chelsea v Fulham: 1X2, over/under 2.5" in text


def test_no_alert_when_nothing_is_missing_or_webhook_unset(monkeypatch):
    with patch("app.backend.slack_alerts.requests.post") as mock_post:
        assert alert_missing_live_odds_for_batch("E0", "2026-10-31", [], total_fixtures=10) is False
        assert alert_missing_live_odds_for_match("E0", "m1", "A", "B", []) is False
        monkeypatch.delenv("SLACK_ALERT_WEBHOOK_URL")
        assert alert_missing_live_odds_for_match("E0", "m1", "A", "B", ["1X2"]) is False
    mock_post.assert_not_called()


def test_match_alert_names_the_fixture_and_markets():
    with patch("app.backend.slack_alerts.requests.post") as mock_post:
        alert_missing_live_odds_for_match("SP1", "m9", "Barça", "Real Madrid", ["Asian handicap"])
    text = mock_post.call_args.kwargs["json"]["text"]
    assert "T-30" in text and "SP1 Barça v Real Madrid: Asian handicap" in text


def test_eod_batch_alerts_on_generated_fixture_missing_asian_handicap(tmp_path: Path):
    """End to end: the fixture still generates (nothing is blocked), and the
    batch sends one alert naming the missing market."""
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [_fixture("m1", "Arsenal", "Everton")]
    odds_client = MagicMock()
    odds_client.get_odds.return_value = [
        NormalizedOdds(
            home_team="Arsenal", away_team="Everton", commence_time="2026-08-22T15:00:00Z",
            home_odds=1.8, draw_odds=3.6, away_odds=4.5, event_id="evt1",
        ),
    ]
    odds_client.get_event_odds.return_value = NormalizedSecondaryOdds(
        total_goals={"over_2.5": 1.9, "under_2.5": 1.95}, btts=None, asian_handicap=None,
    )
    cache = RecommendationCache(db_path=tmp_path / "cache.db")

    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION), \
         patch("app.backend.slack_alerts.requests.post") as mock_post:
        result = asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache,
                config=AgentConfig.default(), schedule_t30=lambda f: None, date_str=_future_date(1),
            )
        )

    assert result.generated == 1
    mock_post.assert_called_once()
    assert "Arsenal v Everton: Asian handicap" in mock_post.call_args.kwargs["json"]["text"]
