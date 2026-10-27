"""BUG-096 follow-up (direct user request, 2026-10-27): notify on Slack when
live odds are missing, instead of blocking bets on missing model features.

Missing live odds don't block a recommendation here. A market with no live
price is already kept out of bets by its own edge calculation (BUG-013's
null-odds downgrade): no price, no value edge. What a missing price ALSO
does is starve the ML forecast of real market features -- no over/under
2.5 or Asian handicap price means MKT_IMPLIED_OVER25/MKT_AH_*/MKT_LAMBDA_*
are placeholders for every goal target, and for result_3way, which is
built from the home/away goal models. This alert makes that visible.

Markets checked, per fixture, after add_secondary_odds() has run:
- 1X2 (h2h): match_info["odds"]
- over/under 2.5: match_info["total_goals_odds"]
- Asian handicap: match_info["asian_handicap_odds"]

EOD sends one summary per league and date; T-30 sends one per match. Both
go through slack_alerts.post_slack_alert (off unless SLACK_ALERT_WEBHOOK_URL
is set, debounced, never raises).
"""

from __future__ import annotations

from app.backend.slack_alerts import post_slack_alert


def missing_live_odds(match_info: dict) -> list[str]:
    """Names of the live markets this fixture has no usable price for."""
    missing: list[str] = []
    h2h = match_info.get("odds") or {}
    if not all(h2h.get(k) for k in ("home", "draw", "away")):
        missing.append("1X2")
    if not (match_info.get("total_goals_odds") or {}).get("over_2.5"):
        missing.append("over/under 2.5")
    ah = match_info.get("asian_handicap_odds") or {}
    if ah.get("line") is None or not ah.get("home") or not ah.get("away"):
        missing.append("Asian handicap")
    return missing


def _fixture_label(home_team: str, away_team: str) -> str:
    return f"{home_team} v {away_team}"


def alert_missing_live_odds_for_batch(
    league: str, date_str: str, reports: list[tuple[str, str, list[str]]], total_fixtures: int,
) -> bool:
    """One message for an EOD batch. `reports` is (home, away, missing
    markets) for each fixture with at least one missing market."""
    if not reports:
        return False
    lines = [f"• {_fixture_label(home, away)}: {', '.join(missing)}" for home, away, missing in reports]
    text = (
        f":warning: Missing live odds -- {league} {date_str}: {len(reports)} of {total_fixtures} "
        f"fixture(s) generated without some live prices. Forecasts for these used placeholder "
        f"market features; bets in a market with no price are skipped.\n" + "\n".join(lines)
    )
    return post_slack_alert(text, debounce_key=f"missing_live_odds:eod:{league}:{date_str}")


def alert_missing_live_odds_for_match(
    league: str, match_id: str, home_team: str, away_team: str, missing: list[str],
) -> bool:
    """One message for a T-30 refresh of a single fixture."""
    if not missing:
        return False
    text = (
        f":warning: Missing live odds at T-30 -- {league} {_fixture_label(home_team, away_team)}: "
        f"{', '.join(missing)}. The near-kickoff forecast used placeholder market features."
    )
    return post_slack_alert(text, debounce_key=f"missing_live_odds:t30:{match_id}")
