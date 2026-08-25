"""W28: sandbox odds source -- real historical 1X2 odds from raw_matches
(football-data.co.uk), not synthetic. Seeds a real temp DuckDB with the
same raw_matches columns production uses, so this exercises a real DuckDB
query end-to-end rather than a mocked connection. Deliberately excludes
odds movement (a single closing-line snapshot, not a time series) -- see
W32."""

from __future__ import annotations

from pathlib import Path
import sys

import duckdb
import yaml

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.historical_odds_client import HistoricalOddsClient


def _seed_db(tmp_path: Path):
    from src.utils.db_manager import DuckDBManager

    db_path = tmp_path / "sandbox.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}))

    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE raw_matches (
            match_id TEXT, league TEXT, date TIMESTAMP,
            home_team TEXT, away_team TEXT,
            odds_h FLOAT, odds_d FLOAT, odds_a FLOAT
        )
        """
    )
    conn.execute(
        "INSERT INTO raw_matches VALUES "
        "('1', 'E0', '2026-03-01', 'Arsenal', 'Everton', 1.80, 3.60, 4.20), "
        "('2', 'E0', '2026-03-01', 'Chelsea', 'Fulham', 1.50, 4.00, 6.50), "
        "('3', 'E0', '2026-03-02', 'Liverpool', 'Burnley', 1.30, 5.50, 9.00), "  # different date -- must be excluded
        # BUG-034: same date, different league -- must not leak into an E0 query.
        "('4', 'SP1', '2026-03-01', 'Barcelona', 'Sevilla', 1.40, 4.50, 7.00)"
    )
    conn.close()
    return DuckDBManager(config_path=str(config_path))


def test_get_odds_returns_real_odds_for_the_sandbox_date(tmp_path: Path) -> None:
    manager = _seed_db(tmp_path)
    client = HistoricalOddsClient(sandbox_date="2026-03-01", db_manager=manager)

    result = client.get_odds()

    assert result is not None
    assert len(result) == 2
    by_home = {odds.home_team: odds for odds in result}
    assert by_home["Arsenal"].home_odds == 1.80
    assert by_home["Arsenal"].draw_odds == 3.60
    assert by_home["Arsenal"].away_odds == 4.20
    assert by_home["Chelsea"].away_team == "Fulham"


def test_get_odds_excludes_fixtures_on_other_dates(tmp_path: Path) -> None:
    manager = _seed_db(tmp_path)
    client = HistoricalOddsClient(sandbox_date="2026-03-01", db_manager=manager)

    result = client.get_odds()

    assert all(odds.home_team != "Liverpool" for odds in result)


def test_get_odds_returns_none_when_no_fixtures_that_date(tmp_path: Path) -> None:
    manager = _seed_db(tmp_path)
    client = HistoricalOddsClient(sandbox_date="2099-01-01", db_manager=manager)

    assert client.get_odds() is None


def test_get_odds_matches_the_normalizedodds_shape_odds_api_client_uses() -> None:
    import dataclasses

    from app.backend.odds_api_client import NormalizedOdds

    assert {f.name for f in dataclasses.fields(NormalizedOdds)} == {
        "home_team", "away_team", "commence_time", "home_odds", "draw_odds", "away_odds", "event_id",
    }


def test_get_odds_date_param_overrides_the_constructor_sandbox_date(tmp_path: Path) -> None:
    """W54: a sandbox fallback batch needs odds for fixture dates other than
    the client's own construction-time sandbox_date (e.g. precompute for
    SANDBOX_DATE=2026-03-08 covering a real fixture on 2026-03-02, the
    fallback window's nearest matchday) -- an explicit `date` must win over
    the instance default, per-call, without needing a new client instance."""
    manager = _seed_db(tmp_path)
    client = HistoricalOddsClient(sandbox_date="2026-03-01", db_manager=manager)

    result = client.get_odds(date="2026-03-02")

    assert result is not None
    assert len(result) == 1
    assert result[0].home_team == "Liverpool"


def test_get_odds_without_date_param_falls_back_to_constructor_sandbox_date(tmp_path: Path) -> None:
    manager = _seed_db(tmp_path)
    client = HistoricalOddsClient(sandbox_date="2026-03-01", db_manager=manager)

    result = client.get_odds()

    assert result is not None
    assert {odds.home_team for odds in result} == {"Arsenal", "Chelsea"}


def test_get_odds_scopes_to_the_requested_sport_keys_league_bug034(tmp_path: Path) -> None:
    """BUG-034: sport_key was previously accepted but silently ignored --
    this always queried league=E0 regardless of what was requested. A
    sport_key mapping to a different competition (La Liga) must return that
    competition's own odds, not E0's same-date rows."""
    manager = _seed_db(tmp_path)
    client = HistoricalOddsClient(sandbox_date="2026-03-01", db_manager=manager)

    result = client.get_odds(sport_key="soccer_spain_la_liga")

    assert result is not None
    assert len(result) == 1
    assert result[0].home_team == "Barcelona"
    assert result[0].home_odds == 1.40


def test_get_odds_default_sport_key_still_scopes_to_e0(tmp_path: Path) -> None:
    """Regression guard: the pre-BUG-034 default behavior (no sport_key
    passed, or the explicit "soccer_epl" default) must be byte-identical --
    E0 only, SP1's same-date row excluded."""
    manager = _seed_db(tmp_path)
    client = HistoricalOddsClient(sandbox_date="2026-03-01", db_manager=manager)

    result = client.get_odds(sport_key="soccer_epl")

    assert result is not None
    assert {odds.home_team for odds in result} == {"Arsenal", "Chelsea"}


def test_get_odds_unrecognized_sport_key_falls_back_to_e0(tmp_path: Path) -> None:
    """An unmapped/unexpected sport_key degrades to the same E0 default as
    before, rather than returning nothing or raising."""
    manager = _seed_db(tmp_path)
    client = HistoricalOddsClient(sandbox_date="2026-03-01", db_manager=manager)

    result = client.get_odds(sport_key="soccer_some_unknown_league")

    assert result is not None
    assert {odds.home_team for odds in result} == {"Arsenal", "Chelsea"}


def test_odds_lookup_and_match_odds_resolve_historical_client_output_unmodified(tmp_path: Path) -> None:
    """W28's acceptance: a real football-data.org fixture for the sandbox
    date must successfully match to HistoricalOddsClient's output via the
    existing, unmodified odds_lookup()/match_odds() (BUG-015's team-name
    resolution) -- zero changes needed to either function."""
    from app.backend.eod_batch import match_odds, odds_lookup
    from app.backend.football_data_client import NormalizedMatch

    manager = _seed_db(tmp_path)
    client = HistoricalOddsClient(sandbox_date="2026-03-01", db_manager=manager)
    odds_events = client.get_odds()
    odds_by_teams = odds_lookup(odds_events)

    fixture = NormalizedMatch(
        match_id="1", utc_date="2026-03-01T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )

    matched = match_odds(fixture, odds_by_teams)

    assert matched == {"home": 1.80, "draw": 3.60, "away": 4.20}
