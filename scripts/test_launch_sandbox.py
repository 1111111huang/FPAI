"""W44: launch_sandbox.py tests. Only the deterministic, non-networked logic
is unit-tested here -- preflight DB queries, state-file read/write/clear,
--stop's process-group teardown, and argument parsing -- matching the
precedent already established by scripts/test_sandbox_runbook.py (only the
argument-guard is tested there; live server launches are verified manually,
recorded in documents/sandbox_testing_runbook.md)."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

import duckdb
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.backend.football_data_client import NormalizedMatch
from scripts.launch_sandbox import (
    clear_state,
    fetch_sandbox_fixtures,
    find_preflight_info,
    parse_args,
    read_state,
    stop_running,
    write_state,
)


@pytest.fixture()
def sample_db(tmp_path) -> Path:
    db_path = tmp_path / "sample.db"
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE raw_matches (league TEXT, date TIMESTAMP, home_team TEXT, away_team TEXT, fthg INTEGER, ftag INTEGER)"
    )
    con.execute(
        "INSERT INTO raw_matches VALUES "
        "('E0', '2025-03-08', 'Nott''m Forest', 'Manchester City', 1, 0), "
        "('E0', '2025-03-08', 'Brighton', 'Fulham', 2, 1), "
        "('E0', '2025-03-15', 'Arsenal', 'Chelsea', 2, 2), "
        "('E0', '2024-11-01', 'Liverpool', 'Everton', 3, 0)"
    )
    con.close()
    return db_path


class TestFindPreflightInfo:
    def test_exact_date_match_returns_its_fixtures(self, sample_db):
        info = find_preflight_info("2025-03-08", db_path=sample_db)

        assert len(info["fixtures"]) == 2
        assert info["nearest_alternative"] is None

    def test_date_with_no_fixtures_suggests_nearest_matchday(self, sample_db):
        info = find_preflight_info("2025-03-05", db_path=sample_db)

        assert info["fixtures"] == []
        assert info["nearest_alternative"] is not None
        assert info["nearest_alternative"]["date"] == "2025-03-08"
        assert len(info["nearest_alternative"]["fixtures"]) == 2

    def test_nearest_matchday_picks_the_closer_of_two_candidates(self, sample_db):
        # 2025-03-10 is 2 days from 2025-03-08 and 5 days from 2025-03-15 -- nearer date wins.
        info = find_preflight_info("2025-03-10", db_path=sample_db)

        assert info["nearest_alternative"]["date"] == "2025-03-08"

    def test_empty_league_with_no_data_at_all_reports_no_alternative(self, sample_db):
        info = find_preflight_info("2025-03-08", db_path=sample_db, league="SP1")

        assert info["fixtures"] == []
        assert info["nearest_alternative"] is None


class TestStateFile:
    def test_write_then_read_round_trips(self, tmp_path):
        state_path = tmp_path / "state.json"
        write_state({"backend_pid": 123, "frontend_pid": 456}, path=state_path)

        assert read_state(path=state_path) == {"backend_pid": 123, "frontend_pid": 456}

    def test_read_missing_file_returns_none(self, tmp_path):
        assert read_state(path=tmp_path / "does_not_exist.json") is None

    def test_clear_removes_the_file(self, tmp_path):
        state_path = tmp_path / "state.json"
        write_state({"backend_pid": 1}, path=state_path)

        clear_state(path=state_path)

        assert not state_path.exists()

    def test_clear_on_missing_file_does_not_raise(self, tmp_path):
        clear_state(path=tmp_path / "does_not_exist.json")  # should be a no-op, not an error


class TestStopRunning:
    def test_no_state_file_reports_nothing_to_stop(self, tmp_path, capsys):
        result = stop_running(state_path=tmp_path / "state.json")

        assert result is False
        assert "nothing to stop" in capsys.readouterr().out.lower()

    def test_kills_both_recorded_process_groups_and_clears_state(self, tmp_path):
        state_path = tmp_path / "state.json"
        write_state({"backend_pid": 111, "frontend_pid": 222}, path=state_path)
        killed = []

        result = stop_running(
            state_path=state_path,
            kill_fn=lambda pgid, sig: killed.append((pgid, sig)),
            getpgid_fn=lambda pid: pid * 10,  # fake pgid, just needs to be traceable
        )

        assert result is True
        assert killed == [(1110, 15), (2220, 15)]  # signal.SIGTERM == 15
        assert not state_path.exists()

    def test_a_process_already_gone_does_not_crash_the_teardown(self, tmp_path):
        state_path = tmp_path / "state.json"
        write_state({"backend_pid": 111, "frontend_pid": 222}, path=state_path)

        def kill_fn(pgid, sig):
            raise ProcessLookupError()

        result = stop_running(state_path=state_path, kill_fn=kill_fn, getpgid_fn=lambda pid: pid)

        assert result is True
        assert not state_path.exists()  # still cleaned up despite both being already-gone


class TestParseArgs:
    def test_date_is_required_unless_stop_is_given(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_stop_alone_is_valid_with_no_date(self):
        args = parse_args(["--stop"])

        assert args.stop is True
        assert args.date is None

    def test_invalid_date_format_is_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["not-a-date"])

    def test_valid_date_parses(self):
        args = parse_args(["2025-03-08"])

        assert args.date == "2025-03-08"
        assert args.dry_run is False

    def test_dry_run_flag(self):
        args = parse_args(["2025-03-08", "--dry-run"])

        assert args.dry_run is True

    def test_custom_ports(self):
        args = parse_args(["2025-03-08", "--backend-port", "9000", "--frontend-port", "4000"])

        assert args.backend_port == 9000
        assert args.frontend_port == 4000

    def test_precompute_flag_defaults_off(self):
        args = parse_args(["2025-03-08"])

        assert args.precompute is False

    def test_precompute_flag(self):
        args = parse_args(["2025-03-08", "--precompute"])

        assert args.precompute is True


class TestFetchSandboxFixtures:
    """W50: a sandbox date is always in the past, so fixture sourcing must
    go through get_results() (status=FINISHED) -- get_fixtures()
    (status=SCHEDULED) structurally can never return anything for an
    already-played date (the same root cause W45 fixed for /api/fixtures)."""

    def test_calls_get_results_not_get_fixtures(self):
        fixtures_client = MagicMock()
        expected = [
            NormalizedMatch(
                match_id="1", utc_date="2025-03-08T15:00:00Z", status="FINISHED",
                home_team="Arsenal", away_team="Everton", home_goals=2, away_goals=1,
            ),
        ]
        fixtures_client.get_results.return_value = expected

        result = fetch_sandbox_fixtures(fixtures_client, "2025-03-08")

        assert result == expected
        fixtures_client.get_results.assert_called_once_with(
            competition_code="PL", date_from="2025-03-08", date_to="2025-03-08",
        )
        fixtures_client.get_fixtures.assert_not_called()

    def test_passes_through_custom_competition_code(self):
        fixtures_client = MagicMock()
        fixtures_client.get_results.return_value = []

        fetch_sandbox_fixtures(fixtures_client, "2025-03-08", competition_code="SP1")

        fixtures_client.get_results.assert_called_once_with(
            competition_code="SP1", date_from="2025-03-08", date_to="2025-03-08",
        )

    def test_no_fixtures_returns_empty_list(self):
        fixtures_client = MagicMock()
        fixtures_client.get_results.return_value = []

        assert fetch_sandbox_fixtures(fixtures_client, "2025-03-08") == []
