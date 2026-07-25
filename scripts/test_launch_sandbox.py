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

import os
from unittest.mock import AsyncMock, patch

import duckdb
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.backend.eod_batch import EodBatchResult
from app.backend.football_data_client import NormalizedMatch
from scripts.launch_sandbox import (
    clear_state,
    fetch_sandbox_fixtures,
    find_preflight_info,
    parse_args,
    precompute_recommendations,
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

        result, used_fallback = fetch_sandbox_fixtures(fixtures_client, "2025-03-08")

        assert result == expected
        assert used_fallback is False
        fixtures_client.get_results.assert_called_once_with(
            competition_code="PL", date_from="2025-03-08", date_to="2025-03-08",
        )
        fixtures_client.get_fixtures.assert_not_called()

    def test_passes_through_custom_competition_code(self):
        # Exact date returns a (non-empty) result here so the W51 90-day
        # fallback never fires -- this test is purely about competition_code
        # pass-through on the exact-date query, not the fallback path
        # (covered separately below).
        fixtures_client = MagicMock()
        fixtures_client.get_results.return_value = [
            NormalizedMatch(
                match_id="1", utc_date="2025-03-08T15:00:00Z", status="FINISHED",
                home_team="Real Sociedad", away_team="Sevilla", home_goals=1, away_goals=1,
            ),
        ]

        fetch_sandbox_fixtures(fixtures_client, "2025-03-08", competition_code="SP1")

        fixtures_client.get_results.assert_called_once_with(
            competition_code="SP1", date_from="2025-03-08", date_to="2025-03-08",
        )

    def test_no_fixtures_returns_empty_list(self):
        fixtures_client = MagicMock()
        fixtures_client.get_results.return_value = []

        result, used_fallback = fetch_sandbox_fixtures(fixtures_client, "2025-03-08")

        assert result == []
        assert used_fallback is True

    def test_exact_date_empty_falls_back_to_90_day_window_sorted_and_capped(self):
        # W51: DashboardPage (MatchUI.tsx, W46) falls back to a 90-day
        # forward window, sorted kickoff-ascending and capped at 10, when
        # the exact sandbox date has zero real fixtures. Before this fix,
        # fetch_sandbox_fixtures() only ever queried the exact date, so
        # --precompute silently generated nothing for exactly the dates
        # where the Dashboard's own fallback actually shows the user real
        # matches. 12 out-of-order fixtures (more than the cap of 10) are
        # used so a broken/no-op sort or a missing cap would both be caught
        # -- a list that happens to already be sorted, or exactly 10 long,
        # wouldn't prove either property.
        fixtures_client = MagicMock()

        def _make(n: int, day: int) -> NormalizedMatch:
            return NormalizedMatch(
                match_id=str(n), utc_date=f"2025-03-{day:02d}T15:00:00Z", status="FINISHED",
                home_team=f"Home{n}", away_team=f"Away{n}", home_goals=1, away_goals=0,
            )

        # Deliberately out of kickoff order.
        unordered_days = [20, 10, 30, 9, 25, 11, 22, 12, 28, 15, 13, 18]
        upcoming = [_make(i, day) for i, day in enumerate(unordered_days)]

        def _get_results(**kwargs):
            if kwargs["date_from"] == kwargs["date_to"] == "2025-03-08":
                return []
            assert kwargs == {"competition_code": "PL", "date_from": "2025-03-08", "date_to": "2025-06-06"}
            return upcoming

        fixtures_client.get_results.side_effect = _get_results

        result, used_fallback = fetch_sandbox_fixtures(fixtures_client, "2025-03-08")

        assert used_fallback is True
        assert fixtures_client.get_results.call_count == 2
        assert len(result) == 10
        assert [f.utc_date for f in result] == sorted(f.utc_date for f in result)
        # The two latest (by kickoff) fixtures -- days 28 and 30 -- must have
        # been dropped by the cap, not two arbitrary ones.
        assert {f.match_id for f in result} == {
            str(i) for i, day in enumerate(unordered_days) if day not in (28, 30)
        }

    def test_exact_date_empty_and_fallback_window_also_empty_returns_empty_list(self):
        fixtures_client = MagicMock()
        fixtures_client.get_results.return_value = []

        result, used_fallback = fetch_sandbox_fixtures(fixtures_client, "2025-03-08")

        assert result == []
        assert used_fallback is True
        assert fixtures_client.get_results.call_count == 2


class TestPrecomputeRecommendationsOrdering:
    """W50 code-quality review finding: precompute_recommendations()'s
    critical property is that SANDBOX_MODE/SANDBOX_DATE are set in the
    environment *before* build_odds_client()/get_cache() are constructed
    (both resolve their sandbox-scoped variants by reading os.environ at
    call time), and before run_eod_batch() itself runs -- a later refactor
    could silently reorder this without any of the other unit tests (which
    mock these dependencies individually, not together) catching it. This
    test mocks the full chain and snapshots os.environ at the moment each
    dependency is actually invoked, so a reordering regression fails here
    even though no real network/LLM call happens."""

    def test_sandbox_env_vars_are_set_before_odds_client_cache_and_batch_are_constructed(self):
        date_str = "2025-03-08"
        env_snapshots: dict[str, tuple[str | None, str | None]] = {}

        def _snapshot(name: str):
            env_snapshots[name] = (os.environ.get("SANDBOX_MODE"), os.environ.get("SANDBOX_DATE"))

        fixture = NormalizedMatch(
            match_id="1", utc_date="2025-03-08T15:00:00Z", status="FINISHED",
            home_team="Arsenal", away_team="Everton", home_goals=2, away_goals=1,
        )

        with patch("scripts.launch_sandbox.FootballDataClient") as mock_client_cls, \
             patch("app.backend.scheduler_wiring.build_odds_client") as mock_build_odds, \
             patch("app.backend.recommendations.get_cache") as mock_get_cache, \
             patch("app.backend.sweden_fixtures_client.historical_results_from_raw_matches", return_value=[]), \
             patch("app.backend.eod_batch.run_eod_batch", new_callable=AsyncMock) as mock_run_batch:
            mock_client_cls.return_value.get_results.side_effect = lambda **_: (
                _snapshot("fetch_fixtures"), [fixture]
            )[1]
            mock_build_odds.side_effect = lambda: (_snapshot("build_odds_client"), None)[1]
            mock_get_cache.side_effect = lambda: (_snapshot("get_cache"), object())[1]
            mock_run_batch.side_effect = lambda **_: (
                _snapshot("run_eod_batch"), EodBatchResult(fixtures=[fixture], generated=1, skipped=0)
            )[1]

            try:
                precompute_recommendations(date_str)
            finally:
                os.environ.pop("SANDBOX_MODE", None)
                os.environ.pop("SANDBOX_DATE", None)

        # Every dependency must observe the env vars already set to this
        # date's sandbox values -- not (None, None), not a stale prior value.
        for name in ("fetch_fixtures", "build_odds_client", "get_cache", "run_eod_batch"):
            assert env_snapshots[name] == ("1", date_str), (
                f"{name} was called before SANDBOX_MODE/SANDBOX_DATE were correctly set "
                f"(observed {env_snapshots[name]!r})"
            )
        mock_run_batch.assert_awaited_once()


class TestPrecomputeRecommendationsFallbackReporting:
    """W51 code-quality review finding: precompute_recommendations()'s
    consumption of fetch_sandbox_fixtures()'s `used_fallback` flag -- the
    honest status line distinguishing "N fixtures found for {date}" from
    "no fixtures on {date}, falling back to the next N in the following
    90 days" -- had no test exercising the fallback branch specifically,
    only the exact-date-has-fixtures case (via the ordering test above)."""

    def test_prints_the_fallback_message_and_precomputes_the_fallback_fixtures(self, capsys):
        date_str = "2025-03-08"
        fallback_fixture = NormalizedMatch(
            match_id="99", utc_date="2025-03-14T15:00:00Z", status="FINISHED",
            home_team="Sunderland", away_team="Brighton Hove", home_goals=0, away_goals=1,
        )

        with patch("scripts.launch_sandbox.FootballDataClient") as mock_client_cls, \
             patch("app.backend.scheduler_wiring.build_odds_client", return_value=None), \
             patch("app.backend.recommendations.get_cache", return_value=object()), \
             patch("app.backend.sweden_fixtures_client.historical_results_from_raw_matches", return_value=[]), \
             patch("app.backend.eod_batch.run_eod_batch", new_callable=AsyncMock) as mock_run_batch:

            def _get_results(**kwargs):
                if kwargs["date_from"] == kwargs["date_to"] == date_str:
                    return []
                return [fallback_fixture]

            mock_client_cls.return_value.get_results.side_effect = _get_results
            mock_run_batch.return_value = EodBatchResult(fixtures=[fallback_fixture], generated=1, skipped=0)

            try:
                precompute_recommendations(date_str)
            finally:
                os.environ.pop("SANDBOX_MODE", None)
                os.environ.pop("SANDBOX_DATE", None)

        mock_run_batch.assert_awaited_once()
        assert mock_run_batch.await_args.kwargs["fixtures"] == [fallback_fixture]

        output = capsys.readouterr().out
        assert f"no real fixtures on {date_str}" in output
        assert "falling back to the next 1 match(es) in the following 90 days" in output
        # The exact-date-has-fixtures wording must NOT also appear -- these
        # are mutually exclusive status lines, not both printed.
        assert "real fixture(s) found for" not in output


class TestPrecomputeRecommendationsBothLeagues:
    """W72 acceptance criterion (documents/app_user_stories.md): a precompute
    run for a date with both real E0 and real SWE fixtures must generate
    recommendations for both leagues, not just E0 -- before this fix,
    precompute_recommendations() only ever looked at football-data.org (E0),
    so a sandbox session's SWE fixtures were silently never precomputed
    regardless of real data availability."""

    def test_precomputes_both_leagues_with_correct_league_tags(self):
        date_str = "2025-03-08"
        e0_fixture = NormalizedMatch(
            match_id="1", utc_date="2025-03-08T15:00:00Z", status="FINISHED",
            home_team="Arsenal", away_team="Everton", home_goals=2, away_goals=1,
        )
        swe_fixture = NormalizedMatch(
            match_id="2", utc_date="2025-03-08T17:00:00Z", status="FINISHED",
            home_team="Malmo FF", away_team="AIK", home_goals=1, away_goals=0,
        )

        with patch("scripts.launch_sandbox.FootballDataClient") as mock_client_cls, \
             patch("app.backend.scheduler_wiring.build_odds_client", return_value=None), \
             patch("app.backend.recommendations.get_cache", return_value=object()), \
             patch(
                 "app.backend.sweden_fixtures_client.historical_results_from_raw_matches",
                 return_value=[swe_fixture],
             ), \
             patch("app.backend.eod_batch.run_eod_batch", new_callable=AsyncMock) as mock_run_batch:
            mock_client_cls.return_value.get_results.return_value = [e0_fixture]
            mock_run_batch.return_value = EodBatchResult(fixtures=[e0_fixture], generated=1, skipped=0)

            try:
                precompute_recommendations(date_str)
            finally:
                os.environ.pop("SANDBOX_MODE", None)
                os.environ.pop("SANDBOX_DATE", None)

        assert mock_run_batch.await_count == 2
        leagues_called = {call.kwargs["league"] for call in mock_run_batch.await_args_list}
        assert leagues_called == {"E0", "SWE"}

        fixtures_by_league = {
            call.kwargs["league"]: call.kwargs["fixtures"] for call in mock_run_batch.await_args_list
        }
        assert fixtures_by_league["E0"] == [e0_fixture]
        assert fixtures_by_league["SWE"] == [swe_fixture]
