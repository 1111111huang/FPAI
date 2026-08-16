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

    def test_config_flag_defaults_to_none(self):
        args = parse_args(["2025-03-08", "--precompute"])

        assert args.config is None

    def test_config_flag(self):
        args = parse_args(["2025-03-08", "--precompute", "--config", "config/agent_config_deepseek.yaml"])

        assert args.config == "config/agent_config_deepseek.yaml"


class TestFetchSandboxFixtures:
    """W50: a sandbox date is always in the past, so fixture sourcing must
    go through get_results() (status=FINISHED) -- get_fixtures()
    (status=SCHEDULED) structurally can never return anything for an
    already-played date (the same root cause W45 fixed for /api/fixtures).

    W86: always queries the same unconditional 90-day-forward window
    DashboardPage itself always queries now, not just as an empty-date
    fallback (W51's original design, superseded -- see
    fetch_sandbox_fixtures()'s own docstring for the live-found drift bug
    this replaced)."""

    def test_calls_get_results_with_the_90_day_window_not_get_fixtures(self):
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
            competition_code="PL", date_from="2025-03-08", date_to="2025-06-06",
        )
        fixtures_client.get_fixtures.assert_not_called()

    def test_passes_through_custom_competition_code(self):
        fixtures_client = MagicMock()
        fixtures_client.get_results.return_value = [
            NormalizedMatch(
                match_id="1", utc_date="2025-03-08T15:00:00Z", status="FINISHED",
                home_team="Real Sociedad", away_team="Sevilla", home_goals=1, away_goals=1,
            ),
        ]

        fetch_sandbox_fixtures(fixtures_client, "2025-03-08", competition_code="SP1")

        fixtures_client.get_results.assert_called_once_with(
            competition_code="SP1", date_from="2025-03-08", date_to="2025-06-06",
        )

    def test_no_fixtures_returns_empty_list(self):
        fixtures_client = MagicMock()
        fixtures_client.get_results.return_value = []

        result, used_fallback = fetch_sandbox_fixtures(fixtures_client, "2025-03-08")

        assert result == []
        assert used_fallback is False
        assert fixtures_client.get_results.call_count == 1

    def test_window_sorted_and_capped_at_10(self):
        # 12 out-of-order fixtures (more than the cap of 10) are used so a
        # broken/no-op sort or a missing cap would both be caught -- a list
        # that happens to already be sorted, or exactly 10 long, wouldn't
        # prove either property. Direct user report (2026-08-08): before
        # this test's underlying fix, a league whose exact sandbox date had
        # *some* (but not 0, and not >=10) real matches never queried this
        # window at all, silently leaving the Dashboard's later-dated cards
        # ("next 10" can span many days, W86) uncovered by --precompute.
        fixtures_client = MagicMock()

        def _make(n: int, day: int) -> NormalizedMatch:
            return NormalizedMatch(
                match_id=str(n), utc_date=f"2025-03-{day:02d}T15:00:00Z", status="FINISHED",
                home_team=f"Home{n}", away_team=f"Away{n}", home_goals=1, away_goals=0,
            )

        # Deliberately out of kickoff order.
        unordered_days = [20, 10, 30, 9, 25, 11, 22, 12, 28, 15, 13, 18]
        upcoming = [_make(i, day) for i, day in enumerate(unordered_days)]
        fixtures_client.get_results.return_value = upcoming

        result, used_fallback = fetch_sandbox_fixtures(fixtures_client, "2025-03-08")

        assert used_fallback is False
        assert fixtures_client.get_results.call_count == 1
        assert len(result) == 10
        assert [f.utc_date for f in result] == sorted(f.utc_date for f in result)
        # The two latest (by kickoff) fixtures -- days 28 and 30 -- must have
        # been dropped by the cap, not two arbitrary ones.
        assert {f.match_id for f in result} == {
            str(i) for i, day in enumerate(unordered_days) if day not in (28, 30)
        }


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
        # W140: E0, SP1, I1, D1, and F1 all go through the same mocked
        # FootballDataClient (get_results blanket-returns [fixture]
        # regardless of competition_code) -- SWE finds nothing
        # (historical_results_from_raw_matches mocked to []), so
        # run_eod_batch fires five times, not once.
        assert mock_run_batch.await_count == 5


class TestPrecomputeRecommendationsConfigSelection:
    """`--config` lets --precompute opt into a non-default agent config (e.g.
    config/agent_config_deepseek.yaml) without changing what every other
    entry point (webapp, plain CLI) defaults to -- same opt-in-only
    convention A42 already established. Omitting it must preserve the exact
    prior behavior: AgentConfig.default()."""

    def _run(self, date_str: str, config_path: str | None):
        fixture = NormalizedMatch(
            match_id="1", utc_date=f"{date_str}T15:00:00Z", status="FINISHED",
            home_team="Arsenal", away_team="Everton", home_goals=2, away_goals=1,
        )
        with patch("scripts.launch_sandbox.FootballDataClient") as mock_client_cls, \
             patch("app.backend.scheduler_wiring.build_odds_client", return_value=None), \
             patch("app.backend.recommendations.get_cache", return_value=object()), \
             patch("app.backend.sweden_fixtures_client.historical_results_from_raw_matches", return_value=[]), \
             patch("app.backend.eod_batch.run_eod_batch", new_callable=AsyncMock) as mock_run_batch, \
             patch("src.agent.agent_config.AgentConfig.default") as mock_default, \
             patch("src.agent.agent_config.AgentConfig.from_yaml") as mock_from_yaml:
            mock_client_cls.return_value.get_results.return_value = [fixture]
            mock_run_batch.return_value = EodBatchResult(fixtures=[fixture], generated=1, skipped=0)
            try:
                precompute_recommendations(date_str, config_path=config_path)
            finally:
                os.environ.pop("SANDBOX_MODE", None)
                os.environ.pop("SANDBOX_DATE", None)
        return mock_default, mock_from_yaml

    def test_omitting_config_path_uses_agent_config_default(self):
        mock_default, mock_from_yaml = self._run("2025-03-08", config_path=None)

        mock_default.assert_called_once()
        mock_from_yaml.assert_not_called()

    def test_config_path_loads_that_yaml_instead_of_the_default(self):
        mock_default, mock_from_yaml = self._run(
            "2025-03-08", config_path="config/agent_config_deepseek.yaml"
        )

        mock_from_yaml.assert_called_once_with("config/agent_config_deepseek.yaml")
        mock_default.assert_not_called()


class TestPrecomputeRecommendationsFallbackReporting:
    """W51/W86 code-quality review finding: precompute_recommendations()'s
    status line for what fetch_sandbox_fixtures() found -- covers a fixture
    dated after the exact sandbox date (i.e. only reachable via the 90-day
    window), which the exact-date-only ordering test above doesn't exercise."""

    def test_prints_the_window_message_and_precomputes_a_later_dated_fixture(self, capsys):
        date_str = "2025-03-08"
        later_fixture = NormalizedMatch(
            match_id="99", utc_date="2025-03-14T15:00:00Z", status="FINISHED",
            home_team="Sunderland", away_team="Brighton Hove", home_goals=0, away_goals=1,
        )

        with patch("scripts.launch_sandbox.FootballDataClient") as mock_client_cls, \
             patch("app.backend.scheduler_wiring.build_odds_client", return_value=None), \
             patch("app.backend.recommendations.get_cache", return_value=object()), \
             patch("app.backend.sweden_fixtures_client.historical_results_from_raw_matches", return_value=[]), \
             patch("app.backend.eod_batch.run_eod_batch", new_callable=AsyncMock) as mock_run_batch:

            mock_client_cls.return_value.get_results.return_value = [later_fixture]
            mock_run_batch.return_value = EodBatchResult(fixtures=[later_fixture], generated=1, skipped=0)

            try:
                precompute_recommendations(date_str)
            finally:
                os.environ.pop("SANDBOX_MODE", None)
                os.environ.pop("SANDBOX_DATE", None)

        # W140: E0, SP1, I1, D1, and F1 all go through the same mocked
        # FootballDataClient (keyed only on date range, not competition_code),
        # so all five find the same fixture -- run_eod_batch fires five
        # times, not once. SWE finds nothing (historical_results_from_raw_matches
        # mocked to []) so contributes no call.
        assert mock_run_batch.await_count == 5
        for call in mock_run_batch.await_args_list:
            assert call.kwargs["fixtures"] == [later_fixture]

        output = capsys.readouterr().out
        assert f"1 match(es) found in the next 90 days from {date_str}" in output


class TestPrecomputeRecommendationsBothLeagues:
    """W72 acceptance criterion (documents/app_user_stories.md): a precompute
    run for a date with both real E0 and real SWE fixtures must generate
    recommendations for both leagues, not just E0 -- before this fix,
    precompute_recommendations() only ever looked at football-data.org (E0),
    so a sandbox session's SWE fixtures were silently never precomputed
    regardless of real data availability. W81 extended this to a third
    competition, SP1 -- sharing E0's football-data.org client/mock but
    distinguished by competition_code, so the mock's side_effect keys off
    that kwarg rather than a blanket return_value."""

    def test_precomputes_all_six_leagues_with_correct_league_tags(self):
        """W140: COMPETITIONS grew from 3 (E0/SWE/SP1) to 6 (+I1/D1/F1) --
        confirms the existing competition-list-driven loop extends with zero
        further generalization needed, mirroring W81's own "extends cleanly"
        finding when SP1 was added as the third."""
        date_str = "2025-03-08"
        e0_fixture = NormalizedMatch(
            match_id="1", utc_date="2025-03-08T15:00:00Z", status="FINISHED",
            home_team="Arsenal", away_team="Everton", home_goals=2, away_goals=1,
        )
        swe_fixture = NormalizedMatch(
            match_id="2", utc_date="2025-03-08T17:00:00Z", status="FINISHED",
            home_team="Malmo FF", away_team="AIK", home_goals=1, away_goals=0,
        )
        sp1_fixture = NormalizedMatch(
            match_id="3", utc_date="2025-03-08T19:00:00Z", status="FINISHED",
            home_team="Real Madrid", away_team="Sevilla FC", home_goals=3, away_goals=1,
        )
        i1_fixture = NormalizedMatch(
            match_id="4", utc_date="2025-03-08T14:00:00Z", status="FINISHED",
            home_team="Juventus", away_team="AC Milan", home_goals=1, away_goals=1,
        )
        d1_fixture = NormalizedMatch(
            match_id="5", utc_date="2025-03-08T14:30:00Z", status="FINISHED",
            home_team="Bayern Munich", away_team="Borussia Dortmund", home_goals=2, away_goals=2,
        )
        f1_fixture = NormalizedMatch(
            match_id="6", utc_date="2025-03-08T20:00:00Z", status="FINISHED",
            home_team="Paris Saint-Germain", away_team="Marseille", home_goals=3, away_goals=0,
        )

        _RESULTS_BY_CODE = {
            "PD": [sp1_fixture], "SA": [i1_fixture], "BL1": [d1_fixture], "FL1": [f1_fixture],
        }

        def _get_results(**kwargs):
            return _RESULTS_BY_CODE.get(kwargs.get("competition_code"), [e0_fixture])

        with patch("scripts.launch_sandbox.FootballDataClient") as mock_client_cls, \
             patch("app.backend.scheduler_wiring.build_odds_client", return_value=None), \
             patch("app.backend.recommendations.get_cache", return_value=object()), \
             patch(
                 "app.backend.sweden_fixtures_client.historical_results_from_raw_matches",
                 return_value=[swe_fixture],
             ), \
             patch("app.backend.eod_batch.run_eod_batch", new_callable=AsyncMock) as mock_run_batch:
            mock_client_cls.return_value.get_results.side_effect = _get_results

            def _run_batch(**kwargs):
                return EodBatchResult(fixtures=kwargs["fixtures"], generated=1, skipped=0)

            mock_run_batch.side_effect = _run_batch

            try:
                precompute_recommendations(date_str)
            finally:
                os.environ.pop("SANDBOX_MODE", None)
                os.environ.pop("SANDBOX_DATE", None)

        assert mock_run_batch.await_count == 6
        leagues_called = {call.kwargs["league"] for call in mock_run_batch.await_args_list}
        assert leagues_called == {"E0", "SWE", "SP1", "I1", "D1", "F1"}

        fixtures_by_league = {
            call.kwargs["league"]: call.kwargs["fixtures"] for call in mock_run_batch.await_args_list
        }
        assert fixtures_by_league["E0"] == [e0_fixture]
        assert fixtures_by_league["SWE"] == [swe_fixture]
        assert fixtures_by_league["SP1"] == [sp1_fixture]
        assert fixtures_by_league["I1"] == [i1_fixture]
        assert fixtures_by_league["D1"] == [d1_fixture]
        assert fixtures_by_league["F1"] == [f1_fixture]
