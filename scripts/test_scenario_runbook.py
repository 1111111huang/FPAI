"""Unit tests for scripts/scenario_runbook.py (W73).

Scoped like scripts/test_launch_sandbox.py: the pure, testable surface here
is sample_dates() and parse_precompute_output() -- text parsing of
launch_sandbox.py's captured stdout+stderr. run_one_scenario() itself
shells out to a real sandbox launch and is exercised live by the runbook,
not here (same reasoning sandbox_runbook.py's own single smoke test uses --
most of this script's value is in a real, evidence-producing run, not a
mocked one)."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from scenario_runbook import parse_precompute_output, sample_dates  # noqa: E402


class TestSampleDates:
    def test_single_day_range_returns_just_that_day(self) -> None:
        assert sample_dates("2025-08-01", "2025-08-01", 30) == ["2025-08-01"]

    def test_steps_by_every_days_and_includes_the_start(self) -> None:
        assert sample_dates("2025-08-01", "2025-09-15", 30) == ["2025-08-01", "2025-08-31"]

    def test_does_not_overshoot_past_to_date(self) -> None:
        dates = sample_dates("2025-08-01", "2025-08-29", 30)
        assert dates == ["2025-08-01"]

    def test_full_plan_range_at_30_days_yields_a_meaningful_sample_count(self) -> None:
        # Guards against silently degenerating to too few samples to be a
        # meaningful check (the plan calls for at least ~4-5 across the
        # full Aug 2025-May 2026 range).
        dates = sample_dates("2025-08-01", "2026-05-31", 30)
        assert len(dates) >= 5
        assert dates[0] == "2025-08-01"
        assert dates[-1] <= "2026-05-31"


class TestParsePrecomputeOutput:
    def test_empty_output_returns_no_leagues_and_no_replay_misses(self) -> None:
        parsed = parse_precompute_output("")
        assert parsed["leagues"] == {}
        assert parsed["replay_miss_count"] == 0

    def test_real_captured_output_e0_and_swe_exact_date_both_skipped(self) -> None:
        # Real output captured from a live `launch_sandbox.py 2025-08-16
        # --precompute` run during this task's Step 1/3 investigation
        # (Anthropic billing was exhausted, so every fixture that reached
        # the LLM came back "skipped").
        output = (
            "sandbox_agent_replay_miss | match=335337e9 | retrying_in_record_mode\n"
            "sandbox_agent_replay_miss | match=7c4d9fab | retrying_in_record_mode\n"
            "Precompute [E0]: 5 real fixture(s) found for 2025-08-16.\n"
            "Precompute [E0]: [1/5] Sunderland vs West Ham: skipped (generated=0 skipped=1)\n"
            "Precompute [E0]: [5/5] Wolverhampton vs Man City: skipped (generated=0 skipped=5)\n"
            "Precompute [E0] complete: generated=0 skipped=5 of 5 fixture(s).\n"
            "Precompute [SWE]: 2 real fixture(s) found for 2025-08-16.\n"
            "Precompute [SWE] complete: generated=0 skipped=2 of 2 fixture(s).\n"
        )
        parsed = parse_precompute_output(output)
        assert parsed["replay_miss_count"] == 2
        assert parsed["leagues"]["E0"] == {
            "fixtures_found": 5, "used_fallback": False, "discovery_failed": False,
            "generated": 0, "skipped": 5, "total": 5, "reported_complete": True,
        }
        assert parsed["leagues"]["SWE"]["generated"] == 0
        assert parsed["leagues"]["SWE"]["skipped"] == 2

    def test_fallback_window_line_is_not_miscounted_as_an_exact_date_hit(self) -> None:
        output = (
            "Precompute [E0]: no real fixtures on 2026-06-15 -- falling back to the next 3 "
            "match(es) in the following 90 days (same window/cap DashboardPage itself falls "
            "back to, W46/W51).\n"
            "Precompute [E0] complete: generated=3 skipped=0 of 3 fixture(s).\n"
        )
        parsed = parse_precompute_output(output)
        entry = parsed["leagues"]["E0"]
        assert entry["used_fallback"] is True
        assert entry["fixtures_found"] == 3
        assert entry["generated"] == 3

    def test_nothing_to_generate_records_the_league_with_zero_fixtures(self) -> None:
        output = (
            "Precompute [SWE]: no real fixtures on 2025-08-01 -- falling back to the next 0 "
            "match(es) in the following 90 days.\n"
            "Precompute [SWE]: nothing to generate.\n"
        )
        parsed = parse_precompute_output(output)
        entry = parsed["leagues"]["SWE"]
        assert entry["fixtures_found"] == 0
        assert entry["reported_complete"] is False

    def test_fixture_discovery_failure_is_flagged_per_league(self) -> None:
        output = "Precompute [E0]: fixture discovery failed -- skipping this competition, others unaffected.\n"
        parsed = parse_precompute_output(output)
        assert parsed["leagues"]["E0"]["discovery_failed"] is True

    def test_no_complete_line_at_all_reports_fixtures_found_but_no_tally(self) -> None:
        # Simulates a truncated/timed-out capture -- the fixtures-found
        # line printed before the process was killed, but the "complete"
        # summary line never got a chance to print.
        output = "Precompute [E0]: 4 real fixture(s) found for 2025-09-01.\n"
        parsed = parse_precompute_output(output)
        entry = parsed["leagues"]["E0"]
        assert entry["fixtures_found"] == 4
        assert entry["reported_complete"] is False
