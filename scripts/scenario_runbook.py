"""Scenario-testing runbook (W73) -- walks a range of dates, launching the
sandbox and running --precompute for both leagues at each sampled date, and
reports whether each one replayed from the agent-snapshot corpus (W70) or
had to fall back to a live call. Repeatable, evidence-producing check for
"does scenario testing across Aug 2025-May 2026 actually work end-to-end,"
mirroring W31/W44's runbook convention -- not a CI suite (this project has
none), a documented manual/scriptable check.

Each sampled date shells out to scripts/launch_sandbox.py (W44) with
--precompute, which runs the real precompute step (W50/W72, both leagues)
in-process before starting the backend/frontend servers, then this script
tears the launch down with --stop. The precompute step's own stdout+stderr
(uvicorn/frontend logs go to separate files launch_sandbox.py itself
manages -- this script only ever sees what launch_sandbox.py's own process
prints) is parsed for:

  - "Precompute [<league>]: N real fixture(s) found for <date>." / the
    90-day-fallback message (W51) / "nothing to generate." -- what fixtures
    were found, if any (see precompute_recommendations() in
    launch_sandbox.py, which prints all of these verbatim);
  - "Precompute [<league>] complete: generated=G skipped=S of N
    fixture(s)." -- the authoritative per-league tally launch_sandbox.py
    itself reports;
  - "sandbox_agent_replay_miss" -- app/backend/recommendations.py's own log
    line (W43/W70) for a fixture that matched a corpus entry (the W70
    bridge found something to replay) but whose specific replay lookup
    still missed (a non-reproducible tool-call argument, per W43's
    documented limitation) and fell back to a fresh record-mode call.

Its absence does NOT by itself prove every fixture on that date cleanly
replayed -- "generated"/"skipped" can equally come from a fixture with no
corpus entry at all (which logs nothing, live or not), or from the
no-odds insufficient_data path (which never reaches the LLM either way).
recommendations.py has no positive "clean replay succeeded" log line, only
warnings for the miss/failure cases -- so this script surfaces the raw
per-date signal it actually has (fixture counts, generated/skipped tallies,
replay-miss occurrences) rather than asserting a stronger claim than the
evidence supports. Cross-referencing against the corpus coverage report
(see the Step 1 snippet in the W73 plan) is how a human confirms which
mode actually ran for a given date.

Usage:
    python scripts/scenario_runbook.py --from-date 2025-08-01 --to-date 2026-05-31 --sample-every-days 30
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAUNCH_SCRIPT = ROOT / "scripts" / "launch_sandbox.py"

# Exact text launch_sandbox.py's precompute_recommendations() prints (see
# that function for the f-strings these mirror) -- kept as separate,
# narrow patterns rather than one loose match so a fallback-window line
# (which also contains the substring "real fixtures") isn't miscounted as
# an exact-date hit.
_FOUND_RE = re.compile(r"Precompute \[(\w+)\]: (\d+) real fixture\(s\) found for")
_FALLBACK_RE = re.compile(r"Precompute \[(\w+)\]: no real fixtures on \S+ -- falling back to the next (\d+) match\(es\)")
_NOTHING_RE = re.compile(r"Precompute \[(\w+)\]: nothing to generate\.")
_DISCOVERY_FAILED_RE = re.compile(r"Precompute \[(\w+)\]: fixture discovery failed")
_COMPLETE_RE = re.compile(r"Precompute \[(\w+)\] complete: generated=(\d+) skipped=(\d+) of (\d+) fixture\(s\)\.")
_REPLAY_MISS_RE = re.compile(r"sandbox_agent_replay_miss")


def sample_dates(from_date: str, to_date: str, every_days: int) -> list[str]:
    start = date.fromisoformat(from_date)
    end = date.fromisoformat(to_date)
    out = []
    d = start
    while d <= end:
        out.append(d.isoformat())
        d += timedelta(days=every_days)
    return out


def _league_entry(leagues: dict, league: str) -> dict:
    return leagues.setdefault(
        league,
        {
            "fixtures_found": 0,
            "used_fallback": False,
            "discovery_failed": False,
            "generated": 0,
            "skipped": 0,
            "total": 0,
            "reported_complete": False,
        },
    )


def parse_precompute_output(output: str) -> dict:
    """Pure text-parsing of launch_sandbox.py's captured stdout+stderr for
    one sampled date. Separated from run_one_scenario() so it can be unit
    tested against realistic captured text without shelling out to a real
    sandbox launch (see scripts/test_scenario_runbook.py)."""
    leagues: dict = {}

    for m in _FOUND_RE.finditer(output):
        entry = _league_entry(leagues, m.group(1))
        entry["fixtures_found"] = int(m.group(2))

    for m in _FALLBACK_RE.finditer(output):
        entry = _league_entry(leagues, m.group(1))
        entry["fixtures_found"] = int(m.group(2))
        entry["used_fallback"] = True

    for m in _NOTHING_RE.finditer(output):
        _league_entry(leagues, m.group(1))

    for m in _DISCOVERY_FAILED_RE.finditer(output):
        _league_entry(leagues, m.group(1))["discovery_failed"] = True

    for m in _COMPLETE_RE.finditer(output):
        entry = _league_entry(leagues, m.group(1))
        entry["generated"] = int(m.group(2))
        entry["skipped"] = int(m.group(3))
        entry["total"] = int(m.group(4))
        entry["reported_complete"] = True

    replay_miss_count = len(_REPLAY_MISS_RE.findall(output))

    return {"leagues": leagues, "replay_miss_count": replay_miss_count}


def run_one_scenario(date_str: str, timeout: float = 300.0) -> dict:
    """Launches the sandbox for date_str with --precompute, captures
    whether each league found fixtures (exact date or the W51 90-day
    fallback) and the real generated/skipped tally, plus any
    sandbox_agent_replay_miss occurrences, then always stops the sandbox
    afterward (even if the launch itself errored or timed out, so a failed
    date doesn't leave a backend/frontend running for the next one).
    Returns a result dict for the summary report."""
    result: dict = {"date": date_str, "leagues": {}, "replay_miss_count": 0, "errors": []}
    output = ""
    try:
        launch = subprocess.run(
            [sys.executable, str(LAUNCH_SCRIPT), date_str, "--precompute"],
            capture_output=True, text=True, timeout=timeout,
        )
        output = launch.stdout + launch.stderr
        if launch.returncode != 0:
            result["errors"].append(f"launch failed (exit {launch.returncode}): {output[-2000:]}")
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        result["errors"].append(f"launch timed out after {timeout}s")
    finally:
        try:
            stop = subprocess.run(
                [sys.executable, str(LAUNCH_SCRIPT), "--stop"],
                capture_output=True, text=True, timeout=60,
            )
            stop_output = stop.stdout + stop.stderr
            if stop.returncode != 0 and "No recorded sandbox launch found" not in stop_output:
                result["errors"].append(f"--stop reported an error: {stop_output[-500:]}")
        except subprocess.TimeoutExpired:
            result["errors"].append("--stop itself timed out after 60s -- a backend/frontend may still be running.")

    parsed = parse_precompute_output(output)
    result["leagues"] = parsed["leagues"]
    result["replay_miss_count"] = parsed["replay_miss_count"]
    return result


def _format_league(league: str, entry: dict) -> str:
    if entry["discovery_failed"]:
        return f"{league}=discovery_failed"
    tag = "fallback-window" if entry["used_fallback"] else "exact-date"
    if entry["reported_complete"]:
        return f"{league}={entry['fixtures_found']}fx({tag}) generated={entry['generated']} skipped={entry['skipped']}"
    return f"{league}={entry['fixtures_found']}fx({tag}) [no complete tally reported]"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--to-date", required=True)
    parser.add_argument("--sample-every-days", type=int, default=30)
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Per-date subprocess timeout (seconds) for the launch+precompute step.",
    )
    args = parser.parse_args(argv)

    dates = sample_dates(args.from_date, args.to_date, args.sample_every_days)
    print(
        f"Scenario runbook: {len(dates)} sampled date(s) from {args.from_date} to {args.to_date} "
        f"(every {args.sample_every_days}d)."
    )
    results = [run_one_scenario(d, timeout=args.timeout) for d in dates]

    print("\n=== Summary ===")
    total_replay_miss = 0
    for r in results:
        status = "ERROR" if r["errors"] else "OK"
        league_bits = [_format_league(league, r["leagues"][league]) for league in sorted(r["leagues"])]
        total_replay_miss += r["replay_miss_count"]
        replay_note = f" replay_miss={r['replay_miss_count']}" if r["replay_miss_count"] else ""
        print(f"{r['date']}: {' '.join(league_bits) or 'no leagues reported'}{replay_note} [{status}]")
        for err in r["errors"]:
            print(f"  {err}")

    dates_with_replay_miss = sum(1 for r in results if r["replay_miss_count"] > 0)
    error_count = sum(1 for r in results if r["errors"])
    print(
        f"\nTotal sampled dates: {len(results)}. Dates with >=1 sandbox_agent_replay_miss "
        f"(corpus hit found, but the specific replay lookup missed and fell back to a live call): "
        f"{dates_with_replay_miss}. Total replay_miss log lines: {total_replay_miss}."
    )
    if error_count:
        print(f"{error_count} date(s) had errors -- see details above.")


if __name__ == "__main__":
    main()
