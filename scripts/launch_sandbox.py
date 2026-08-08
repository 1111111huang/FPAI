#!/usr/bin/env python3
"""W44: one-command launch of the app in sandbox mode for a given date --
preflight (real-data availability) + backend + frontend, so a human can
browse the real UI against a chosen historical "today" without manually
repeating the launch steps in documents/sandbox_testing_runbook.md by hand
every time.

Usage:
    python scripts/launch_sandbox.py 2025-03-08
    python scripts/launch_sandbox.py 2025-03-08 --dry-run   # preflight only, no servers
    python scripts/launch_sandbox.py --stop                  # tear down a running launch

Complements scripts/sandbox_runbook.py (W31), which drives the backend-only
fixtures->recommendation->bet->settlement flow with no server/browser
involved. This script is for interactive use: it boots real HTTP servers so
a human can click through the actual frontend in a browser. Verifying what
renders is still a manual step -- this script does not drive a browser.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import datetime as datetime_mod
from datetime import date as date_cls
from pathlib import Path

import duckdb
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# Allows `python scripts/launch_sandbox.py` (script dir on sys.path, not
# ROOT) to import app.backend/src modules directly for the --precompute
# step below -- same convention scripts/sandbox_runbook.py (W31) already
# uses.
sys.path.append(str(ROOT))

from app.backend.football_data_client import FootballDataClient, NormalizedMatch  # noqa: E402
from app.backend.football_data_competition_codes import FOOTBALL_DATA_CODE_BY_LEAGUE  # noqa: E402

DB_PATH = ROOT / "data" / "fpai_core.db"
SNAPSHOT_DIR = ROOT / "data" / "agent_snapshots" / "sandbox"
STATE_FILE = Path("/tmp/fpai_sandbox_launch_state.json")
BACKEND_LOG = Path("/tmp/fpai_sandbox_backend.log")
FRONTEND_LOG = Path("/tmp/fpai_sandbox_frontend.log")

DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_PORT = 3000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("date", nargs="?", help="Date to treat as 'today', YYYY-MM-DD (required unless --stop)")
    parser.add_argument("--dry-run", action="store_true", help="Run the preflight check only, launch nothing")
    parser.add_argument("--stop", action="store_true", help="Tear down a previously launched sandbox instance")
    parser.add_argument(
        "--precompute", action="store_true",
        help="Before starting servers, pre-populate the sandbox recommendation cache for all of the date's "
             "real fixtures using the same batch-generation logic as the live EOD job (odds-fetching + "
             "per-match fault tolerance included), so cards are already populated when the browser opens "
             "instead of each needing an individual click-and-wait. Opt-in: makes real network/LLM calls "
             "and can take several minutes for a full matchday.",
    )
    parser.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT)
    parser.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT)
    parser.add_argument("--startup-timeout", type=float, default=45.0, help="Seconds to wait for each server to become healthy")
    parser.add_argument(
        "--config", default=None,
        help="Agent config YAML for the --precompute step (e.g. config/agent_config_deepseek.yaml). "
             "Defaults to AgentConfig.default() (config/agent_config.yaml) when omitted -- matches every "
             "other entry point's default, DeepSeek/other providers stay opt-in per-invocation.",
    )
    args = parser.parse_args(argv)

    if not args.stop and not args.date:
        parser.error("a date argument is required unless --stop is given")
    if args.date:
        try:
            date_cls.fromisoformat(args.date)
        except ValueError:
            parser.error(f"invalid date {args.date!r} -- expected YYYY-MM-DD")
    return args


def _to_plain_date(value) -> date_cls:
    """DuckDB returns TIMESTAMP columns as datetime.datetime -- normalize to
    a plain date so day-difference arithmetic against a date_cls works."""
    if isinstance(value, datetime_mod.datetime):
        return value.date()
    return value


# ---------------------------------------------------------------------------
# Preflight: real-data availability + existing-snapshot report (no live calls)
# ---------------------------------------------------------------------------

def find_preflight_info(date_str: str, db_path: Path = DB_PATH, league: str = "E0") -> dict:
    """Reports what real data exists for `date_str` without making any live
    calls. If the exact date has no fixtures, suggests the nearest date that
    does, so a human isn't left guessing why the app looks empty."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        exact = con.execute(
            "SELECT home_team, away_team, fthg, ftag FROM raw_matches "
            "WHERE league = ? AND date = ? ORDER BY home_team",
            [league, date_str],
        ).fetchall()

        nearest = None
        if not exact:
            all_dates = con.execute(
                "SELECT DISTINCT date FROM raw_matches WHERE league = ?", [league],
            ).fetchall()
            target = date_cls.fromisoformat(date_str)
            if all_dates:
                distinct_dates = {_to_plain_date(row[0]) for row in all_dates}
                closest_date = min(distinct_dates, key=lambda d: abs((d - target).days))
                closest_date_str = closest_date.isoformat()
                nearest_fixtures = con.execute(
                    "SELECT home_team, away_team, fthg, ftag FROM raw_matches "
                    "WHERE league = ? AND date = ? ORDER BY home_team",
                    [league, closest_date_str],
                ).fetchall()
                nearest = {"date": closest_date_str, "fixtures": nearest_fixtures}
    finally:
        con.close()

    return {
        "date": date_str,
        "league": league,
        "fixtures": exact,
        "nearest_alternative": nearest,
    }


def count_existing_snapshots(date_str: str, fixtures: list[tuple], snapshot_dir: Path = SNAPSHOT_DIR) -> int:
    """How many snapshot directories already exist for this date, regardless
    of which fixture they belong to. NOTE: this is a best-effort count, not
    an exact per-fixture match -- snapshot directory names are keyed on
    whatever team-name spelling the live caller (the frontend, via
    football-data.org's naming) used at record time, which does not
    reliably match raw_matches's own Football-Data CSV spelling (e.g.
    "Nott'm Forest" vs "Nottingham Forest" vs "Nottingham" -- all three
    exist as real, different snapshot-directory prefixes for the same team
    in this repo's own snapshot corpus). A nonzero count means *some*
    recommendation call for this date will replay rather than hit a live
    LLM; it does not promise which fixture."""
    if not snapshot_dir.exists():
        return 0
    return sum(1 for d in snapshot_dir.iterdir() if d.is_dir() and d.name.endswith(f"__{date_str}"))


def print_preflight_report(info: dict) -> None:
    date_str, fixtures = info["date"], info["fixtures"]
    print(f"=== Preflight for {date_str} ({info['league']}) ===")
    if fixtures:
        print(f"{len(fixtures)} real fixture(s) found:")
        for home, away, fthg, ftag in fixtures:
            score = f"{fthg}-{ftag}" if fthg is not None else "not yet played"
            print(f"  {home} vs {away} ({score})")
        existing = count_existing_snapshots(date_str, fixtures)
        if existing:
            print(f"{existing} snapshot directory(ies) already recorded for {date_str} -- some recommendation calls may replay instead of hitting a live LLM (exact fixture-to-snapshot match isn't reliable across team-name spelling variants).")
        else:
            print("No existing snapshots for this date -- recommendation calls will be real, live LLM calls.")
    else:
        print(f"No real {info['league']} fixtures on {date_str}.")
        alt = info["nearest_alternative"]
        if alt:
            print(f"Nearest matchday with data: {alt['date']} ({len(alt['fixtures'])} fixture(s)):")
            for home, away, fthg, ftag in alt["fixtures"][:5]:
                score = f"{fthg}-{ftag}" if fthg is not None else "not yet played"
                print(f"  {home} vs {away} ({score})")
        else:
            print(f"No {info['league']} data exists in raw_matches at all.")
    print()


# ---------------------------------------------------------------------------
# Environment checks (non-blocking -- warnings only)
# ---------------------------------------------------------------------------

def check_environment() -> dict:
    load_dotenv(ROOT / ".env")
    ollama_reachable = False
    try:
        ollama_reachable = requests.get("http://localhost:11434/api/tags", timeout=2).ok
    except requests.RequestException:
        ollama_reachable = False
    return {
        "ollama_reachable": ollama_reachable,
        "football_data_api_key": bool(os.environ.get("FOOTBALL_DATA_API_KEY")),
        "tavily_api_key": bool(os.environ.get("TAVILY_API_KEY")),
    }


def print_environment_report(env: dict) -> None:
    print("=== Environment ===")
    print(f"  Ollama reachable:          {'yes' if env['ollama_reachable'] else 'NO -- recommendation generation will fail'}")
    print(f"  FOOTBALL_DATA_API_KEY set: {'yes' if env['football_data_api_key'] else 'NO -- fixtures will fail to load'}")
    print(f"  TAVILY_API_KEY set:        {'yes' if env['tavily_api_key'] else 'NO -- agent web_search will report unavailable'}")
    print()


# ---------------------------------------------------------------------------
# State file (PID tracking, so --stop only ever kills what this script started)
# ---------------------------------------------------------------------------

def write_state(state: dict, path: Path = STATE_FILE) -> None:
    path.write_text(json.dumps(state))


def read_state(path: Path = STATE_FILE) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def clear_state(path: Path = STATE_FILE) -> None:
    path.unlink(missing_ok=True)


def stop_running(state_path: Path = STATE_FILE, kill_fn=os.killpg, getpgid_fn=os.getpgid) -> bool:
    """Reads the recorded launch state and terminates both process groups.
    Returns False (and prints a message) if there's nothing recorded."""
    state = read_state(state_path)
    if state is None:
        print("No recorded sandbox launch found (nothing to stop).")
        return False

    for name in ("backend_pid", "frontend_pid"):
        pid = state.get(name)
        if pid is None:
            continue
        try:
            kill_fn(getpgid_fn(pid), signal.SIGTERM)
            print(f"Stopped {name.replace('_pid', '')} (pid {pid}).")
        except ProcessLookupError:
            print(f"{name.replace('_pid', '')} (pid {pid}) was already stopped.")

    clear_state(state_path)
    return True


# ---------------------------------------------------------------------------
# Launch + health-check polling
# ---------------------------------------------------------------------------

def wait_for_http(url: str, timeout: float, interval: float = 0.5, get_fn=requests.get) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if get_fn(url, timeout=2).status_code < 500:
                return True
        except requests.RequestException:
            pass
        time.sleep(interval)
    return False


def launch_backend(date_str: str, port: int) -> subprocess.Popen:
    env = {**os.environ, "SANDBOX_MODE": "1", "SANDBOX_DATE": date_str}
    log = BACKEND_LOG.open("w")
    return subprocess.Popen(
        ["uvicorn", "app.backend.main:app", "--port", str(port)],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def launch_frontend(backend_port: int, frontend_port: int) -> subprocess.Popen:
    env = {**os.environ, "NEXT_PUBLIC_API_BASE": f"http://localhost:{backend_port}"}
    log = FRONTEND_LOG.open("w")
    return subprocess.Popen(
        ["npm", "run", "dev", "--", "-p", str(frontend_port)],
        cwd=ROOT / "app" / "frontend", env=env, stdout=log, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


# ---------------------------------------------------------------------------
# W50: opt-in precompute -- runs the sandbox date's real fixtures through
# the same batch-generation logic the live EOD job (W09) uses, before the
# servers start, so the frontend doesn't open to an empty "Not yet
# generated" state.
# ---------------------------------------------------------------------------

def fetch_sandbox_fixtures(
    fixtures_client: FootballDataClient, date_str: str, competition_code: str = "PL",
) -> tuple[list[NormalizedMatch], bool]:
    """A sandbox date is always in the past relative to real wall-clock time
    (SANDBOX_DATE names a historical scenario by construction), so unlike
    main.py's _split_fixture_date_range (W45 -- a general-purpose endpoint
    that must handle a range spanning past/today/future), sourcing here is
    unconditionally get_results() (status=FINISHED): get_fixtures()
    (status=SCHEDULED) structurally can never return anything for an
    already-played date. Same root cause W45 fixed for /api/fixtures,
    applied here for this script's own fixture sourcing -- see
    eod_batch.py's run_eod_batch() `fixtures` parameter (W50), which this
    feeds.

    W51: if the exact date genuinely has zero real fixtures, falls back to
    the same 90-day-forward window DashboardPage itself falls back to
    (MatchUI.tsx, W46) -- query the exact date first; if that's empty,
    query date_str..date_str+90d, sort by kickoff ascending (utc_date is
    already ISO 8601, so a plain string sort is correct -- same as the
    frontend's `.sort((a, b) => a.kickoffIso.localeCompare(b.kickoffIso))`),
    and cap at the same 10 matches the Dashboard renders. Keep this in sync
    with DashboardPage's own fallback in MatchUI.tsx if either changes --
    a matching comment lives there pointing back here.

    Returns `(fixtures, used_fallback)`: `used_fallback` lets
    precompute_recommendations() print an accurate status line for either
    case without issuing a second, duplicate network call of its own."""
    exact = fixtures_client.get_results(competition_code=competition_code, date_from=date_str, date_to=date_str)
    if exact:
        return exact, False

    to_date = (date_cls.fromisoformat(date_str) + datetime_mod.timedelta(days=90)).isoformat()
    upcoming = fixtures_client.get_results(competition_code=competition_code, date_from=date_str, date_to=to_date)
    return sorted(upcoming, key=lambda m: m.utc_date)[:10], True


def fetch_sandbox_fixtures_swe(date_str: str) -> tuple[list[NormalizedMatch], bool]:
    """W72: SWE analogue of fetch_sandbox_fixtures(), sourced from W71's
    historical_results_from_raw_matches() instead of FootballDataClient --
    SwedenFixturesClient's own get_results() can't serve an arbitrary past
    date at all (see W71), so there's no equivalent single-client call to
    parameterize the way the E0 version does."""
    from app.backend.sweden_fixtures_client import historical_results_from_raw_matches

    exact = historical_results_from_raw_matches(date_str, date_str)
    if exact:
        return exact, False
    to_date = (date_cls.fromisoformat(date_str) + datetime_mod.timedelta(days=90)).isoformat()
    upcoming = historical_results_from_raw_matches(date_str, to_date)
    return sorted(upcoming, key=lambda m: m.utc_date)[:10], True


def _fetch_sandbox_fixtures_for_league(
    league: str, fixtures_client: FootballDataClient, date_str: str,
) -> tuple[list[NormalizedMatch], bool]:
    """W72/W76/W81: dispatches to the right per-league fetch function, mirroring
    scheduler_wiring.py's _fetch_fixtures_for_league shape (same
    per-competition dispatch idea), adapted for precompute's need to also
    track the used_fallback flag for its status-line reporting. Any league
    other than SWE is football-data.org-sourced (E0, SP1) -- looked up via
    FOOTBALL_DATA_CODE_BY_LEAGUE rather than fetch_sandbox_fixtures()'s own
    "PL" default, or SP1 would silently fetch E0's fixtures instead."""
    if league == "SWE":
        return fetch_sandbox_fixtures_swe(date_str)
    competition_code = FOOTBALL_DATA_CODE_BY_LEAGUE.get(league, "PL")
    return fetch_sandbox_fixtures(fixtures_client, date_str, competition_code=competition_code)


def precompute_recommendations(date_str: str, config_path: str | None = None) -> None:
    """Pre-populates the sandbox-scoped RecommendationCache (W29) for every
    real fixture on `date_str` across every competition in
    scheduler_wiring.COMPETITIONS (E0 via fetch_sandbox_fixtures(), SWE via
    W72's fetch_sandbox_fixtures_swe() -- or, per W51, the nearest real
    fixtures in the following 90 days if `date_str` itself has none for a
    given league), using run_eod_batch()'s exact generation/odds-matching/
    fault-tolerance logic (W09, benefiting from W49's odds-attachment fix)
    -- the same code path the live EOD batch uses, just fed this date's
    already-played fixtures instead of tomorrow's scheduled ones. W62's
    `league` parameter is passed through per-league, and per-league fixture
    discovery is wrapped in its own try/except (mirroring
    scheduler_wiring.py's _eod_job) so one competition's fetch failure
    (e.g. a football-data.org outage for E0) doesn't prevent the other
    competition -- which doesn't even depend on that client -- from
    running.

    Sets SANDBOX_MODE/SANDBOX_DATE in this process's own environment (mirroring
    launch_backend()'s subprocess env) so recommendations.get_cache() and
    scheduler_wiring.build_odds_client() route to the same sandbox-scoped
    cache/odds source the backend subprocess will use once it starts --
    results land in the same on-disk cache file, so they're already there
    when the backend (started afterward, in main()) opens it.

    No T-30 job is scheduled here (schedule_t30 is a no-op) -- there's no
    long-lived scheduler process behind this one-shot CLI step, and a
    sandbox fixture's kickoff is always already in the past, so a T-30
    "refresh before kickoff" has nothing left to do.

    Backend modules (agent/LLM/ML dependencies) are imported here, not at
    module load time, so a plain launch/--stop/--dry-run invocation that
    never passes --precompute doesn't pay for importing them.

    `config_path`, when given, loads that YAML instead of AgentConfig's own
    default (e.g. config/agent_config_deepseek.yaml) -- same opt-in-only
    convention A42 already established for the standalone CLI (`--config` on
    any agent-* command), now threaded through this script's own --precompute
    step too. Omitting it preserves the exact prior default (AgentConfig.default(),
    whatever config/agent_config.yaml currently says)."""
    os.environ["SANDBOX_MODE"] = "1"
    os.environ["SANDBOX_DATE"] = date_str

    from app.backend import recommendations
    from app.backend.eod_batch import run_eod_batch
    from app.backend.scheduler_wiring import COMPETITIONS, build_odds_client
    from src.agent.agent_config import AgentConfig

    odds_client = build_odds_client()
    cache = recommendations.get_cache()
    config = AgentConfig.from_yaml(config_path) if config_path else AgentConfig.default()
    fixtures_client = FootballDataClient(api_key=os.environ.get("FOOTBALL_DATA_API_KEY", ""))

    for league in COMPETITIONS:
        try:
            fixtures, used_fallback = _fetch_sandbox_fixtures_for_league(league, fixtures_client, date_str)
        except Exception:
            print(f"Precompute [{league}]: fixture discovery failed -- skipping this competition, others unaffected.")
            continue

        if used_fallback:
            print(
                f"Precompute [{league}]: no real fixtures on {date_str} -- falling back to the next "
                f"{len(fixtures)} match(es) in the following 90 days (same window/cap DashboardPage itself "
                "falls back to, W46/W51)."
            )
        else:
            print(f"Precompute [{league}]: {len(fixtures)} real fixture(s) found for {date_str}.")
        if not fixtures:
            print(f"Precompute [{league}]: nothing to generate.")
            continue

        tally = {"generated": 0, "skipped": 0}

        def _on_progress(
            fixture: NormalizedMatch, outcome: str, _league=league, _tally=tally, _total=len(fixtures),
        ) -> None:
            _tally[outcome] += 1
            done = _tally["generated"] + _tally["skipped"]
            print(
                f"Precompute [{_league}]: [{done}/{_total}] {fixture.home_team} vs {fixture.away_team}: "
                f"{outcome} (generated={_tally['generated']} skipped={_tally['skipped']})"
            )

        result = asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda fixture: None, date_str=date_str, fixtures=fixtures,
                on_progress=_on_progress, league=league,
            )
        )
        print(
            f"Precompute [{league}] complete: generated={result.generated} skipped={result.skipped} "
            f"of {len(fixtures)} fixture(s)."
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.stop:
        stop_running()
        return

    info = find_preflight_info(args.date)
    print_preflight_report(info)

    if args.dry_run:
        return

    env = check_environment()
    print_environment_report(env)

    if read_state() is not None:
        print("A sandbox launch is already recorded as running. Run with --stop first, or check "
              f"{STATE_FILE} manually if that seems wrong.")
        sys.exit(1)

    if not (ROOT / "app" / "frontend" / "node_modules").exists():
        print("app/frontend/node_modules is missing -- run `npm install` in app/frontend first.")
        sys.exit(1)

    if args.precompute:
        print(f"=== Precompute (SANDBOX_DATE={args.date}) === -- this can take several minutes")
        precompute_recommendations(args.date, config_path=args.config)
        print()

    print(f"Starting backend on :{args.backend_port} (SANDBOX_DATE={args.date}) -- log: {BACKEND_LOG}")
    backend = launch_backend(args.date, args.backend_port)
    if not wait_for_http(f"http://localhost:{args.backend_port}/api/health", args.startup_timeout):
        print(f"Backend did not become healthy within {args.startup_timeout}s -- check {BACKEND_LOG}")
        os.killpg(os.getpgid(backend.pid), signal.SIGTERM)
        sys.exit(1)

    status = requests.get(f"http://localhost:{args.backend_port}/api/sandbox/status", timeout=5).json()
    if status.get("as_of") != args.date:
        print(f"Warning: backend reports as_of={status.get('as_of')!r}, expected {args.date!r}")

    print(f"Starting frontend on :{args.frontend_port} -- log: {FRONTEND_LOG}")
    frontend = launch_frontend(args.backend_port, args.frontend_port)
    if not wait_for_http(f"http://localhost:{args.frontend_port}", args.startup_timeout):
        print(f"Frontend did not become healthy within {args.startup_timeout}s -- check {FRONTEND_LOG}")
        os.killpg(os.getpgid(backend.pid), signal.SIGTERM)
        os.killpg(os.getpgid(frontend.pid), signal.SIGTERM)
        sys.exit(1)

    write_state({
        "date": args.date,
        "backend_pid": backend.pid,
        "frontend_pid": frontend.pid,
        "backend_port": args.backend_port,
        "frontend_port": args.frontend_port,
        "started_at": datetime_mod.datetime.now().isoformat(),
    })

    print()
    print("=== Sandbox running ===")
    print(f"  Frontend: http://localhost:{args.frontend_port}")
    print(f"  Backend:  http://localhost:{args.backend_port}")
    print(f"  As of:    {args.date}")
    print(f"  Stop with: python scripts/launch_sandbox.py --stop")


if __name__ == "__main__":
    main()
