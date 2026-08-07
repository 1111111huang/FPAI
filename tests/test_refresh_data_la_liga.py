"""Regression tests for US#150: extend refresh-data/schedule-refresh to
include La Liga (SP1).

Part 1 (live investigation, 2026-08-06) tabulated all 3,800 real SP1 rows'
weekdays and found La Liga is weekend-heavy -- 80.6% of matches fall on
Fri/Sat/Sun, actually *higher* than E0's own 77.7% -- and its occasional
full midweek round (e.g. Tue 2025-09-23/Wed 09-24/Thu 09-25) followed
swiftly by the next weekend round (Fri 09-26) is the *same* pattern E0
itself already has (e.g. Tue 2025-12-02/Wed 12-03/Thu 12-04 followed by
Sat 12-06) under its existing, accepted weekly Sunday cadence -- unlike
Sweden's genuinely tighter Allsvenskan schedule (US#132), no new cadence
machinery is warranted for SP1; `schedule-refresh --league SP1` already
falls through to the same weekly-Sunday `build_weekly_refresh_scheduler`
path E0 uses, with no SP1-specific branch needed.

Part 2 found a real, separate gap in `run_refresh_data`'s *scrape* step:
`run_scrape(app_settings, force=force)` is called with no override, so it
always scrapes `config.yaml`'s default page (englandm.php/["E0"]) --
`refresh-data --league SP1` would silently re-scrape E0's page instead of
La Liga's own `spainm.php` (US#144), leaving SP1's current-season CSV
never refreshed and any new results never ingested. Fixed by having
run_refresh_data pass SP1's real page/league override into run_scrape,
mirroring the override mechanism US#144 already built (no new scraper
code needed -- just wiring the existing optional args for a second
caller)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main


def test_run_refresh_data_la_liga_scrapes_the_spain_page_not_the_default_england_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`refresh-data --league SP1` must scrape football-data.co.uk's spainm.php
    page for SP1, not silently fall through to config.yaml's E0/englandm.php
    default -- otherwise SP1's current-season CSV never gets refreshed."""
    scrape_calls: list[dict] = []
    monkeypatch.setattr(main, "run_scrape", lambda *a, **kw: scrape_calls.append(kw))
    monkeypatch.setattr(main, "run_ingest", lambda *a, **kw: None)
    monkeypatch.setattr(main, "run_fetch_understat", lambda *a, **kw: None)
    monkeypatch.setattr(main, "run_fetch_fotmob", lambda *a, **kw: None)
    monkeypatch.setattr(
        "src.ingestion.fotmob.lineup.backfill_lineups_from_player_stats",
        lambda db_manager: 0,
    )

    from src.utils.config_loader import settings as app_settings

    main.run_refresh_data(app_settings, db_manager=None, league="SP1")

    assert len(scrape_calls) == 1
    assert scrape_calls[0]["league_page_url"] == "https://www.football-data.co.uk/spainm.php"
    assert scrape_calls[0]["leagues"] == ["SP1"]


def test_run_refresh_data_epl_still_scrapes_its_own_default_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """E0's own refresh must stay byte-for-byte unaffected -- no override
    passed, config.yaml's default englandm.php/["E0"] still applies."""
    scrape_calls: list[dict] = []
    monkeypatch.setattr(main, "run_scrape", lambda *a, **kw: scrape_calls.append(kw))
    monkeypatch.setattr(main, "run_ingest", lambda *a, **kw: None)
    monkeypatch.setattr(main, "run_fetch_understat", lambda *a, **kw: None)
    monkeypatch.setattr(main, "run_fetch_fotmob", lambda *a, **kw: None)
    monkeypatch.setattr(
        "src.ingestion.fotmob.lineup.backfill_lineups_from_player_stats",
        lambda db_manager: 0,
    )

    from src.utils.config_loader import settings as app_settings

    main.run_refresh_data(app_settings, db_manager=None, league="E0")

    assert len(scrape_calls) == 1
    assert scrape_calls[0].get("league_page_url") is None
    assert scrape_calls[0].get("leagues") is None


def test_schedule_refresh_sp1_uses_the_same_weekly_sunday_default_as_e0() -> None:
    """No SP1-specific scheduler branch -- La Liga's own weekend-heavy
    schedule (80.6%, higher than E0's 77.7%) doesn't need Sweden's tighter
    twice-weekly cadence (US#132); confirm it falls through to the same
    generic weekly-Sunday scheduler E0 uses."""
    from src.scheduling.data_refresh_scheduler import build_weekly_refresh_scheduler

    scheduler = build_weekly_refresh_scheduler(refresh_fn=lambda: None, league="SP1")
    jobs = scheduler.get_jobs()
    assert len(jobs) == 1
    fields = {f.name: str(f) for f in jobs[0].trigger.fields}
    assert fields["day_of_week"] == "sun"
