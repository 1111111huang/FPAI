"""US#162: generalize scraper configuration to accept a list of
(league_page_url, leagues) pairs in a single `scrape` invocation, closing the
gap US#144 deliberately deferred ("only build the list-of-pairs version if a
fourth big-five league is added and running scrape twice per setup becomes a
real operational pain") -- Phase 27 adds three more leagues at once, making
six manual `scrape` invocations per refresh cycle a real cost.

Lightest option (mirrors US#144's own precedent): a new `run_scrape_all()`
that loops over `_SCRAPE_SOURCE_OVERRIDE_BY_LEAGUE` (US#150's existing
per-league page table, now extended with I1/D1/F1) and calls the existing,
already-tested `run_scrape()` once per league -- no new scraper-class code,
`run_scrape()` itself is untouched and byte-for-byte backward compatible."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main


def _settings():
    from src.utils.config_loader import AppSettings

    return AppSettings()


def test_run_scrape_all_scrapes_every_big_five_league_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(main, "run_scrape", lambda *a, **kw: calls.append(kw))

    main.run_scrape_all(_settings())

    assert [c.get("league_page_url") for c in calls] == [
        None,
        "https://www.football-data.co.uk/spainm.php",
        "https://www.football-data.co.uk/italym.php",
        "https://www.football-data.co.uk/germanym.php",
        "https://www.football-data.co.uk/francem.php",
    ]
    assert [c.get("leagues") for c in calls] == [None, ["SP1"], ["I1"], ["D1"], ["F1"]]


def test_run_scrape_all_respects_an_explicit_leagues_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(main, "run_scrape", lambda *a, **kw: calls.append(kw))

    main.run_scrape_all(_settings(), leagues=["I1", "D1"])

    assert len(calls) == 2
    assert calls[0]["league_page_url"] == "https://www.football-data.co.uk/italym.php"
    assert calls[1]["league_page_url"] == "https://www.football-data.co.uk/germanym.php"


def test_run_scrape_all_threads_force_through_to_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(main, "run_scrape", lambda *a, **kw: calls.append(kw))

    main.run_scrape_all(_settings(), force=True, leagues=["E0", "F1"])

    assert all(c["force"] is True for c in calls)


def test_run_scrape_single_page_call_is_completely_unaffected() -> None:
    """The existing single-pair `run_scrape()` override (US#144) must keep
    working byte-for-byte unchanged -- run_scrape_all is purely additive."""
    import inspect

    sig = inspect.signature(main.run_scrape)
    assert list(sig.parameters) == ["app_settings", "force", "league_page_url", "leagues"]
