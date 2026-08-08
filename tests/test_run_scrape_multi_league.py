"""US#144: run_scrape() must accept an optional league_page_url/leagues override
so a second football-data.co.uk page (e.g. spainm.php for La Liga's SP1) can be
scraped without touching config.yaml's own E0 defaults. No new config schema --
an override arg that falls back to the existing config.yaml values when omitted.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from main import run_scrape
from src.utils.config_loader import AppSettings


def _settings() -> AppSettings:
    return AppSettings()


@patch("main.FootballDataScraper")
def test_run_scrape_defaults_to_config_values_when_no_override_given(mock_scraper_cls: MagicMock) -> None:
    mock_scraper = mock_scraper_cls.return_value
    mock_scraper.download_all.return_value = 3

    run_scrape(_settings())

    mock_scraper_cls.assert_called_once_with(
        league_page_url="https://www.football-data.co.uk/englandm.php",
        timeout_seconds=30,
    )
    mock_scraper.download_all.assert_called_once_with(
        limit_seasons=10, leagues=["E0"], start_year=2015, force=False,
    )


@patch("main.FootballDataScraper")
def test_run_scrape_league_page_url_override_reaches_the_scraper(mock_scraper_cls: MagicMock) -> None:
    mock_scraper = mock_scraper_cls.return_value
    mock_scraper.download_all.return_value = 33

    run_scrape(_settings(), league_page_url="https://www.football-data.co.uk/spainm.php")

    mock_scraper_cls.assert_called_once_with(
        league_page_url="https://www.football-data.co.uk/spainm.php",
        timeout_seconds=30,
    )


@patch("main.FootballDataScraper")
def test_run_scrape_leagues_override_reaches_download_all(mock_scraper_cls: MagicMock) -> None:
    mock_scraper = mock_scraper_cls.return_value
    mock_scraper.download_all.return_value = 33

    run_scrape(_settings(), league_page_url="https://www.football-data.co.uk/spainm.php", leagues=["SP1"])

    mock_scraper.download_all.assert_called_once_with(
        limit_seasons=10, leagues=["SP1"], start_year=2015, force=False,
    )


@patch("main.FootballDataScraper")
def test_run_scrape_force_still_threads_through_with_overrides_present(mock_scraper_cls: MagicMock) -> None:
    mock_scraper = mock_scraper_cls.return_value
    mock_scraper.download_all.return_value = 0

    run_scrape(_settings(), force=True, league_page_url="https://www.football-data.co.uk/spainm.php", leagues=["SP1"])

    mock_scraper.download_all.assert_called_once_with(
        limit_seasons=10, leagues=["SP1"], start_year=2015, force=True,
    )
