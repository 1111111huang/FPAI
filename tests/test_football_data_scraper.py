"""BUG-062: a transient error fetching football-data.co.uk's own league-listing
page (e.g. a 503, confirmed live 2026-09-08) must not crash the whole
refresh-data chain before it ever reaches run_ingest() -- the actually-needed
step (feature_store schema recompute, see A103/BUG-061's neighbor investigation)
doesn't depend on a fresh scrape succeeding at all, since CSVLoader processes
whatever CSV files already exist on disk from a prior successful scrape.
Per-season CSV downloads already degrade safely this way (download_all's own
try/except a few lines down); only the initial page-listing fetch lacked it."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from src.ingestion.football_data.scraper import FootballDataScraper


def _scraper(tmp_path) -> FootballDataScraper:
    scraper = FootballDataScraper()
    scraper.raw_data_dir = tmp_path
    return scraper


def test_download_all_degrades_to_zero_when_the_league_page_itself_errors(tmp_path):
    scraper = _scraper(tmp_path)
    error_response = MagicMock()
    error_response.raise_for_status.side_effect = requests.HTTPError("503 Server Error")
    with patch.object(scraper.session, "get", return_value=error_response):
        downloaded = scraper.download_all(force=True)
    assert downloaded == 0


def test_download_all_degrades_to_zero_on_a_connection_error(tmp_path):
    """Same degrade-safely contract for a plain network failure (DNS/connection
    refused/timeout), not just an HTTP status error -- both are equally
    transient and equally shouldn't block ingest."""
    scraper = _scraper(tmp_path)
    with patch.object(scraper.session, "get", side_effect=requests.ConnectionError("boom")):
        downloaded = scraper.download_all(force=True)
    assert downloaded == 0
