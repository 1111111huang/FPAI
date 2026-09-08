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


_LISTING_HTML = '<a href="mmz4281/2526/E0.csv">Premier League</a>'


def _ok_response() -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.text = _LISTING_HTML
    return response


def test_fetch_csv_urls_falls_back_to_bare_domain_when_www_host_fails(tmp_path):
    """Confirmed live 2026-09-08: www.football-data.co.uk 503'd while the bare
    apex domain served identical content (same page, same CSV links) with a
    clean 200 -- a real, independent second frontend for the same site, not a
    different data source. Falling back to it -- and resolving the returned
    CSV URLs against *it*, not the failed www host -- turns today's outage
    into a non-event."""
    scraper = _scraper(tmp_path)
    error_response = MagicMock()
    error_response.raise_for_status.side_effect = requests.HTTPError("503 Server Error")
    with patch.object(scraper.session, "get", side_effect=[error_response, _ok_response()]) as mock_get:
        urls = scraper.fetch_csv_urls("https://www.football-data.co.uk/englandm.php")

    assert urls == ["https://football-data.co.uk/mmz4281/2526/E0.csv"]
    assert mock_get.call_args_list[0].args == ("https://www.football-data.co.uk/englandm.php",)
    assert mock_get.call_args_list[1].args == ("https://football-data.co.uk/englandm.php",)


def test_fetch_csv_urls_raises_when_url_has_no_www_to_strip(tmp_path):
    """A failure on a URL that's already bare (or otherwise not a 'www.' host)
    has no fallback to try -- must raise, same as before this change, rather
    than retry the identical URL pointlessly."""
    scraper = _scraper(tmp_path)
    with patch.object(scraper.session, "get", side_effect=requests.HTTPError("503")):
        try:
            scraper.fetch_csv_urls("https://football-data.co.uk/englandm.php")
            assert False, "expected HTTPError to propagate"
        except requests.HTTPError:
            pass


def test_download_all_degrades_to_zero_when_both_www_and_fallback_fail(tmp_path):
    scraper = _scraper(tmp_path)
    error_response = MagicMock()
    error_response.raise_for_status.side_effect = requests.HTTPError("503 Server Error")
    with patch.object(scraper.session, "get", return_value=error_response):
        downloaded = scraper.download_all(force=True)
    assert downloaded == 0
