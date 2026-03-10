"""Automated multi-season data scraping for Football-Data.co.uk."""

from __future__ import annotations

from datetime import datetime
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from src.utils.config_loader import load_settings
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)
SEASON_LINK_PATTERN = re.compile(r"/(?P<season>\d{4})/(?P<file>[A-Za-z0-9]+\.csv)$")
LEAGUE_LABELS = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
}


class FootballDataScraper:
    """Fetch and download historical/current season CSV files from league pages."""

    def __init__(
        self,
        config_path: str = "config.yaml",
        league_page_url: str = "https://www.football-data.co.uk/englandm.php",
        timeout_seconds: int = 30,
    ) -> None:
        """Initialize scraper with downloader session and storage paths."""
        settings = load_settings(config_path)
        self.league_page_url = league_page_url
        self.timeout_seconds = timeout_seconds
        self.raw_data_dir = Path(settings.paths.raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def fetch_csv_urls(self, league_page_url: str) -> list[str]:
        """Fetch all season CSV URLs from a league page, including historical/current."""
        response = self.session.get(league_page_url, timeout=self.timeout_seconds)
        response.raise_for_status()
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError("beautifulsoup4 is required for scraping. Please install requirements.txt.") from exc
        soup = BeautifulSoup(response.text, "html.parser")

        matched_links: list[tuple[str, str]] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            absolute_url = urljoin(league_page_url, href)
            match = SEASON_LINK_PATTERN.search(urlparse(absolute_url).path)
            if not match:
                continue
            season = match.group("season")
            matched_links.append((season, absolute_url))

        if not matched_links:
            LOGGER.warning("No season CSV links found on %s", league_page_url)
            return []

        # Sort newest season first, then stable URL order.
        matched_links.sort(key=lambda item: (item[0], item[1]), reverse=True)
        urls = [url for _, url in matched_links]
        LOGGER.info(
            "Discovered %s CSV links across %s seasons on %s.",
            len(urls),
            len({season for season, _ in matched_links}),
            league_page_url,
        )
        return urls

    @staticmethod
    def _season_start_year_from_code(season_code: str) -> int:
        """Convert season code (e.g., 2324, 9394) into season start year (e.g., 2023, 1993)."""
        start_two_digits = int(season_code[:2])
        current_two_digits = datetime.now().year % 100
        if start_two_digits > current_two_digits + 1:
            return 1900 + start_two_digits
        return 2000 + start_two_digits

    def cleanup_old_raw_files(self, start_year: int = 2015) -> int:
        """Delete local CSV files older than start_year based on *_YYZZ.csv naming."""
        removed = 0
        for csv_file in self.raw_data_dir.glob("*.csv"):
            match = re.search(r"_(\d{4})\.csv$", csv_file.name)
            if not match:
                continue
            season_start = self._season_start_year_from_code(match.group(1))
            if season_start < start_year:
                csv_file.unlink(missing_ok=True)
                removed += 1
        if removed > 0:
            LOGGER.info("Removed %s old CSV files older than %s from %s", removed, start_year, self.raw_data_dir)
        return removed

    def download_all(
        self,
        limit_seasons: int = 5,
        leagues: list[str] | None = None,
        start_year: int = 2015,
    ) -> int:
        """Download selected leagues and seasons; overwrite current season and skip unchanged history."""
        target_leagues = [league.upper() for league in (leagues or ["E0"])]
        self.cleanup_old_raw_files(start_year=start_year)

        all_urls = self.fetch_csv_urls(self.league_page_url)
        if not all_urls:
            return 0

        urls_by_season: dict[str, list[tuple[str, str]]] = {}
        for url in all_urls:
            match = SEASON_LINK_PATTERN.search(urlparse(url).path)
            if not match:
                continue
            season = match.group("season")
            season_start_year = self._season_start_year_from_code(season)
            if season_start_year < start_year:
                continue
            league_code = Path(match.group("file")).stem.upper()
            if league_code not in target_leagues:
                continue
            urls_by_season.setdefault(season, []).append((league_code, url))

        if not urls_by_season:
            LOGGER.warning(
                "No CSV URLs matched filters | leagues=%s | start_year>=%s",
                target_leagues,
                start_year,
            )
            return 0

        selected_seasons = sorted(urls_by_season.keys(), reverse=True)[: max(1, int(limit_seasons))]
        current_season = selected_seasons[0]
        selected_urls: list[tuple[str, str, str]] = []
        for season in selected_seasons:
            for league_code, url in urls_by_season[season]:
                selected_urls.append((season, league_code, url))

        downloaded_files = 0
        skipped_files = 0
        for season, league_code, link in selected_urls:
            match = SEASON_LINK_PATTERN.search(urlparse(link).path)
            if not match:
                continue
            file_name = match.group("file")
            local_name = f"{league_code}_{season}.csv"
            local_path = self.raw_data_dir / local_name

            should_overwrite = season == current_season
            if local_path.exists() and not should_overwrite:
                LOGGER.info("Skipping existing historical file: %s", local_name)
                skipped_files += 1
                continue

            try:
                file_response = self.session.get(link, timeout=self.timeout_seconds)
                if file_response.status_code == 404:
                    LOGGER.error("CSV not found (404): %s", link)
                    skipped_files += 1
                    continue
                file_response.raise_for_status()
                local_path.write_bytes(file_response.content)
                downloaded_files += 1
                LOGGER.info(
                    "Downloaded%s %s (%s) from %s",
                    " (overwritten current season)" if should_overwrite else "",
                    local_name,
                    LEAGUE_LABELS.get(league_code, league_code),
                    link,
                )
            except requests.Timeout:
                LOGGER.error("Timeout while downloading %s", link)
                skipped_files += 1
            except requests.HTTPError:
                LOGGER.exception("HTTP error while downloading %s", link)
                skipped_files += 1
            except requests.RequestException:
                LOGGER.exception("Network error while downloading %s", link)
                skipped_files += 1
            except Exception:
                LOGGER.exception("Unexpected error while handling downloaded CSV %s", local_path)
                skipped_files += 1

        LOGGER.info(
            "Multi-season download finished | leagues=%s | seasons=%s | files_downloaded=%s | files_skipped=%s",
            ",".join(target_leagues),
            len(selected_seasons),
            downloaded_files,
            skipped_files,
        )
        return downloaded_files

    def update_latest_data(self) -> int:
        """Backward-compatible alias to download current/latest season only."""
        return self.download_all(limit_seasons=1, leagues=["E0"], start_year=2015)
