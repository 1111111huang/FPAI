"""Fetch Sweden (Allsvenskan) match data from football-data.co.uk's "new" CSV format.

football-data.co.uk publishes a handful of newer leagues (Sweden, Norway,
Finland, Denmark, Ireland, ...) as a single static file per country under
`/new/<CODE>.csv`, covering the full history (2012-present) in one file. This
is structurally different from `FootballDataScraper` (`scraper.py`), which is
built around `englandm.php`'s per-season page-scrape pattern (discover season
links via `SEASON_LINK_PATTERN`, download one CSV per season/league code).
There is nothing to scrape here -- the URL is fixed and the whole history
comes back in one response -- so this is a single HTTP GET rather than
forcing the fetch through the season-link-discovery machinery.

Live-verified 2026-07-21: a plain `requests.get` with no headers returned
HTTP 200 on this occasion, but a browser-like `User-Agent` is set anyway
(matching the convention already used by `src/ingestion/understat/fetcher.py`
and `src/ingestion/fotmob/fetcher.py`) since this endpoint has been observed
to 429 non-browser-like requests (e.g. plain WebFetch-style clients).

This module is fetch-and-parse only. Mapping the returned DataFrame's columns
into the `raw_matches` schema (HG -> fthg, AG -> ftag, AvgCH/AvgCD/AvgCA ->
avgh/avgd/avga-equivalent, league='SWE' tagging, NULLs for columns this
source doesn't provide) is a separate, later step -- see US#125.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

SWEDEN_CSV_URL = "https://www.football-data.co.uk/new/SWE.csv"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.8",
}

# Exact header confirmed live 2026-07-21. Unlike the main englandm.php CSVs
# (two-year season codes like "2526"), Season here is a plain single year
# (e.g. 2012, 2026) -- intentionally not forced into the E0 season-code shape.
EXPECTED_COLUMNS: list[str] = [
    "Country", "League", "Season", "Date", "Time", "Home", "Away", "HG", "AG", "Res",
    "PSCH", "PSCD", "PSCA", "MaxCH", "MaxCD", "MaxCA", "AvgCH", "AvgCD", "AvgCA",
    "BFECH", "BFECD", "BFECA", "B365CH", "B365CD", "B365CA",
]


def fetch_sweden_csv(timeout_seconds: int = 30) -> pd.DataFrame:
    """Download and parse football-data.co.uk's Sweden (Allsvenskan) CSV.

    A single HTTP GET against a fixed URL covering the whole 2012-present
    history -- no page-scraping or season-link discovery needed.

    Returns a DataFrame with the CSV's own columns (see EXPECTED_COLUMNS).
    `Date` is parsed day-first (DD/MM/YYYY, the same convention
    football-data.co.uk uses across all its CSVs -- see the day-first
    handling in `loader.py`'s `process_v1_csv`) into pandas Timestamps.
    Odds columns are frequently blank for recent rows (most commonly PSCH/
    PSCD/PSCA, the Pinnacle closing odds); those are left as NaN rather than
    raising or being silently dropped.

    Raises `requests.HTTPError` (via `raise_for_status`) on a non-2xx
    response, e.g. rate-limiting (429) or a server error -- callers should
    handle/log this the same way other fetchers in this package do.

    Mapping the returned columns into the `raw_matches` schema is handled by
    a later ingestion step, not here -- see US#125.
    """
    LOGGER.info("Fetching Sweden CSV -> %s", SWEDEN_CSV_URL)
    response = requests.get(SWEDEN_CSV_URL, headers=_HEADERS, timeout=timeout_seconds)
    response.raise_for_status()

    # US#130 follow-up fix: `response.text` decodes bytes using `requests`'
    # own encoding guess *before* pandas ever sees the content, so a UTF-8
    # BOM (present on the live file's header line) gets mis-decoded into
    # visible "ï»¿" mojibake prefixed onto the first column name ("ï»¿Country")
    # rather than being stripped -- pandas' BOM-handling only works on the
    # raw bytes it reads itself. Read from `response.content` (raw bytes)
    # with `encoding="utf-8-sig"` instead, which correctly strips the BOM.
    df = pd.read_csv(io.BytesIO(response.content), encoding="utf-8-sig")

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing_columns:
        LOGGER.warning(
            "Sweden CSV missing expected columns: %s (source format may have changed)",
            missing_columns,
        )

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    LOGGER.info("Got %d Sweden match rows", len(df))
    return df
