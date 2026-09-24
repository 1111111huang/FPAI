"""Fetches Transfermarkt player market values via plain requests -- same
pattern as src/ingestion/fotmob/fetcher.py, confirmed live that (unlike
sofifa.com's Cloudflare-protected pages) a plain requests.get() with a
browser User-Agent gets through cleanly, no browser automation needed.
"""

from __future__ import annotations

import re
import time

from bs4 import BeautifulSoup
import requests

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

_UNIT_MULTIPLIERS = {"k": 1_000, "m": 1_000_000, "bn": 1_000_000_000}
_VALUE_PATTERN = re.compile(r"€\s*([\d.,]+)\s*(k|m|bn)", re.IGNORECASE)


def _parse_market_value_eur(text: str) -> int | None:
    """Parses Transfermarkt's own display format (e.g. "€ 220.00 m") into a
    plain integer EUR value. Returns None for text with no recognizable
    value (e.g. a page section that's present but genuinely says nothing,
    or malformed input) -- never raises on this, matching this codebase's
    "degrade, don't crash" convention for optional lookups."""
    match = _VALUE_PATTERN.search(text)
    if match is None:
        return None
    number = float(match.group(1).replace(",", ""))
    multiplier = _UNIT_MULTIPLIERS[match.group(2).lower()]
    return int(number * multiplier)


def fetch_market_value(transfermarkt_id: int, delay: float = 1.0) -> int | None:
    """Returns the player's current market value in EUR, or None if the
    page has no listed value (a real, non-error case confirmed live for
    lower-profile players) -- never raises for this. The URL's slug segment
    ("player") is a placeholder; Transfermarkt redirects to the canonical
    slug based on the numeric ID alone, and requests follows redirects by
    default."""
    url = f"https://www.transfermarkt.com/player/profil/spieler/{transfermarkt_id}"
    response = requests.get(url, headers=_HEADERS, timeout=30)
    response.raise_for_status()
    time.sleep(delay)

    soup = BeautifulSoup(response.text, "html.parser")
    wrapper = soup.find(class_="data-header__market-value-wrapper")
    if wrapper is None:
        return None
    return _parse_market_value_eur(wrapper.get_text(" ", strip=True))
