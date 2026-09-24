"""Tests for the Transfermarkt market-value fetcher."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.transfermarkt.fetcher import _parse_market_value_eur, fetch_market_value


def _mock_resp(html: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.text = html
    return resp


def test_parse_market_value_eur_millions():
    assert _parse_market_value_eur("€ 220.00 m Last update: 22/07/2026") == 220_000_000


def test_parse_market_value_eur_thousands():
    assert _parse_market_value_eur("€ 450 k Last update: 22/07/2026") == 450_000


def test_parse_market_value_eur_billions():
    assert _parse_market_value_eur("€ 1.20 bn Last update: 22/07/2026") == 1_200_000_000


def test_fetch_market_value_extracts_from_a_real_shaped_page():
    html = '<div class="data-header__market-value-wrapper">€ 220.00 m Last update: 22/07/2026</div>'
    with patch("src.ingestion.transfermarkt.fetcher.requests.get", return_value=_mock_resp(html)) as mock_get, \
         patch("src.ingestion.transfermarkt.fetcher.time.sleep"):
        result = fetch_market_value(418560)

    assert result == 220_000_000
    args, kwargs = mock_get.call_args
    assert args[0] == "https://www.transfermarkt.com/player/profil/spieler/418560"
    assert "User-Agent" in kwargs.get("headers", {})


def test_fetch_market_value_returns_none_when_no_value_listed():
    """Confirmed live -- not every player has a market value wrapper on
    their page (lower-profile/inactive players). A real, non-error gap."""
    html = "<div>no market value section here</div>"
    with patch("src.ingestion.transfermarkt.fetcher.requests.get", return_value=_mock_resp(html)), \
         patch("src.ingestion.transfermarkt.fetcher.time.sleep"):
        result = fetch_market_value(96148)

    assert result is None
