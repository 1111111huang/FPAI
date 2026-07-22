"""Tests for the Sweden (Allsvenskan) football-data.co.uk fetch path (US#124).

The `new/SWE.csv` format is a single static file, structurally different from
`FootballDataScraper`'s per-season page-scrape pattern, so it gets its own
fetch function (`fetch_sweden_csv`) in
`src/ingestion/football_data/sweden_fetcher.py`. These tests mock the HTTP
layer with a local fixture CSV -- no live network access in the suite.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.football_data.sweden_fetcher import (
    EXPECTED_COLUMNS,
    SWEDEN_CSV_URL,
    fetch_sweden_csv,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "football_data_sweden_sample.csv"


def _mock_response(text: str, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = text
    # US#130 follow-up fix: fetch_sweden_csv() reads response.content (raw
    # bytes) with encoding="utf-8-sig", not response.text -- see that
    # module's docstring for why (BOM mis-decoding via requests' own text
    # decoding). Mock both so tests exercise the real code path.
    mock.content = text.encode("utf-8-sig")
    if status_code >= 400:
        mock.raise_for_status.side_effect = requests.HTTPError(f"{status_code} error")
    else:
        mock.raise_for_status = MagicMock()
    return mock


def test_fetch_sweden_csv_returns_dataframe_with_expected_shape():
    fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")

    with patch(
        "src.ingestion.football_data.sweden_fetcher.requests.get",
        return_value=_mock_response(fixture_text),
    ) as mock_get:
        df = fetch_sweden_csv()

    mock_get.assert_called_once()
    called_url = mock_get.call_args.args[0]
    assert called_url == SWEDEN_CSV_URL

    assert len(df) == 3
    assert list(df.columns) == EXPECTED_COLUMNS

    last_row = df.iloc[-1]
    assert last_row["Home"] == "Malmo FF"
    assert last_row["Away"] == "Goteborg"
    assert int(last_row["HG"]) == 4
    assert int(last_row["AG"]) == 0
    assert last_row["Res"] == "H"
    assert last_row["Season"] == 2026


def test_fetch_sweden_csv_sends_browser_user_agent():
    """The site 429s plain non-browser-like requests; assert we set a UA."""
    fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")
    captured_headers = {}

    def fake_get(url, headers=None, **kwargs):
        captured_headers.update(headers or {})
        return _mock_response(fixture_text)

    with patch("src.ingestion.football_data.sweden_fetcher.requests.get", side_effect=fake_get):
        fetch_sweden_csv()

    assert "User-Agent" in captured_headers
    assert "Mozilla" in captured_headers["User-Agent"]


def test_fetch_sweden_csv_parses_date_dayfirst():
    fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")

    with patch(
        "src.ingestion.football_data.sweden_fetcher.requests.get",
        return_value=_mock_response(fixture_text),
    ):
        df = fetch_sweden_csv()

    # 12/07/2026 is day-first (12 July), not month-first (which would be invalid: month 12, day 07 -> Dec 7).
    parsed = df.iloc[-1]["Date"]
    assert isinstance(parsed, pd.Timestamp)
    assert parsed == pd.Timestamp(year=2026, month=7, day=12)


def test_fetch_sweden_csv_tolerates_blank_odds_columns():
    """PSCH/PSCD/PSCA (Pinnacle closing odds) are frequently blank for recent
    rows; the fetch must not crash and must surface them as NaN, not as a
    parse error or a dropped row."""
    fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")

    with patch(
        "src.ingestion.football_data.sweden_fetcher.requests.get",
        return_value=_mock_response(fixture_text),
    ):
        df = fetch_sweden_csv()

    # Rows 2 and 3 in the fixture have blank PSCH/PSCD/PSCA.
    assert pd.isna(df.iloc[1]["PSCH"])
    assert pd.isna(df.iloc[1]["PSCD"])
    assert pd.isna(df.iloc[1]["PSCA"])
    assert pd.isna(df.iloc[2]["PSCH"])

    # Row 1 has populated PSCH/PSCD/PSCA -- confirm they parse as floats, not NaN.
    assert df.iloc[0]["PSCH"] == pytest.approx(2.10)

    # No rows should be dropped just because some odds columns are blank.
    assert len(df) == 3

    # MaxC*/AvgC*/BFEC*/B365C* are always populated per the documented format;
    # confirm they parse cleanly even on rows where PSC* is blank.
    assert df.iloc[1]["AvgCH"] == pytest.approx(3.30)
    assert df.iloc[2]["B365CA"] == pytest.approx(3.3)


def test_fetch_sweden_csv_raises_on_http_error():
    """A bad response (e.g. 429 rate-limit) should propagate as an
    HTTPError, matching the convention used by the understat/fotmob
    fetchers (raise_for_status, no silent swallow)."""
    with patch(
        "src.ingestion.football_data.sweden_fetcher.requests.get",
        return_value=_mock_response("", status_code=429),
    ):
        with pytest.raises(requests.HTTPError):
            fetch_sweden_csv()


def test_fetch_sweden_csv_warns_but_does_not_crash_on_missing_columns(caplog):
    """If the source format changes (columns renamed/removed), the fetch
    should still return a DataFrame and log a warning rather than raising --
    silent format drift should be visible, not fatal."""
    truncated_csv = "Country,League,Season,Date,Home,Away,HG,AG\nSweden,Allsvenskan,2026,12/07/2026,Malmo FF,Goteborg,4,0\n"

    with patch(
        "src.ingestion.football_data.sweden_fetcher.requests.get",
        return_value=_mock_response(truncated_csv),
    ):
        df = fetch_sweden_csv()

    assert len(df) == 1
    assert "PSCH" not in df.columns
