"""Tests for the pure matching logic in scripts/build_ucl_uel_raw_matches.py
(Task 17): joining OddsPapi's UCL/UEL manifest entries against fotmob's
finished-match dicts by standardized team name + calendar date.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import scripts.build_ucl_uel_raw_matches as build_module
from scripts.build_ucl_uel_raw_matches import match_oddspapi_to_fotmob


def _fotmob_match(match_date_str: str, home: str, away: str, hg: int, ag: int) -> dict:
    return {
        "fotmob_match_id": 123,
        "match_date": pd.Timestamp(match_date_str),
        "home_team": home,
        "away_team": away,
        "home_score": hg,
        "away_score": ag,
    }


def test_exact_match_on_same_date_and_names() -> None:
    manifest = {
        "fid1": {
            "league": "UEL", "date": "2026-09-17", "status": "ok",
            "odds_1x2": {"home": 1.257, "draw": 4.89, "away": 16.32},
            "participant1Name": "Besiktas Istanbul", "participant2Name": "Olympique Marseille",
            "startTime": "2026-09-17T19:00:00.000Z",
        },
    }
    fotmob_by_date = {
        "2026-09-17": [_fotmob_match("2026-09-17", "Besiktas Istanbul", "Olympique Marseille", 2, 1)],
    }

    matched, unmatched = match_oddspapi_to_fotmob(manifest, fotmob_by_date)

    assert unmatched == []
    assert len(matched) == 1
    row = matched[0]
    assert row["league"] == "UEL"
    assert row["date"] == "2026-09-17"
    assert row["home_team"] == "Besiktas Istanbul"
    assert row["away_team"] == "Olympique Marseille"
    assert row["fthg"] == 2
    assert row["ftag"] == 1
    assert row["odds_h"] == 1.257
    assert row["odds_d"] == 4.89
    assert row["odds_a"] == 16.32


def test_name_variant_solved_by_standardize_team_name() -> None:
    """standardize_team_name resolves this via its hand-curated alias map
    ("man utd" -> "Manchester United") -- a real name-variant case it IS
    built to fix, unlike cross-provider international-club spelling
    differences, which it will NOT solve (see module docstring)."""
    manifest = {
        "fid2": {
            "league": "UCL", "date": "2026-10-01", "status": "ok",
            "odds_1x2": {"home": 2.1, "draw": 3.4, "away": 3.2},
            "participant1Name": "Man Utd", "participant2Name": "Real Madrid",
            "startTime": "2026-10-01T20:00:00.000Z",
        },
    }
    fotmob_by_date = {
        "2026-10-01": [_fotmob_match("2026-10-01", "Manchester United", "Real Madrid", 3, 3)],
    }

    matched, unmatched = match_oddspapi_to_fotmob(manifest, fotmob_by_date)

    assert unmatched == []
    assert len(matched) == 1
    assert matched[0]["fthg"] == 3
    assert matched[0]["ftag"] == 3


def test_off_by_one_day_falls_back_to_adjacent_date() -> None:
    """A late-UTC kickoff can land on fotmob's next local day. Manifest date
    2026-09-17 doesn't exist in fotmob_by_date but 2026-09-18 has the match."""
    manifest = {
        "fid3": {
            "league": "UCL", "date": "2026-09-17", "status": "ok",
            "odds_1x2": {"home": 1.5, "draw": 4.0, "away": 6.0},
            "participant1Name": "Ajax", "participant2Name": "Benfica",
            "startTime": "2026-09-17T23:30:00.000Z",
        },
    }
    fotmob_by_date = {
        "2026-09-18": [_fotmob_match("2026-09-18", "Ajax", "Benfica", 1, 1)],
    }

    matched, unmatched = match_oddspapi_to_fotmob(manifest, fotmob_by_date)

    assert unmatched == []
    assert len(matched) == 1
    assert matched[0]["date"] == "2026-09-18"


def test_genuinely_unmatched_entry_is_reported_not_dropped() -> None:
    manifest = {
        "fid4": {
            "league": "UEL", "date": "2026-11-05", "status": "ok",
            "odds_1x2": {"home": 1.8, "draw": 3.6, "away": 4.5},
            "participant1Name": "Some Team", "participant2Name": "Other Team",
            "startTime": "2026-11-05T19:00:00.000Z",
        },
    }
    fotmob_by_date = {
        "2026-11-05": [_fotmob_match("2026-11-05", "Completely Different", "Nobody Else", 0, 0)],
    }

    matched, unmatched = match_oddspapi_to_fotmob(manifest, fotmob_by_date)

    assert matched == []
    assert len(unmatched) == 1
    assert unmatched[0]["fixtureId"] == "fid4"
    assert unmatched[0]["participant1Name"] == "Some Team"
    assert unmatched[0]["participant2Name"] == "Other Team"
    assert unmatched[0]["date"] == "2026-11-05"


def test_crosswalk_alias_resolves_real_unmatched_case() -> None:
    """OddsPapi's "Besiktas Istanbul" vs fotmob's native "Beşiktaş" -- the
    exact real Task 17 unmatched case this crosswalk was built to fix.
    standardize_team_name alone cannot bridge this (different alphabet
    characters, not an alias-map/whitespace/case issue)."""
    manifest = {
        "fid5": {
            "league": "UEL", "date": "2026-09-17", "status": "ok",
            "odds_1x2": {"home": 1.257, "draw": 4.89, "away": 16.32},
            "participant1Name": "Besiktas Istanbul", "participant2Name": "Olympique Marseille",
            "startTime": "2026-09-17T19:00:00.000Z",
        },
    }
    fotmob_by_date = {
        "2026-09-17": [_fotmob_match("2026-09-17", "Beşiktaş", "Marseille", 4, 1)],
    }

    matched, unmatched = match_oddspapi_to_fotmob(manifest, fotmob_by_date)

    assert unmatched == []
    assert len(matched) == 1
    assert matched[0]["home_team"] == "Beşiktaş"
    assert matched[0]["fthg"] == 4


def test_crosswalk_tuple_alias_matches_either_real_fotmob_spelling() -> None:
    """fotmob itself spelled "Atletico Madrid" two different ways across
    real matchdays this task diagnosed (plain "Atletico Madrid" on most
    dates, accented "Atlético Madrid" on 2026-09-09) -- the crosswalk's
    tuple-valued entries must match either confirmed real spelling."""
    manifest = {
        "fid6": {
            "league": "UCL", "date": "2026-09-09", "status": "ok",
            "odds_1x2": {"home": 2.0, "draw": 3.5, "away": 3.8},
            "participant1Name": "Liverpool FC", "participant2Name": "Atletico Madrid",
            "startTime": "2026-09-09T20:00:00.000Z",
        },
    }
    fotmob_by_date = {
        "2026-09-09": [_fotmob_match("2026-09-09", "Liverpool", "Atlético Madrid", 2, 1)],
    }

    matched, unmatched = match_oddspapi_to_fotmob(manifest, fotmob_by_date)

    assert unmatched == []
    assert len(matched) == 1
    assert matched[0]["away_team"] == "Atlético Madrid"


def test_crosswalk_does_not_cause_false_positive_against_wrong_opponent() -> None:
    """A crosswalk alias resolving the home side must not cause a match
    against the WRONG away side -- both must still agree, same conservative
    guarantee as before this crosswalk existed."""
    manifest = {
        "fid7": {
            "league": "UEL", "date": "2026-09-17", "status": "ok",
            "odds_1x2": {"home": 1.257, "draw": 4.89, "away": 16.32},
            "participant1Name": "Besiktas Istanbul", "participant2Name": "Olympique Marseille",
            "startTime": "2026-09-17T19:00:00.000Z",
        },
    }
    fotmob_by_date = {
        # Right home-side spelling (Beşiktaş), but a different away side --
        # must NOT be treated as a match.
        "2026-09-17": [_fotmob_match("2026-09-17", "Beşiktaş", "Some Other Club", 4, 1)],
    }

    matched, unmatched = match_oddspapi_to_fotmob(manifest, fotmob_by_date)

    assert matched == []
    assert len(unmatched) == 1


def test_fetch_all_matches_cached_writes_then_reads_cache(tmp_path) -> None:
    """First call with no cache file hits the (mocked) network and writes a
    cache file; a second call for the same date must NOT call the network
    again, and must return the same match data read back from disk."""
    day = date(2026, 9, 17)
    live_result = [
        {
            "fotmob_match_id": 123,
            "match_date": pd.Timestamp("2026-09-17"),
            "home_team": "Besiktas Istanbul",
            "away_team": "Olympique Marseille",
            "home_score": 2,
            "away_score": 1,
        }
    ]

    with patch.object(build_module, "FOTMOB_CACHE_DIR", tmp_path), \
         patch.object(build_module, "fetch_all_matches", return_value=live_result) as mock_fetch:
        first = build_module._fetch_all_matches_cached(day)
        assert mock_fetch.call_count == 1
        assert first[0]["home_team"] == "Besiktas Istanbul"
        assert (tmp_path / "2026-09-17.json").exists()

        second = build_module._fetch_all_matches_cached(day)
        assert mock_fetch.call_count == 1  # not called again -- served from cache
        assert second[0]["home_team"] == "Besiktas Istanbul"
        assert second[0]["match_date"] == pd.Timestamp("2026-09-17")
