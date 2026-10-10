from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.date_range_utils import contiguous_date_ranges, date_range


def test_date_range_single_day():
    assert date_range("2026-08-21", "2026-08-21") == ["2026-08-21"]


def test_date_range_multi_day_inclusive():
    assert date_range("2026-08-21", "2026-08-23") == ["2026-08-21", "2026-08-22", "2026-08-23"]


def test_contiguous_date_ranges_empty_list():
    assert contiguous_date_ranges([]) == []


def test_contiguous_date_ranges_single_span():
    assert contiguous_date_ranges(["2026-08-21", "2026-08-22", "2026-08-23"]) == [("2026-08-21", "2026-08-23")]


def test_contiguous_date_ranges_splits_on_a_gap():
    assert contiguous_date_ranges(["2026-08-21", "2026-08-22", "2026-08-24"]) == [
        ("2026-08-21", "2026-08-22"), ("2026-08-24", "2026-08-24"),
    ]
