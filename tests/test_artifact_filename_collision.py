"""Regression tests for a real, confirmed bug found live during US#139:
model artifact filenames had no competition_id component at all, so
training two different competitions' models for the same target+model-type
on the same day silently collided on disk.

Confirmed live: training `international`'s `result_3way` (13 MKT-only
features) the same day as `SWE`'s `result_3way` (74 features) overwrote
SWE's already-committed model file with international's content --
`config/model_selection.yaml`'s `contexts.SWE` entry kept pointing at the
now-wrong file, and the corruption went undetected until a forecast was
manually re-verified (`select-best-models`'s own tie-breaking logic doesn't
catch this case, since the "current" and "best" candidates had numerically
identical metrics -- it only checks that a model_path resolves to *some*
existing file, not that the file's content matches what's expected).

Recovered by retraining SWE's affected 5 targets and manually re-pointing
their model_selection.yaml entries to the new, safely-named files. This
module tests the actual fix: `build_artifact_filename()` now includes
competition_id, so this class of collision is structurally prevented going
forward.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.model_manager import build_artifact_filename


def test_different_competitions_produce_different_filenames_same_day() -> None:
    """The exact collision that occurred live: SWE and international training
    the same target with the same model type on the same date."""
    swe_filename = build_artifact_filename("result_3way", "SWE", "xgboost", "20260721")
    intl_filename = build_artifact_filename("result_3way", "international", "xgboost", "20260721")
    e0_filename = build_artifact_filename("result_3way", "E0", "xgboost", "20260721")

    assert swe_filename != intl_filename
    assert swe_filename != e0_filename
    assert intl_filename != e0_filename


def test_e0_filename_shape_is_unchanged_for_backward_compatibility() -> None:
    """E0's filenames must keep their pre-existing unsuffixed shape, since
    every already-recorded model_selection.yaml entry for E0 (and every
    not-yet-updated app/agent reference to an E0 artifact path) assumes it."""
    assert build_artifact_filename("result_3way", "E0", "xgboost", "20260721") == "result_3way_xgboost_v1_20260721.joblib"


def test_none_competition_id_also_collapses_to_unsuffixed() -> None:
    """A caller that never set competition_id at all (competition_id=None)
    gets the same unsuffixed shape as E0, not a literal '_none' suffix."""
    assert build_artifact_filename("btts", None, "lr", "20260721") == "btts_lr_v1_20260721.joblib"


def test_non_e0_competition_gets_a_lowercase_suffix() -> None:
    assert build_artifact_filename("btts", "SWE", "xgboost", "20260721") == "btts_swe_xgboost_v1_20260721.joblib"
    assert build_artifact_filename("btts", "international", "lr", "20260721") == "btts_international_lr_v1_20260721.joblib"
