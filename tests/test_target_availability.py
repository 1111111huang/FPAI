"""Regression tests for US#129: per-competition target availability.

Sweden's Allsvenskan (football-data.co.uk "New Leagues" CSV, US#124/US#125)
has no `hc`/`ac` (corners) columns at all -- `raw_matches.hc`/`.ac` are NULL
for every row, by design (see US#125's `_NULL_COLUMNS`). Before this story,
`target_registry.py` assumed every competition could train every target;
training `home_corners`/`away_corners`/`total_corners` for such a competition
would reach `ModelManager.prepare_training_data()`, where
`dropna(subset=["target"])` would silently drop every training row -- the
same failure shape as BUG-001 (documents/bugs.md), but on labels instead of
features -- and raise a confusing "No rows left" error, or worse.

These tests use a *fictional* competition ("SWE" in a tmp_path registry, not
the real one -- US#128, Sweden's actual competitions.yaml registration, is a
separate, parallel story) to prove:
  1. The underlying failure path is real (ModelManager.prepare_training_data
     genuinely raises when a target's label columns are 100% NULL).
  2. `CompetitionDefinition.available_targets` / `is_target_available` /
     `config/competitions.yaml` parsing work correctly.
  3. `train-forecast-suite --context SWE` skips exactly the unavailable
     targets with an explicit, readable per-target log reason, while E0
     (no `available_targets` set) is completely unaffected.
  4. `train-target` (single target) fails fast and explicitly rather than
     falling through to the confusing dropna crash.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

import main
from src.logic.competition_registry import (
    CompetitionDefinition,
    get_competition_definition,
    is_target_available,
)
from src.logic.target_registry import list_target_definitions
from src.models.base_model import XGBoostRegressorModel
from src.models.model_manager import ModelManager


# ---------------------------------------------------------------------------
# 1. Prove the underlying failure path is real (BUG-001 shape, on labels).
# ---------------------------------------------------------------------------


def _write_minimal_config(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    model_dir = tmp_path / "models"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path), "model_dir": str(model_dir)}}),
        encoding="utf-8",
    )
    schema_path = tmp_path / "config" / "schema.yaml"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(
        yaml.safe_dump({"training_setup": {"selected_features": ["MKT_IMPLIED_HOME"]}}),
        encoding="utf-8",
    )
    return config_path


def _seed_raw_matches_with_null_corners(config_path: Path) -> None:
    # US#131: raw_matches.league is real in production and prepare_training_data
    # now filters on it (against the real config/competitions.yaml, since this
    # test doesn't pass its own registry_path) -- must be present and match the
    # ModelManager's competition_id ("SWE") below, or the league filter alone
    # would produce zero rows and mask the dropna behavior this test targets.
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    db_path = config["paths"]["database_path"]
    with duckdb.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE raw_matches (
                match_id TEXT PRIMARY KEY, date TIMESTAMP, odds_h FLOAT,
                fthg INTEGER, ftag INTEGER, hc FLOAT, ac FLOAT, league TEXT
            )
            """
        )
        conn.execute('CREATE TABLE feature_store (match_id TEXT PRIMARY KEY, "MKT_IMPLIED_HOME" FLOAT)')
        for i in range(5):
            match_id = f"m{i}"
            conn.execute(
                "INSERT INTO raw_matches VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)",
                [match_id, f"2025-08-{10 + i:02d} 15:00:00", 1.9, 1, 1, "SWE"],
            )
            conn.execute("INSERT INTO feature_store VALUES (?, ?)", [match_id, 0.5])


def test_prepare_training_data_raises_when_corners_labels_are_all_null(tmp_path: Path) -> None:
    """Reproduces the bug this story prevents: a competition whose source
    never populates `hc`/`ac` must not silently train `home_corners` --
    dropna(subset=["target"]) drops every row and prepare_training_data
    raises a clear (if low-level) error. This is exactly the failure the
    upstream `train-forecast-suite`/`train-target` guards added by this
    story exist to keep a caller from ever reaching."""
    config_path = _write_minimal_config(tmp_path)
    _seed_raw_matches_with_null_corners(config_path)

    manager = ModelManager(
        model=XGBoostRegressorModel(),
        config_path=str(config_path),
        target_config={"target": "home_corners"},
        feature_subset=["MKT_IMPLIED_HOME"],
        competition_id="SWE",
    )

    with pytest.raises(ValueError, match="No rows left after dropping records with missing labels or features"):
        manager.prepare_training_data()


# ---------------------------------------------------------------------------
# 2. CompetitionDefinition.available_targets / is_target_available / yaml parsing
# ---------------------------------------------------------------------------


def test_is_target_available_default_none_allows_every_target() -> None:
    definition = CompetitionDefinition(
        competition_id="E0", tier="competition_specific", league_code="E0", enabled_feature_groups=()
    )
    assert definition.available_targets is None
    for target in list_target_definitions():
        assert is_target_available(definition, target.name)


def test_is_target_available_restricts_to_listed_targets() -> None:
    definition = CompetitionDefinition(
        competition_id="SWE",
        tier="competition_specific",
        league_code="SWE",
        enabled_feature_groups=(),
        available_targets=("result_3way", "btts", "home_goals", "away_goals", "total_goals"),
    )
    for corners_target in ("home_corners", "away_corners", "total_corners"):
        assert not is_target_available(definition, corners_target)
    for allowed_target in ("result_3way", "btts", "home_goals", "away_goals", "total_goals"):
        assert is_target_available(definition, allowed_target)


def test_e0_has_no_available_targets_restriction_in_real_registry() -> None:
    """Regression guard: E0's real config/competitions.yaml entry must not
    declare available_targets -- it must keep training every target."""
    definition = get_competition_definition("E0")
    assert definition.available_targets is None


def test_available_targets_parses_from_yaml_and_normalizes_aliases(tmp_path: Path) -> None:
    registry_path = tmp_path / "competitions.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "competitions": {
                    "SWE": {
                        "competition_id": "SWE",
                        "tier": "competition_specific",
                        "league_code": "SWE",
                        "available_targets": ["3way", "both_teams_to_score", "home_goals"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    definition = get_competition_definition("SWE", registry_path=registry_path)
    assert definition.available_targets == ("result_3way", "btts", "home_goals")


def test_available_targets_rejects_unknown_target_name(tmp_path: Path) -> None:
    registry_path = tmp_path / "competitions.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "competitions": {
                    "SWE": {
                        "competition_id": "SWE",
                        "tier": "competition_specific",
                        "league_code": "SWE",
                        "available_targets": ["home_goals", "shots_on_target"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown available_targets"):
        get_competition_definition("SWE", registry_path=registry_path)


# ---------------------------------------------------------------------------
# 3. train-forecast-suite --context SWE: skips exactly the 3 corners targets,
#    with a readable reason per skip; E0 unaffected.
# ---------------------------------------------------------------------------


def _write_sweden_like_registry(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "competitions.yaml").write_text(
        yaml.safe_dump(
            {
                "competitions": {
                    "SWE": {
                        "competition_id": "SWE",
                        "tier": "competition_specific",
                        "league_code": "SWE",
                        "available_targets": [
                            "result_3way",
                            "btts",
                            "home_goals",
                            "away_goals",
                            "total_goals",
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_train_forecast_suite_skips_corners_targets_for_sweden_like_competition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _write_sweden_like_registry(tmp_path)
    monkeypatch.chdir(tmp_path)

    trained: list[str] = []
    monkeypatch.setattr(main, "run_train_target", lambda target_name, context="E0": trained.append(target_name))

    with caplog.at_level(logging.INFO):
        main.run_train_forecast_suite(context="SWE")

    # Acceptance: exactly the 5 available targets are trained.
    assert sorted(trained) == sorted(["result_3way", "btts", "home_goals", "away_goals", "total_goals"])

    # Acceptance: an explicit, readable skip reason is logged for each of the
    # 3 unavailable corners targets -- not a generic exception message.
    skip_messages = [record.getMessage() for record in caplog.records if "SKIPPING" in record.getMessage()]
    assert len(skip_messages) == 3
    for corners_target, label_columns in (
        ("home_corners", "'hc'"),
        ("away_corners", "'ac'"),
        ("total_corners", "'hc'"),
    ):
        matches = [msg for msg in skip_messages if f"target={corners_target} " in msg]
        assert len(matches) == 1, f"expected exactly one skip log line for {corners_target}"
        message = matches[0]
        assert "SWE" in message
        assert label_columns in message
        assert "available_targets" in message


def test_train_forecast_suite_default_context_e0_trains_full_target_set_unaffected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E0 has no available_targets restriction -- this story must not change
    its behavior: it still trains the full default suite (every target
    except legacy home_win)."""
    trained: list[str] = []
    monkeypatch.setattr(main, "run_train_target", lambda target_name, context="E0": trained.append(target_name))

    main.run_train_forecast_suite()  # context defaults to "E0"

    expected = sorted(
        definition.name for definition in list_target_definitions() if definition.name != "home_win"
    )
    assert sorted(trained) == expected
    assert len(trained) == 8
    assert "home_corners" in trained
    assert "away_corners" in trained
    assert "total_corners" in trained


def test_train_forecast_suite_all_requested_targets_unavailable_trains_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If every explicitly-requested target is unavailable, the suite must
    log a clear warning and return -- not silently do nothing with no trace,
    and not crash."""
    _write_sweden_like_registry(tmp_path)
    monkeypatch.chdir(tmp_path)

    trained: list[str] = []
    monkeypatch.setattr(main, "run_train_target", lambda target_name, context="E0": trained.append(target_name))

    with caplog.at_level(logging.INFO):
        main.run_train_forecast_suite(targets=["home_corners", "away_corners"], context="SWE")

    assert trained == []
    assert any("no available targets to train" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# 4. train-target: fails fast and explicitly for a single unavailable target.
# ---------------------------------------------------------------------------


def test_run_train_target_raises_clear_error_for_unavailable_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_sweden_like_registry(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="not available for competition 'SWE'"):
        main.run_train_target("home_corners", context="SWE")
