"""A124: check_model_staleness() gives non-blocking visibility into whether
a backtest/train report's underlying recorded snapshots still reflect the
currently-promoted model, or one that's since been replaced -- so a reported
ROI number isn't misread as "the currently-promoted model's real
performance" when it's actually months stale. BUG-036/W96's own fingerprint
check only ever covered the live-serving sandbox replay cache, deliberately
never this corpus."""

from __future__ import annotations

from pathlib import Path
import sys

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agent.backtest import BacktestRecord, check_model_staleness


def _record(league: str, target_versions: dict, full_state_override: dict | None = None) -> BacktestRecord:
    full_state = full_state_override if full_state_override is not None else {
        "forecast_payload": {"diagnostics": {"target_versions": target_versions}},
    }
    return BacktestRecord(
        match_id="m1", home_team="A", away_team="B", date="2026-01-01", league=league,
        recommendation={}, actual={}, full_state=full_state,
    )


def _config(tmp_path: Path, contexts: dict) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"contexts": contexts}), encoding="utf-8")
    return str(path)


def test_flags_a_target_whose_recorded_artifact_no_longer_matches_current(tmp_path: Path) -> None:
    config_path = _config(tmp_path, {"E0": {"total_goals": {"model_path": "models/total_goals_v2.joblib"}}})
    records = [_record("E0", {"total_goals": {"artifact": "total_goals_v1.joblib"}})]

    result = check_model_staleness(records, config_path=config_path)

    assert result["matches_checked"] == 1
    assert result["matches_with_a_stale_target"] == 1
    entry = result["by_context_target"]["E0/total_goals"]
    assert entry == {
        "current_artifact": "total_goals_v2.joblib", "fresh": 0, "stale": 1,
        "stale_artifacts_seen": ["total_goals_v1.joblib"],
    }


def test_does_not_flag_a_target_whose_recorded_artifact_still_matches_current(tmp_path: Path) -> None:
    config_path = _config(tmp_path, {"E0": {"total_goals": {"model_path": "models/total_goals_v2.joblib"}}})
    records = [_record("E0", {"total_goals": {"artifact": "total_goals_v2.joblib"}})]

    result = check_model_staleness(records, config_path=config_path)

    assert result["matches_checked"] == 1
    assert result["matches_with_a_stale_target"] == 0
    entry = result["by_context_target"]["E0/total_goals"]
    assert entry["fresh"] == 1
    assert entry["stale"] == 0


def test_a_match_with_multiple_targets_only_needs_one_stale_target_to_count(tmp_path: Path) -> None:
    config_path = _config(tmp_path, {
        "E0": {
            "total_goals": {"model_path": "models/total_goals_v2.joblib"},
            "btts": {"model_path": "models/btts_v1.joblib"},
        },
    })
    records = [_record("E0", {
        "total_goals": {"artifact": "total_goals_v1.joblib"},  # stale
        "btts": {"artifact": "btts_v1.joblib"},  # fresh
    })]

    result = check_model_staleness(records, config_path=config_path)

    assert result["matches_with_a_stale_target"] == 1
    assert result["by_context_target"]["E0/total_goals"]["stale"] == 1
    assert result["by_context_target"]["E0/btts"]["fresh"] == 1


def test_a_target_with_no_current_model_selection_entry_is_reported_but_not_counted_stale(tmp_path: Path) -> None:
    """A target that's since been removed from model_selection.yaml entirely
    (rather than replaced) has nothing to compare against -- report it
    (current_artifact=None) rather than silently dropping it, but don't
    count it toward matches_with_a_stale_target since "no longer promoted
    at all" isn't the same claim as "a newer version is now promoted"."""
    config_path = _config(tmp_path, {"E0": {}})
    records = [_record("E0", {"total_goals": {"artifact": "total_goals_v1.joblib"}})]

    result = check_model_staleness(records, config_path=config_path)

    assert result["matches_with_a_stale_target"] == 0
    entry = result["by_context_target"]["E0/total_goals"]
    assert entry["current_artifact"] is None
    assert entry["fresh"] == 1
    assert entry["stale"] == 0


def test_a_record_with_forecast_payload_explicitly_none_is_skipped_not_errored(tmp_path: Path) -> None:
    """Real shape found live: full_state can have a "forecast_payload" key
    present but set to None (not merely absent) -- a plain .get(key, {})
    would return None here, not the default, and crash the next .get() in
    the chain."""
    config_path = _config(tmp_path, {"E0": {"total_goals": {"model_path": "models/total_goals_v2.joblib"}}})
    records = [_record("E0", {}, full_state_override={"forecast_payload": None})]

    result = check_model_staleness(records, config_path=config_path)

    assert result["matches_checked"] == 0


def test_records_with_no_full_state_are_skipped_not_errored(tmp_path: Path) -> None:
    config_path = _config(tmp_path, {"E0": {"total_goals": {"model_path": "models/total_goals_v2.joblib"}}})
    records = [
        BacktestRecord(match_id="m1", home_team="A", away_team="B", date="2026-01-01", league="E0", recommendation={}, actual={}, full_state=None),
        _record("E0", {"total_goals": {"artifact": "total_goals_v2.joblib"}}),
    ]

    result = check_model_staleness(records, config_path=config_path)

    assert result["matches_checked"] == 1  # only the one with real diagnostics


def test_pools_stale_matches_across_multiple_contexts_independently(tmp_path: Path) -> None:
    config_path = _config(tmp_path, {
        "E0": {"total_goals": {"model_path": "models/tg_e0_v2.joblib"}},
        "SP1": {"total_goals": {"model_path": "models/tg_sp1_v1.joblib"}},
    })
    records = [
        _record("E0", {"total_goals": {"artifact": "tg_e0_v1.joblib"}}),  # stale
        _record("SP1", {"total_goals": {"artifact": "tg_sp1_v1.joblib"}}),  # fresh
    ]

    result = check_model_staleness(records, config_path=config_path)

    assert result["matches_checked"] == 2
    assert result["matches_with_a_stale_target"] == 1
    assert result["by_context_target"]["E0/total_goals"]["stale"] == 1
    assert result["by_context_target"]["SP1/total_goals"]["fresh"] == 1
