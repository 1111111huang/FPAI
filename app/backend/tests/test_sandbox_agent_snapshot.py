"""W37: wires SnapshotStore record/replay into the sandbox agent-invocation
path. When sandbox mode is active, recommendations.run_agent() records the
first live run of a sandboxed match (date-filtered web_search, per A10) and
replays every subsequent run of the same match (zero live calls) --
otherwise it passes straight through to the real, live run_agent,
unchanged from before this story."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend import recommendations


_MATCH_INFO = {"home_team": "Arsenal", "away_team": "Everton", "date": "2026-03-01"}
_RECOMMENDATION = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-03-01", "league": "E0"},
    "overall": "no_bet", "markets": [], "explanation": "test", "confidence": "low",
    "limitations": [], "prediction_basis": "market_odds_only",
}


@pytest.fixture(autouse=True)
def _reset_sandbox_recording_state():
    recommendations._sandbox_recorded_matches.clear()
    yield
    recommendations._sandbox_recorded_matches.clear()


def test_passes_through_to_the_real_run_agent_when_sandbox_mode_is_off(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    with patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION) as mock_run, \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        result = recommendations.run_agent(_MATCH_INFO)

    assert result == _RECOMMENDATION
    mock_run.assert_called_once_with(_MATCH_INFO, config=None)
    mock_configure.assert_not_called()


def test_first_sandboxed_run_of_a_match_uses_record_mode(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    with patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION), \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        recommendations.run_agent(_MATCH_INFO)

    modes_used = [call.args[0] for call in mock_configure.call_args_list]
    assert modes_used == ["record", "live"]


def test_second_sandboxed_run_of_the_same_match_uses_replay_mode(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    with patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION), \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        recommendations.run_agent(_MATCH_INFO)  # first run: records
        mock_configure.reset_mock()
        recommendations.run_agent(_MATCH_INFO)  # second run: must replay

    modes_used = [call.args[0] for call in mock_configure.call_args_list]
    assert modes_used == ["replay", "live"]


def test_sandboxed_run_uses_the_sandbox_snapshot_namespace(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    with patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION), \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        recommendations.run_agent(_MATCH_INFO)

    record_call = mock_configure.call_args_list[0]
    assert record_call.kwargs["base_dir"] == recommendations._SANDBOX_SNAPSHOT_BASE_DIR


def test_configure_snapshot_store_resets_to_live_and_match_not_recorded_on_agent_exception(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    match_key = f"{_MATCH_INFO['home_team']}__{_MATCH_INFO['away_team']}__{_MATCH_INFO['date']}"
    with patch("app.backend.recommendations._real_run_agent", side_effect=RuntimeError("boom")), \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        with pytest.raises(RuntimeError):
            recommendations.run_agent(_MATCH_INFO)

    modes_used = [call.args[0] for call in mock_configure.call_args_list]
    assert modes_used == ["record", "live"]
    assert match_key not in recommendations._sandbox_recorded_matches
