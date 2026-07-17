"""W31: sandbox scenario runbook. Only the argument-guard is unit-tested
here (no real network/LLM calls needed) -- the full live run against a real
historical date is Step 5/6 below, recorded in
documents/sandbox_testing_runbook.md, same as W23's original checklist."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.sandbox_runbook import main


def test_main_exits_with_a_clear_message_when_sandbox_mode_not_set(monkeypatch, capsys) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.delenv("SANDBOX_DATE", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "SANDBOX_MODE=1" in capsys.readouterr().out
