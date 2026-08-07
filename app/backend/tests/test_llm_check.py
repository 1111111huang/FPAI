"""W01: startup performs a soft reachability check for the configured LLM
provider (Ollama daemon or Anthropic key) and must never raise -- only warn."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import requests

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.llm_check import check_llm_reachable
from src.agent.agent_config import AgentConfig


def _config(provider: str, model: str = "llama3.1:8b") -> AgentConfig:
    return AgentConfig(
        model=model, provider=provider, temperature=0.1, max_tool_calls=10,
        min_odds_threshold=1.2, max_odds_threshold=11.0, min_value_edge=0.05, markets=["result_3way"],
        system_prompt_version="v1",
    )


def test_ollama_reachable_when_daemon_responds() -> None:
    config = _config("ollama")
    with patch("app.backend.llm_check.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        assert check_llm_reachable(config) is True
        mock_get.assert_called_once()


def test_ollama_unreachable_when_connection_fails() -> None:
    config = _config("ollama")
    with patch("app.backend.llm_check.requests.get", side_effect=requests.ConnectionError()):
        assert check_llm_reachable(config) is False


def test_ollama_unreachable_on_non_200() -> None:
    config = _config("ollama")
    with patch("app.backend.llm_check.requests.get") as mock_get:
        mock_get.return_value.status_code = 500
        assert check_llm_reachable(config) is False


def test_anthropic_reachable_when_api_key_present(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    config = _config("anthropic", model="claude-haiku-4-5-20251001")
    assert check_llm_reachable(config) is True


def test_anthropic_unreachable_when_api_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = _config("anthropic", model="claude-haiku-4-5-20251001")
    assert check_llm_reachable(config) is False


def test_deepseek_reachable_when_api_key_present(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    config = _config("deepseek", model="deepseek-chat")
    assert check_llm_reachable(config) is True


def test_deepseek_unreachable_when_api_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config = _config("deepseek", model="deepseek-chat")
    assert check_llm_reachable(config) is False


def test_check_never_raises_on_unexpected_error() -> None:
    config = _config("ollama")
    with patch("app.backend.llm_check.requests.get", side_effect=RuntimeError("boom")):
        assert check_llm_reachable(config) is False
