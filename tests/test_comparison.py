"""Tests for config comparison framework (A16)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.agent.comparison import compare_configs, print_comparison_table, save_comparison


def test_compare_configs_runs_each_config_and_collects_reports():
    fake_report_a = {"roi": 0.05, "hit_rate": 0.5, "bet_frequency": 0.2, "max_drawdown": 0.1, "insufficient_data_rate": 0.0}
    fake_report_b = {"roi": -0.02, "hit_rate": 0.4, "bet_frequency": 0.3, "max_drawdown": 0.2, "insufficient_data_rate": 0.1}

    with patch("src.agent.comparison.AgentConfig") as MockCfg, \
         patch("src.agent.comparison.BacktestHarness") as MockHarness, \
         patch("src.agent.comparison.simulate_flat_stake") as mock_stake, \
         patch("src.agent.comparison.build_evaluation_report", side_effect=[fake_report_a, fake_report_b]):
        MockCfg.from_yaml.side_effect = lambda p: MagicMock(name=p)
        instance_a = MagicMock()
        instance_b = MagicMock()
        MockHarness.side_effect = [instance_a, instance_b]
        instance_a.run.return_value = ["rec-a"]
        instance_b.run.return_value = ["rec-b"]
        mock_stake.return_value = MagicMock()

        results = compare_configs(
            ["config/a.yaml", "config/b.yaml"],
            from_date="2025-01-01", to_date="2025-06-01", league="E0", sample=20,
        )

    assert results == {"config/a.yaml": fake_report_a, "config/b.yaml": fake_report_b}
    instance_a.run.assert_called_once_with("2025-01-01", "2025-06-01", league="E0", sample=20)


def test_print_comparison_table_runs_without_error(capsys):
    results = {
        "config/a.yaml": {"roi": 0.05, "hit_rate": 0.5, "bet_frequency": 0.2, "max_drawdown": 0.1, "insufficient_data_rate": 0.0},
    }
    print_comparison_table(results)
    captured = capsys.readouterr()
    assert "config/a.yaml" in captured.out
    assert "roi" in captured.out


def test_save_comparison_writes_json(tmp_path):
    results = {"config/a.yaml": {"roi": 0.05}}
    path = save_comparison(results, base_dir=str(tmp_path))
    assert path.exists()
    assert json.loads(path.read_text()) == results
