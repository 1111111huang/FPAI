"""Config comparison framework: re-run BacktestHarness for multiple configs
over the identical (seeded) match sample so the only varying factor is the
agent config itself (A16)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent.agent_config import AgentConfig
from src.agent.backtest import BacktestHarness
from src.agent.evaluation import build_evaluation_report
from src.agent.staking import simulate_flat_stake, simulate_kelly_stake


def compare_configs(
    config_paths: list[str],
    from_date: str,
    to_date: str,
    league: str | None = None,
    sample: int | None = None,
    stake_mode: str = "flat",
) -> dict[str, dict[str, Any]]:
    """Run each config's agent over the same match set (same from_date/to_date/league/
    sample -> BacktestHarness._stratified_sample's fixed random_state=42 guarantees an
    identical sample across configs) and return {config_path: evaluation_report}."""
    stake_fn = simulate_kelly_stake if stake_mode == "kelly" else simulate_flat_stake
    results: dict[str, dict[str, Any]] = {}
    for path in config_paths:
        cfg = AgentConfig.from_yaml(path)
        harness = BacktestHarness(config=cfg)
        records = harness.run(from_date, to_date, league=league, sample=sample)
        bankroll_result = stake_fn(records)
        results[path] = build_evaluation_report(records, bankroll_result)
    return results


def print_comparison_table(results: dict[str, dict[str, Any]]) -> None:
    metrics = ["roi", "hit_rate", "bet_frequency", "max_drawdown", "insufficient_data_rate"]
    header = f"{'config':<40}" + "".join(f"{m:>16}" for m in metrics)
    print(header)
    print("-" * len(header))
    for path, report in results.items():
        row = f"{path:<40}" + "".join(f"{report.get(m, ''):>16}" for m in metrics)
        print(row)


def save_comparison(results: dict[str, dict[str, Any]], base_dir: str = "reports/agent_backtest") -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"comparison_{timestamp}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path
