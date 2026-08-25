"""Tests for evaluation report computation (A13)."""
from __future__ import annotations

import json

from src.agent.agent_config import AgentConfig
from src.agent.evaluation import (
    build_evaluation_report,
    compute_max_drawdown,
    config_hash,
    save_report,
)
from src.agent.staking import BankrollResult, BetOutcome


def _make_config(**overrides) -> AgentConfig:
    defaults = dict(
        model="llama3.1:8b", provider="ollama", temperature=0.1, max_tool_calls=10,
        min_odds_threshold=1.2, max_odds_threshold=11.0, min_conditional_odds_threshold=1.5, min_value_edge=0.05, markets=["result_3way", "btts"],
        system_prompt_version="v1",
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


def test_compute_max_drawdown_simple_peak_to_trough():
    curve = [1000, 1100, 900, 950, 1200]
    # peak 1100 -> trough 900 = (1100-900)/1100 = 0.1818...
    assert round(compute_max_drawdown(curve), 4) == 0.1818


def test_compute_max_drawdown_no_drawdown_when_monotonic_increase():
    assert compute_max_drawdown([1000, 1100, 1200]) == 0.0


def test_build_evaluation_report_computes_roi_and_hit_rate():
    bankroll = BankrollResult(
        starting_bankroll=1000.0, ending_bankroll=1010.0,
        equity_curve=[1000.0, 1010.0],
        bets=[BetOutcome(match_id="m1", market="result_3way", selection="home", odds=2.0, stake=10.0, won=True, payout=10.0)],
    )

    class _Rec:
        def __init__(self, overall):
            self.recommendation = {"overall": overall}

    records = [_Rec("direct_bet"), _Rec("insufficient_data")]
    report = build_evaluation_report(records, bankroll)

    assert report["bets_placed"] == 1
    assert report["bets_won"] == 1
    assert report["hit_rate"] == 1.0
    assert report["roi"] == 1.0  # 10 profit / 10 staked
    assert report["bet_frequency"] == 0.5  # 1 bet / 2 matches
    assert report["insufficient_data_rate"] == 0.5
    assert report["matches_evaluated"] == 2
    assert report["total_staked"] == 10.0
    assert report["total_profit"] == 10.0


def test_build_evaluation_report_handles_zero_bets():
    bankroll = BankrollResult(starting_bankroll=1000.0, ending_bankroll=1000.0, equity_curve=[1000.0], bets=[])

    class _Rec:
        recommendation = {"overall": "no_bet"}

    report = build_evaluation_report([_Rec()], bankroll)
    assert report["roi"] == 0.0
    assert report["hit_rate"] == 0.0
    assert report["bets_placed"] == 0


def test_build_evaluation_report_total_staked_and_profit_zero_when_no_bets():
    bankroll = BankrollResult(starting_bankroll=1000.0, ending_bankroll=1000.0, equity_curve=[1000.0], bets=[])

    class _Rec:
        recommendation = {"overall": "no_bet"}

    report = build_evaluation_report([_Rec()], bankroll)
    assert report["total_staked"] == 0.0
    assert report["total_profit"] == 0.0


def test_config_hash_deterministic_and_order_independent():
    cfg_a = _make_config(markets=["btts", "result_3way"])
    cfg_b = _make_config(markets=["result_3way", "btts"])
    assert config_hash(cfg_a) == config_hash(cfg_b)
    assert len(config_hash(cfg_a)) == 8


def test_config_hash_differs_for_different_model():
    cfg_a = _make_config(model="llama3.1:8b")
    cfg_b = _make_config(model="llama3.2:3b")
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_save_report_writes_json_file(tmp_path):
    report = {"roi": 0.05, "hit_rate": 0.5}
    path = save_report(report, _make_config(), base_dir=str(tmp_path))
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded == report
