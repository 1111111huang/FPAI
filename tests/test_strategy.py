from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.strategy.strategy_engine import StrategyEngine


def test_calculate_ev_for_half_probability_and_2_2_odds() -> None:
    engine = StrategyEngine()
    ev = engine.calculate_ev(win_prob=0.5, odds=2.2)
    assert ev == pytest.approx(0.10)
