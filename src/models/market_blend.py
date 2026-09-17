"""Explicit market-probability blending -- a linear shrinkage of the raw
model probability toward the market-implied one, as a tunable alternative
to relying on MKT_IMPLIED_* being effectively used inside a tree ensemble
itself. market_weight=0 is a pure no-op (today's behavior); market_weight=1
means "just use the market's own price." The right value, if any, is an
empirical question answered per-market/per-league by its own real-edge
sweep (see US#200), not assumed here.
"""

from __future__ import annotations

import numpy as np


def blend_with_market(model_proba: np.ndarray, market_proba: np.ndarray, market_weight: float) -> np.ndarray:
    if not 0.0 <= market_weight <= 1.0:
        raise ValueError(f"market_weight must be in [0, 1], got {market_weight}")
    blended = (1 - market_weight) * model_proba + market_weight * market_proba
    return blended / blended.sum(axis=1, keepdims=True)
