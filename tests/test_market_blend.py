from __future__ import annotations

import numpy as np
import pytest

from src.models.market_blend import blend_with_market


def test_blend_weight_zero_returns_model_probability_unchanged():
    model_proba = np.array([[0.2, 0.3, 0.5]])
    market_proba = np.array([[0.33, 0.33, 0.34]])
    blended = blend_with_market(model_proba, market_proba, market_weight=0.0)
    np.testing.assert_allclose(blended, model_proba)


def test_blend_weight_one_returns_market_probability_unchanged():
    model_proba = np.array([[0.2, 0.3, 0.5]])
    market_proba = np.array([[0.33, 0.33, 0.34]])
    blended = blend_with_market(model_proba, market_proba, market_weight=1.0)
    np.testing.assert_allclose(blended, market_proba)


def test_blend_rows_still_sum_to_one():
    model_proba = np.array([[0.1, 0.2, 0.7], [0.5, 0.25, 0.25]])
    market_proba = np.array([[0.3, 0.3, 0.4], [0.4, 0.3, 0.3]])
    blended = blend_with_market(model_proba, market_proba, market_weight=0.4)
    np.testing.assert_allclose(blended.sum(axis=1), [1.0, 1.0])


def test_market_weight_out_of_range_raises():
    model_proba = np.array([[0.5, 0.5]])
    market_proba = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError, match="market_weight must be in"):
        blend_with_market(model_proba, market_proba, market_weight=1.5)
