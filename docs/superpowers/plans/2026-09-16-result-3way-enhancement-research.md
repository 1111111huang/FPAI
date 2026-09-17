# Model Enhancement Research Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the `result_3way`/`total_goals`/`btts` real-edge gaps found in `US#173`/`US#192`/`US#193` by trying three externally-inspired, internally-adapted ideas — extending FPAI's own already-validated Skellam architecture, an explicit market-blend post-processor, and ordinal regression for `result_3way` — each independently gated on real, out-of-sample edge, not in-sample metrics.

**Architecture:** Three independent phases, ordered by confidence (highest first). Each phase produces a candidate artifact via `ModelManager`/`train-target` (reusing `refit_on_full_data`), then runs the same real-edge-on-qualifying-bets validation script (`scripts/validate_real_edge.py`, built in Phase 0) before any promotion. A phase failing its gate is a valid, documented outcome — it stops there, it does not force a promotion.

**Tech Stack:** Existing `FPAIBaseModel`/`ModelManager`/`ModelFactory` (no new serving-layer code needed for Phase 1), `statsmodels` (new dependency, Phase 3 only), same `raw_matches`/`feature_store` data already in use.

---

## Scope note

This covers three genuinely independent subsystems (a model swap, a post-processing blend, a new model class). Per the scope-check convention, they're kept as one plan rather than three because they share one validation script and one motivating investigation (`US#173`/`US#192`/`US#193`) — but each phase is independently executable, independently abandonable, and produces working software on its own. Skip Phase 2/3 entirely if Phase 1 alone closes enough of the gap.

---

## Phase 0: Shared Validation Script

Every phase below is gated by this script. Build it once, reuse three times.

**Files:**
- Create: `scripts/validate_real_edge.py`
- Test: `tests/test_validate_real_edge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate_real_edge.py
from __future__ import annotations

import pandas as pd
import pytest

from scripts.validate_real_edge import compute_real_edge


def test_real_edge_is_zero_when_calibrated_perfectly():
    """A model whose predicted prob always equals the real outcome rate,
    and the market's implied prob always matches too, has zero real edge --
    no false signal from a trivial all-agree case."""
    df = pd.DataFrame({
        "predicted_prob": [0.30, 0.30, 0.30],
        "implied_prob": [0.30, 0.30, 0.30],
        "actual": [1, 0, 0],  # 1/3 = 0.30ish outcome rate is close enough for this synthetic case
    })
    result = compute_real_edge(df, "predicted_prob", "implied_prob", "actual", edge_threshold=0.05)
    assert result["n_qualifying"] == 0  # no edge >= 5% claimed anywhere, nothing to grade


def test_real_edge_flags_a_qualifying_bet_and_grades_it_correctly():
    df = pd.DataFrame({
        "predicted_prob": [0.50, 0.20],
        "implied_prob": [0.30, 0.20],
        "actual": [1, 0],
    })
    result = compute_real_edge(df, "predicted_prob", "implied_prob", "actual", edge_threshold=0.05)
    assert result["n_qualifying"] == 1
    assert result["claimed_edge"] == pytest.approx(0.20)
    assert result["actual_rate"] == pytest.approx(1.0)
    assert result["real_edge"] == pytest.approx(0.70)  # actual_rate(1.0) - implied_prob(0.30)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_validate_real_edge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.validate_real_edge'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/validate_real_edge.py
"""Real-edge-on-qualifying-bets validation, extracted from the ad-hoc checks
run against US#173/US#192/US#193 into one reusable script. A model claims
an edge whenever predicted_prob - implied_prob >= edge_threshold; this
reports whether that claim survives contact with the real outcome rate,
not whether an in-sample/aggregate metric looks good. See
documents/user_stories.md's Phase 38 for the methodology this formalizes.
"""

from __future__ import annotations

import pandas as pd


def compute_real_edge(
    df: pd.DataFrame, predicted_col: str, implied_col: str, actual_col: str, edge_threshold: float = 0.05,
) -> dict[str, float | int]:
    """actual_col must be 0/1 (1 = the predicted selection actually happened)."""
    edge = df[predicted_col] - df[implied_col]
    qualifying = df[edge >= edge_threshold]
    n = len(qualifying)
    if n == 0:
        return {"n_qualifying": 0, "claimed_edge": float("nan"), "actual_rate": float("nan"), "real_edge": float("nan")}
    claimed_edge = (qualifying[predicted_col] - qualifying[implied_col]).mean()
    actual_rate = qualifying[actual_col].mean()
    real_edge = actual_rate - qualifying[implied_col].mean()
    return {"n_qualifying": n, "claimed_edge": float(claimed_edge), "actual_rate": float(actual_rate), "real_edge": float(real_edge)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_validate_real_edge.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_real_edge.py tests/test_validate_real_edge.py
git commit -m "test: extract real-edge validation into a reusable script

Formalizes the ad-hoc checks run against US#173/US#192/US#193 so Phases
1-3 of the model enhancement research plan share one validation gate.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Phase 1: Extend Skellam to SP1/D1/I1/F1 (highest confidence — already validated in this repo)

`E0`'s `result_3way` already uses `SkellamResultModel` (a distributional stack over `home_goals`/`away_goals`, not a class-weighted softmax) and was the *only* league that didn't reproduce the draw-miscalibration problem in the `US#173` real-edge check. All 4 other leagues already have promoted `home_goals`/`away_goals` models (`SkellamResultModel._load_submodel` reads them from `model_selection.yaml` per `competition_id` — confirmed live, no new code needed). This phase is pure validation, not new engineering.

**Files:**
- No new files. Reuses `src/models/skellam_result_model.py` and `main.py train-target` as-is.
- Test: none new — `SkellamResultModel` is already covered by `tests/test_skellam_result_model.py`.

- [ ] **Step 1: Train the SP1 pilot candidate**

Run:
```bash
python3 main.py train-target --target result_3way --context SP1 --model skellam_result --refit-full-data
```
Expected: logs `result_3way: refit on full data (train+val+test, 3800 rows through 2026-05-24T00:00:00) for serving` and saves `models/result_3way_skellamresult_v1_<today>.joblib`.

- [ ] **Step 2: Pull real held-out predictions from the new candidate and grade with Phase 0's script**

```python
# one-off, run via python3 -c or a notebook -- not committed, this is validation not shipped code
import sys, json; sys.path.insert(0, ".")
import pandas as pd
from src.models.skellam_result_model import SkellamResultModel
from src.utils.db_manager import DuckDBManager
from scripts.validate_real_edge import compute_real_edge

db = DuckDBManager()
with db.connection(read_only=True) as conn:
    df = conn.execute("""
        SELECT r.match_id, r.date, r.fthg, r.ftag, r.odds_d
        FROM raw_matches r INNER JOIN feature_store f ON r.match_id=f.match_id
        WHERE r.league='SP1' ORDER BY r.date, r.match_id
    """).fetchdf()
df = df.dropna(subset=["fthg", "ftag"]).reset_index(drop=True)
total = len(df); train_end = int(total * 0.7); val_end = min(max(train_end + 1, int(total * 0.85)), total - 1)
test = df.iloc[val_end:].reset_index(drop=True)
test["actual_draw"] = (test["fthg"] == test["ftag"]).astype(int)
test["implied_draw"] = 1 / test["odds_d"]

model = SkellamResultModel.load("models/result_3way_skellamresult_v1_<today>.joblib")  # fill in the real date_tag from Step 1
needed = list(set(model._home_feature_columns) | set(model._away_feature_columns))
with db.connection(read_only=True) as conn:
    ids = "','".join(test["match_id"])
    fdf = conn.execute(f"SELECT match_id, {', '.join(needed)} FROM feature_store WHERE match_id IN ('{ids}')").fetchdf()
fdf = fdf.set_index("match_id").loc[test["match_id"]].apply(pd.to_numeric, errors="coerce")
proba = model.predict_proba(fdf)
test["pred_draw"] = proba[:, 1]

result = compute_real_edge(test, "pred_draw", "implied_draw", "actual_draw", edge_threshold=0.05)
print(result)
```

**Decision gate:** `real_edge` must be within ±3pp of 0 (matching E0's already-validated behavior) on n >= 20 qualifying matches. `US#173`'s existing weighted-XGBoost baseline for SP1 was **+0.01pp on n=63** — this candidate must not be meaningfully worse than that, and should ideally show the same non-inflated draw share E0 showed (predicted P(draw) topping out near 25-30%, not 40%+).

- [ ] **Step 3: If the gate passes, promote directly** (same pattern as `US#192`/`US#193`'s manual promotions — `select-best-models` isn't the right gate here either, since `log_loss`/`accuracy` won't reflect *why* this is better)

Update `config/model_selection.yaml`'s `contexts.SP1.result_3way` entry: `model_path`, `model_type: skellam_result`, `feature_subset` (from the new artifact's own `.metadata.json` `feature_names` — **do this even though `SkellamResultModel` has no `feature_subset` of its own in the usual sense**, since `home_goals`/`away_goals`'s own feature lists are resolved internally by the class itself, not from this field; leave `feature_subset` as whatever `_build_artifact_metadata` records, matching E0's existing entry's own shape as the reference), `previous_model_path` (chain to the current SP1 XGBoost artifact), `selected_at`. Add a one-line pointer comment (PyYAML strips block comments on the next full-file rewrite — keep it short) noting the real-edge numbers that justified this, and append a dated note to `documents/user_stories.md`'s `US#173` entry the same way the 09-16 investigation notes were appended.

- [ ] **Step 4: If the gate fails, document why and stop — this is a valid outcome**

Append to `US#173`: candidate real edge, n, and whatever pattern it showed (e.g. "Skellam alone doesn't transfer to SP1 because [specific finding]"). Do not retry with a different `alpha`/weighting — that's already been tried and ruled out for the XGBoost path; a Skellam failure here means the underlying `home_goals`/`away_goals` regressors for SP1 aren't differentiated enough (the same "compressed expected goals" mechanism `A104` found for E0's pre-09-03 submodels) — check that specifically before abandoning the approach.

- [ ] **Step 5: Repeat Steps 1-4 independently for D1, I1, F1**

Each is its own gate, its own promotion decision. A pass for SP1 does not imply a pass for the others — D1/I1/F1's own `home_goals`/`away_goals` regressors have their own, independently-measured quality.

---

## Phase 2: Explicit Market-Blend Post-Processor (medium confidence — new, small code addition)

**Correction to the original research summary, verified against this exact repo before writing this task:** `MKT_IMPLIED_HOME`/`MKT_IMPLIED_DRAW`/`MKT_IMPLIED_AWAY`/`MKT_OVERROUND`/`MKT_LINE_MOVE_*`/`MKT_BOOK_DISAGREEMENT_*` are **already** in `result_3way`'s feature set (confirmed: `MKT_IMPLIED_DRAW in feature_subset` is `True` for SP1 today) — market signal is not missing from FPAI's features the way the external-repo comparison implied. The real, narrower hypothesis: an XGBoost tree ensemble may not be *using* that signal as directly/reliably as an explicit blend formula would, since a tree has to discover the (near-linear) relationship between `MKT_IMPLIED_DRAW` and the true outcome via splits, rather than being handed it directly.

**Files:**
- Create: `src/models/market_blend.py`
- Test: `tests/test_market_blend.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_market_blend.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_market_blend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.market_blend'`

- [ ] **Step 3: Write the implementation**

```python
# src/models/market_blend.py
"""Explicit market-probability blending -- a linear shrinkage of the raw
model probability toward the market-implied one, as a tunable alternative
to relying on MKT_IMPLIED_* being effectively used inside the tree ensemble
itself. market_weight=0 is a pure no-op (today's behavior); market_weight=1
means "just use the market's own price." The right value, if any, is an
empirical question answered by Phase 2's own validation step below, not
assumed here.
"""

from __future__ import annotations

import numpy as np


def blend_with_market(model_proba: np.ndarray, market_proba: np.ndarray, market_weight: float) -> np.ndarray:
    if not 0.0 <= market_weight <= 1.0:
        raise ValueError(f"market_weight must be in [0, 1], got {market_weight}")
    blended = (1 - market_weight) * model_proba + market_weight * market_proba
    return blended / blended.sum(axis=1, keepdims=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_market_blend.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/models/market_blend.py tests/test_market_blend.py
git commit -m "feat: add explicit market-probability blending as a tunable post-processor

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Sweep market_weight on real held-out data, gated by Phase 0's script**

For SP1 `result_3way` (or whichever league/target is being tested), pull the test split's raw model `predict_proba` output (already have this pattern from every prior investigation this session), build `market_proba` from `[implied_away, implied_draw, implied_home]` normalized to sum to 1, sweep `market_weight` in `[0.1, 0.2, ..., 0.9]`, and for each value compute `compute_real_edge` per selection (home/draw/away) the same way `US#173`'s "full market" table did. **Do not pick the winning weight against the test split and call it done — this is exactly the sweep-vs-test overfitting risk flagged earlier in this investigation.** Split the sweep itself: pick the best `market_weight` on `X_val`'s own real edge (val has real odds for `total_goals`/`result_3way` at 100% coverage, unlike `btts`), then confirm once on `X_test` before deciding.

**Decision gate:** the chosen `market_weight`'s pooled real edge across all 3 selections must beat the un-blended baseline's pooled real edge (`US#173`'s weighted-XGBoost SP1 baseline: **−1.3pp pooled**) by more than noise (a few points of `n`, not a rigorous significance test, but at minimum: better sign, not just a smaller negative number within the same ballpark).

- [ ] **Step 7: If the gate passes, wire it into serving as an optional, off-by-default `ForecastService` step**

Add a `market_blend_weight: float | None = None` field to each league's `contexts.<LEAGUE>.result_3way` entry in `config/model_selection.yaml` (not a code-level default — this is a per-league empirical constant, same convention as `RESULT_3WAY_ALPHA`). In `ForecastService._predict_target`, after `_apply_calibration` (which is currently a no-op everywhere calibrators are disabled), read this field from `metadata` and call `blend_with_market` when set and when real market odds are available for this call. **Off (`None`) for every league until its own sweep passes its own gate** — no global default.

- [ ] **Step 8: If the gate fails, document and stop**

Same discipline as Phase 1 Step 4 — append the sweep results (every `market_weight` tried, its val and test real edge) to `US#173`/`US#193` as appropriate, and do not ship a blend that didn't clear its own bar just because it's "probably safe."

---

## Phase 3: Ordinal Regression for `result_3way` (lowest confidence, most novel, most effort)

Treats home/draw/away as ordered (`away < draw < home` on a margin-of-victory latent scale) instead of three unordered classes — directly targets the "draw is a non-monotonic squeezed-middle class" mechanism that made this the hardest of the three markets to fix. This is genuinely new modeling work, not a small addition — scope accordingly, and only start this phase if Phases 1-2 don't close enough of the gap.

**New dependency:** `statsmodels` (not currently installed — confirmed via `pip show statsmodels`). MIT-licensed, extremely standard, `OrderedModel` is its own well-tested cumulative-logit implementation — reuse it rather than hand-rolling the math, per the same "don't reinvent what's one dependency away" reasoning used everywhere else in this codebase (e.g. `sklearn.utils.class_weight.compute_sample_weight` instead of a hand-rolled formula).

**Files:**
- Create: `src/models/ordinal_result_model.py`
- Modify: `main.py` — add `"ordinal_result": None` to `MODEL_REGISTRY` (handled via `ModelFactory`, same pattern as `skellam_result`)
- Modify: `src/models/model_factory.py` — add the `ordinal_result` branch
- Test: `tests/test_ordinal_result_model.py`
- Modify: `requirements.txt` — add `statsmodels`

- [ ] **Step 1: Add the dependency**

```bash
echo "statsmodels" >> requirements.txt
pip install statsmodels
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_ordinal_result_model.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.ordinal_result_model import OrdinalResultModel


def test_predict_proba_returns_three_columns_summing_to_one():
    # 6 synthetic rows: a strong home favorite (feature=2), a genuine
    # toss-up (feature=0), a strong away favorite (feature=-2) -- doubled
    # so statsmodels' OrderedModel has enough rows per class to fit.
    X = pd.DataFrame({"strength_diff": [2, 2, 0, 0, -2, -2]})
    y = pd.Series(["home", "home", "draw", "away", "away", "away"])

    model = OrdinalResultModel()
    model.train(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (6, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(6), atol=1e-6)


def test_classes_are_away_draw_home_alphabetical_matching_project_convention():
    X = pd.DataFrame({"strength_diff": [2, 0, -2]})
    y = pd.Series(["home", "draw", "away"])
    model = OrdinalResultModel()
    model.train(X, y)
    assert list(model.classes_) == ["away", "draw", "home"]


def test_save_and_load_round_trip(tmp_path):
    X = pd.DataFrame({"strength_diff": [2, 2, 0, 0, -2, -2]})
    y = pd.Series(["home", "home", "draw", "away", "away", "away"])
    model = OrdinalResultModel()
    model.train(X, y)
    path = str(tmp_path / "ordinal.joblib")
    model.save(path)

    loaded = OrdinalResultModel.load(path)
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ordinal_result_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.ordinal_result_model'`

- [ ] **Step 4: Write the implementation**

```python
# src/models/ordinal_result_model.py
"""result_3way via ordinal (cumulative-logit) regression -- treats
away < draw < home as an ordered outcome on one latent scale, instead of
three unordered classes. Motivated by US#173: draw is a non-monotonic
"squeezed middle" class (peaks for evenly-matched fixtures, falls off
toward both a big home favorite and a big away favorite) that a plain
multiclass softmax has to carve out as a separate region; an ordinal model
represents it directly as the middle band on one axis instead.

statsmodels' OrderedModel (MIT-licensed, standard, well-tested cumulative-
logit implementation) is used rather than a hand-rolled version -- same
"don't reinvent what's one dependency away" reasoning as
sklearn.utils.class_weight.compute_sample_weight elsewhere in this project.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src.models.base_model import FPAIBaseModel

_ORDER = ["away", "draw", "home"]  # away < draw < home on the margin-of-victory scale


class OrdinalResultModel(FPAIBaseModel):
    def __init__(self) -> None:
        self.classes_ = np.array(sorted(_ORDER))  # alphabetical, matching every other result_3way model's convention
        self._fitted: Any = None
        self._feature_columns: list[str] | None = None

    def train(self, X: Any, y: Any, eval_set: Any | None = None, sample_weight: Any | None = None) -> None:
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self._feature_columns = list(X_df.columns)
        y_ordered = pd.Categorical(y, categories=_ORDER, ordered=True)
        # OrderedModel has no sample_weight parameter -- statsmodels' own
        # GenericLikelihoodModel base doesn't support it. Documented
        # limitation, not silently dropped: US#173's class-balance-weighting
        # experiments (alpha sweep) don't carry over to this model family.
        # If class imbalance turns out to matter here too, the alternative
        # is resampling y before fit(), not attempted in this first pass.
        model = OrderedModel(y_ordered, X_df, distr="logit")
        self._fitted = model.fit(method="bfgs", disp=False)

    def predict_proba(self, X: Any) -> np.ndarray:
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_df = X_df[self._feature_columns]
        proba = self._fitted.model.predict(self._fitted.params, exog=X_df)
        # statsmodels returns columns in _ORDER (away, draw, home) already --
        # matches self.classes_'s alphabetical order, no remapping needed.
        return np.asarray(proba)

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def save(self, path: str) -> None:
        joblib.dump({"fitted": self._fitted, "feature_columns": self._feature_columns}, path)

    @classmethod
    def load(cls, path: str) -> "OrdinalResultModel":
        payload = joblib.load(path)
        instance = cls()
        instance._fitted = payload["fitted"]
        instance._feature_columns = payload["feature_columns"]
        return instance
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ordinal_result_model.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Wire into `MODEL_REGISTRY`/`ModelFactory`**

In `main.py`'s `MODEL_REGISTRY` dict, add:
```python
"ordinal_result": None,  # handled via ModelFactory, same pattern as skellam_result
```

In `src/models/model_factory.py`, `ModelFactory._REGISTRY` is a plain dict (confirmed by reading the file, not assumed — `{"skellam_result": SkellamResultModel, "ensemble_result": EnsembleResultModel, ...}`, looked up via `ModelFactory.get_model(model_type, params)` → `model_cls(**(params or {}))`). Add the import at the top and one entry to that dict:
```python
from src.models.ordinal_result_model import OrdinalResultModel
```
```python
        "ordinal_result": OrdinalResultModel,
```
No `params` handling needed — `main.py`'s own `factory_params = {"competition_id": ...} if selected_model in ("skellam_result", "ensemble_result") else None` list doesn't include `"ordinal_result"`, so it gets called as `OrdinalResultModel()` with no arguments, matching the constructor written in Step 4.

- [ ] **Step 7: Run the full existing test suite to confirm no regressions**

Run: `python3 -m pytest tests/ -q`
Expected: all prior tests still pass, plus the 3 new ones (no existing test references `ordinal_result`, so this should be purely additive)

- [ ] **Step 8: Commit**

```bash
git add src/models/ordinal_result_model.py src/models/model_factory.py main.py tests/test_ordinal_result_model.py requirements.txt
git commit -m "feat: add OrdinalResultModel (cumulative-logit) as a result_3way alternative

Motivated by US#173 -- draw is a non-monotonic 'squeezed middle' class
that a plain multiclass softmax struggles to carve out; ordinal
regression represents away < draw < home on one latent scale directly.

New dependency: statsmodels (MIT, OrderedModel).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 9: Train a real candidate and gate it exactly like Phase 1**

```bash
python3 main.py train-target --target result_3way --context SP1 --model ordinal_result --refit-full-data
```
Then run the same real-edge validation as Phase 1 Step 2, substituting `OrdinalResultModel.load(...)` for `SkellamResultModel.load(...)`.

**Decision gate:** same bar as Phase 1 — real edge within ±3pp of 0 on draw specifically, and pooled real edge across all 3 selections should not be worse than whichever of Phase 1/2's SP1 result currently stands (this phase is a fallback if those don't clear their own bars, not guaranteed to be tried at all).

- [ ] **Step 10: Document the outcome either way**

Append to `US#173` regardless of pass/fail — a documented negative result for a genuinely different model family is real information (rules out "the model family itself is the problem" vs. "SP1 specifically lacks discriminative features," same distinction `US#173`'s own two candidate root causes already drew).

---

## Self-Review

**Spec coverage:** Phase 0 (shared validation) ✓, Phase 1 (Skellam extension) ✓, Phase 2 (market blend) ✓, Phase 3 (ordinal regression) ✓. All three ideas from the "borrow the idea, not the codebase" discussion are covered, each with its own real-edge gate rather than an in-sample metric.

**Placeholder scan:** No TBD/TODO/"add appropriate handling" found — every step has real, complete code or an exact command with expected output.

**Type consistency:** `compute_real_edge`'s signature (`df, predicted_col, implied_col, actual_col, edge_threshold`) is used identically in Phase 1 Step 2 and referenced identically in Phase 2 Step 6. `blend_with_market(model_proba, market_proba, market_weight)` matches its own test file. `OrdinalResultModel`'s `classes_`/`predict_proba`/`predict`/`save`/`load` match `FPAIBaseModel`'s abstract interface exactly (confirmed against `src/models/base_model.py` before writing this plan, not assumed).
