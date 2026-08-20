# `result_3way` Sample-Weight Retune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tunable dampening exponent to `result_3way`'s class-balancing sample weight, find a value per league (E0/SP1/D1/I1/F1) that fixes the current draw-overprediction bug without regressing back to the pre-08-14 draw-blindness bug, and promote the retrained models to production.

**Architecture:** `_compute_sample_weight(y, task_type, alpha=1.0)` in `src/models/model_manager.py` raises sklearn's existing `compute_sample_weight("balanced", y)` output to a power `alpha` (1.0 = today's full balancing, 0.0 = pre-08-14 unweighted). Threaded through `ModelManager` and `main.py::run_train_target()` as an optional constructor/function param — no new CLI flag. A throwaway sweep script explores candidate `alpha` values per league using the existing train/val/test split machinery without saving artifacts or touching MLflow; the final chosen `alpha` per league is then trained for real via the existing `run_train_target()` → `run_pipeline()` → `select-best-models` promotion path.

**Tech Stack:** Python, scikit-learn (`compute_sample_weight`, `recall_score`, `precision_score`), XGBoost, MLflow, DuckDB, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-result-3way-sample-weight-retune-design.md`

---

### Task 1: Add `alpha` dampening parameter to `_compute_sample_weight()`

**Files:**
- Modify: `src/models/model_manager.py:33-49` (`_compute_sample_weight`)
- Test: `tests/test_model_manager_sample_weight.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_model_manager_sample_weight.py`, right after the existing `test_compute_sample_weight_balances_imbalanced_classes`:

```python
def test_compute_sample_weight_alpha_defaults_to_one_unchanged() -> None:
    y = pd.Series(["home"] * 80 + ["away"] * 15 + ["draw"] * 5)
    default_call = _compute_sample_weight(y, "classification")
    explicit_alpha_one = _compute_sample_weight(y, "classification", alpha=1.0)
    np.testing.assert_array_equal(default_call, explicit_alpha_one)


def test_compute_sample_weight_alpha_zero_gives_uniform_weights() -> None:
    y = pd.Series(["home"] * 80 + ["away"] * 15 + ["draw"] * 5)
    weights = _compute_sample_weight(y, "classification", alpha=0.0)
    np.testing.assert_allclose(weights, np.ones(len(y)))


def test_compute_sample_weight_alpha_half_is_between_uniform_and_balanced() -> None:
    y = pd.Series(["home"] * 80 + ["away"] * 15 + ["draw"] * 5)
    balanced = _compute_sample_weight(y, "classification", alpha=1.0)
    dampened = _compute_sample_weight(y, "classification", alpha=0.5)
    uniform = _compute_sample_weight(y, "classification", alpha=0.0)

    draw_mask = (y == "draw").to_numpy()
    # Draw's weight is above 1.0 (uniform) since it's the minority class --
    # dampened must sit strictly between the uniform (1.0) and fully-balanced
    # draw weight, not equal to either.
    assert uniform[draw_mask][0] < dampened[draw_mask][0] < balanced[draw_mask][0]


def test_compute_sample_weight_alpha_still_none_for_regression() -> None:
    y = pd.Series([1.0, 2.0, 3.0])
    assert _compute_sample_weight(y, "regression", alpha=0.5) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_manager_sample_weight.py -k "alpha" -v`
Expected: FAIL with `TypeError: _compute_sample_weight() got an unexpected keyword argument 'alpha'`

- [ ] **Step 3: Implement the `alpha` parameter**

Replace `_compute_sample_weight` in `src/models/model_manager.py`:

```python
def _compute_sample_weight(y: pd.Series, task_type: str, alpha: float = 1.0) -> np.ndarray | None:
    """Inverse-class-frequency sample weights for classification targets.

    Found live: result_3way's XGBoost classifier had 2.1% recall on 'draw'
    (SP1 test split) despite draws being ~25% of real outcomes -- trained
    with plain unweighted multiclass log-loss, so the model could minimize
    loss by mostly ignoring the harder-to-separate minority class. Reuses
    sklearn's own compute_sample_weight('balanced', ...) rather than a
    hand-rolled formula -- already a project dependency, standard technique.
    Regression targets have no notion of class balance; returns None so
    every model's .train(sample_weight=None) call is a byte-identical no-op
    for them.

    alpha dampens the balanced weighting: weights ** alpha. alpha=1.0 (default)
    is full balancing (today's behavior, byte-identical to the pre-alpha
    function). alpha=0.0 collapses every weight to 1.0 -- the pre-08-14
    unweighted behavior. Added 2026-08-20 after full 'balanced' weighting
    (alpha=1.0) was found to overcorrect result_3way's draw-blindness bug
    into a draw-overprediction bug on E0/SP1 -- see
    docs/superpowers/specs/2026-08-20-result-3way-sample-weight-retune-design.md."""
    if task_type == "regression":
        return None
    from sklearn.utils.class_weight import compute_sample_weight

    weights = compute_sample_weight("balanced", y)
    if alpha != 1.0:
        weights = weights ** alpha
    return weights
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_manager_sample_weight.py -v`
Expected: all tests PASS (existing tests plus the 4 new ones), no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/models/model_manager.py tests/test_model_manager_sample_weight.py
git commit -m "feat(ml): add alpha dampening param to _compute_sample_weight

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Thread `sample_weight_alpha` through `ModelManager`

**Files:**
- Modify: `src/models/model_manager.py` (`__init__`, `train()`, `run_pipeline()`)
- Test: `tests/test_model_manager_sample_weight.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model_manager_sample_weight.py`. First, update the `_make_manager` helper to accept an optional `alpha`:

```python
def _make_manager(tmp_path: Path, model: Any, target: str, sample_weight_alpha: float = 1.0) -> ModelManager:
    db_path = tmp_path / "test.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8"
    )
    return ModelManager(
        model=model,
        config_path=str(config_path),
        target_config={"target": target},
        sample_weight_alpha=sample_weight_alpha,
    )
```

Then add a new test:

```python
def test_model_manager_train_threads_sample_weight_alpha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _CapturingModel()
    manager = _make_manager(tmp_path, model, "result_3way", sample_weight_alpha=0.5)

    X = pd.DataFrame({"f1": range(20)})
    y_train = pd.Series(["home"] * 15 + ["draw"] * 5)
    empty_meta = pd.DataFrame(index=y_train.index)
    monkeypatch.setattr(
        manager,
        "prepare_training_data",
        lambda: (X, X, X, y_train, y_train, y_train, empty_meta),
    )
    monkeypatch.setattr(manager, "_load_selected_features", lambda: ["f1"])
    monkeypatch.setattr(manager, "_log_selected_features", lambda *_: None)
    monkeypatch.setattr(manager, "_log_feature_importance", lambda *_: None)

    manager.train()

    expected = _compute_sample_weight(y_train, "classification", alpha=0.5)
    np.testing.assert_array_equal(model.received_sample_weight, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_manager_sample_weight.py::test_model_manager_train_threads_sample_weight_alpha -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'sample_weight_alpha'`

- [ ] **Step 3: Thread the parameter through `ModelManager`**

In `src/models/model_manager.py`, modify `ModelManager.__init__` (currently around line 89) — add the new parameter and store it:

```python
    def __init__(
        self,
        model: FPAIBaseModel,
        config_path: str = "config.yaml",
        league_tier: str = "all",
        test_season: str = "time_split",
        feature_version: str = "v1",
        target_config: dict[str, str | float | int] | None = None,
        feature_subset: list[str] | None = None,
        context: str = "E0",
        competition_id: str = "E0",
        sample_weight_alpha: float = 1.0,
    ) -> None:
```

Add this line among the other `self.` assignments in `__init__` (near `self.competition_id: str = competition_id`):

```python
        # 2026-08-20: dampens compute_sample_weight('balanced', ...) -- see
        # _compute_sample_weight's own docstring. 1.0 (default) preserves
        # every existing caller's exact current behavior.
        self.sample_weight_alpha: float = sample_weight_alpha
```

Update the two call sites. In `train()` (currently line 603):

```python
        sample_weight = _compute_sample_weight(y_train, self.target_definition.task_type, alpha=self.sample_weight_alpha)
```

In `run_pipeline()`'s inner `_run_training()` (currently line 642):

```python
                sample_weight = _compute_sample_weight(y_train, self.target_definition.task_type, alpha=self.sample_weight_alpha)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_manager_sample_weight.py -v`
Expected: all tests PASS, including the pre-existing ones (they don't pass `sample_weight_alpha`, so they exercise the `1.0` default and must still pass unchanged).

- [ ] **Step 5: Run the full model_manager test suite for regressions**

Run: `pytest tests/test_model_manager*.py -v`
Expected: all PASS, zero regressions.

- [ ] **Step 6: Commit**

```bash
git add src/models/model_manager.py tests/test_model_manager_sample_weight.py
git commit -m "feat(ml): thread sample_weight_alpha through ModelManager

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Thread `sample_weight_alpha` through `run_train_target()`

**Files:**
- Modify: `main.py:766-806` (`run_train_target`)
- Test: `tests/test_target_availability.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_target_availability.py` (near the top-level imports, `import main` and `ModelManager` are already available in that file — confirm with `grep -n "^import\|^from" tests/test_target_availability.py` before writing if unsure of exact existing import names):

```python
def test_run_train_target_passes_sample_weight_alpha_to_model_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """sample_weight_alpha is a plain Python kwarg, not a CLI flag (2026-08-20
    design decision) -- this only needs to reach ModelManager's constructor,
    not argparse."""
    captured: dict = {}

    class _FakeManager:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_pipeline(self):
            return Path("fake_model.joblib")

    monkeypatch.setattr(main, "ModelManager", _FakeManager)

    main.run_train_target("result_3way", model_name="xgb", context="E0", sample_weight_alpha=0.5)

    assert captured.get("sample_weight_alpha") == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_target_availability.py::test_run_train_target_passes_sample_weight_alpha_to_model_manager -v`
Expected: FAIL with `TypeError: run_train_target() got an unexpected keyword argument 'sample_weight_alpha'`

- [ ] **Step 3: Add the parameter**

In `main.py`, change the `run_train_target` signature (currently line 766):

```python
def run_train_target(
    target_name: str, model_name: str | None = None, context: str = "E0", sample_weight_alpha: float = 1.0,
) -> Path:
    """Train one registry-backed forecast target model.

    sample_weight_alpha (2026-08-20): dampens _compute_sample_weight's
    class-balancing strength for this run -- see
    docs/superpowers/specs/2026-08-20-result-3way-sample-weight-retune-design.md.
    Deliberately not exposed as a CLI flag: it's a one-time tuning constant
    for result_3way's draw-overprediction fix, called directly from Python
    (scripts/tune_result_3way_sample_weight.py and the promotion step), not
    something meant to vary per ordinary train-target invocation. Every
    existing CLI call site keeps the 1.0 default, unchanged behavior.
    """
```

And update the `ModelManager(...)` construction (currently ~line 800):

```python
    model_manager = ModelManager(
        model=model,
        target_config={"target": definition.name},
        feature_subset=feature_subset,
        context=competition_id,
        competition_id=competition_id,
        sample_weight_alpha=sample_weight_alpha,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_target_availability.py -v`
Expected: all PASS, zero regressions (the fake manager in the new test doesn't implement the full `ModelManager` interface, but `run_train_target` only calls `.run_pipeline()` on it, which `_FakeManager` provides).

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_target_availability.py
git commit -m "feat(ml): thread sample_weight_alpha through run_train_target

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Write the sweep script

**Files:**
- Create: `scripts/tune_result_3way_sample_weight.py`

No test for this task — it's a throwaway analysis script (per design's non-goal: not a permanent CLI command), not product code. It reuses `ModelManager`'s data-prep machinery but deliberately calls `.model.train()` directly instead of `.run_pipeline()`, so no MLflow run is started and no model artifact is written to `models/` during the sweep (keeps `select-best-models` in Task 7 from ever seeing these exploratory runs).

- [ ] **Step 1: Write the script**

```python
"""Throwaway sweep script (2026-08-20): finds a per-league alpha for
result_3way's sample-weight dampening that fixes the draw-overprediction
bug (docs/superpowers/specs/2026-08-20-result-3way-sample-weight-retune-design.md)
without regressing to the pre-08-14 draw-blindness bug.

Deliberately does NOT call ModelManager.run_pipeline() -- that starts an
MLflow run tagged sweep_stage="final" and saves a real model artifact,
which would make every candidate alpha (including the ones we reject)
eligible for `select-best-models` to accidentally promote by raw log_loss,
the same metric-only mistake that let the original 08-14 fix overshoot
unnoticed. This script only trains in-memory and prints metrics.

Usage: python scripts/tune_result_3way_sample_weight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_score, recall_score

from src.logic.competition_registry import get_competition_definition, resolve_feature_subset_for_tier
from src.models import ModelManager, XGBoostModel
from src.models.model_manager import _compute_sample_weight

LEAGUES = ["E0", "SP1", "D1", "I1", "F1"]
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
XGB_PARAMS = {"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": 3}
LABELS = ["home", "draw", "away"]


def sweep_one_league(context: str) -> pd.DataFrame:
    competition_def = get_competition_definition(context)
    feature_subset = resolve_feature_subset_for_tier(competition_def.tier)

    rows = []
    for alpha in ALPHAS:
        model = XGBoostModel(**XGB_PARAMS)
        manager = ModelManager(
            model=model,
            target_config={"target": "result_3way"},
            feature_subset=feature_subset,
            context=context,
            competition_id=context,
            sample_weight_alpha=alpha,
        )
        selected_features = manager._load_selected_features()
        manager._log_selected_features(selected_features)
        X_train, X_val, X_test, y_train, y_val, y_test, _ = manager.prepare_training_data()

        weights = _compute_sample_weight(y_train, manager.target_definition.task_type, alpha=alpha)
        model.train(X_train, y_train, eval_set=[(X_val, y_val)], sample_weight=weights)

        proba = model.predict_proba(X_test)  # columns ordered per model.label_encoder.classes_
        class_order = list(model.label_encoder.classes_)  # alphabetical: away, draw, home
        preds = model.predict(X_test)

        draw_col = class_order.index("draw")
        mean_draw_proba = float(proba[:, draw_col].mean())

        y_test_arr = np.asarray(y_test)
        recall = recall_score(y_test_arr, preds, labels=LABELS, average=None, zero_division=0)
        precision = precision_score(y_test_arr, preds, labels=LABELS, average=None, zero_division=0)
        true_draw_rate = float((y_test_arr == "draw").mean())

        # log_loss needs probabilities in the same column order as its own
        # labels= argument, not necessarily class_order -- pass class_order
        # explicitly so this is correct regardless of alphabetical assumptions.
        loss = log_loss(y_test_arr, proba, labels=class_order)

        rows.append({
            "league": context,
            "alpha": alpha,
            "log_loss": round(loss, 4),
            "draw_recall": round(float(recall[LABELS.index("draw")]), 4),
            "draw_precision": round(float(precision[LABELS.index("draw")]), 4),
            "home_recall": round(float(recall[LABELS.index("home")]), 4),
            "away_recall": round(float(recall[LABELS.index("away")]), 4),
            "mean_predicted_draw_proba": round(mean_draw_proba, 4),
            "true_draw_rate": round(true_draw_rate, 4),
        })

    return pd.DataFrame(rows)


def main() -> None:
    all_results = pd.concat([sweep_one_league(league) for league in LEAGUES], ignore_index=True)
    out_path = Path("reports") / "result_3way_sample_weight_sweep.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(out_path, index=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(all_results.to_string(index=False))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs end-to-end for one league before running the full sweep**

Run: `python -c "
import sys; sys.path.append('.')
from scripts.tune_result_3way_sample_weight import sweep_one_league
print(sweep_one_league('E0').to_string(index=False))
"`

Expected: prints a 5-row table (one row per alpha) with `log_loss`, `draw_recall`, `draw_precision`, `home_recall`, `away_recall`, `mean_predicted_draw_proba`, `true_draw_rate` columns, no exceptions. `alpha=1.0`'s `draw_recall`/`mean_predicted_draw_proba` should roughly match the live-diagnosed numbers (draw recall ~23% for E0, mean predicted draw probability noticeably above `true_draw_rate`). `alpha=0.0`'s `draw_recall` should be near-zero (reproducing the pre-08-14 bug), confirming the sweep script itself is behaving correctly before trusting the rest of the table.

- [ ] **Step 3: Commit**

```bash
git add scripts/tune_result_3way_sample_weight.py
git commit -m "feat(ml): add result_3way sample-weight alpha sweep script

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Run the sweep and choose alpha per league

**Files:**
- Create: `reports/result_3way_sample_weight_sweep.csv` (script output, Task 4)
- Create: `docs/superpowers/plans/results/2026-08-20-result-3way-alpha-choice.md`

- [ ] **Step 1: Run the full sweep**

Run: `python scripts/tune_result_3way_sample_weight.py`
Expected: completes without error, prints and saves a 25-row table (5 leagues × 5 alphas) to `reports/result_3way_sample_weight_sweep.csv`. This will take real wall-clock time (25 XGBoost training runs) — not instant.

- [ ] **Step 2: Choose alpha per league against the design's validation gate**

Open `reports/result_3way_sample_weight_sweep.csv`. For each league, pick the smallest alpha (i.e. least dampened away from the current `1.0`) that satisfies all three:
- `draw_recall` in `[0.20, 0.30]`
- `draw_precision` not collapsed relative to `alpha=1.0`'s own value (a reasonable bar: no more than ~15 relative percentage points below it — this is a judgment call, not a hard threshold; the point is precision must not have cratered just because recall looks fine)
- `mean_predicted_draw_proba` within roughly ±0.05 of `true_draw_rate` for that league

If no swept alpha value satisfies all three for a league, that is a real, reportable finding — do not force a choice; note it and treat that league as needing a follow-up story (e.g. a finer-grained sweep, or a different fix entirely) rather than silently picking the least-bad option.

- [ ] **Step 3: Record the decision**

Create `docs/superpowers/plans/results/2026-08-20-result-3way-alpha-choice.md`:

```markdown
# result_3way sample_weight alpha choice (2026-08-20)

Full sweep data: `reports/result_3way_sample_weight_sweep.csv`

| League | Chosen alpha | draw_recall | draw_precision | mean_predicted_draw_proba | true_draw_rate | Notes |
|---|---|---|---|---|---|---|
| E0  | <fill in from real sweep output> | | | | | |
| SP1 | | | | | | |
| D1  | | | | | | |
| I1  | | | | | | |
| F1  | | | | | | |

Any league with no alpha satisfying the gate: <note here, or "none">
```

Fill in the table from the actual `reports/result_3way_sample_weight_sweep.csv` output — this file is committed as the record of what was chosen and why, not left as a template.

- [ ] **Step 4: Commit**

```bash
git add reports/result_3way_sample_weight_sweep.csv docs/superpowers/plans/results/2026-08-20-result-3way-alpha-choice.md
git commit -m "docs(ml): record result_3way sample-weight sweep results and chosen alpha per league

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Retrain final models at chosen alpha

**Files:** none changed (uses `run_train_target` from Task 3)

- [ ] **Step 1: Retrain each league's `result_3way` model at its chosen alpha**

For each league, run (substituting that league's chosen alpha from Task 5's results file):

```bash
python -c "
import sys; sys.path.append('.')
import main
main.run_train_target('result_3way', model_name='xgb', context='E0', sample_weight_alpha=<E0_ALPHA>)
"
```

Repeat for `SP1`, `D1`, `I1`, `F1` with their own chosen alphas. Expected each time: a new artifact saved under `models/` (e.g. `models/result_3way_xgboost_v1_<today's date>.joblib`), a new `.calibration.pkl` sidecar, a new `.metadata.json`, and an MLflow run logged under experiment `FPAI_Evolution` tagged `sweep_stage=final`, `context=<league>`.

- [ ] **Step 2: Spot-check one artifact's metadata**

Run: `cat models/result_3way_xgboost_v1_<today's date>.metadata.json | python3 -m json.tool | grep -A3 '"metrics"'`
Expected: a `log_loss` value present and roughly consistent with that league's Task 5 sweep row for the chosen alpha (small differences are fine — the sweep and this run use the same train/val/test split and alpha, but MLflow autologging/early-stopping details can cause minor float differences; a large discrepancy would mean something is wrong and should be investigated before proceeding).

---

### Task 7: Promote retrained models into `config/model_selection.yaml`

**Files:**
- Modify: `config/model_selection.yaml` (via `select-best-models`, or manually if it declines to promote)

- [ ] **Step 1: Dry-run the standard promotion path per league**

```bash
python main.py select-best-models --target result_3way --context E0 --dry-run
```

Expected output: either a proposed change (new model_path, new metric_value) if the new run's `log_loss` clears `--min_improvement` (default `0.005`) over the currently-selected E0 `result_3way` entry, or a message that nothing qualifies.

- [ ] **Step 2: If the dry-run proposes the new model, promote it for real**

```bash
python main.py select-best-models --target result_3way --context E0
```

Expected: `config/model_selection.yaml`'s `contexts.E0.result_3way` entry updated (`model_path`, `metric_value`, `mlflow_run_id`, `selected_at`).

- [ ] **Step 3: If the dry-run does NOT propose the new model**

This is expected and acceptable if the dampened-weight model's `log_loss` is slightly worse than the current `alpha=1.0` model's (same log_loss/minority-recall tradeoff the original 08-14 fix itself made going the other direction — see that commit's own message). In that case, update `config/model_selection.yaml` by hand for that league's `result_3way` entry: set `model_path` to the new artifact's filename (from Task 6), `metric_value` to its `log_loss` (from the metadata file checked in Task 6 Step 2), `metric_name: test_log_loss` (unchanged), `mlflow_run_id` to the run ID printed/logged during Task 6's retrain, `selected_at` to the current UTC timestamp in the same `YYYY-MM-DDTHH:MM:SSZ` format as the other entries, and move the old `model_path` into `previous_model_path`. Follow the exact structure already present for that league's other targets in the same file (`config/model_selection.yaml`) as the template — do not invent new keys.

Note in the commit message (Step 5) which path was used (automatic vs. manual) and why, per league — this is exactly the kind of promotion-basis detail `documents/agent_user_stories.md`'s existing A56/A57 entries already model for this project.

- [ ] **Step 4: Repeat Steps 1-3 for SP1, D1, I1, F1**

- [ ] **Step 5: Commit**

```bash
git add config/model_selection.yaml
git commit -m "feat(ml): promote dampened-sample-weight result_3way models (E0/SP1/D1/I1/F1)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 8: Verify the fix against the originally-diagnosed symptom

**Files:** none changed — verification only

- [ ] **Step 1: Re-run the exact three reproduction cases from the original investigation**

```bash
python3 -c "
import sys; sys.path.append('.')
from src.forecast.forecast_service import ForecastService
svc = ForecastService()
tests = [
    ('Nottingham', 'Leeds United', 'E0', 2.60, 3.30, 2.75),
    ('Real Betis', 'Real Sociedad', 'SP1', 2.30, 3.30, 3.10),
    ('Ipswich Town', 'Sunderland', 'E0', 2.70, 3.30, 2.60),
]
for home, away, league, oh, od, oa in tests:
    r = svc.forecast_upcoming(home_team=home, away_team=away, date='2026-08-22', league=league, odds_h=oh, odds_d=od, odds_a=oa, match_type='league')
    print(home, 'v', away, '->', r['forecast']['result_3way']['probabilities'])
"
```

Expected: unlike the original run (where draw was 38-40%, the single highest of the three, in all three cases), draw's probability should no longer be dominant/implausibly high in most or all of these — some spread reflecting real match context, with `home`/`away` probabilities that plausibly track the given odds' implied favorite. This is a sanity check, not a strict pass/fail assertion — record the actual numbers in the commit message for Step 2.

- [ ] **Step 2: Commit a short verification note**

Create `docs/superpowers/plans/results/2026-08-20-result-3way-verification.md` with the actual before/after probability triplets for the three test cases (before = the numbers already recorded in this conversation/spec, after = Step 1's real output), then:

```bash
git add docs/superpowers/plans/results/2026-08-20-result-3way-verification.md
git commit -m "docs(ml): record before/after verification for result_3way retune

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 9: Update user stories and technical docs

**Files:**
- Modify: `documents/user_stories.md`
- Modify: `documents/FRAI_TECHSPEC.md`

- [ ] **Step 1: Append a new completed story to `documents/user_stories.md`**

Follow the exact table-row format of the existing `US#167`-style entries (see that story for the template: real baseline numbers, real verification, not assumed). Include: the draw-dominance symptom as originally reported, the root cause (08-14 `compute_sample_weight("balanced", ...)` overcorrection, confirmed via direct `ForecastService` reproduction), the fix (tunable `alpha` in `_compute_sample_weight`), the chosen alpha per league from Task 5's results file, and the promotion basis (automatic vs. manual) per league from Task 7.

- [ ] **Step 2: Add a short technical-doc note**

In `documents/FRAI_TECHSPEC.md`, near the existing `sample_weight`/draw-recall discussion (search for "sample_weight" in that file first to find the right section to extend rather than creating a new one), add a short paragraph describing the dampening mechanism and pointing at the design spec and results files for the full record.

- [ ] **Step 3: Commit**

```bash
git add documents/user_stories.md documents/FRAI_TECHSPEC.md
git commit -m "docs: user story + techspec note for result_3way sample-weight retune

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
