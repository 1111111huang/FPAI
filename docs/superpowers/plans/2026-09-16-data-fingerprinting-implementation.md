# Data Fingerprinting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the version-mismatch gaps found across this session's investigation (`BUG-066`, `US#173`/`US#192`/`US#193`) by (1) fixing the root cause that let broken calibrators ship silently — an in-sample-only validation gate — and (2) adding content-hash fingerprints at the boundaries that currently have none, so a stale/mismatched artifact fails loudly instead of silently.

**Architecture:** One shared `data_fingerprint()` primitive (mirrors `agent_config_hash.py`'s own "sort keys, JSON dump, SHA-256" pattern — reused, not reinvented), threaded through four independent, additive touch points: the calibrator promotion gate itself, the calibrator sidecar's self-check, model training metadata, and live/snapshot forecast diagnostics.

**Tech Stack:** Existing `hashlib`/`json` (stdlib, no new dependency), `sklearn.metrics.log_loss` (already a dependency).

---

## Scope note

Four independent, additive pieces. Each can ship and be reverted on its own — no piece depends on another being merged first, except Phase 2 (the gate fix) and Phase 3 (the sidecar fingerprint) both touch `_fit_and_save_calibrator`, so do them in order to avoid a merge conflict with yourself.

**Explicitly out of scope for this plan** (noted so it isn't lost, not attempted here):
- `raw_matches`/`feature_store` gaining an `updated_at` column — real, but a schema migration on live tables, a bigger and separate change.
- Folding model/calibrator fingerprints into `agent_config_hash` (`app/backend/agent_config_hash.py`) so `recommendation_cache`'s key busts automatically on a model swap — genuinely valuable, but requires threading per-match target/league resolution into the hash-compute call site, a larger change than the four pieces below. Flagged as a natural Phase 5 if this plan's pieces prove useful.

---

## Phase 1: Shared `data_fingerprint()` Primitive

**Files:**
- Create: `src/utils/fingerprint.py`
- Test: `tests/test_fingerprint.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fingerprint.py
from __future__ import annotations

from pathlib import Path

from src.utils.fingerprint import data_fingerprint, file_fingerprint


def test_same_dict_gives_same_fingerprint_regardless_of_key_order():
    a = data_fingerprint({"b": 2, "a": 1})
    b = data_fingerprint({"a": 1, "b": 2})
    assert a == b


def test_different_values_give_different_fingerprints():
    a = data_fingerprint({"a": 1})
    b = data_fingerprint({"a": 2})
    assert a != b


def test_fingerprint_is_a_short_hex_string():
    fp = data_fingerprint({"a": 1})
    assert len(fp) == 16
    int(fp, 16)  # raises ValueError if not valid hex


def test_file_fingerprint_changes_when_file_content_changes(tmp_path: Path):
    f = tmp_path / "model.joblib"
    f.write_bytes(b"version one")
    fp1 = file_fingerprint(f)
    f.write_bytes(b"version two")
    fp2 = file_fingerprint(f)
    assert fp1 != fp2


def test_file_fingerprint_is_stable_for_unchanged_file(tmp_path: Path):
    f = tmp_path / "model.joblib"
    f.write_bytes(b"same content")
    assert file_fingerprint(f) == file_fingerprint(f)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fingerprint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.fingerprint'`

- [ ] **Step 3: Write the implementation**

```python
# src/utils/fingerprint.py
"""Content-hash fingerprinting, shared across the training/serving boundary
gaps found in this session's investigation (BUG-066, US#173/192/193): a
model file, a training-data summary, or a live feature row can each be
hashed the same way `app/backend/agent_config_hash.py` already hashes an
AgentConfig -- sort keys, JSON dump, SHA-256 -- so two things that should
match can be compared cheaply, without needing a central version registry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def data_fingerprint(payload: dict) -> str:
    """16-hex-char fingerprint of a JSON-serializable dict. Key order and
    nesting don't matter; two dicts with the same content always match."""
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def file_fingerprint(path: str | Path) -> str:
    """16-hex-char fingerprint of a file's raw bytes -- used to detect when
    a model artifact has been silently replaced under an unchanged filename
    (the exact gap that let a stale calibrator sit next to a swapped model
    with no way to notice short of BUG-066's own manual forensic dig)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_fingerprint.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/utils/fingerprint.py tests/test_fingerprint.py
git commit -m "feat: add shared data_fingerprint()/file_fingerprint() primitives

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Phase 2: Fix the Calibrator Promotion Gate (the actual root-cause fix, not just a symptom patch)

`BUG-066`'s own root-cause writeup already names this exactly: *"nothing checked whether calibration was actually safe to apply before saving/serving it (the existing ll_before/ll_after log-loss check is measured in-sample on the exact set the calibrator was fit on, so it can't and didn't catch this)."* This session's own out-of-sample test proved it concretely: **7 of 8 real calibrators tested made predictions measurably worse on genuinely held-out data**, despite every one showing a clean in-sample "improvement." This phase makes that the actual gate, so a bad calibrator can no longer be *saved* in the first place — not just detected after the fact.

**Files:**
- Modify: `src/models/model_manager.py:304-371` (`_fit_and_save_calibrator`)
- Modify: `src/models/model_manager.py` (its one call site inside `run_pipeline`)
- Test: `tests/test_model_manager_calibration_small_sample.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_model_manager_calibration_small_sample.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.model_manager import ModelManager


class _AlwaysWorseOnTestModel:
    """A fake binary classifier whose predict_proba is well-behaved on
    whatever data it's asked about, but where the fitted calibrator
    (isotonic, on a tiny synthetic val set) will provably NOT generalize --
    reproduces the exact BUG-066 shape (clean in-sample gain, real
    out-of-sample loss) without needing a real 1000+-row dataset."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Constant, uninformative predictions -- deliberately gives the
        # calibrator nothing real to learn, so any isotonic fit on a tiny
        # val set here is pure noise-fitting, matching BUG-066's shape.
        n = len(X)
        return np.tile(np.array([0.5, 0.5]), (n, 1))


def test_fit_and_save_calibrator_rejects_a_calibrator_that_hurts_on_held_out_test(tmp_path):
    model = _AlwaysWorseOnTestModel()
    # Tiny val set: isotonic will happily overfit a plateau to these 6 points.
    X_val = pd.DataFrame({"f": range(6)})
    y_val = pd.Series([1, 0, 1, 0, 1, 0])
    # Test set the SAME shape must generalize to -- genuinely different
    # labels than what the val-set plateau would predict well.
    X_test = pd.DataFrame({"f": range(6)})
    y_test = pd.Series([0, 0, 0, 0, 0, 0])  # real rate 0.0, nothing like val's 0.5

    model_path = tmp_path / "fake_model.joblib"
    model_path.write_bytes(b"fake model bytes")

    result = ModelManager._fit_and_save_calibrator(model, X_val, y_val, model_path, X_test=X_test, y_test=y_test)

    assert result is None, "a calibrator that doesn't generalize to held-out test data must not be saved at all"
    assert not model_path.with_suffix(model_path.suffix + ".calibration.pkl").exists()


def test_fit_and_save_calibrator_still_saves_a_calibrator_that_genuinely_generalizes(tmp_path):
    # A real, separable dataset large enough that sigmoid calibration
    # (below _MIN_ISOTONIC_SAMPLES) genuinely helps and generalizes --
    # val and test drawn from the same real distribution, not adversarial.
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    y = (x + rng.normal(scale=0.5, size=200) > 0).astype(int)
    X_val = pd.DataFrame({"f": x[:100]})
    y_val = pd.Series(y[:100])
    X_test = pd.DataFrame({"f": x[100:]})
    y_test = pd.Series(y[100:])

    lr = LogisticRegression().fit(X_val[["f"]], y_val)

    class _RealModel:
        def predict_proba(self, X):
            return lr.predict_proba(X)

    model_path = tmp_path / "real_model.joblib"
    model_path.write_bytes(b"real model bytes")

    result = ModelManager._fit_and_save_calibrator(_RealModel(), X_val, y_val, model_path, X_test=X_test, y_test=y_test)

    assert result is not None
    assert model_path.with_suffix(model_path.suffix + ".calibration.pkl").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_model_manager_calibration_small_sample.py -v -k held_out`
Expected: FAIL with `TypeError: _fit_and_save_calibrator() got an unexpected keyword argument 'X_test'`

- [ ] **Step 3: Modify `_fit_and_save_calibrator` to take and enforce the held-out gate**

In `src/models/model_manager.py`, change the signature and add the gate right after `sidecar` is built (both the binary and multiclass branches already compute `ll_before`/`ll_after` on `X_val`/`y_val` — leave those as reported diagnostics, they're informative even though they're not the gate anymore) and before `joblib.dump(sidecar, ...)`:

```python
    @staticmethod
    def _fit_and_save_calibrator(
        model: FPAIBaseModel,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_path: Path,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
    ) -> dict[str, float] | None:
        """Fit a probability calibrator on val-set probabilities and save as sidecar.

        X_test/y_test (optional, but always passed by run_pipeline): the
        REAL promotion gate, added after BUG-066 found the val-set-only
        ll_before/ll_after check can't distinguish a genuinely better
        calibrator from one that just overfits a plateau to the exact data
        it was fit on -- confirmed live, 7 of 8 real production calibrators
        tested showed a clean in-sample gain and a real out-of-sample loss.
        A calibrator that doesn't ALSO improve log_loss on X_test (never
        touched during fitting) is not saved at all. Omitting X_test/y_test
        preserves the old val-only behavior for any caller that hasn't been
        updated yet -- but every current caller (run_pipeline) now passes
        both, so this only matters for tests exercising the pre-gate shape.

        Returns a dict with log_loss before/after calibration (both
        measured on X_val, for continuity with existing diagnostics/mlflow
        logging), or None if calibration was skipped or failed the gate.
        """
```

Then, right after the `sidecar = {...}` line in each branch (both binary and multiclass), before `cal_path = model_path.with_suffix(...)`, insert the gate:

```python
            if X_test is not None and y_test is not None:
                test_raw_proba = np.asarray(model.predict_proba(X_test))
                if sidecar["type"] == "binary":
                    y_test_arr = pd.to_numeric(y_test, errors="coerce").astype(float).to_numpy()
                    test_cal_pos = sidecar["calibrator"].predict(test_raw_proba[:, 1])
                    test_cal_proba = np.stack([1 - test_cal_pos, test_cal_pos], axis=1)
                    test_ll_before = float(log_loss(y_test_arr, test_raw_proba))
                    test_ll_after = float(log_loss(y_test_arr, test_cal_proba))
                else:
                    y_test_raw = y_test.to_numpy()
                    test_cal_proba = np.zeros_like(test_raw_proba)
                    for c, cal in enumerate(sidecar["calibrator"]):
                        test_cal_proba[:, c] = cal.predict(test_raw_proba[:, c])
                    test_cal_proba /= test_cal_proba.sum(axis=1, keepdims=True).clip(min=1e-9)
                    test_ll_before = float(log_loss(y_test_raw, test_raw_proba, labels=sidecar.get("classes")))
                    test_ll_after = float(log_loss(y_test_raw, test_cal_proba, labels=sidecar.get("classes")))
                if test_ll_after >= test_ll_before:
                    LOGGER.warning(
                        "Calibration REJECTED (fails held-out gate) | val: before=%.4f after=%.4f | "
                        "test: before=%.4f after=%.4f -- not saved",
                        ll_before, ll_after, test_ll_before, test_ll_after,
                    )
                    return None
```

- [ ] **Step 4: Update the one call site to pass `X_test`/`y_test`**

In `run_pipeline`'s `_run_training()` closure, `X_test`/`y_test` are already in scope (used a few lines earlier for `_evaluate_target`). Change:
```python
                    cal_metrics = self._fit_and_save_calibrator(self.model, X_val, y_val, save_path)
```
to:
```python
                    cal_metrics = self._fit_and_save_calibrator(self.model, X_val, y_val, save_path, X_test=X_test, y_test=y_test)
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `python3 -m pytest tests/test_model_manager_calibration_small_sample.py -v`
Expected: PASS (all tests, including the 2 new ones)

- [ ] **Step 6: Run the full calibration/sample-weight test suite to confirm no regressions**

Run: `python3 -m pytest tests/test_model_manager_calibration_small_sample.py tests/test_model_manager_calibration_multiclass.py tests/test_model_manager_sample_weight.py tests/test_model_manager_time_decay.py tests/test_model_manager_refit_full_data.py -q`
Expected: all pass (the existing small-sample/multiclass tests call `_fit_and_save_calibrator` without `X_test`/`y_test` — confirm they still pass under the now-optional defaults, i.e. the gate is skipped and old behavior is preserved when a caller doesn't pass them)

- [ ] **Step 7: Commit**

```bash
git add src/models/model_manager.py tests/test_model_manager_calibration_small_sample.py
git commit -m "fix: gate calibrator promotion on real held-out (X_test) improvement, not just in-sample (X_val)

BUG-066's own root-cause writeup named this exact gap: the ll_before/
ll_after check was measured on the same data the calibrator was fit to,
so it couldn't distinguish a genuinely better calibrator from one that
overfits a plateau to that data. Confirmed this session: 7 of 8 real
production calibrators showed a clean in-sample gain and a real
out-of-sample loss. A calibrator that doesn't also improve on genuinely
held-out X_test is no longer saved at all.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Phase 3: `model_fingerprint` in the Calibrator Sidecar (defense in depth)

Phase 2 stops a *new* bad calibrator from being saved. This phase catches the *other* half of `BUG-066`'s failure mode: a calibrator that was fine when saved, sitting next to a model file that's since been silently replaced under the same filename (exactly the manual-`.broken_bug066`-rename mechanism this session used, made automatic instead of requiring a human to notice).

**Files:**
- Modify: `src/models/model_manager.py` (`_fit_and_save_calibrator` — store the fingerprint)
- Modify: `src/forecast/forecast_service.py` (`_load_calibrator_sidecar` — check it)
- Test: `tests/test_forecast_service_calibration.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_forecast_service_calibration.py
import joblib

from src.forecast.forecast_service import ForecastService


def test_load_calibrator_sidecar_self_disables_when_model_file_changed(tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"original model bytes")

    cal_path = tmp_path / "model.joblib.calibration.pkl"
    from src.utils.fingerprint import file_fingerprint
    joblib.dump({"type": "binary", "calibrator": object(), "model_fingerprint": file_fingerprint(model_path)}, cal_path)

    # Sidecar matches the model right now.
    assert ForecastService._load_calibrator_sidecar(model_path) is not None

    # Model file silently replaced under the same filename (the exact
    # BUG-066 gap) -- the sidecar's stored fingerprint no longer matches.
    model_path.write_bytes(b"a completely different model")
    assert ForecastService._load_calibrator_sidecar(model_path) is None


def test_load_calibrator_sidecar_still_works_for_a_pre_fingerprint_sidecar(tmp_path):
    """A sidecar saved before this change has no model_fingerprint key at
    all -- must not crash, and must still load (can't validate what was
    never recorded, same as any other additive metadata field in this
    codebase's own convention)."""
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"some model bytes")
    cal_path = tmp_path / "model.joblib.calibration.pkl"
    joblib.dump({"type": "binary", "calibrator": object()}, cal_path)

    assert ForecastService._load_calibrator_sidecar(model_path) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_forecast_service_calibration.py -v -k self_disables`
Expected: FAIL (either an `ImportError` for `file_fingerprint`, once Phase 1 exists this becomes a real behavioral failure: `_load_calibrator_sidecar` currently returns the sidecar unconditionally, so the second assertion fails)

- [ ] **Step 3: Store the fingerprint at save time**

In `src/models/model_manager.py`, add the import:
```python
from src.utils.fingerprint import file_fingerprint
```
Then, right after `joblib.dump(sidecar, str(cal_path))` in `_fit_and_save_calibrator`, add one line before the dump instead — the fingerprint must be *inside* the dumped dict, so insert it into `sidecar` right before the `joblib.dump` call:
```python
            sidecar["model_fingerprint"] = file_fingerprint(model_path)
            cal_path = model_path.with_suffix(model_path.suffix + ".calibration.pkl")
            joblib.dump(sidecar, str(cal_path))
```
(This replaces the existing two lines with the same two lines plus the one new line directly above them — `model_path` is already a parameter of this function, and by this point in the flow the model has already been saved to `model_path` by `run_pipeline` before `_fit_and_save_calibrator` is ever called, so hashing it here reflects the real, final artifact.)

- [ ] **Step 4: Check the fingerprint at load time**

In `src/forecast/forecast_service.py`, there is currently no module-level `LOGGER` at all (confirmed by reading the file, not assumed — `grep -n "^LOGGER\|get_logger"` returns nothing). Add both the fingerprint import and a real logger, matching every other file in this codebase's own convention (`src/utils/logger.py`'s `get_logger`):
```python
from src.utils.fingerprint import file_fingerprint
from src.utils.logger import get_logger
```
```python
LOGGER = get_logger(__name__)
```
(Place the `LOGGER = get_logger(__name__)` line after the existing imports, same position every other module in this codebase uses it.) Then modify `_load_calibrator_sidecar`:
```python
    @staticmethod
    def _load_calibrator_sidecar(model_path: Path) -> dict[str, Any] | None:
        """US#186: load a model's saved isotonic-calibration sidecar, if
        ModelManager._fit_and_save_calibrator wrote one alongside it.

        Self-disables (returns None) if the sidecar's recorded
        model_fingerprint no longer matches the model file currently on
        disk -- the automatic version of BUG-066's manual `.broken_bug066`
        rename: a model silently replaced under the same filename without
        its calibrator being refit no longer serves a stale calibration
        undetected. A sidecar with no model_fingerprint at all (saved
        before this change) is trusted as before -- nothing to check it
        against."""
        cal_path = model_path.with_suffix(model_path.suffix + ".calibration.pkl")
        if not cal_path.exists():
            return None
        sidecar = joblib.load(str(cal_path))
        stored_fingerprint = sidecar.get("model_fingerprint")
        if stored_fingerprint is not None and stored_fingerprint != file_fingerprint(model_path):
            LOGGER.warning(
                "Calibrator sidecar for %s is stale (model file changed since calibration was fit) -- ignoring.",
                model_path.name,
            )
            return None
        return sidecar
```
(Add `LOGGER = get_logger(__name__)` at module level if `forecast_service.py` doesn't already have one — check first; several other files in this codebase already follow this exact pattern, likely already present.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_forecast_service_calibration.py -v`
Expected: PASS (all tests, including the 2 new ones)

- [ ] **Step 6: Run the full forecast_service test suite to confirm no regressions**

Run: `python3 -m pytest tests/test_forecast_service_calibration.py tests/test_forecast_service_composite_model_loading.py tests/test_forecast_registry_fallback.py tests/test_per_competition_context.py -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/models/model_manager.py src/forecast/forecast_service.py tests/test_forecast_service_calibration.py
git commit -m "feat: self-disable a calibrator sidecar when its model file has been replaced

Automatic version of BUG-066's manual .broken_bug066 rename -- a stale
calibrator no longer requires a human to notice the model changed
underneath it.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Phase 4: `training_data_fingerprint` + Live `feature_fingerprint` (closes the snapshot-vs-live drift gap)

This is the piece that would have turned this session's hour-long forensic reconstruction (comparing raw vs. calibrated probabilities to guess whether a snapshot matched live) into a single equality check.

**Files:**
- Modify: `src/models/model_manager.py` (`_build_artifact_metadata`)
- Modify: `src/forecast/forecast_service.py` (`_predict_target`)
- Test: `tests/test_model_manager_time_decay.py` or a new small test file — extending an existing one that already exercises `_build_artifact_metadata`-adjacent behavior keeps this DRY; use whichever already imports `ModelManager` and has the lightest fixture setup (`test_model_manager_refit_full_data.py`'s `_wire_common_mocks` pattern is the closest match).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_model_manager_refit_full_data.py
def test_metadata_includes_training_data_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = _CapturingModel()
    manager = _make_manager(tmp_path, model, refit_on_full_data=False)
    written, _ = _wire_common_mocks(manager, monkeypatch, n_train=10, n_val=3, n_test=3)

    manager.run_pipeline(external_run=True)

    assert "training_data_fingerprint" in written["metadata"]
    assert len(written["metadata"]["training_data_fingerprint"]) == 16
```

Note: `_wire_common_mocks` mocks `_build_artifact_metadata` to return `{}` — for this test to be meaningful, stop mocking it away and let the real method run. Add a second, more targeted unit test instead that calls `_build_artifact_metadata` directly:

```python
def test_build_artifact_metadata_fingerprint_is_deterministic_for_same_data(tmp_path: Path):
    model = _CapturingModel()
    manager = _make_manager(tmp_path, model, refit_on_full_data=False)
    X_val = pd.DataFrame({"f1": [1, 2, 3]})
    y_val = pd.Series(["home", "draw", "away"])
    model_path = tmp_path / "fake.joblib"
    model_path.write_bytes(b"x")

    meta1 = manager._build_artifact_metadata(model_path, ["f1"], X_val, y_val)
    meta2 = manager._build_artifact_metadata(model_path, ["f1"], X_val, y_val)

    assert meta1["training_data_fingerprint"] == meta2["training_data_fingerprint"]
    assert len(meta1["training_data_fingerprint"]) == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_model_manager_refit_full_data.py -v -k fingerprint`
Expected: FAIL with `KeyError: 'training_data_fingerprint'`

- [ ] **Step 3: Add the fingerprint to `_build_artifact_metadata`**

In `src/models/model_manager.py`, add the import:
```python
from src.utils.fingerprint import data_fingerprint, file_fingerprint
```
(This subsumes the `file_fingerprint` import already added in Phase 3 — one import line covering both.) Then, inside `_build_artifact_metadata`, after `metadata = {...}` is built, add one line before `return metadata`:
```python
        metadata["training_data_fingerprint"] = data_fingerprint({
            "n_rows": len(X_val),
            "columns": sorted(X_val.columns),
            "value_summary": {col: round(float(X_val[col].mean()), 6) for col in sorted(X_val.columns) if pd.api.types.is_numeric_dtype(X_val[col])},
        })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_model_manager_refit_full_data.py -v`
Expected: PASS (all tests, including the 2 new ones; the original 4 tests from the `refit_on_full_data` plan still pass since `_build_artifact_metadata` is mocked away there and unaffected)

- [ ] **Step 5: Stamp `feature_fingerprint` into live/snapshot forecast diagnostics**

In `src/forecast/forecast_service.py`, add the import (if not already added in Phase 3):
```python
from src.utils.fingerprint import data_fingerprint
```
In `_predict_target`, right after `raw_proba = np.asarray(model.predict_proba(feature_row))` (the very first line inside the classification branch), the `feature_row` DataFrame is already sliced to this target's own columns — stamp it once, right where the existing `raw_probabilities` diagnostic already gets added (same `if calibrator is not None:` block is the wrong place — this should be **unconditional**, not gated on a calibrator existing, since the whole point is knowing what fed *any* prediction). Add immediately after `result = {"probabilities": probability_map, "uncertainty": ...}` is built:
```python
            result["feature_fingerprint"] = data_fingerprint({
                col: round(float(val), 6) if pd.notna(val) else None
                for col, val in feature_row.iloc[0].items()
            })
```

- [ ] **Step 6: Run the forecast_service test suite to confirm no regressions**

Run: `python3 -m pytest tests/test_forecast_service_calibration.py tests/test_forecast_payload.py tests/test_forecast_service_composite_model_loading.py -q`
Expected: all pass (this is a purely additive dict key — no existing test asserts an exact/closed dict shape for this response, but confirm by running rather than assuming)

- [ ] **Step 7: Commit**

```bash
git add src/models/model_manager.py src/forecast/forecast_service.py tests/test_model_manager_refit_full_data.py
git commit -m "feat: stamp training_data_fingerprint (metadata) and feature_fingerprint (live/snapshot) for drift detection

Closes the gap this session spent real time forensically reconstructing
by hand -- 'did this snapshot see the same feature values live serving
sees now' becomes an equality check instead of a raw-vs-calibrated
probability comparison.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** shared primitive (Phase 1) ✓, the actual root-cause fix for the calibrator gate (Phase 2) ✓, model-swap self-detection (Phase 3) ✓, training/serving drift detection (Phase 4) ✓. The two explicitly-deferred pieces (`updated_at` columns, `agent_config_hash` folding) are named in the Scope section, not silently dropped.

**Placeholder scan:** every step has real, complete code or an exact runnable command — no TBD/"add appropriate handling" found.

**Type consistency:** `_fit_and_save_calibrator`'s new `X_test`/`y_test` parameters are optional (`None` default) in both the signature (Phase 2 Step 3) and its one call site (Phase 2 Step 4) — a caller that doesn't pass them (every existing calibration unit test) keeps working exactly as before, confirmed by Phase 2 Step 6 rather than assumed. `data_fingerprint`/`file_fingerprint` signatures match between Phase 1's implementation and every later phase's usage (Phase 3's sidecar check, Phase 4's metadata/live stamping).
