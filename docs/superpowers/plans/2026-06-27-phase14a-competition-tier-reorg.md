# Phase 14a: Model Tier Reorg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded `MKT_FEATURES if context == "international" else None` logic in `main.py` with a config-driven competition registry (`config/competitions.yaml`) that resolves a competition's model tier (`general_purpose` vs `competition_specific`) and feature subset, without changing any existing CLI/MCP behavior.

**Architecture:** A new `src/logic/competition_registry.py` module (mirroring the existing `src/logic/target_registry.py` pattern) loads `config/competitions.yaml` and exposes `get_competition_definition()`, `list_competition_definitions()`, and `resolve_feature_subset_for_tier()`. `main.py:run_train_target` maps the existing `context` CLI value to an implicit `competition_id` (`"league"` → `"E0"`, `"international"` → `"international"`) and resolves the feature subset through the registry instead of the inline constant. No new tables, no ingestion changes — this phase only touches the model-tier resolution seam. Corresponds to US#87–90 in `documents/user_stories.md`.

**Tech Stack:** Python, PyYAML (already a dependency), pytest.

---

### Task 1: Create the competition registry config

**Files:**
- Create: `config/competitions.yaml`

- [ ] **Step 1: Write the registry file**

```yaml
# Competition registry (Phase 14a, US#87).
# Maps each competition_id to its model tier and feature scope.
# tier: "general_purpose" (market-odds-only, works for any competition)
#       or "competition_specific" (full team-form feature set; player
#       features are added here in Phase 14c, gated by enabled_feature_groups).
competitions:
  E0:
    competition_id: E0
    tier: competition_specific
    league_code: E0
    enabled_feature_groups:
      - OFF
      - DEF
      - DIS
      - CTX
      - MKT
      - STRENGTH
      - INTERACTION
      - EFFICIENCY
    player_data_sources: []
  international:
    competition_id: international
    tier: general_purpose
    league_code: null
    enabled_feature_groups:
      - MKT
    player_data_sources: []
```

- [ ] **Step 2: Validate the YAML parses**

Run: `python -c "import yaml; print(yaml.safe_load(open('config/competitions.yaml'))['competitions'].keys())"`
Expected: `dict_keys(['E0', 'international'])`

- [ ] **Step 3: Commit**

```bash
git add config/competitions.yaml
git commit -m "feat: add competition registry config (US#87)"
```

---

### Task 2: Build the competition registry module (TDD)

**Files:**
- Create: `src/logic/competition_registry.py`
- Test: `tests/test_competition_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logic.competition_registry import (
    GENERAL_PURPOSE_FEATURES,
    get_competition_definition,
    list_competition_definitions,
    resolve_feature_subset_for_tier,
)


def test_get_competition_definition_for_e0() -> None:
    definition = get_competition_definition("E0")
    assert definition.competition_id == "E0"
    assert definition.tier == "competition_specific"
    assert definition.league_code == "E0"
    assert "MKT" in definition.enabled_feature_groups


def test_get_competition_definition_for_international() -> None:
    definition = get_competition_definition("international")
    assert definition.tier == "general_purpose"
    assert definition.league_code is None


def test_get_competition_definition_rejects_unknown_competition() -> None:
    with pytest.raises(ValueError, match="Unknown competition"):
        get_competition_definition("nonexistent")


def test_list_competition_definitions_is_stable() -> None:
    names = [definition.competition_id for definition in list_competition_definitions()]
    assert names == sorted(names)
    assert set(names) == {"E0", "international"}


def test_resolve_feature_subset_for_general_purpose_matches_legacy_mkt_features() -> None:
    # Regression guard: this must equal the old MKT_FEATURES constant that
    # used to live in main.py, so behavior is unchanged after the refactor.
    assert resolve_feature_subset_for_tier("general_purpose") == [
        "MKT_IMPLIED_HOME",
        "MKT_IMPLIED_DRAW",
        "MKT_IMPLIED_AWAY",
        "MKT_OVERROUND",
        "MKT_LAMBDA_TOTAL",
        "MKT_LAMBDA_HOME",
        "MKT_LAMBDA_AWAY",
        "MKT_POISSON_BTTS_PROB",
        "MKT_LAMBDA_AH_DIFF",
        "MKT_AH_LINE",
        "MKT_AH_HOME_ODDS",
        "MKT_AH_AWAY_ODDS",
        "MKT_IMPLIED_OVER25",
    ]


def test_resolve_feature_subset_for_competition_specific_is_none() -> None:
    # None tells ModelManager to use the full schema.yaml selected_features list.
    assert resolve_feature_subset_for_tier("competition_specific") is None


def test_resolve_feature_subset_for_unknown_tier_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tier"):
        resolve_feature_subset_for_tier("not_a_tier")


def test_invalid_tier_in_yaml_raises(tmp_path: Path) -> None:
    bad_registry = tmp_path / "bad_competitions.yaml"
    bad_registry.write_text(
        yaml.dump({"competitions": {"X0": {"competition_id": "X0", "tier": "made_up_tier"}}})
    )
    with pytest.raises(ValueError, match="invalid tier"):
        get_competition_definition("X0", registry_path=bad_registry)


def test_general_purpose_features_are_subset_of_full_schema() -> None:
    # US#89: enforce the feature-superset invariant. competition_specific
    # resolves to "all of config/schema.yaml's selected_features" (feature_subset=None),
    # so general_purpose's feature list must be fully contained within it.
    schema_path = Path(__file__).resolve().parents[1] / "config" / "schema.yaml"
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = yaml.safe_load(handle)
    full_feature_set = set(schema["training_setup"]["selected_features"])
    assert set(GENERAL_PURPOSE_FEATURES) <= full_feature_set
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_competition_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.logic.competition_registry'`

- [ ] **Step 3: Write the implementation**

```python
"""Central registry for competitions and their model-tier configuration.

Two tiers exist today: "general_purpose" (market-odds-only, usable for any
competition) and "competition_specific" (full team-form feature set). A
competition_specific feature list must always be a superset of
general_purpose's (see Phase 14, FRAI_TECHSPEC.md Section 27.2). If a future
tier needs an architecture where a literal feature superset doesn't apply,
the design reserves room for a competition_specific model to instead consume
the general_purpose model's own prediction as an input feature (stacking).
That stacking path is not implemented here (US#90) — this module only
resolves which feature subset a tier uses today.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

GENERAL_PURPOSE_FEATURES: list[str] = [
    "MKT_IMPLIED_HOME",
    "MKT_IMPLIED_DRAW",
    "MKT_IMPLIED_AWAY",
    "MKT_OVERROUND",
    "MKT_LAMBDA_TOTAL",
    "MKT_LAMBDA_HOME",
    "MKT_LAMBDA_AWAY",
    "MKT_POISSON_BTTS_PROB",
    "MKT_LAMBDA_AH_DIFF",
    "MKT_AH_LINE",
    "MKT_AH_HOME_ODDS",
    "MKT_AH_AWAY_ODDS",
    "MKT_IMPLIED_OVER25",
]

VALID_TIERS = ("general_purpose", "competition_specific")

DEFAULT_REGISTRY_PATH = Path("config/competitions.yaml")


@dataclass(frozen=True)
class CompetitionDefinition:
    """Definition of a registered competition's model tier and feature scope."""

    competition_id: str
    tier: str
    league_code: str | None
    enabled_feature_groups: tuple[str, ...]
    player_data_sources: tuple[str, ...] = ()


def _load_registry(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, CompetitionDefinition]:
    path = Path(registry_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing competition registry: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    competitions = raw.get("competitions", {})
    if not competitions:
        raise ValueError(f"No competitions defined in {path}")

    registry: dict[str, CompetitionDefinition] = {}
    for competition_id, entry in competitions.items():
        tier = entry.get("tier")
        if tier not in VALID_TIERS:
            raise ValueError(
                f"Competition '{competition_id}' has invalid tier '{tier}'. Must be one of {VALID_TIERS}."
            )
        registry[competition_id] = CompetitionDefinition(
            competition_id=competition_id,
            tier=tier,
            league_code=entry.get("league_code"),
            enabled_feature_groups=tuple(entry.get("enabled_feature_groups") or ()),
            player_data_sources=tuple(entry.get("player_data_sources") or ()),
        )
    return registry


def get_competition_definition(
    competition_id: str, registry_path: str | Path = DEFAULT_REGISTRY_PATH
) -> CompetitionDefinition:
    """Return the competition definition or raise a helpful error."""
    registry = _load_registry(registry_path)
    try:
        return registry[competition_id]
    except KeyError as exc:
        valid = ", ".join(sorted(registry))
        raise ValueError(f"Unknown competition '{competition_id}'. Registered competitions: {valid}") from exc


def list_competition_definitions(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[CompetitionDefinition]:
    """Return registered competition definitions in stable competition_id order."""
    registry = _load_registry(registry_path)
    return [registry[name] for name in sorted(registry)]


def resolve_feature_subset_for_tier(tier: str) -> list[str] | None:
    """Return the feature subset for a tier, or None to use the full schema.yaml list."""
    if tier == "general_purpose":
        return list(GENERAL_PURPOSE_FEATURES)
    if tier == "competition_specific":
        return None
    raise ValueError(f"Unknown tier '{tier}'. Must be one of {VALID_TIERS}.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_competition_registry.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/logic/competition_registry.py tests/test_competition_registry.py
git commit -m "feat: add competition registry module with tier resolution (US#87-90)"
```

---

### Task 3: Wire `main.py` to resolve feature subsets through the registry

**Files:**
- Modify: `main.py:21` (imports)
- Modify: `main.py:399-414` (remove `MKT_FEATURES` constant)
- Modify: `main.py:417-441` (`run_train_target`)

- [ ] **Step 1: Add the import**

In `main.py`, after the existing line:

```python
from src.logic.target_registry import get_target_definition, list_target_definitions
```

add:

```python
from src.logic.competition_registry import get_competition_definition, resolve_feature_subset_for_tier
```

- [ ] **Step 2: Remove the hardcoded `MKT_FEATURES` constant**

Delete these lines from `main.py` (currently lines 399-414):

```python
# MKT_* features used for international context models (US#85)
MKT_FEATURES = [
    "MKT_IMPLIED_HOME",
    "MKT_IMPLIED_DRAW",
    "MKT_IMPLIED_AWAY",
    "MKT_OVERROUND",
    "MKT_LAMBDA_TOTAL",
    "MKT_LAMBDA_HOME",
    "MKT_LAMBDA_AWAY",
    "MKT_POISSON_BTTS_PROB",
    "MKT_LAMBDA_AH_DIFF",
    "MKT_AH_LINE",
    "MKT_AH_HOME_ODDS",
    "MKT_AH_AWAY_ODDS",
    "MKT_IMPLIED_OVER25",
]
```

- [ ] **Step 3: Replace the inline conditional in `run_train_target`**

Change this block in `main.py` (currently lines 417-441):

```python
def run_train_target(target_name: str, model_name: str | None = None, context: str = "league") -> Path:
    """Train one registry-backed forecast target model."""
    definition = get_target_definition(target_name)
    selected_model = (model_name or _default_model_for_target(definition.name)).strip().lower()
    if selected_model not in MODEL_REGISTRY:
        valid_models = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unsupported model '{selected_model}'. Available options: {valid_models}")
    LOGGER.info("Training forecast target | target=%s | task_type=%s | model=%s | context=%s", definition.name, definition.task_type, selected_model, context)
    model_cls = MODEL_REGISTRY.get(selected_model)
    if model_cls is None:
        model = ModelFactory.get_model(selected_model)
    else:
        xgb_params = _xgb_params_for_target(target_name, selected_model)
        model = model_cls(**xgb_params)

    feature_subset = MKT_FEATURES if context == "international" else None
    model_manager = ModelManager(
        model=model,
        target_config={"target": definition.name},
        feature_subset=feature_subset,
        context=context,
    )
    model_path = model_manager.run_pipeline()
    LOGGER.info("Target model saved to %s", model_path)
    return model_path
```

to:

```python
def run_train_target(target_name: str, model_name: str | None = None, context: str = "league") -> Path:
    """Train one registry-backed forecast target model."""
    definition = get_target_definition(target_name)
    selected_model = (model_name or _default_model_for_target(definition.name)).strip().lower()
    if selected_model not in MODEL_REGISTRY:
        valid_models = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unsupported model '{selected_model}'. Available options: {valid_models}")
    LOGGER.info("Training forecast target | target=%s | task_type=%s | model=%s | context=%s", definition.name, definition.task_type, selected_model, context)
    model_cls = MODEL_REGISTRY.get(selected_model)
    if model_cls is None:
        model = ModelFactory.get_model(selected_model)
    else:
        xgb_params = _xgb_params_for_target(target_name, selected_model)
        model = model_cls(**xgb_params)

    # US#88: resolve tier/feature-subset through the competition registry
    # instead of a hardcoded context check. "league" has no fixed competition_id
    # input yet, so it maps to the one currently-registered league competition (E0).
    competition_id = "international" if context == "international" else "E0"
    competition_def = get_competition_definition(competition_id)
    feature_subset = resolve_feature_subset_for_tier(competition_def.tier)
    model_manager = ModelManager(
        model=model,
        target_config={"target": definition.name},
        feature_subset=feature_subset,
        context=context,
    )
    model_path = model_manager.run_pipeline()
    LOGGER.info("Target model saved to %s", model_path)
    return model_path
```

- [ ] **Step 4: Run the full test suite**

Run: `pytest -q`
Expected: all tests pass (same pass count as before this change, plus the 9 new competition registry tests)

- [ ] **Step 5: Manual smoke test against real data**

Run: `python main.py train-target --target btts --model xgb --context international`
Expected: log line `Training forecast target | target=btts | task_type=binary_classification | model=xgb | context=international`, training completes, and a new `models/btts_xgb_v1_<today>.joblib` artifact is written — same observable behavior as before the refactor.

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "refactor: resolve tier feature subset via competition registry (US#88)"
```

---

### Task 4: Mark Phase 14a complete in documentation

**Files:**
- Modify: `documents/FRAI_TECHSPEC.md` (Section 27 status line)
- Modify: `documents/user_stories.md` (Phase 14a heading)

- [ ] **Step 1: Update the techspec status line**

In `documents/FRAI_TECHSPEC.md` Section 27, change:

```
**Status: planned, not yet implemented.** This section records the design agreed during brainstorming so implementation can proceed against a stable spec. Story breakdown lives in `documents/user_stories.md` Phase 14.
```

to:

```
**Status: Phase 14a (tier reorg) implemented — see 27.2 for the design and `config/competitions.yaml` / `src/logic/competition_registry.py` for the implementation. Phase 14b and 14c remain planned.** Story breakdown lives in `documents/user_stories.md` Phase 14.
```

- [ ] **Step 2: Update the user stories heading**

In `documents/user_stories.md`, change:

```
### Phase 14a: Model Tier Reorg
```

to:

```
### Phase 14a: Model Tier Reorg — Completed
```

- [ ] **Step 3: Commit**

```bash
git add documents/FRAI_TECHSPEC.md documents/user_stories.md
git commit -m "docs: mark Phase 14a model tier reorg complete (US#87-90)"
```
