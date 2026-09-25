"""A127: model-version fingerprinting for lesson staleness. Whole-competition
scope (not per-market/target) -- agent_lessons rows aren't tagged by which
market they concern, so the fingerprint covers a competition's entire
model_selection.yaml entry (every target). Coarser than a per-target trigger,
deliberately -- see the design doc's "Trigger scope" decision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

_DEFAULT_SELECTION_PATH = Path("config/model_selection.yaml")


def compute_model_fingerprint(
    competition_id: str | None, selection_path: str | Path = _DEFAULT_SELECTION_PATH,
) -> str | None:
    """Hash of every target's full model_selection.yaml entry for one
    competition -- changes whenever ANY target's feature_subset/model_path/
    model_type/metric/selected_at changes for this competition, whether via
    ModelSelector.run() or a direct hand-edit (both are real, confirmed
    promotion paths in this codebase). None when the competition has no
    contexts entry at all (e.g. an unrecognized/leagueless competition_id)
    -- callers treat None the same as any other mismatch, never as "nothing
    to check"."""
    with open(selection_path) as f:
        config = yaml.safe_load(f) or {}
    entry = (config.get("contexts") or {}).get(competition_id)
    if entry is None:
        return None
    payload = json.dumps(entry, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
