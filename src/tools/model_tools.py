"""Model status tools for agent/MCP consumption (US#82)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

MODEL_SELECTION_PATH = Path("config/model_selection.yaml")


def get_model_status() -> dict[str, Any]:
    """Return per-context per-target model selection status.

    Reads from config/model_selection.yaml if present; falls back to scanning
    the models/ directory for artifact filenames.

    Returns:
        Dict with keys 'league' and 'international', each mapping target names
        to {model_type, primary_metric_value, metric_name, selected_at}.
    """
    if MODEL_SELECTION_PATH.exists():
        with MODEL_SELECTION_PATH.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        contexts_data = config.get("contexts", {})
        result: dict[str, Any] = {}
        for ctx in ["league", "international"]:
            ctx_data = contexts_data.get(ctx, {})
            result[ctx] = {
                tgt: {
                    "model_type": info.get("model_type"),
                    "primary_metric_value": info.get("metric_value"),
                    "metric_name": info.get("metric_name"),
                    "selected_at": info.get("selected_at"),
                }
                for tgt, info in ctx_data.items()
            }
        return result

    # Fallback: scan models/ directory
    models_dir = Path("models")
    if not models_dir.exists():
        return {"league": {}, "international": {}}

    artifacts = sorted(models_dir.glob("*.joblib"))
    league_models: dict[str, Any] = {}
    for path in artifacts:
        parts = path.stem.split("_")
        if len(parts) >= 2:
            target = parts[0]
            model_type = parts[1] if len(parts) > 1 else "unknown"
            league_models[target] = {
                "model_type": model_type,
                "primary_metric_value": None,
                "metric_name": None,
                "selected_at": None,
                "artifact_path": str(path),
            }
    return {"league": league_models, "international": {}}
