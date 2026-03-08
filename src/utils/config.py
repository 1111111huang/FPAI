"""Shared configuration helpers for FPAI modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.config_loader import load_settings


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Return cached configuration as a plain dictionary."""
    return load_settings(config_path).model_dump()


def get_database_path(config: dict[str, Any]) -> Path:
    """Extract database path from loaded config."""
    db_path = config.get("paths", {}).get("database_path")
    if not db_path:
        raise ValueError("Missing 'paths.database_path' in config.yaml")
    return Path(db_path)
