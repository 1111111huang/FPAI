"""Centralized configuration loading for FPAI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Filesystem paths used by the application."""

    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    database_path: str = "data/fpai_core.db"
    model_dir: str = "models"


class RuntimeConfig(BaseModel):
    """Runtime settings used across the pipeline."""

    rolling_window: int = 5
    min_matches_required: int = 10
    test_size: float = 0.2
    initial_bankroll: float = 1000.0


class ScraperConfig(BaseModel):
    """Scraper settings for automated data collection."""

    league_page_url: str = "https://www.football-data.co.uk/englandm.php"
    limit_seasons: int = 10
    timeout_seconds: int = 30
    leagues: list[str] = Field(default_factory=lambda: ["E0"])
    start_year: int = 2015


class AppSettings(BaseModel):
    """Top-level application settings parsed from config.yaml."""

    project_name: str = "FPAI-Football-Predictor"
    version: str = "1.0.0"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    settings: RuntimeConfig = Field(default_factory=RuntimeConfig)
    scraper: ScraperConfig = Field(default_factory=ScraperConfig)


@lru_cache(maxsize=32)
def load_settings(config_path: str = "config.yaml") -> AppSettings:
    """Load and validate settings from YAML, cached by path."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    # Support both Pydantic v2 (model_validate) and v1 (parse_obj).
    if hasattr(AppSettings, "model_validate"):
        return AppSettings.model_validate(raw_config)
    return AppSettings.parse_obj(raw_config)


settings = load_settings()
