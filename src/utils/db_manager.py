"""Shared DuckDB connection management for FPAI modules."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb

from src.utils.config_loader import load_settings


class DuckDBManager:
    """Create and manage DuckDB connections from project configuration."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize manager using the database path in config.yaml."""
        self.settings = load_settings(config_path)
        self.db_path: Path = Path(self.settings.paths.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self, read_only: bool = False) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Yield a DuckDB connection and guarantee clean close after use."""
        conn = duckdb.connect(str(self.db_path), read_only=read_only)
        try:
            yield conn
        finally:
            conn.close()
