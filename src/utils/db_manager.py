"""Shared DuckDB connection management for FPAI modules."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time
from typing import Callable, Generator

import duckdb

from src.utils.config_loader import load_settings
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class DuckDBManager:
    """Create and manage DuckDB connections from project configuration."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize manager using the database path in config.yaml."""
        # US#155: stored so callers holding only a DuckDBManager instance
        # (e.g. run_ingest, main.py) can still re-derive the exact config it
        # was built from -- see this field's own usage there for why that
        # matters.
        self.config_path = config_path
        self.settings = load_settings(config_path)
        self.db_path: Path = Path(self.settings.paths.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(
        self,
        read_only: bool = False,
        max_retries: int = 5,
        retry_delay_seconds: float = 1.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Yield a DuckDB connection and guarantee clean close after use.

        W95 (documents/app_user_stories.md Phase 21): DuckDB enforces a real
        exclusive file lock (confirmed live, W93/W94's own investigation) --
        a second process opening *any* connection (even read-only) while
        another is open elsewhere fails immediately with
        duckdb.IOException; it does not block/wait on its own. Retries a
        fixed number of times with a fixed delay before giving up, rather
        than failing on the first attempt, since a collision is expected to
        be transient -- both the ML-engine refresh's own connections
        (US#154/US#155, opened/closed per operation, never held across the
        whole multi-step chain) and the app's own per-match reads are
        short-lived. Direct user decision: this should retry rather than
        leave `run_refresh_data` (and every other caller) to fail and wait
        for the next scheduled fire, or wait for W93's own 503 to surface
        to the end user, on what's expected to be at most a few seconds'
        genuine overlap. `max_retries=0` preserves the exact prior
        fail-immediately behavior for any caller that wants it (or a test
        asserting the un-retried failure path)."""
        last_exc: duckdb.IOException | None = None
        conn: duckdb.DuckDBPyConnection | None = None
        for attempt in range(max_retries + 1):
            try:
                conn = duckdb.connect(str(self.db_path), read_only=read_only)
                break
            except duckdb.IOException as exc:
                last_exc = exc
                if attempt < max_retries:
                    LOGGER.warning(
                        "DuckDB file locked (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, max_retries + 1, retry_delay_seconds, exc,
                    )
                    sleep_fn(retry_delay_seconds)
        if conn is None:
            assert last_exc is not None
            raise last_exc
        try:
            yield conn
        finally:
            conn.close()
