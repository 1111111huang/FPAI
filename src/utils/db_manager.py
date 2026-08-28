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

    def __init__(
        self,
        config_path: str = "config.yaml",
        default_max_retries: int = 5,
        default_retry_delay_seconds: float = 1.0,
    ) -> None:
        """Initialize manager using the database path in config.yaml.

        US#159: default_max_retries/default_retry_delay_seconds let a
        caller that constructs its own DuckDBManager pick a retry window
        appropriate to how it's used, without touching every individual
        .connection() call site downstream (CSVLoader/FeatureFactory/etc.
        all just call self.db_manager.connection() with no args, inheriting
        whatever this instance was built with). Found live: main.py's CLI
        (refresh-data) and the app's live HTTP endpoints share the exact
        same W95 retry mechanism, but need very different tolerances -- a
        live user-facing request should fail fast (a few seconds) rather
        than hang a browser tab, while a CLI refresh-data run (invoked
        deliberately, unattended, for a whole league) can and should
        tolerate a much longer wait rather than crash outright, e.g. when
        two different leagues' refreshes genuinely overlap for several
        minutes (confirmed live: two refresh-data subprocesses for
        different leagues collided on data/fpai_core.db's real exclusive
        lock, and the previous fixed 5-attempt/1s window wasn't long
        enough to outlast the first one still mid-run). Defaults here
        (5/1.0) are unchanged from before this story -- every existing
        caller that doesn't pass these explicitly keeps the exact same
        short-window behavior; only main.py's CLI construction opts into a
        longer one."""
        # US#155: stored so callers holding only a DuckDBManager instance
        # (e.g. run_ingest, main.py) can still re-derive the exact config it
        # was built from -- see this field's own usage there for why that
        # matters.
        self.config_path = config_path
        self.default_max_retries = default_max_retries
        self.default_retry_delay_seconds = default_retry_delay_seconds
        self.settings = load_settings(config_path)
        self.db_path: Path = Path(self.settings.paths.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(
        self,
        read_only: bool = False,
        max_retries: int | None = None,
        retry_delay_seconds: float | None = None,
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
        asserting the un-retried failure path).

        US#159: `max_retries`/`retry_delay_seconds` left unset (`None`,
        the default) fall back to this manager instance's own
        `default_max_retries`/`default_retry_delay_seconds` (see
        `__init__`) -- lets a caller that constructs its own DuckDBManager
        pick an appropriately longer or shorter window for how it's used,
        without threading new parameters through every individual
        `.connection()` call site downstream. Passing either explicitly
        here still overrides the instance default for that one call, same
        as before."""
        if max_retries is None:
            max_retries = self.default_max_retries
        if retry_delay_seconds is None:
            retry_delay_seconds = self.default_retry_delay_seconds
        # W185 code-quality review: duckdb.ConnectionException (raised when
        # two threads concurrently duckdb.connect() the same file with
        # different configs -- confirmed live via the new weekly live-lesson
        # review job racing the existing daily one on RecoverableScheduler's
        # own concurrent-catch-up threads, W160) is a sibling of
        # IOException, not a subclass -- the retry loop below silently
        # never covered it before. Widened rather than added a second loop:
        # both are "another connection is in the way right now, try again
        # shortly" cases, same transient-collision reasoning as W95 above.
        last_exc: duckdb.IOException | duckdb.ConnectionException | None = None
        conn: duckdb.DuckDBPyConnection | None = None
        for attempt in range(max_retries + 1):
            try:
                conn = duckdb.connect(str(self.db_path), read_only=read_only)
                break
            except (duckdb.IOException, duckdb.ConnectionException) as exc:
                last_exc = exc
                if attempt < max_retries:
                    LOGGER.warning(
                        "DuckDB connection failed (attempt %d/%d), retrying in %.1fs: %s",
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
