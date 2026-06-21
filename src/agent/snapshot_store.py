"""Record/replay interceptor for agent tool calls (A09).

Lets every tool function in src/agent/tools.py run unmodified in three modes:
  - live:   call the real implementation, no interception
  - record: call the real implementation, save {tool, inputs, response} to disk
  - replay: never call the real implementation — load the saved response or
            raise SnapshotMissingError immediately (no silent fallback)

Mode and match context are stored in thread-local state so concurrent backtest
runs (each on its own thread via asyncio.to_thread) never clobber each other's
snapshot context. This must not be relaxed to plain instance attributes without
re-checking A14 (agent-backtest --concurrency).
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

SnapshotMode = Literal["live", "record", "replay"]

_DEFAULT_BASE_DIR = Path("data/agent_snapshots")
_VALID_MODES = {"live", "record", "replay"}


class SnapshotMissingError(Exception):
    """Raised in replay mode when no recorded snapshot exists for a tool call."""

    def __init__(self, tool: str, match_id: str | None, key: str):
        self.tool = tool
        self.match_id = match_id
        self.key = key
        super().__init__(
            f"No snapshot found for tool={tool!r} match_id={match_id!r} key={key} "
            "(run agent-snapshot in record mode for this match first)"
        )


class SnapshotStore:
    """Intercepts tool calls to record live responses or replay recorded ones."""

    def __init__(self, base_dir: str | Path = _DEFAULT_BASE_DIR) -> None:
        self.base_dir = Path(base_dir)
        self._local = threading.local()

    @property
    def mode(self) -> SnapshotMode:
        return getattr(self._local, "mode", "live")

    @property
    def match_id(self) -> str | None:
        return getattr(self._local, "match_id", None)

    @property
    def match_date(self) -> str | None:
        return getattr(self._local, "match_date", None)

    def set_mode(self, mode: SnapshotMode) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown snapshot mode: {mode!r}")
        self._local.mode = mode

    def set_match(self, match_id: str, match_date: str | None = None) -> None:
        self._local.match_id = match_id
        self._local.match_date = match_date

    @staticmethod
    def key_for(inputs: dict[str, Any]) -> str:
        canonical = json.dumps(inputs, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, tool: str, key: str) -> Path:
        match_id = self.match_id
        if not match_id:
            raise ValueError("SnapshotStore.set_match() must be called before record/replay use")
        return self.base_dir / match_id / f"{tool}_{key}.json"

    def wrap(self, tool: str, fn: Callable[..., str]) -> Callable[..., str]:
        """Return a callable that records or replays fn's output based on the current mode."""

        def wrapped(**kwargs: Any) -> str:
            mode = self.mode
            if mode == "live":
                return fn(**kwargs)

            key = self.key_for(kwargs)
            path = self._path(tool, key)

            if mode == "replay":
                if not path.exists():
                    raise SnapshotMissingError(tool, self.match_id, key)
                payload = json.loads(path.read_text(encoding="utf-8"))
                return payload["response"]

            # record
            response = fn(**kwargs)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "tool": tool,
                "inputs": kwargs,
                "response": response,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            return response

        return wrapped
