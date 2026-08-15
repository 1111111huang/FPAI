"""Record/replay interceptor for agent tool calls (A09).

Lets every tool function in src/agent/tools.py run unmodified in three modes:
  - live:   call the real implementation, no interception
  - record: call the real implementation, save {tool, inputs, response} to disk
  - replay: never call the real implementation — load the saved response or
            raise SnapshotMissingError immediately (no silent fallback)

Mode and match context are stored in contextvars.ContextVar, NOT threading.local().
This matters because LangGraph's ToolNode executes every tool call — even a single
one — via langchain_core's get_executor_for_config(), which returns a
ContextThreadPoolExecutor. That executor explicitly copies the calling thread's
contextvars.Context into its worker thread (copy_context().run(...)); it does
NOT carry over plain threading.local() state, which is strictly per-OS-thread.
With threading.local(), configure_snapshot_store() on the calling thread was
invisible inside ToolNode's worker thread, so every tool call silently read the
default ("live") mode no matter what record/replay mode was actually configured —
record mode wrote zero snapshot files, and replay mode never replayed anything
(see agent_techspec.md Section 18 for the full incident writeup).

contextvars.ContextVar still gives the cross-match isolation A09/A14 need:
asyncio.to_thread() (used by agent-backtest --concurrency) and
ContextThreadPoolExecutor both copy context on dispatch, so each concurrently
running match gets its own independent context snapshot — but a bare
threading.Thread() (not used anywhere in this codebase) would NOT inherit it,
since plain threads don't copy context automatically. Do not revert to
threading.local() without re-verifying this propagates through ToolNode.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

SnapshotMode = Literal["live", "record", "replay"]

_DEFAULT_BASE_DIR = Path("data/agent_snapshots")
DEFAULT_BASE_DIR = _DEFAULT_BASE_DIR
_VALID_MODES = {"live", "record", "replay"}


def league_base_dir(league: str | None, base_dir: str | Path = _DEFAULT_BASE_DIR) -> Path:
    """BUG-022: match_id is a content hash with no league component, so
    without this, every league's recordings land in the same flat directory
    and nothing on disk distinguishes them -- a cleanup scoped to one league
    can silently destroy another's in-progress work (this actually happened:
    an E0 corpus cleanup wiped a concurrently-running SWE snapshot job).
    Normalizes league to a stable, case-insensitive directory name."""
    safe_league = (league or "").strip().upper() or "unknown"
    return Path(base_dir) / safe_league


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
        self._mode_var: contextvars.ContextVar[SnapshotMode] = contextvars.ContextVar(
            "snapshot_mode", default="live"
        )
        self._match_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "snapshot_match_id", default=None
        )
        self._match_date_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "snapshot_match_date", default=None
        )
        # A41: lets lessons_node load approved lessons during a *replay* run,
        # not just "live" -- only meaningful for a held-out test-split
        # backtest (agent-backtest --split test --use-lessons), never set for
        # an ordinary backtest/train replay. Default False preserves A33's
        # original leakage guard for every other caller.
        self._allow_lessons_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
            "snapshot_allow_lessons_in_replay", default=False
        )
        # Per-tool mode override, taking precedence over `mode` in wrap()'s
        # dispatch -- lets a caller run e.g. mode="replay" globally (frozen
        # web_search/resolve_competition, no new Tavily calls) while forcing
        # just forecast_league/forecast_international into "record" to pick
        # up a newly retrained model, without re-recording everything else.
        # Never mutated in place (always replaced wholesale via .set()), so
        # sharing the same default {} across contexts before any .set() is safe.
        self._tool_overrides_var: contextvars.ContextVar[dict[str, "SnapshotMode"]] = contextvars.ContextVar(
            "snapshot_tool_mode_overrides", default={}
        )

    @property
    def mode(self) -> SnapshotMode:
        return self._mode_var.get()

    @property
    def match_id(self) -> str | None:
        return self._match_id_var.get()

    @property
    def match_date(self) -> str | None:
        return self._match_date_var.get()

    @property
    def allow_lessons_in_replay(self) -> bool:
        return self._allow_lessons_var.get()

    @property
    def tool_mode_overrides(self) -> dict[str, SnapshotMode]:
        return self._tool_overrides_var.get()

    def set_mode(self, mode: SnapshotMode) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown snapshot mode: {mode!r}")
        self._mode_var.set(mode)

    def set_tool_mode_overrides(self, overrides: dict[str, SnapshotMode]) -> None:
        for tool, mode in overrides.items():
            if mode not in _VALID_MODES:
                raise ValueError(f"Unknown snapshot mode: {mode!r} for tool {tool!r}")
        self._tool_overrides_var.set(dict(overrides))

    def set_match(self, match_id: str, match_date: str | None = None) -> None:
        self._match_id_var.set(match_id)
        self._match_date_var.set(match_date)

    def set_allow_lessons_in_replay(self, allow: bool) -> None:
        self._allow_lessons_var.set(allow)

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
            mode = self.tool_mode_overrides.get(tool, self.mode)
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
