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

SnapshotMode = Literal["live", "record", "replay", "record_missing"]

_DEFAULT_BASE_DIR = Path("data/agent_snapshots")
DEFAULT_BASE_DIR = _DEFAULT_BASE_DIR
_VALID_MODES = {"live", "record", "replay", "record_missing"}


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


# Tools that degrade to a soft "unavailable" sentinel instead of raising when
# every underlying attempt fails (e.g. _web_search_impl on a Tavily timeout/
# quota exhaustion with both keys) prefix the string this way. That's the
# right behavior for a *live* agent run (the LLM sees a clear stop signal),
# but wrong for record/record_missing: persisting it would freeze a fake
# "the tool is broken" answer into the corpus forever, indistinguishable on
# replay from a genuine result.
_DEGRADED_RESPONSE_PREFIX = "TOOL_PERMANENTLY_UNAVAILABLE"


class SnapshotRecordingDegraded(Exception):
    """Raised in record/record_missing mode when the live tool call itself
    degraded to a soft sentinel instead of a real response, instead of
    silently persisting that sentinel as the recorded answer (found live,
    BUG-072 2026-09-25: 16 matches across D1/F1 already had this baked in --
    6 from the original recording weeks earlier, 10 from a same-session
    backfill run, both invisible until grepped for directly)."""

    def __init__(self, tool: str, match_id: str | None, key: str, response: str):
        self.tool = tool
        self.match_id = match_id
        self.key = key
        super().__init__(
            f"tool={tool!r} match_id={match_id!r} key={key} returned a degraded sentinel "
            f"instead of a real response -- refusing to persist it: {response[:200]!r}"
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
        """Raw, byte-exact hash -- unchanged, still the only key format every
        snapshot in the corpus was ever written under before canonicalization
        existed. wrap() falls back to this after canonical_key_for() misses,
        so every already-recorded file stays reachable with no migration."""
        canonical = json.dumps(inputs, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_for_hashing(value: Any) -> Any:
        """Collapses superficial variance the LLM can reintroduce between an
        original recording and a later replay of the 'same' tool call, even
        at temperature=0 (provider batching effects, agent_config.yaml's own
        comment) -- whitespace/capitalization drift in free-form query text,
        and float precision noise in a re-stated odds value. Does not attempt
        to resolve a team referred to by a genuinely different name/alias
        (e.g. "Man Utd" vs "Manchester United") inside free-form query text --
        that would need parsing team names out of arbitrary LLM prose, a much
        larger and less reliable undertaking than this scoped normalization."""
        if isinstance(value, str):
            return " ".join(value.split()).lower()
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, dict):
            return {k: SnapshotStore._normalize_for_hashing(v) for k, v in value.items()}
        if isinstance(value, list):
            return [SnapshotStore._normalize_for_hashing(v) for v in value]
        return value

    @staticmethod
    def canonical_key_for(inputs: dict[str, Any]) -> str:
        """The primary key going forward: key_for(), but over normalized
        inputs first. New recordings are always written under this key
        (see wrap()) -- the corpus becomes more replay-resilient over time
        as it's extended/re-recorded, with no bulk migration required."""
        return SnapshotStore.key_for(SnapshotStore._normalize_for_hashing(inputs))

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

            # Canonicalization (whitespace/case/float-precision normalization,
            # see _normalize_for_hashing) is the primary lookup -- tolerant of
            # superficial LLM re-generation drift between the original
            # recording and this replay. The raw, byte-exact key is checked
            # second, purely for backward compatibility: every snapshot ever
            # written before this existed is keyed that way, and staying
            # readable via this fallback means the whole existing corpus
            # never needs a bulk migration -- it organically moves onto the
            # canonical key as matches get re-recorded/topped-up going
            # forward, since every new write below always uses it.
            canonical_key = self.canonical_key_for(kwargs)
            canonical_path = self._path(tool, canonical_key)
            raw_key = self.key_for(kwargs)
            raw_path = self._path(tool, raw_key)

            if mode == "replay":
                if canonical_path.exists():
                    payload = json.loads(canonical_path.read_text(encoding="utf-8"))
                    return payload["response"]
                if raw_path.exists():
                    payload = json.loads(raw_path.read_text(encoding="utf-8"))
                    return payload["response"]
                raise SnapshotMissingError(tool, self.match_id, canonical_key)

            if mode == "record_missing":
                # BUG-072/A56: a key this exact match already has -- reuse it
                # unchanged, same as replay, rather than re-fetching/re-billing
                # a call whose recording is still perfectly valid.
                if canonical_path.exists():
                    payload = json.loads(canonical_path.read_text(encoding="utf-8"))
                    return payload["response"]
                if raw_path.exists():
                    payload = json.loads(raw_path.read_text(encoding="utf-8"))
                    return payload["response"]

            # record (also reached by record_missing for a key with no
            # existing file under either convention -- live-fetch and write
            # it, same as pure record). Always written under the canonical
            # key, never the raw one, so the corpus migrates onto it over time.
            response = fn(**kwargs)
            if isinstance(response, str) and response.startswith(_DEGRADED_RESPONSE_PREFIX):
                raise SnapshotRecordingDegraded(tool, self.match_id, canonical_key, response)
            canonical_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "tool": tool,
                "inputs": kwargs,
                "response": response,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            canonical_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            return response

        return wrapped
