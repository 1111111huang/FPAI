"""W27: sandbox clock -- an app-wide, overridable "as-of now". Driven by two
env vars, SANDBOX_MODE=1 and SANDBOX_DATE=YYYY-MM-DD, both absent by default
so normal operation is completely unaffected (purely additive, not a new
mode every code path must branch on defensively). Not a literal container
(Docker etc.) -- "sandbox" here means an isolated *configuration and data*
mode within the existing app.

Every backend call site that currently computes "today"/"now" directly for
date-window purposes should route through sandbox_now()/is_sandbox_mode()
instead of a bare datetime.now()."""

from __future__ import annotations

from datetime import date, datetime, tzinfo
import os
from pathlib import Path


def is_sandbox_mode() -> bool:
    return os.environ.get("SANDBOX_MODE") == "1"


def sandbox_scoped_path(filename: str) -> Path:
    """Resolves filename under app/data/sandbox/ -- the one shared root for
    every sandbox-mode scratch db (RecommendationCache/BetTracker/JobRunLog),
    so the app/backend/ vs app/ parent-depth arithmetic that already caused
    one real bug (main.py's JobRunLog path, W29) only needs to be right in
    one place."""
    return Path(__file__).parent.parent / "data" / "sandbox" / filename


def sandbox_date() -> date | None:
    """The active override date, or None if sandbox mode is off or no
    SANDBOX_DATE is set."""
    if not is_sandbox_mode():
        return None
    raw = os.environ.get("SANDBOX_DATE")
    if not raw:
        return None
    return date.fromisoformat(raw)


def sandbox_now(tz: tzinfo | None = None) -> datetime:
    """Real wall-clock 'now' (in the given tz, if any) unless sandbox mode
    is active with a SANDBOX_DATE set, in which case it returns that date
    at midnight in the given tz -- a stand-in "as-of" instant for
    date-window computations, not a literal simulated clock-tick."""
    override = sandbox_date()
    if override is not None:
        return datetime(override.year, override.month, override.day, tzinfo=tz)
    return datetime.now(tz)


def sandbox_status() -> dict:
    override = sandbox_date()
    return {"sandbox_mode": is_sandbox_mode(), "as_of": override.isoformat() if override else None}
