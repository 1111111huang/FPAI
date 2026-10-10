"""Per-day fixture/result cache shared by main.py's GET /api/fixtures route
and scheduler_wiring.py's fixture-reconciliation job (W249) -- extracted
out of main.py specifically so the reconciliation job can warm/invalidate
it without creating a circular import (scheduler_wiring.py is already
imported BY main.py; this module depends on neither).

TTL is computed per day from the matches that day actually contains,
instead of a blind constant -- see compute_ttl_seconds's own docstring.
Two different concurrency domains touch this module's module-level state:
the HTTP route (always the same uvicorn event loop) and the reconciliation
job (APScheduler's own background thread, via asyncio.run() -- a
*different* event loop). See get_range()/force_refresh_range()'s own
docstrings for how each is made safe."""

from __future__ import annotations

from datetime import datetime, timezone

from app.backend.football_data_client import NormalizedMatch

TTL_FLOOR_SECONDS = 60.0
TTL_CEILING_SECONDS = 4 * 3600.0
TTL_DEFAULT_SECONDS = 15 * 60.0
_LIVE_STATUSES = frozenset({"LIVE", "IN_PLAY", "PAUSED"})
_UPCOMING_STATUSES = frozenset({"SCHEDULED", "TIMED"})


def compute_ttl_seconds(matches: list[NormalizedMatch], now: datetime) -> float:
    """How long a single day's cached fixture/result list stays valid,
    computed from the matches it actually contains:

    - Empty day (nothing fetched for it): the plain default -- nothing to
      compute a kickoff-driven expiry from, but not worth the ceiling
      either in case a fixture gets added before the reconciliation job's
      own next cycle notices.
    - Any live/in-progress match (LIVE/IN_PLAY/PAUSED): the floor --
      scores update continuously, this cache layer must not sit on stale
      data for long.
    - A not-yet-kicked-off SCHEDULED/TIMED match expires at its own
      kickoff time (when it may flip to LIVE) -- the EARLIEST such
      kickoff among the day's matches, clamped to [floor, ceiling]. A
      SCHEDULED match whose kickoff has already passed (a stale fetch
      snapshot, mirrors eod_batch.has_kicked_off's own "status reflects
      whenever the fixture was fetched, not right now" reasoning) is
      treated the same as FINISHED, not as "about to kick off".
    - Everything else (only FINISHED matches, or an unparseable utc_date)
      is effectively immutable from this cache's point of view -- the
      ceiling. Postponement/reschedule is NOT covered by this TTL at all;
      that is register_fixture_reconciliation_job's job (a periodic live
      re-check independent of this cache's own expiry), not this
      function's."""
    if not matches:
        return TTL_DEFAULT_SECONDS

    if any(m.status in _LIVE_STATUSES for m in matches):
        return TTL_FLOOR_SECONDS

    upcoming_kickoffs: list[datetime] = []
    for m in matches:
        if m.status not in _UPCOMING_STATUSES:
            continue
        try:
            kickoff = datetime.fromisoformat(m.utc_date.replace("Z", "+00:00"))
        except ValueError:
            continue
        if kickoff > now:
            upcoming_kickoffs.append(kickoff)

    if not upcoming_kickoffs:
        return TTL_CEILING_SECONDS

    seconds_to_earliest = (min(upcoming_kickoffs) - now).total_seconds()
    return max(TTL_FLOOR_SECONDS, min(TTL_CEILING_SECONDS, seconds_to_earliest))
