"""Shared week-bucketing for the /ops dashboard's trend endpoints
(odds-availability-trend, feature-health-trend, performance-trend). Weeks run
Thursday-to-Wednesday (direct user request, 2026-11-03) -- the newest bucket
is always "today"'s own week, necessarily partial until the following
Wednesday; shown as-is, same as any other in-progress aggregation elsewhere in
this codebase, not specially called out."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta


def thursday_week_start(date_str: str) -> str:
    """ISO date (YYYY-MM-DD) -> the ISO date of the Thursday that starts its
    week bucket. date.weekday() is Mon=0..Sun=6, so Thursday=3; the modulo
    handles every day of the week in one formula, including Thursday itself
    (maps to 0 days back)."""
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    days_since_thursday = (d.weekday() - 3) % 7
    return (d - timedelta(days=days_since_thursday)).isoformat()


def recent_week_starts(today: date, weeks: int) -> list[str]:
    """Oldest-to-newest list of `weeks` Thursday week_start ISO dates ending
    at `today`'s own (possibly partial) week. `weeks` must be >= 1."""
    current_week_start = datetime.strptime(thursday_week_start(today.isoformat()), "%Y-%m-%d").date()
    return [(current_week_start - timedelta(weeks=n)).isoformat() for n in range(weeks - 1, -1, -1)]


# Tunable, first-pass thresholds -- see design doc section 6. A feature needs
# at least this many appearances in BOTH the baseline and recent half to be
# eligible for flagging at all (guards against a 1-2-sample feature looking
# like it "drifted" when it's just noise from rarely surfacing).
MIN_APPEARANCES_PER_HALF = 3
# Flag drift when the recent-half pooled mean is at least this many times
# the baseline pooled mean (or its reciprocal, for a drop) -- a magnitude-
# ratio heuristic, not a statistical test; revisit once this has real
# production weeks to tune against.
DRIFT_RATIO_THRESHOLD = 2.0
# Flag an elevated null rate when the recent-half null rate exceeds the
# baseline by at least this many percentage points.
NULL_RATE_DELTA_THRESHOLD = 0.05


@dataclass(frozen=True)
class FeatureWeekStats:
    appearances: int
    mean: float | None
    null_rate: float


def aggregate_feature_weeks(
    entries: list[dict], week_starts: list[str]
) -> dict[str, dict[str, FeatureWeekStats]]:
    """`entries`: one dict per cached recommendation with `top_features`
    (already `forecast_payload["explainability"]["top_features"]`) already
    extracted, plus the entry's own pre-computed `week_start` (callers bucket
    via thursday_week_start() before calling this -- kept as a plain
    precondition rather than re-bucketing here, since main.py's endpoint
    already needs the raw date for other filtering too). Returns
    {feature_name: {week_start: FeatureWeekStats}}, only for weeks actually
    present in `week_starts` -- a feature with zero appearances in a given
    week simply has no key for that week (not a zero-appearance stats
    object), so callers can tell "never surfaced that week" from "surfaced
    and was null" by key presence."""
    # name -> week_start -> running totals
    totals: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {
        "appearances": 0, "null_count": 0, "value_sum": 0.0, "value_count": 0,
    }))
    week_set = set(week_starts)
    for entry in entries:
        week_start = entry["week_start"]
        if week_start not in week_set:
            continue
        for feature in entry.get("top_features") or []:
            name = feature.get("name")
            if not name:
                continue
            bucket = totals[name][week_start]
            bucket["appearances"] += 1
            value = feature.get("value")
            if value is None:
                bucket["null_count"] += 1
            else:
                bucket["value_sum"] += float(value)
                bucket["value_count"] += 1

    result: dict[str, dict[str, FeatureWeekStats]] = {}
    for name, by_week in totals.items():
        result[name] = {}
        for week_start, bucket in by_week.items():
            mean = bucket["value_sum"] / bucket["value_count"] if bucket["value_count"] else None
            null_rate = bucket["null_count"] / bucket["appearances"] if bucket["appearances"] else 0.0
            result[name][week_start] = FeatureWeekStats(
                appearances=bucket["appearances"], mean=mean, null_rate=round(null_rate, 6),
            )
    return result


def _pooled(stats: list[FeatureWeekStats]) -> tuple[int, float | None, float]:
    """Pools several weeks' FeatureWeekStats into one (appearances, mean,
    null_rate) triple -- sums counts rather than averaging per-week means, so
    weeks with more samples correctly carry more weight.

    Reconstructs each week's null_count as round(s.null_rate * s.appearances).
    This round-trips exactly against aggregate_feature_weeks's own
    round(null_count/appearances, 6): verified by brute force (every
    appearances/null_count pair up to appearances=2000) that
    round(round(null_count/appearances, 6) * appearances) == null_count
    always -- the rounding error from 6 decimal places (<= 5e-7) only
    accumulates past the 0.5 needed to flip the final round() once
    appearances exceeds ~1,000,000, far beyond any realistic weekly
    recommendation count. No need for FeatureWeekStats to carry raw counts
    instead of null_rate."""
    total_appearances = sum(s.appearances for s in stats)
    total_null = sum(round(s.null_rate * s.appearances) for s in stats)
    value_count = sum(s.appearances - round(s.null_rate * s.appearances) for s in stats)
    value_sum = sum((s.mean or 0.0) * (s.appearances - round(s.null_rate * s.appearances)) for s in stats)
    mean = value_sum / value_count if value_count else None
    null_rate = total_null / total_appearances if total_appearances else 0.0
    return total_appearances, mean, null_rate


def flag_drifting_features(
    weekly: dict[str, dict[str, FeatureWeekStats]], week_starts: list[str]
) -> list[dict]:
    """Splits `week_starts` into an earlier baseline half and a later recent
    half (floor split -- e.g. 8 weeks -> 4/4, 5 weeks -> 2/3), pools each
    feature's stats within each half, and flags features whose recent pooled
    mean/null_rate deviates from baseline beyond the module's threshold
    constants. Only features with >= MIN_APPEARANCES_PER_HALF total
    appearances in BOTH halves are eligible -- see that constant's own
    comment for why. Returns a list sorted by severity (largest deviation
    first), each entry: {"name", "category", "flags": [...], "baseline_mean",
    "recent_mean", "baseline_null_rate", "recent_null_rate", "magnitude",
    "weekly": [{"week_start", "mean", "null_rate"}, ...]}."""
    split = len(week_starts) // 2
    baseline_weeks, recent_weeks = week_starts[:split], week_starts[split:]
    flagged: list[dict] = []
    for name, by_week in weekly.items():
        baseline_stats = [by_week[w] for w in baseline_weeks if w in by_week]
        recent_stats = [by_week[w] for w in recent_weeks if w in by_week]
        baseline_n, baseline_mean, baseline_null = _pooled(baseline_stats)
        recent_n, recent_mean, recent_null = _pooled(recent_stats)
        if baseline_n < MIN_APPEARANCES_PER_HALF or recent_n < MIN_APPEARANCES_PER_HALF:
            continue

        flags: list[str] = []
        magnitude = 0.0
        if baseline_mean is not None and recent_mean is not None:
            if baseline_mean == 0:
                if abs(recent_mean) > 0.01:
                    flags.append("drift")
                    magnitude = max(magnitude, abs(recent_mean))
            else:
                ratio = abs(recent_mean / baseline_mean)
                if ratio >= DRIFT_RATIO_THRESHOLD or ratio <= (1.0 / DRIFT_RATIO_THRESHOLD):
                    flags.append("drift")
                    magnitude = max(magnitude, max(ratio, 1.0 / ratio) if ratio else 0.0)
        if (recent_null - baseline_null) >= NULL_RATE_DELTA_THRESHOLD:
            flags.append("nullrate")
            magnitude = max(magnitude, (recent_null - baseline_null) * 10)  # scaled to be roughly comparable to a drift ratio

        if not flags:
            continue
        flagged.append({
            "name": name,
            "category": _feature_category(name),
            "flags": flags,
            "baseline_mean": round(baseline_mean, 6) if baseline_mean is not None else None,
            "recent_mean": round(recent_mean, 6) if recent_mean is not None else None,
            "baseline_null_rate": round(baseline_null, 6),
            "recent_null_rate": round(recent_null, 6),
            "magnitude": round(magnitude, 3),
            "weekly": [
                {"week_start": w, "mean": by_week[w].mean, "null_rate": by_week[w].null_rate}
                for w in week_starts if w in by_week
            ],
        })
    flagged.sort(key=lambda f: f["magnitude"], reverse=True)
    return flagged


_CATEGORY_PREFIXES = [
    "OFF_", "DEF_", "MKT_", "CTX_", "DIS_", "SQUAD_", "H2H_", "LUCK_", "EFFICIENCY_",
    "OPP_ADJ_", "DC_", "XOC", "DEF_ANCHOR",
]


def _feature_category(name: str) -> str:
    """Best-effort category tag from the feature name's own existing
    namespace-prefix convention (already visible across this codebase's real
    feature names, e.g. OFF_AWAY_SHOT_ACCURACY_R5, MKT_POISSON_BTTS_PROB) --
    a simple prefix lookup, not a new taxonomy to maintain. Falls back to
    "other" for a name matching none of these."""
    for prefix in _CATEGORY_PREFIXES:
        if name.startswith(prefix):
            return prefix.rstrip("_").lower()
    return "other"
