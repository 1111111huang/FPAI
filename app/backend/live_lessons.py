"""W177: turns the app's own daily finished-match recommendations into
lesson candidates for src/agent/lessons.py's existing human-review pipeline
(A33/A39-A47) -- the same agent_lessons table agent-train writes to, sourced
here from live recommendation_outcomes (W167) instead of a backtest corpus.

Internal use only: every candidate this writes lands as status='pending',
exactly like a training-sourced one -- it only reaches live serving once a
human runs `agent-lessons approve <id> --scope ...` (main.py), unchanged.

Kept out of recommendation_stats.py (needs real DB I/O beyond pure
aggregation -- same separation agent_performance_dashboard.py already
established for its own DB-touching enrichment)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import duckdb

from app.backend.football_data_client import FootballDataClient
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import (
    RecommendationOutcome,
    RecommendationOutcomeStore,
    resolve_pending_recommendations,
)
from src.agent.backtest import BacktestRecord, load_match_stats
from src.agent.lessons import (
    approve_lesson,
    find_conflicting_rule,
    generate_match_reflection,
    generate_rule_from_lesson,
    insert_lesson_candidate,
    judge_lesson_candidate,
    list_pending_by_source,
    load_approved_lessons,
    reject_lesson,
)
from src.agent.market_resolution import build_actual_outcome
from src.agent.schema import reported_teams
from src.ingestion.common.team_mapping import TeamNameMapper
from src.logic.competition_registry import get_competition_definition
from src.utils.db_manager import DuckDBManager
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

LIVE_SOURCE_NOTE = (
    "Live-sourced batch: reflects only the market actually recommended per "
    "match, not every market the agent evaluated."
)


@dataclass
class PreparedLessonBatch:
    """Output of prepare_lesson_batches() -- everything needed to write one
    agent_lessons row, computed with NO DuckDB connection HELD across its
    slow work (an LLM reflection call per match) -- only a brief, already-
    closed read for match_stats grounding happens first (A109). Kept
    separate from the actual write (commit_lesson_batches()) specifically
    so the DuckDB exclusive file lock is never held across that LLM work --
    found during Task 4's code-quality review, which traced the lock being
    held across both network calls and LLM work in the original
    single-phase generate_daily_lessons(); resolve_pending_recommendations'
    own football-data.org lookups moved out of this phase entirely
    (2026-09-12, now the caller's own daily-cadence responsibility, see
    register_lessons_job)."""
    competition_id: str
    tier: str
    lesson_text: str
    match_ids: str
    outcome_ids: list[int]


def _to_lesson_record(outcome: RecommendationOutcome, cache: RecommendationCache) -> BacktestRecord:
    """Enrichment-complete adapter -- unlike recommendation_stats.py's own
    minimal _to_backtest_records (which only needs market_results for the
    Kelly simulation), generate_match_reflection (and its generate_lesson_text
    fallback) also read home_team/away_team/recommendation.{overall,confidence,
    explanation,limitations}/actual.result. A cache miss or a pre-migration
    outcome (competition_id/home_goals/away_goals all NULL, resolved before
    W175) degrades to blank fields rather than raising -- the record still
    joins its batch, just with less color, matching the dashboard's own
    degrade-one-row discipline (agent_performance_dashboard.py's
    _enrich_bet)."""
    entry = cache.get_latest_any_config(outcome.match_id, outcome.date)
    recommendation = entry.recommendation if entry is not None else {}
    teams = reported_teams(recommendation.get("match") or {}) if entry is not None else None
    home_team, away_team = teams if teams is not None else ("", "")
    actual = (
        build_actual_outcome(outcome.home_goals, outcome.away_goals)
        if outcome.home_goals is not None and outcome.away_goals is not None
        else {}
    )
    return BacktestRecord(
        match_id=outcome.match_id,
        home_team=home_team,
        away_team=away_team,
        date=outcome.date,
        league=outcome.competition_id or "",
        recommendation=recommendation,
        actual=actual,
        market_results=[{
            "market": outcome.market,
            "selection": outcome.selection,
            "correct": outcome.correct,
        }],
    )


def _match_stats_lookup(
    duckdb_manager: DuckDBManager | None, groups: dict[tuple[str, str], list[RecommendationOutcome]],
) -> dict[tuple[str, str], Any]:
    """2026-09-12: one brief upfront read (closed before any LLM work
    starts below, same discipline this module's own docstrings already
    established) for match_stats grounding -- the box-score columns
    (shots, cards) A109's generate_match_reflection uses for train, never
    available to live before now since FootballDataClient's own API
    genuinely has no such field (confirmed live against the real API; not
    an extraction gap like A69/A73's). raw_matches DOES eventually get real
    shots/cards, but only via the weekly raw_matches refresh
    (schedule-refresh) -- register_lessons_job now runs lesson generation
    weekly too (not daily) specifically so that refresh has usually already
    caught up with last week's matches by the time this runs. Returns one
    DataFrame per (competition_id, date) group, scoped to that league+date
    (raw_matches has no live match_id to join on -- football-data.org's own
    ids and raw_matches' content-hashed ones are different ID systems
    entirely); empty for a group with no raw_matches row yet, or if
    duckdb_manager is None (callers that don't have one, e.g. fast unit
    tests uninterested in match_stats) or raw_matches doesn't exist at all
    in that DB."""
    # The file itself not existing yet (e.g. this is the very first run,
    # before commit_lesson_batches' create_lessons_tables has ever created
    # it) is routine, not transient -- checked explicitly rather than
    # falling into DuckDBManager.connection()'s own retry-on-IOException
    # loop, which would otherwise burn several real seconds of sleep-and-
    # retry on every such call for no benefit (the file still won't exist
    # on attempt 6 either).
    if duckdb_manager is None or not duckdb_manager.db_path.exists():
        return {}
    lookup: dict[tuple[str, str], Any] = {}
    try:
        with duckdb_manager.connection(read_only=True) as conn:
            for competition_id, date in groups:
                lookup[(competition_id, date)] = conn.execute(
                    'SELECT home_team, away_team, hs, "as", hst, ast, hy, ay, hr, ar '
                    "FROM raw_matches WHERE league = ? AND date = ?",
                    [competition_id, date],
                ).fetchdf()
    except Exception:
        LOGGER.warning("live_lessons: match_stats lookup against raw_matches failed -- proceeding without it.", exc_info=True)
        return {}
    return lookup


def _match_stats_for_record(record: BacktestRecord, raw_today: Any, mapper: TeamNameMapper) -> dict[str, Any] | None:
    """Join one live match to its raw_matches row by (mapped team names),
    reusing the exact fuzzy-resolution TeamNameMapper src/ingestion/fotmob/
    merge.py's resolve_match_ids already established for this identical
    problem (a source's own team-name spelling vs. raw_matches' canonical
    one) -- scoped to one (league, date)'s candidate pool the same way,
    not the whole season's, since that's already this group's scope."""
    if raw_today is None or raw_today.empty or not record.home_team or not record.away_team:
        return None
    team_pool = set(raw_today["home_team"]).union(raw_today["away_team"])
    mapped_home = mapper.map_team(record.home_team, team_pool)
    mapped_away = mapper.map_team(record.away_team, team_pool)
    matched = raw_today[(raw_today["home_team"] == mapped_home) & (raw_today["away_team"] == mapped_away)]
    return load_match_stats(matched.iloc[0]) if not matched.empty else None


def prepare_lesson_batches(
    cache: RecommendationCache,
    store: RecommendationOutcomeStore,
    duckdb_manager: DuckDBManager | None = None,
    llm_invoke: Callable[[str], str] | None = None,
) -> list[PreparedLessonBatch]:
    """All the LLM-bound work (grouping, enrichment, reflection generation)
    -- deliberately does NOT hold DuckDB open during any of it, so it can
    run for as long as it needs (an LLM call per match) without holding
    data/fpai_core.db's exclusive file lock. Call commit_lesson_batches()
    with the result to actually write.

    2026-09-12: no longer resolves outcomes itself (that moved to the
    caller -- register_lessons_job's daily job, kept daily so anything
    reading recommendation_outcomes stays fresh even though lesson
    generation itself moved to weekly, see that function's docstring) --
    this only ever batches whatever's already unbatched.

    duckdb_manager (A109): optional -- when given, used for one brief
    match_stats lookup against raw_matches before any LLM work starts (see
    _match_stats_lookup); None skips it entirely (match_stats stays None
    for every match), same degrade-gracefully contract generate_match_
    reflection already has for a record with no match_stats at all.

    llm_invoke=None makes every per-match reflection fall back to the
    deterministic template (generate_match_reflection's own contract) --
    used by callers that can't or don't want to pay for the LLM call (e.g.
    a fast unit test), not a distinct product mode."""
    pending = store.list_unbatched_for_lessons()
    groups: dict[tuple[str, str], list[RecommendationOutcome]] = defaultdict(list)
    for outcome in pending:
        if outcome.competition_id is None:
            LOGGER.warning(
                "live_lessons: skipping outcome match_id=%s (date=%s) -- no verified "
                "competition_id (likely resolved before W175's migration).",
                outcome.match_id, outcome.date,
            )
            continue
        groups[(outcome.competition_id, outcome.date)].append(outcome)

    raw_by_group = _match_stats_lookup(duckdb_manager, groups)
    mapper = TeamNameMapper()

    prepared: list[PreparedLessonBatch] = []
    for (competition_id, date), group in groups.items():
        try:
            tier = get_competition_definition(competition_id).tier
        except (ValueError, FileNotFoundError):
            # ValueError: unknown competition_id. FileNotFoundError:
            # config/competitions.yaml missing -- same two exceptions
            # src/agent/tools.py's _resolve_competition_impl already guards
            # against calling this same function. Without both, a missing
            # registry file would crash the whole run instead of just
            # skipping this one batch.
            LOGGER.warning("live_lessons: skipping batch for unrecognized competition_id=%s.", competition_id)
            continue

        raw_today = raw_by_group.get((competition_id, date))
        records = [_to_lesson_record(outcome, cache) for outcome in group]
        reflections = []
        for outcome, record in zip(group, records):
            entry = cache.get_latest_any_config(outcome.match_id, outcome.date)
            reasoning_trace = entry.reasoning_trace if entry is not None else None
            match_stats = _match_stats_for_record(record, raw_today, mapper)
            reflections.append(generate_match_reflection(record, reasoning_trace, llm_invoke, match_stats))
        lesson_text = f"{LIVE_SOURCE_NOTE}\n\n" + "\n\n".join(reflections)

        prepared.append(PreparedLessonBatch(
            competition_id=competition_id,
            tier=tier,
            lesson_text=lesson_text,
            match_ids=",".join(outcome.match_id for outcome in group),
            outcome_ids=[outcome.id for outcome in group],
        ))
    return prepared


def commit_lesson_batches(
    duckdb_conn: duckdb.DuckDBPyConnection,
    store: RecommendationOutcomeStore,
    batches: list[PreparedLessonBatch],
) -> list[int]:
    """The brief write phase -- no network or LLM calls happen here, only
    DuckDB inserts and SQLite updates. Call with an already-open
    duckdb_conn; hold it for only as long as this function runs."""
    lesson_ids: list[int] = []
    for batch in batches:
        lesson_id = insert_lesson_candidate(
            duckdb_conn, batch.lesson_text, batch.competition_id, batch.tier, batch.match_ids, source="live",
        )
        store.mark_lesson_batched(batch.outcome_ids)
        lesson_ids.append(lesson_id)
    return lesson_ids


def generate_daily_lessons(
    cache: RecommendationCache,
    store: RecommendationOutcomeStore,
    client: FootballDataClient,
    duckdb_conn: duckdb.DuckDBPyConnection,
    sweden_client: object | None = None,
    llm_invoke: Callable[[str], str] | None = None,
    duckdb_manager: DuckDBManager | None = None,
) -> list[int]:
    """Thin orchestrator combining resolve_pending_recommendations() +
    prepare_lesson_batches() + commit_lesson_batches() -- kept for
    direct/test convenience where lock-hold-duration doesn't matter (e.g.
    an in-memory DuckDB connection in a test, or a one-off manual sanity
    check). The real jobs (scheduler_wiring.py's register_lessons_job) call
    resolution daily and generation+commit weekly instead, each opening its
    own DuckDB connection only around its own brief step -- see that
    function's own docstring.

    duckdb_manager (A109): optional, threaded straight through to
    prepare_lesson_batches for its match_stats lookup -- None (the default,
    every pre-existing caller) skips that lookup entirely, unaffected."""
    resolve_pending_recommendations(cache, store, client, sweden_client)
    batches = prepare_lesson_batches(cache, store, duckdb_manager, llm_invoke)
    return commit_lesson_batches(duckdb_conn, store, batches)


def _format_group_lesson_text(candidates: list[dict[str, Any]]) -> str:
    """Joins one (competition_id, tier) group's individual daily
    candidates into a single combined lesson_text, in date order, each
    section labeled with its own date and source match ids -- so a week's
    worth of daily reports reads as one document instead of a single day's,
    giving judge_lesson_candidate real sample size to apply its existing
    "reject if noise, approve if clearly systematic" test to, rather than
    the n=1 it always got when judged one candidate at a time."""
    ordered = sorted(candidates, key=lambda c: c["created_at"])
    sections = [
        f"--- {c['created_at'].date()} (match_ids: {c['source_match_id']}) ---\n{c['lesson_text']}"
        for c in ordered
    ]
    return "\n\n".join(sections)


def auto_judge_live_lessons(
    duckdb_manager: DuckDBManager,
    llm_invoke: Callable[[str], str] | None,
) -> list[dict[str, Any]]:
    """2026-08-27 (W183-W185, superseding the 2026-08-26 per-candidate
    version): judges every still-pending source='live' candidate for a
    (competition_id, tier) *together*, once a week, instead of one
    candidate at a time right after it's created. A single day's batch is
    typically n=1 match -- judge_lesson_candidate's own "reject on a thin
    sample" prompt could never clear that bar even when the exact same
    failure mode recurred for weeks, since it never saw more than one
    day's evidence. Grouping several days' already-computed lesson_text
    into one combined document (_format_group_lesson_text) gives it real
    sample size instead, with zero changes to judge_lesson_candidate,
    generate_rule_from_lesson, or find_conflicting_rule themselves.

    Never touches source='train' (or pre-migration, source IS NULL) rows --
    list_pending_by_source(source='live') structurally excludes both.

    llm_invoke=None means the weekly job's own LLM client failed to build --
    a no-op; every pending candidate simply waits for the following week.

    Three phases, same discipline as prepare_lesson_batches/
    commit_lesson_batches: a brief read, all LLM work with no DuckDB
    connection open, then a brief write. Both the per-group conflict check
    and the per-row write are isolated in their own try/except -- a
    failure in either only defers/skips what it was working on, never
    discarding another group's (or another row's) already-computed
    decision. Returns a list of dicts, one per underlying row (a group of
    N candidates contributes N entries, all sharing the same decision) --
    {id, action, reasoning}."""
    if llm_invoke is None:
        return []

    with duckdb_manager.connection(read_only=True) as conn:
        pending = list_pending_by_source(conn, source="live")

    groups: dict[tuple[str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in pending:
        groups[(candidate["competition_id"], candidate["tier"])].append(candidate)

    results: list[dict[str, Any]] = []
    for (competition_id, tier), candidates in groups.items():
        combined_text = _format_group_lesson_text(candidates)
        row_ids = [candidate["id"] for candidate in candidates]

        decision = judge_lesson_candidate(combined_text, competition_id, tier, llm_invoke)
        action = "reject" if not decision.approve else "approve"
        scope = decision.scope
        rule_text: str | None = None
        reasoning = decision.reasoning

        if decision.approve:
            rule_text = generate_rule_from_lesson(combined_text, llm_invoke)
            if rule_text is None:
                action = "defer"
                reasoning = f"{decision.reasoning} (rule distillation failed -- left pending for retry)"
            else:
                # find_conflicting_rule deliberately doesn't catch its own
                # exceptions (see its docstring) -- callers decide fail-open
                # vs fail-closed. Unlike main.py's run_agent_lessons_approve
                # (a human is right there, so it fails open), there's no
                # human backstop on this autonomous path -- approving a rule
                # whose conflict-check silently never ran is worse than
                # deferring it, so this fails CLOSED to defer.
                try:
                    with duckdb_manager.connection(read_only=True) as conn:
                        existing_rules = load_approved_lessons(conn, competition_id, tier)
                    conflict = find_conflicting_rule(rule_text, existing_rules, llm_invoke)
                except Exception as exc:
                    action = "defer"
                    reasoning = f"{decision.reasoning} (conflict check failed: {exc!r} -- left pending for retry)"
                    LOGGER.warning(
                        "live_lessons: conflict check failed for lesson group competition_id=%s tier=%s.",
                        competition_id, tier, exc_info=True,
                    )
                else:
                    if conflict is not None:
                        action = "defer"
                        reasoning = f"Would approve, but a conflict was found: {conflict}"

        for row_id in row_ids:
            results.append({
                "id": row_id, "action": action, "scope": scope,
                "rule_text": rule_text, "reasoning": reasoning,
            })

    with duckdb_manager.connection() as conn:
        for result in results:
            try:
                # Re-check status right before writing -- a human can run
                # `agent-lessons approve/reject <id>` on this exact row (the
                # CLI applies to any row id, no source filter) at any point
                # during this function's judge/distill/conflict-check phase
                # above, which holds no DuckDB connection open and can run
                # for a while (LLM calls). Without this, our write here would
                # silently clobber that human decision with a stale one
                # computed before it happened. Applied uniformly to
                # approve/reject/defer -- a defer that hits a changed-status
                # row just skips harmlessly too, same as the others.
                current_status = conn.execute(
                    "SELECT status FROM agent_lessons WHERE id = ?", [result["id"]]
                ).fetchone()
                if current_status is None or current_status[0] != "pending":
                    LOGGER.warning(
                        "live_lessons: skipping auto-judge write for lesson id=%s -- "
                        "already reviewed (status=%s) since this run started.",
                        result["id"], current_status[0] if current_status else "missing",
                    )
                    continue
                if result["action"] == "approve":
                    approve_lesson(conn, result["id"], result["scope"], reviewer="agent-auto", rule_text=result["rule_text"])
                elif result["action"] == "reject":
                    reject_lesson(conn, result["id"], reviewer="agent-auto")
                # "defer" -- leave status as-is, just record the reasoning below.
                conn.execute(
                    "UPDATE agent_lessons SET auto_decision_reasoning = ? WHERE id = ?",
                    [result["reasoning"], result["id"]],
                )
            except Exception:
                LOGGER.warning(
                    "live_lessons: failed to write auto-judge decision for lesson id=%s -- left as-is for retry.",
                    result["id"], exc_info=True,
                )

    return results
