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
from typing import Callable

import duckdb

from app.backend.football_data_client import FootballDataClient
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import (
    RecommendationOutcome,
    RecommendationOutcomeStore,
    resolve_pending_recommendations,
)
from src.agent.backtest import BacktestRecord
from src.agent.lessons import generate_batch_lesson_text, generate_batch_reflection, insert_lesson_candidate
from src.agent.market_resolution import build_actual_outcome
from src.agent.schema import reported_teams
from src.logic.competition_registry import get_competition_definition
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

LIVE_SOURCE_NOTE = (
    "Live-sourced batch: reflects only the market actually recommended per "
    "match, not every market the agent evaluated."
)


def _to_lesson_record(outcome: RecommendationOutcome, cache: RecommendationCache) -> BacktestRecord:
    """Enrichment-complete adapter -- unlike recommendation_stats.py's own
    minimal _to_backtest_records (which only needs market_results for the
    Kelly simulation), generate_batch_lesson_text/generate_batch_reflection
    also read home_team/away_team/recommendation.{overall,confidence,
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


def generate_daily_lessons(
    cache: RecommendationCache,
    store: RecommendationOutcomeStore,
    client: FootballDataClient,
    duckdb_conn: duckdb.DuckDBPyConnection,
    sweden_client: object | None = None,
    llm_invoke: Callable[[str], str] | None = None,
) -> list[int]:
    """The daily job body (wired by scheduler_wiring.py's
    register_lessons_job). Resolves pending outcomes first (W167) so this
    runs as one unattended pipeline, then batches whatever hasn't yet been
    folded into a lesson by (competition_id, date) -- one candidate per
    league per day, per direct user decision.

    llm_invoke=None skips generate_batch_reflection entirely (a stats-only
    candidate) -- used by callers that can't or don't want to pay for the
    LLM call (e.g. a fast unit test), not a distinct product mode."""
    resolve_pending_recommendations(cache, store, client, sweden_client)

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

    lesson_ids: list[int] = []
    for (competition_id, _date), group in groups.items():
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

        records = [_to_lesson_record(outcome, cache) for outcome in group]
        stats_text = generate_batch_lesson_text(records)
        lesson_text = f"{LIVE_SOURCE_NOTE} {stats_text}"
        if llm_invoke is not None:
            reflection = generate_batch_reflection(records, stats_text, llm_invoke)
            if reflection:
                lesson_text = f"{lesson_text}\n\nReflection: {reflection}"

        match_ids = ",".join(outcome.match_id for outcome in group)
        lesson_id = insert_lesson_candidate(duckdb_conn, lesson_text, competition_id, tier, match_ids)
        store.mark_lesson_batched([outcome.id for outcome in group])
        lesson_ids.append(lesson_id)

    return lesson_ids
