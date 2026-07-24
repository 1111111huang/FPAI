"""A33: DuckDB persistence for critic/train mode -- reviewed, competition/
tier-scoped lesson candidates and per-run evidence telemetry. See
docs/superpowers/specs/2026-07-22-agent-phase11-design.md (A33 section,
revised 2026-07-24 for competition scoping).

load_approved_lessons() is the ONLY function this module exposes that the
live agent path (src/agent/pipeline.py's lessons_node) imports -- its SQL
hardcodes status='approved' and never touches an outcome-bearing table, so
live mode is structurally unable to read match outcomes or pending/rejected
lessons, not just conventionally forbidden from it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import duckdb

_VALID_SCOPES = ("competition", "tier")


def create_lessons_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create agent_lessons and agent_telemetry if they don't already exist."""
    conn.execute("CREATE SEQUENCE IF NOT EXISTS agent_lessons_id_seq START 1")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_lessons (
            id INTEGER PRIMARY KEY DEFAULT nextval('agent_lessons_id_seq'),
            lesson_text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            competition_id TEXT,
            tier TEXT NOT NULL,
            scope TEXT,
            source_match_id TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            reviewed_at TIMESTAMP,
            reviewer TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_telemetry (
            match_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            competition_resolution TEXT,
            research_evidence TEXT,
            forecast_payload TEXT,
            recommendation TEXT,
            created_at TIMESTAMP NOT NULL,
            PRIMARY KEY (match_id, run_id)
        )
        """
    )


def insert_lesson_candidate(
    conn: duckdb.DuckDBPyConnection,
    lesson_text: str,
    competition_id: str | None,
    tier: str,
    source_match_id: str,
) -> int:
    """Insert a pending, unscoped lesson candidate. Returns its id."""
    row = conn.execute(
        """
        INSERT INTO agent_lessons (lesson_text, status, competition_id, tier, source_match_id, created_at)
        VALUES (?, 'pending', ?, ?, ?, ?)
        RETURNING id
        """,
        [lesson_text, competition_id, tier, source_match_id, datetime.now(timezone.utc)],
    ).fetchone()
    return int(row[0])


def insert_telemetry(
    conn: duckdb.DuckDBPyConnection,
    match_id: str,
    run_id: str,
    competition_resolution: dict[str, Any] | None,
    research_evidence: dict[str, Any] | None,
    forecast_payload: dict[str, Any] | None,
    recommendation: dict[str, Any] | None,
) -> None:
    conn.execute(
        """
        INSERT INTO agent_telemetry
            (match_id, run_id, competition_resolution, research_evidence, forecast_payload, recommendation, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            match_id,
            run_id,
            json.dumps(competition_resolution),
            json.dumps(research_evidence),
            json.dumps(forecast_payload),
            json.dumps(recommendation),
            datetime.now(timezone.utc),
        ],
    )


def _require_lesson_exists(conn: duckdb.DuckDBPyConnection, lesson_id: int) -> None:
    count = conn.execute("SELECT COUNT(*) FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()[0]
    if not count:
        raise ValueError(f"No lesson with id={lesson_id}")


def approve_lesson(conn: duckdb.DuckDBPyConnection, lesson_id: int, scope: str, reviewer: str) -> None:
    """Approve a lesson, requiring the reviewer to pick a scope explicitly.

    scope='competition' pins the lesson to its recorded competition_id;
    scope='tier' widens it to every match resolving to its recorded tier.
    """
    if scope not in _VALID_SCOPES:
        raise ValueError(f"scope must be one of {_VALID_SCOPES}, got {scope!r}")
    _require_lesson_exists(conn, lesson_id)
    conn.execute(
        "UPDATE agent_lessons SET status = 'approved', scope = ?, reviewed_at = ?, reviewer = ? WHERE id = ?",
        [scope, datetime.now(timezone.utc), reviewer, lesson_id],
    )


def reject_lesson(conn: duckdb.DuckDBPyConnection, lesson_id: int, reviewer: str) -> None:
    _require_lesson_exists(conn, lesson_id)
    conn.execute(
        "UPDATE agent_lessons SET status = 'rejected', reviewed_at = ?, reviewer = ? WHERE id = ?",
        [datetime.now(timezone.utc), reviewer, lesson_id],
    )


def load_approved_lessons(conn: duckdb.DuckDBPyConnection, competition_id: str | None, tier: str) -> list[str]:
    """Approved lesson text for one match's competition_id/tier. Tolerates a
    missing agent_lessons table (e.g. agent-train has never been run yet) by
    returning no lessons rather than raising -- live recommendation runs must
    never fail just because train mode hasn't produced anything yet."""
    try:
        rows = conn.execute(
            """
            SELECT lesson_text FROM agent_lessons
            WHERE status = 'approved'
              AND ((scope = 'competition' AND competition_id = ?)
                OR (scope = 'tier' AND tier = ?))
            ORDER BY created_at
            """,
            [competition_id, tier],
        ).fetchall()
    except duckdb.CatalogException:
        return []
    return [row[0] for row in rows]


def extract_competition_scope(full_state: dict[str, Any]) -> tuple[str | None, str]:
    """(competition_id, tier) from an AgentState-shaped dict's
    competition_resolution block, defaulting tier to general_purpose when
    absent -- mirrors resolve_competition_node's own leagueless-international
    default so a lesson from that path still records a real tier."""
    resolution = full_state.get("competition_resolution") or {}
    return resolution.get("competition"), resolution.get("tier") or "general_purpose"


def generate_lesson_text(record: Any) -> str:
    """Deterministic lesson-candidate template from a BacktestRecord-shaped
    object (duck-typed: .league, .recommendation, .market_results, .actual --
    see src/agent/backtest.py). Not an attempt at insightful NLG -- the
    reviewer judges usefulness at approval time; this just surfaces a
    structured summary of what happened for them to judge."""
    context_label = record.league or "an unlabeled competition"
    overall = record.recommendation.get("overall", "unknown")
    confidence = record.recommendation.get("confidence", "unknown")
    basis = record.recommendation.get("prediction_basis", "unknown")
    limitations = record.recommendation.get("limitations") or []

    market_lines = []
    for market in record.market_results:
        correct = market.get("correct")
        outcome = "correct" if correct is True else "incorrect" if correct is False else "unresolved"
        market_lines.append(f"{market.get('market')}={market.get('selection')} ({outcome})")
    markets_summary = "; ".join(market_lines) if market_lines else "no markets recommended"
    limitations_summary = "; ".join(limitations) if limitations else "none noted"

    return (
        f"WHEN evaluating {context_label} matches: a recommendation of '{overall}' "
        f"(confidence={confidence}, basis={basis}) had actual result={record.actual.get('result')}. "
        f"Markets: {markets_summary}. Limitations noted at the time: {limitations_summary}."
    )
