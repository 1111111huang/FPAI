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
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable

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
    # A44: rule_text added after agent_lessons already shipped (A33) and has
    # live data -- CREATE TABLE IF NOT EXISTS alone wouldn't add it to an
    # existing table, so this migrates in place every time. Deliberately
    # nullable: lesson_text stays the full audit trail (stats + reflection,
    # team names, dates -- what a reviewer judges), rule_text is the
    # generalized, prompt-ready sentence populated only once a lesson is
    # actually approved (see approve_lesson/generate_rule_from_lesson) --
    # load_approved_lessons() reads ONLY rule_text, never lesson_text, so the
    # live agent's prompt never sees match-specific noise.
    conn.execute("ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS rule_text TEXT")
    # 2026-08-26 (autonomous live-lesson judging, Phase 1): source
    # distinguishes an agent-train-sourced candidate ('train', the default
    # below and the only source that ever existed before this) from a
    # live-deployment-sourced one ('live', intended to be set by
    # live_lessons.py's commit_lesson_batches once it's updated to pass
    # source="live" -- not yet wired as of this commit, see Task 3).
    # auto_decision_reasoning is the audit trail
    # for an autonomous approve/reject decision -- a human reviewer's own
    # judgment call is visible in the CLI transcript; this is the
    # equivalent for a decision nobody watched happen. Both nullable/
    # additive -- a pre-migration row simply carries NULL for both, and
    # NULL is never matched by `source = 'live'` (SQL semantics), so
    # existing rows are structurally excluded from live-lesson-only
    # queries like list_pending_by_source() below, not just conventionally.
    conn.execute("ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS source TEXT")
    conn.execute("ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS auto_decision_reasoning TEXT")
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
    source: str = "train",
) -> int:
    """Insert a pending, unscoped lesson candidate. Returns its id.

    source: 'train' (default, preserves every pre-existing caller
    unchanged -- agent-train's own CLI path) or 'live' (intended for
    live_lessons.py's commit_lesson_batches to pass explicitly once it's
    updated to do so -- not yet wired as of this commit, see Task 3)."""
    row = conn.execute(
        """
        INSERT INTO agent_lessons (lesson_text, status, competition_id, tier, source_match_id, created_at, source)
        VALUES (?, 'pending', ?, ?, ?, ?, ?)
        RETURNING id
        """,
        [lesson_text, competition_id, tier, source_match_id, datetime.now(timezone.utc), source],
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


def approve_lesson(conn: duckdb.DuckDBPyConnection, lesson_id: int, scope: str, reviewer: str, rule_text: str) -> None:
    """Approve a lesson, requiring the reviewer to pick a scope explicitly.

    scope='competition' pins the lesson to its recorded competition_id;
    scope='tier' widens it to every match resolving to its recorded tier.

    rule_text (A44) is required, not optional -- an approved lesson with no
    rule_text would silently vanish from live use (load_approved_lessons
    only reads rule_text), which is a worse failure mode than forcing every
    approval to supply one. Callers (main.py's run_agent_lessons_approve)
    either take it from --rule or auto-distill via generate_rule_from_lesson
    before calling this."""
    if scope not in _VALID_SCOPES:
        raise ValueError(f"scope must be one of {_VALID_SCOPES}, got {scope!r}")
    if not rule_text or not rule_text.strip():
        raise ValueError("rule_text must be a non-empty string")
    _require_lesson_exists(conn, lesson_id)
    conn.execute(
        "UPDATE agent_lessons SET status = 'approved', scope = ?, rule_text = ?, reviewed_at = ?, reviewer = ? WHERE id = ?",
        [scope, rule_text.strip(), datetime.now(timezone.utc), reviewer, lesson_id],
    )


def generate_rule_from_lesson(lesson_text: str, llm_invoke: Callable[[str], str]) -> str | None:
    """A44: distill a reviewed lesson's raw stats+reflection text into one
    clean, generalized, prompt-ready rule -- stripped of team names, match
    dates, and batch statistics. Requested directly by the user (2026-07-28)
    after reviewing raw batch lessons: "every post match reflection needs to
    be summarized into clear rules... the agent should not read directly
    from the lesson's table but the summary." Mirrors the exact extraction
    task performed manually earlier in that same session.

    Same decoupled-from-langchain design as generate_batch_reflection: takes
    a plain str -> str callable, not an LLM object.

    Returns None on any failure (exception or blank response) -- callers
    must not silently approve a lesson with no usable rule_text; see
    run_agent_lessons_approve, which requires either a working distillation
    or an explicit --rule override before approval can proceed."""
    prompt = (
        "Below is a reviewed post-mortem analysis of a batch of an automated betting agent's historical "
        "recommendations, including deterministic statistics and a reflective narrative.\n\n"
        f"{lesson_text}\n\n"
        "Extract ONLY the single most programmatic, highly action-oriented rule from this analysis. "
        "Strip out all historical team names, match dates, and batch statistics. Output exactly one "
        "clean sentence starting with 'NEVER...' or 'IF...', suitable for inclusion as a standing "
        "instruction in the agent's system prompt. Output nothing else -- no preamble, no explanation, "
        "just the single sentence."
    )
    try:
        rule = llm_invoke(prompt)
    except Exception:
        return None
    rule = rule.strip()
    return rule or None


def reject_lesson(conn: duckdb.DuckDBPyConnection, lesson_id: int, reviewer: str) -> None:
    _require_lesson_exists(conn, lesson_id)
    conn.execute(
        "UPDATE agent_lessons SET status = 'rejected', reviewed_at = ?, reviewer = ? WHERE id = ?",
        [datetime.now(timezone.utc), reviewer, lesson_id],
    )


def load_approved_lessons(conn: duckdb.DuckDBPyConnection, competition_id: str | None, tier: str) -> list[str]:
    """Approved, distilled rule text (A44: rule_text, never the raw
    lesson_text -- the live agent's prompt should never see match-specific
    noise like team names/dates/stats, only the generalized rule a reviewer
    approved) for one match's competition_id/tier. Excludes any approved row
    with a NULL rule_text (shouldn't normally happen -- approve_lesson
    requires a non-empty rule_text -- but old rows approved before A44
    shipped have no rule_text, and this is the correct way for them to just
    not apply live rather than injecting a NULL/empty string into the
    prompt). Tolerates a missing agent_lessons table (e.g. agent-train has
    never been run yet) by returning no lessons rather than raising -- live
    recommendation runs must never fail just because train mode hasn't
    produced anything yet."""
    try:
        rows = conn.execute(
            """
            SELECT rule_text FROM agent_lessons
            WHERE status = 'approved'
              AND rule_text IS NOT NULL
              AND ((scope = 'competition' AND competition_id = ?)
                OR (scope = 'tier' AND tier = ?))
            ORDER BY created_at
            """,
            [competition_id, tier],
        ).fetchall()
    except duckdb.CatalogException:
        return []
    return [row[0] for row in rows]


def list_pending_by_source(conn: duckdb.DuckDBPyConnection, source: str) -> list[dict[str, Any]]:
    """Every pending lesson candidate from one source ('train'/'live') --
    intended for use by the not-yet-built auto_judge_live_lessons() (Task 3,
    src/live_lessons.py) to find only its own population. WHERE source = ?
    naturally excludes both the other source and any pre-migration row
    (source IS NULL, since SQL's `NULL = 'live'` is never true) --
    agent-train's human-reviewed queue is structurally unreachable from
    here, not just conventionally avoided."""
    rows = conn.execute(
        "SELECT id, lesson_text, competition_id, tier FROM agent_lessons "
        "WHERE status = 'pending' AND source = ? ORDER BY created_at",
        [source],
    ).fetchall()
    return [{"id": row[0], "lesson_text": row[1], "competition_id": row[2], "tier": row[3]} for row in rows]


def find_conflicting_rule(new_rule_text: str, existing_rules: list[str], llm_invoke: Callable[[str], str]) -> str | None:
    """A45 (design 3, chosen 2026-07-28): pairwise LLM contradiction check
    between a candidate rule and every already-approved rule that could
    co-occur with it in a live prompt. Concrete motivating case from this
    session: batch 156's reflection concluded "never bet result_3way when
    only BTTS is forecast," batch 157's concluded the opposite ("fall back
    to result_3way via odds-implied probability when only BTTS is
    forecast") -- both plausible, independently-reviewed rules that would
    silently coexist in the same live prompt with no warning.

    Returns None when no conflict is found, including when existing_rules is
    empty (nothing to conflict with). Returns the LLM's explanation --
    naming which existing rule conflicts and why -- when one is found.

    Deliberately does NOT catch exceptions from llm_invoke (unlike
    generate_batch_reflection/generate_rule_from_lesson, which both collapse
    "the call failed" into the same None as "nothing found"): a failed check
    and a clean check both returning None would make it impossible for a
    caller to fail open vs. fail closed differently, and those two outcomes
    genuinely warrant different handling -- see run_agent_lessons_approve,
    which fails open (warns, still approves) on a raised exception here but
    fails closed (refuses to approve without --force) when a real conflict
    is returned."""
    if not existing_rules:
        return None
    existing_numbered = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(existing_rules))
    prompt = (
        "You are checking a new proposed rule for an automated betting agent's system prompt against "
        "rules already approved and live for the same competition/tier.\n\n"
        f"NEW RULE:\n{new_rule_text}\n\n"
        f"ALREADY-APPROVED RULES:\n{existing_numbered}\n\n"
        "Does the new rule directly contradict any already-approved rule (i.e. following both together "
        "is impossible or produces opposite behavior in the same situation)? Rules about different "
        "situations, or rules that are merely both about the same market without contradicting, are NOT "
        "conflicts.\n\n"
        "If there is no contradiction, respond with exactly: NONE\n"
        "If there is a contradiction, respond with one sentence naming which numbered rule it conflicts "
        "with and why. Output nothing else."
    )
    response = llm_invoke(prompt).strip()
    if not response or response.upper().startswith("NONE"):
        return None
    return response


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


_LIMITATION_THEMES = {
    "injury/availability": ("injury",),
    "research coverage gap": ("research coverage gap",),
    "generic historical-data caveat": ("historical data", "current team performance", "current form"),
}


def generate_batch_lesson_text(records: list[Any]) -> str:
    """A39: deterministic aggregation of N BacktestRecord-shaped objects into
    one lesson candidate, mirroring mini-batch training -- aggregate a batch
    of outcomes into one update instead of one per sample. Same "structured
    summary, not NLG" contract as generate_lesson_text(): Counter-based
    tallies only, zero LLM calls, nothing here asks a model to synthesize
    the batch itself. Callers are responsible for only ever batching records
    that share one competition_id/tier (insert_lesson_candidate takes a
    single value of each per row) -- this function has no scope awareness
    of its own."""
    if not records:
        raise ValueError("generate_batch_lesson_text requires at least one record")

    n = len(records)
    league_counts = Counter(r.league or "an unlabeled competition" for r in records)
    league_label = (
        next(iter(league_counts))
        if len(league_counts) == 1
        else ", ".join(f"{lg} (n={c})" for lg, c in league_counts.most_common())
    )
    dates = sorted(r.date for r in records)
    date_range = dates[0] if dates[0] == dates[-1] else f"{dates[0]} to {dates[-1]}"

    overall_counts = Counter(r.recommendation.get("overall", "unknown") for r in records)
    overall_str = ", ".join(f"{count} {overall}" for overall, count in overall_counts.most_common())

    market_outcome_counts: Counter[tuple[str, str]] = Counter()
    confidence_outcome_counts: Counter[tuple[str, str]] = Counter()
    for r in records:
        confidence = r.recommendation.get("confidence", "unknown")
        for market in r.market_results:
            correct = market.get("correct")
            outcome = "correct" if correct is True else "incorrect" if correct is False else "unresolved"
            market_outcome_counts[(market.get("market", "unknown"), outcome)] += 1
            confidence_outcome_counts[(confidence, outcome)] += 1

    total_correct = sum(v for (_, o), v in market_outcome_counts.items() if o == "correct")
    total_incorrect = sum(v for (_, o), v in market_outcome_counts.items() if o == "incorrect")
    total_unresolved = sum(v for (_, o), v in market_outcome_counts.items() if o == "unresolved")
    resolved = total_correct + total_incorrect
    hit_rate_str = f"{total_correct / resolved:.0%}" if resolved else "n/a"

    per_market_incorrect: Counter[str] = Counter()
    per_market_total: Counter[str] = Counter()
    for (market, outcome), count in market_outcome_counts.items():
        per_market_total[market] += count
        if outcome == "incorrect":
            per_market_incorrect[market] += count
    if per_market_incorrect:
        worst_market, worst_count = per_market_incorrect.most_common(1)[0]
        worst_market_str = f"{worst_market} ({worst_count}/{per_market_total[worst_market]} incorrect)"
    else:
        worst_market_str = "none"

    conf_parts = []
    for confidence in sorted({c for c, _ in confidence_outcome_counts}):
        c_correct = confidence_outcome_counts.get((confidence, "correct"), 0)
        c_incorrect = confidence_outcome_counts.get((confidence, "incorrect"), 0)
        c_resolved = c_correct + c_incorrect
        if c_resolved:
            conf_parts.append(f"{confidence}={c_correct}/{c_resolved} correct")
    conf_str = "; ".join(conf_parts) if conf_parts else "no resolved markets"

    theme_counts: Counter[str] = Counter()
    for r in records:
        text = " ".join(r.recommendation.get("limitations") or []).lower()
        for theme, keywords in _LIMITATION_THEMES.items():
            if any(kw in text for kw in keywords):
                theme_counts[theme] += 1
    themes_str = (
        ", ".join(f"{theme} ({count}/{n})" for theme, count in theme_counts.most_common())
        if theme_counts
        else "none noted"
    )

    return (
        f"WHEN evaluating a batch of {n} matches ({league_label}, {date_range}): "
        f"overall recommendations were {overall_str}. "
        f"Markets: {total_correct} correct / {total_incorrect} incorrect / {total_unresolved} unresolved "
        f"({hit_rate_str} hit rate on resolved markets). "
        f"Most frequently wrong market: {worst_market_str}. "
        f"Confidence vs accuracy: {conf_str}. "
        f"Common limitation themes: {themes_str}."
    )


_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


def _classify_and_rank(records: list[Any]) -> tuple[list[Any], list[Any]]:
    """Split a batch into (misses, hits) by whether more of a record's
    resolved markets were incorrect than correct, each ranked
    highest-confidence-first -- the most informative examples for a
    reflection are the calls the agent was most sure about, not a random
    sample. Records with no resolved markets (insufficient_data, or every
    market unresolved) land in neither list."""
    misses, hits = [], []
    for r in records:
        correct = sum(1 for m in r.market_results if m.get("correct") is True)
        incorrect = sum(1 for m in r.market_results if m.get("correct") is False)
        if incorrect > correct:
            misses.append(r)
        elif correct > incorrect:
            hits.append(r)
    rank_key = lambda r: -_CONFIDENCE_RANK.get(r.recommendation.get("confidence", ""), 0)
    misses.sort(key=rank_key)
    hits.sort(key=rank_key)
    return misses, hits


def _describe_record(r: Any) -> str:
    overall = r.recommendation.get("overall", "unknown")
    confidence = r.recommendation.get("confidence", "unknown")
    # Found live: explanation is list[str] in the real schema (schema.py's
    # normalize_explanation, "one item per aspect") -- every real
    # recommendation has it as a list, not a string. .strip() on that raised
    # uncaught (generate_batch_reflection has no try/except of its own around
    # this), aborting the whole agent-train run's lesson-writing step.
    explanation_raw = r.recommendation.get("explanation") or []
    explanation = "; ".join(explanation_raw) if isinstance(explanation_raw, list) else str(explanation_raw).strip()
    markets_str = "; ".join(
        f"{m.get('market')}={m.get('selection')} "
        f"({'correct' if m.get('correct') is True else 'incorrect' if m.get('correct') is False else 'unresolved'})"
        for m in r.market_results
    ) or "no markets recommended"
    return (
        f"{r.home_team} vs {r.away_team} ({r.date}): recommended {overall} (confidence={confidence}). "
        f"Markets: {markets_str}. Actual result: {r.actual.get('result')}. "
        f'Agent\'s reasoning at the time: "{explanation}"'
    )


def generate_batch_reflection(
    records: list[Any], stats_text: str, llm_invoke: Callable[[str], str], n_examples: int = 5,
) -> str | None:
    """A42-follow-up (2026-07-28): LLM-synthesized reflective narrative over
    a batch, layered on top of generate_batch_lesson_text()'s deterministic
    stats rather than replacing them -- requested directly by the user after
    reviewing the pure-stats version ("not very sensible... I hope to see
    model's reasoning on reflecting the mistakes/accomplishments"). The
    stats stay the trustworthy, unhallucinatable anchor (passed in as
    stats_text, computed once by the caller rather than recomputed here);
    this adds the qualitative judgment on top.

    Deliberately takes a plain llm_invoke: str -> str callable instead of a
    langchain LLM object, so this module stays decoupled from langchain and
    trivially testable (pass a lambda/fake in tests, no message objects or
    mocked client needed). Callers (main.py) are responsible for wrapping
    their actual LLM into that shape.

    Returns None on any failure (network error, provider error, anything) --
    the caller falls back to stats-only rather than losing the whole lesson
    candidate over a transient API problem. This is a best-effort narrative,
    not a required field."""
    misses, hits = _classify_and_rank(records)
    misses_text = "\n".join(f"{i + 1}. {_describe_record(r)}" for i, r in enumerate(misses[:n_examples])) or "(none)"
    hits_text = "\n".join(f"{i + 1}. {_describe_record(r)}" for i, r in enumerate(hits[:n_examples])) or "(none)"

    prompt = (
        "You are reviewing a batch of betting recommendations an automated agent made for historical matches, "
        "now that the actual results are known. Below are deterministic statistics for the batch, followed by "
        "the agent's highest-confidence mistakes and its highest-confidence correct calls, each with the agent's "
        "own reasoning at the time.\n\n"
        f"STATISTICS:\n{stats_text}\n\n"
        f"NOTABLE MISSES (high-confidence calls that were wrong):\n{misses_text}\n\n"
        f"NOTABLE HITS (high-confidence calls that were right):\n{hits_text}\n\n"
        "Write a short reflective analysis (3-5 sentences) covering: (a) any systematic pattern behind the "
        "misses -- what kind of reasoning or evidence gap led the agent astray, (b) what the agent got right "
        "and why, (c) one concrete, actionable adjustment for future recommendations in this competition. "
        "Reference the specific examples above. Do not invent facts not present in the statistics or examples, "
        "and do not use generic hedging language like 'more data would help'."
    )
    try:
        reflection = llm_invoke(prompt)
    except Exception:
        return None
    return reflection.strip() or None
