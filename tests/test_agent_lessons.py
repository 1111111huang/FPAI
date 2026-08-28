"""Tests for A33's lesson/telemetry persistence (src/agent/lessons.py)."""
from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone

import duckdb
import pytest

from src.agent.lessons import (
    approve_lesson,
    create_lessons_tables,
    extract_competition_scope,
    find_conflicting_rule,
    generate_batch_lesson_text,
    generate_batch_reflection,
    generate_lesson_text,
    generate_rule_from_lesson,
    insert_lesson_candidate,
    insert_telemetry,
    judge_lesson_candidate,
    LessonDecision,
    list_pending_by_source,
    load_approved_lessons,
    reject_lesson,
)


def _conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    return conn


def test_create_lessons_tables_creates_both_tables():
    conn = _conn()
    lesson_cols = {row[1] for row in conn.execute("PRAGMA table_info('agent_lessons')").fetchall()}
    telemetry_cols = {row[1] for row in conn.execute("PRAGMA table_info('agent_telemetry')").fetchall()}
    assert lesson_cols == {
        "id", "lesson_text", "status", "competition_id", "tier", "scope",
        "source_match_id", "created_at", "reviewed_at", "reviewer", "rule_text",
        "source", "auto_decision_reasoning",
    }
    assert telemetry_cols == {
        "match_id", "run_id", "competition_resolution", "research_evidence",
        "forecast_payload", "recommendation", "created_at",
    }


def test_create_lessons_tables_is_idempotent():
    conn = _conn()
    create_lessons_tables(conn)  # second call must not raise
    assert conn.execute("SELECT COUNT(*) FROM agent_lessons").fetchone()[0] == 0


def test_insert_lesson_candidate_defaults_to_pending_with_null_scope():
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "WHEN evaluating E0 matches...", "E0", "competition_specific", "m1")
    row = conn.execute(
        "SELECT status, scope, competition_id, tier, source_match_id FROM agent_lessons WHERE id = ?", [lesson_id]
    ).fetchone()
    assert row == ("pending", None, "E0", "competition_specific", "m1")


def test_insert_lesson_candidate_allows_null_competition_id():
    """Leagueless internationals (resolve_competition returns competition=None)."""
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "WHEN evaluating international matches...", None, "general_purpose", "m2")
    row = conn.execute("SELECT competition_id, tier FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()
    assert row == (None, "general_purpose")


def test_insert_lesson_candidate_defaults_source_to_train():
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "text", "E0", "competition_specific", "m1")
    source = conn.execute("SELECT source FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()[0]
    assert source == "train"


def test_insert_lesson_candidate_accepts_explicit_live_source():
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "text", "E0", "competition_specific", "m1", source="live")
    source = conn.execute("SELECT source FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()[0]
    assert source == "live"


def test_list_pending_by_source_excludes_other_sources_and_null():
    conn = _conn()
    live_id = insert_lesson_candidate(conn, "live text", "E0", "competition_specific", "m1", source="live")
    train_id = insert_lesson_candidate(conn, "train text", "E0", "competition_specific", "m2", source="train")
    # A pre-migration row -- source column exists but was never populated for it.
    conn.execute(
        "INSERT INTO agent_lessons (lesson_text, status, competition_id, tier, source_match_id, created_at) "
        "VALUES ('legacy text', 'pending', 'E0', 'competition_specific', 'm3', ?)",
        [datetime.now(timezone.utc)],
    )

    pending = list_pending_by_source(conn, source="live")

    assert [p["id"] for p in pending] == [live_id]
    assert pending[0]["lesson_text"] == "live text"
    assert pending[0]["competition_id"] == "E0"
    assert pending[0]["tier"] == "competition_specific"

    # Proves this is a genuine source == ? match, not a hardcoded
    # "source != 'train'" exclusion -- querying by "train" on this same
    # fixture must return exactly the train row, not the live one.
    train_pending = list_pending_by_source(conn, source="train")
    assert [p["id"] for p in train_pending] == [train_id]
    assert train_pending[0]["lesson_text"] == "train text"


def test_list_pending_by_source_includes_created_at_and_source_match_id():
    """New fields needed by app/backend/live_lessons.py's weekly grouped
    judge to label each candidate's section when joining several days'
    lesson_texts into one combined document (W184)."""
    conn = _conn()
    insert_lesson_candidate(conn, "live text", "E0", "competition_specific", "m1,m2", source="live")

    pending = list_pending_by_source(conn, source="live")

    assert pending[0]["source_match_id"] == "m1,m2"
    assert isinstance(pending[0]["created_at"], datetime)


def test_list_pending_by_source_excludes_already_reviewed_rows():
    conn = _conn()
    live_id = insert_lesson_candidate(conn, "live text", "E0", "competition_specific", "m1", source="live")
    reject_lesson(conn, live_id, reviewer="test")

    assert list_pending_by_source(conn, source="live") == []


def test_insert_telemetry_round_trips_json_fields():
    conn = _conn()
    insert_telemetry(
        conn,
        match_id="m1",
        run_id="run-1",
        competition_resolution={"competition": "E0", "tier": "competition_specific"},
        research_evidence={"availability": "ok"},
        forecast_payload={"result_3way": {"probabilities": {"home": 0.5}}},
        recommendation={"overall": "no_bet"},
    )
    row = conn.execute(
        "SELECT match_id, run_id, competition_resolution, research_evidence, forecast_payload, recommendation "
        "FROM agent_telemetry WHERE match_id = 'm1'"
    ).fetchone()
    assert row[0] == "m1"
    assert row[1] == "run-1"
    assert json.loads(row[2]) == {"competition": "E0", "tier": "competition_specific"}
    assert json.loads(row[3]) == {"availability": "ok"}
    assert json.loads(row[4]) == {"result_3way": {"probabilities": {"home": 0.5}}}
    assert json.loads(row[5]) == {"overall": "no_bet"}


def test_approve_lesson_sets_status_scope_rule_text_reviewed_at_reviewer():
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "text", "E0", "competition_specific", "m1")
    approve_lesson(conn, lesson_id, "competition", "alice", "NEVER bet BTTS without checking BTTS odds.")
    row = conn.execute(
        "SELECT status, scope, rule_text, reviewer FROM agent_lessons WHERE id = ?", [lesson_id]
    ).fetchone()
    assert row == ("approved", "competition", "NEVER bet BTTS without checking BTTS odds.", "alice")
    reviewed_at = conn.execute("SELECT reviewed_at FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()[0]
    assert reviewed_at is not None


def test_approve_lesson_rejects_invalid_scope():
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "text", "E0", "competition_specific", "m1")
    with pytest.raises(ValueError, match="scope"):
        approve_lesson(conn, lesson_id, "league", "alice", "NEVER do X.")


def test_approve_lesson_rejects_empty_rule_text():
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "text", "E0", "competition_specific", "m1")
    with pytest.raises(ValueError, match="rule_text"):
        approve_lesson(conn, lesson_id, "competition", "alice", "   ")


def test_approve_lesson_raises_for_unknown_id():
    conn = _conn()
    with pytest.raises(ValueError, match="999"):
        approve_lesson(conn, 999, "competition", "alice", "NEVER do X.")


def test_reject_lesson_sets_status_rejected():
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "text", "E0", "competition_specific", "m1")
    reject_lesson(conn, lesson_id, "bob")
    row = conn.execute("SELECT status, reviewer FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()
    assert row == ("rejected", "bob")


def test_reject_lesson_raises_for_unknown_id():
    conn = _conn()
    with pytest.raises(ValueError, match="999"):
        reject_lesson(conn, 999, "bob")


def test_load_approved_lessons_matches_competition_scope_only_for_same_competition():
    conn = _conn()
    e0_id = insert_lesson_candidate(conn, "E0 lesson", "E0", "competition_specific", "m1")
    sp1_id = insert_lesson_candidate(conn, "SP1 lesson", "SP1", "competition_specific", "m2")
    approve_lesson(conn, e0_id, "competition", "alice", "E0 rule")
    approve_lesson(conn, sp1_id, "competition", "alice", "SP1 rule")

    result = load_approved_lessons(conn, "E0", "competition_specific")
    assert result == ["E0 rule"]


def test_load_approved_lessons_matches_tier_scope_regardless_of_competition():
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "tier lesson", "SWE_ALLS", "general_purpose", "m1")
    approve_lesson(conn, lesson_id, "tier", "alice", "tier rule")

    result = load_approved_lessons(conn, "SOME_OTHER_LEAGUE", "general_purpose")
    assert result == ["tier rule"]


def test_load_approved_lessons_excludes_approved_rows_with_null_rule_text():
    """A44: an approved row with no rule_text (only reachable pre-A44, since
    approve_lesson now requires one) must never leak lesson_text into the
    live prompt -- simulated here via direct SQL, bypassing approve_lesson's
    own validation, the way a genuinely pre-migration row would look."""
    conn = _conn()
    legacy_id = insert_lesson_candidate(conn, "legacy lesson", "E0", "competition_specific", "m1")
    conn.execute(
        "UPDATE agent_lessons SET status = 'approved', scope = 'competition', rule_text = NULL WHERE id = ?",
        [legacy_id],
    )

    result = load_approved_lessons(conn, "E0", "competition_specific")
    assert result == []


def test_load_approved_lessons_excludes_pending_and_rejected():
    conn = _conn()
    pending_id = insert_lesson_candidate(conn, "pending lesson", "E0", "competition_specific", "m1")
    rejected_id = insert_lesson_candidate(conn, "rejected lesson", "E0", "competition_specific", "m2")
    reject_lesson(conn, rejected_id, "alice")
    # pending_id stays pending -- never approved

    result = load_approved_lessons(conn, "E0", "competition_specific")
    assert result == []


def test_load_approved_lessons_returns_empty_list_when_table_missing():
    conn = duckdb.connect(":memory:")  # create_lessons_tables() never called
    assert load_approved_lessons(conn, "E0", "competition_specific") == []


def test_load_approved_lessons_signature_has_no_status_override_parameter():
    """A33 acceptance: live mode must structurally be unable to fetch
    pending/rejected lessons -- proven here by the function itself having no
    parameter that could select anything but the hardcoded status='approved'."""
    params = set(inspect.signature(load_approved_lessons).parameters)
    assert params == {"conn", "competition_id", "tier"}


def test_extract_competition_scope_reads_competition_and_tier():
    full_state = {"competition_resolution": {"competition": "E0", "tier": "competition_specific"}}
    assert extract_competition_scope(full_state) == ("E0", "competition_specific")


def test_extract_competition_scope_defaults_tier_general_purpose_when_missing():
    assert extract_competition_scope({}) == (None, "general_purpose")
    full_state = {"competition_resolution": {"competition": None, "tier": None}}
    assert extract_competition_scope(full_state) == (None, "general_purpose")


class _FakeRecord:
    def __init__(
        self, league, recommendation, market_results, actual,
        match_id="m", date="2025-01-01", home_team="Home", away_team="Away",
    ):
        self.league = league
        self.recommendation = recommendation
        self.market_results = market_results
        self.actual = actual
        self.match_id = match_id
        self.date = date
        self.home_team = home_team
        self.away_team = away_team


def test_generate_lesson_text_includes_context_overall_and_market_outcomes():
    record = _FakeRecord(
        league="E0",
        recommendation={
            "overall": "direct_bet", "confidence": "high",
            "prediction_basis": "team_history_and_market", "limitations": [],
        },
        market_results=[{"market": "result_3way", "selection": "home", "correct": True}],
        actual={"result": "home"},
    )
    text = generate_lesson_text(record)
    assert text.startswith("WHEN evaluating E0 matches")
    assert "direct_bet" in text
    assert "result_3way=home (correct)" in text


def test_generate_lesson_text_handles_no_markets_and_limitations():
    record = _FakeRecord(
        league=None,
        recommendation={
            "overall": "insufficient_data", "confidence": "low",
            "prediction_basis": "unknown", "limitations": ["no odds available"],
        },
        market_results=[],
        actual={"result": "draw"},
    )
    text = generate_lesson_text(record)
    assert "an unlabeled competition" in text
    assert "no markets recommended" in text
    assert "no odds available" in text


def test_generate_batch_lesson_text_rejects_empty_batch():
    with pytest.raises(ValueError):
        generate_batch_lesson_text([])


def test_generate_batch_lesson_text_aggregates_hit_rate_and_worst_market():
    records = [
        _FakeRecord(
            league="E0", match_id="m1", date="2025-01-01",
            recommendation={"overall": "direct_bet", "confidence": "high", "limitations": []},
            market_results=[
                {"market": "result_3way", "selection": "home", "correct": True},
                {"market": "btts", "selection": "yes", "correct": False},
            ],
            actual={"result": "home"},
        ),
        _FakeRecord(
            league="E0", match_id="m2", date="2025-01-08",
            recommendation={"overall": "conditional", "confidence": "medium", "limitations": ["injury news not available"]},
            market_results=[
                {"market": "result_3way", "selection": "draw", "correct": False},
                {"market": "btts", "selection": "no", "correct": False},
            ],
            actual={"result": "away"},
        ),
    ]
    text = generate_batch_lesson_text(records)

    assert text.startswith("WHEN evaluating a batch of 2 matches (E0, 2025-01-01 to 2025-01-08)")
    assert "1 correct / 3 incorrect / 0 unresolved" in text
    assert "25% hit rate" in text
    assert "btts (2/2 incorrect)" in text  # worst market: btts wrong both times
    assert "high=1/2 correct" in text  # record 1's 2 markets (1 correct, 1 incorrect), both confidence=high
    assert "medium=0/2 correct" in text
    assert "injury/availability (1/2)" in text


def test_generate_batch_reflection_includes_stats_and_examples_in_prompt():
    # explanation is a list[str] in the real schema (src/agent/schema.py:40,
    # normalize_explanation's "one item per aspect" bullet-point design) --
    # every real recommendation this whole session has had it as a list, not
    # a plain string. Found live: a string here masked a real crash in
    # _describe_record's .strip() call on what's actually always a list.
    records = [
        _FakeRecord(
            league="E0", match_id="m1", date="2025-01-01", home_team="City", away_team="Villa",
            recommendation={"overall": "direct_bet", "confidence": "high", "explanation": ["Confident home win pick."], "limitations": []},
            market_results=[{"market": "result_3way", "selection": "home", "correct": False}],
            actual={"result": "away"},
        ),
        _FakeRecord(
            league="E0", match_id="m2", date="2025-01-08", home_team="Spurs", away_team="Fulham",
            recommendation={"overall": "direct_bet", "confidence": "high", "explanation": ["Strong home form.", "Good recent H2H record."], "limitations": []},
            market_results=[{"market": "result_3way", "selection": "home", "correct": True}],
            actual={"result": "home"},
        ),
    ]
    stats_text = generate_batch_lesson_text(records)
    captured = {}

    def fake_invoke(prompt: str) -> str:
        captured["prompt"] = prompt
        return "The agent overrated home favourites; City's loss shows overconfidence in home form alone."

    reflection = generate_batch_reflection(records, stats_text, fake_invoke)

    assert reflection == "The agent overrated home favourites; City's loss shows overconfidence in home form alone."
    assert stats_text in captured["prompt"]
    assert "City vs Villa" in captured["prompt"]
    assert "Confident home win pick." in captured["prompt"]
    assert "Spurs vs Fulham" in captured["prompt"]
    assert "Strong home form." in captured["prompt"]


def test_generate_batch_reflection_returns_none_on_llm_failure():
    records = [
        _FakeRecord(
            league="E0",
            recommendation={"overall": "direct_bet", "confidence": "high", "explanation": ["x"], "limitations": []},
            market_results=[{"market": "result_3way", "selection": "home", "correct": False}],
            actual={"result": "away"},
        ),
    ]
    stats_text = generate_batch_lesson_text(records)

    def failing_invoke(prompt: str) -> str:
        raise RuntimeError("API down")

    assert generate_batch_reflection(records, stats_text, failing_invoke) is None


def test_generate_batch_reflection_returns_none_on_empty_response():
    records = [
        _FakeRecord(
            league="E0",
            recommendation={"overall": "direct_bet", "confidence": "high", "explanation": ["x"], "limitations": []},
            market_results=[{"market": "result_3way", "selection": "home", "correct": False}],
            actual={"result": "away"},
        ),
    ]
    stats_text = generate_batch_lesson_text(records)

    assert generate_batch_reflection(records, stats_text, lambda p: "   ") is None


def test_generate_rule_from_lesson_returns_llm_output_stripped():
    captured = {}

    def fake_invoke(prompt: str) -> str:
        captured["prompt"] = prompt
        return "  NEVER bet BTTS without checking BTTS odds.  \n"

    result = generate_rule_from_lesson("WHEN evaluating a batch of 80 matches...", fake_invoke)

    assert result == "NEVER bet BTTS without checking BTTS odds."
    assert "WHEN evaluating a batch of 80 matches..." in captured["prompt"]
    assert "NEVER" in captured["prompt"] and "IF" in captured["prompt"]


def test_generate_rule_from_lesson_returns_none_on_failure():
    def failing_invoke(prompt: str) -> str:
        raise RuntimeError("API down")

    assert generate_rule_from_lesson("some lesson text", failing_invoke) is None


def test_generate_rule_from_lesson_returns_none_on_blank_response():
    assert generate_rule_from_lesson("some lesson text", lambda p: "   ") is None


def test_find_conflicting_rule_returns_none_without_calling_llm_when_no_existing_rules():
    calls = []
    result = find_conflicting_rule("NEVER do X.", [], lambda p: calls.append(p) or "NONE")
    assert result is None
    assert calls == []


def test_find_conflicting_rule_returns_none_when_llm_says_none():
    result = find_conflicting_rule(
        "NEVER bet BTTS without checking odds.",
        ["IF confidence is low, prefer no_bet."],
        lambda p: "NONE",
    )
    assert result is None


def test_find_conflicting_rule_returns_explanation_when_conflict_found():
    captured = {}

    def fake_invoke(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Conflicts with rule 1: it recommends the exact action this rule forbids."

    result = find_conflicting_rule(
        "NEVER recommend result_3way without a direct model probability.",
        ["IF only BTTS is forecast, evaluate result_3way via odds-implied probability."],
        fake_invoke,
    )

    assert result == "Conflicts with rule 1: it recommends the exact action this rule forbids."
    assert "NEVER recommend result_3way" in captured["prompt"]
    assert "IF only BTTS is forecast" in captured["prompt"]


def test_find_conflicting_rule_propagates_llm_exceptions():
    """Unlike generate_batch_reflection/generate_rule_from_lesson, failures
    here must NOT be swallowed into None -- callers need to distinguish
    'check failed' from 'check ran, found nothing' to fail open vs. closed
    correctly (see run_agent_lessons_approve)."""
    def failing_invoke(prompt: str) -> str:
        raise RuntimeError("API down")

    with pytest.raises(RuntimeError, match="API down"):
        find_conflicting_rule("NEVER do X.", ["some existing rule"], failing_invoke)


def test_judge_lesson_candidate_parses_a_plain_json_approval():
    captured = {}

    def fake_invoke(prompt: str) -> str:
        captured["prompt"] = prompt
        return '{"approve": true, "scope": "competition", "reasoning": "Clear systematic pattern."}'

    decision = judge_lesson_candidate("WHEN evaluating a batch...", "E0", "competition_specific", fake_invoke)

    assert decision.approve is True
    assert decision.scope == "competition"
    assert decision.reasoning == "Clear systematic pattern."
    assert "E0" in captured["prompt"]
    assert "competition_specific" in captured["prompt"]


def test_judge_lesson_candidate_parses_json_wrapped_in_a_markdown_fence():
    def fake_invoke(prompt: str) -> str:
        return '```json\n{"approve": false, "scope": null, "reasoning": "Sample too thin."}\n```'

    decision = judge_lesson_candidate("WHEN evaluating a batch...", "E0", "competition_specific", fake_invoke)

    assert decision.approve is False
    assert decision.scope is None
    assert decision.reasoning == "Sample too thin."


def test_judge_lesson_candidate_rejects_on_invalid_scope_value():
    def fake_invoke(prompt: str) -> str:
        return '{"approve": true, "scope": "everywhere", "reasoning": "Broad pattern."}'

    decision = judge_lesson_candidate("some lesson", "E0", "competition_specific", fake_invoke)

    assert decision.approve is False
    assert decision.scope is None
    assert "everywhere" in decision.reasoning


def test_judge_lesson_candidate_rejects_on_malformed_json():
    def fake_invoke(prompt: str) -> str:
        return "not json at all"

    decision = judge_lesson_candidate("some lesson", "E0", "competition_specific", fake_invoke)

    assert decision.approve is False
    assert decision.scope is None


def test_judge_lesson_candidate_rejects_on_llm_exception():
    def failing_invoke(prompt: str) -> str:
        raise RuntimeError("API down")

    decision = judge_lesson_candidate("some lesson", "E0", "competition_specific", failing_invoke)

    assert decision.approve is False
    assert "API down" in decision.reasoning


def test_judge_lesson_candidate_rejects_on_stringified_approve_false():
    def fake_invoke(prompt: str) -> str:
        return '{"approve": "false", "scope": "competition", "reasoning": "..."}'

    decision = judge_lesson_candidate("some lesson", "E0", "competition_specific", fake_invoke)

    assert decision.approve is False


def test_generate_batch_lesson_text_handles_no_resolved_markets():
    records = [
        _FakeRecord(
            league=None, match_id="m1", date="2025-02-01",
            recommendation={"overall": "insufficient_data", "confidence": "low", "limitations": []},
            market_results=[],
            actual={"result": "draw"},
        ),
    ]
    text = generate_batch_lesson_text(records)
    assert "0 correct / 0 incorrect / 0 unresolved" in text
    assert "n/a hit rate" in text
    assert "Most frequently wrong market: none." in text
    assert "no resolved markets" in text
    assert "none noted" in text
