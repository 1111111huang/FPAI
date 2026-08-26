"""W177: live_lessons.py -- adapts RecommendationOutcome rows into
BacktestRecord-shaped objects and batches them into agent_lessons
candidates via src/agent/lessons.py's existing, unmodified functions."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import duckdb

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.football_data_client import NormalizedMatch
from app.backend.live_lessons import (
    PreparedLessonBatch,
    _to_lesson_record,
    auto_judge_live_lessons,
    commit_lesson_batches,
    generate_daily_lessons,
)
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcome, RecommendationOutcomeStore
from src.agent.lessons import approve_lesson, create_lessons_tables, insert_lesson_candidate
from src.utils.db_manager import DuckDBManager


def _outcome(**overrides) -> RecommendationOutcome:
    defaults = dict(
        id=1, match_id="m1", date="2026-08-22", competition="Premier League",
        market="result_3way", selection="home", recommendation_type="direct_bet",
        confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2026-08-22T10:00:00+00:00", resolved_at="2026-08-23T00:00:00+00:00",
        competition_id="E0", home_goals=2, away_goals=1, lesson_batched_at=None,
    )
    defaults.update(overrides)
    return RecommendationOutcome(**defaults)


def _rec(league: str = "E0") -> dict:
    return {
        "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": league},
        "overall": "direct_bet",
        "markets": [{
            "market": "result_3way", "selection": "home", "recommendation_type": "direct_bet",
            "current_odds": 2.0, "value_edge": 0.1,
        }],
        "confidence": "medium", "explanation": ["good value"], "limitations": [],
        "prediction_basis": "team_history_and_market",
    }


def _duckdb_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    return conn


def test_to_lesson_record_enriches_from_a_real_cache_hit(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec(), "scheduled")

    record = _to_lesson_record(_outcome(), cache)

    assert record.home_team == "Arsenal"
    assert record.away_team == "Everton"
    assert record.league == "E0"
    assert record.actual["result"] == "home"
    assert record.market_results == [{"market": "result_3way", "selection": "home", "correct": True}]


def test_to_lesson_record_degrades_gracefully_on_cache_miss(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")  # nothing recorded

    record = _to_lesson_record(_outcome(), cache)

    assert record.home_team == ""
    assert record.away_team == ""
    assert record.recommendation == {}


def test_to_lesson_record_degrades_gracefully_on_a_pre_migration_missing_score(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec(), "scheduled")

    record = _to_lesson_record(_outcome(home_goals=None, away_goals=None), cache)

    assert record.actual == {}


def test_generate_daily_lessons_groups_by_competition_and_date(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="m1", date="2026-08-22", competition="Premier League", market="result_3way",
        selection="home", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=True, generated_at="2026-08-22T10:00:00+00:00",
        competition_id="E0", home_goals=2, away_goals=1,
    )
    store.insert(
        match_id="m2", date="2026-08-22", competition="La Liga", market="result_3way",
        selection="away", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=False, generated_at="2026-08-22T10:00:00+00:00",
        competition_id="SP1", home_goals=0, away_goals=1,
    )
    client = MagicMock()
    client.get_results.return_value = []  # nothing new to resolve this run
    conn = _duckdb_conn()

    lesson_ids = generate_daily_lessons(cache, store, client, conn, llm_invoke=None)

    assert len(lesson_ids) == 2  # E0 batch and SP1 batch, never merged
    rows = conn.execute(
        "SELECT competition_id, source_match_id FROM agent_lessons ORDER BY competition_id"
    ).fetchall()
    assert rows == [("E0", "m1"), ("SP1", "m2")]


def test_generate_daily_lessons_skips_outcomes_with_no_competition_id(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="m1", date="2026-08-22", competition="Premier League", market="result_3way",
        selection="home", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=True, generated_at="2026-08-22T10:00:00+00:00",
    )  # competition_id deliberately omitted -- simulates a pre-W175 row
    client = MagicMock()
    client.get_results.return_value = []
    conn = _duckdb_conn()

    lesson_ids = generate_daily_lessons(cache, store, client, conn, llm_invoke=None)

    assert lesson_ids == []
    assert len(store.list_unbatched_for_lessons()) == 1  # still unbatched, not silently marked done


def test_generate_daily_lessons_skips_a_batch_with_an_unregistered_competition_id(tmp_path: Path) -> None:
    """get_competition_definition raises ValueError for an id not in the
    registry (config/competitions.yaml) -- that batch must be skipped and
    logged, not crash the whole run, and its outcome must stay unbatched
    for a human to fix. A sibling valid-competition batch in the same call
    must still succeed."""
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="m1", date="2026-08-22", competition="Premier League", market="result_3way",
        selection="home", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=True, generated_at="2026-08-22T10:00:00+00:00",
        competition_id="E0", home_goals=2, away_goals=1,
    )
    store.insert(
        match_id="m2", date="2026-08-22", competition="Unknown League", market="result_3way",
        selection="home", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=True, generated_at="2026-08-22T10:00:00+00:00",
        competition_id="ZZ", home_goals=1, away_goals=0,
    )
    client = MagicMock()
    client.get_results.return_value = []
    conn = _duckdb_conn()

    lesson_ids = generate_daily_lessons(cache, store, client, conn, llm_invoke=None)

    assert len(lesson_ids) == 1  # only the E0 batch made it
    rows = conn.execute("SELECT competition_id FROM agent_lessons").fetchall()
    assert rows == [("E0",)]
    unbatched_match_ids = [o.match_id for o in store.list_unbatched_for_lessons()]
    assert unbatched_match_ids == ["m2"]  # ZZ's outcome left unbatched, not silently marked done


def test_generate_daily_lessons_marks_outcomes_batched(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="m1", date="2026-08-22", competition="Premier League", market="result_3way",
        selection="home", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=True, generated_at="2026-08-22T10:00:00+00:00",
        competition_id="E0", home_goals=2, away_goals=1,
    )
    client = MagicMock()
    client.get_results.return_value = []
    conn = _duckdb_conn()

    generate_daily_lessons(cache, store, client, conn, llm_invoke=None)

    assert store.list_unbatched_for_lessons() == []


def test_generate_daily_lessons_prepends_the_live_source_note_and_skips_reflection_without_an_llm(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="m1", date="2026-08-22", competition="Premier League", market="result_3way",
        selection="home", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=True, generated_at="2026-08-22T10:00:00+00:00",
        competition_id="E0", home_goals=2, away_goals=1,
    )
    client = MagicMock()
    client.get_results.return_value = []
    conn = _duckdb_conn()

    generate_daily_lessons(cache, store, client, conn, llm_invoke=None)

    lesson_text = conn.execute("SELECT lesson_text FROM agent_lessons").fetchone()[0]
    assert lesson_text.startswith("Live-sourced batch:")
    assert "Reflection:" not in lesson_text


def test_generate_daily_lessons_appends_reflection_when_llm_invoke_given(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="m1", date="2026-08-22", competition="Premier League", market="result_3way",
        selection="home", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=True, generated_at="2026-08-22T10:00:00+00:00",
        competition_id="E0", home_goals=2, away_goals=1,
    )
    client = MagicMock()
    client.get_results.return_value = []
    conn = _duckdb_conn()

    generate_daily_lessons(cache, store, client, conn, llm_invoke=lambda prompt: "a real reflection")

    lesson_text = conn.execute("SELECT lesson_text FROM agent_lessons").fetchone()[0]
    assert "Reflection: a real reflection" in lesson_text


def test_generate_daily_lessons_resolves_pending_recommendations_first(tmp_path: Path) -> None:
    """End-to-end: a brand-new, not-yet-resolved recommendation gets
    resolved and then batched in the same call -- proves the two steps
    run as one pipeline, not something a caller must sequence itself."""
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec(), "scheduled")
    client = MagicMock()

    def fake_get_results(competition_code, date_from, date_to):
        if competition_code == "PL":
            return [NormalizedMatch(
                match_id="m1", utc_date="2026-08-22T15:00:00Z", status="FINISHED",
                home_team="Arsenal", away_team="Everton", home_goals=2, away_goals=1,
            )]
        return []

    client.get_results.side_effect = fake_get_results
    conn = _duckdb_conn()

    lesson_ids = generate_daily_lessons(cache, store, client, conn, llm_invoke=None)

    assert len(lesson_ids) == 1
    assert store.list_all()[0].correct is True


def test_commit_lesson_batches_writes_source_live(tmp_path: Path) -> None:
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    conn = _duckdb_conn()
    batch = PreparedLessonBatch(
        competition_id="E0", tier="competition_specific", lesson_text="Live-sourced batch: test.",
        match_ids="m1", outcome_ids=[],
    )

    commit_lesson_batches(conn, store, [batch])

    source = conn.execute("SELECT source FROM agent_lessons").fetchone()[0]
    assert source == "live"


def _dm(tmp_path: Path) -> DuckDBManager:
    dm = DuckDBManager()
    dm.db_path = tmp_path / "fpai_core.db"
    return dm


def test_auto_judge_live_lessons_approves_a_good_candidate(tmp_path: Path) -> None:
    dm = _dm(tmp_path)
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    with dm.connection() as conn:
        create_lessons_tables(conn)
        insert_lesson_candidate(conn, "Live-sourced batch: strong pattern.", "E0", "competition_specific", "m1", source="live")

    def fake_invoke(prompt: str) -> str:
        if "deciding whether to promote" in prompt:
            return '{"approve": true, "scope": "competition", "reasoning": "Clear pattern."}'
        if "Extract ONLY the single most programmatic" in prompt:
            return "NEVER bet result_3way on a thin sample."
        if "checking a new proposed rule" in prompt:
            return "NONE"
        raise AssertionError(f"unexpected prompt: {prompt}")

    outcomes = auto_judge_live_lessons(dm, fake_invoke)

    assert len(outcomes) == 1
    with dm.connection(read_only=True) as conn:
        row = conn.execute(
            "SELECT status, scope, rule_text, reviewer, auto_decision_reasoning FROM agent_lessons"
        ).fetchone()
    assert row[0] == "approved"
    assert row[1] == "competition"
    assert row[2] == "NEVER bet result_3way on a thin sample."
    assert row[3] == "agent-auto"
    assert row[4] == "Clear pattern."


def test_auto_judge_live_lessons_rejects_a_weak_candidate(tmp_path: Path) -> None:
    dm = _dm(tmp_path)
    with dm.connection() as conn:
        create_lessons_tables(conn)
        insert_lesson_candidate(conn, "Live-sourced batch: n=1.", "E0", "competition_specific", "m1", source="live")

    def fake_invoke(prompt: str) -> str:
        return '{"approve": false, "scope": null, "reasoning": "Sample too thin."}'

    auto_judge_live_lessons(dm, fake_invoke)

    with dm.connection(read_only=True) as conn:
        row = conn.execute("SELECT status, reviewer, auto_decision_reasoning FROM agent_lessons").fetchone()
    assert row[0] == "rejected"
    assert row[1] == "agent-auto"
    assert row[2] == "Sample too thin."


def test_auto_judge_live_lessons_leaves_a_conflict_pending(tmp_path: Path) -> None:
    dm = _dm(tmp_path)
    with dm.connection() as conn:
        create_lessons_tables(conn)
        # An already-approved rule in the same scope to conflict with.
        existing_id = insert_lesson_candidate(conn, "existing text", "E0", "competition_specific", "m0", source="train")
        approve_lesson(conn, existing_id, scope="competition", reviewer="test", rule_text="ALWAYS bet result_3way when confident.")
        insert_lesson_candidate(conn, "Live-sourced batch: new pattern.", "E0", "competition_specific", "m1", source="live")

    def fake_invoke(prompt: str) -> str:
        if "deciding whether to promote" in prompt:
            return '{"approve": true, "scope": "competition", "reasoning": "Looks solid."}'
        if "Extract ONLY the single most programmatic" in prompt:
            return "NEVER bet result_3way when confident."
        if "checking a new proposed rule" in prompt:
            return "Conflicts with rule 1: it recommends the exact opposite action."
        raise AssertionError(f"unexpected prompt: {prompt}")

    auto_judge_live_lessons(dm, fake_invoke)

    with dm.connection(read_only=True) as conn:
        rows = conn.execute(
            "SELECT status, auto_decision_reasoning FROM agent_lessons WHERE source = 'live'"
        ).fetchall()
    assert rows[0][0] == "pending"
    assert "Conflicts with rule 1" in rows[0][1]


def test_auto_judge_live_lessons_leaves_pending_on_distillation_failure(tmp_path: Path) -> None:
    dm = _dm(tmp_path)
    with dm.connection() as conn:
        create_lessons_tables(conn)
        insert_lesson_candidate(conn, "Live-sourced batch: pattern.", "E0", "competition_specific", "m1", source="live")

    def fake_invoke(prompt: str) -> str:
        if "deciding whether to promote" in prompt:
            return '{"approve": true, "scope": "competition", "reasoning": "Looks solid."}'
        if "Extract ONLY the single most programmatic" in prompt:
            return "   "  # blank -- generate_rule_from_lesson returns None for this
        raise AssertionError(f"unexpected prompt: {prompt}")

    auto_judge_live_lessons(dm, fake_invoke)

    with dm.connection(read_only=True) as conn:
        row = conn.execute("SELECT status, auto_decision_reasoning FROM agent_lessons").fetchone()
    assert row[0] == "pending"
    assert "distillation failed" in row[1]


def test_auto_judge_live_lessons_never_touches_train_sourced_rows(tmp_path: Path) -> None:
    dm = _dm(tmp_path)
    with dm.connection() as conn:
        create_lessons_tables(conn)
        train_id = insert_lesson_candidate(conn, "train text", "E0", "competition_specific", "m1", source="train")

    def fake_invoke(prompt: str) -> str:
        raise AssertionError("should never be called -- no live-sourced pending rows exist")

    outcomes = auto_judge_live_lessons(dm, fake_invoke)

    assert outcomes == []
    with dm.connection(read_only=True) as conn:
        status = conn.execute("SELECT status FROM agent_lessons WHERE id = ?", [train_id]).fetchone()[0]
    assert status == "pending"


def test_auto_judge_live_lessons_is_a_noop_when_llm_invoke_is_none(tmp_path: Path) -> None:
    dm = _dm(tmp_path)
    with dm.connection() as conn:
        create_lessons_tables(conn)
        insert_lesson_candidate(conn, "text", "E0", "competition_specific", "m1", source="live")

    outcomes = auto_judge_live_lessons(dm, None)

    assert outcomes == []
    with dm.connection(read_only=True) as conn:
        status = conn.execute("SELECT status FROM agent_lessons").fetchone()[0]
    assert status == "pending"


def test_auto_judge_live_lessons_isolates_a_conflict_check_failure_to_its_own_candidate(tmp_path: Path) -> None:
    """A raised exception from find_conflicting_rule (fail-closed by
    design -- see its own docstring) must defer only the candidate it was
    checking, not silently discard a sibling candidate's already-computed
    decision in the same batch."""
    dm = _dm(tmp_path)
    with dm.connection() as conn:
        create_lessons_tables(conn)
        existing_id = insert_lesson_candidate(conn, "existing text", "E0", "competition_specific", "m0", source="train")
        approve_lesson(conn, existing_id, scope="competition", reviewer="test", rule_text="ALWAYS bet result_3way when confident.")
        insert_lesson_candidate(conn, "Live-sourced batch: pattern A.", "E0", "competition_specific", "m1", source="live")
        insert_lesson_candidate(conn, "Live-sourced batch: pattern B.", "E0", "competition_specific", "m2", source="live")

    def fake_invoke(prompt: str) -> str:
        if "deciding whether to promote" in prompt:
            reasoning = "A looks solid." if "pattern A" in prompt else "B looks solid."
            return f'{{"approve": true, "scope": "competition", "reasoning": "{reasoning}"}}'
        if "Extract ONLY the single most programmatic" in prompt:
            return "NEVER bet result_3way on pattern A." if "pattern A" in prompt else "NEVER bet result_3way on pattern B."
        if "checking a new proposed rule" in prompt:
            if "pattern A" in prompt:
                raise RuntimeError("conflict-check API down")
            return "NONE"
        raise AssertionError(f"unexpected prompt: {prompt}")

    results = auto_judge_live_lessons(dm, fake_invoke)

    assert len(results) == 2  # both candidates processed -- neither batch member lost
    with dm.connection(read_only=True) as conn:
        rows = {
            row[0]: row for row in conn.execute(
                "SELECT lesson_text, status, rule_text, auto_decision_reasoning FROM agent_lessons WHERE source = 'live'"
            ).fetchall()
        }
    a_row = rows["Live-sourced batch: pattern A."]
    b_row = rows["Live-sourced batch: pattern B."]
    assert a_row[1] == "pending"
    assert "conflict check failed" in a_row[3]
    assert b_row[1] == "approved"
    assert b_row[2] == "NEVER bet result_3way on pattern B."


def test_auto_judge_live_lessons_isolates_a_write_failure_to_its_own_candidate(tmp_path: Path) -> None:
    """A raised exception writing one candidate's decision (e.g. its row
    vanished between phase 1's read and phase 3's write) must not abort the
    write loop and skip every later candidate's already-decided write."""
    dm = _dm(tmp_path)
    with dm.connection() as conn:
        create_lessons_tables(conn)
        id1 = insert_lesson_candidate(conn, "Live-sourced batch: candidate one.", "E0", "competition_specific", "m1", source="live")
        id2 = insert_lesson_candidate(conn, "Live-sourced batch: candidate two.", "E0", "competition_specific", "m2", source="live")

    def fake_invoke(prompt: str) -> str:
        return '{"approve": false, "scope": null, "reasoning": "Sample too thin."}'

    from src.agent.lessons import reject_lesson as real_reject_lesson

    def flaky_reject_lesson(conn, lesson_id, reviewer):
        if lesson_id == id1:
            raise RuntimeError("write failed")
        return real_reject_lesson(conn, lesson_id, reviewer)

    with patch("app.backend.live_lessons.reject_lesson", side_effect=flaky_reject_lesson):
        auto_judge_live_lessons(dm, fake_invoke)

    with dm.connection(read_only=True) as conn:
        row1 = conn.execute("SELECT status, auto_decision_reasoning FROM agent_lessons WHERE id = ?", [id1]).fetchone()
        row2 = conn.execute("SELECT status, auto_decision_reasoning FROM agent_lessons WHERE id = ?", [id2]).fetchone()
    assert row1[0] == "pending"
    assert row1[1] is None  # untouched -- the write for id1 never got past reject_lesson raising
    assert row2[0] == "rejected"
    assert row2[1] == "Sample too thin."


def test_auto_judge_live_lessons_does_not_clobber_a_concurrent_human_decision(tmp_path: Path) -> None:
    """A human can run `agent-lessons approve/reject <id>` on a live-sourced
    row (the CLI applies to any row id, no source filter) at any point during
    auto_judge_live_lessons's judge/distill/conflict-check phase, which holds
    no DuckDB connection open and can run for a while. The write phase must
    re-check status right before writing and skip (not overwrite) a row a
    human already reviewed in the meantime."""
    dm = _dm(tmp_path)
    with dm.connection() as conn:
        create_lessons_tables(conn)
        lesson_id = insert_lesson_candidate(
            conn, "Live-sourced batch: pattern.", "E0", "competition_specific", "m1", source="live",
        )

    from src.agent.lessons import judge_lesson_candidate as real_judge_lesson_candidate

    def judge_then_human_intervenes(lesson_text, competition_id, tier, llm_invoke):
        decision = real_judge_lesson_candidate(lesson_text, competition_id, tier, llm_invoke)
        # Simulate a human approving this exact row via the CLI while the
        # job is still mid-flight (i.e. before this function's own write
        # phase runs).
        with dm.connection() as human_conn:
            approve_lesson(
                human_conn, lesson_id, "tier", reviewer="human-reviewer",
                rule_text="ALWAYS bet result_3way -- human call.",
            )
        return decision

    def fake_invoke(prompt: str) -> str:
        if "deciding whether to promote" in prompt:
            return '{"approve": false, "scope": null, "reasoning": "Sample too thin."}'
        raise AssertionError(f"unexpected prompt: {prompt}")

    with patch("app.backend.live_lessons.judge_lesson_candidate", side_effect=judge_then_human_intervenes):
        auto_judge_live_lessons(dm, fake_invoke)

    with dm.connection(read_only=True) as conn:
        row = conn.execute(
            "SELECT status, scope, rule_text, reviewer, auto_decision_reasoning FROM agent_lessons WHERE id = ?",
            [lesson_id],
        ).fetchone()
    assert row[0] == "approved"
    assert row[1] == "tier"
    assert row[2] == "ALWAYS bet result_3way -- human call."
    assert row[3] == "human-reviewer"
    assert row[4] is None  # auto_decision_reasoning untouched -- the job's write was skipped entirely
