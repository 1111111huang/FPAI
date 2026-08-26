"""W177: live_lessons.py -- adapts RecommendationOutcome rows into
BacktestRecord-shaped objects and batches them into agent_lessons
candidates via src/agent/lessons.py's existing, unmodified functions."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

import duckdb

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.football_data_client import NormalizedMatch
from app.backend.live_lessons import _to_lesson_record, generate_daily_lessons
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcome, RecommendationOutcomeStore
from src.agent.lessons import create_lessons_tables


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
