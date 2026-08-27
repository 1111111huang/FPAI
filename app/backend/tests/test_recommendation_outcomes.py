"""W167: recommendation_outcomes storage + resolve_pending_recommendations,
mirroring test_settlement.py's own structure and cases."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import sys
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.football_data_client import NormalizedMatch
from app.backend.football_data_competition_codes import FOOTBALL_DATA_CODE_BY_LEAGUE
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcomeStore, resolve_pending_recommendations


def _match(match_id: str, home_goals: int | None, away_goals: int | None) -> NormalizedMatch:
    return NormalizedMatch(
        match_id=match_id, utc_date="2026-08-22T15:00:00Z", status="FINISHED",
        home_team="Arsenal", away_team="Everton", home_goals=home_goals, away_goals=away_goals,
    )


def _rec(overall: str, market: str, selection: str, recommendation_type: str, current_odds, value_edge=0.1, league="E0", confidence="medium") -> dict:
    return {
        "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": league},
        "overall": overall,
        "markets": [{
            "market": market, "selection": selection, "recommendation_type": recommendation_type,
            "current_odds": current_odds, "value_edge": value_edge,
        }],
        "confidence": confidence,
        "explanation": [], "limitations": [], "prediction_basis": "team_history_and_market",
    }


def test_resolves_a_won_direct_bet_pick(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0), "scheduled")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    resolved = resolve_pending_recommendations(cache, store, client)

    assert len(resolved) == 1
    assert resolved[0].correct is True
    assert resolved[0].market == "result_3way"
    assert store.list_all()[0].match_id == "m1"


def test_resolves_a_lost_pick(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "away", "direct_bet", 2.0), "scheduled")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    resolved = resolve_pending_recommendations(cache, store, client)

    assert resolved[0].correct is False


def test_skips_no_bet_recommendations(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("no_bet", "result_3way", "home", "no_bet", 2.0), "scheduled")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    resolved = resolve_pending_recommendations(cache, store, client)

    assert resolved == []
    client.get_results.assert_not_called()


def test_skips_unresolvable_markets(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "home_corners", "over_4.5", "direct_bet", 1.9), "scheduled")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    resolved = resolve_pending_recommendations(cache, store, client)

    assert resolved == []
    client.get_results.assert_not_called()


def test_persists_the_verified_competition_id_not_the_self_reported_league(tmp_path: Path) -> None:
    """W175: `competition` stays the LLM's self-reported string (existing,
    unchanged behavior) -- the new `competition_id` is the real code the
    results lookup actually matched against. Deliberately different values
    here to prove these are two independent columns, not aliases."""
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation(
        "m1", "2026-08-22", "hash1", {},
        _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0, league="a made-up league name"),
        "scheduled",
    )
    client = MagicMock()

    def fake_get_results(competition_code, date_from, date_to):
        return [_match("m1", 2, 1)] if competition_code == "PD" else []

    client.get_results.side_effect = fake_get_results

    resolved = resolve_pending_recommendations(cache, store, client)

    assert resolved[0].competition == "a made-up league name"
    assert resolved[0].competition_id == "SP1"


def test_persists_the_raw_final_score(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0), "scheduled")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    resolved = resolve_pending_recommendations(cache, store, client)

    assert resolved[0].home_goals == 2
    assert resolved[0].away_goals == 1


def test_persists_sweden_competition_id_via_the_sweden_client(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0), "scheduled")
    client = MagicMock()
    client.get_results.return_value = []
    sweden_client = MagicMock()
    sweden_client.get_results.return_value = [_match("m1", 2, 1)]

    resolved = resolve_pending_recommendations(cache, store, client, sweden_client)

    assert resolved[0].competition_id == "SWE"


def test_reopening_store_after_schema_migration_is_idempotent(tmp_path: Path) -> None:
    """A genuine pre-W175 table: the original 13-column DDL (no
    competition_id/home_goals/away_goals), with one real row inserted
    directly via sqlite3 -- then opened through RecommendationOutcomeStore,
    which must migrate it in place without raising and without losing the
    row, and tolerate being opened a second time after that."""
    db_path = tmp_path / "outcomes.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE recommendation_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT NOT NULL,
            date TEXT NOT NULL,
            competition TEXT,
            market TEXT NOT NULL,
            selection TEXT NOT NULL,
            recommendation_type TEXT NOT NULL,
            confidence TEXT,
            odds REAL,
            value_edge REAL,
            correct INTEGER NOT NULL,
            generated_at TEXT NOT NULL,
            resolved_at TEXT NOT NULL,
            UNIQUE(match_id, date)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO recommendation_outcomes
        (match_id, date, competition, market, selection, recommendation_type,
         confidence, odds, value_edge, correct, generated_at, resolved_at)
        VALUES ('old-m1', '2020-01-01', 'E0', 'result_3way', 'home', 'direct_bet',
                'medium', 2.0, 0.1, 1, '2020-01-01T00:00:00+00:00', '2020-01-01T00:00:00+00:00')
        """
    )
    conn.commit()
    conn.close()

    store = RecommendationOutcomeStore(db_path=db_path)  # must migrate in place, not raise

    rows = store.list_all()
    assert len(rows) == 1
    assert rows[0].match_id == "old-m1"
    assert rows[0].competition_id is None
    assert rows[0].home_goals is None
    assert rows[0].away_goals is None

    RecommendationOutcomeStore(db_path=db_path)  # second open must not raise


def test_skips_not_yet_finished_matches(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0), "scheduled")
    client = MagicMock()
    client.get_results.return_value = []  # not finished yet

    resolved = resolve_pending_recommendations(cache, store, client)

    assert resolved == []


def test_idempotent_rerun_does_not_duplicate(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0), "scheduled")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    first = resolve_pending_recommendations(cache, store, client)
    second = resolve_pending_recommendations(cache, store, client)

    assert len(first) == 1
    assert second == []
    assert len(store.list_all()) == 1


def test_resolves_a_non_epl_league_via_the_correct_competition_code(tmp_path: Path) -> None:
    """Deliberately does not trust the recommendation's self-reported
    match.league for routing -- merges results across every football-data.org
    competition code instead (same reasoning settlement.py already uses to
    merge EPL + Sweden results)."""
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0, league="SP1"), "scheduled")
    client = MagicMock()
    # Only the "PD" (La Liga) call returns the match; every other competition
    # code call returns nothing.
    client.get_results.side_effect = lambda competition_code, date_from, date_to: (
        [_match("m1", 2, 1)] if competition_code == "PD" else []
    )

    resolved = resolve_pending_recommendations(cache, store, client)

    assert len(resolved) == 1
    assert resolved[0].correct is True
    assert resolved[0].competition == "SP1"


def test_one_competitions_connection_error_does_not_abort_the_others(tmp_path: Path) -> None:
    """W178 regression: found live (2026-08-27) that a single competition's
    transient RemoteDisconnected propagated straight out of
    resolve_pending_recommendations, aborting the whole daily_live_lessons
    job -- every other league's candidates went unresolved too, not just
    the flaky one's."""
    import requests

    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0, league="SP1"), "scheduled")
    client = MagicMock()

    def _get_results(competition_code, date_from, date_to):
        if competition_code == "PL":
            raise requests.exceptions.ConnectionError("Remote end closed connection without response")
        return [_match("m1", 2, 1)] if competition_code == "PD" else []

    client.get_results.side_effect = _get_results

    resolved = resolve_pending_recommendations(cache, store, client)

    assert len(resolved) == 1
    assert resolved[0].correct is True
    # Every competition code was still tried -- PL's failure didn't short-circuit the loop.
    assert client.get_results.call_count == len(FOOTBALL_DATA_CODE_BY_LEAGUE)


def test_sweden_clients_connection_error_does_not_abort_football_data_resolution(tmp_path: Path) -> None:
    import requests

    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0), "scheduled")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]
    sweden_client = MagicMock()
    sweden_client.get_results.side_effect = requests.exceptions.ConnectionError("boom")

    resolved = resolve_pending_recommendations(cache, store, client, sweden_client=sweden_client)

    assert len(resolved) == 1
    assert resolved[0].correct is True


def test_groups_api_calls_by_date_not_per_candidate(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0), "scheduled")
    cache.record_generation("m2", "2026-08-22", "hash1", {}, _rec("direct_bet", "btts", "yes", "direct_bet", 1.8), "scheduled")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1), _match("m2", 1, 1)]

    resolved = resolve_pending_recommendations(cache, store, client)

    assert len(resolved) == 2
    # One call per football-data.org competition code for the single shared
    # date -- not one per candidate. FOOTBALL_DATA_CODE_BY_LEAGUE has 5
    # entries, so 5 calls total for 2 candidates on the same date, not 10.
    assert client.get_results.call_count == len(FOOTBALL_DATA_CODE_BY_LEAGUE)


def test_list_unbatched_for_lessons_excludes_already_batched(tmp_path: Path) -> None:
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    o1 = store.insert(
        match_id="m1", date="2026-08-22", competition="E0", market="result_3way", selection="home",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2026-08-22T10:00:00+00:00", competition_id="E0", home_goals=2, away_goals=1,
    )
    store.insert(
        match_id="m2", date="2026-08-22", competition="E0", market="result_3way", selection="away",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=False,
        generated_at="2026-08-22T10:00:00+00:00", competition_id="E0", home_goals=0, away_goals=1,
    )

    store.mark_lesson_batched([o1.id])

    unbatched = store.list_unbatched_for_lessons()
    assert [o.match_id for o in unbatched] == ["m2"]


def test_mark_lesson_batched_handles_multiple_ids_in_one_call(tmp_path: Path) -> None:
    """The realistic production case (a whole day's batch of several
    matches) -- exercises the comma-joined placeholder-building logic,
    unlike the other tests here which only ever pass a single id."""
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    outcomes = [
        store.insert(
            match_id=f"m{i}", date="2026-08-22", competition="E0", market="result_3way", selection="home",
            recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
            generated_at="2026-08-22T10:00:00+00:00", competition_id="E0", home_goals=2, away_goals=1,
        )
        for i in range(3)
    ]
    still_unbatched = store.insert(
        match_id="m-unbatched", date="2026-08-22", competition="E0", market="result_3way", selection="away",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=False,
        generated_at="2026-08-22T10:00:00+00:00", competition_id="E0", home_goals=0, away_goals=1,
    )

    store.mark_lesson_batched([o.id for o in outcomes])

    unbatched = store.list_unbatched_for_lessons()
    assert [o.match_id for o in unbatched] == [still_unbatched.match_id]


def test_mark_lesson_batched_is_a_noop_for_an_empty_list(tmp_path: Path) -> None:
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.mark_lesson_batched([])  # must not raise


def test_list_unbatched_for_lessons_has_no_date_filter(tmp_path: Path) -> None:
    """Unlike list_all(since=...) -- a prior run could have resolved an
    outcome it never got to batch (e.g. a crash between steps), and that
    must still surface here no matter how old it is."""
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="old", date="2020-01-01", competition="E0", market="result_3way", selection="home",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2020-01-01T10:00:00+00:00", competition_id="E0", home_goals=1, away_goals=0,
    )
    assert len(store.list_unbatched_for_lessons()) == 1


def test_uses_sweden_client_when_provided(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("sw1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0, league="SWE"), "scheduled")
    client = MagicMock()
    client.get_results.return_value = []
    sweden_client = MagicMock()
    sweden_client.get_results.return_value = [
        NormalizedMatch(match_id="sw1", utc_date="2026-08-22T15:00:00Z", status="FINISHED", home_team="Malmo FF", away_team="AIK", home_goals=2, away_goals=1)
    ]

    resolved = resolve_pending_recommendations(cache, store, client, sweden_client=sweden_client)

    assert len(resolved) == 1
    assert resolved[0].correct is True
