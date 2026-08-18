"""W11: recommendation caching layer. Not a reuse of SnapshotStore (purpose-
built for backtest determinism, keyed by tool-call SHA-256) -- this is a new
store keyed by (match_id, date, agent_config_hash), append-only so a
lightweight generation history survives alongside the latest entry, and
each row records the odds it was generated against so a future consumer
(W10) can cheaply detect "no new data" before deciding whether to
regenerate."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.agent_config_hash import compute_agent_config_hash
from app.backend.recommendation_cache import RecommendationCache
from src.agent.agent_config import AgentConfig


def _config(**overrides) -> AgentConfig:
    defaults = dict(
        model="llama3.1:8b", provider="ollama", temperature=0.1, max_tool_calls=10,
        min_odds_threshold=1.2, max_odds_threshold=11.0, min_conditional_odds_threshold=1.5, min_value_edge=0.05,
        markets=["result_3way"], system_prompt_version="v1",
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


def test_get_latest_returns_none_when_nothing_cached(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    assert cache.get_latest("m1", "2026-08-22", "cfg-hash") is None


def test_record_and_retrieve_a_generation(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    odds = {"home": 1.5, "draw": 4.0, "away": 6.0}
    rec = {"overall": "no_bet", "markets": []}

    cache.record_generation(
        match_id="m1", date="2026-08-22", agent_config_hash="cfg-hash",
        odds=odds, recommendation=rec, triggered_by="manual_regenerate",
    )
    entry = cache.get_latest("m1", "2026-08-22", "cfg-hash")

    assert entry is not None
    assert entry.odds == odds
    assert entry.recommendation == rec
    assert entry.triggered_by == "manual_regenerate"


def test_get_latest_returns_the_most_recent_generation(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    cache.record_generation(
        match_id="m1", date="2026-08-22", agent_config_hash="cfg-hash",
        odds={"home": 1.5, "draw": 4.0, "away": 6.0}, recommendation={"overall": "no_bet"},
        triggered_by="scheduled", generated_at="2026-08-21T23:00:00Z",
    )
    cache.record_generation(
        match_id="m1", date="2026-08-22", agent_config_hash="cfg-hash",
        odds={"home": 1.6, "draw": 3.9, "away": 5.5}, recommendation={"overall": "direct_bet"},
        triggered_by="scheduled", generated_at="2026-08-22T21:30:00Z",
    )

    entry = cache.get_latest("m1", "2026-08-22", "cfg-hash")

    assert entry.recommendation == {"overall": "direct_bet"}
    assert entry.odds == {"home": 1.6, "draw": 3.9, "away": 5.5}


def test_history_keeps_every_generation_not_just_the_latest(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    cache.record_generation(
        match_id="m1", date="2026-08-22", agent_config_hash="cfg-hash",
        odds={"home": 1.5, "draw": 4.0, "away": 6.0}, recommendation={"overall": "no_bet"},
        triggered_by="scheduled",
    )
    cache.record_generation(
        match_id="m1", date="2026-08-22", agent_config_hash="cfg-hash",
        odds={"home": 1.6, "draw": 3.9, "away": 5.5}, recommendation={"overall": "direct_bet"},
        triggered_by="scheduled",
    )

    history = cache.get_history("m1", "2026-08-22", "cfg-hash")

    assert len(history) == 2
    assert [h.recommendation["overall"] for h in history] == ["no_bet", "direct_bet"]


def test_different_agent_config_hash_is_a_different_cache_entry(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    cache.record_generation(
        match_id="m1", date="2026-08-22", agent_config_hash="cfg-a",
        odds={"home": 1.5, "draw": 4.0, "away": 6.0}, recommendation={"overall": "no_bet"},
        triggered_by="scheduled",
    )
    assert cache.get_latest("m1", "2026-08-22", "cfg-b") is None
    assert cache.get_latest("m1", "2026-08-22", "cfg-a") is not None


# --- A65/A66 follow-up: get_latest_any_config() -- a fallback for when a
# config change (busts every match's agent_config_hash at once) coincides
# with regeneration under the new config failing, leaving a still-good
# prior recommendation unreachable via get_latest()'s exact-hash lookup. ---

def test_get_latest_any_config_returns_none_when_nothing_cached(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    assert cache.get_latest_any_config("m1", "2026-08-22") is None


def test_get_latest_any_config_finds_an_entry_under_a_different_hash(tmp_path: Path) -> None:
    """The exact scenario: a match was generated fine under cfg-old, then a
    config change bumps the hash to cfg-new, and regeneration under
    cfg-new fails (or hasn't run yet) -- get_latest("cfg-new") sees
    nothing, but get_latest_any_config() still finds the old-config row."""
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    cache.record_generation(
        match_id="m1", date="2026-08-22", agent_config_hash="cfg-old",
        odds={"home": 1.5, "draw": 4.0, "away": 6.0}, recommendation={"overall": "direct_bet"},
        triggered_by="scheduled",
    )

    assert cache.get_latest("m1", "2026-08-22", "cfg-new") is None
    entry = cache.get_latest_any_config("m1", "2026-08-22")
    assert entry is not None
    assert entry.agent_config_hash == "cfg-old"
    assert entry.recommendation == {"overall": "direct_bet"}


def test_get_latest_any_config_returns_the_most_recent_across_all_hashes(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    cache.record_generation(
        match_id="m1", date="2026-08-22", agent_config_hash="cfg-old",
        odds={"home": 1.5, "draw": 4.0, "away": 6.0}, recommendation={"overall": "no_bet"},
        triggered_by="scheduled", generated_at="2026-08-21T23:00:00Z",
    )
    cache.record_generation(
        match_id="m1", date="2026-08-22", agent_config_hash="cfg-new",
        odds={"home": 1.6, "draw": 3.9, "away": 5.5}, recommendation={"overall": "direct_bet"},
        triggered_by="scheduled", generated_at="2026-08-22T21:30:00Z",
    )

    entry = cache.get_latest_any_config("m1", "2026-08-22")

    assert entry.recommendation == {"overall": "direct_bet"}
    assert entry.agent_config_hash == "cfg-new"


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    """SQLite-backed -- a restart shouldn't lose the cache."""
    db_path = tmp_path / "cache.db"
    RecommendationCache(db_path=db_path).record_generation(
        match_id="m1", date="2026-08-22", agent_config_hash="cfg-hash",
        odds={"home": 1.5, "draw": 4.0, "away": 6.0}, recommendation={"overall": "no_bet"},
        triggered_by="scheduled",
    )
    reopened = RecommendationCache(db_path=db_path)
    assert reopened.get_latest("m1", "2026-08-22", "cfg-hash") is not None


def test_agent_config_hash_is_stable_for_identical_config() -> None:
    assert compute_agent_config_hash(_config()) == compute_agent_config_hash(_config())


def test_agent_config_hash_changes_when_a_threshold_changes() -> None:
    h1 = compute_agent_config_hash(_config())
    h2 = compute_agent_config_hash(_config(min_odds_threshold=1.5))
    assert h1 != h2
