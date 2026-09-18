# Odds Team-Name Token Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the recurring "Unmapped team" / silently-missing-odds bug class (BUG-057, W191, W192) by giving `eod_batch.py`'s and `t30_refresh.py`'s odds/fixtures join a real, candidate-based fuzzy match, so a new club-type-prefix spelling (the next "TSG Hoffenheim") resolves automatically instead of needing another manual `config/team_mapping.json` entry.

**Architecture:** `TeamNameMapper.suggest()`/`map_team()` (`src/ingestion/common/team_mapping.py`) gain an opt-in `use_token_match` parameter that adds token-containment scoring (e.g. `{"hoffenheim"} <= {"tsg", "hoffenheim"}`) on top of the existing Levenshtein fallback, plus an ambiguity guard so a tie between two equally-good candidates never gets silently resolved. `eod_batch.py`'s `odds_lookup()` and `t30_refresh.py`'s `refresh_match_at_t30()` are the only two callers that opt in, each building its own tightly-scoped candidate pool (that batch's fixtures, or a single fixture's two team names) from data already in hand. `fotmob`/`understat` merge.py's existing calls are untouched -- their season-spanning candidate pools make token-subset matching unsafe there (see Task 2's "GFC Ajaccio" test).

**Tech Stack:** Python, pytest.

**Design doc:** `docs/superpowers/specs/2026-09-17-odds-team-mapping-token-match-design.md`

---

### Task 1: Token-containment scoring helpers

**Files:**
- Modify: `src/ingestion/common/team_mapping.py`
- Test (create): `tests/test_team_mapping_token_match.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_team_mapping_token_match.py`:

```python
"""W234: closes the structural gap W192 flagged -- `map_team()`'s two
no-candidate-pool call sites in eod_batch.py/t30_refresh.py never attempt a
fuzzy match at all, and even with a candidate pool wired in, plain Levenshtein
similarity can't bridge a club-type-prefix mismatch ("TSG Hoffenheim" vs
"Hoffenheim" scores 0.71, "1. FC Koln" vs "FC Koln" scores ~0.70, both below
`min_similarity=0.82`). This module tests the token-containment scoring added
to close that gap, opt-in via `use_token_match` so `fotmob`/`understat`
merge.py's existing season-spanning candidate pools keep their exact current
behavior (see `test_three_leagues_team_mapping.py`'s "GFC Ajaccio" case)."""

from __future__ import annotations

from src.ingestion.common.team_mapping import _token_containment_score, _tokenize


def test_tokenize_folds_accents_lowercases_and_strips_punctuation():
    assert _tokenize("TSG Hoffenheim") == {"tsg", "hoffenheim"}
    assert _tokenize("1. FC Köln") == {"1", "fc", "koln"}


def test_token_containment_score_equal_sets_scores_1():
    assert _token_containment_score("Le Havre", "Le Havre") == 1.0


def test_token_containment_score_strict_subset_scores_0_95():
    assert _token_containment_score("TSG Hoffenheim", "Hoffenheim") == 0.95
    assert _token_containment_score("Hoffenheim", "TSG Hoffenheim") == 0.95  # symmetric


def test_token_containment_score_shared_word_only_is_not_containment():
    """'Manchester United' and 'West Ham United' merely share one word --
    neither token set is a subset of the other, so this must fall back to
    Levenshtein (return None), not silently treat them as related."""
    assert _token_containment_score("Manchester United", "West Ham United") is None


def test_token_containment_score_empty_tokens_is_none():
    assert _token_containment_score("", "Hoffenheim") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_team_mapping_token_match.py -v`
Expected: FAIL with `ImportError: cannot import name '_token_containment_score'`

- [ ] **Step 3: Implement `_tokenize` and `_token_containment_score`**

In `src/ingestion/common/team_mapping.py`, add these two functions directly after `_similarity_score` (i.e. right before `class TeamNameMapper:`):

```python
def _tokenize(value: str) -> set[str]:
    """Lowercase, accent-folded, punctuation-stripped word set -- 'TSG
    Hoffenheim' -> {'tsg', 'hoffenheim'}. Used by `_token_containment_score`
    below, not by the exact/accent-fold lookup paths in `map_team()`, which
    stay untouched."""
    folded = _fold_accents(value).lower()
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in folded)
    return set(cleaned.split())


def _token_containment_score(left: str, right: str) -> float | None:
    """Token-containment score for `use_token_match=True` callers, or None
    if the two names don't relate this way (caller falls back to
    `_similarity_score`). Equal token sets (a pure word-order variant) score
    1.0; a genuine strict subset -- one side is the other plus extra words,
    e.g. a club-type prefix like 'TSG Hoffenheim' vs 'Hoffenheim' -- scores
    0.95. Ranking equal-set above strict-subset means an exact token match in
    a candidate list always outright wins over a merely-prefixed one, rather
    than tying with it under suggest()'s ambiguity guard. Two names that just
    happen to share one common word ('Manchester United' vs 'West Ham
    United') relate neither way -- neither full token set is a subset of the
    other -- so this returns None for them, same as any unrelated pair."""
    left_tokens, right_tokens = _tokenize(left), _tokenize(right)
    if not left_tokens or not right_tokens:
        return None
    if left_tokens == right_tokens:
        return 1.0
    if left_tokens <= right_tokens or right_tokens <= left_tokens:
        return 0.95
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_team_mapping_token_match.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/common/team_mapping.py tests/test_team_mapping_token_match.py
git commit -m "$(cat <<'EOF'
feat(ingestion): W234 -- add token-containment scoring to team_mapping

New _tokenize/_token_containment_score helpers, not yet wired into
suggest()/map_team() -- next task.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Wire `use_token_match` into `suggest()`/`map_team()`, plus an ambiguity guard

**Files:**
- Modify: `src/ingestion/common/team_mapping.py:138-198` (the `map_team`/`suggest` methods)
- Test (extend): `tests/test_team_mapping_token_match.py`

- [ ] **Step 1: Write the failing tests**

First, add these three lines to `tests/test_team_mapping_token_match.py`'s existing import block at the top of the file (alongside the `from src.ingestion.common.team_mapping import ...` line Task 1 added):

```python
import json
from pathlib import Path

from src.ingestion.common.team_mapping import TeamNameMapper
```

Then append the following to the end of `tests/test_team_mapping_token_match.py`:

```python


def _write_mapping(tmp_path: Path, mapping: dict[str, str]) -> str:
    path = tmp_path / "team_mapping.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return str(path)


def test_suggest_with_token_match_scores_prefixed_candidate_above_threshold(tmp_path: Path) -> None:
    mapper = TeamNameMapper(mapping_path=_write_mapping(tmp_path, {}))
    suggestion, score = mapper.suggest("TSG Hoffenheim", ["Hoffenheim", "Freiburg"], use_token_match=True)
    assert suggestion == "Hoffenheim"
    assert score == 0.95


def test_suggest_without_token_match_flag_stays_below_threshold(tmp_path: Path) -> None:
    """Confirms the real-world number from the design doc: plain Levenshtein
    alone can't bridge this, which is exactly why this change exists."""
    mapper = TeamNameMapper(mapping_path=_write_mapping(tmp_path, {}))
    suggestion, score = mapper.suggest("TSG Hoffenheim", ["Hoffenheim", "Freiburg"])
    assert score < mapper.min_similarity


def test_map_team_resolves_prefixed_club_via_token_match_with_candidates(tmp_path: Path) -> None:
    """The actual BUG-057-class case: a club-type-prefixed odds spelling with
    no direct entry in the mapping file, resolved purely via a candidate pool
    that happens to include the bare canonical name -- no manual mapping
    entry needed."""
    mapper = TeamNameMapper(mapping_path=_write_mapping(tmp_path, {"Hoffenheim": "Hoffenheim"}))
    resolved = mapper.map_team("TSG Hoffenheim", ["Hoffenheim", "Freiburg"], use_token_match=True)
    assert resolved == "Hoffenheim"


def test_map_team_without_use_token_match_leaves_prefixed_name_unmapped(tmp_path: Path) -> None:
    """Control: the same case, without opting in, behaves exactly as it does
    today (falls through to the "closest match below threshold" branch,
    returns the name unchanged)."""
    mapper = TeamNameMapper(mapping_path=_write_mapping(tmp_path, {"Hoffenheim": "Hoffenheim"}))
    resolved = mapper.map_team("TSG Hoffenheim", ["Hoffenheim", "Freiburg"])
    assert resolved == "TSG Hoffenheim"


def test_suggest_ambiguous_token_match_returns_no_suggestion(tmp_path: Path) -> None:
    """Two candidates both strictly contain the input's tokens -- e.g. two
    'Real ...' clubs sharing the word 'Real' -- must not silently pick
    whichever the loop reaches first."""
    mapper = TeamNameMapper(mapping_path=_write_mapping(tmp_path, {}))
    suggestion, score = mapper.suggest("Real", ["Real Madrid", "Real Sociedad"], use_token_match=True)
    assert suggestion is None


def test_map_team_ambiguous_token_match_falls_through_to_unmapped_warning(tmp_path: Path) -> None:
    mapper = TeamNameMapper(mapping_path=_write_mapping(tmp_path, {}))
    resolved = mapper.map_team("Real", ["Real Madrid", "Real Sociedad"], use_token_match=True)
    assert resolved == "Real"


def test_gfc_ajaccio_style_case_unaffected_by_default(tmp_path: Path) -> None:
    """Regression guard for the real documented case
    (tests/test_three_leagues_team_mapping.py): a genuinely different, older
    club whose name is a strict token superset of a later, unrelated club's
    name must not auto-merge when a caller doesn't opt in to token matching
    -- i.e. fotmob/understat's existing calls (which never pass
    use_token_match=True) are provably unaffected by this change."""
    mapper = TeamNameMapper(mapping_path=_write_mapping(tmp_path, {"Ajaccio": "Ajaccio"}))
    resolved = mapper.map_team("GFC Ajaccio", ["Ajaccio"])
    assert resolved == "GFC Ajaccio"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_team_mapping_token_match.py -v`
Expected: FAIL -- `TypeError: suggest() got an unexpected keyword argument 'use_token_match'` (and `test_suggest_with_token_match_scores_prefixed_candidate_above_threshold` etc. fail with it; the other pre-existing tests from Task 1 still pass).

- [ ] **Step 3: Implement `use_token_match` and the ambiguity guard**

In `src/ingestion/common/team_mapping.py`, replace the entire `map_team` method:

```python
    def map_team(
        self,
        team_name: str,
        candidates: Iterable[str] | None = None,
        use_token_match: bool = False,
    ) -> str:
        """Map a team name using explicit mappings or a fuzzy fallback."""
        normalized = " ".join(str(team_name).strip().split())
        if not normalized:
            return normalized
        if normalized in self.mapping:
            return self.mapping[normalized]

        folded = _fold_accents(normalized)
        if folded in self._folded_mapping:
            return self._folded_mapping[folded]

        if candidates is None:
            LOGGER.warning(
                "Unmapped team '%s'. Add mapping to %s.",
                normalized,
                self.mapping_path,
            )
            return normalized

        suggestion, score = self.suggest(normalized, candidates, use_token_match=use_token_match)
        if suggestion is None:
            LOGGER.warning(
                "Unmapped team '%s'. Add mapping to %s.",
                normalized,
                self.mapping_path,
            )
            return normalized

        if score >= self.min_similarity:
            LOGGER.warning(
                "Unmapped team '%s'. Using fuzzy match '%s' (score=%.2f). "
                "Add mapping to %s.",
                normalized,
                suggestion,
                score,
                self.mapping_path,
            )
            return suggestion

        LOGGER.warning(
            "Unmapped team '%s'. Closest match '%s' (score=%.2f). "
            "Add mapping to %s.",
            normalized,
            suggestion,
            score,
            self.mapping_path,
        )
        return normalized
```

(This is identical to today's `map_team` except the new `use_token_match` parameter and passing it through to `self.suggest(...)`.)

Then replace the entire `suggest` method:

```python
    def suggest(
        self, team_name: str, candidates: Iterable[str], use_token_match: bool = False
    ) -> tuple[str | None, float]:
        """Suggest the closest mapping candidate for a new team name.

        `use_token_match=True` (opt-in) additionally scores each candidate by
        `_token_containment_score` before falling back to Levenshtein --
        closes the club-type-prefix gap ("TSG Hoffenheim" vs "Hoffenheim")
        plain Levenshtein can't bridge (0.71 similarity, below the 0.82
        threshold). Left False (the default) for every pre-existing caller
        (fotmob/understat merge.py), whose candidate pools span whole-league
        history and can contain genuinely distinct, differently-named clubs a
        token-subset match would wrongly conflate (see
        tests/test_three_leagues_team_mapping.py's "GFC Ajaccio" case).

        If two or more candidates tie for the best score, returns (None,
        best_score) rather than guessing one -- more likely to matter once
        token matching makes an exact-token tie between two real candidates
        possible (e.g. two "Real ..." clubs sharing the word "Real")."""
        best_name: str | None = None
        best_score = -1.0
        tied = False
        for candidate in candidates:
            candidate_name = standardize_team_name(str(candidate))
            token_score = _token_containment_score(team_name, candidate_name) if use_token_match else None
            score = token_score if token_score is not None else _similarity_score(team_name, candidate_name)
            if score > best_score:
                best_score = score
                best_name = candidate_name
                tied = False
            elif score == best_score:
                tied = True
        if tied and best_score >= self.min_similarity:
            return None, best_score
        return best_name, best_score
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_team_mapping_token_match.py -v`
Expected: PASS (12 passed)

- [ ] **Step 5: Run the full existing team-mapping test suite to confirm no regressions**

Run: `pytest tests/test_la_liga_team_mapping.py tests/test_sweden_team_mapping.py tests/test_three_leagues_team_mapping.py tests/test_build_for_match_team_mapping_logging.py app/backend/tests/test_la_liga_football_data_team_mapping.py app/backend/tests/test_bundesliga_odds_team_mapping.py app/backend/tests/test_new_leagues_football_data_team_mapping.py app/backend/tests/test_sweden_odds_team_mapping.py app/backend/tests/test_new_leagues_odds_team_mapping.py -v`
Expected: PASS, same counts as before this task (no test in this list passes `use_token_match=True`, so none of them can be affected by it).

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/common/team_mapping.py tests/test_team_mapping_token_match.py
git commit -m "$(cat <<'EOF'
feat(ingestion): W234 -- opt-in token-containment matching in suggest()/map_team()

use_token_match=False by default, so fotmob/understat merge.py's existing
season-spanning candidate pools are byte-for-byte unaffected (see the
GFC Ajaccio regression test). Also adds an ambiguity guard to suggest():
a tie between two top-scoring candidates now returns no suggestion instead
of silently picking one.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Wire a candidate pool into `eod_batch.py`'s odds/fixtures join

**Files:**
- Modify: `app/backend/eod_batch.py:37` (imports), `:72-74` (`odds_lookup`), `:269-279` (`run_eod_batch`'s odds-fetching block)
- Test (extend): `app/backend/tests/test_eod_batch.py`

- [ ] **Step 1: Write the failing test**

Append to `app/backend/tests/test_eod_batch.py`:

```python
def test_odds_matched_via_token_containment_for_club_type_prefix_mismatch(tmp_path: Path) -> None:
    """W234: a club-type-prefix mismatch ('FC Testopolis' vs the fixture's
    bare 'Testopolis') that plain exact/accent-fold lookup can't bridge --
    the documented recurring class behind BUG-057/W191/W192 (TSG Hoffenheim,
    1. FC Koln, ...). Token-containment matching against this batch's own
    fixture-derived candidate pool must resolve it without a manual mapping
    entry. Invented team names, not real clubs, so this doesn't depend on
    config/team_mapping.json's current (or future) real content."""
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [_fixture("m1", "Testopolis", "Rivertown")]
    odds_client = MagicMock()
    odds_client.get_odds.return_value = [
        NormalizedOdds(
            home_team="FC Testopolis", away_team="Rivertown", commence_time="2026-08-22T15:00:00Z",
            home_odds=1.9, draw_odds=3.4, away_odds=4.2,
        ),
    ]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    captured_match_info = {}

    def _capture(match_info, config):
        captured_match_info.update(match_info)
        return _RECOMMENDATION

    with patch("app.backend.recommendations.run_agent", side_effect=_capture):
        asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda f: None, date_str=_future_date(1),
            )
        )

    assert captured_match_info["odds"] == {"home": 1.9, "draw": 3.4, "away": 4.2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest app/backend/tests/test_eod_batch.py::test_odds_matched_via_token_containment_for_club_type_prefix_mismatch -v`
Expected: FAIL -- `assert {} == {'home': 1.9, ...}` (`captured_match_info` has no `"odds"` key: the odds side "FC Testopolis" never joins the fixture side "Testopolis" today).

- [ ] **Step 3: Add `Iterable` to the typing import**

In `app/backend/eod_batch.py`, change:

```python
from typing import Callable
```

to:

```python
from typing import Callable, Iterable
```

- [ ] **Step 4: Add a `candidates` parameter to `odds_lookup`**

Replace:

```python
def odds_lookup(odds_events: list[NormalizedOdds]) -> dict[tuple[str, str], NormalizedOdds]:
    mapper = TeamNameMapper(mapping_path=str(_TEAM_MAPPING_PATH))
    return {(mapper.map_team(o.home_team), mapper.map_team(o.away_team)): o for o in odds_events}
```

with:

```python
def odds_lookup(
    odds_events: list[NormalizedOdds], candidates: Iterable[str] | None = None
) -> dict[tuple[str, str], NormalizedOdds]:
    """W234: `candidates`, when supplied, lets a club-type-prefixed odds-side
    spelling ("TSG Hoffenheim") resolve via token-containment matching
    against the batch's own fixture-derived canonical names, instead of
    needing a manual config/team_mapping.json entry for every new instance
    of the same recurring pattern (BUG-057/W191/W192)."""
    mapper = TeamNameMapper(mapping_path=str(_TEAM_MAPPING_PATH))
    return {
        (
            mapper.map_team(o.home_team, candidates, use_token_match=True),
            mapper.map_team(o.away_team, candidates, use_token_match=True),
        ): o
        for o in odds_events
    }
```

- [ ] **Step 5: Build and pass the fixture-derived candidate pool in `run_eod_batch`**

Replace:

```python
    fixture_dates = {_fixture_date(fixture) for fixture in fixtures}
    sport_key = ODDS_SPORT_KEY_BY_COMPETITION[league]
    if odds_client is None:
        odds_by_teams_by_date: dict[str, dict] = {}
    elif fixture_dates <= {date_str}:
        odds_by_teams_by_date = {date_str: odds_lookup(odds_client.get_odds(sport_key=sport_key) or [])}
    else:
        odds_by_teams_by_date = {
            fixture_date: odds_lookup(odds_client.get_odds(sport_key=sport_key, date=fixture_date) or [])
            for fixture_date in fixture_dates
        }
```

with:

```python
    fixture_dates = {_fixture_date(fixture) for fixture in fixtures}
    sport_key = ODDS_SPORT_KEY_BY_COMPETITION[league]
    if odds_client is None:
        odds_by_teams_by_date: dict[str, dict] = {}
    else:
        # W234: this batch's own fixture team names (already resolved to
        # their canonical form) become the candidate pool for the odds
        # side's own mapping -- lets odds_lookup() bridge a club-type-prefix
        # mismatch (e.g. "TSG Hoffenheim") that has no direct
        # config/team_mapping.json entry yet.
        fixture_mapper = TeamNameMapper(mapping_path=str(_TEAM_MAPPING_PATH))
        fixture_team_candidates = sorted({
            fixture_mapper.map_team(name)
            for fixture in fixtures
            for name in (fixture.home_team, fixture.away_team)
        })
        if fixture_dates <= {date_str}:
            odds_by_teams_by_date = {
                date_str: odds_lookup(odds_client.get_odds(sport_key=sport_key) or [], fixture_team_candidates)
            }
        else:
            odds_by_teams_by_date = {
                fixture_date: odds_lookup(
                    odds_client.get_odds(sport_key=sport_key, date=fixture_date) or [], fixture_team_candidates
                )
                for fixture_date in fixture_dates
            }
```

- [ ] **Step 6: Run the new test to verify it passes**

Run: `pytest app/backend/tests/test_eod_batch.py::test_odds_matched_via_token_containment_for_club_type_prefix_mismatch -v`
Expected: PASS

- [ ] **Step 7: Run the full `test_eod_batch.py` suite to confirm no regressions**

Run: `pytest app/backend/tests/test_eod_batch.py -v`
Expected: PASS, same count as before this task plus the 1 new test (`odds_lookup`'s `candidates` parameter defaults to `None`, so every existing direct caller and test is unaffected).

- [ ] **Step 8: Commit**

```bash
git add app/backend/eod_batch.py app/backend/tests/test_eod_batch.py
git commit -m "$(cat <<'EOF'
fix(app): W234 -- eod_batch odds/fixtures join resolves prefixed club names

odds_lookup() now accepts a candidate pool (this batch's own fixture team
names) and opts in to token-containment matching, so a club-type-prefixed
odds-side spelling (the next "TSG Hoffenheim") joins its fixture and
attaches real odds without needing a manual config/team_mapping.json entry.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Wire the same candidate pool into `t30_refresh.py`

**Files:**
- Modify: `app/backend/t30_refresh.py:28-40` (imports), `:98` (the `odds_lookup` call inside `refresh_match_at_t30`)
- Test (extend): `app/backend/tests/test_t30_refresh.py`

- [ ] **Step 1: Write the failing test**

Append to `app/backend/tests/test_t30_refresh.py`:

```python
def test_odds_matched_via_token_containment_for_club_type_prefix_mismatch(tmp_path: Path) -> None:
    """W234, mirroring the eod_batch.py regression test: T-30's own
    single-fixture candidate pool must also resolve a club-type-prefix
    mismatch via token-containment matching."""
    config = AgentConfig.default()
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    fixture = _fixture(home="Testopolis", away="Rivertown")
    odds_client = MagicMock()
    odds_client.get_odds.return_value = [
        NormalizedOdds(
            home_team="FC Testopolis", away_team="Rivertown", commence_time="2026-08-22T15:00:00Z",
            home_odds=1.9, draw_odds=3.4, away_odds=4.2,
        ),
    ]

    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION) as mock_run_agent:
        result = refresh_match_at_t30(
            fixture, odds_client=odds_client, cache=cache, config=config, date_str=_future_date(1)
        )

    mock_run_agent.assert_called_once()
    assert result.outcome == "refreshed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest app/backend/tests/test_t30_refresh.py::test_odds_matched_via_token_containment_for_club_type_prefix_mismatch -v`
Expected: FAIL -- `assert 'skipped_no_odds' == 'refreshed'` (the odds side "FC Testopolis" never joins fixture side "Testopolis" today, so `match_odds` returns `None`).

- [ ] **Step 3: Import `TeamNameMapper` and `_TEAM_MAPPING_PATH`**

In `app/backend/t30_refresh.py`, replace:

```python
from app.backend.eod_batch import (
    LEAGUE_CODE, add_secondary_odds, has_kicked_off, match_odds, odds_lookup,
)
```

with:

```python
from app.backend.eod_batch import (
    LEAGUE_CODE, _TEAM_MAPPING_PATH, add_secondary_odds, has_kicked_off, match_odds, odds_lookup,
)
```

And add, alongside the other `src.` imports a few lines below:

```python
from src.ingestion.common.team_mapping import TeamNameMapper
```

- [ ] **Step 4: Build and pass a per-fixture candidate pool**

In `refresh_match_at_t30`, replace:

```python
    odds_by_teams = odds_lookup(odds_events)
    fresh_odds = match_odds(fixture, odds_by_teams)
```

with:

```python
    # W234: a tight 2-name candidate pool (just this fixture's own home/away)
    # lets odds_lookup() resolve a club-type-prefixed odds-side spelling via
    # token-containment matching -- even lower collision risk than
    # eod_batch.py's whole-batch pool, since there are only ever two names to
    # choose between.
    fixture_mapper = TeamNameMapper(mapping_path=str(_TEAM_MAPPING_PATH))
    fixture_candidates = [fixture_mapper.map_team(fixture.home_team), fixture_mapper.map_team(fixture.away_team)]
    odds_by_teams = odds_lookup(odds_events, fixture_candidates)
    fresh_odds = match_odds(fixture, odds_by_teams)
```

- [ ] **Step 5: Run the new test to verify it passes**

Run: `pytest app/backend/tests/test_t30_refresh.py::test_odds_matched_via_token_containment_for_club_type_prefix_mismatch -v`
Expected: PASS

- [ ] **Step 6: Run the full `test_t30_refresh.py` suite to confirm no regressions**

Run: `pytest app/backend/tests/test_t30_refresh.py -v`
Expected: PASS, same count as before this task plus the 1 new test.

- [ ] **Step 7: Commit**

```bash
git add app/backend/t30_refresh.py app/backend/tests/test_t30_refresh.py
git commit -m "$(cat <<'EOF'
fix(app): W234 -- T-30 refresh odds join resolves prefixed club names

Same fix as eod_batch.py's odds_lookup() wiring, applied to
refresh_match_at_t30's own single-fixture odds join.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Full-suite verification and documentation

**Files:**
- Modify: `documents/app_user_stories.md` (W234 row, PHASE 46)

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ app/backend/tests/ -v 2>&1 | tail -60`
Expected: PASS, with the same pre-existing failures/skips already documented in W233's completion notes (5 pre-existing, unrelated `test_fixtures_endpoint.py` failures, 1 pre-existing unrelated `BetTracker.race.test.tsx` failure -- that one's a frontend test file and won't be collected by `pytest` at all, just noting it's not a new regression if seen elsewhere) -- plus this plan's 19 new tests (5 + 7 in `tests/test_team_mapping_token_match.py`, 1 in `test_eod_batch.py`, 1 in `test_t30_refresh.py`, plus whatever the full team-mapping regression run in Task 2 Step 5 already covered) all passing, and no other change in pass/fail counts.

- [ ] **Step 2: Update the W234 row in `documents/app_user_stories.md` to completed**

Find the row (PHASE 46 table):

```
| W234 | planned | **Close the structural gap W192 flagged but didn't build:
```

Change `planned` to `completed`, and append a new sentence to the end of the Comments cell (after the existing "**Design notes (2026-09-17):**" text), starting `**Completion notes (2026-09-17):**`, summarizing: the two-tier token-containment score (1.0 equal-set / 0.95 strict-subset) added as `_token_containment_score`/`_tokenize`; the opt-in `use_token_match` parameter on `suggest()`/`map_team()` (confirmed via the GFC Ajaccio regression test that fotmob/understat merge.py's existing calls are unaffected); the ambiguity guard added to `suggest()`; `eod_batch.py`'s `odds_lookup()`/`run_eod_batch()` and `t30_refresh.py`'s `refresh_match_at_t30()` wired to pass a fixture-derived candidate pool with `use_token_match=True`; and the full suite pass count from Step 1.

- [ ] **Step 3: Commit the documentation update**

```bash
git add documents/app_user_stories.md
git commit -m "$(cat <<'EOF'
docs: W234 -- mark odds team-mapping token-match complete

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```
