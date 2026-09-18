# Odds Team-Name Token Matching -- Design

**Date:** 2026-09-17
**Status:** Approved, pending implementation plan
**Related:** BUG-057, W59, W78 (`documents/bugs.md`, `documents/app_user_stories.md` PHASE 46)

## Problem

`TeamNameMapper.map_team()` (`src/ingestion/common/team_mapping.py`) already has a
fuzzy-match fallback (accent-folding + Levenshtein similarity), but it only runs
when the caller passes a `candidates` pool. Two real call sites never do:
`eod_batch.py`'s `odds_lookup()`/`matched_odds_event()` and
`t30_refresh.py`'s `refresh_match_at_t30()`. Every miss there goes straight to
"Unmapped team '...'. Add mapping to config/team_mapping.json." with no attempt
to resolve it.

This has already produced three rounds of the same manual fix (W59 Sweden,
W78 La Liga shortNames, BUG-057 Bundesliga): The Odds API spells a club with
its club-type prefix ("TSG Hoffenheim", "1. FC Köln") while
`config/team_mapping.json`'s canonical short name doesn't ("Hoffenheim", "FC
Koln"). Each round was fixed by enumerating the specific instances found live
in Railway logs, not by closing the underlying gap -- so a new promoted/newly
covered club with the same prefix pattern will recreate the identical warning
and, more importantly, the identical silent join failure: the odds side keys
on the unmapped raw name, the fixture side keys on the canonical name, the two
dict keys never match, and that match's real odds never attach to its
recommendation.

Confirmed live: plain Levenshtein similarity is not sufficient to close this
even with a candidate pool wired in. `"TSG Hoffenheim"` vs `"Hoffenheim"`
scores 0.71 (4-char edit distance / 14 chars); `"1. FC Köln"` vs `"FC Koln"`
scores ~0.70. Both are below the mapper's existing 0.82 `min_similarity`
threshold. Prefix words push absolute edit distance up relative to short club
names, so the existing scoring function under-scores exactly the pattern that
keeps recurring.

## Scope

This spec covers **only** the odds/fixture join in `eod_batch.py` and
`t30_refresh.py` (the confirmed, live, recurring bug class). It does not touch
the other `map_team()` call sites with no candidate pool
(`forecast_service.py`, `feature_factory.py`, `recommendations.py`'s sandbox
lookup, `agent/schema.py`) -- those either don't have an obvious natural
candidate pool without further investigation, or are unconfirmed as actually
affected. Out of scope for this change; revisit separately if they turn up
the same warning live.

## Design

### 1. Token-containment matching (`src/ingestion/common/team_mapping.py`)

Two new module-level helpers:

- `_tokenize(value: str) -> set[str]` -- fold accents (reuse `_fold_accents`),
  lowercase, replace non-alphanumeric characters with spaces, split into a
  word set. `"TSG Hoffenheim"` -> `{"tsg", "hoffenheim"}`.
- `_token_subset_match(left: str, right: str) -> bool` -- true when one side's
  full token set is a non-empty subset of the other's. `{"hoffenheim"} <=
  {"tsg", "hoffenheim"}` -> match. `{"manchester", "united"} <= {"west",
  "ham", "united"}`? No (neither side is a full subset of the other) -> no
  match. This directional, whole-token-set requirement is what keeps it from
  false-matching two unrelated clubs that merely share one common word.

`TeamNameMapper.suggest()` scores each candidate by
`_token_subset_match` first (a fixed score, `0.95` -- confidently above
`min_similarity` but distinguishable in logs from a true exact/accent-fold
match) and falls back to the existing `_similarity_score` (Levenshtein)
otherwise. The rest of `suggest()`'s "keep the best-scoring candidate" loop
is unchanged.

### 2. Ambiguity guard (`suggest()`)

If two or more candidates tie for the best score, `suggest()` returns `(None,
best_score)` instead of silently keeping whichever the loop reached first.
Today this tie-break almost never matters (untuned Levenshtein ties are
coincidental); once token-subset scoring makes exact-token-match ties more
likely (e.g. two "Real ..." clubs live on the same La Liga matchday sharing
a token), guessing the first one is exactly the kind of silent wrong-merge
this file's own docstring (US#141) already flags as the real risk to avoid.
An ambiguous result falls through to `map_team()`'s existing "no suggestion"
warning branch -- no new log branch needed.

### 3. Wiring a candidate pool at the two call sites

- `odds_lookup(odds_events, candidates=None)` (`eod_batch.py`) -- new
  optional parameter, passed straight through to `mapper.map_team(name,
  candidates)` for both home and away.
- `run_eod_batch()` builds the candidate pool once per batch from that
  batch's own `fixtures`: each fixture's home/away team name, mapped through
  `map_team()` with no candidates (the reliable, already-working side of the
  join per BUG-057's own finding). Passed into both existing `odds_lookup(...)`
  call sites (the primary date and the fallback-window date).
- `t30_refresh.py`'s `refresh_match_at_t30()` builds a tight 2-name candidate
  pool from just its own single `fixture` (home + away, canonical) and passes
  it to its own `odds_lookup()` call -- even lower collision risk than the
  batch case, since there are only ever two names to choose between.
- `matched_odds_event()` / `match_odds()` / `add_secondary_odds()` are
  unchanged. The fixture side of the join already resolves correctly; only
  the odds side needed a candidate pool.

## Testing

- New unit tests for `_token_subset_match` (positive case, negative
  shared-word case, empty-set case) and the ambiguity guard (two tied
  candidates -> `None`).
- Regression test replaying the actual BUG-057 case: `"TSG Hoffenheim"`
  resolves to `"Hoffenheim"` given a Bundesliga fixture-derived candidate
  pool.
- Full existing `team_mapping` test suite plus `test_fotmob_merge.py` /
  `test_understat.py` (both already pass a `candidates` pool through the same
  shared `suggest()`) re-run to confirm no behavior change for their existing
  cases.
- Existing `eod_batch`/`t30_refresh` test suites re-run unchanged.

## Explicitly not doing

- Not lowering `min_similarity` or changing Levenshtein scoring itself --
  token-containment is a separate, additive scoring path.
- Not adding a prefix stoplist ("TSG", "1. FC", "VfL", ...) -- token-subset
  matching handles the general case without enumerating club-type words.
- Not touching the other three no-candidate `map_team()` call sites (see
  Scope).
- Not auto-writing resolved names back into `config/team_mapping.json` --
  resolution stays runtime-only, exactly as it already works for every
  existing fuzzy-match case today. A human still adds a permanent mapping
  entry if they want to silence the warning outright.
