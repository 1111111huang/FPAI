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

import json
from pathlib import Path

from src.ingestion.common.team_mapping import TeamNameMapper, _token_containment_score, _tokenize


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
