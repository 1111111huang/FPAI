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
