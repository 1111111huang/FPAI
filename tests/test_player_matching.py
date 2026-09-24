"""Tests for the 4-layer player-name matcher (get_player_rating's identity
resolution): exact -> accent-fold -> surname -> difflib fuzzy fallback.
Ambiguous matches must resolve to None, never a guess."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agent.player_matching import match_player_name

_ROSTER = ["Kevin De Bruyne", "Erling Haaland", "Rodri", "Kylian Mbappé"]


def test_exact_match_case_insensitive():
    assert match_player_name("erling haaland", _ROSTER) == "Erling Haaland"


def test_accent_folded_match():
    assert match_player_name("Kylian Mbappe", _ROSTER) == "Kylian Mbappé"


def test_surname_only_match():
    assert match_player_name("De Bruyne", _ROSTER) == "Kevin De Bruyne"


def test_fuzzy_fallback_match():
    assert match_player_name("Erlin Haaland", _ROSTER) == "Erling Haaland"


def test_ambiguous_surname_returns_none_not_a_guess():
    roster = ["Kevin De Bruyne", "Bruno De Bruyne"]  # shared surname, contrived
    assert match_player_name("De Bruyne", roster) is None


def test_no_plausible_match_returns_none():
    assert match_player_name("Someone Totally Unrelated", _ROSTER) is None
