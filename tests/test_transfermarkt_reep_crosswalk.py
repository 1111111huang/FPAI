"""Tests for loading the REEP FotMob<->Transfermarkt player identity crosswalk."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.transfermarkt.reep_crosswalk import load_reep_crosswalk

_CSV = (
    "reep_id,type,name,key_fotmob,key_transfermarkt\n"
    "reep_p1,player,Aaron Doran,162549,96148\n"
    "reep_p2,player,No Transfermarkt Player,555555,\n"
    "reep_p3,manager,Some Manager,666666,777777\n"
)


def _mock_resp(text: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.text = text
    return resp


def test_load_reep_crosswalk_keeps_only_players_with_both_ids():
    with patch("src.ingestion.transfermarkt.reep_crosswalk.requests.get", return_value=_mock_resp(_CSV)):
        crosswalk = load_reep_crosswalk()

    assert list(crosswalk.columns) == ["fotmob_player_id", "transfermarkt_player_id"]
    assert crosswalk.to_dict("records") == [{"fotmob_player_id": 162549, "transfermarkt_player_id": 96148}]
