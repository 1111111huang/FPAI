"""Loads REEP's (github.com/withqwerty/reep, CC0) FotMob<->Transfermarkt
player identity crosswalk -- the player-level equivalent of
config/team_mapping.json, which already solves this exact class of problem
for team names.

Known limitation (accepted per the design spec, same non-blocking-degrade
discipline BUG-057 already established for unmapped team names): REEP's
public snapshot is a point-in-time export, so this season's newest transfers
may be missing until REEP's own next refresh -- not a crash, just a smaller
crosswalk than the true current universe.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

REEP_PEOPLE_CSV_URL = "https://raw.githubusercontent.com/withqwerty/reep/main/data/people.csv"


def load_reep_crosswalk(url: str = REEP_PEOPLE_CSV_URL) -> pd.DataFrame:
    """Returns a DataFrame with columns [fotmob_player_id,
    transfermarkt_player_id] (both Int64), one row per REEP person who is a
    player with both IDs mapped. Rows missing either ID, or not
    type=='player' (e.g. managers), are dropped -- not an error, just
    outside this crosswalk's scope."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), usecols=["type", "key_fotmob", "key_transfermarkt"])
    players = df[(df["type"] == "player") & df["key_fotmob"].notna() & df["key_transfermarkt"].notna()]
    return (
        players[["key_fotmob", "key_transfermarkt"]]
        .astype("int64")
        .rename(columns={"key_fotmob": "fotmob_player_id", "key_transfermarkt": "transfermarkt_player_id"})
        .reset_index(drop=True)
    )
