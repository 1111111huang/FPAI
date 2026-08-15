"""Regression tests for US#164: config/team_mapping.json's Serie A (I1),
Bundesliga (D1), and Ligue 1 (F1) entries.

US#163 ingested all three leagues' football-data.co.uk CSV team names
verbatim (only whitespace-normalized) into `raw_matches`, with no fuzzy
mapping applied -- mirrors every prior competition's own precedent (E0/SWE/
SP1). This module:

1. Confirms every distinct raw team-name string actually observed in the
   real `raw_matches` table (10 seasons each, ingested 2026-08-15 -- see
   US#164's completion notes in `documents/user_stories.md` for how this
   list was derived) round-trips through `TeamNameMapper.map_team()` to
   itself, i.e. the raw CSV spelling is self-mapped as the canonical short
   name -- closes the "logs a warning every call" gap the exact-lookup path
   (`feature_factory.py::build_for_match`, no `candidates` passed) would
   otherwise hit for every one of these 96 clubs on every match built.
2. Re-runs the collision check (Levenshtein `_similarity_score`,
   `min_similarity=0.82`, US#126/US#141/US#145's precedent) against every
   existing E0/SWE/SP1 entry -- the first genuinely six-way check.

**Deliberately out of scope here** (unlike US#145's La Liga precedent,
which proactively guessed at diacritic/full-official-name variants): no
Understat/FotMob/football-data.org/Odds-API data has been fetched for any
of these three leagues yet (US#165, `documents/app_user_stories.md` W138),
so there's no real spelling to verify a guessed variant against -- adding
unverified full-name variants here would be exactly the kind of assumption
this codebase's own "measure, don't assume" precedent argues against.
Real spelling gaps (e.g. "Bayern Muenchen" vs. "Bayern Munich", "PSG" vs.
"Paris SG") will surface against live data in US#165/W138, the same way
La Liga's real gaps surfaced in W78 rather than being fully anticipated
here.
"""

from __future__ import annotations

import json

from src.ingestion.common.team_mapping import TeamNameMapper, _similarity_score

MAPPING_PATH = "config/team_mapping.json"

# Every distinct home/away team-name string observed in the real, ingested
# `raw_matches` table for league in ('I1', 'D1', 'F1') (10 seasons each,
# 2016/17-2025/26, US#163's real scrape+ingest run, 2026-08-15) -- i.e.
# exactly what `CSVLoader.process_v1_csv` writes into
# `raw_matches.home_team`/`away_team` today.
RAW_SERIE_A_TEAM_NAMES: list[str] = [
    "Atalanta", "Benevento", "Bologna", "Brescia", "Cagliari", "Chievo",
    "Como", "Cremonese", "Crotone", "Empoli", "Fiorentina", "Frosinone",
    "Genoa", "Inter", "Juventus", "Lazio", "Lecce", "Milan", "Monza",
    "Napoli", "Palermo", "Parma", "Pescara", "Pisa", "Roma", "Salernitana",
    "Sampdoria", "Sassuolo", "Spal", "Spezia", "Torino", "Udinese",
    "Venezia", "Verona",
]

RAW_BUNDESLIGA_TEAM_NAMES: list[str] = [
    "Augsburg", "Bayern Munich", "Bielefeld", "Bochum", "Darmstadt",
    "Dortmund", "Ein Frankfurt", "FC Koln", "Fortuna Dusseldorf",
    "Freiburg", "Greuther Furth", "Hamburg", "Hannover", "Heidenheim",
    "Hertha", "Hoffenheim", "Holstein Kiel", "Ingolstadt", "Leverkusen",
    "M'gladbach", "Mainz", "Nurnberg", "Paderborn", "RB Leipzig",
    "Schalke 04", "St Pauli", "Stuttgart", "Union Berlin", "Werder Bremen",
    "Wolfsburg",
]

RAW_LIGUE_1_TEAM_NAMES: list[str] = [
    "Ajaccio", "Amiens", "Angers", "Auxerre", "Bastia", "Bordeaux", "Brest",
    "Caen", "Clermont", "Dijon", "Guingamp", "Le Havre", "Lens", "Lille",
    "Lorient", "Lyon", "Marseille", "Metz", "Monaco", "Montpellier",
    "Nancy", "Nantes", "Nice", "Nimes", "Paris FC", "Paris SG", "Reims",
    "Rennes", "St Etienne", "Strasbourg", "Toulouse", "Troyes",
]

ALL_NEW_RAW_TEAM_NAMES: list[str] = (
    RAW_SERIE_A_TEAM_NAMES + RAW_BUNDESLIGA_TEAM_NAMES + RAW_LIGUE_1_TEAM_NAMES
)


def test_every_raw_team_name_round_trips_to_itself():
    mapper = TeamNameMapper(mapping_path=MAPPING_PATH)
    for raw_name in ALL_NEW_RAW_TEAM_NAMES:
        assert mapper.map_team(raw_name) == raw_name, (
            f"{raw_name!r} should resolve to itself (already the canonical short form)"
        )


def test_no_duplicate_raw_names_across_the_three_new_leagues():
    """A club short name colliding across two of these three leagues (not
    just against E0/SWE/SP1) would silently merge under one mapping key --
    confirm the three real rosters are disjoint."""
    assert len(ALL_NEW_RAW_TEAM_NAMES) == len(set(ALL_NEW_RAW_TEAM_NAMES))


def test_no_high_similarity_collision_between_new_leagues_and_existing_entries():
    """Collision re-check (Levenshtein `_similarity_score`, `min_similarity
    =0.82`, the exact threshold `TeamNameMapper` itself uses) between every
    new I1/D1/F1 short name and every existing E0/SWE/SP1 key/value in the
    file -- the first genuinely six-way check. Fails loudly if a future edit
    to any roster introduces a real name collision."""
    with open(MAPPING_PATH, encoding="utf-8") as handle:
        mapping: dict[str, str] = json.load(handle)

    new_names = set(ALL_NEW_RAW_TEAM_NAMES)
    other_names = {k for k in mapping if k not in new_names} | {
        v for v in mapping.values() if v not in new_names
    }

    min_similarity = TeamNameMapper(mapping_path=MAPPING_PATH).min_similarity
    collisions = []
    for new in new_names:
        for other in other_names:
            score = _similarity_score(new, other)
            if score >= min_similarity:
                collisions.append((score, new, other))

    assert not collisions, f"New-league/existing-competition name collision(s) at >= {min_similarity}: {collisions}"
