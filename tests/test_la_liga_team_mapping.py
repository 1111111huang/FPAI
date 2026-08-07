"""Regression tests for US#145: config/team_mapping.json's La Liga entries.

US#143 ingests La Liga's football-data.co.uk CSV team names verbatim (only
whitespace-normalized via `standardize_team_name`, e.g. "Ath Bilbao", "La
Coruna") into `raw_matches` with no fuzzy mapping applied -- that's this
story's job, mirroring US#126's Sweden precedent exactly. This module:

1. Confirms every distinct raw team-name string actually observed in the real
   `raw_matches` table (SP1, 10 seasons ingested 2026-08-06 -- see US#145's
   completion notes in `documents/user_stories.md` for how this list was
   derived) round-trips through `TeamNameMapper.map_team()` to itself (the
   raw CSV spelling is already the canonical short name here, same
   convention as Sweden's).
2. Confirms diacritic-restored/full-official-name variants a fuzzy source
   (Understat/FotMob/manual entry) might plausibly send through resolve onto
   the same canonical short form.
3. Re-runs the collision check (Levenshtein `_similarity_score`,
   `min_similarity=0.82`, US#126/US#141's precedent) against *both* existing
   competitions' entries (E0 and SWE) -- the first three-way check, since
   La Liga is the third competition registered in this file.
"""

from __future__ import annotations

import json

from src.ingestion.common.team_mapping import TeamNameMapper, _similarity_score

MAPPING_PATH = "config/team_mapping.json"

# Every distinct home/away team-name string observed in the real, ingested
# `raw_matches` table for league='SP1' (10 seasons, 2015/16-2025/26,
# US#143/US#144's real scrape+ingest run, 2026-08-06) -- i.e. exactly what
# `CSVLoader.process_v1_csv` writes into `raw_matches.home_team`/`away_team`
# today (`standardize_team_name` is a no-op for these, since none match its
# EPL-specific alias map).
RAW_LA_LIGA_TEAM_NAMES: list[str] = [
    "Alaves", "Almeria", "Ath Bilbao", "Ath Madrid", "Barcelona", "Betis",
    "Cadiz", "Celta", "Eibar", "Elche", "Espanol", "Getafe", "Girona",
    "Granada", "Huesca", "La Coruna", "Las Palmas", "Leganes", "Levante",
    "Malaga", "Mallorca", "Osasuna", "Oviedo", "Real Madrid", "Sevilla",
    "Sociedad", "Sp Gijon", "Valencia", "Valladolid", "Vallecano",
    "Villarreal",
]


def test_every_raw_la_liga_team_name_round_trips_to_itself():
    mapper = TeamNameMapper(mapping_path=MAPPING_PATH)
    for raw_name in RAW_LA_LIGA_TEAM_NAMES:
        assert mapper.map_team(raw_name) == raw_name, (
            f"{raw_name!r} should resolve to itself (already the canonical short form)"
        )


def test_diacritic_and_full_official_names_map_onto_raw_short_forms():
    """Spelling variants a fuzzy source might plausibly send through --
    diacritic-restored and full official club names -- must resolve to the
    same canonical short form as the raw CSV spelling."""
    mapper = TeamNameMapper(mapping_path=MAPPING_PATH)
    cases = {
        "Deportivo Alaves": "Alaves",
        "UD Almeria": "Almeria",
        "Athletic Bilbao": "Ath Bilbao",
        "Athletic Club": "Ath Bilbao",
        "Atletico Madrid": "Ath Madrid",
        "Atletico de Madrid": "Ath Madrid",
        "FC Barcelona": "Barcelona",
        "Barca": "Barcelona",
        "Real Betis": "Betis",
        "Celta Vigo": "Celta",
        "RC Celta": "Celta",
        "Espanyol": "Espanol",
        "RCD Espanyol": "Espanol",
        "Deportivo La Coruna": "La Coruna",
        "Deportivo de La Coruna": "La Coruna",
        "CD Leganes": "Leganes",
        "Levante UD": "Levante",
        "Malaga CF": "Malaga",
        "RCD Mallorca": "Mallorca",
        "Real Oviedo": "Oviedo",
        "Real Madrid CF": "Real Madrid",
        "Sevilla FC": "Sevilla",
        "Real Sociedad": "Sociedad",
        "Sporting Gijon": "Sp Gijon",
        "Sporting de Gijon": "Sp Gijon",
        "Rayo Vallecano": "Vallecano",
        "Villarreal CF": "Villarreal",
    }
    for name, expected in cases.items():
        assert mapper.map_team(name) == expected


def test_no_high_similarity_collision_between_la_liga_e0_and_swe_entries():
    """Collision re-check (Levenshtein `_similarity_score`, `min_similarity
    =0.82`, the exact threshold `TeamNameMapper` itself uses) between every
    La Liga short name/full-name variant added by US#145 and every existing
    E0/SWE key/value in the file -- the first genuinely three-way check,
    since La Liga is the third competition registered here. Fails loudly if
    a future edit to any roster introduces a real name collision."""
    with open(MAPPING_PATH, encoding="utf-8") as handle:
        mapping: dict[str, str] = json.load(handle)

    la_liga_raw = set(RAW_LA_LIGA_TEAM_NAMES)
    la_liga_names = {k for k, v in mapping.items() if v in la_liga_raw} | la_liga_raw

    other_names = {k for k in mapping if k not in la_liga_names} | {
        v for v in mapping.values() if v not in la_liga_raw
    }

    min_similarity = TeamNameMapper(mapping_path=MAPPING_PATH).min_similarity
    collisions = []
    for ll in la_liga_names:
        for other in other_names:
            score = _similarity_score(ll, other)
            if score >= min_similarity:
                collisions.append((score, ll, other))

    assert not collisions, f"La Liga/other-competition name collision(s) at >= {min_similarity}: {collisions}"
