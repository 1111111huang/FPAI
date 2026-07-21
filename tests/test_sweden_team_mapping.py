"""Regression tests for US#126: config/team_mapping.json's Allsvenskan entries.

US#125 ingests Sweden's football-data.co.uk CSV team names verbatim (only
whitespace-normalized via `standardize_team_name`, e.g. "Malmo FF",
"Goteborg", "Djurgarden") into `raw_matches` with no fuzzy mapping applied --
that was deliberately deferred to this story. This module:

1. Confirms every distinct raw team-name string actually observed in a live
   `fetch_sweden_csv()` pull (recorded below -- see US#126's completion notes
   in `documents/user_stories.md` for how this list was derived) round-trips
   through `TeamNameMapper.map_team()` to one canonical short name per club,
   the same "explicit mapping" pattern already proven for EPL entries in
   `tests/test_understat.py::test_team_name_mapper_production_mapping`.
2. Confirms the two source-spelling variants of the same club seen across
   different eras of the CSV ("Oster", 2022-2025, and "Osters", 2013) both
   resolve to a single canonical short name ("Oster").
3. Re-runs the Sweden/EPL collision check US#141 established (Levenshtein
   `_similarity_score`, `min_similarity=0.82`) against the *actual* new
   Sweden short names now in the file, so a future edit that introduces a
   real collision fails loudly here instead of silently fuzzy-matching two
   different clubs from different competitions together.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.common.team_mapping import TeamNameMapper, _similarity_score

MAPPING_PATH = "config/team_mapping.json"

# Every distinct home/away team-name string observed in a live
# `fetch_sweden_csv()` pull (2026-07-21, 3489 rows, 2012-2026 history) --
# i.e. exactly what `sweden_loader.py::upsert_sweden_matches` writes into
# `raw_matches.home_team`/`away_team` today (standardize_team_name is a
# no-op for these, since none match its EPL-specific alias map).
RAW_SWEDEN_TEAM_NAMES: list[str] = [
    "AFC Eskilstuna", "AIK", "Atvidabergs", "Brage", "Brommapojkarna",
    "Dalkurd", "Degerfors", "Djurgarden", "Elfsborg", "Falkenbergs",
    "GAIS", "Gefle", "Goteborg", "Hacken", "Halmstad", "Hammarby",
    "Helsingborg", "Jonkopings", "Kalmar", "Landskrona", "Ljungskile",
    "Malmo FF", "Mjallby", "Norrkoping", "Orebro", "Orgryte", "Oster",
    "Osters", "Ostersunds", "Sirius", "Sundsvall", "Syrianska",
    "Trelleborgs", "Varberg", "Varnamo", "Vasteras SK",
]

# Expected canonical short name each raw string above maps to. "Oster" and
# "Osters" are the same real-world club (Östers IF) spelled two different
# ways in different eras of the source CSV -- both must converge on one
# canonical value ("Oster", the spelling used in the more recent rows).
EXPECTED_CANONICAL: dict[str, str] = {name: name for name in RAW_SWEDEN_TEAM_NAMES}
EXPECTED_CANONICAL["Osters"] = "Oster"


def test_every_raw_sweden_team_name_round_trips_to_canonical():
    mapper = TeamNameMapper(mapping_path=MAPPING_PATH)
    for raw_name, expected in EXPECTED_CANONICAL.items():
        assert mapper.map_team(raw_name) == expected, (
            f"{raw_name!r} should resolve to canonical {expected!r}"
        )


def test_oster_and_osters_are_the_same_club():
    """Same club, two source spellings across different CSV eras -- must merge."""
    mapper = TeamNameMapper(mapping_path=MAPPING_PATH)
    assert mapper.map_team("Oster") == mapper.map_team("Osters") == "Oster"


def test_diacritic_and_full_official_names_map_onto_raw_short_forms():
    """Spelling variants a fuzzy source (Understat/FotMob/manual entry) might
    plausibly send through -- diacritic-restored and full official club
    names -- must resolve to the same canonical short form as the raw CSV
    spelling, mirroring the EPL "full name / FC-suffixed name / short name"
    grouping already in the file (e.g. Wolverhampton Wanderers / Wolverhampton
    Wanderers FC / Wolves)."""
    mapper = TeamNameMapper(mapping_path=MAPPING_PATH)
    cases = {
        "Djurgårdens IF": "Djurgarden",
        "Djurgården": "Djurgarden",
        "IFK Göteborg": "Goteborg",
        "Göteborg": "Goteborg",
        "BK Häcken": "Hacken",
        "Malmö FF": "Malmo FF",
        "Östers IF": "Oster",
        "Öster": "Oster",
        "Östersunds FK": "Ostersunds",
        "Örebro SK": "Orebro",
        "Jönköpings Södra IF": "Jonkopings",
        "Västerås SK": "Vasteras SK",
    }
    for name, expected in cases.items():
        assert mapper.map_team(name) == expected


def test_no_high_similarity_collision_between_sweden_and_epl_entries():
    """US#141-style collision re-check (Levenshtein `_similarity_score`,
    `min_similarity=0.82`, the exact threshold `TeamNameMapper` itself uses)
    between every Sweden short name/full-name variant added by US#126 and
    every EPL key/value already in the file. Fails loudly if a future edit
    to either roster introduces a genuine name collision."""
    with open(MAPPING_PATH, encoding="utf-8") as handle:
        mapping: dict[str, str] = json.load(handle)

    sweden_values = set(EXPECTED_CANONICAL.values())
    sweden_names = {k for k, v in mapping.items() if v in sweden_values} | sweden_values

    # EPL names = every key/value in the file that isn't part of the Sweden set.
    epl_names = {k for k in mapping if k not in sweden_names} | {
        v for v in mapping.values() if v not in sweden_values
    }

    min_similarity = TeamNameMapper(mapping_path=MAPPING_PATH).min_similarity
    collisions = []
    for sw in sweden_names:
        for epl in epl_names:
            score = _similarity_score(sw, epl)
            if score >= min_similarity:
                collisions.append((score, sw, epl))

    assert not collisions, f"Sweden/EPL name collision(s) at >= {min_similarity}: {collisions}"


def test_team_mapping_file_is_valid_json_dict_of_strings():
    """Lightweight structural sanity check (no dedicated schema test existed
    for this file prior to US#126) -- every key/value must be a plain
    string, since `TeamNameMapper._load_mapping` coerces everything via
    `str()` and a non-string entry would silently mask a malformed edit."""
    with open(MAPPING_PATH, encoding="utf-8") as handle:
        mapping = json.load(handle)
    assert isinstance(mapping, dict)
    assert len(mapping) > 0
    for key, value in mapping.items():
        assert isinstance(key, str) and key.strip(), f"bad key: {key!r}"
        assert isinstance(value, str) and value.strip(), f"bad value for {key!r}: {value!r}"
