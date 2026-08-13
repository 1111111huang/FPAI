"""W78: football-data.org's La Liga team-name spellings vs the app's
canonical team names (mirrors W59's Odds-API verification for Sweden).

config/team_mapping.json already has ~70 La Liga entries from US#145, but
those were sourced from the ML engine's own training-data ingestion
(football-data.co.uk's SP1.csv short forms). W76's live fixtures client
(app/backend/football_data_client.py's `_normalize()`) uses a *different*
field from a *different* provider -- football-data.org's `shortName`, not
football-data.co.uk's CSV team column -- so there's no guarantee the two
providers spell every club identically, the exact class of gap W59 already
found once for Sweden/The Odds API.

Real finding (2026-08-07), cross-referencing football-data.org's live
20-team La Liga roster (`GET /v4/competitions/PD/teams`) against
config/team_mapping.json: 13 of 20 shortNames already resolve correctly.
7 don't:
  - "Athletic" (Athletic Club/Bilbao) -- only "Athletic Club"/"Athletic
    Bilbao" were mapped, not the bare shortName.
  - "Atleti" (Atlético Madrid) -- only "Atletico Madrid"/"Atletico de
    Madrid" were mapped.
  - "Barça" (FC Barcelona) -- only unaccented "Barca"/"Barcelona"/"FC
    Barcelona" were mapped; football-data.org's shortName keeps the
    cedilla.
  - "Málaga" (accented shortName) -- only unaccented "Malaga"/"Malaga CF"
    were mapped.
  - "Alavés" (accented shortName) -- only unaccented "Alaves"/"Deportivo
    Alaves" were mapped.
  - "Deportivo" (bare shortName for Deportivo La Coruña) -- only
    "Deportivo de La Coruna"/"Deportivo La Coruna" (full names) were
    mapped.
  - "Santander" (Real Racing Club de Santander) -- a genuine cold-start
    case, not a mapping gap: this club does not appear anywhere in
    raw_matches' real 31-club SP1 roster (US#145's audit) at all, meaning
    it was promoted to La Liga for a season this repo's training data
    (2015/16-2025/26) doesn't cover. There is no existing canonical name
    to point it at -- left unmapped deliberately, degrading to the
    existing unknown-team cold-start handling (tests/test_unknown_team_flag.py)
    rather than inventing a fake canonical entry for a club with zero
    history."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.ingestion.common.team_mapping import TeamNameMapper

MAPPING_PATH = "config/team_mapping.json"

# The exact 20 team shortName strings football-data.org returned live
# (2026-08-07) for GET /v4/competitions/PD/teams, mapped to the canonical
# short name config/team_mapping.json's other (football-data.co.uk-sourced)
# entries already use for the same real-world club. "Santander" is
# deliberately excluded -- see the cold-start note above, not a mapping gap.
FOOTBALL_DATA_ORG_LA_LIGA_SHORT_NAMES: dict[str, str] = {
    "Athletic": "Ath Bilbao",
    "Atleti": "Ath Madrid",
    "Osasuna": "Osasuna",
    "Espanyol": "Espanol",
    "Barça": "Barcelona",
    "Getafe": "Getafe",
    "Málaga": "Malaga",
    "Real Madrid": "Real Madrid",
    "Rayo Vallecano": "Vallecano",
    "Levante": "Levante",
    "Real Betis": "Betis",
    "Real Sociedad": "Sociedad",
    "Villarreal": "Villarreal",
    "Valencia": "Valencia",
    "Alavés": "Alaves",
    "Elche": "Elche",
    "Celta": "Celta",
    "Sevilla FC": "Sevilla",
    "Deportivo": "La Coruna",
}


def test_every_football_data_org_la_liga_short_name_resolves_to_the_shared_canonical_name():
    mapper = TeamNameMapper(mapping_path=MAPPING_PATH)
    for short_name, expected in FOOTBALL_DATA_ORG_LA_LIGA_SHORT_NAMES.items():
        assert mapper.map_team(short_name) == expected, (
            f"{short_name!r} (football-data.org's shortName) should resolve to "
            f"canonical {expected!r}, the same short name football-data.co.uk-"
            f"sourced entries already map to."
        )


def test_santander_is_a_genuine_cold_start_not_a_mapping_gap():
    """Real Racing Club de Santander has zero rows in raw_matches (US#145's
    31-club audit never saw it) -- confirms it's correctly left unmapped
    (returned unchanged) rather than silently resolved to some other club's
    canonical name."""
    mapper = TeamNameMapper(mapping_path=MAPPING_PATH)
    assert mapper.map_team("Santander") == "Santander"
    assert mapper.map_team("Real Racing Club de Santander") == "Real Racing Club de Santander"


# 2026-08-13: live warnings on the deployed instance ("Unmapped team
# 'Atlético Madrid'"/"'Deportivo La Coruña'.") -- traced to a *third* team
# name source for SP1, distinct from both this file's football-data.org
# shortNames (W78) and W59's Sweden Odds-API audit: The Odds API's SP1 feed
# (odds_sport_keys.py's "soccer_spain_la_liga", the same provider W59 found
# returns Sweden's full accented club names) returns La Liga's full,
# diacritic-carrying official names too -- names config/team_mapping.json's
# existing entries never needed to spell accented (they were added from
# football-data.co.uk's own ASCII-only SP1.csv). Rather than a third manual
# audit enumerating The Odds API's own SP1 spellings (mirroring W59/W78),
# TeamNameMapper.map_team()'s accent-folding fallback (root cause fix, same
# session) closes this for every existing entry at once.
def test_the_odds_apis_full_accented_la_liga_names_resolve_via_the_existing_mapping():
    mapper = TeamNameMapper(mapping_path=MAPPING_PATH)
    assert mapper.map_team("Atlético Madrid") == "Ath Madrid"
    assert mapper.map_team("Deportivo La Coruña") == "La Coruna"
