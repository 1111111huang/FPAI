"""W138: football-data.org's Serie A/Bundesliga/Ligue 1 team-name spellings
vs the app's canonical team names (mirrors W59/W78's precedent exactly for
Sweden/La Liga).

Real finding (2026-08-15), cross-referencing football-data.org's live
rosters (`GET /v4/competitions/{SA,BL1,FL1}/teams`) against
config/team_mapping.json: 42 of 56 shortNames across all three leagues
already resolved correctly (self-mapped by US#164/US#165's own real-data
audits). 14 didn't -- all real spelling divergences for clubs this repo
already has training history for, added here as verified mappings:
  I1: "Venezia FC" -> "Venezia", "Como 1907" -> "Como"
  D1: "1. FC Köln" -> "FC Koln", "Bayern" -> "Bayern Munich",
      "Schalke" -> "Schalke 04", "HSV" -> "Hamburg",
      "Bremen" -> "Werder Bremen", "Frankfurt" -> "Ein Frankfurt",
      "SC Paderborn" -> "Paderborn"
  F1: "Olympique Lyon" -> "Lyon", "PSG" -> "Paris SG",
      "Stade Rennais" -> "Rennes", "Angers SCO" -> "Angers",
      "RC Lens" -> "Lens"

Two genuine cold-start cases (not mapping gaps) -- confirmed via a direct
`raw_matches` query, not assumed from the fuzzy-suggest score alone,
mirroring W78's own "Santander" precedent:
  D1: "Elversberg" -- zero rows in raw_matches for this club at all;
      promoted outside this repo's 2016/17-2025/26 training window.
  F1: "Le Mans" -- same, zero rows; football-data.org's free-tier roster
      listing includes it despite no current top-flight presence."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.ingestion.common.team_mapping import TeamNameMapper

MAPPING_PATH = "config/team_mapping.json"

# The exact shortName strings football-data.org returned live (2026-08-15)
# for GET /v4/competitions/{SA,BL1,FL1}/teams, mapped to the canonical
# short name config/team_mapping.json's other (football-data.co.uk-sourced)
# entries already use for the same real-world club. "Elversberg"/"Le Mans"
# are deliberately excluded -- see the cold-start note above.
FOOTBALL_DATA_ORG_SHORT_NAMES: dict[str, str] = {
    # I1 (Serie A)
    "Milan": "Milan", "Fiorentina": "Fiorentina", "Roma": "Roma",
    "Atalanta": "Atalanta", "Bologna": "Bologna", "Cagliari": "Cagliari",
    "Genoa": "Genoa", "Inter": "Inter", "Juventus": "Juventus",
    "Lazio": "Lazio", "Parma": "Parma", "Napoli": "Napoli",
    "Udinese": "Udinese", "Venezia FC": "Venezia", "Frosinone": "Frosinone",
    "Sassuolo": "Sassuolo", "Torino": "Torino", "Lecce": "Lecce",
    "Monza": "Monza", "Como 1907": "Como",
    # D1 (Bundesliga)
    "1. FC Köln": "FC Koln", "Hoffenheim": "Hoffenheim",
    "Leverkusen": "Leverkusen", "Dortmund": "Dortmund",
    "Bayern": "Bayern Munich", "Schalke": "Schalke 04", "HSV": "Hamburg",
    "Stuttgart": "Stuttgart", "Bremen": "Werder Bremen", "Mainz": "Mainz",
    "Augsburg": "Augsburg", "Freiburg": "Freiburg",
    "M'gladbach": "M'gladbach", "Frankfurt": "Ein Frankfurt",
    "Union Berlin": "Union Berlin", "SC Paderborn": "Paderborn",
    "RB Leipzig": "RB Leipzig",
    # F1 (Ligue 1)
    "Toulouse": "Toulouse", "Brest": "Brest", "Marseille": "Marseille",
    "Auxerre": "Auxerre", "Lille": "Lille", "Nice": "Nice",
    "Olympique Lyon": "Lyon", "PSG": "Paris SG", "Lorient": "Lorient",
    "Stade Rennais": "Rennes", "Troyes": "Troyes",
    "Angers SCO": "Angers", "Le Havre": "Le Havre", "RC Lens": "Lens",
    "Monaco": "Monaco", "Strasbourg": "Strasbourg", "Paris FC": "Paris FC",
}


def test_every_real_football_data_org_short_name_resolves_correctly():
    mapper = TeamNameMapper(mapping_path=MAPPING_PATH)
    for short_name, expected in FOOTBALL_DATA_ORG_SHORT_NAMES.items():
        assert mapper.map_team(short_name) == expected, (
            f"{short_name!r} should resolve to {expected!r}"
        )


def test_elversberg_and_le_mans_are_genuine_cold_starts_not_silently_mismapped():
    """Confirmed via a direct raw_matches query (not assumed) -- both have
    zero rows in this repo's training window, so there's no canonical name
    to point them at. Must round-trip to themselves (the existing
    unknown-team cold-start path), not silently resolve to an unrelated club."""
    mapper = TeamNameMapper(mapping_path=MAPPING_PATH)
    assert mapper.map_team("Elversberg") == "Elversberg"
    assert mapper.map_team("Le Mans") == "Le Mans"
