"""W59: The Odds API's Allsvenskan team-name spellings vs the app's canonical
team names (mirrors W06's football-data.org verification).

config/team_mapping.json already has ~40 Swedish entries from US#126, but
those are scoped to football-data.co.uk's spelling (the ML engine's own
ingestion source) -- not guaranteed to match The Odds API's spelling, a
completely different provider used only at the app layer (W57).

Real finding (2026-07-23), cross-referencing The Odds API's live 16-team
upcoming-fixture list (soccer_sweden_allsvenskan) against
config/team_mapping.json: 13 of 16 already resolve correctly as existing
mapping keys. 3 don't, all for the same reason -- The Odds API
inconsistently ASCII-strips Swedish diacritics team-by-team within the same
response (Västerås SK/Mjällby AIF/Örgryte IS keep å/ä/ö; these three don't):
"Djurgardens IF" (only the diacritic form "Djurgårdens IF" was mapped),
"IFK Goteborg" (only "IFK Göteborg"), "BK Hacken" (only "BK Häcken").

Critically, TeamNameMapper.map_team() has no fuzzy fallback when called
without an explicit `candidates` list (see its docstring) -- and
eod_batch.py's odds_lookup()/match_odds() call it with no candidates at all.
So an unmapped name isn't "close enough, fuzzy-matched" -- it's returned
completely unchanged, silently failing to match against the canonical name
used elsewhere, exactly like a real mismatch would (BUG-015's original
finding, for a different provider)."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.ingestion.common.team_mapping import TeamNameMapper

MAPPING_PATH = "config/team_mapping.json"

# The exact 16 team-name strings The Odds API returned live (2026-07-23) for
# soccer_sweden_allsvenskan's upcoming fixtures, mapped to the canonical
# short name config/team_mapping.json's other (football-data.co.uk-sourced)
# entries already use for the same real-world club.
ODDS_API_SWEDISH_TEAM_NAMES: dict[str, str] = {
    "AIK": "AIK",
    "BK Hacken": "Hacken",
    "Degerfors IF": "Degerfors",
    "Djurgardens IF": "Djurgarden",
    "GAIS": "GAIS",
    "Halmstads BK": "Halmstad",
    "Hammarby IF": "Hammarby",
    "IF Brommapojkarna": "Brommapojkarna",
    "IF Elfsborg": "Elfsborg",
    "IFK Goteborg": "Goteborg",
    "IK Sirius": "Sirius",
    "Kalmar FF": "Kalmar",
    "Malmo FF": "Malmo FF",
    "Mjällby AIF": "Mjallby",
    "Västerås SK": "Vasteras SK",
    "Örgryte IS": "Orgryte",
}


def test_every_odds_api_swedish_team_name_resolves_to_the_shared_canonical_name():
    mapper = TeamNameMapper(mapping_path=MAPPING_PATH)
    for odds_api_name, expected in ODDS_API_SWEDISH_TEAM_NAMES.items():
        assert mapper.map_team(odds_api_name) == expected, (
            f"{odds_api_name!r} (The Odds API's spelling) should resolve to "
            f"canonical {expected!r}, the same short name football-data.co.uk-"
            f"sourced entries already map to."
        )
