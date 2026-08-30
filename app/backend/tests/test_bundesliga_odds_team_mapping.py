"""BUG-057 / 2026-08-30: The Odds API's Bundesliga team-name spellings vs
the app's canonical team names (mirrors test_sweden_odds_team_mapping.py's
W59 precedent exactly, one league later).

config/team_mapping.json already has D1 entries from W138's
football-data.org audit, but those are scoped to football-data.org's own
shortName spelling -- The Odds API, used only at the app layer for
eod_batch.py's odds/fixtures join (not the ML engine's ingestion), turns
out to spell several Bundesliga clubs with their full club-type prefix
instead ("TSG Hoffenheim" vs football-data.org's bare "Hoffenheim").

Found live 2026-08-30 via user-shared Railway deploy logs: repeated
"Unmapped team '...'. Add mapping to config/team_mapping.json." warnings
for these exact 5 clubs, every EOD run. Since TeamNameMapper.map_team() has
no fuzzy fallback when called without an explicit `candidates` list (see
its docstring, and W59's own identical finding for Sweden) -- and
eod_batch.py's odds_lookup()/match_odds() call it with no candidates at
all -- an unmapped odds-side name never joins its fixture: the odds side
resolves to (e.g.) "TSG Hoffenheim" unchanged while the fixtures side
resolves to "Hoffenheim", two different dict keys, so the match's odds
silently never attach to its recommendation.

"Elversberg" (also logged as unmapped in the same run) is deliberately NOT
included here -- test_new_leagues_football_data_team_mapping.py's own
docstring already documents this as a genuine cold-start case (zero rows
in raw_matches, promoted outside this repo's training window), not a
mapping gap; adding a mapping wouldn't give the model any real history to
predict from."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.ingestion.common.team_mapping import TeamNameMapper

MAPPING_PATH = "config/team_mapping.json"

# The exact strings logged live (2026-08-30) as "Unmapped team", mapped to
# the canonical short name config/team_mapping.json's football-data.org-
# sourced entries already use for the same real-world club.
ODDS_API_BUNDESLIGA_TEAM_NAMES: dict[str, str] = {
    "TSG Hoffenheim": "Hoffenheim",
    "Borussia Monchengladbach": "M'gladbach",
    "SC Freiburg": "Freiburg",
    "FC Schalke 04": "Schalke 04",
    "FSV Mainz 05": "Mainz",
}


def test_every_odds_api_bundesliga_team_name_resolves_to_the_shared_canonical_name():
    mapper = TeamNameMapper(mapping_path=MAPPING_PATH)
    for odds_api_name, expected in ODDS_API_BUNDESLIGA_TEAM_NAMES.items():
        assert mapper.map_team(odds_api_name) == expected, (
            f"{odds_api_name!r} (The Odds API's spelling) should resolve to "
            f"canonical {expected!r}, the same short name football-data.org-"
            f"sourced entries already map to."
        )
