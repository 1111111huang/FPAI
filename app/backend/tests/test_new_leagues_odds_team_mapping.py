"""BUG-057 (2026-08-30) follow-up: The Odds API's Serie A/Ligue 1 team-name
spellings vs the app's canonical team names -- same gap as
test_bundesliga_odds_team_mapping.py, one more pair of leagues.

Found live 2026-08-30 via user-shared Railway deploy logs, in a second
batch after the Bundesliga fix already shipped: repeated "Unmapped team
'...'. Add mapping to config/team_mapping.json." warnings for Serie A's
"Inter Milan"/"AS Roma"/"Atalanta BC" and Ligue 1's "AS Monaco" -- every
EOD run. Same root cause as the Bundesliga case: config/team_mapping.json
has these clubs' football-data.org shortNames (W138) but never the
equivalent audit pass for The Odds API's own longer, club-type-suffixed/
prefixed spelling. TeamNameMapper.map_team() has no fuzzy fallback when
called without an explicit `candidates` list (eod_batch.py's odds/fixtures
join calls it with none), so an unmapped odds-side name never joins its
fixture.

"Le Mans FC" (also logged as unmapped in the same batch) is deliberately
NOT included here, for the identical reason "Elversberg" was excluded from
the Bundesliga fix: test_new_leagues_football_data_team_mapping.py already
documents "Le Mans" as a genuine cold-start case (zero rows in
raw_matches, football-data.org's free-tier roster lists it despite no
current top-flight presence) -- not a mapping gap."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.ingestion.common.team_mapping import TeamNameMapper

MAPPING_PATH = "config/team_mapping.json"

# The exact strings logged live (2026-08-30) as "Unmapped team", mapped to
# the canonical short name config/team_mapping.json's football-data.org-
# sourced entries already use for the same real-world club.
ODDS_API_NEW_LEAGUE_TEAM_NAMES: dict[str, str] = {
    "Inter Milan": "Inter",
    "AS Roma": "Roma",
    "Atalanta BC": "Atalanta",
    "AS Monaco": "Monaco",
}


def test_every_odds_api_new_league_team_name_resolves_to_the_shared_canonical_name():
    mapper = TeamNameMapper(mapping_path=MAPPING_PATH)
    for odds_api_name, expected in ODDS_API_NEW_LEAGUE_TEAM_NAMES.items():
        assert mapper.map_team(odds_api_name) == expected, (
            f"{odds_api_name!r} (The Odds API's spelling) should resolve to "
            f"canonical {expected!r}, the same short name football-data.org-"
            f"sourced entries already map to."
        )
