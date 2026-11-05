"""One-off spike (UCL/UEL pre-flight, 2026-11-04): bucket every club from
Task 1's real UCL/UEL club list into (a) reachable via a football-data.co.uk
country confirmed in Task 2, (b) neither -- a candidate for the fotmob-only
check (Task 4) or genuine no-coverage.

Two real bugs found and fixed while building this (not guessed in
advance):
1. football-data.co.uk has TWO different CSV formats. The 6 modeled
   leagues' format (mmz4281/<season>/<CODE>.csv, one file per season,
   HomeTeam/AwayTeam columns) only covers some of these 15 countries.
   Others (confirmed live: Austria) use a different "new leagues" format
   (new/<CODE>.csv, one file with EVERY season since 2012 concatenated,
   Home/Away columns) -- the same format already used for Sweden
   elsewhere in this codebase. Both are tried per country.
2. Team-name matching must be word-level, not whole-string. football-data.co.uk
   uses short team names ("Celtic", "Ajax") that never appear as a
   substring of the UCL/UEL manifest's longer names ("Celtic Glasgow",
   "Ajax Amsterdam") -- checking the full normalized name as one substring
   against the raw CSV text (an early draft's approach) found zero matches
   even for teams confirmed present. Fixed to compare real extracted team
   names against UCL/UEL club names in both directions.

This is still a spike-level approximation for bucketing, not
TeamNameMapper-grade resolution.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests

CLUB_LIST_PATH = Path(__file__).parent.parent / "data" / "ucl_uel_club_list.json"
COUNTRY_COVERAGE_PATH = Path(__file__).parent.parent / "data" / "football_data_wider_country_coverage.json"

# The 11-league universe (6 modeled leagues + 5 second divisions from the
# domestic-cup design) -- real football-data.co.uk codes, confirmed live
# either this session (E0/SP1/I1/D1/F1) or during the domestic-cup spike
# (E1/SP2/I2/D2/F2, cherry-picked into this branch already).
ELEVEN_LEAGUE_CODES = ["E0", "E1", "SP1", "SP2", "I1", "I2", "D1", "D2", "F1", "F2"]


def _normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\b(fc|cf|ac|sc|sk|bc|if|afc|club|de|football)\b", "", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def _discover_real_teams(country: str) -> tuple[str, set[str]] | None:
    """Try both known football-data.co.uk CSV formats for this country,
    return (format_used, real_team_name_set) for whichever one works."""
    page_url = f"https://www.football-data.co.uk/{country}m.php"
    page_resp = requests.get(page_url, timeout=15)
    if page_resp.status_code != 200:
        return None

    # Format 1: mmz4281/<season>/<CODE>.csv -- the 6 modeled leagues' own
    # format. Prefer this when both are linked.
    mmz_codes = re.findall(r"mmz4281/\d+/([A-Z0-9]+)\.csv", page_resp.text)
    if mmz_codes:
        code = mmz_codes[0]
        csv_resp = requests.get(f"https://www.football-data.co.uk/mmz4281/2526/{code}.csv", timeout=15)
        if csv_resp.status_code == 200:
            lines = csv_resp.text.splitlines()
            teams: set[str] = set()
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) > 4:
                    teams.add(parts[3])
                    teams.add(parts[4])
            if teams:
                return f"mmz4281/{code}", teams

    # Format 2: new/<CODE>.csv -- the "new leagues" format (same one
    # already used for Sweden), Home/Away columns, every season in one file.
    new_codes = re.findall(r"new/([A-Z0-9]+)\.csv", page_resp.text)
    if new_codes:
        code = new_codes[0]
        csv_resp = requests.get(f"https://www.football-data.co.uk/new/{code}.csv", timeout=15)
        if csv_resp.status_code == 200:
            lines = csv_resp.text.splitlines()
            teams = set()
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) > 6:
                    teams.add(parts[5])
                    teams.add(parts[6])
            if teams:
                return f"new/{code}", teams

    return None


def _fetch_mmz_teams(code: str) -> set[str]:
    resp = requests.get(f"https://www.football-data.co.uk/mmz4281/2526/{code}.csv", timeout=15)
    if resp.status_code != 200:
        return set()
    teams: set[str] = set()
    for line in resp.text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) > 4:
            teams.add(parts[3])
            teams.add(parts[4])
    return teams


def _match_against(club_norm: str, real_teams_norm: dict[str, str]) -> str | None:
    for team_norm, real_name in real_teams_norm.items():
        if len(team_norm) >= 4 and (team_norm in club_norm or club_norm in team_norm):
            return real_name
    return None


def main() -> None:
    with open(CLUB_LIST_PATH) as f:
        all_clubs = json.load(f)["all_clubs"]
    with open(COUNTRY_COVERAGE_PATH) as f:
        found_countries = json.load(f)["found"]

    # Exclude the 11-league universe FIRST -- those clubs are already
    # covered by the cup-transfer model (design spec Section 4), not
    # candidates for the wider Dixon-Coles check below, and leaving them
    # in risks a false positive (PSG's normalized name happened to contain
    # Greek club "Aris" as a coincidental substring on an earlier run).
    eleven_league_teams: set[str] = set()
    for code in ELEVEN_LEAGUE_CODES:
        eleven_league_teams |= _fetch_mmz_teams(code)
    eleven_league_norm = {_normalize(t): t for t in eleven_league_teams if _normalize(t)}

    already_in_universe = []
    remaining = set()
    for club in all_clubs:
        club_norm = _normalize(club)
        hit = _match_against(club_norm, eleven_league_norm)
        if hit:
            already_in_universe.append((club, hit))
        else:
            remaining.add(club)
    print(f"{len(already_in_universe)} clubs already in the 11-league universe, excluded from the wider check:")
    for club, matched_as in sorted(already_in_universe):
        print(f"  {club} (matched {matched_as!r})")

    buckets: dict[str, list[str]] = {
        "already_in_11_league_universe": sorted(c for c, _ in already_in_universe),
        "football_data_country": [],
    }
    matched_by_country: dict[str, dict[str, object]] = {}

    for country in found_countries:
        result = _discover_real_teams(country)
        if result is None:
            print(f"{country}: no usable CSV found in either format")
            continue
        fmt, real_teams = result
        real_teams_norm = {_normalize(t): t for t in real_teams if _normalize(t)}

        matched_this_country = []
        for club in list(remaining):
            if _match_against(_normalize(club), real_teams_norm):
                matched_this_country.append(club)

        for club in matched_this_country:
            remaining.discard(club)
        buckets["football_data_country"].extend(matched_this_country)
        matched_by_country[country] = {"format": fmt, "matched_clubs": matched_this_country}
        print(f"{country} ({fmt}): matched {len(matched_this_country)} clubs -- {matched_this_country}")

    buckets["unresolved"] = sorted(remaining)
    print(f"\n{len(buckets['football_data_country'])} clubs matched to a football-data.co.uk country.")
    print(f"{len(buckets['unresolved'])} clubs still unresolved -- candidates for Task 4's fotmob check.")

    out_path = Path(__file__).parent.parent / "data" / "ucl_uel_club_buckets.json"
    out_path.write_text(json.dumps({**buckets, "matched_by_country": matched_by_country}, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
