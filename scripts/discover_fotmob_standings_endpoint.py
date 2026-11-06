"""One-off spike (UCL/UEL pre-flight, 2026-11-04): discover a working
fotmob standings/table endpoint for UCL/UEL's league-phase, needed for
Swiss-format dead-rubber/qualification-scenario detection
(design spec Section 5). Starting leads already confirmed this session:
real league IDs 943230 (UCL) / 943229 (UEL); /api/data/leagues?id=943230
responds 200 but with a null body; /api/leaguedetails?id=943230 404s.
Tries a wider set of plausible real fotmob API paths/params -- confirm
whichever one actually returns real standings data before using it,
don't assume the first 200 response is the right shape."""
from __future__ import annotations

import requests

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
UCL_ID = 943230

CANDIDATE_URLS = [
    f"https://www.fotmob.com/api/data/tltable?leagueId={UCL_ID}",
    f"https://www.fotmob.com/api/tltable?leagueId={UCL_ID}",
    f"https://www.fotmob.com/api/data/leagues?id={UCL_ID}&type=league",
    f"https://www.fotmob.com/api/leagues?id={UCL_ID}&ccode3=INT",
    f"https://www.fotmob.com/api/data/table?leagueId={UCL_ID}",
    f"https://www.fotmob.com/api/data/leagues?id={UCL_ID}&tab=table&type=league",
    f"https://www.fotmob.com/api/standings?leagueId={UCL_ID}",
]


def main() -> None:
    for url in CANDIDATE_URLS:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        print(f"{url} -> status={resp.status_code} len={len(resp.content)}")
        if resp.status_code != 200 or not resp.content:
            continue
        try:
            payload = resp.json()
        except ValueError:
            print("  not JSON")
            continue
        if payload is None:
            print("  null body")
            continue
        print("  REAL RESPONSE -- top-level keys:", list(payload.keys())[:20] if isinstance(payload, dict) else type(payload))

    # Confirmed live, 2026-11-04: /api/data/tltable is the real standings
    # endpoint. It returns a list with one element; that element's
    # "data" key has leagueId/leagueName/legend (qualification-zone
    # definitions with color + team-index ranges)/table.all (one row per
    # team: played/wins/draws/losses/scoresStr/goalConDiff/pts/idx/
    # qualColor). 36 rows for both UCL and UEL, matching the Swiss-format
    # league phase exactly. Internally this endpoint reports leagueId=42
    # for UCL and leagueId=73 for UEL -- the commonly-cited-but-unverified
    # public IDs, which were correctly treated with suspicion earlier in
    # this session (the /api/data/matches endpoint uses different,
    # season-specific IDs: 943230/943229) -- both now confirmed live for
    # their respective endpoints, neither guessed.
    for league_id, label in [(42, "UCL"), (73, "UEL")]:
        resp = requests.get(f"https://www.fotmob.com/api/data/tltable?leagueId={league_id}", headers=_HEADERS, timeout=30)
        payload = resp.json()
        data = payload[0]["data"]
        print(f"\n{label} (leagueId={league_id}): {data['leagueName']}, {len(data['table']['all'])} teams")
        print(f"  qualification zones: {[entry['title'] for entry in data['legend']]}")


if __name__ == "__main__":
    main()
