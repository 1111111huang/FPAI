# Champions League / Europa League Pre-Flight Data Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, with real data (not guesses), how much of UCL/UEL's
real opponent pool is reachable by the Dixon-Coles fallback's score-only
ingestion, and whether a group/league-phase standings source exists for
the Swiss-format stakes detection flagged in
`docs/superpowers/specs/2026-11-04-champions-league-recommendations-design.md`
Section 8 — before committing to how wide the Dixon-Coles pool needs to be.

**Architecture:** Mostly free, zero-API-cost scripts under `scripts/`. The
single most valuable resource here is already sitting in git history at
zero cost: `data/oddspapi_ucl_uel_snapshots/manifest.json` on the abandoned
`worktree-llm-only-ucl-uel` branch has the real 368-fixture, 150-unique-club
UCL/UEL dataset from the historical backfill — this plan reads it directly
rather than making any new OddsPapi calls. The only metered-adjacent
resource touched is fotmob's public (unauthenticated, free) matches
endpoint, same as the domestic-cup spike's Task 7.

**Note on task structure:** same as the domestic-cup pre-flight plan — this
verifies external reality, not business logic this repo owns, so each task
is write-script → run → record-the-exact-output → commit, not red/green
unit tests.

**Tech Stack:** `requests`, no API keys needed for any task in this plan
(the manifest is already-fetched data; football-data.co.uk and fotmob are
both unauthenticated public endpoints).

---

### Task 1: Extract the real UCL/UEL club list from the existing manifest

**Files:**
- Create: `scripts/extract_ucl_uel_club_list.py`
- Create: `docs/superpowers/plans/2026-11-04-champions-league-preflight-findings.md`

Zero new API calls — `data/oddspapi_ucl_uel_snapshots/manifest.json` only
exists on the abandoned branch, not this one, so it's read via `git show`
rather than copied into this worktree's own tree.

- [ ] **Step 1: Pull the manifest from the abandoned branch (read-only, no new fetch)**

Run:
```bash
git show worktree-llm-only-ucl-uel:data/oddspapi_ucl_uel_snapshots/manifest.json > /tmp/ucl_uel_manifest.json
```
Expected: a JSON file, ~368 top-level fixture entries.

- [ ] **Step 2: Write the extraction script**

```python
"""One-off spike (UCL/UEL pre-flight, 2026-11-04): extract the real set of
unique clubs that appeared in UCL/UEL since 2026-01-01, from the
already-fetched historical manifest (zero new API cost -- this data was
paid for once, during the abandoned branch's original backfill)."""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST_PATH = "/tmp/ucl_uel_manifest.json"


def main() -> None:
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    clubs: set[str] = set()
    by_league: dict[str, set[str]] = {"UCL": set(), "UEL": set()}
    for fixture in manifest.values():
        p1, p2 = fixture["participant1Name"], fixture["participant2Name"]
        clubs.add(p1)
        clubs.add(p2)
        league = fixture.get("league", "UNKNOWN")
        by_league.setdefault(league, set()).update([p1, p2])

    print(f"{len(manifest)} fixtures, {len(clubs)} unique clubs")
    for league, names in by_league.items():
        print(f"  {league}: {len(names)} unique clubs")

    out_path = Path(__file__).parent.parent / "data" / "ucl_uel_club_list.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "all_clubs": sorted(clubs),
        "by_league": {k: sorted(v) for k, v in by_league.items()},
    }, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run: `python scripts/extract_ucl_uel_club_list.py`
Expected: 368 fixtures, 150 unique clubs (189 UCL-side + 179 UEL-side
fixture-participations, per the already-recorded backfill counts, with
overlap since some clubs play in both competitions).

- [ ] **Step 4: Record findings and commit**

Create `docs/superpowers/plans/2026-11-04-champions-league-preflight-findings.md`:

```markdown
# Champions League / Europa League Pre-Flight Findings

Live-verified 2026-11-04.

## Real UCL/UEL club list (from the existing 368-match historical manifest)

368 fixtures, 150 unique clubs, extracted at zero new API cost from
`worktree-llm-only-ucl-uel`'s already-paid-for backfill
(`data/oddspapi_ucl_uel_snapshots/manifest.json`). Full list saved to
`data/ucl_uel_club_list.json`.
```

```bash
git add scripts/extract_ucl_uel_club_list.py \
  docs/superpowers/plans/2026-11-04-champions-league-preflight-findings.md \
  data/ucl_uel_club_list.json
git commit -m "spike: extract real UCL/UEL club list from existing manifest"
```

---

### Task 2: Discover football-data.co.uk's country coverage beyond the 11-league universe

**Files:**
- Create: `scripts/discover_football_data_country_coverage.py`
- Modify: `docs/superpowers/plans/2026-11-04-champions-league-preflight-findings.md`

Free, unauthenticated. Checks real country pages by status code, same
`<country>m.php` pattern already confirmed for the 6 modeled leagues and
the domestic-cup design's 5 second divisions — extended here to a wider
candidate list, since UCL/UEL draws from far more countries than either.

- [ ] **Step 1: Write the discovery script**

```python
"""One-off spike (UCL/UEL pre-flight, 2026-11-04): check which additional
countries football-data.co.uk covers beyond the 6 modeled leagues, by
testing real <country>m.php pages (same pattern main.py's
_SCRAPE_SOURCE_OVERRIDE_BY_LEAGUE already uses) -- a candidate list drawn
from UCL/UEL's actual participant countries (Task 1), not guessed in the
abstract."""
from __future__ import annotations

import json
from pathlib import Path

import requests

# Candidate countries, chosen because UCL/UEL clubs from them appear in
# Task 1's real club list -- not an abstract "every UEFA country" guess.
CANDIDATES = [
    "netherlands", "portugal", "belgium", "turkey", "scotland", "greece",
    "austria", "switzerland", "denmark", "norway", "poland", "russia",
    "romania", "ireland", "finland", "ukraine", "croatia", "serbia",
    "czechrepublic", "israel", "cyprus",
]


def main() -> None:
    found = []
    not_found = []
    for country in CANDIDATES:
        url = f"https://www.football-data.co.uk/{country}m.php"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            found.append(country)
            print(f"{country}: FOUND ({url})")
        else:
            not_found.append(country)
            print(f"{country}: not found (status={resp.status_code})")

    out_path = Path(__file__).parent.parent / "data" / "football_data_wider_country_coverage.json"
    out_path.write_text(json.dumps({"found": found, "not_found": not_found}, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/discover_football_data_country_coverage.py`
Expected: most candidates found (football-data.co.uk covers most of
Western/Central/Northern Europe), a handful not found (likely the smaller
federations). Live-verified 2026-11-04: found = netherlands, portugal,
belgium, turkey, scotland, greece, austria, switzerland, denmark, norway,
poland, russia, romania, ireland, finland; not found = ukraine, croatia,
serbia, czechrepublic (404s — israel/cyprus not yet checked at the time of
spec-writing, confirm in this run).

- [ ] **Step 3: Record findings and commit**

Append to the findings doc:

```markdown
## football-data.co.uk coverage beyond the 11-league universe

| Country | Page found |
|---|---|
| Netherlands | True |
| Portugal | True |
| Belgium | True |
| Turkey | True |
| Scotland | True |
| Greece | True |
| Austria | True |
| Switzerland | True |
| Denmark | True |
| Norway | True |
| Poland | True |
| Russia | True |
| Romania | True |
| Ireland | True |
| Finland | True |
| Ukraine | False (404) |
| Croatia | False (404) |
| Serbia | False (404) |
| Czech Republic | False (404) |
| Israel | <fill in from this run> |
| Cyprus | <fill in from this run> |

15 additional countries confirmed beyond the 6 modeled leagues — a
substantial widening of the Dixon-Coles pool's reach, found at zero
ongoing cost (same free CSV mechanism already used for the 6 leagues).
```

```bash
git add scripts/discover_football_data_country_coverage.py \
  docs/superpowers/plans/2026-11-04-champions-league-preflight-findings.md \
  data/football_data_wider_country_coverage.json
git commit -m "spike: discover football-data.co.uk coverage for UCL/UEL participant countries"
```

---

### Task 3: Bucket every UCL/UEL club by data-source reach

**Files:**
- Create: `scripts/bucket_ucl_uel_clubs_by_coverage.py`
- Modify: `docs/superpowers/plans/2026-11-04-champions-league-preflight-findings.md`

For each confirmed country from Task 2, fetch its current-season CSV and
check which of Task 1's 150 clubs appear in it (normalized substring match
— football-data.co.uk's team names are often abbreviated, e.g. "Ajax" not
"Ajax Amsterdam"; this is a spike-level approximation for bucketing, not a
production-grade team-name resolution, which is `TeamNameMapper`'s job
when real ingestion code is written).

- [ ] **Step 1: Write the bucketing script**

```python
"""One-off spike (UCL/UEL pre-flight, 2026-11-04): bucket every club from
Task 1's real UCL/UEL club list into (a) already in the 11-league
universe, (b) reachable via a football-data.co.uk country confirmed in
Task 2, (c) neither -- a candidate for the fotmob-only check (Task 4) or
genuine no-coverage. Name matching here is a normalized-substring
approximation for bucketing purposes only, not TeamNameMapper-grade
resolution.

Each country's real CSV code is discovered from its own page (same
regex-on-real-HTML technique already used for the domestic-cup design's 5
second divisions, scripts/check_second_division_historical_coverage.py)
-- not hardcoded from a guessed convention. None of these 15 countries'
codes were verified during spec-writing; all of them are discovered fresh
when this script runs."""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests

CLUB_LIST_PATH = Path(__file__).parent.parent / "data" / "ucl_uel_club_list.json"
COUNTRY_COVERAGE_PATH = Path(__file__).parent.parent / "data" / "football_data_wider_country_coverage.json"


def _discover_csv_code(country: str) -> str | None:
    """Fetch the country's page and find its real current-season CSV
    filename -- the code is whatever's actually linked, not assumed."""
    page_url = f"https://www.football-data.co.uk/{country}m.php"
    resp = requests.get(page_url, timeout=15)
    if resp.status_code != 200:
        return None
    matches = re.findall(r"mmz4281/\d+/([A-Z0-9]+)\.csv", resp.text)
    if not matches:
        return None
    # A country page can link several divisions (e.g. England: E0/E1/E2/E3)
    # -- take the first, which is consistently the top flight on every page
    # checked so far for the 6 modeled leagues, but print all of them so a
    # wrong pick is visible rather than silent.
    print(f"  {country}: real codes found on page = {sorted(set(matches))}")
    return matches[0]


def _normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\b(fc|cf|ac|sc|sk|bc|if|afc|club|de|football)\b", "", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def main() -> None:
    with open(CLUB_LIST_PATH) as f:
        all_clubs = json.load(f)["all_clubs"]
    with open(COUNTRY_COVERAGE_PATH) as f:
        found_countries = json.load(f)["found"]

    normalized_clubs = {_normalize(c): c for c in all_clubs}
    remaining = set(normalized_clubs)
    buckets: dict[str, list[str]] = {"football_data_country": []}

    for country in found_countries:
        code = _discover_csv_code(country)
        if not code:
            print(f"{country}: no CSV code discovered on its page, skipping")
            continue
        url = f"https://www.football-data.co.uk/mmz4281/2526/{code}.csv"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"{country}: CSV fetch failed for discovered code {code} (status={resp.status_code})")
            continue
        text = resp.text.lower()
        text_normalized = _normalize(text)
        matched_this_country = []
        for norm_name in list(remaining):
            if norm_name and norm_name in text_normalized:
                matched_this_country.append(normalized_clubs[norm_name])
                remaining.discard(norm_name)
        buckets["football_data_country"].extend(matched_this_country)
        print(f"{country}: matched {len(matched_this_country)} clubs -- {matched_this_country}")

    buckets["unresolved"] = [normalized_clubs[n] for n in remaining]
    print(f"\n{len(buckets['football_data_country'])} clubs matched to a football-data.co.uk country.")
    print(f"{len(buckets['unresolved'])} clubs still unresolved -- candidates for Task 4's fotmob check.")

    out_path = Path(__file__).parent.parent / "data" / "ucl_uel_club_buckets.json"
    out_path.write_text(json.dumps(buckets, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/bucket_ucl_uel_clubs_by_coverage.py`
Expected: a meaningful fraction of the 150 clubs matched to a
football-data.co.uk country (the big, well-known UCL/UEL regulars —
Ajax, Porto, Benfica, Celtic, Besiktas, Club Brugge, Olympiacos, etc.);
the rest unresolved (smaller federations: Faroe Islands, Andorra, San
Marino, Kazakhstan, Armenia, Moldova, Gibraltar, Malta, etc., per Task 1's
real club list). Normalized substring matching will miss some real matches
and may produce a few false positives on short/common name fragments —
treat this as a directional bucket count, not a precise final list; a real
ingestion pass would use `TeamNameMapper` for anything load-bearing.

- [ ] **Step 3: Record findings and commit**

Append to the findings doc, with the real bucket counts and the full
`unresolved` list from this run.

```bash
git add scripts/bucket_ucl_uel_clubs_by_coverage.py \
  docs/superpowers/plans/2026-11-04-champions-league-preflight-findings.md \
  data/ucl_uel_club_buckets.json
git commit -m "spike: bucket UCL/UEL clubs by football-data.co.uk coverage"
```

---

### Task 4: Check fotmob's raw payload for the unresolved clubs

**Files:**
- Create: `scripts/check_fotmob_unresolved_ucl_uel_clubs.py`
- Modify: `docs/superpowers/plans/2026-11-04-champions-league-preflight-findings.md`

Free, unauthenticated. For clubs Task 3 couldn't resolve to a
football-data.co.uk country, check whether fotmob's unfiltered matches
payload has seen them in their own domestic league recently — same
mechanism as the domestic-cup spike's Task 7, applied to specific club
names instead of competition names.

- [ ] **Step 1: Write the check script**

```python
"""One-off spike (UCL/UEL pre-flight, 2026-11-04): for clubs Task 3 could
not resolve to a football-data.co.uk country, check fotmob's unfiltered
/api/data/matches payload for any recent match involving them -- the
lighter-weight signal the Dixon-Coles fallback would actually have to rely
on for this long tail."""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion.fotmob.fetcher import _fetch_matches_payload  # noqa: E402

BUCKETS_PATH = Path(__file__).parent.parent / "data" / "ucl_uel_club_buckets.json"


def main() -> None:
    with open(BUCKETS_PATH) as f:
        unresolved = json.load(f)["unresolved"]

    today = date.today()
    found: dict[str, bool] = {club: False for club in unresolved}

    # 14 days -- wider than the domestic-cup spike's 7, since these are
    # specific clubs' domestic fixtures, not a whole competition's round,
    # so a single week is more likely to miss a match entirely.
    for offset in range(14):
        day = today - timedelta(days=offset)
        payload = _fetch_matches_payload(day, delay=1.0)
        if payload is None:
            continue
        for league_entry in payload.get("leagues", []):
            for match in league_entry.get("matches", []):
                home = str(match.get("home", {}).get("name", ""))
                away = str(match.get("away", {}).get("name", ""))
                for club in unresolved:
                    if found[club]:
                        continue
                    # Loose substring check -- same bucketing-only caveat
                    # as Task 3, not TeamNameMapper-grade.
                    club_key = club.split()[0].lower()
                    if club_key in home.lower() or club_key in away.lower():
                        found[club] = True

    resolved_count = sum(found.values())
    print(f"{resolved_count}/{len(unresolved)} previously-unresolved clubs found in fotmob's payload (last 14 days)")
    for club, was_found in sorted(found.items()):
        print(f"  {club}: {'FOUND' if was_found else 'not found'}")

    out_path = Path(__file__).parent.parent / "data" / "ucl_uel_fotmob_long_tail_check.json"
    out_path.write_text(json.dumps(found, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/check_fotmob_unresolved_ucl_uel_clubs.py`
Expected: some fraction found (fotmob's unfiltered payload is broad), a
remainder genuinely not found in the last 14 days — those are the real
no-coverage candidates for this pass (true no-coverage, or their domestic
league simply didn't play in this window; note the caveat either way, same
as the domestic-cup spike's equivalent finding).

- [ ] **Step 3: Record findings and commit**

Append final counts and the genuinely-unresolved club list to the findings
doc.

```bash
git add scripts/check_fotmob_unresolved_ucl_uel_clubs.py \
  docs/superpowers/plans/2026-11-04-champions-league-preflight-findings.md \
  data/ucl_uel_fotmob_long_tail_check.json
git commit -m "spike: check fotmob coverage for UCL/UEL's long-tail clubs"
```

---

### Task 5: Discover a group/league-phase standings source

**Files:**
- Create: `scripts/discover_fotmob_standings_endpoint.py`
- Modify: `docs/superpowers/plans/2026-11-04-champions-league-preflight-findings.md`

This is a genuine open discovery task, not a known-answer check. Starting
point already established (free, during this planning session): fotmob's
real numeric league IDs for the current competitions are
**943230 (Champions League)** and **943229 (Europa League)**, confirmed
live via `/api/data/matches` (the `name`/`id` fields on a real recent
day's payload) — not guessed from public knowledge, which would have been
wrong (42/73 is the commonly-cited but unverified pair). `/api/data/leagues?id=943230`
responds `200` but returns a `null` body regardless of extra params tried
(`tab=table`, `season=...`) — the endpoint exists but doesn't resolve this
ID the way `/api/data/matches` does. `/api/leaguedetails?id=943230` 404s.

- [ ] **Step 1: Write a broader discovery script**

```python
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

import json

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


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it, and manually extend the candidate list if none hit**

Run: `python scripts/discover_fotmob_standings_endpoint.py`
If none of the candidates return real data: open fotmob.com/leagues/943230
in a browser, open its network tab, and find the actual XHR request the
page itself makes for the standings tab — the real path is whatever that
request shows, not a further guess. Record the real working URL and its
real response shape (which field holds the group/league-phase standings
table) once found.

- [ ] **Step 3: Record findings and commit**

Append to the findings doc: either the confirmed working endpoint + its
real shape, or an honest "not resolved in this pass" if no candidate
worked and the browser-network-tab method wasn't run — don't claim success
without a real 200 response containing recognizable standings data.

```bash
git add scripts/discover_fotmob_standings_endpoint.py \
  docs/superpowers/plans/2026-11-04-champions-league-preflight-findings.md
git commit -m "spike: discover (or document non-discovery of) a UCL/UEL standings source"
```

---

### Task 6: Consolidated write-up

**Files:**
- Modify: `docs/superpowers/plans/2026-11-04-champions-league-preflight-findings.md`
- Modify: `docs/superpowers/specs/2026-11-04-champions-league-recommendations-design.md`

- [ ] **Step 1: Write the consolidated recommendation**

Append a final section to the findings doc covering: what fraction of the
150-club real UCL/UEL opponent pool is reachable via football-data.co.uk
(Task 3), what fraction of the remainder fotmob can at least confirm exists
(Task 4), and what's genuinely uncovered — a concrete, data-backed answer
to "how wide does the Dixon-Coles pool actually need to be," replacing the
abstract "20-30 countries" estimate in the design spec with real numbers.
Also state plainly whether Task 5 found a usable standings source or not.

- [ ] **Step 2: Update the design spec's Section 8 open items**

In `docs/superpowers/specs/2026-11-04-champions-league-recommendations-design.md`,
replace Section 8's "Dixon-Coles pool breadth" and "Group/league-phase
standings source" bullets with a pointer to the real findings:

```markdown
**Resolved 2026-11-04:** see
`docs/superpowers/plans/2026-11-04-champions-league-preflight-findings.md`
for the real club-coverage breakdown and standings-source discovery result.
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-11-04-champions-league-preflight-findings.md \
  docs/superpowers/specs/2026-11-04-champions-league-recommendations-design.md
git commit -m "docs: consolidate UCL/UEL pre-flight findings, update design spec"
```
