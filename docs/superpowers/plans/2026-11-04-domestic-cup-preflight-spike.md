# Domestic Cup Pre-Flight Data Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, with real live API calls (not guesses), whether odds and
score data actually exist for the 5 second divisions (Championship, Segunda,
Serie B, 2.Bundesliga, Ligue 2) and the 6 domestic cups (FA Cup, Carabao Cup,
Copa del Rey, DFB-Pokal, Coupe de France, Coppa Italia) in scope for
`docs/superpowers/specs/2026-11-04-domestic-cup-recommendations-design.md`,
before any onboarding/ingestion work is built against them.

**Architecture:** Seven small, throwaway, standalone scripts under
`scripts/` (Tasks 1-7), following this codebase's existing one-off-script
convention (`scripts/pull_oddspapi_btts_corners.py`: `load_dotenv()`, read
the API key from env, plain `requests` calls, no production client classes),
plus a final consolidation task (Task 8) with no new script. Each script
makes a handful of real API/page calls and prints/saves exact findings.
Nothing here touches production code paths — this is discovery only, the
deliverable is a findings document that the later ingestion/onboarding plans
will be written against.

**Note on task structure:** this plan verifies external reality, not business
logic we own, so there is no red/green unit test per step. Each task instead
follows write-script → run-against-the-live-API → record-the-exact-output →
commit, which is this codebase's own established pattern for this kind of
work (see `scripts/pull_oddspapi_btts_corners.py`'s header comment, itself
the record of a prior live-discovery pass).

**Tech Stack:** `requests`, `python-dotenv`. Existing env vars:
`ODDSPAPI_API_KEY` (OddsPapi, tournament-ID-based, used for cup competitions
— same provider as the already-discovered UCL=7/UEL=679), `ODDS_API_KEY`
(The Odds API, sport-key-based, used for the 6 modeled leagues).

**Important distinction this plan checks separately — live serving vs.
training data:** `OddsPapi`'s `/v4/odds` and The Odds API's live endpoint are
both **current-odds-only, no historical replay**
(`app/backend/historical_odds_client.py`'s own docstring: *"The Odds API
(W07) is a live-current-odds-only service with no historical replay"*). The
6 leagues' actual *training* odds (`odds_h`/`odds_d`/`odds_a` in
`raw_matches`) come from a third, different source entirely:
football-data.co.uk CSVs (`B365H`/`B365D`/`B365A` columns, parsed in
`src/ingestion/football_data/loader.py:73-75`), covering 2016-08-13 onward.
OddsPapi does have a separate historical endpoint (`/v4/historical-odds`,
already used in `scripts/pull_oddspapi_btts_corners.py` for BTTS/corners),
which is the real candidate for domestic cups' training odds — football-data.co.uk
has no cup data at all, only round-robin league CSVs. So this plan checks
live and historical coverage as separate questions, not one: Tasks 2/4 check
live serving; Tasks 3/5 (new) check whether there's any training-data
foundation at all, which is the more fundamental question for the
cup-transfer model specifically.

---

### Task 1: Discover OddsPapi tournament IDs for the 6 domestic cups

**Files:**
- Create: `scripts/discover_domestic_cup_tournament_ids.py`
- Create: `docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md`

- [x] **Step 1: Write the discovery script**

```python
"""One-off spike (domestic cup pre-flight, 2026-11-04): discover OddsPapi
tournament IDs for the 6 domestic cups in scope. Follows the same
confirmed-live pattern already used for the 5 modeled leagues and UCL/UEL
(scripts/pull_oddspapi_btts_corners.py's LEAGUE_TOURNAMENT_IDS comment):
/v4/tournaments?sportId=10 lists every soccer tournament OddsPapi knows
about; fixture/tournament discovery calls are (almost) free, unlike
/v4/odds, so this script is safe to re-run.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ODDSPAPI_API_KEY")
if not API_KEY:
    sys.exit("No API key: set ODDSPAPI_API_KEY in .env.")
BASE_URL = "https://api.oddspapi.io"

# Name fragments + category, one per cup in scope. OddsPapi reuses generic
# names like "FA Cup"/"League Cup" across dozens of countries -- name alone
# is not enough, "categoryName" (the real per-tournament country/region
# field) is required to disambiguate.
CUP_NAME_FRAGMENTS = {
    "FA_CUP": (["fa cup"], "England"),
    "CARABAO_CUP": (["efl cup"], "England"),
    "COPA_DEL_REY": (["copa del rey"], "Spain"),
    "DFB_POKAL": (["dfb pokal"], "Germany"),
    "COUPE_DE_FRANCE": (["coupe de france"], "France"),
    "COPPA_ITALIA": (["coppa italia"], "Italy"),
}


def main() -> None:
    resp = requests.get(
        f"{BASE_URL}/v4/tournaments", params={"apiKey": API_KEY, "sportId": 10}, timeout=30,
    )
    resp.raise_for_status()
    tournaments = resp.json()
    print(f"Fetched {len(tournaments)} total soccer tournaments.")

    # Real field names confirmed live (not "id"/"name" as the inline
    # LEAGUE_TOURNAMENT_IDS comments elsewhere implied): tournamentId,
    # tournamentName, categoryName.
    matches: dict[str, list[dict]] = {cup: [] for cup in CUP_NAME_FRAGMENTS}
    for t in tournaments:
        name_lower = str(t.get("tournamentName", "")).lower()
        category = str(t.get("categoryName", ""))
        for cup, (fragments, expected_category) in CUP_NAME_FRAGMENTS.items():
            if any(frag in name_lower for frag in fragments) and category == expected_category:
                matches[cup].append({
                    "id": t.get("tournamentId"), "name": t.get("tournamentName"), "category": category,
                })

    for cup, found in matches.items():
        print(f"\n{cup}:")
        if not found:
            print("  NO MATCH FOUND")
        for f in found:
            print(f"  id={f['id']}  name={f['name']!r}  category={f['category']!r}")

    out_path = Path(__file__).parent.parent / "data" / "domestic_cup_tournament_discovery.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(matches, indent=2))
    print(f"\nSaved raw matches to {out_path}")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run it against the live API**

Run: `python scripts/discover_domestic_cup_tournament_ids.py`
Expected: prints one block per cup with `id=<int> name='<real name>'
category='<real category>'`, or `NO MATCH FOUND` for any cup OddsPapi
doesn't carry under that name+category combination. This is real-world
discovery — a `NO MATCH FOUND` is a valid, important finding, not a bug.
Live-verified 2026-11-04: all 6 resolved to a single unambiguous ID once
category was added as a filter (name alone matched 14 "FA Cup"-named
tournaments across different countries) — FA Cup=19, EFL Cup=21,
Copa del Rey=329, DFB Pokal=217, Coupe de France=335, Coppa Italia=328.

- [x] **Step 3: Record findings**

Create `docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md`
with this starting content (fill in the real `id=`/`name=` values or
`NOT FOUND` from Step 2's actual output — do not leave placeholders):

```markdown
# Domestic Cup Pre-Flight Findings

Live-verified 2026-11-04. Each section records what Step 2's real API output
showed, not an assumption.

## OddsPapi tournament IDs (domestic cups)

| Cup | Tournament ID | OddsPapi name |
|---|---|---|
| FA Cup | <fill in> | <fill in> |
| Carabao Cup | <fill in> | <fill in> |
| Copa del Rey | <fill in> | <fill in> |
| DFB-Pokal | <fill in> | <fill in> |
| Coupe de France | <fill in> | <fill in> |
| Coppa Italia | <fill in> | <fill in> |
```

- [x] **Step 4: Commit**

```bash
git add scripts/discover_domestic_cup_tournament_ids.py \
  docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md
git commit -m "spike: discover OddsPapi tournament IDs for 6 domestic cups"
```

---

### Task 2: Check real OddsPapi fixture + 1X2 odds coverage for each cup

**Files:**
- Create: `scripts/check_domestic_cup_odds_coverage.py`
- Modify: `docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md`

This task spends real OddsPapi credits (one `/v4/odds` call per cup = up to 6
credits, against the shared 250/month free-tier budget already partly used
by the UCL/UEL historical backfill and the live BTTS/corners pull). Check
the current balance first so this can't accidentally blow the budget.

- [x] **Step 1: Check current OddsPapi credit usage before spending more**

Run:
```bash
cat data/oddspapi_credit_counter.json 2>/dev/null || echo "no counter file yet -- 0 used"
```
Expected: JSON with a `credits_used`/similar field, or the fallback message.
If `credits_used` is within ~10 credits of 250, stop here and flag it in the
findings doc instead of spending more — this task only needs 6.

- [x] **Step 2: Write the coverage-check script**

```python
"""One-off spike (domestic cup pre-flight, 2026-11-04): for each domestic
cup tournament ID discovered in Task 1, fetch real fixtures and check
whether real 1X2 odds exist. Reuses the already-confirmed-live market ID
(MATCH_WINNER_MARKET_ID = "101", outcomes 101/102/103 for home/draw/away --
discovered for the UCL/UEL backfill, not re-guessed here) and the
already-confirmed statusId gotcha (statusId=0 = Pre-Game; statusId=1 always
returns zero rows, per W200/W236).

Spends real credits: one /v4/odds call per cup with at least one upcoming
fixture. Run Task 1's discovery output through this script's
TOURNAMENT_IDS dict below -- fill those in from the real Task 1 output
before running.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ODDSPAPI_API_KEY")
if not API_KEY:
    sys.exit("No API key: set ODDSPAPI_API_KEY in .env.")
BASE_URL = "https://api.oddspapi.io"
MATCH_WINNER_MARKET_ID = "101"
MATCH_WINNER_OUTCOME_HOME = "101"
MATCH_WINNER_OUTCOME_DRAW = "102"
MATCH_WINNER_OUTCOME_AWAY = "103"

# Fill in from Task 1's real output -- do not guess, leave a cup out of this
# dict entirely if Task 1 found NO MATCH for it.
TOURNAMENT_IDS: dict[str, int] = {
    # "FA_CUP": 123,
}


def _get(path: str, **params) -> tuple[int, dict | list]:
    params["apiKey"] = API_KEY
    resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
    try:
        body = resp.json() if resp.content else {}
    except ValueError:
        body = {}
    return resp.status_code, body


def _last_price(outcomes: dict, outcome_id: str) -> float | None:
    outcome = outcomes.get(outcome_id)
    if not outcome:
        return None
    # "players"->"0" is a single price dict in the real payload (confirmed
    # live, 2026-11-04: {"active": true, ..., "price": 9.86}), not a list of
    # historical "ticks" as production's own _last_price
    # (app/backend/oddspapi_client.py) assumes.
    player = outcome.get("players", {}).get("0", {})
    return player.get("price")


def main() -> None:
    findings = {}
    for cup, tournament_id in TOURNAMENT_IDS.items():
        status, fixtures = _get("/v4/fixtures", tournamentId=tournament_id, statusId=0)
        if status == 404:
            fixtures = []
        elif not isinstance(fixtures, list):
            print(f"{cup}: unexpected fixtures response (status={status}): {fixtures}")
            fixtures = []

        print(f"\n{cup}: {len(fixtures)} upcoming (statusId=0) fixtures")
        if not fixtures:
            findings[cup] = {"fixture_count": 0, "has_1x2_odds": None}
            continue

        # /v4/fixtures does not return results sorted by date -- picking
        # fixtures[0] directly can land on an arbitrary far-future fixture
        # with no odds posted yet, which looks like "no coverage" but is
        # really just "too early for this specific match." Sort by
        # startTime and use the nearest one instead.
        fixtures_sorted = sorted(fixtures, key=lambda f: f.get("startTime", "9999"))
        sample = fixtures_sorted[0]
        fixture_id = str(sample["fixtureId"])
        print(f"  nearest fixture start_time={sample.get('startTime')}")
        print(f"  ALERT: spending 1 real OddsPapi credit -- /v4/odds for fixture {fixture_id}")
        time.sleep(5.0)  # same cooldown convention as pull_oddspapi_btts_corners.py
        odds_status, odds_payload = _get("/v4/odds", fixtureId=fixture_id, bookmakers="pinnacle")
        outcomes = {}
        if odds_status == 200 and isinstance(odds_payload, dict):
            # The real top-level key is "bookmakerOdds", not "bookmakers" --
            # confirmed live, 2026-11-04, against a real payload where
            # market 101 (match-winner) had real prices sitting under
            # "bookmakerOdds" and nothing under "bookmakers". The same wrong
            # key also exists in production (app/backend/oddspapi_client.py's
            # _parse_corners_odds) and two other scripts -- out of scope to
            # fix as part of this plan, flagged separately.
            outcomes = (
                odds_payload.get("bookmakerOdds", {}).get("pinnacle", {})
                .get("markets", {}).get(MATCH_WINNER_MARKET_ID, {}).get("outcomes", {})
            )
        home = _last_price(outcomes, MATCH_WINNER_OUTCOME_HOME)
        draw = _last_price(outcomes, MATCH_WINNER_OUTCOME_DRAW)
        away = _last_price(outcomes, MATCH_WINNER_OUTCOME_AWAY)
        has_odds = all(v is not None for v in (home, draw, away))
        print(f"  sample fixture {fixture_id}: home={home} draw={draw} away={away} -> has_1x2_odds={has_odds}")
        findings[cup] = {"fixture_count": len(fixtures), "has_1x2_odds": has_odds}

    out_path = Path(__file__).parent.parent / "data" / "domestic_cup_odds_coverage.json"
    out_path.write_text(json.dumps(findings, indent=2))
    print(f"\nSaved findings to {out_path}")


if __name__ == "__main__":
    main()
```

- [x] **Step 3: Fill in `TOURNAMENT_IDS` from Task 1's real output, then run**

Run: `python scripts/check_domestic_cup_odds_coverage.py`
Expected: one block per cup showing `fixture_count` and `has_1x2_odds`
(`True`/`False`/`None` if there were no upcoming fixtures to check at all).
Live-verified 2026-11-04: two real bugs caught and fixed while running
this (both documented inline above) — the response's real top-level key is
`bookmakerOdds` not `bookmakers`, and `players["0"]` is a single price dict
not a list of ticks. Both bugs also exist in production code
(`app/backend/oddspapi_client.py`'s `_parse_corners_odds`), masking each
other there (the wrong key always produced an empty `outcomes` dict, so the
list-indexing bug was never reached with real data) — flagged separately,
not fixed as part of this plan. Real result: FA Cup and Carabao Cup
confirmed real 1X2 odds on their nearest fixture; Copa del Rey, DFB-Pokal,
Coupe de France, and Coppa Italia came back with no price posted yet on a
single-sample check (Copa del Rey's "nearest" fixture under `statusId=0`
had a start time a month in the past — likely a postponed/rescheduled match
still flagged pre-game, a data-quality wrinkle for later ingestion work, not
resolved here).

- [x] **Step 4: Append real findings to the findings doc**

Add to `docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md`:

```markdown
## OddsPapi odds coverage (domestic cups)

| Cup | Upcoming fixtures (statusId=0) | Has real 1X2 odds |
|---|---|---|
| FA Cup | <fill in> | <fill in> |
| Carabao Cup | <fill in> | <fill in> |
| Copa del Rey | <fill in> | <fill in> |
| DFB-Pokal | <fill in> | <fill in> |
| Coupe de France | <fill in> | <fill in> |
| Coppa Italia | <fill in> | <fill in> |
```

- [x] **Step 5: Commit**

```bash
git add scripts/check_domestic_cup_odds_coverage.py \
  docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md
git commit -m "spike: check real OddsPapi odds coverage for 6 domestic cups"
```

---

### Task 3: Check OddsPapi *historical* odds coverage for domestic cups (training data)

**Files:**
- Create: `scripts/check_domestic_cup_historical_odds.py`
- Modify: `docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md`

Task 2 checked live (`/v4/odds`) coverage — relevant to serving recommendations
once this ships, but irrelevant to training the cup-transfer model, which
needs *historical* results+odds. This task checks OddsPapi's actual
historical endpoint (`/v4/historical-odds`, fixture-ID-based), confirmed
live for UCL/UEL with a 2026-01-01 cutoff — that cutoff is specific to
UCL/UEL and must not be assumed for the 6 domestic cups without checking.

- [x] **Step 1: Re-check current OddsPapi credit usage**

Run: `cat data/oddspapi_credit_counter.json 2>/dev/null || echo "no counter file yet -- 0 used"`
Expected: confirm there's budget left after Task 2's spend before this task
spends more (one `/v4/historical-odds` call per cup with a finished fixture
= up to 6 more credits).

- [x] **Step 2: Write the historical-coverage script**

```python
"""One-off spike (domestic cup pre-flight, 2026-11-04): check whether
OddsPapi's historical-odds endpoint (/v4/historical-odds) -- the real
source for *training* odds, not /v4/odds which is live-only -- actually
covers the 6 domestic cups, and from what date. Mirrors
scripts/pull_oddspapi_btts_corners.py's already-confirmed-live call shape
and the already-confirmed match-winner market ID (101)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ODDSPAPI_API_KEY")
if not API_KEY:
    sys.exit("No API key: set ODDSPAPI_API_KEY in .env.")
BASE_URL = "https://api.oddspapi.io"
MATCH_WINNER_MARKET_ID = "101"
MATCH_WINNER_OUTCOME_HOME = "101"
MATCH_WINNER_OUTCOME_DRAW = "102"
MATCH_WINNER_OUTCOME_AWAY = "103"

# Same tournament IDs discovered in Task 1 -- fill in from that real output.
TOURNAMENT_IDS: dict[str, int] = {
    # "FA_CUP": 123,
}


def _get(path: str, **params) -> tuple[int, dict | list]:
    params["apiKey"] = API_KEY
    resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
    try:
        body = resp.json() if resp.content else {}
    except ValueError:
        body = {}
    return resp.status_code, body


def _last_price(outcomes: dict, outcome_id: str) -> float | None:
    outcome = outcomes.get(outcome_id)
    if not outcome:
        return None
    ticks = outcome.get("players", {}).get("0", [])
    return ticks[-1].get("price") if ticks else None


def main() -> None:
    findings = {}
    for cup, tournament_id in TOURNAMENT_IDS.items():
        # statusId=2 = finished, same convention pull_oddspapi_btts_corners.py
        # already confirmed live for the 5 modeled leagues.
        status, fixtures = _get("/v4/fixtures", tournamentId=tournament_id, statusId=2)
        if not isinstance(fixtures, list):
            print(f"{cup}: unexpected fixtures response (status={status}): {fixtures}")
            findings[cup] = {"finished_fixture_count": 0, "earliest_with_odds": None}
            continue

        fixtures_sorted = sorted(fixtures, key=lambda f: f.get("startTime", ""))
        print(f"\n{cup}: {len(fixtures_sorted)} finished fixtures total")
        if not fixtures_sorted:
            findings[cup] = {"finished_fixture_count": 0, "earliest_with_odds": None}
            continue

        # Check the oldest and the newest finished fixture -- enough to tell
        # whether historical odds exist at all and roughly how far back,
        # without spending a full per-match sweep's worth of credits here.
        checked = []
        for fixture in (fixtures_sorted[0], fixtures_sorted[-1]):
            fixture_id = str(fixture["fixtureId"])
            start_time = fixture.get("startTime")
            time.sleep(5.0)
            odds_status, payload = _get("/v4/historical-odds", fixtureId=fixture_id, bookmakers="pinnacle")
            outcomes = {}
            if odds_status == 200 and isinstance(payload, dict):
                outcomes = (
                    payload.get("bookmakers", {}).get("pinnacle", {})
                    .get("markets", {}).get(MATCH_WINNER_MARKET_ID, {}).get("outcomes", {})
                )
            has_odds = all(
                _last_price(outcomes, oid) is not None
                for oid in (MATCH_WINNER_OUTCOME_HOME, MATCH_WINNER_OUTCOME_DRAW, MATCH_WINNER_OUTCOME_AWAY)
            )
            print(f"  fixture {fixture_id} ({start_time}): status={odds_status} has_1x2_odds={has_odds}")
            checked.append({"start_time": start_time, "has_odds": has_odds})

        findings[cup] = {"finished_fixture_count": len(fixtures_sorted), "checked_samples": checked}

    out_path = Path(__file__).parent.parent / "data" / "domestic_cup_historical_odds_coverage.json"
    out_path.write_text(json.dumps(findings, indent=2))
    print(f"\nSaved findings to {out_path}")


if __name__ == "__main__":
    main()
```

- [x] **Step 3: Fill in `TOURNAMENT_IDS` from Task 1's real output, then run**

Run: `python scripts/check_domestic_cup_historical_odds.py`
Expected: per cup, whether the oldest and newest finished fixtures sampled
have real 1X2 odds. If the oldest sample lacks odds but the newest has them,
that's a real cutoff date, same shape as UCL/UEL's confirmed 2026-01-01 —
narrow it down with a couple more sampled fixtures between the two dates if
the gap matters for deciding whether there's enough training history.
Live-verified 2026-11-04: every cup's oldest (2024) sample returned 404,
consistent with the known 2026-01-01 cutoff. 4/6 cups' newest sample (FA
Cup, Carabao Cup, DFB-Pokal, Coppa Italia) confirmed real 1X2 odds; Copa del
Rey and Coupe de France's single newest sample came back empty — not
conclusive on one sample each, re-check with a wider sample before treating
as "no coverage."

- [x] **Step 4: Record findings and commit**

Append to the findings doc:

```markdown
## OddsPapi historical odds coverage (domestic cups, for training)

| Cup | Finished fixtures (statusId=2) | Oldest sample has odds | Newest sample has odds |
|---|---|---|---|
| FA Cup | 571 | False (2024, pre-cutoff) | True |
| Carabao Cup | 235 | False (2024, pre-cutoff) | True |
| Copa del Rey | 323 | False (2024, pre-cutoff) | False (one sample, not conclusive) |
| DFB-Pokal | 148 | False (2024, pre-cutoff) | True |
| Coupe de France | 411 | False (2024, pre-cutoff) | False (one sample, not conclusive) |
| Coppa Italia | 98 | False (2024, pre-cutoff) | True |
```

```bash
git add scripts/check_domestic_cup_historical_odds.py \
  docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md
git commit -m "spike: check OddsPapi historical odds coverage for 6 domestic cups"
```

---

### Task 4: Discover The Odds API sport_keys for the 5 second divisions

**Files:**
- Create: `scripts/discover_second_division_sport_keys.py`
- Modify: `docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md`

Mirrors the already-documented live-confirmation pattern in
`app/backend/odds_sport_keys.py` (every existing entry there has a comment
recording the exact confirmed `{"key": ..., "title": ..., "active": ...}`
response that justified it — same discipline here, including the D1 gotcha:
`soccer_germany_bundesliga2` (2. Bundesliga) and `soccer_austria_bundesliga`
both exist in the same listing and must not be confused.

- [x] **Step 1: Write the discovery script**

```python
"""One-off spike (domestic cup pre-flight, 2026-11-04): discover The Odds
API sport_keys for the 5 second divisions in scope, the same way every
existing entry in app/backend/odds_sport_keys.py was confirmed (live
/v4/sports/?all=true listing, matched by real title, not a name-pattern
guess)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    sys.exit("No API key: set ODDS_API_KEY in .env.")
BASE_URL = "https://api.the-odds-api.com"

# Name fragments, one per second division in scope. Confirm against the real
# listing below -- The Odds API's own key naming isn't known in advance.
DIVISION_NAME_FRAGMENTS = {
    "CHAMPIONSHIP": ["championship"],
    # "segunda" alone matches nothing live -- the real title is
    # "La Liga 2 - Spain" (key soccer_spain_segunda_division), confirmed
    # 2026-11-04. Both fragments kept so the script is self-explanatory
    # about why.
    "SEGUNDA": ["segunda", "la liga 2"],
    "SERIE_B": ["serie b"],
    "BUNDESLIGA_2": ["bundesliga"],  # must disambiguate from top-flight D1 + Austria, see Step 2
    "LIGUE_2": ["ligue 2"],
}


def main() -> None:
    resp = requests.get(f"{BASE_URL}/v4/sports/", params={"apiKey": API_KEY, "all": "true"}, timeout=30)
    resp.raise_for_status()
    sports = resp.json()
    print(f"Fetched {len(sports)} total sports/leagues.")

    matches: dict[str, list[dict]] = {d: [] for d in DIVISION_NAME_FRAGMENTS}
    for s in sports:
        if not str(s.get("key", "")).startswith("soccer_"):
            continue
        title_lower = str(s.get("title", "")).lower()
        for division, fragments in DIVISION_NAME_FRAGMENTS.items():
            if any(frag in title_lower for frag in fragments):
                matches[division].append(
                    {"key": s.get("key"), "title": s.get("title"), "active": s.get("active")}
                )

    for division, found in matches.items():
        print(f"\n{division}:")
        if not found:
            print("  NO MATCH FOUND")
        for f in found:
            print(f"  key={f['key']}  title={f['title']!r}  active={f['active']}")

    out_path = Path(__file__).parent.parent / "data" / "second_division_sport_key_discovery.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(matches, indent=2))
    print(f"\nSaved raw matches to {out_path}")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run it, disambiguating BUNDESLIGA_2 by hand**

Run: `python scripts/discover_second_division_sport_keys.py`
Expected: `BUNDESLIGA_2` prints multiple candidates (at minimum
`soccer_germany_bundesliga` (top flight, already D1's key — exclude) and
`soccer_germany_bundesliga2`). Pick the one whose real `title` says
"2. Bundesliga" / "Bundesliga 2 - Germany", not the top-flight one.
Live-verified 2026-11-04: all 5 resolved to a single unambiguous key.
Incidental finding: The Odds API also carries `soccer_spain_copa_del_rey` —
a second potential live-odds source for that cup, alongside OddsPapi — not
pursued further here, just noted.

- [x] **Step 3: Record findings**

Append to the findings doc:

```markdown
## The Odds API sport_keys (second divisions)

| Division | sport_key | Real title | active |
|---|---|---|---|
| Championship | soccer_efl_champ | Championship | True |
| Segunda División | soccer_spain_segunda_division | La Liga 2 - Spain | True |
| Serie B | soccer_italy_serie_b | Serie B - Italy | True |
| 2. Bundesliga | soccer_germany_bundesliga2 | Bundesliga 2 - Germany | True |
| Ligue 2 | soccer_france_ligue_two | Ligue 2 - France | True |
```

- [x] **Step 4: Commit**

```bash
git add scripts/discover_second_division_sport_keys.py \
  docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md
git commit -m "spike: discover The Odds API sport_keys for 5 second divisions"
```

---

### Task 5: Confirm real LIVE odds rows exist for each second-division sport_key

**Files:**
- Create: `scripts/check_second_division_odds_coverage.py`
- Modify: `docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md`

Being listed in `/v4/sports/` with `active: true` means the sport exists —
it does not mean a bookmaker is actually offering odds on it right now. This
step confirms real odds rows come back, reusing the existing production
`OddsAPIClient.get_odds()` rather than re-implementing the call.

- [x] **Step 1: Write the coverage-check script**

```python
"""One-off spike (domestic cup pre-flight, 2026-11-04): confirm The Odds API
actually returns real odds rows (not just a listed-but-dormant sport) for
each second division discovered in Task 4. Reuses the existing production
OddsAPIClient instead of re-implementing the HTTP call."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.backend.odds_api_client import CreditCounter, OddsAPIClient  # noqa: E402

load_dotenv()

# ODDS_API_KEY and _2 confirmed exhausted live, 2026-11-04 (401
# OUT_OF_USAGE_CREDITS, x-requests-remaining=0) against soccer_epl itself --
# a production-already-live league, so this is unrelated to second
# divisions. Found while testing this task, flagged separately (not fixed
# here). _3 confirmed to have real remaining capacity, used here instead.
API_KEY = os.environ.get("ODDS_API_KEY_3")
if not API_KEY:
    sys.exit("No API key: set ODDS_API_KEY_3 in .env.")

# Fill in from Task 4's real output.
SPORT_KEYS: dict[str, str] = {
    # "CHAMPIONSHIP": "soccer_england_championship",
}


def main() -> None:
    # A fresh, zeroed CreditCounter -- this spike's own handful of calls
    # (cost = len(markets) * len(regions) = 1 each with the client's
    # defaults) shouldn't be tallied against the real production counter
    # persisted at data/odds_api_credit_counter_3.json, and doesn't need to
    # be -- reconciled manually afterward instead.
    client = OddsAPIClient(api_key=API_KEY, credit_counter=CreditCounter())
    findings = {}
    for division, sport_key in SPORT_KEYS.items():
        print(f"ALERT: spending 1 real Odds API credit -- get_odds({sport_key!r})")
        odds = client.get_odds(sport_key=sport_key)
        count = len(odds) if odds else 0
        print(f"{division} ({sport_key}): {count} real odds rows")
        findings[division] = {"sport_key": sport_key, "odds_row_count": count}

    out_path = Path(__file__).parent.parent / "data" / "second_division_odds_coverage.json"
    out_path.write_text(json.dumps(findings, indent=2))
    print(f"\nSaved findings to {out_path}")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Fill in `SPORT_KEYS` from Task 4's real output, then run**

Run: `python scripts/check_second_division_odds_coverage.py`
Expected: one line per division with a real `odds_row_count`. `0` is a valid,
important finding (thin/no current liquidity), not a script bug — if every
division returns `0`, re-run closer to a matchday before concluding there's
no coverage at all, since pre-match odds for lower-tier leagues may post
later than for the 6 modeled leagues. Live-verified 2026-11-04: all 5
confirmed real odds rows (9-12 matches each) — a genuine positive signal
for live serving.

**Separate, higher-severity finding surfaced while running this task:**
`ODDS_API_KEY` and `ODDS_API_KEY_2` are both exhausted
(`401 OUT_OF_USAGE_CREDITS`, provider-reported `x-requests-remaining=0`)
against `soccer_epl` itself — a league already live in production, so this
is unrelated to domestic cups or second divisions. Checked all 8 configured
keys: `_3` has 169/500 remaining, `_4` through `_8` have 499/500 each
(barely touched) — the fallback rotation has real capacity, so this reads
as normal rotation past two exhausted keys at the front of a healthy chain,
not an active production outage. Not fixed or root-caused here (why keys
1-2 specifically are exhausted — expected monthly usage vs. something
abnormal — is a separate investigation). The two confirmed-dead keys'
local credit-counter files were updated to reflect real exhaustion so
production doesn't keep retrying them blind.

- [x] **Step 3: Record findings and commit**

Append the real counts to the findings doc's sport_key table (add an
"Odds rows found" column), then:

```bash
git add scripts/check_second_division_odds_coverage.py \
  docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md
git commit -m "spike: confirm real odds rows for 5 second divisions"
```

---

### Task 6: Check football-data.co.uk coverage for second divisions (training data)

**Files:**
- Create: `scripts/check_second_division_historical_coverage.py`
- Modify: `docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md`

Task 5 checked live odds via The Odds API — relevant to serving, not to
training the cup-transfer model, which needs each second division's
rolling-form features computed the same way the 6 leagues' already are: from
football-data.co.uk CSVs. This task checks whether each country's
football-data.co.uk page actually lists a second-division CSV, without
guessing the URL/filename — `main.py`'s existing per-league override table
(`_SCRAPE_SOURCE_OVERRIDE_BY_LEAGUE`) already proves each country's page is
one page covering several divisions (e.g. `englandm.php` lists E0 alongside
lower English divisions), so the second divisions are plausibly already on
pages this codebase has scraped before — confirm, don't assume.

- [x] **Step 1: Write the coverage-check script**

```python
"""One-off spike (domestic cup pre-flight, 2026-11-04): check whether each
country's football-data.co.uk page (the same pages main.py's
_SCRAPE_SOURCE_OVERRIDE_BY_LEAGUE already points at for the 6 leagues) also
lists a second-division CSV, and what season range it covers. Read-only --
does not run the actual scraper, just inspects the page for a CSV link."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests

# Same pages main.py already uses for the top-flight leagues (englandm.php
# has no override entry because E0 is football-data.co.uk's own default --
# see main.py's _SCRAPE_SOURCE_OVERRIDE_BY_LEAGUE comment).
COUNTRY_PAGES = {
    "CHAMPIONSHIP": "https://www.football-data.co.uk/englandm.php",
    "SEGUNDA": "https://www.football-data.co.uk/spainm.php",
    "SERIE_B": "https://www.football-data.co.uk/italym.php",
    "BUNDESLIGA_2": "https://www.football-data.co.uk/germanym.php",
    "LIGUE_2": "https://www.football-data.co.uk/francem.php",
}

# Division codes to look for in each page's CSV links -- football-data.co.uk's
# own convention (E0=Premier League, E1=Championship, SP1/SP2, I1/I2, D1/D2,
# F1/F2 per the existing adding-a-league pattern). Confirm these appear as
# real links below, don't assume the convention holds for every country.
EXPECTED_CSV_CODE = {
    "CHAMPIONSHIP": "E1",
    "SEGUNDA": "SP2",
    "SERIE_B": "I2",
    "BUNDESLIGA_2": "D2",
    "LIGUE_2": "F2",
}


def main() -> None:
    findings = {}
    for division, page_url in COUNTRY_PAGES.items():
        resp = requests.get(page_url, timeout=30)
        resp.raise_for_status()
        html = resp.text
        code = EXPECTED_CSV_CODE[division]
        # football-data.co.uk links each division's current-season CSV as
        # mmz4281/<season>/<CODE>.csv -- look for the exact code as its own
        # path segment, not a loose substring (avoids e.g. "E1" matching
        # inside an unrelated "E10" or similar).
        csv_links = sorted(set(re.findall(rf"mmz4281/\d+/{code}\.csv", html)))
        print(f"{division} ({code}): {len(csv_links)} CSV link(s) found on {page_url}")
        for link in csv_links:
            print(f"  {link}")
        findings[division] = {"page_url": page_url, "csv_code": code, "csv_links_found": csv_links}

    out_path = Path(__file__).parent.parent / "data" / "second_division_historical_coverage.json"
    out_path.write_text(json.dumps(findings, indent=2))
    print(f"\nSaved findings to {out_path}")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run it**

Run: `python scripts/check_second_division_historical_coverage.py`
Expected: one or more real CSV links per division (football-data.co.uk
typically only links the *current* season per page — historical seasons
live under the same `mmz4281/<season>/` pattern with other season codes,
which a later ingestion plan would enumerate, not this spike). A division
with zero links found means either the code guess is wrong (check the page
by hand in a browser before concluding "no data") or that division genuinely
isn't covered. Live-verified 2026-11-04: all 5 divisions found far more
than "the current season" — the full historical archive, back to the
1990s, is linked on each page (30-34 season CSVs each). This substantially
de-risks the training-data question for Option A.

- [x] **Step 3: Record findings and commit**

Append to the findings doc:

```markdown
## football-data.co.uk coverage (second divisions, for training)

| Division | Page | Expected CSV code | Seasons found | Earliest season |
|---|---|---|---|---|
| Championship | englandm.php | E1 | 34 | 1993/94 |
| Segunda División | spainm.php | SP2 | 31 | 1996/97 |
| Serie B | italym.php | I2 | 30 | 1997/98 |
| 2. Bundesliga | germanym.php | D2 | 34 | 1993/94 |
| Ligue 2 | francem.php | F2 | 31 | 1996/97 |
```

```bash
git add scripts/check_second_division_historical_coverage.py \
  docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md
git commit -m "spike: check football-data.co.uk historical coverage for 5 second divisions"
```

---

### Task 7: Check fotmob's raw payload for second-division and cup coverage

**Files:**
- Create: `scripts/check_fotmob_lower_tier_coverage.py`
- Modify: `docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md`

This determines whether the Dixon-Coles fallback's score-only ingestion path
(design doc Section 4, bucket 2) has a viable source. `fetch_all_matches()`
(`src/ingestion/fotmob/fetcher.py:181`) already discards which competition
each match belongs to — it calls `_parse_finished_matches()`, which keeps
only `fotmob_match_id`/`match_date`/`home_team`/`away_team`. This script
reads the raw payload's `leagues` list directly instead, so competition
identity isn't lost before it can be checked.

- [x] **Step 1: Write the coverage-check script**

```python
"""One-off spike (domestic cup pre-flight, 2026-11-04): check whether
fotmob's /api/data/matches payload (the same unfiltered endpoint
fetch_all_matches() already uses for UCL/UEL, per US#212's docstring) also
carries the 5 second divisions and 6 domestic cups in scope -- reading the
raw 'leagues' list directly, since fetch_all_matches()/_parse_finished_matches()
already discard league identity before this check could run."""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion.fotmob.fetcher import _fetch_matches_payload  # noqa: E402

NAME_FRAGMENTS = {
    "CHAMPIONSHIP": ["championship"],
    "SEGUNDA": ["segunda", "laliga 2", "la liga 2"],
    "SERIE_B": ["serie b"],
    "BUNDESLIGA_2": ["2. bundesliga", "bundesliga 2"],
    "LIGUE_2": ["ligue 2"],
    "FA_CUP": ["fa cup"],
    "CARABAO_CUP": ["carabao", "efl cup", "league cup"],
    "COPA_DEL_REY": ["copa del rey"],
    "DFB_POKAL": ["dfb", "pokal"],
    "COUPE_DE_FRANCE": ["coupe de france"],
    "COPPA_ITALIA": ["coppa italia"],
}


def main() -> None:
    today = date.today()
    found: dict[str, set[str]] = {k: set() for k in NAME_FRAGMENTS}

    # Check the last 7 days -- enough to catch at least one matchday for
    # most of these, without a long multi-week backfill-style loop.
    for offset in range(7):
        day = today - timedelta(days=offset)
        payload = _fetch_matches_payload(day, delay=1.0)
        if payload is None:
            continue
        for league_entry in payload.get("leagues", []):
            name = str(league_entry.get("name", ""))
            name_lower = name.lower()
            for key, fragments in NAME_FRAGMENTS.items():
                if any(frag in name_lower for frag in fragments):
                    found[key].add(name)

    for key, names in found.items():
        if names:
            print(f"{key}: FOUND -- real league names seen: {sorted(names)}")
        else:
            print(f"{key}: NOT FOUND in the last 7 days")

    out_path = Path(__file__).parent.parent / "data" / "fotmob_lower_tier_coverage.json"
    out_path.write_text(json.dumps({k: sorted(v) for k, v in found.items()}, indent=2))
    print(f"\nSaved findings to {out_path}")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run it**

Run: `python scripts/check_fotmob_lower_tier_coverage.py`
Expected: `FOUND` or `NOT FOUND` per competition, based on real matches in the
last 7 days. A cup showing `NOT FOUND` may simply mean no round was played
that week — note this caveat rather than treating it as a final answer if
any domestic cup isn't currently mid-round. Live-verified 2026-11-04: all 5
second divisions and 4/6 cups found (loose name fragments also pulled in
unrelated noise for CHAMPIONSHIP/CARABAO_CUP/DFB_POKAL — the real
competition was still clearly present among the matches for each). Coupe
de France and Coppa Italia weren't found this week, consistent with their
OddsPapi-confirmed next fixtures being 1-4 weeks out (Task 2) — read as
"no round this week," not "unavailable."

- [x] **Step 3: Record findings and commit**

Append to the findings doc:

```markdown
## FotMob raw payload coverage (last 7 days, 2026-11-04)

| Competition | Found in fotmob's unfiltered payload |
|---|---|
| Championship | True ("Championship", among unrelated same-named leagues) |
| Segunda | True ("Segunda Division") |
| Serie B | True |
| 2. Bundesliga | True |
| Ligue 2 | True |
| FA Cup | True |
| Carabao Cup | True ("EFL Cup", "League Cup") |
| Copa del Rey | True |
| DFB-Pokal | False this week (women's + an unrelated Danish cup matched, not the men's comp) |
| Coupe de France | False this week (consistent with next fixture being 10 days out) |
| Coppa Italia | False this week (consistent with next fixture being ~4 weeks out) |
```

```bash
git add scripts/check_fotmob_lower_tier_coverage.py \
  docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md
git commit -m "spike: check fotmob raw payload coverage for lower-tier competitions"
```

---

### Task 8: Consolidated go/no-go write-up

**Files:**
- Modify: `docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md`
- Modify: `docs/superpowers/specs/2026-11-04-domestic-cup-recommendations-design.md`

- [x] **Step 1: Write the consolidated recommendation**

Append a final section to the findings doc, using the real data gathered in
Tasks 1-7 (not a restatement of the task list). Address live-serving
coverage and training-data coverage **separately** for each competition —
a competition can be fine for one and not the other (e.g. real live odds
but no historical odds deep enough to train on yet, same shape as the
already-known UCL/UEL 2026-01-01 cutoff):

```markdown
## Go/no-go recommendation

For each of the 11 competitions in scope, state plainly:
- Training-data readiness (Tasks 3/6 for cups/second-divisions respectively)
  -- does a real historical source exist, and how far back/how many
  fixtures, for the cup-transfer model and Dixon-Coles pool to train on.
- Live-serving readiness (Tasks 2/5) -- does a real current-odds source
  exist, for once this ships.
- Dixon-Coles-pool breadth (Task 7) -- is this competition visible at all
  in fotmob's unfiltered payload, for the score-only fallback path.

State whether Plan 2 onward (registry/schema skeleton, cup ingestion,
Dixon-Coles pool, league onboardings) should proceed for each competition,
proceed with a flagged caveat (e.g. thin-but-real odds, short historical
window), or be dropped from this pass entirely -- citing the specific
finding from Tasks 1-7 that justifies each call, not a general impression.
```

- [x] **Step 2: Update the design spec's Section 4 pre-flight note**

In `docs/superpowers/specs/2026-11-04-domestic-cup-recommendations-design.md`,
Section 4, replace the "Pre-flight check before committing further build
time" paragraph's final sentence with a pointer to the real result:

```markdown
**Resolved 2026-11-04:** see
`docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md` for
the live-verified odds/score coverage per competition and the resulting
go/no-go call for each.
```

- [x] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-11-04-domestic-cup-preflight-findings.md \
  docs/superpowers/specs/2026-11-04-domestic-cup-recommendations-design.md
git commit -m "docs: consolidate domestic cup pre-flight findings, update design spec"
```
