# UCL/UEL Ingestion Cherry-Pick Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get a real, correctly-canonicalized `raw_matches` table for UCL/UEL
in this branch — reusing the abandoned branch's already-paid-for 368-match
historical odds (zero new OddsPapi credits) and its hand-verified
fixture-pairing logic, but writing each side's *canonical* team name (the
same one the domestic SP1/D1/etc. feature pipeline already uses) instead of
fotmob's raw spelling, per the corrected approach in
`docs/superpowers/specs/2026-11-04-champions-league-recommendations-design.md`
Section 3.

**Architecture:** Port the abandoned branch's odds manifest (`git show`,
zero cost) and its fixture-pairing script largely unchanged — that part
already achieves 100% match rate (368/368) via hand-verified exact aliases
and is not the part that was wrong. Add one new canonicalization step
before the `raw_matches` write: resolve the matched fotmob team names
through `TeamNameMapper.map_team(name, team_pool)` (exact + accent-fold +
conservative Levenshtein fuzzy, `use_token_match` left at its default
`False`) against the real pool of existing `raw_matches` team names —
reusing the existing mechanism, not new infrastructure. **`use_token_match=True`
was tried and rejected during Task 2's implementation**: against the real
150-club UCL/UEL list and the real unscoped `config/team_mapping.json`
pool, it resolves exactly one club, and resolves it wrong — "Inter Club de
Escaldes" (a real, tiny Andorran club) to "Inter" (Inter Milan) at a 0.95
token-containment score, a false positive matching the exact failure shape
`US#141` already documented and fixed elsewhere (`src/ingestion/common/team_mapping.py`'s
module docstring) for an unscoped, cross-competition candidate pool. Zero
real benefit, one confirmed corruption — not used. No OddsPapi calls in
this plan at all; fotmob calls
(unauthenticated, free) will take real wall-clock time (~20-40 min for the
~105 unique dates needed) when the join actually runs.

**Tech Stack:** Python, DuckDB (`src/utils/db_manager.py`), pandas. No new
dependencies.

---

### Task 1: Port the historical odds manifest (zero new API cost)

**Files:**
- Create: `data/oddspapi_ucl_uel_snapshots/manifest.json`

- [ ] **Step 1: Pull the manifest from the abandoned branch**

Run:
```bash
mkdir -p data/oddspapi_ucl_uel_snapshots
git show worktree-llm-only-ucl-uel:data/oddspapi_ucl_uel_snapshots/manifest.json > data/oddspapi_ucl_uel_snapshots/manifest.json
```
Expected: a 368-entry JSON file (same one already inspected this session —
189 UCL + 179 UEL fixtures, 2026-01-20 to 2026-09-17, real 1X2 odds).

- [ ] **Step 2: Verify it's the real, complete file**

Run: `python3 -c "import json; m = json.load(open('data/oddspapi_ucl_uel_snapshots/manifest.json')); print(len(m))"`
Expected: `368`.

- [ ] **Step 3: Commit**

```bash
git add data/oddspapi_ucl_uel_snapshots/manifest.json
git commit -m "feat(ingestion): port the UCL/UEL historical odds manifest (zero new API cost)"
```

---

### Task 2: Write the canonicalization helper (TDD)

**Files:**
- Create: `src/ingestion/ucl_uel_join.py`
- Test: `tests/test_ucl_uel_join.py`

This is the one genuinely new piece of logic — everything else in this
plan ports existing, already-proven code. Isolate it in its own small
module so it can be tested without needing a live fotmob fetch or the
full join script around it.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the UCL/UEL team-name canonicalization step -- resolving a
fotmob-sourced team name into the same canonical identity the domestic
feature pipeline already uses, via map_team()'s existing exact/accent-fold/
conservative-fuzzy mechanism, not a bespoke crosswalk. use_token_match is
deliberately not enabled here -- see
test_canonicalize_does_not_falsely_resolve_inter_club_de_escaldes_to_inter
below for the confirmed real-data false positive that ruled it out."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.ucl_uel_join import canonicalize_team_name
from src.ingestion.common.team_mapping import TeamNameMapper


def test_canonicalize_resolves_exact_alias_already_in_the_shared_mapping() -> None:
    mapper = TeamNameMapper()
    pool = {"Ath Madrid", "Barcelona", "Real Madrid"}

    assert canonicalize_team_name("Atletico Madrid", mapper, pool) == "Ath Madrid"


def test_canonicalize_resolves_accent_variant_via_folding() -> None:
    mapper = TeamNameMapper()
    pool = {"Ath Madrid"}

    assert canonicalize_team_name("Atlético Madrid", mapper, pool) == "Ath Madrid"


def test_canonicalize_resolves_via_existing_exact_alias_for_known_la_liga_clubs() -> None:
    """These three names are already exact keys in the real
    config/team_mapping.json, so map_team() resolves them on its
    exact-match check, same path as the first test above."""
    mapper = TeamNameMapper()
    pool = {"Sociedad", "Betis", "Celta"}

    assert canonicalize_team_name("Real Sociedad", mapper, pool) == "Sociedad"
    assert canonicalize_team_name("Real Betis", mapper, pool) == "Betis"
    assert canonicalize_team_name("Celta Vigo", mapper, pool) == "Celta"


def test_canonicalize_does_not_falsely_resolve_inter_club_de_escaldes_to_inter() -> None:
    """Regression test for a confirmed real-data false positive:
    canonicalize_team_name used to pass use_token_match=True, and against
    this module's real-world pool -- every value in config/team_mapping.json
    combined across leagues, the shape raw_matches naturally has, since
    there is no per-league scoping here (see team_mapping.py's own US#141
    docstring on why an unscoped multi-league candidate pool is already a
    known risk factor for fuzzy matching) -- "Inter Club de Escaldes" (a
    real, tiny Andorran club) wrongly resolved to "Inter" (Inter Milan's
    canonical name) via a token-containment score of 0.95.

    use_token_match is no longer passed (defaults to False), which uses
    plain Levenshtein similarity instead: the closest candidate scores only
    0.27, well below the 0.82 threshold, so the name correctly falls
    through to the unmapped/unchanged branch. This test uses the real
    TeamNameMapper (real config/team_mapping.json) and the real full pool
    of every canonical name in it, not a synthetic fixture, since the bug
    only reproduced against real data."""
    mapper = TeamNameMapper()
    full_pool = set(mapper.mapping.values())

    assert canonicalize_team_name("Inter Club de Escaldes", mapper, full_pool) == "Inter Club de Escaldes"


def test_canonicalize_returns_the_name_unchanged_for_a_genuinely_new_club() -> None:
    """A club with no existing representation anywhere in raw_matches --
    e.g. Besiktas, never seen before -- has nothing to resolve against.
    Returns the (normalized) input itself, which becomes the de-facto new
    canonical identity from this point forward, same as how every other
    first-time club already bootstraps through TeamNameMapper elsewhere."""
    mapper = TeamNameMapper()
    pool = {"Ath Madrid", "Barcelona"}

    assert canonicalize_team_name("Besiktas Istanbul", mapper, pool) == "Besiktas Istanbul"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_ucl_uel_join.py -v`
Expected: `ModuleNotFoundError: No module named 'src.ingestion.ucl_uel_join'` (or
`ImportError` for `canonicalize_team_name`) — fails because the module
doesn't exist yet, not a typo in the test.

- [ ] **Step 3: Implement**

```python
"""UCL/UEL-specific ingestion helpers. The team-name canonicalization step
here is the one piece of new logic in this cherry-pick -- everything else
(fixture pairing, the historical odds backfill) ports the abandoned
branch's already-proven code largely unchanged. See
docs/superpowers/specs/2026-11-04-champions-league-recommendations-design.md
Section 3 for why fotmob's raw spelling must not be written directly into
raw_matches."""
from __future__ import annotations

from src.ingestion.common.team_mapping import TeamNameMapper


def canonicalize_team_name(name: str, mapper: TeamNameMapper, team_pool: set[str]) -> str:
    """Resolve a fotmob-sourced team name into the same canonical identity
    the domestic feature pipeline already uses for that club, via
    map_team()'s existing exact/accent-fold/conservative-fuzzy mechanism --
    not a bespoke OddsPapi/fotmob-specific crosswalk. Returns the name
    unchanged when no existing representation is found, which is correct:
    that becomes the club's first-time canonical identity, not an error.

    Deliberately does NOT pass use_token_match=True. This module's real
    candidate pool is unscoped across every competition (every value in
    config/team_mapping.json, the shape `raw_matches` naturally has here --
    see the US#141 precedent in team_mapping.py's docstring for why an
    unscoped multi-league pool is already a known risk factor). Against
    that real pool, token-containment scoring produces a confirmed false
    positive on real data: "Inter Club de Escaldes" (a real Andorran club)
    resolves to "Inter" (Inter Milan) at score 0.95. Plain Levenshtein
    (use_token_match's default, False) correctly leaves it unchanged
    (closest score 0.27, well below the 0.82 threshold)."""
    return mapper.map_team(name, team_pool)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_ucl_uel_join.py -v`
Expected: 5 passed, using plain `pytest tests/test_ucl_uel_join.py -v` (not
`python -m pytest`) — the `sys.path.append` line already in the test file
above is required for the plain console-script invocation to find `src.*`
at all, since there's no `conftest.py`/`pytest.ini` doing path setup for
this project; `python -m pytest` would mask a missing one.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/ucl_uel_join.py tests/test_ucl_uel_join.py
git commit -m "feat(ingestion): add UCL/UEL team-name canonicalization via the existing TeamNameMapper mechanism"
```

---

### Task 3: Port the fixture-pairing script, with canonicalization wired in

**Files:**
- Create: `scripts/build_ucl_uel_raw_matches.py`

Ports the abandoned branch's fixture-pairing logic (date + hand-verified
exact-alias matching, already proven at 368/368) essentially unchanged —
it isn't the part that was wrong. The only change: the matched row's
`home_team`/`away_team` get canonicalized (Task 2's helper) before the
`raw_matches` write, instead of writing fotmob's raw spelling.

- [ ] **Step 1: Pull the abandoned branch's script as a starting point**

Run: `git show worktree-llm-only-ucl-uel:scripts/build_ucl_uel_raw_matches.py > scripts/build_ucl_uel_raw_matches.py`

- [ ] **Step 2: Wire in canonicalization before the `raw_matches` insert**

In `scripts/build_ucl_uel_raw_matches.py`, add the import:
```python
from src.ingestion.common.team_mapping import TeamNameMapper
from src.ingestion.ucl_uel_join import canonicalize_team_name
```

In `_insert_matched_rows`, build the real team pool once (same pattern as
`src/ingestion/fotmob/merge.py`) and canonicalize each row's team names
before building the `match_id` and the insert tuple:

```python
def _insert_matched_rows(matched_rows: list[dict], mapper: TeamNameMapper) -> tuple[int, set[str]]:
    """... (docstring: same rationale as before, plus a note that the
    returned team_pool is the pre-insert snapshot, reused by main()'s
    new-clubs report below -- see Step 3 -- and a `ponytail:` comment
    flagging that match_id is hashed from the canonicalized name at
    insertion time, so a club whose canonical spelling changes in a LATER
    run would get a different match_id and INSERT OR IGNORE would no
    longer recognize it as the same fixture; no structural fix attempted,
    pinning canonical identity at first insert is a separate design
    question)."""
    db_manager = DuckDBManager()

    with db_manager.connection() as conn:
        CSVLoader._create_raw_matches_table(conn)
        pool_df = conn.execute("SELECT home_team, away_team FROM raw_matches").fetchdf()
        team_pool = set(pool_df["home_team"]) | set(pool_df["away_team"])

        records: list[tuple] = []
        for row in matched_rows:
            home = canonicalize_team_name(row["home_team"], mapper, team_pool)
            away = canonicalize_team_name(row["away_team"], mapper, team_pool)
            match_id = generate_match_id(date=row["date"], home_team=home, away_team=away, league=row["league"])
            records.append(
                (
                    match_id, row["league"], None, row["date"],
                    home, away, row["fthg"], row["ftag"],
                    row["odds_h"], row["odds_d"], row["odds_a"],
                    *([None] * 28),
                )
            )

        before = conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()[0]
        conn.executemany(
            """
            INSERT OR IGNORE INTO raw_matches
            (match_id, league, tier, date, home_team, away_team, fthg, ftag, odds_h, odds_d, odds_a,
             hs, "as", hst, ast, hc, ac, hy, ay, hr, ar, avgh, avgd, avga,
             maxch, maxcd, maxca, avgch, avgcd, avgca, xg_h, xg_a, xga_h, xga_a,
             over25_odds, under25_odds, ah_line, ah_home_odds, ah_away_odds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        after = conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()[0]
    return after - before, team_pool
```

Note: the team pool is read once, before any of this run's own rows are
inserted. Note on "pre-insert" here: it means "whatever `raw_matches` held
immediately before this run touched it" — UCL/UEL rows from *earlier* runs
included, not "every other league, forever." That distinction matters for
Step 3's report below, which reuses this exact pool rather than
re-querying with a different, narrower filter.

- [ ] **Step 3: Also report newly-registered (never-seen-before) clubs**

In `main()`, construct `mapper` once, pass it into `_insert_matched_rows`,
and reuse its *returned* pool for the report — don't re-query
`raw_matches` with a separate filter (an earlier draft of this step did
exactly that, filtering `WHERE league NOT IN ('UCL', 'UEL')`, which
permanently excludes this script's own prior-run rows and reprints
already-resolved clubs as "new" forever on every re-run — caught and
fixed during implementation, not a hypothetical):

```python
    mapper = TeamNameMapper()
    added, established_pool = _insert_matched_rows(matched_rows, mapper)

    # Report which clubs had no existing representation to resolve
    # against -- their fotmob spelling becomes their first-time canonical
    # identity from this run, not an error, but worth a human glance.
    # established_pool is the pool _insert_matched_rows read BEFORE this
    # run's own rows went in, reused as-is here (same mapper too) rather
    # than re-querying raw_matches with a separate filter.
    new_clubs = sorted({
        name for row in matched_rows
        for name in (row["home_team"], row["away_team"])
        if canonicalize_team_name(name, mapper, established_pool) == name
        and name not in established_pool
    })
    print(f"\n=== {len(new_clubs)} clubs with no prior representation (new canonical identities) ===")
    for name in new_clubs:
        print(f"  {name}")
```

- [ ] **Step 4: Commit**

```bash
git add scripts/build_ucl_uel_raw_matches.py
git commit -m "feat(ingestion): port UCL/UEL fixture-pairing script, canonicalize team names before the raw_matches write"
```

---

### Task 4: Port the fotmob score-extraction fix (real bug found running Task 5 the first time)

**Files:**
- Modify: `src/ingestion/fotmob/fetcher.py`

**Why this task exists:** an earlier attempt at Task 5 (running the real
join) crashed immediately with `KeyError: 'home_score'` in
`match_oddspapi_to_fotmob` — `fetch_all_matches`'s shared parser
(`_parse_finished_matches`) never extracted score fields at all, only
`fotmob_match_id`/`match_date`/`home_team`/`away_team`. This is a real,
100%-reproducible gap (not a flake — it fails on every match, every run),
caused by this plan originally scoping Task 3 to port only
`build_ucl_uel_raw_matches.py` without also porting the abandoned branch's
separate `fetcher.py` commit that script actually depends on. The
abandoned branch's own historical 368/368 match-rate claim is real — it
ran against ITS version of `fetcher.py`, which had this fix; this branch
never got it.

- [ ] **Step 1: Port the fix**

Apply this exact diff to `src/ingestion/fotmob/fetcher.py`'s
`_parse_finished_matches` function (confirmed via `git show
fa14fb9851082783e3bef19c69f0191c09034263 -- src/ingestion/fotmob/fetcher.py`
on the abandoned branch — a small, already-tested change, not new logic
to design):

```python
def _parse_finished_matches(matches_payload: list[dict]) -> list[dict]:
    """Shared parsing for one league's raw 'matches' list -> our own finished-
    match dicts (fotmob_match_id, match_date, home_team, away_team, home_score,
    away_score). Used by both fetch_finished_match_ids (single league) and
    fetch_matches_for_leagues (US#190, several leagues from one shared
    request) so the two never drift."""
    matches: list[dict] = []
    for match in matches_payload:
        status = match.get("status", {})
        if not status.get("finished"):
            continue
        utc_time = status.get("utcTime")
        if not utc_time:
            continue
        try:
            home_score = match["home"]["score"]
            away_score = match["away"]["score"]
            if home_score is None or away_score is None:
                # "finished": true with a null score happens on real
                # abandoned/awarded/walkover matches -- treat it the same
                # as a missing key rather than letting None reach callers
                # (e.g. build_ucl_uel_raw_matches.py inserts it straight
                # into raw_matches.fthg with no numeric/null check).
                raise ValueError("score present but null")
            matches.append(
                {
                    "fotmob_match_id": match["id"],
                    "match_date": pd.to_datetime(utc_time).tz_localize(None).normalize(),
                    "home_team": match["home"]["name"],
                    "away_team": match["away"]["name"],
                    "home_score": home_score,
                    "away_score": away_score,
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("Skipping malformed match entry id=%s: %s", match.get("id"), exc)
    return matches
```

Two new dict keys, plus an explicit null-score check. The first draft of
this fix (additive keys only, no null check) was caught by code-quality
review: a present-but-`null` score — real on fotmob for
abandoned/awarded/walkover matches still flagged `"finished": true` —
doesn't raise `KeyError` (the key exists, its value is just `None`), so it
would have flowed straight through into `home_score`/`away_score` as
`None`, silently reaching `raw_matches.fthg` with no numeric/null check in
the new consumer, unlike the existing precedent for this exact risk
(`sweden_loader.py`'s `pd.to_numeric(..., errors="coerce")` + `dropna`).
Explicitly raising `ValueError` for a `None` score routes it through the
same existing skip-and-log `except` branch as a missing key — same log
line, no new branch.

- [ ] **Step 2: Check for any other real caller this could affect**

Run: `grep -rn "fetch_finished_match_ids\|fetch_matches_for_leagues" --include="*.py" . | grep -v "fetcher.py\|/tests/"`
Expected: confirms the only real (non-test) callers are `main.py` (player-stats
pipeline) and `src/ingestion/fotmob/lineup.py` (lineup backfill) — both
consume specific known dict keys (`fotmob_match_id`, `home_team`, etc.),
so two new keys appearing is additive and should not break either. Read
both call sites briefly to confirm they don't do strict key-set validation
(e.g. `assert set(d.keys()) == {...}`) that an added key would break — if
either does, stop and report rather than assuming it's fine.

- [ ] **Step 3: Run the existing fotmob fetcher test suite**

Run: `pytest tests/test_fotmob_fetcher.py app/backend/tests -k fotmob -v` (adjust
paths if the real test file names differ from this guess — check
`find . -iname "*fotmob*test*" -o -iname "*test*fotmob*"` first rather
than assuming).
Expected: all pass, confirming the additive change didn't break any
existing fotmob-fetcher-adjacent test.

- [ ] **Step 4: Commit**

```bash
git add src/ingestion/fotmob/fetcher.py
git commit -m "fix(ingestion): extract final score from fotmob finished-match payload

Ported from the abandoned UCL/UEL branch's own fix
(fa14fb9851082783e3bef19c69f0191c09034263) -- this branch's version of
_parse_finished_matches never extracted score fields at all, which
Task 5's first real run discovered by crashing on every single match
(KeyError: home_score), not a flake."
```

---

### Task 5: Run the real join and verify it

**Files:** none new — running Task 3's script for real.

**If this is a retry after Task 4's fix:** the first attempt already ran
the full fetch phase (105 unique dates) before crashing on the
`home_score` `KeyError` — `data/fotmob_date_cache/*.json` has all 105
dates cached from that run (the script's own `_fetch_all_matches_cached`
checks this cache first). A retry after Task 4's fix should be
near-instant for the fetch phase, not another 20-40 minute wait — but the
*cached* payloads were captured before Task 4's fetcher fix existed, so
they're from `fetch_all_matches`'s OLD (pre-fix) output shape, missing
`home_score`/`away_score` entirely. **Delete the stale cache before
retrying** (`rm -rf data/fotmob_date_cache`) so the fetch re-runs through
the now-fixed parser and actually gets score fields this time — otherwise
Task 4's fix will have no effect and the same `KeyError` will recur.

**Pool-scoping safety, already checked exhaustively (not just a spot-check):**
a code-quality review of Task 2 raised whether an unscoped, all-leagues
candidate pool (Task 3's design) risks *other* false positives beyond the
one already found and fixed (Inter Club de Escaldes). Checked directly
against the real, full 150-club UCL/UEL list and the real unscoped
`config/team_mapping.json` pool, under the shipped default
(`use_token_match` left `False`): **zero** additional matches. 39/150
resolve via exact/accent-fold; the remaining 111 correctly fall through
unchanged. Plain Levenshtein fuzzy matching doesn't fire at all for this
real dataset — there is nothing else lurking, confirmed exhaustively, not
by sampling. Task 3's unscoped pool is safe as designed; no league-scoping
needed for the data this plan actually covers.

This takes real wall-clock time: ~105 unique dates fetched from fotmob's
public, unauthenticated, free endpoint, roughly 20-40 minutes depending on
the 1-second per-call delay already built into `fetch_all_matches`. No
OddsPapi calls at all — this step is free, just slow.

- [ ] **Step 1: Run the join**

Run: `python scripts/build_ucl_uel_raw_matches.py`
Expected: a self-check block reporting match rate (should be 368/368,
100% — same as the original run, since the fixture-pairing logic is
unchanged), the count of newly-inserted rows, and the new-canonical-clubs
report from Task 3 Step 3.

- [ ] **Step 2: Spot-check the 6 known-tricky clubs got the correct canonical name**

Run:
```bash
python3 -c "
from src.utils.db_manager import DuckDBManager
db = DuckDBManager()
with db.connection() as conn:
    rows = conn.execute(\"SELECT DISTINCT home_team FROM raw_matches WHERE league IN ('UCL','UEL') AND home_team IN ('Ath Madrid','Ath Bilbao','Sociedad','Betis','Celta','Bayern Munich')\").fetchall()
    print(rows)
"
```
Expected: the 6 real canonical names actually appear in the UCL/UEL rows —
confirms the canonicalization step worked on live data, not just the unit
test's constructed pool. If any are missing, check whether that specific
club actually played a UCL/UEL match in this sample before concluding the
canonicalization step is broken.

- [ ] **Step 3: Record the real match rate and new-club count, commit any follow-up findings**

If the match rate is below 368/368 or the new-canonical-club list looks
wrong (e.g. a club that should have resolved to an existing name didn't),
stop and investigate before proceeding — don't assume it's fine. If it
matches expectations, no commit needed for this task (the data lives in
the local DuckDB file, not a tracked artifact).

---

### Task 6: Port the historical-odds backfill script (not run, for future reuse)

**Files:**
- Create: `scripts/pull_oddspapi_ucl_uel_1x2.py`

This is what would extend the ~7-week-stale sample forward in time —
ported for future use, not executed now (running it now would just
re-confirm matches already in the manifest, spending real OddsPapi
credits for zero new data, contrary to the data budget policy in the
design spec).

- [ ] **Step 1: Pull the script**

Run: `git show worktree-llm-only-ucl-uel:scripts/pull_oddspapi_ucl_uel_1x2.py > scripts/pull_oddspapi_ucl_uel_1x2.py`

- [ ] **Step 2: Confirm it still imports cleanly (no execution)**

Run: `python3 -c "import ast; ast.parse(open('scripts/pull_oddspapi_ucl_uel_1x2.py').read())"`
Expected: no output (valid syntax) — this only checks the file parses, it
does not run `main()` or make any network call.

- [ ] **Step 3: Commit**

```bash
git add scripts/pull_oddspapi_ucl_uel_1x2.py
git commit -m "feat(ingestion): port the UCL/UEL historical odds backfill script (not run -- see data budget policy)"
```

---

### Task 7: Documentation

**Files:**
- Modify: `documents/FRAI_TECHSPEC.md`
- Modify: `documents/user_stories.md`

Per `CLAUDE.md` convention.

- [ ] **Step 1: Add a techspec section**

Append a new subsection (parallel to the existing `llm_only` tier section,
§30.2) documenting: the `raw_matches` population path for UCL/UEL
(manifest → fixture-pairing → canonicalization → insert), the
canonicalization mechanism (reusing `TeamNameMapper`/`fotmob/merge.py`'s
pattern, not a bespoke crosswalk), and the real match rate achieved.

- [ ] **Step 2: Add user story entries**

Append new `US#` entries to `documents/user_stories.md` for this
ingestion work, marked `completed` with the real match rate and new-club
count from Task 4 in the completion notes, following the existing phase
structure.

- [ ] **Step 3: Commit**

```bash
git add documents/FRAI_TECHSPEC.md documents/user_stories.md
git commit -m "docs: document UCL/UEL raw_matches ingestion and canonicalization"
```
