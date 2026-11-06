"""Task 17 (llm_only tier): join fotmob's finished UCL/UEL match results
against OddsPapi's historical 1X2 odds (already pulled, Task 15, into
data/oddspapi_ucl_uel_snapshots/manifest.json) and write the result into
raw_matches, so the backtest harness has real truth + real odds for this
tier the same way every other league already does.

Only match_id, league, tier, date, home_team, away_team, fthg, ftag, odds_h,
odds_d, odds_a are ever populated here -- everything else in raw_matches'
column list (shots, corners, cards, xG, Asian handicap) stays NULL for
UCL/UEL by design: this tier needs no feature-store parity, just backtest
truth (see Task 17 spec).
"""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.common.team_mapping import TeamNameMapper
from src.ingestion.fotmob.fetcher import fetch_all_matches
from src.ingestion.football_data.loader import CSVLoader
from src.ingestion.ucl_uel_join import canonicalize_team_name
from src.utils.db_manager import DuckDBManager
from src.utils.helpers import generate_match_id, standardize_team_name
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "oddspapi_ucl_uel_snapshots" / "manifest.json"

# Local cache of fetch_all_matches(day) results, one JSON file per date.
# Two full 105-date fotmob fetches (~18 min each) already happened for this
# task's diagnostics -- re-fetching live on every iteration of the
# name-matching logic below would repeat that cost for no reason. A re-run
# after the first successful fetch of a given date is near-instant.
FOTMOB_CACHE_DIR = Path(__file__).parent.parent / "data" / "fotmob_date_cache"


def _fetch_all_matches_cached(day: date) -> list[dict]:
    """fetch_all_matches(day), through a local JSON cache keyed by date.
    match_date (a pd.Timestamp, not JSON-serializable) round-trips as an
    ISO string on disk and is restored to a Timestamp on cache read, so a
    cached result is indistinguishable from a live one to callers.

    Deliberately does NOT cache an empty result: fetch_all_matches returns
    [] both for "this day genuinely has zero finished matches" and for a
    degraded fetch (_fetch_matches_payload returning None on a malformed/
    failed response, swallowed without raising -- a real, already-known gap
    in that function, not introduced here). Since every date in this
    script's corpus comes from a real OddsPapi fixture, a genuine zero-match
    day should not occur -- caching [] would risk permanently baking in a
    transient fetch failure instead of retrying it live next run."""
    cache_path = FOTMOB_CACHE_DIR / f"{day.isoformat()}.json"
    if cache_path.exists():
        cached: list[dict] = json.loads(cache_path.read_text())
        for match in cached:
            match["match_date"] = pd.Timestamp(match["match_date"])
        return cached

    matches = fetch_all_matches(day)
    if matches:
        FOTMOB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps([{**m, "match_date": m["match_date"].isoformat()} for m in matches])
        )
    return matches


# UCL/UEL-specific team-name crosswalk: OddsPapi's formal name -> fotmob's
# short/native name, built from this task's real unmatched-entry diagnostics
# (2026-10-08) against the full 368-entry manifest and every one of the 105
# fotmob dates it needed -- NOT a general-purpose fuzzy matcher (that risks
# a false-positive join writing wrong odds/scores against the wrong
# fixture); every entry here was individually confirmed by cross-referencing
# a real matchday's fotmob payload and verifying the opponent on the other
# side of the fixture also lines up, not guessed from the name alone.
#
# A handful of values are a tuple instead of a single string because fotmob
# itself spelled the same club differently across different real matchdays
# (e.g. "Atletico Madrid" on most dates but "Atlético Madrid" on 2026-09-09;
# "FC Viktoria Plzen" as both "Viktoria Plzen" and "Viktoria Plzeň") -- both
# spellings were individually confirmed against real data, so both are kept
# as exact-match candidates. This is still pure exact-string matching after
# normalization, never substring/fuzzy.
ODDSPAPI_TO_FOTMOB_ALIASES: dict[str, str | tuple[str, ...]] = {
    "Besiktas Istanbul": "Beşiktaş",
    "RSC Anderlecht": "Anderlecht",
    "GNK Dinamo Zagreb": "Dinamo Zagreb",
    "AC Milan": "Milan",
    "Hapoel Be`er Sheva FC": "Hapoel Beer Sheva",
    "Olympiacos Piraeus": "Olympiacos",
    "Olympique Lyon": "Lyon",
    "SK Sturm Graz": "Sturm Graz",
    "Sunderland AFC": "Sunderland",
    "FC Ararat-Armenia": "Ararat Armenia",
    "AC Omonia Nicosia": "Omonia Nicosia",
    "RC Celta de Vigo": "Celta Vigo",
    "FC Barcelona": "Barcelona",
    "Feyenoord Rotterdam": "Feyenoord",
    "FC Kairat Almaty": "Kairat Almaty",
    "FC Salzburg": "Salzburg",
    "Mjallby AIF": "Mjällby",
    "FC Thun": "Thun",
    "AGF Aarhus": "AGF",
    "PFC CSKA Sofia": "CSKA Sofia",
    "FC Viktoria Plzen": ("Viktoria Plzen", "Viktoria Plzeň"),
    "FK Crvena Zvezda Belgrade": "FK Crvena Zvezda",
    "CS Universitatea Craiova": "Universitatea Craiova",
    "FC Iberia 1999": "Iberia 1999",
    "Jagiellonia Bialystok": "Jagiellonia Białystok",
    "PFC Levski Sofia": "Levski Sofia",
    "Viking FK": "Viking",
    "SK Slovan Bratislava": "Slovan Bratislava",
    "LASK Linz": "LASK",
    "Celtic Glasgow": "Celtic",
    "Bodoe/Glimt": "Bodø/Glimt",
    "Sabah Masazir": "Sabah FK",
    "KI Klaksvik": "Klaksvik",
    "KKS Lech Poznan": "Lech Poznań",
    "KF Egnatia Rrogozhine": "Egnatia",
    "Heart of Midlothian FC": "Heart of Midlothian",
    "Glasgow Rangers": "Rangers",
    "Maccabi Tel Aviv FC": "Maccabi Tel Aviv",
    "Lincoln Red Imps": "Lincoln Red Imps FC",
    "Gornik Zabrze": "Górnik Zabrze",
    "FK Kauno Zalgiris": ("FK Kauno Zalgiris", "FK Kauno Žalgiris"),
    "Union Saint-Gilloise": "Union St.Gilloise",
    "Larne FC": "Larne",
    "SL Benfica": "Benfica",
    "FC St. Gallen 1879": "St. Gallen",
    "Hammarby IF": "Hammarby",
    "FC Dynamo Kyiv": "Dynamo Kyiv",
    "HNK Hajduk Split": "Hajduk Split",
    "FC Sheriff Tiraspol": "FC Sheriff",
    "Fenerbahce Istanbul": "Fenerbahçe",
    "FK Sutjeska Niksic": "Sutjeska",
    "Petrocub Hincesti": "CS Petrocub",
    "Maxline Vitebsk": ("Maxline Vitebsk", "ML Vitebsk"),
    "SP Tre Fiori": "Tre Fiori",
    "Floriana FC": "Floriana",
    "FC Drita": "Drita",
    "FK Borac Banja Luka": "Borac Banja Luka",
    "WKW ETO FC Gyor": "Györi ETO",
    "Inter Club de Escaldes": "Inter Club d'Escaldes",
    "Tallinna FC Flora": "Flora Tallinn",
    "MSK Zilina": "Zilina",
    "Aluminij Kidricevo": "Aluminij",
    "FC Universitatea Cluj": "Universitatea Cluj",
    "Derry City FC": "Derry City",
    "IF Vestri": "Vestri",
    "Arsenal FC": "Arsenal",
    "Atletico Madrid": ("Atletico Madrid", "Atlético Madrid"),
    "SC Freiburg": "Freiburg",
    "SC Braga": "Braga",
    "Bayern Munich": "Bayern München",
    "Real Betis Seville": "Real Betis",
    "Bologna FC": "Bologna",
    "Liverpool FC": "Liverpool",
    "Lille OSC": "Lille",
    "AS Roma": "Roma",
    "Panathinaikos Athens": "Panathinaikos",
    "KRC Genk": "Genk",
    "Galatasaray Istanbul": "Galatasaray",
    "Atalanta BC": "Atalanta",
    "Chelsea FC": "Chelsea",
    "AS Monaco": "Monaco",
    "Juventus Turin": "Juventus",
    "Inter Milano": "Inter",
    "SK Brann": "Brann",
    "PFC Ludogorets 1945 Razgrad": "Ludogorets Razgrad",
    "FC Basel 1893": "Basel",
    "Fotbal Club FCSB": "FCSB",
    "Malmo FF": "Malmö FF",
    "OGC Nice": "Nice",
    "Young Boys Bern": "Young Boys",
    "SSC Napoli": "Napoli",
    "Olympique Marseille": "Marseille",
    "Villarreal CF": "Villarreal",
    "Athletic Bilbao": "Athletic Club",
    "Ajax Amsterdam": "Ajax",
    "TSG Hoffenheim": "Hoffenheim",
    "Real Sociedad San Sebastian": "Real Sociedad",
    "St. Truidense VV": "St.Truiden",
    "Como 1907": "Como",
    "Racing Club De Lens": "Lens",
    "FC Shakhtar Donetsk": "Shakhtar Donetsk",
    "FC Copenhagen": "FC København",
    "SCU Torreense": "Torreense",
    "Lillestroem SK": "Lillestrøm",
    "Stade Rennais FC": "Rennes",
    "Ferencvarosi Budapest": ("Ferencváros", "Ferencvaros"),
    "FC Hradec Kralove": "Hradec Kralove",
    "Tromsoe IL": "Tromsø",
    "FK Vojvodina Novi Sad": "Vojvodina",
    "Kuopion Palloseura": "KuPS",
    "The New Saints FC": "TNS",
    "FC Twente Enschede": "FC Twente",
}


def _oddspapi_name_candidates(name: str) -> set[str]:
    """All exact-match candidate spellings to try for one OddsPapi
    participant name: the hand-curated alias(es) above (if any), plus
    standardize_team_name's own output as a fallback (same as before this
    crosswalk existed) -- so a name not in the crosswalk still gets every
    chance it had under Task 17's original logic. Pure exact-string
    matching after normalization, never substring/fuzzy."""
    candidates: set[str] = set()
    alias = ODDSPAPI_TO_FOTMOB_ALIASES.get(name)
    if alias is not None:
        candidates.update(alias if isinstance(alias, tuple) else (alias,))
    candidates.add(standardize_team_name(name))
    return candidates


# Day offsets to try, in priority order: exact manifest date first, then -1
# and +1 calendar day. This absorbs a real, plausible risk: OddsPapi's
# startTime is UTC ISO-8601, while fotmob's /api/data/matches buckets a
# match into a calendar day that can shift by one depending on kickoff time
# and whatever timezone fotmob's own day-bucketing uses internally. Widening
# is explicit and counted -- see match_oddspapi_to_fotmob's docstring and
# each matched row's "date_offset_days" -- never silent.
_DATE_OFFSETS = (0, -1, 1)


def _offset_date(date_str: str, offset_days: int) -> str:
    return (date.fromisoformat(date_str) + timedelta(days=offset_days)).isoformat()


def match_oddspapi_to_fotmob(
    manifest_entries: dict[str, dict],
    fotmob_matches_by_date: dict[str, list[dict]],
) -> tuple[list[dict], list[dict]]:
    """Join OddsPapi manifest entries to fotmob finished-match dicts.

    Matching rule: standardize_team_name(fotmob home_team) is in the set of
    exact-match candidates for participant1Name (the UCL/UEL crosswalk's
    alias(es) for that name, if any, plus standardize_team_name's own
    output -- see _oddspapi_name_candidates), and the same for away, on the
    same calendar date -- tried exact first (the manifest's own `date`
    field), then -1 and +1 day (see _DATE_OFFSETS) if the exact date finds
    no fotmob entry at all for that date. Beyond the hand-curated crosswalk
    (ODDSPAPI_TO_FOTMOB_ALIASES), standardize_team_name is a narrow
    English-league alias map plus whitespace/case cleanup -- it will NOT
    resolve a cross-provider spelling difference outside that crosswalk;
    those fall out as unmatched, which is correct and reported, not
    silently papered over.

    Returns (matched_rows, unmatched_entries):
      - matched_rows: one dict per successful join -- fixtureId, league,
        date (the date that actually matched; may differ from the
        manifest's own date if an offset was used), home_team, away_team
        (fotmob's own spelling, matching the rest of raw_matches), fthg,
        ftag, odds_h, odds_d, odds_a, date_offset_days (0 if the exact date
        matched, else the offset that did).
      - unmatched_entries: one dict per manifest entry with no fotmob
        counterpart at ANY tried offset -- fixtureId, league,
        participant1Name, participant2Name, date -- so nothing is ever
        silently dropped; every unmatched row is nameable.
    """
    matched_rows: list[dict] = []
    unmatched_entries: list[dict] = []

    for fixture_id, entry in manifest_entries.items():
        home_candidates = _oddspapi_name_candidates(entry["participant1Name"])
        away_candidates = _oddspapi_name_candidates(entry["participant2Name"])
        manifest_date = entry["date"]

        found_date = None
        found_match = None
        found_offset = None
        for offset in _DATE_OFFSETS:
            try:
                candidate_date = _offset_date(manifest_date, offset)
            except ValueError:
                continue
            for fotmob_match in fotmob_matches_by_date.get(candidate_date, []):
                if (
                    standardize_team_name(fotmob_match["home_team"]) in home_candidates
                    and standardize_team_name(fotmob_match["away_team"]) in away_candidates
                ):
                    found_date, found_match, found_offset = candidate_date, fotmob_match, offset
                    break
            if found_match is not None:
                break

        if found_match is None:
            unmatched_entries.append(
                {
                    "fixtureId": fixture_id,
                    "league": entry.get("league"),
                    "participant1Name": entry.get("participant1Name"),
                    "participant2Name": entry.get("participant2Name"),
                    "date": manifest_date,
                }
            )
            continue

        if found_offset != 0:
            LOGGER.info(
                "Matched fixtureId=%s via %+d day offset (manifest date=%s, fotmob date=%s)",
                fixture_id, found_offset, manifest_date, found_date,
            )

        odds = entry.get("odds_1x2") or {}
        matched_rows.append(
            {
                "fixtureId": fixture_id,
                "league": entry.get("league"),
                "date": found_date,
                "home_team": found_match["home_team"],
                "away_team": found_match["away_team"],
                "fthg": found_match["home_score"],
                "ftag": found_match["away_score"],
                "odds_h": odds.get("home"),
                "odds_d": odds.get("draw"),
                "odds_a": odds.get("away"),
                "date_offset_days": found_offset,
            }
        )

    return matched_rows, unmatched_entries


def _dates_to_fetch(manifest: dict[str, dict]) -> list[date]:
    """Unique calendar dates fotmob needs to be queried for: every manifest
    date, plus its -1/+1 neighbors (so the ±1-day fallback above actually
    has data to search against -- fetching only the exact manifest dates
    would make that fallback a no-op)."""
    unique_dates: set[date] = set()
    for entry in manifest.values():
        base = date.fromisoformat(entry["date"])
        for offset in _DATE_OFFSETS:
            unique_dates.add(base + timedelta(days=offset))
    return sorted(unique_dates)


def _build_fotmob_matches_by_date(manifest: dict[str, dict]) -> dict[str, list[dict]]:
    fotmob_matches_by_date: dict[str, list[dict]] = {}
    dates = _dates_to_fetch(manifest)
    LOGGER.info("Fetching fotmob finished matches for %d unique dates", len(dates))
    for day in dates:
        matches = _fetch_all_matches_cached(day)
        fotmob_matches_by_date[day.isoformat()] = matches
    return fotmob_matches_by_date


def _insert_matched_rows(matched_rows: list[dict], mapper: TeamNameMapper) -> tuple[int, set[str]]:
    """Upsert matched rows into raw_matches via the same INSERT OR IGNORE
    pattern as every other ingestion path (src/ingestion/football_data/
    loader.py). Only the UCL/UEL-relevant columns are populated; everything
    else stays NULL.

    `tier` is left NULL rather than routed through map_league_code_to_tier:
    that helper's unknown-code fallback would silently store `4`, a real
    English "League Two" tier value that has no meaning for a continental
    cup -- the exact trap src/ingestion/football_data/sweden_loader.py
    already documents and avoids for the same reason. `tier` has no NOT
    NULL constraint, so this needs no raw_matches migration.

    home_team/away_team are canonicalized (src.ingestion.ucl_uel_join.
    canonicalize_team_name) against the existing raw_matches team pool
    before match_id generation and the insert itself, rather than writing
    fotmob's raw spelling -- see ucl_uel_join.py's module docstring. The
    pool is read once, before any of this run's own rows are inserted, so
    it reflects only pre-existing rows and one UCL/UEL row can't influence
    another's resolution (same pattern as fotmob/merge.py). That same
    pre-insert pool is returned (alongside the inserted-row count) so
    main()'s new-clubs report can reuse it directly instead of re-querying
    raw_matches with its own (previously buggy) filter -- see the
    docstring's note on match_id drift below for why "pre-insert" here
    deliberately does NOT mean "every other league, forever": it means
    "whatever raw_matches held immediately before this run touched it",
    UCL/UEL rows from earlier runs included.

    ponytail: match_id is a hash over the canonicalized team names at
    insertion time, resolved against whatever raw_matches currently holds.
    If a club's canonical spelling ever changes later (e.g. a new league
    onboarding reassigns its mapped name), re-running this script would
    hash a DIFFERENT match_id for the same real fixture, so INSERT OR
    IGNORE would no longer recognize it as a duplicate and a second row
    for that fixture would be inserted. No fix attempted here -- pinning
    a club's canonical identity permanently at first-insert time is a
    separate, larger design question. Revisit if/when this script needs
    to be safely re-run across a canonical-name change."""
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
                    *([None] * 28),  # hs through ah_away_odds: unpopulated for this tier (see module docstring)
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


def main() -> None:
    manifest: dict[str, dict] = json.loads(MANIFEST_PATH.read_text())
    print(f"Loaded manifest: {len(manifest)} entries")

    fotmob_matches_by_date = _build_fotmob_matches_by_date(manifest)
    total_fotmob_matches = sum(len(v) for v in fotmob_matches_by_date.values())
    print(f"Fetched fotmob matches for {len(fotmob_matches_by_date)} unique dates "
          f"({total_fotmob_matches} finished matches across all competitions)")

    matched_rows, unmatched_entries = match_oddspapi_to_fotmob(manifest, fotmob_matches_by_date)
    offset_used = [r for r in matched_rows if r["date_offset_days"] != 0]

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

    print("\n=== Self-check ===")
    print(f"Manifest entries total: {len(manifest)}")
    print(f"Matched: {len(matched_rows)}")
    print(f"Unmatched: {len(unmatched_entries)}")
    print(f"Match rate: {len(matched_rows) / len(manifest) * 100:.1f}%")
    print(f"Matches that needed the +/-1 day fallback: {len(offset_used)} "
          f"({'none -- exact-date matching alone was sufficient' if not offset_used else 'see below'})")
    for row in offset_used:
        print(f"  offset={row['date_offset_days']:+d} fixtureId={row['fixtureId']} "
              f"{row['home_team']} vs {row['away_team']} -> matched date {row['date']}")
    print(f"New rows inserted into raw_matches: {added} "
          f"(<= matched count; fewer if some were already present from a prior run)")

    if unmatched_entries:
        print(f"\n=== Unmatched entries ({len(unmatched_entries)}) -- never silently dropped ===")
        for entry in unmatched_entries:
            print(f"  fixtureId={entry['fixtureId']} league={entry['league']} date={entry['date']} "
                  f"{entry['participant1Name']} vs {entry['participant2Name']}")


if __name__ == "__main__":
    main()
