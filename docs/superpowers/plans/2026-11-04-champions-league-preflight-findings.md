# Champions League / Europa League Pre-Flight Findings

Live-verified 2026-11-04.

## Real UCL/UEL club list (from the existing 368-match historical manifest)

368 fixtures, 150 unique clubs (96 UCL-side + 99 UEL-side unique clubs,
overlapping for clubs that played in both), extracted at zero new API cost
from `worktree-llm-only-ucl-uel`'s already-paid-for backfill
(`data/oddspapi_ucl_uel_snapshots/manifest.json`). Full list saved to
`data/ucl_uel_club_list.json`.

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
| Israel | False (404) |
| Cyprus | False (404) |

15 additional countries confirmed beyond the 6 modeled leagues — a
substantial widening of the Dixon-Coles pool's reach, found at zero
ongoing cost (same free CSV mechanism already used for the 6 leagues).

## UCL/UEL club bucketing by data-source reach

Two real bugs found and fixed while building this (not pre-existing
knowledge): football-data.co.uk has two CSV formats (the 6 leagues'
`mmz4281/<season>/<CODE>.csv`, one file per season; and a "new leagues"
format `new/<CODE>.csv`, every season in one file — confirmed live for
Austria/Switzerland/Denmark/Norway/Poland/Russia/Romania/Ireland/Finland,
the same format already used for Sweden elsewhere in this codebase).
Team-name matching also had to be word-level, not whole-string — an early
draft searched for the full UCL/UEL name ("Celtic Glasgow") as one
unbroken substring of the raw CSV text and found zero matches even for
teams confirmed present, because football-data.co.uk uses short names
("Celtic").

**Raw script output, then 3 spot-checked corrections.** The script's own
exclusion pass (clubs already in the 6 modeled leagues + 5 second
divisions) found 40 matches, 3 of which are coincidental short-substring
false positives, caught by eye, not by the script:
- "Glasgow Rangers" matched "Angers" (a French club) — Rangers is Scottish.
- "Lillestroem SK" matched "Lille" (a French club) — Lillestrøm is
  Norwegian.
- "Inter Club de Escaldes" (a small Andorran club) matched "Inter" (Inter
  Milan).

All 3 are moved to the `unresolved` bucket below (not re-verified against
their real country here, to avoid overclaiming without checking) rather
than left as wrongly-excluded. **Corrected tally: 37 already in the
11-league universe, 43 matched to a wider football-data.co.uk country, 70
unresolved** (150 total). This is a spike-level approximation for
bucketing — not `TeamNameMapper`-grade — and it has a real, visible error
rate (3 caught just by spot-checking the exclusion list; the two match
lists below likely have some too, not individually re-verified).

| Country | Format | Matched clubs |
|---|---|---|
| Netherlands | mmz4281/N1 | Go Ahead Eagles, PSV Eindhoven, FC Utrecht, Ajax Amsterdam, Feyenoord Rotterdam, FC Twente Enschede, NEC Nijmegen, AZ Alkmaar |
| Portugal | mmz4281/P1 | FC Porto, SC Braga, SL Benfica |
| Belgium | mmz4281/B1 | St. Truidense VV, Club Brugge, KRC Genk, RSC Anderlecht |
| Turkey | mmz4281/T1 | Galatasaray Istanbul, Besiktas Istanbul, Fenerbahce Istanbul, Trabzonspor |
| Scotland | mmz4281/SC0 | Celtic Glasgow |
| Greece | mmz4281/G1 | OFI Crete, PAOK Thessaloniki, Panathinaikos Athens |
| Austria | new/AUT | SK Sturm Graz, LASK Linz, FC Salzburg |
| Switzerland | new/SWZ | Young Boys Bern, FC St. Gallen 1879, FC Thun, FC Basel 1893 |
| Denmark | new/DNK | FC Midtjylland, FC Copenhagen, AGF Aarhus |
| Norway | new/NOR | Tromsoe IL, SK Brann, Viking FK, Vikingur Reykjavik |
| Poland | new/POL | Gornik Zabrze, KKS Lech Poznan, Jagiellonia Bialystok |
| Russia | new/RUS | (none matched) |
| Romania | new/ROU | Fotbal Club FCSB |
| Ireland | new/IRL | Derry City FC, Shamrock Rovers |
| Finland | new/FIN | (none matched) |

43 clubs reachable via a wider football-data.co.uk country — real
historical training data, same mechanism as the 6 modeled leagues, not
just a current snapshot (the "new leagues" format covers every season
back to 2012). 70 clubs (67 from the script + 3 corrected) remain
unresolved — candidates for Task 4's fotmob check, including Glasgow
Rangers, Lillestroem SK, and Inter Club de Escaldes despite the script's
raw output.

## FotMob raw payload coverage for unresolved clubs (last 14 days)

58/70 previously-unresolved clubs found in fotmob's unfiltered payload.
The 3 manually-corrected clubs from Task 3 all resolved here: Glasgow
Rangers and Inter Club de Escaldes found; Lillestroem SK not found in this
14-day window (worth a longer window before concluding no coverage).

**Important miscount discovered here, affecting Task 3's numbers:**
several clubs "found" in this check are ones that should already belong to
an *earlier* bucket (11-league universe or a wider football-data.co.uk
country) — Manchester City, Manchester United, Nottingham Forest
(England), Atletico Madrid, Athletic Bilbao (Spain), Eintracht Frankfurt
(Germany), Stade Rennais (France), Sporting CP (Portugal, a wider-country
match). They were missed in Task 3 because football-data.co.uk abbreviates
names in ways that aren't simple truncations ("Man City" vs "Manchester
City", "Nott'm Forest" vs "Nottingham Forest") — a substring check can't
catch a different short-form word. **This means Task 3's "already
covered" counts are undercounts, not overcounts — the real new-coverage
need is smaller than the raw numbers suggest.** Safe direction of error
for a pre-flight spike (overestimates remaining work, doesn't understate
data risk), but a reminder that real ingestion code must use
`TeamNameMapper`'s actual alias dictionary, not this spike's substring
heuristic, which was never intended to replace it.

12 clubs genuinely not found in 14 days: Bodoe/Glimt, Ferencvarosi
Budapest, Floriana FC, GNK Dinamo Zagreb, Kuopion Palloseura, Lillestroem
SK, MSK Zilina, Maxline Vitebsk, Mjallby AIF, Qarabag FK, Tallinna FC
Flora, WKW ETO FC Gyor — some of these (Ferencváros, Dinamo Zagreb,
Bodø/Glimt) are well-known, regular UCL/UEL participants, so "not found in
14 days" most likely reflects their domestic league being between
matchdays, not genuine unavailability — re-check over a longer window
before treating any of these as a real gap.
