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
