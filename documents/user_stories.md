# FPAI Forecast Engine User Stories

This document tracks story-level actionable items for the forecast-engine pivot. Default status is `active`. Completed stories are archived in `documents/FRAI_TECHSPEC.md` Section 23.

## Story Dependencies & Execution Order

**PHASE 1–11: All completed** — See Section 23 in `documents/FRAI_TECHSPEC.md` for the full archive.

**PHASE 12: CLI & Model Lifecycle — All completed** — See Sections 23 and 25 in `documents/FRAI_TECHSPEC.md`.

**PHASE 13: Agent Tool Layer — All completed** — See Sections 23 and 26 in `documents/FRAI_TECHSPEC.md`.

**PHASE 14: Player-Level Data & Competition Tiers — All completed (US#87–99).** Design recorded in `documents/FRAI_TECHSPEC.md` Section 27. All three sub-phases complete: 14a (competition registry), 14b (FotMob player ingestion), and 14c (SQUAD_* squad features with competition-gated model training).

| ID | Status | Description | Comments |
|---|---|---|---|
| Phase 1–11 | completed | Full archive of all stories from these phases. | See `documents/FRAI_TECHSPEC.md` Section 23 |
| Phase 12 | completed | CLI & Model Lifecycle stories. | See `documents/FRAI_TECHSPEC.md` Sections 23 and 25 |
| Phase 13 | completed | Agent Tool Layer stories. | See `documents/FRAI_TECHSPEC.md` Sections 23 and 26 |

### Phase 14a: Model Tier Reorg — Completed

| ID | Status | Description | Comments |
|---|---|---|---|
| US#87 | completed | Define `config/competitions.yaml` competition registry — `competition_id` mapped to `tier`, `league_code`, `enabled_feature_groups`, `player_data_sources`. | — |
| US#88 | completed | Resolve model tier (`general_purpose` / `competition_specific`) from the competition registry in `model_manager.py`/`model_factory.py`, replacing the hardcoded `league`/`international` context list. Existing `--context`/`match_type` CLI and MCP contracts are unchanged; `international` becomes a caller of `general_purpose`. | — |
| US#89 | completed | Add a feature-superset validation check ensuring every `competition_specific` feature list is a superset of `general_purpose` for the same target. | — |
| US#90 | completed | Document the future "general-purpose prediction as a feature" stacking seam in the model-manager interface (no implementation — keeps the door open for tiers whose architectures diverge). | — |

### Phase 14b: Player Data Sourcing & Ingestion — Completed

> Source pivoted from FBref to FotMob (2026-06-27) after live verification: FBref now serves a Cloudflare JS challenge (403) to non-browser requests and would require a headless browser; Sofascore's API returns 403 Forbidden; Understat's player data is season-cumulative only. FotMob's internal JSON API (`fotmob.com/api/data/...`) returned real per-match player ratings/xG/xA with no anti-bot wall — see `documents/FRAI_TECHSPEC.md` Section 27.3.

| ID | Status | Description | Comments |
|---|---|---|---|
| US#91 | completed | Restructure `src/ingestion/` into per-source subpackages (`football_data/`, `understat/`, `common/`) and namespace `data/raw/` to match. | Registry uses `fotmob/` rather than `fbref/` for the player-data source — see US#92 |
| US#92 | completed | Build the FotMob fetcher (`src/ingestion/fotmob/fetcher.py`) for per-match player stats (rating, xG, xA, xGOT, shots, minutes). | — |
| US#93 | completed | Build player identity resolution — `player_dim` table keyed by FotMob's native player `id` (with Opta `optaId` as a secondary column) — and extend `config/team_mapping.json` with FotMob team-name variants. | — |
| US#94 | completed | Create `raw_player_match_stats` DuckDB table and merge/upsert logic (`src/ingestion/fotmob/merge.py`) from fetched FotMob data. | — |
| US#95 | completed | Add FotMob backfill + incremental refresh support, extending the `refresh-data` CLI command. | — |

### Phase 14c: Squad Feature Engineering & Model Integration — Completed

> Depends on Phase 14a and Phase 14b.

| ID | Status | Description | Comments |
|---|---|---|---|
| US#96 | completed | Add `SQUAD_*` rolling feature family to `feature_factory.py`, aggregating `raw_player_match_stats` into pre-match-safe squad-level form (e.g. `SQUAD_HOME_XG_MEAN_R5`). | — |
| US#97 | completed | Gate `SQUAD_*` features to competitions with `"SQUAD"` in `enabled_feature_groups` (i.e. `competition_specific` tier only), via the Phase 14a registry. | — |
| US#98 | completed | Retrain `competition_specific` models with the expanded feature set; re-run `select-best-models` to update `model_selection.yaml`. | — |
| US#99 | completed | Surface `SQUAD_*` feature contributions in forecast payload explainability output where used. | — |

### Phase 15a: Lineup Data Foundation

> All lineup-gated features (Phase 15b, 15c) depend on this phase. Three features from the original proposal were excluded as blocked: xG Chain Concentration (not exposed by FotMob summary API — requires StatsBomb/FBref event-level data), Deep Completion Share (same — spatial pass data not in FotMob player stats), and Big-League Minutes Ratio (requires multi-league global player history ingestion — separate project-scale effort).

| ID | Status | Description | Comments |
|---|---|---|---|
| US#100 | completed | **Explore FotMob lineup API**: verify what pre-match lineup data FotMob exposes (player IDs, positions, timing of announcement relative to kickoff), document the endpoint path and schema, and confirm player IDs join to `player_dim`. Produce a short findings note before any implementation. | **Findings (2026-07-03):** Same `matchDetails` endpoint (`content.lineup`), no new auth. Schema: `homeTeam/awayTeam → starters[]` each with `id` (matches `player_dim`), `name`, `positionId`, `usualPlayingPositionId`; `subs[]` list lacks `positionId`. Position ID ranges: 11=GK, 30–39=DEF, 60–69=MID, 80–89=ATT/WNG, 110+=ST. `lineupType` field distinguishes confirmed vs provisional — must verify pre-match values once PL season resumes (tested only on completed matches). Physical metrics (sprints, distance) confirmed **absent** from all stat groups (Top stats / Attack / Defense / Duels) — see US#105. Interceptions and Recoveries **available** in "Defense" group — feeds US#104. |
| US#101 | completed | **Implement FotMob lineup ingestion**: fetch and store pre-match starting XI — player_id, team_id, position_group (GK/DEF/MID/FWD), match_id — in a new `match_lineups` DuckDB table; extend `refresh-data` CLI to include lineup backfill. Depends on US#100. | Shipped 2026-07-03: `src/ingestion/fotmob/lineup.py` with `fetch_match_lineup`, `upsert_match_lineups`, `backfill_lineups_from_player_stats`; `fetch-lineups` CLI subcommand; `refresh-data` extended. 6 tests pass. Note: `backfill` re-derives FotMob match IDs from date-range scan (not stored in DB). Pre-match `lineupType` values to be verified once PL season resumes. |

### Phase 15b: General-Purpose Lineup Features

> Depends on Phase 15a. These features are competition-agnostic by design — raw rolling averages are replaced by relative, context-normalised signals that are meaningful for World Cup, Champions League, and mixed cup competitions.

| ID | Status | Description | Comments |
|---|---|---|---|
| US#102 | completed | **FRDS (FotMob Rating Dominance Share)**: for each match, compute `sum(rolling-avg rating of starting 11) / sum(rolling-avg rating of all players with ≥1 appearance for this team in the last 90 days)`. The denominator proxies the full available squad. Add as a general-purpose feature (home and away). Depends on US#101. | **Shipped (2026-07-04):** `compute_frds()` + `_resolve_fotmob_to_raw()` in `lineup_features.py`; `_compute_frds_features()` method in `feature_factory.py`; `FRDS_HOME`/`FRDS_AWAY` in schema.yaml; gated with SQUAD group in model_manager.py. 6 tests pass. Pool window = 90 days (`SQUAD_POOL_DAYS`). |
| US#103 | completed | **xOC (Top-3 Offensive Concentration)**: add `config/league_coefficients.yaml` mapping league_code → UEFA/FIFA coefficient (static config, not FotMob-derived); compute `sum of xG+xA per 90 for top-3 forward starters (by xG+xA rolling avg) / coefficient of their club league`. Add as a general-purpose feature (home and away). Depends on US#101. | **Shipped (2026-07-04):** `compute_xoc()` in `lineup_features.py`; `_compute_xoc_features()` in `feature_factory.py`; `XOC_HOME`/`XOC_AWAY` in schema.yaml; gated with SQUAD group. `league_coefficients.yaml` added (E0=1.00 baseline). 5 tests pass. |
| US#104 | completed | **Defensive Interception & Recovery Anchor**: extend `src/ingestion/fotmob/fetcher.py` to collect `interceptions` and `recoveries` from the Defense stat group per player; compute top-2 mean (interceptions + recoveries)/90 rolling R5 among starting DEF/MID. Add as a general-purpose feature. Depends on US#101 for position filter. | **Shipped (2026-07-04):** `_extract_defense_stat()` in fetcher.py; `interceptions`/`recoveries` columns added to merge.py schema + ALTER TABLE migrations; `compute_defensive_anchor()` in lineup_features.py; `_compute_defensive_anchor_features()` in feature_factory.py; `DEF_ANCHOR_HOME`/`DEF_ANCHOR_AWAY` in schema.yaml; gated with SQUAD group. 5 tests pass. |

### Phase 15c: EPL-Specific Features

> Depends on Phase 15a for lineup-gated variants. US#106 team-level variant is independent and can be built immediately.

| ID | Status | Description | Comments |
|---|---|---|---|
| US#105 | blocked | **Physical Performance Intensity Delta**: explore FotMob `matchDetails` for physical metrics (sprints, high-intensity runs, distance covered). | **Blocked (2026-07-03):** FotMob `playerStats` exposes only four stat groups — Top stats, Attack, Defense, Duels — none of which contain sprint count, distance covered, or high-intensity run data. Physical tracking data is not available through FotMob's internal API. Would require a dedicated physical-data provider (e.g. Opta, SkillCorner) as a new source. |
| US#106 | completed | **Luck Burnout (Net Attacking Outperformance)**: compute `(Goals + Assists) − (xG + xA)` rolling 5-match window per team, using existing `raw_player_match_stats` data (no lineup needed). Implement team-level aggregate first and add as an EPL-specific feature. Once US#101 is complete, extend to a forward-only-filtered variant and compare predictive value of both. | **Shipped (2026-07-03):** `LUCK_HOME_BURNOUT_R5` and `LUCK_AWAY_BURNOUT_R5` added to feature_factory.py, schema.yaml, and model_manager.py (gated with SQUAD group). 5 tests pass. Forward-filtered variant deferred — team-level signal deemed sufficient for Phase 15. |
