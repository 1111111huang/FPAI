# FPAI Forecast Engine User Stories

This document tracks story-level actionable items for the forecast-engine pivot. Default status is `active`. Completed stories are archived in `documents/FRAI_TECHSPEC.md` Section 23.

## Story Dependencies & Execution Order

**PHASE 1–11: All completed** — See Section 23 in `documents/FRAI_TECHSPEC.md` for the full archive.

**PHASE 12: CLI & Model Lifecycle — All completed** — See Sections 23 and 25 in `documents/FRAI_TECHSPEC.md`.

**PHASE 13: Agent Tool Layer — All completed** — See Sections 23 and 26 in `documents/FRAI_TECHSPEC.md`.

**PHASE 14: Player-Level Data & Competition Tiers — In progress (14a and 14b fully completed; 14c not started).** Design recorded in `documents/FRAI_TECHSPEC.md` Section 27. 14c depends on both 14a and 14b.

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

### Phase 14c: Squad Feature Engineering & Model Integration

> Depends on Phase 14a and Phase 14b.

| ID | Status | Description | Comments |
|---|---|---|---|
| US#96 | not started | Add `SQUAD_*` rolling feature family to `feature_factory.py`, aggregating `raw_player_match_stats` into pre-match-safe squad-level form (e.g. `SQUAD_XG_PER90_R5`). | — |
| US#97 | not started | Gate `SQUAD_*` features to competitions with `"SQUAD"` in `enabled_feature_groups` (i.e. `competition_specific` tier only), via the Phase 14a registry. | — |
| US#98 | not started | Retrain `competition_specific` models with the expanded feature set; re-run `select-best-models` to update `model_selection.yaml`. | — |
| US#99 | not started | Surface `SQUAD_*` feature contributions in forecast payload explainability output where used. | — |
