# FPAI Forecast Engine User Stories

This document tracks story-level actionable items for the forecast-engine pivot. Default status is `active`. Completed stories are archived in `documents/FRAI_TECHSPEC.md` Section 23.

## Story Dependencies & Execution Order

**PHASE 1–11: All completed** — See Section 23 in `documents/FRAI_TECHSPEC.md` for the full archive.

**PHASE 12: CLI & Model Lifecycle — All completed** — See Sections 23 and 25 in `documents/FRAI_TECHSPEC.md`.

**PHASE 13: Agent Tool Layer — All completed** — See Sections 23 and 26 in `documents/FRAI_TECHSPEC.md`.

**PHASE 14: Player-Level Data & Competition Tiers — Planned, not started.** Design recorded in `documents/FRAI_TECHSPEC.md` Section 27. Sub-phases 14a and 14b have no dependency on each other and may be worked in parallel; 14c depends on both.

### Phase 14a: Model Tier Reorg — Completed
- **US#87**: Define `config/competitions.yaml` competition registry — `competition_id` mapped to `tier`, `league_code`, `enabled_feature_groups`, `player_data_sources`.
- **US#88**: Resolve model tier (`general_purpose` / `competition_specific`) from the competition registry in `model_manager.py`/`model_factory.py`, replacing the hardcoded `league`/`international` context list. Existing `--context`/`match_type` CLI and MCP contracts are unchanged; `international` becomes a caller of `general_purpose`.
- **US#89**: Add a feature-superset validation check ensuring every `competition_specific` feature list is a superset of `general_purpose` for the same target.
- **US#90**: Document the future "general-purpose prediction as a feature" stacking seam in the model-manager interface (no implementation — keeps the door open for tiers whose architectures diverge).

### Phase 14b: Player Data Sourcing & Ingestion
*Source pivoted from FBref to FotMob (2026-06-27) after live verification: FBref now serves a Cloudflare JS challenge (403) to non-browser requests and would require a headless browser; Sofascore's API returns 403 Forbidden; Understat's player data is season-cumulative only. FotMob's internal JSON API (`fotmob.com/api/data/...`) returned real per-match player ratings/xG/xA with no anti-bot wall — see `documents/FRAI_TECHSPEC.md` Section 27.3.*
- **US#91**: Restructure `src/ingestion/` into per-source subpackages (`football_data/`, `understat/`, `common/`) and namespace `data/raw/` to match; update the 3 existing import sites.
- **US#92**: Build the FotMob fetcher (`src/ingestion/fotmob/fetcher.py`) for per-match player stats (rating, xG, xA, xGOT, shots, minutes).
- **US#93**: Build player identity resolution — `player_dim` table keyed by FotMob's native player `id` (with Opta `optaId` as a secondary column) — and extend `config/team_mapping.json` with FotMob team-name variants.
- **US#94**: Create `raw_player_match_stats` DuckDB table and merge/upsert logic (`src/ingestion/fotmob/merge.py`) from fetched FotMob data.
- **US#95**: Add FotMob backfill + incremental refresh support, extending the `refresh-data` CLI command.

### Phase 14c: Squad Feature Engineering & Model Integration
*Depends on Phase 14a and Phase 14b.*
- **US#96**: Add `SQUAD_*` rolling feature family to `feature_factory.py`, aggregating `raw_player_match_stats` into pre-match-safe squad-level form (e.g. `SQUAD_XG_PER90_R5`).
- **US#97**: Gate `SQUAD_*` features to competitions with `"SQUAD"` in `enabled_feature_groups` (i.e. `competition_specific` tier only), via the Phase 14a registry.
- **US#98**: Retrain `competition_specific` models with the expanded feature set; re-run `select-best-models` to update `model_selection.yaml`.
- **US#99**: Surface `SQUAD_*` feature contributions in forecast payload explainability output where used.
