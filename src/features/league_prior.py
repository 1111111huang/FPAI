"""US#153: league-wide per-column prior for fully-cold-start imputation.

_apply_cold_start_imputation's own per-call mean (src/features/feature_factory.py)
has nothing to average from when a match involves two genuinely unseen teams
-- that call's own frame carries zero rows for either side by definition.
This computes the same kind of column mean US#134 already established
(per-competition, never cross-competition), just sourced from the full
persisted feature_store table instead of a single call's own empty frame."""

from __future__ import annotations

import duckdb

from src.utils.db_manager import DuckDBManager


def compute_league_prior(
    db_manager: DuckDBManager, league: str, columns: list[str],
) -> dict[str, float]:
    """Real per-column mean, joined against raw_matches and filtered to one
    league/competition -- never averaged across competitions. A column
    omitted from the returned dict means either every row was NULL for this
    league (a feature family this competition genuinely doesn't have) or no
    rows exist for this league at all -- the caller's own fallback (a flat
    0.0) still applies in that case, unchanged; this never invents a value
    where there is truly nothing to compute it from.

    Tolerates a missing feature_store table entirely (e.g. the offline
    compute_rolling_stats() pipeline has never run against this DB) by
    returning no prior rather than raising -- forecast_upcoming must never
    fail just because feature_store hasn't been populated yet, same
    "missing persistence = no signal, don't crash" contract this codebase
    already applies elsewhere (e.g. load_approved_lessons)."""
    if not columns:
        return {}
    quoted_selects = ", ".join(f'AVG(fs."{c}") AS "{c}"' for c in columns)
    try:
        with db_manager.connection(read_only=True) as conn:
            row = conn.execute(
                f"""
                SELECT {quoted_selects}
                FROM feature_store fs
                JOIN raw_matches rm ON fs.match_id = rm.match_id
                WHERE rm.league = ?
                """,
                [league],
            ).fetchone()
    except duckdb.CatalogException:
        return {}
    return {col: value for col, value in zip(columns, row) if value is not None}
