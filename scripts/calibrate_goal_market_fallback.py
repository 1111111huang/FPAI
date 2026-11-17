"""2026-11-16: offline calibration for the goal-market lambda NaN fallback --
see docs/superpowers/specs/2026-11-16-goal-market-lambda-fallback-design.md.

Fits a per-league linear mapping from a match's own 1X2-implied-probability
gap (MKT_IMPLIED_HOME - MKT_IMPLIED_AWAY) to its real home goal-share
(MKT_LAMBDA_HOME / MKT_LAMBDA_TOTAL), using every feature_store row where a
real Asian Handicap line already exists -- i.e. where the true split is
known, so the fit can be checked against ground truth. The result is used by
FeatureFactory.build_for_match (src/features/feature_factory.py) only when
the AH line is genuinely missing, in place of propagating NaN.

Run against production's DB:

    python scripts/calibrate_goal_market_fallback.py

Paste the printed per-league (slope, intercept) pairs into
HOME_SHARE_COEFFICIENTS in src/features/goal_market_fallback.py, replacing
the interim flat-prior placeholder there.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.db_manager import DuckDBManager  # noqa: E402

# Below this many real-AH-line rows, keep the flat default rather than fit
# noise from too small a sample.
_MIN_ROWS_PER_LEAGUE = 30


def fit_home_share_coefficients(rows: list[tuple[float, float, float, float]]) -> tuple[float, float]:
    """rows: (lambda_total, lambda_home, implied_home, implied_away) tuples,
    all real (caller filters out any NaN/non-positive lambda_total first).
    Returns (slope, intercept) for
        real_home_share ~= intercept + slope * (implied_home - implied_away)
    fit by ordinary least squares (numpy.polyfit, degree 1)."""
    gaps = np.array([implied_home - implied_away for _, _, implied_home, implied_away in rows])
    shares = np.array([lambda_home / lambda_total for lambda_total, lambda_home, _, _ in rows])
    slope, intercept = np.polyfit(gaps, shares, deg=1)
    return float(slope), float(intercept)


def _fetch_rows_by_league(db_manager: DuckDBManager) -> dict[str, list[tuple[float, float, float, float]]]:
    """{league: [(lambda_total, lambda_home, implied_home, implied_away), ...]}
    -- only rows with a real MKT_AH_LINE (the true split is known for these,
    which is what makes them usable as calibration ground truth)."""
    rows_by_league: dict[str, list[tuple[float, float, float, float]]] = {}
    with db_manager.connection(read_only=True) as conn:
        query = """
            SELECT rm.league, fs."MKT_LAMBDA_TOTAL", fs."MKT_LAMBDA_HOME",
                   fs."MKT_IMPLIED_HOME", fs."MKT_IMPLIED_AWAY"
            FROM feature_store fs
            JOIN raw_matches rm ON fs.match_id = rm.match_id
            WHERE fs."MKT_AH_LINE" IS NOT NULL
              AND fs."MKT_LAMBDA_TOTAL" IS NOT NULL AND fs."MKT_LAMBDA_TOTAL" > 0
              AND fs."MKT_LAMBDA_HOME" IS NOT NULL
              AND fs."MKT_IMPLIED_HOME" IS NOT NULL AND fs."MKT_IMPLIED_AWAY" IS NOT NULL
        """
        for league, lambda_total, lambda_home, implied_home, implied_away in conn.execute(query).fetchall():
            rows_by_league.setdefault(league, []).append((lambda_total, lambda_home, implied_home, implied_away))
    return rows_by_league


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml", help="Config file that points at the DB.")
    args = parser.parse_args(argv)

    rows_by_league = _fetch_rows_by_league(DuckDBManager(config_path=args.config))
    if not rows_by_league:
        print("No feature_store rows with a real AH line found.")
        return 1

    print(f"{'league':<8} {'n_rows':>7} {'slope':>10} {'intercept':>10} {'MAE':>8} {'MAE@50/50':>10}")
    for league, rows in sorted(rows_by_league.items()):
        if len(rows) < _MIN_ROWS_PER_LEAGUE:
            print(f"{league:<8} {len(rows):>7}   -- too few rows, keep the flat default --")
            continue
        slope, intercept = fit_home_share_coefficients(rows)
        # The fitted split must beat a naive flat 50/50 split on this same
        # data, or it isn't earning its complexity -- printed here so the
        # person running this script can eyeball it before accepting the
        # fit (not a fixed accept/reject threshold).
        real_shares = [lambda_home / lambda_total for lambda_total, lambda_home, _, _ in rows]
        fitted_shares = [
            intercept + slope * (implied_home - implied_away)
            for _, _, implied_home, implied_away in rows
        ]
        fitted_mae = sum(abs(r - f) for r, f in zip(real_shares, fitted_shares)) / len(rows)
        baseline_mae = sum(abs(r - 0.5) for r in real_shares) / len(rows)
        print(
            f"{league:<8} {len(rows):>7} {slope:>10.4f} {intercept:>10.4f} "
            f"{fitted_mae:>8.4f} {baseline_mae:>10.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
