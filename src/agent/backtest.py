"""Outcome loading and backtest replay (A12).

process_match_row() is the single source of truth for "run one historical
match through the agent in replay mode and score it" — used by both the
synchronous BacktestHarness.run() and the concurrent agent-backtest CLI path
(main.py), so the two never drift out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.agent.agent_config import AgentConfig
from src.agent.market_resolution import build_actual_outcome, market_correct as _market_correct
from src.utils.db_manager import DuckDBManager


@dataclass
class BacktestRecord:
    match_id: str
    home_team: str
    away_team: str
    date: str
    league: str
    recommendation: dict[str, Any]
    actual: dict[str, Any]
    market_results: list[dict[str, Any]] = field(default_factory=list)


def load_outcome(row: pd.Series) -> dict[str, Any]:
    """Derive the resolvable outcome categories for a finished match."""
    return build_actual_outcome(int(row["fthg"]), int(row["ftag"]))


def _date_str(row: pd.Series) -> str:
    value = row["date"]
    return str(value.date()) if hasattr(value, "date") else str(value)


def _build_match_info(row: pd.Series) -> dict[str, Any]:
    match_info: dict[str, Any] = {
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "date": _date_str(row),
        "league": row["league"],
    }
    if row.get("odds_h") and row.get("odds_d") and row.get("odds_a"):
        match_info["odds"] = {"home": row["odds_h"], "draw": row["odds_d"], "away": row["odds_a"]}
    return match_info


def process_match_row(row: pd.Series, config: AgentConfig) -> BacktestRecord:
    """Replay one historical match through the agent and score its recommendation.

    Sets the module-level SnapshotStore to replay mode for this match_id before
    calling run_agent, and always resets it to live mode afterward (even on
    error) so a failed match doesn't leave a later, unrelated call in replay
    mode by accident.
    """
    # Local imports: keep these inside the function — tests patch
    # src.agent.graph.run_agent and src.agent.tools.configure_snapshot_store,
    # which only works if these names are resolved at call time, not import time.
    from src.agent.graph import run_agent
    from src.agent import tools as agent_tools

    match_id = row["match_id"]
    match_info = _build_match_info(row)

    agent_tools.configure_snapshot_store("replay", match_id=match_id)
    try:
        recommendation = run_agent(match_info=match_info, config=config)
    finally:
        agent_tools.configure_snapshot_store("live")

    actual = load_outcome(row)
    market_results = [
        {**m, "correct": _market_correct(m, actual)}
        for m in recommendation.get("markets", [])
    ]
    return BacktestRecord(
        match_id=match_id,
        home_team=row["home_team"],
        away_team=row["away_team"],
        date=match_info["date"],
        league=row["league"],
        recommendation=recommendation,
        actual=actual,
        market_results=market_results,
    )


class BacktestHarness:
    """Loads historical matches and replays them through the agent via process_match_row()."""

    def __init__(self, config: AgentConfig | None = None, db_path: str = "config.yaml") -> None:
        self.config = config or AgentConfig.default()
        self.db = DuckDBManager(config_path=db_path)

    def load_matches(
        self,
        from_date: str,
        to_date: str,
        league: str | None = None,
        sample: int | None = None,
    ) -> pd.DataFrame:
        query = (
            "SELECT match_id, league, date, home_team, away_team, "
            "odds_h, odds_d, odds_a, fthg, ftag, hc, ac "
            "FROM raw_matches WHERE date >= ? AND date <= ? AND fthg IS NOT NULL AND ftag IS NOT NULL"
        )
        params: list[Any] = [from_date, to_date]
        if league:
            query += " AND UPPER(league) = ?"
            params.append(league.upper())
        query += " ORDER BY date"
        with self.db.connection() as conn:
            matches = conn.execute(query, params).fetchdf()

        if sample is not None and len(matches) > sample:
            matches = self._stratified_sample(matches, sample)
        return matches

    @staticmethod
    def _stratified_sample(matches: pd.DataFrame, sample: int) -> pd.DataFrame:
        """Stratify by actual result (home/draw/away) — the only outcome dimension
        known before running the agent. ('bet/no-bet' is the agent's own output,
        so it can't be used to pre-stratify the input sample.) Seeded for
        reproducibility so agent-compare (A16) can re-run different configs over
        the identical sample.
        """

        def _result(row: pd.Series) -> str:
            if row["fthg"] > row["ftag"]:
                return "home"
            if row["fthg"] < row["ftag"]:
                return "away"
            return "draw"

        matches = matches.copy()
        matches["_stratum"] = matches.apply(_result, axis=1)
        n_strata = matches["_stratum"].nunique()
        per_stratum = max(1, sample // n_strata)
        sampled = (
            matches.groupby("_stratum", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_stratum), random_state=42), include_groups=False)
        )
        # include_groups=False already drops "_stratum" from the result, so only
        # drop it if it's somehow still present (e.g. future pandas behavior change).
        if "_stratum" in sampled.columns:
            sampled = sampled.drop(columns="_stratum")
        return sampled.sort_values("date").reset_index(drop=True)

    def run(
        self,
        from_date: str,
        to_date: str,
        league: str | None = None,
        sample: int | None = None,
    ) -> list[BacktestRecord]:
        matches = self.load_matches(from_date, to_date, league=league, sample=sample)
        return [process_match_row(row, self.config) for _, row in matches.iterrows()]
