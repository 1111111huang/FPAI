"""Outcome loading and backtest replay (A12).

process_match_row() is the single source of truth for "run one historical
match through the agent in replay mode and score it" — used by both the
synchronous BacktestHarness.run() and the concurrent agent-backtest CLI path
(main.py), so the two never drift out of sync.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from src.agent.agent_config import AgentConfig
from src.agent.market_resolution import build_actual_outcome, market_correct as _market_correct
from src.utils.db_manager import DuckDBManager

_VALID_SPLITS = ("all", "train", "test")


def match_in_test_split(match_id: str, test_fraction: float) -> bool:
    """A40: stable per-match_id train/test assignment for critic-mode holdout.

    Hashes match_id directly rather than sampling the DataFrame (e.g.
    df.sample(frac=...)) so a given match's assignment never shifts as the
    snapshot corpus grows or as different date/league filters are applied --
    the same match_id always lands in the same split, with no split
    assignment table to persist or keep in sync.
    """
    digest = hashlib.sha256(match_id.encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < test_fraction


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
    full_state: dict[str, Any] | None = None


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


def process_match_row(
    row: pd.Series, config: AgentConfig, capture_state: bool = False, allow_lessons_in_replay: bool = False,
) -> BacktestRecord:
    """Replay one historical match through the agent and score its recommendation.

    Sets the module-level SnapshotStore to replay mode for this match_id before
    calling run_agent, and always resets it to live mode afterward (even on
    error) so a failed match doesn't leave a later, unrelated call in replay
    mode by accident.

    capture_state (A33): when True, also captures the full graph state
    (competition_resolution/research_evidence/forecast_payload) on the
    returned record's full_state, for agent-train's telemetry persistence.

    allow_lessons_in_replay (A41): when True, lessons_node loads approved
    lessons during this replay instead of skipping (A33's default for every
    non-live run). Only meaningful for agent-backtest --split test
    --use-lessons -- evaluating a held-out split against lessons approved
    from the disjoint train split. main.py's CLI layer is responsible for
    refusing --use-lessons on anything but --split test; this function has
    no split awareness of its own and just does what it's told.

    A42: runs the agent with no LLM-callable tools during replay (tools=[]),
    regardless of config.provider. The only LLM-callable tool is web_search
    (A31/A32 moved forecast/resolve_competition off the LLM entirely) --
    research_node already guarantees deterministic baseline evidence before
    the LLM's turn, and the prompt itself says most matches need nothing
    further ("skip straight to step 3 unless there's a specific gap").
    Whenever the LLM chooses to call web_search anyway, its query text is its
    own invention and essentially never byte-matches whatever was recorded
    (deterministic templated queries from research_node do match; this is
    specifically the LLM's *optional* follow-up call) -- SnapshotMissingError
    aborts that whole match. Observed at a 100% rate on a live DeepSeek smoke
    sample and previously implicated in llama3.1:8b's ~69% E0 skip rate.
    Removing the tool during replay doesn't change what evidence the LLM has
    (still the full research_node/forecast_node payload) -- it just removes
    a call that could never succeed in this mode, for any provider.
    """
    # Local imports: keep these inside the function — tests patch
    # src.agent.graph.run_agent and src.agent.tools.configure_snapshot_store,
    # which only works if these names are resolved at call time, not import time.
    from src.agent.graph import run_agent
    from src.agent import tools as agent_tools
    from src.agent.snapshot_store import league_base_dir

    match_id = row["match_id"]
    match_info = _build_match_info(row)

    agent_tools.configure_snapshot_store(
        "replay", match_id=match_id, match_date=match_info["date"], base_dir=league_base_dir(row["league"]),
        allow_lessons_in_replay=allow_lessons_in_replay,
    )
    try:
        if capture_state:
            full_state = run_agent(match_info=match_info, config=config, tools=[], return_full_state=True)
            recommendation = full_state["recommendation"]
        else:
            recommendation = run_agent(match_info=match_info, config=config, tools=[])
            full_state = None
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
        full_state=full_state,
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
        split: Literal["all", "train", "test"] = "all",
        test_fraction: float = 0.2,
    ) -> pd.DataFrame:
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")
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

        if split != "all":
            is_test = matches["match_id"].apply(lambda m: match_in_test_split(m, test_fraction))
            matches = matches[is_test if split == "test" else ~is_test].reset_index(drop=True)

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
        split: Literal["all", "train", "test"] = "all",
        test_fraction: float = 0.2,
    ) -> list[BacktestRecord]:
        matches = self.load_matches(from_date, to_date, league=league, sample=sample, split=split, test_fraction=test_fraction)
        return [process_match_row(row, self.config) for _, row in matches.iterrows()]
