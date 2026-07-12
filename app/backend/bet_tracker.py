"""W12: user_bets storage (D3a) -- logs bets the user actually placed, not
automatic hypothetical tracking. SQLite-backed (app/data/user_bets.db, same
pattern as W11's recommendation_cache), independent of both SnapshotStore and
the recommendation cache -- a different concern entirely.

profit_loss is computed at settlement time (W13 calls settle_bet()), not at
creation -- every bet starts 'open' with profit_loss=None."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Literal

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "user_bets.db"

Outcome = Literal["open", "won", "lost"]
Source = Literal["from_recommendation", "manual"]


@dataclass(frozen=True)
class Bet:
    id: int
    match_id: str
    date: str
    home_team: str
    away_team: str
    market: str
    selection: str
    odds: float
    stake: float
    outcome: Outcome
    profit_loss: float | None
    source: Source
    recommendation_snapshot: dict | None
    created_at: str


class BetTracker:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    market TEXT NOT NULL,
                    selection TEXT NOT NULL,
                    odds REAL NOT NULL,
                    stake REAL NOT NULL,
                    outcome TEXT NOT NULL DEFAULT 'open',
                    profit_loss REAL,
                    source TEXT NOT NULL,
                    recommendation_snapshot_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def create_bet(
        self,
        match_id: str,
        date: str,
        home_team: str,
        away_team: str,
        market: str,
        selection: str,
        odds: float,
        stake: float,
        source: Source,
        recommendation_snapshot: dict | None,
    ) -> Bet:
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        snapshot_json = json.dumps(recommendation_snapshot) if recommendation_snapshot is not None else None
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO user_bets
                (match_id, date, home_team, away_team, market, selection, odds, stake,
                 outcome, profit_loss, source, recommendation_snapshot_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', NULL, ?, ?, ?)
                """,
                (match_id, date, home_team, away_team, market, selection, odds, stake,
                 source, snapshot_json, created_at),
            )
            bet_id = cursor.lastrowid
        return self.get_bet(bet_id)  # type: ignore[return-value]

    def get_bet(self, bet_id: int) -> Bet | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_bets WHERE id = ?", (bet_id,)
            ).fetchone()
        return self._row_to_bet(row) if row else None

    def list_bets(self) -> list[Bet]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM user_bets ORDER BY id ASC").fetchall()
        return [self._row_to_bet(row) for row in rows]

    def list_open_bets(self) -> list[Bet]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM user_bets WHERE outcome = 'open' ORDER BY id ASC"
            ).fetchall()
        return [self._row_to_bet(row) for row in rows]

    def settle_bet(self, bet_id: int, outcome: Literal["won", "lost"]) -> Bet:
        bet = self.get_bet(bet_id)
        if bet is None:
            raise ValueError(f"Bet {bet_id} not found")
        profit_loss = bet.stake * (bet.odds - 1) if outcome == "won" else -bet.stake
        with self._connect() as conn:
            conn.execute(
                "UPDATE user_bets SET outcome = ?, profit_loss = ? WHERE id = ?",
                (outcome, profit_loss, bet_id),
            )
        return self.get_bet(bet_id)  # type: ignore[return-value]

    @staticmethod
    def _row_to_bet(row: tuple) -> Bet:
        (bet_id, match_id, date, home_team, away_team, market, selection, odds, stake,
         outcome, profit_loss, source, snapshot_json, created_at) = row
        return Bet(
            id=bet_id, match_id=match_id, date=date, home_team=home_team, away_team=away_team,
            market=market, selection=selection, odds=odds, stake=stake, outcome=outcome,
            profit_loss=profit_loss, source=source,
            recommendation_snapshot=json.loads(snapshot_json) if snapshot_json else None,
            created_at=created_at,
        )
