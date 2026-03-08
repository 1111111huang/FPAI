"""CSV ingestion logic for historical football match data."""

from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd
from pydantic import ValidationError

from src.ingestion.schema import MatchSchema
from src.utils.db_manager import DuckDBManager
from src.utils.helpers import generate_match_id
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class CSVLoader:
    """Load, validate, and persist football CSV data into DuckDB."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize loader using database settings from YAML config."""
        self.db_manager = DuckDBManager(config_path=config_path)

    def process_v1_csv(self, file_path: str, league_code: str) -> None:
        """Ingest a v1 football CSV file and insert validated rows into DuckDB."""
        df = pd.read_csv(file_path)

        renamed = df.rename(
            columns={
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "FTHG": "fthg",
                "FTAG": "ftag",
                "B365H": "odds_h",
                "B365D": "odds_d",
                "B365A": "odds_a",
            }
        )

        if "BbAv>2.5" in renamed.columns:
            over25_series = renamed["BbAv>2.5"]
        elif "Avg>2.5" in renamed.columns:
            over25_series = renamed["Avg>2.5"]
        else:
            over25_series = pd.Series([None] * len(renamed), index=renamed.index)

        working = renamed[
            ["Date", "home_team", "away_team", "fthg", "ftag", "odds_h", "odds_d", "odds_a"]
        ].copy()
        working["over25_odds_avg"] = over25_series

        # PRD rule: skip rows with missing critical odds inputs.
        before_drop = len(working)
        working = working.dropna(subset=["odds_h", "odds_d", "odds_a"])
        dropped_missing_odds = before_drop - len(working)
        if dropped_missing_odds > 0:
            LOGGER.warning(
                "Skipped %s rows with missing odds in %s.", dropped_missing_odds, file_path
            )

        parsed_dates = pd.to_datetime(working["Date"], dayfirst=True, errors="coerce")
        working["Date"] = [
            value.to_pydatetime() if not pd.isna(value) else None for value in parsed_dates
        ]

        records_to_insert: list[
            tuple[str, str, datetime, str, str, int, int, float, float, float]
        ] = []
        skipped_validation = 0

        for row in working.to_dict(orient="records"):
            try:
                validated = MatchSchema.model_validate(
                    {
                        "Date": row.get("Date"),
                        "HomeTeam": row.get("home_team"),
                        "AwayTeam": row.get("away_team"),
                        "FTHG": row.get("fthg"),
                        "FTAG": row.get("ftag"),
                        "BbAv>2.5": row.get("over25_odds_avg"),
                    }
                )
            except ValidationError:
                skipped_validation += 1
                continue

            match_datetime = pd.Timestamp(validated.match_date).to_pydatetime()
            match_id = generate_match_id(
                date=match_datetime.date().isoformat(),
                home_team=validated.home_team,
                away_team=validated.away_team,
            )

            records_to_insert.append(
                (
                    match_id,
                    league_code,
                    match_datetime,
                    validated.home_team,
                    validated.away_team,
                    validated.fthg,
                    validated.ftag,
                    float(row["odds_h"]),
                    float(row["odds_d"]),
                    float(row["odds_a"]),
                )
            )

        if skipped_validation > 0:
            LOGGER.warning(
                "Skipped %s rows due to schema validation failures in %s.",
                skipped_validation,
                file_path,
            )

        try:
            with self.db_manager.connection() as conn:
                self._create_raw_matches_table(conn)
                if records_to_insert:
                    conn.executemany(
                        """
                        INSERT OR IGNORE INTO raw_matches
                        (match_id, league, date, home_team, away_team, fthg, ftag, odds_h, odds_d, odds_a)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        records_to_insert,
                    )
                LOGGER.info(
                    "CSV ingestion completed for league %s with %s valid records.",
                    league_code,
                    len(records_to_insert),
                )
        except duckdb.Error:
            LOGGER.exception("Database failure while ingesting %s.", file_path)
            raise

    @staticmethod
    def _create_raw_matches_table(conn: duckdb.DuckDBPyConnection) -> None:
        """Create the raw_matches table if it does not already exist."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_matches (
                match_id TEXT PRIMARY KEY,
                league TEXT,
                date TIMESTAMP,
                home_team TEXT,
                away_team TEXT,
                fthg INTEGER,
                ftag INTEGER,
                odds_h FLOAT,
                odds_d FLOAT,
                odds_a FLOAT
            )
            """
        )
