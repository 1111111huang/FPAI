"""CSV ingestion logic for historical football match data."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
from pydantic import ValidationError

from src.ingestion.football_data.match_schema import MatchSchema
from src.utils.db_manager import DuckDBManager
from src.utils.helpers import generate_match_id, standardize_team_name
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class CSVLoader:
    """Load, validate, and persist football CSV data into DuckDB."""

    def __init__(
        self,
        config_path: str = "config.yaml",
        default_max_retries: int = 5,
        default_retry_delay_seconds: float = 1.0,
    ) -> None:
        """Initialize loader using database settings from YAML config.

        US#159: default_max_retries/default_retry_delay_seconds forwarded
        straight to the DuckDBManager this loader builds internally -- a
        caller that wants a longer lock-retry window (main.py's CLI, see
        DuckDBManager.__init__'s own docstring) needs it applied here too,
        not just on its own separate db_manager instance, since this class
        builds its own rather than reusing one passed in."""
        self.db_manager = DuckDBManager(
            config_path=config_path,
            default_max_retries=default_max_retries,
            default_retry_delay_seconds=default_retry_delay_seconds,
        )
        self.raw_data_dir = Path(self.db_manager.settings.paths.raw_data_dir) / "football_data"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def process_v1_csv(self, file_path: str, league_code: str, overwrite: bool = False) -> int:
        """Ingest a v1 football CSV file and return how many matches were added."""
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            LOGGER.warning("Skipping unreadable CSV file %s (%s)", file_path, exc)
            return 0

        # US#156: distinguishes a column that's always been absent for this
        # league (never flagged -- tolerated as known-optional, e.g. an
        # Asian-handicap column a smaller league's CSV has simply never
        # had) from one that *used to be present and just vanished* -- a
        # real signal the upstream source's format changed, worth a human
        # looking at, not silently absorbed the same way as an
        # always-optional column. Runs before the required-columns check
        # below so a HARD-required column disappearing (the more severe
        # case) is flagged too, not just optional ones.
        with self.db_manager.connection() as conn:
            self._check_and_update_schema_baseline(conn, league_code, file_path, set(df.columns))

        renamed = df.rename(
            columns={
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "FTHG": "fthg",
                "FTAG": "ftag",
                "B365H": "odds_h",
                "B365D": "odds_d",
                "B365A": "odds_a",
                "HS": "hs",
                "AS": "as",
                "HST": "hst",
                "AST": "ast",
                "HC": "hc",
                "AC": "ac",
                "HY": "hy",
                "AY": "ay",
                "HR": "hr",
                "AR": "ar",
                "AvgH": "avgh",
                "AvgD": "avgd",
                "AvgA": "avga",
                "xG_H": "xg_h",
                "xG_A": "xg_a",
                "xGA_H": "xga_h",
                "xGA_A": "xga_a",
            }
        )
        required_columns = {"Date", "home_team", "away_team", "fthg", "ftag", "odds_h", "odds_d", "odds_a"}
        missing_required = required_columns.difference(renamed.columns)
        if missing_required:
            LOGGER.warning(
                "Skipping file %s due to missing required columns: %s",
                file_path,
                ", ".join(sorted(missing_required)),
            )
            return 0

        if "BbAv>2.5" in renamed.columns:
            over25_series = renamed["BbAv>2.5"]
        elif "Avg>2.5" in renamed.columns:
            over25_series = renamed["Avg>2.5"]
        else:
            over25_series = pd.Series([None] * len(renamed), index=renamed.index)

        # Under 2.5: modern CSVs use "Avg<2.5"; older seasons use "BbAv<2.5"
        under25_series = (
            renamed["Avg<2.5"] if "Avg<2.5" in renamed.columns
            else renamed.get("BbAv<2.5", pd.Series([None] * len(renamed), index=renamed.index))
        )
        # Asian handicap: modern CSVs use "AHh"/"AvgAHH"/"AvgAHA"; older use "BbAHh"/"BbAvAHH"/"BbAvAHA"
        ah_line_series = (
            renamed["AHh"] if "AHh" in renamed.columns
            else renamed.get("BbAHh", pd.Series([None] * len(renamed), index=renamed.index))
        )
        ah_home_series = (
            renamed["AvgAHH"] if "AvgAHH" in renamed.columns
            else renamed.get("BbAvAHH", pd.Series([None] * len(renamed), index=renamed.index))
        )
        ah_away_series = (
            renamed["AvgAHA"] if "AvgAHA" in renamed.columns
            else renamed.get("BbAvAHA", pd.Series([None] * len(renamed), index=renamed.index))
        )

        # Some historical CSVs do not ship AvgH/AvgD/AvgA or discipline/shot metrics.
        # Ensure optional columns exist so downstream selection doesn't crash.
        optional_columns = [
            "hs",
            "as",
            "hst",
            "ast",
            "hc",
            "ac",
            "hy",
            "ay",
            "hr",
            "ar",
            "avgh",
            "avgd",
            "avga",
            "xg_h",
            "xg_a",
            "xga_h",
            "xga_a",
            "over25_odds",
            "under25_odds",
            "ah_line",
            "ah_home_odds",
            "ah_away_odds",
        ]
        for col in optional_columns:
            if col not in renamed.columns:
                renamed[col] = None

        working = renamed[
            [
                "Date",
                "home_team",
                "away_team",
                "fthg",
                "ftag",
                "odds_h",
                "odds_d",
                "odds_a",
                "hs",
                "as",
                "hst",
                "ast",
                "hc",
                "ac",
                "hy",
                "ay",
                "hr",
                "ar",
                "avgh",
                "avgd",
                "avga",
                "xg_h",
                "xg_a",
                "xga_h",
                "xga_a",
                "over25_odds",
                "under25_odds",
                "ah_line",
                "ah_home_odds",
                "ah_away_odds",
            ]
        ].copy()
        working["home_team"] = working["home_team"].astype(str).map(standardize_team_name)
        working["away_team"] = working["away_team"].astype(str).map(standardize_team_name)
        working["over25_odds_avg"] = over25_series
        working["over25_odds"] = over25_series
        working["under25_odds"] = under25_series
        working["ah_line"] = ah_line_series
        working["ah_home_odds"] = ah_home_series
        working["ah_away_odds"] = ah_away_series

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

        records_to_insert: list[tuple] = []
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
                        "LeagueCode": league_code,
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
                league=league_code,
            )

            def _f(v: object) -> float | None:
                return float(v) if pd.notna(v) else None

            records_to_insert.append(
                (
                    match_id,
                    league_code,
                    validated.tier,
                    match_datetime,
                    validated.home_team,
                    validated.away_team,
                    validated.fthg,
                    validated.ftag,
                    float(row["odds_h"]),
                    float(row["odds_d"]),
                    float(row["odds_a"]),
                    _f(row["hs"]),
                    _f(row["as"]),
                    _f(row["hst"]),
                    _f(row["ast"]),
                    _f(row["hc"]),
                    _f(row["ac"]),
                    _f(row["hy"]),
                    _f(row["ay"]),
                    _f(row["hr"]),
                    _f(row["ar"]),
                    _f(row["avgh"]),
                    _f(row["avgd"]),
                    _f(row["avga"]),
                    _f(row["xg_h"]),
                    _f(row["xg_a"]),
                    _f(row["xga_h"]),
                    _f(row["xga_a"]),
                    _f(row["over25_odds"]),
                    _f(row["under25_odds"]),
                    _f(row["ah_line"]),
                    _f(row["ah_home_odds"]),
                    _f(row["ah_away_odds"]),
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
                before_row = conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()
                before_count = int(before_row[0]) if before_row is not None else 0
                if records_to_insert:
                    insert_clause = "INSERT OR REPLACE" if overwrite else "INSERT OR IGNORE"
                    conn.executemany(
                        f"""
                        {insert_clause} INTO raw_matches
                        (match_id, league, tier, date, home_team, away_team, fthg, ftag, odds_h, odds_d, odds_a,
                         hs, "as", hst, ast, hc, ac, hy, ay, hr, ar, avgh, avgd, avga, xg_h, xg_a, xga_h, xga_a,
                         over25_odds, under25_odds, ah_line, ah_home_odds, ah_away_odds)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        records_to_insert,
                    )
                after_row = conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()
                after_count = int(after_row[0]) if after_row is not None else before_count
                added_matches = max(0, after_count - before_count)
                LOGGER.info(
                    "CSV ingestion completed for league %s with %s valid records, %s new matches added.",
                    league_code,
                    len(records_to_insert),
                    added_matches,
                )
                return added_matches
        except duckdb.Error:
            LOGGER.exception("Database failure while ingesting %s.", file_path)
            raise

    def process_directory(self, pattern: str = "E0_*.csv", force: bool = False) -> int:
        """Process all matching CSV files in raw data directory with change detection."""
        files = sorted(self.raw_data_dir.glob(pattern))
        if not files:
            LOGGER.warning("No files found for pattern %s in %s", pattern, self.raw_data_dir)
            return 0

        total_added = 0
        with self.db_manager.connection() as conn:
            self._create_processed_files_table(conn)

        for file_path in files:
            file_hash = self._compute_file_hash(file_path)
            if not force and self._is_file_unchanged(file_path=file_path, file_hash=file_hash):
                LOGGER.info("Skipping unchanged file: %s", file_path.name)
                continue

            league_code = file_path.stem.split("_")[0]
            added = self.process_v1_csv(
                file_path=str(file_path),
                league_code=league_code,
                overwrite=force,
            )
            total_added += added
            self._mark_file_processed(file_path=file_path, file_hash=file_hash)

        LOGGER.info(
            "Batch ingestion finished for pattern %s | total_new_matches_added=%s",
            pattern,
            total_added,
        )
        return total_added

    @staticmethod
    def _create_raw_matches_table(conn: duckdb.DuckDBPyConnection) -> None:
        """Create the raw_matches table if it does not already exist."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_matches (
                match_id TEXT PRIMARY KEY,
                league TEXT,
                tier INTEGER,
                date TIMESTAMP,
                home_team TEXT,
                away_team TEXT,
                fthg INTEGER,
                ftag INTEGER,
                odds_h FLOAT,
                odds_d FLOAT,
                odds_a FLOAT,
                hs FLOAT,
                "as" FLOAT,
                hst FLOAT,
                ast FLOAT,
                hc FLOAT,
                ac FLOAT,
                hy FLOAT,
                ay FLOAT,
                hr FLOAT,
                ar FLOAT,
                avgh FLOAT,
                avgd FLOAT,
                avga FLOAT,
                xg_h FLOAT,
                xg_a FLOAT,
                xga_h FLOAT,
                xga_a FLOAT,
                over25_odds FLOAT,
                under25_odds FLOAT,
                ah_line FLOAT,
                ah_home_odds FLOAT,
                ah_away_odds FLOAT
            )
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info('raw_matches')").fetchall()}
        if "tier" not in columns:
            conn.execute("ALTER TABLE raw_matches ADD COLUMN tier INTEGER")
        for col in [
            "hs",
            "as",
            "hst",
            "ast",
            "hc",
            "ac",
            "hy",
            "ay",
            "hr",
            "ar",
            "avgh",
            "avgd",
            "avga",
            "xg_h",
            "xg_a",
            "xga_h",
            "xga_a",
            "over25_odds",
            "under25_odds",
            "ah_line",
            "ah_home_odds",
            "ah_away_odds",
        ]:
            if col not in columns:
                if col == "as":
                    conn.execute('ALTER TABLE raw_matches ADD COLUMN "as" FLOAT')
                else:
                    conn.execute(f"ALTER TABLE raw_matches ADD COLUMN {col} FLOAT")

    @staticmethod
    def _create_processed_files_table(conn: duckdb.DuckDBPyConnection) -> None:
        """Create metadata table to track processed input files."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_files (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    @staticmethod
    def _create_schema_baseline_table(conn: duckdb.DuckDBPyConnection) -> None:
        """US#156: a separate table, keyed by league rather than file_path
        (deliberately -- a brand-new season file for an already-established
        league, e.g. E0_2627.csv appearing fresh each August, should still
        be compared against that league's last known-good columns, not
        start with no baseline just because this exact file is new).
        Independent of processed_files' own file_hash-keyed row so this
        never risks being clobbered by _mark_file_processed's own
        INSERT OR REPLACE (which only ever names file_path/file_hash/
        processed_at)."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_baselines (
                league TEXT PRIMARY KEY,
                columns TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def _check_and_update_schema_baseline(
        self, conn: duckdb.DuckDBPyConnection, league_code: str, file_path: str, current_columns: set[str]
    ) -> None:
        """US#156: logs a distinguishable warning (not the same generic
        wording as an always-tolerated missing-optional-column case) when a
        column present in this league's last successful ingestion has
        vanished from the current file -- a real signal of upstream format
        drift, not routine season-to-season variation. No baseline yet
        (this league's first-ever ingestion) means nothing to compare
        against, so it's silently established rather than flagged."""
        self._create_schema_baseline_table(conn)
        row = conn.execute("SELECT columns FROM schema_baselines WHERE league = ?", [league_code]).fetchone()
        if row is not None:
            previous_columns = set(json.loads(row[0]))
            missing = previous_columns - current_columns
            if missing:
                LOGGER.warning(
                    "SCHEMA DRIFT DETECTED | league=%s | file=%s | columns present in a prior "
                    "successful ingestion for this league but now missing: %s",
                    league_code, file_path, sorted(missing),
                )
        conn.execute(
            """
            INSERT OR REPLACE INTO schema_baselines (league, columns, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            [league_code, json.dumps(sorted(current_columns))],
        )

    def _is_file_unchanged(self, file_path: Path, file_hash: str) -> bool:
        """Check if the file has already been processed with the same hash."""
        with self.db_manager.connection() as conn:
            self._create_processed_files_table(conn)
            row = conn.execute(
                "SELECT file_hash FROM processed_files WHERE file_path = ?",
                [str(file_path.resolve())],
            ).fetchone()
        return row is not None and str(row[0]) == file_hash

    def _mark_file_processed(self, file_path: Path, file_hash: str) -> None:
        """Upsert processed file metadata after successful ingestion."""
        with self.db_manager.connection() as conn:
            self._create_processed_files_table(conn)
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_files (file_path, file_hash, processed_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                [str(file_path.resolve()), file_hash],
            )

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 checksum for a local file."""
        hasher = hashlib.sha256()
        with file_path.open("rb") as infile:
            for chunk in iter(lambda: infile.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
