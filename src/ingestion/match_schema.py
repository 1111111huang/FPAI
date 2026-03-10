"""Pydantic schemas for raw match ingestion."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.ingestion.schema import map_league_code_to_tier


class MatchSchema(BaseModel):
    """Minimal raw-match schema required by the FPAI ingestion layer."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    match_date: date = Field(alias="Date")
    home_team: str = Field(alias="HomeTeam", min_length=1)
    away_team: str = Field(alias="AwayTeam", min_length=1)
    fthg: int = Field(alias="FTHG", ge=0)
    ftag: int = Field(alias="FTAG", ge=0)
    over25_odds_avg: float = Field(alias="BbAv>2.5", gt=1.0)
    tier: int = Field(alias="LeagueCode", ge=1, le=4)

    @model_validator(mode="before")
    @classmethod
    def map_over25_aliases(cls, values: object) -> object:
        """Allow either legacy BbAv>2.5 or modern Avg>2.5 source column."""
        if not isinstance(values, dict):
            return values
        if "BbAv>2.5" not in values and "Avg>2.5" in values:
            values["BbAv>2.5"] = values["Avg>2.5"]
        return values

    @field_validator("match_date", mode="before")
    @classmethod
    def parse_match_date(cls, value: object) -> date:
        """Parse dates from Football-Data CSV format (DD/MM/YYYY)."""
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value) if "-" in value else datetime.strptime(value, "%d/%m/%Y").date()
        raise TypeError("Date must be a date object or string")

    @field_validator("tier", mode="before")
    @classmethod
    def parse_tier(cls, value: object) -> int:
        """Convert league code into a numeric tier level."""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return map_league_code_to_tier(value)
        raise TypeError("LeagueCode must be a league code string or integer tier")
