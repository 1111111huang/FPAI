"""MCP server exposing FPAI forecast tools to AI agents (US#83).

Zero business logic here — pure delegation to src/tools/.

Start with:
    python -m src.mcp_server
or configure as a Claude Desktop MCP server pointing to this module.
"""

from __future__ import annotations

from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

from src.tools.data_tools import get_data_freshness, list_matches
from src.tools.forecast_tools import forecast_matches, forecast_upcoming
from src.tools.model_tools import get_model_status

app = Server("fpai-forecast")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="forecast",
            description=(
                "Return structured forecast JSON for one or more historical feature-store matches. "
                "Use list_matches first to find valid match_ids."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "league": {"type": "string", "description": "Optional league code filter (e.g. 'E0')."},
                    "match_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional list of match IDs."},
                    "targets": {"type": "array", "items": {"type": "string"}, "description": "Optional target subset."},
                    "limit": {"type": "integer", "description": "Optional maximum number of matches."},
                },
            },
        ),
        types.Tool(
            name="forecast_upcoming",
            description=(
                "Produce an on-demand forecast for a named upcoming match without requiring it to exist "
                "in the feature store.\n\n"
                "match_type values:\n"
                "  - 'league' (default): full team history + market features. Requires --league.\n"
                "  - 'international': market-odds-only; team history skipped. --league optional.\n\n"
                "Response includes 'data_quality.prediction_basis' indicating which data was used."
            ),
            inputSchema={
                "type": "object",
                "required": ["home_team", "away_team", "date", "odds_h", "odds_d", "odds_a"],
                "properties": {
                    "home_team": {"type": "string"},
                    "away_team": {"type": "string"},
                    "date": {"type": "string", "description": "Match date in YYYY-MM-DD format."},
                    "league": {"type": "string", "description": "League code. Required for match_type='league'."},
                    "odds_h": {"type": "number", "description": "Home win decimal odds."},
                    "odds_d": {"type": "number", "description": "Draw decimal odds."},
                    "odds_a": {"type": "number", "description": "Away win decimal odds."},
                    "match_type": {
                        "type": "string",
                        "enum": ["league", "international"],
                        "description": "Prediction path. 'league' uses full team history; 'international' uses market odds only.",
                    },
                    "over25_odds": {"type": "number", "description": "Optional over 2.5 goals odds."},
                    "ah_line": {"type": "number", "description": "Optional Asian handicap line."},
                    "ah_home_odds": {"type": "number", "description": "Optional AH home odds."},
                    "ah_away_odds": {"type": "number", "description": "Optional AH away odds."},
                    "targets": {"type": "array", "items": {"type": "string"}, "description": "Optional target subset."},
                },
            },
        ),
        types.Tool(
            name="list_matches",
            description=(
                "List historical matches from the feature store. "
                "NOTE: Returns historical matches only — upcoming matches not yet played are NOT included. "
                "For upcoming match forecasts, use forecast_upcoming."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "league": {"type": "string", "description": "Optional league code filter."},
                    "from_date": {"type": "string", "description": "Optional ISO date lower bound (inclusive)."},
                    "to_date": {"type": "string", "description": "Optional ISO date upper bound (inclusive)."},
                    "limit": {"type": "integer", "description": "Optional maximum number of matches."},
                },
            },
        ),
        types.Tool(
            name="model_status",
            description="Return per-context per-target model selection status from model_selection.yaml.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="data_freshness",
            description="Return data freshness metadata: latest match date, days since update, match count, is_stale.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    import json

    try:
        if name == "forecast":
            result = forecast_matches(
                league=arguments.get("league"),
                match_ids=arguments.get("match_ids"),
                targets=arguments.get("targets"),
                limit=arguments.get("limit"),
            )
        elif name == "forecast_upcoming":
            result = forecast_upcoming(
                home_team=str(arguments["home_team"]),
                away_team=str(arguments["away_team"]),
                date=str(arguments["date"]),
                league=arguments.get("league"),
                odds_h=float(arguments["odds_h"]),
                odds_d=float(arguments["odds_d"]),
                odds_a=float(arguments["odds_a"]),
                match_type=str(arguments.get("match_type", "league")),
                over25_odds=arguments.get("over25_odds"),
                ah_line=arguments.get("ah_line"),
                ah_home_odds=arguments.get("ah_home_odds"),
                ah_away_odds=arguments.get("ah_away_odds"),
                targets=arguments.get("targets"),
            )
        elif name == "list_matches":
            result = list_matches(
                league=arguments.get("league"),
                from_date=arguments.get("from_date"),
                to_date=arguments.get("to_date"),
                limit=arguments.get("limit"),
            )
        elif name == "model_status":
            result = get_model_status()
        elif name == "data_freshness":
            result = get_data_freshness()
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        result = {"error": str(exc)}

    return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
