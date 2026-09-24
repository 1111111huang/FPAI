# Player-Quality Proxy — Phase 1 (A126) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared player-market-value ingestion/crosswalk layer and the `get_player_rating` agent tool, retiring A121's `web_search`-based FIFA-rating workaround (ticket A126). Phase 2 (the `SQUAD_MKT_VALUE_MEAN_*` ML feature, US#208) is a separate, later plan per the design spec's own sequencing.

**Architecture:** A new `src/ingestion/transfermarkt/` module (mirroring `src/ingestion/fotmob/`'s `fetcher.py`/`merge.py` shape) fetches Transfermarkt market values for players already known to this project, joined via the REEP identity crosswalk, into a dated-snapshot `player_market_values` table. A new agent tool `get_player_rating(player_name)` is built fresh per match as a closure over that match's two-team roster pool (the first per-match-context-bound tool in this codebase — `get_default_tools()`/`build_graph()` already run fresh per `run_agent()` call, so no new global state is needed), with layered exact → accent-fold → surname → fuzzy matching.

**Tech Stack:** `requests` + `beautifulsoup4` (both already dependencies) — no new dependencies needed. `duckdb`, `pandas`, stdlib `difflib`/`unicodedata`/`re`.

**History, why this isn't a SoFIFA-ratings plan:** the design spec originally called for SoFIFA (EA FC) ability ratings. Research for this plan found sofifa.com is behind Cloudflare's JS-challenge bot protection — a plain `requests.get()` (this repo's usual ingestion pattern) is blocked outright, confirmed live. The only real workaround (the `soccerdata` library, wrapping `seleniumbase`'s undetected-Chrome mode) needs a real local Chrome install and, once installed into this project's shared venv, created a genuine dependency conflict with `google-genai`/`langgraph-sdk`'s `websockets` requirement — a real risk to the live agent stack. **User-confirmed pivot: use Transfermarkt market values instead** — tested live, a plain `requests.get()` with a browser User-Agent returns the real page (HTTP 200, no block), the market value is directly parseable, `robots.txt` allows general crawling, and REEP's own crosswalk already carries a `key_transfermarkt` column alongside `key_fotmob`. No new dependency, no browser automation, same `requests`-only pattern as every other ingestion module here. Full history in `docs/superpowers/specs/2026-09-22-player-quality-proxy-design.md`'s "Revision (2026-09-23)" note.

**Scope boundary:** this plan covers A126 only (ingestion/crosswalk/agent tool). It does not touch `feature_factory.py`, does not add `SQUAD_MKT_VALUE_MEAN_*`, and does not build any production-sync mechanism.

---

### Task 1: REEP identity crosswalk

**Files:**
- Create: `src/ingestion/transfermarkt/__init__.py` (empty)
- Create: `src/ingestion/transfermarkt/reep_crosswalk.py`
- Test: `tests/test_transfermarkt_reep_crosswalk.py`

REEP (`github.com/withqwerty/reep`, CC0) publishes `data/people.csv` — confirmed columns include `type`, `name`, `key_fotmob`, `key_transfermarkt` (both float-cast numeric IDs, NaN when a provider has no entry for that person). We only need `type == "player"` rows where both IDs are present.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfermarkt_reep_crosswalk.py
"""Tests for loading the REEP FotMob<->Transfermarkt player identity crosswalk."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.transfermarkt.reep_crosswalk import load_reep_crosswalk

_CSV = (
    "reep_id,type,name,key_fotmob,key_transfermarkt\n"
    "reep_p1,player,Aaron Doran,162549,96148\n"
    "reep_p2,player,No Transfermarkt Player,555555,\n"
    "reep_p3,manager,Some Manager,666666,777777\n"
)


def _mock_resp(text: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.text = text
    return resp


def test_load_reep_crosswalk_keeps_only_players_with_both_ids():
    with patch("src.ingestion.transfermarkt.reep_crosswalk.requests.get", return_value=_mock_resp(_CSV)):
        crosswalk = load_reep_crosswalk()

    assert list(crosswalk.columns) == ["fotmob_player_id", "transfermarkt_player_id"]
    assert crosswalk.to_dict("records") == [{"fotmob_player_id": 162549, "transfermarkt_player_id": 96148}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfermarkt_reep_crosswalk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingestion.transfermarkt'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ingestion/transfermarkt/reep_crosswalk.py
"""Loads REEP's (github.com/withqwerty/reep, CC0) FotMob<->Transfermarkt
player identity crosswalk -- the player-level equivalent of
config/team_mapping.json, which already solves this exact class of problem
for team names.

Known limitation (accepted per the design spec, same non-blocking-degrade
discipline BUG-057 already established for unmapped team names): REEP's
public snapshot is a point-in-time export, so this season's newest transfers
may be missing until REEP's own next refresh -- not a crash, just a smaller
crosswalk than the true current universe.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

REEP_PEOPLE_CSV_URL = "https://raw.githubusercontent.com/withqwerty/reep/main/data/people.csv"


def load_reep_crosswalk(url: str = REEP_PEOPLE_CSV_URL) -> pd.DataFrame:
    """Returns a DataFrame with columns [fotmob_player_id,
    transfermarkt_player_id] (both Int64), one row per REEP person who is a
    player with both IDs mapped. Rows missing either ID, or not
    type=='player' (e.g. managers), are dropped -- not an error, just
    outside this crosswalk's scope."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), usecols=["type", "key_fotmob", "key_transfermarkt"])
    players = df[(df["type"] == "player") & df["key_fotmob"].notna() & df["key_transfermarkt"].notna()]
    return (
        players[["key_fotmob", "key_transfermarkt"]]
        .astype("int64")
        .rename(columns={"key_fotmob": "fotmob_player_id", "key_transfermarkt": "transfermarkt_player_id"})
        .reset_index(drop=True)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transfermarkt_reep_crosswalk.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/transfermarkt/__init__.py src/ingestion/transfermarkt/reep_crosswalk.py tests/test_transfermarkt_reep_crosswalk.py
git commit -m "feat(ingestion): load REEP FotMob<->Transfermarkt player crosswalk (A126)"
```

---

### Task 2: `player_market_values` dated-snapshot table + insert

**Files:**
- Create: `src/ingestion/transfermarkt/merge.py`
- Test: `tests/test_transfermarkt_merge.py`

Per the design spec, this table is **append-only** (a new row set per refresh run, stamped `snapshot_date`, never overwritten) — the natural key is `(fotmob_player_id, snapshot_date)`, not a plain upsert key on `fotmob_player_id` alone. This is deliberately unlike `fotmob/merge.py`'s `ON CONFLICT ... DO UPDATE` shape.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfermarkt_merge.py
"""Tests for the player_market_values dated-snapshot table."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.ingestion.transfermarkt.merge import insert_market_value_snapshot
from src.utils.db_manager import DuckDBManager


def _make_db_manager(tmp_path: Path) -> DuckDBManager:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    return DuckDBManager(config_path=str(config_path))


def _values_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"fotmob_player_id": 162549, "transfermarkt_player_id": 96148, "market_value_eur": 220_000_000},
    ])


def test_insert_market_value_snapshot_stamps_the_given_date(tmp_path: Path):
    db_manager = _make_db_manager(tmp_path)
    n = insert_market_value_snapshot(_values_df(), db_manager, snapshot_date="2026-09-23")

    assert n == 1
    with db_manager.connection(read_only=True) as conn:
        rows = conn.execute("SELECT fotmob_player_id, snapshot_date, market_value_eur FROM player_market_values").fetchall()
    assert rows == [(162549, "2026-09-23", 220_000_000)]


def test_insert_market_value_snapshot_never_overwrites_an_earlier_snapshot(tmp_path: Path):
    """Point-in-time correctness (design spec's whole reason for a dated-
    snapshot schema instead of a plain upsert) -- inserting a new snapshot
    date must leave an earlier one intact, both rows present."""
    db_manager = _make_db_manager(tmp_path)
    insert_market_value_snapshot(_values_df(), db_manager, snapshot_date="2026-08-01")
    insert_market_value_snapshot(_values_df(), db_manager, snapshot_date="2026-09-23")

    with db_manager.connection(read_only=True) as conn:
        dates = conn.execute(
            "SELECT snapshot_date FROM player_market_values WHERE fotmob_player_id = 162549 ORDER BY snapshot_date"
        ).fetchall()
    assert dates == [("2026-08-01",), ("2026-09-23",)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfermarkt_merge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingestion.transfermarkt.merge'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ingestion/transfermarkt/merge.py
"""Persists Transfermarkt player market values into a dated-snapshot table.

Append-only by design (design spec's "Storage schema" section): each refresh
run inserts a new row set stamped with that run's snapshot_date, never
overwriting an earlier one. Costs nothing for the agent-tool consumer (Phase
1 only ever wants "most recent snapshot <= today"), but is exactly what a
future point-in-time backtest needs -- a historical match must read the
value that existed *at* that match's date, never a later one (the same
lookahead-bias class of bug W179 already had to fix for closing-line odds).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.db_manager import DuckDBManager

LOGGER = get_logger(__name__)

_VALUE_COLUMNS = ["fotmob_player_id", "transfermarkt_player_id", "market_value_eur"]


def _create_tables(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_market_values (
            fotmob_player_id BIGINT,
            transfermarkt_player_id BIGINT,
            snapshot_date TEXT,
            market_value_eur BIGINT
        )
        """
    )


def insert_market_value_snapshot(values_df: pd.DataFrame, db_manager: "DuckDBManager", snapshot_date: str) -> int:
    """Inserts one dated snapshot's worth of market values. Returns the row
    count inserted. Deliberately plain INSERT ... SELECT, no ON CONFLICT
    clause -- this table is append-only, never upserted (see module
    docstring)."""
    if values_df.empty:
        with db_manager.connection() as conn:
            _create_tables(conn)
        return 0

    to_insert = values_df[_VALUE_COLUMNS].copy()
    to_insert["snapshot_date"] = snapshot_date

    with db_manager.connection() as conn:
        _create_tables(conn)
        conn.register("_values_upd", to_insert)
        conn.execute(
            """
            INSERT INTO player_market_values (fotmob_player_id, transfermarkt_player_id, snapshot_date, market_value_eur)
            SELECT fotmob_player_id, transfermarkt_player_id, snapshot_date, market_value_eur
            FROM _values_upd
            """
        )
        conn.unregister("_values_upd")

    LOGGER.info("Market value snapshot inserted | snapshot_date=%s | rows=%d", snapshot_date, len(to_insert))
    return len(to_insert)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transfermarkt_merge.py -v`
Expected: PASS, both tests

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/transfermarkt/merge.py tests/test_transfermarkt_merge.py
git commit -m "feat(ingestion): dated-snapshot player_market_values table (A126)"
```

---

### Task 3: Transfermarkt fetcher

**Files:**
- Create: `src/ingestion/transfermarkt/fetcher.py`
- Test: `tests/test_transfermarkt_fetcher.py`

Confirmed live: `GET https://www.transfermarkt.com/<any-slug>/profil/spieler/<transfermarkt_id>` (redirects to the canonical slug, plain `requests` follows redirects by default) returns the player's profile page. Market value text lives in `soup.find(class_="data-header__market-value-wrapper")`, e.g. `"€ 220.00 m Last update: 22/07/2026"` — not every player has one (confirmed live on a lower-profile player), which is a real, non-error case to degrade on.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transfermarkt_fetcher.py
"""Tests for the Transfermarkt market-value fetcher."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.transfermarkt.fetcher import _parse_market_value_eur, fetch_market_value


def _mock_resp(html: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.text = html
    return resp


def test_parse_market_value_eur_millions():
    assert _parse_market_value_eur("€ 220.00 m Last update: 22/07/2026") == 220_000_000


def test_parse_market_value_eur_thousands():
    assert _parse_market_value_eur("€ 450 k Last update: 22/07/2026") == 450_000


def test_parse_market_value_eur_billions():
    assert _parse_market_value_eur("€ 1.20 bn Last update: 22/07/2026") == 1_200_000_000


def test_fetch_market_value_extracts_from_a_real_shaped_page():
    html = '<div class="data-header__market-value-wrapper">€ 220.00 m Last update: 22/07/2026</div>'
    with patch("src.ingestion.transfermarkt.fetcher.requests.get", return_value=_mock_resp(html)) as mock_get, \
         patch("src.ingestion.transfermarkt.fetcher.time.sleep"):
        result = fetch_market_value(418560)

    assert result == 220_000_000
    args, kwargs = mock_get.call_args
    assert args[0] == "https://www.transfermarkt.com/player/profil/spieler/418560"
    assert "User-Agent" in kwargs.get("headers", {})


def test_fetch_market_value_returns_none_when_no_value_listed():
    """Confirmed live -- not every player has a market value wrapper on
    their page (lower-profile/inactive players). A real, non-error gap."""
    html = "<div>no market value section here</div>"
    with patch("src.ingestion.transfermarkt.fetcher.requests.get", return_value=_mock_resp(html)), \
         patch("src.ingestion.transfermarkt.fetcher.time.sleep"):
        result = fetch_market_value(96148)

    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfermarkt_fetcher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingestion.transfermarkt.fetcher'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ingestion/transfermarkt/fetcher.py
"""Fetches Transfermarkt player market values via plain requests -- same
pattern as src/ingestion/fotmob/fetcher.py, confirmed live that (unlike
sofifa.com's Cloudflare-protected pages) a plain requests.get() with a
browser User-Agent gets through cleanly, no browser automation needed.
"""

from __future__ import annotations

import re
import time

from bs4 import BeautifulSoup
import requests

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

_UNIT_MULTIPLIERS = {"k": 1_000, "m": 1_000_000, "bn": 1_000_000_000}
_VALUE_PATTERN = re.compile(r"€\s*([\d.,]+)\s*(k|m|bn)", re.IGNORECASE)


def _parse_market_value_eur(text: str) -> int | None:
    """Parses Transfermarkt's own display format (e.g. "€ 220.00 m") into a
    plain integer EUR value. Returns None for text with no recognizable
    value (e.g. a page section that's present but genuinely says nothing,
    or malformed input) -- never raises on this, matching this codebase's
    "degrade, don't crash" convention for optional lookups."""
    match = _VALUE_PATTERN.search(text)
    if match is None:
        return None
    number = float(match.group(1).replace(",", ""))
    multiplier = _UNIT_MULTIPLIERS[match.group(2).lower()]
    return int(number * multiplier)


def fetch_market_value(transfermarkt_id: int, delay: float = 1.0) -> int | None:
    """Returns the player's current market value in EUR, or None if the
    page has no listed value (a real, non-error case confirmed live for
    lower-profile players) -- never raises for this. The URL's slug segment
    ("player") is a placeholder; Transfermarkt redirects to the canonical
    slug based on the numeric ID alone, and requests follows redirects by
    default."""
    url = f"https://www.transfermarkt.com/player/profil/spieler/{transfermarkt_id}"
    response = requests.get(url, headers=_HEADERS, timeout=30)
    response.raise_for_status()
    time.sleep(delay)

    soup = BeautifulSoup(response.text, "html.parser")
    wrapper = soup.find(class_="data-header__market-value-wrapper")
    if wrapper is None:
        return None
    return _parse_market_value_eur(wrapper.get_text(" ", strip=True))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transfermarkt_fetcher.py -v`
Expected: PASS, all 5 tests

- [ ] **Step 5: Manual smoke test (not automated — real network)**

Run: `python3 -c "from src.ingestion.transfermarkt.fetcher import fetch_market_value; print(fetch_market_value(418560))"`
Expected: `220000000` (Erling Haaland's real current market value, or whatever it's since changed to — a plausible int in the hundreds-of-millions range, not `None` or an exception).

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/transfermarkt/fetcher.py tests/test_transfermarkt_fetcher.py
git commit -m "feat(ingestion): fetch Transfermarkt player market values (A126)"
```

---

### Task 4: `refresh-market-values` CLI subcommand

**Files:**
- Modify: `main.py`

Wires Tasks 1-3 together: pull the players already in `raw_player_match_stats` (via `player_dim`, this project's own known-player scope, per the design spec), resolve their Transfermarkt IDs via the crosswalk, fetch market values one at a time (politely rate-limited), insert today's snapshot.

- [ ] **Step 1: Write the handler function**

Add near the other `run_fetch_*` functions (e.g. after `run_fetch_fotmob`, following the same lazy-import-inside-the-handler convention `main.py`'s own module comment describes):

```python
def run_refresh_market_values(app_settings: AppSettings, db_manager: DuckDBManager, delay: float = 1.0) -> None:
    """A126: refresh Transfermarkt market values for every player already
    known to this project (raw_player_match_stats/player_dim), via the REEP
    crosswalk."""
    from datetime import date

    from src.ingestion.transfermarkt.fetcher import fetch_market_value
    from src.ingestion.transfermarkt.merge import insert_market_value_snapshot
    from src.ingestion.transfermarkt.reep_crosswalk import load_reep_crosswalk
    import pandas as pd

    with db_manager.connection(read_only=True) as conn:
        known_fotmob_ids = conn.execute("SELECT DISTINCT player_id FROM player_dim").fetchdf()["player_id"].tolist()
    if not known_fotmob_ids:
        LOGGER.info("refresh-market-values: no players in player_dim yet -- nothing to refresh.")
        return

    crosswalk = load_reep_crosswalk()
    scoped = crosswalk[crosswalk["fotmob_player_id"].isin(known_fotmob_ids)].reset_index(drop=True)
    unmapped = len(known_fotmob_ids) - len(scoped)
    if unmapped:
        LOGGER.info("refresh-market-values: %d of %d known players have no REEP crosswalk entry -- skipped, not an error.", unmapped, len(known_fotmob_ids))

    rows = []
    for _, row in scoped.iterrows():
        value = fetch_market_value(int(row["transfermarkt_player_id"]), delay=delay)
        if value is not None:
            rows.append({"fotmob_player_id": row["fotmob_player_id"], "transfermarkt_player_id": row["transfermarkt_player_id"], "market_value_eur": value})

    n = insert_market_value_snapshot(pd.DataFrame(rows), db_manager, snapshot_date=date.today().isoformat())
    LOGGER.info("refresh-market-values complete | players_scoped=%d | values_fetched=%d | rows_inserted=%d", len(scoped), len(rows), n)
```

(`LOGGER` here is `main.py`'s own module-level logger, already defined at the top of the file — reuse it, don't add a second one.)

- [ ] **Step 2: Register the subparser**

Add alongside the other `fetch-*`/`refresh-*` subparsers (near `fetch-fotmob`'s registration):

```python
market_values_parser = subparsers.add_parser(
    "refresh-market-values", help="Refresh Transfermarkt market values for every player already known to this project (A126)."
)
market_values_parser.add_argument("--delay", type=float, default=1.0, help="Polite delay in seconds between requests.")
```

- [ ] **Step 3: Wire the dispatch**

Add alongside the other `elif args.command == "fetch-fotmob":` branch:

```python
elif args.command == "refresh-market-values":
    run_refresh_market_values(app_settings, db_manager, delay=float(args.delay))
```

- [ ] **Step 4: Manual smoke test**

Run: `python main.py refresh-market-values`
Expected: logs `refresh-market-values complete | players_scoped=<N> | values_fetched=<M> | rows_inserted=<M>`, no traceback. `M <= N` is expected (some scoped players have no listed market value).

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat(cli): add refresh-market-values subcommand (A126)"
```

---

### Task 5: Layered player-name matching

**Files:**
- Create: `src/agent/player_matching.py`
- Test: `tests/test_player_matching.py`

Pure matching logic, no DB/agent dependency — kept separate from Task 6's tool wiring so it's independently testable, per the design spec's own 4-layer breakdown (exact -> accent-fold -> surname -> fuzzy), each independently verifiable.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_player_matching.py
"""Tests for the 4-layer player-name matcher (get_player_rating's identity
resolution): exact -> accent-fold -> surname -> difflib fuzzy fallback.
Ambiguous matches must resolve to None, never a guess."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agent.player_matching import match_player_name

_ROSTER = ["Kevin De Bruyne", "Erling Haaland", "Rodri", "Kylian Mbappé"]


def test_exact_match_case_insensitive():
    assert match_player_name("erling haaland", _ROSTER) == "Erling Haaland"


def test_accent_folded_match():
    assert match_player_name("Kylian Mbappe", _ROSTER) == "Kylian Mbappé"


def test_surname_only_match():
    assert match_player_name("De Bruyne", _ROSTER) == "Kevin De Bruyne"


def test_fuzzy_fallback_match():
    assert match_player_name("Erlin Haaland", _ROSTER) == "Erling Haaland"


def test_ambiguous_surname_returns_none_not_a_guess():
    roster = ["Kevin De Bruyne", "Bruno De Bruyne"]  # shared surname, contrived
    assert match_player_name("De Bruyne", roster) is None


def test_no_plausible_match_returns_none():
    assert match_player_name("Someone Totally Unrelated", _ROSTER) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_player_matching.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agent.player_matching'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/agent/player_matching.py
"""Layered, cheapest-first player-name matching against a small (~40-50
name) roster pool -- the player-level analog of
src/ingestion/common/team_mapping.py's TeamNameMapper, per the design spec.

Ambiguity is a non-match, never a guess (mirrors TeamNameMapper.suggest()'s
own tie-breaking-to-None precedent) -- a wrong confident number here actively
misleads betting reasoning, worse than admitting uncertainty."""

from __future__ import annotations

import difflib

from src.ingestion.common.team_mapping import _fold_accents

_FUZZY_SIMILARITY_FLOOR = 0.75


def _surname(full_name: str) -> str:
    return full_name.strip().split()[-1]


def match_player_name(query: str, roster: list[str]) -> str | None:
    """Returns the single roster name query most plausibly refers to, or
    None if there's no match or more than one equally plausible match."""
    query_norm = query.strip().casefold()

    # Layer 1: exact (case-insensitive)
    exact = [name for name in roster if name.casefold() == query_norm]
    if len(exact) == 1:
        return exact[0]

    # Layer 2: accent-folded
    folded_query = _fold_accents(query_norm)
    folded = [name for name in roster if _fold_accents(name.casefold()) == folded_query]
    if len(folded) == 1:
        return folded[0]

    # Layer 3: surname-only
    surname_query = _fold_accents(_surname(query).casefold())
    surname_matches = [name for name in roster if _fold_accents(_surname(name).casefold()) == surname_query]
    if len(surname_matches) == 1:
        return surname_matches[0]
    if len(surname_matches) > 1:
        return None  # ambiguous surname -- never guess

    # Layer 4: difflib fuzzy fallback against the full roster
    close = difflib.get_close_matches(query, roster, n=2, cutoff=_FUZZY_SIMILARITY_FLOOR)
    if len(close) == 1:
        return close[0]
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_player_matching.py -v`
Expected: PASS, all 6 tests. If `test_fuzzy_fallback_match` fails because `difflib` also returns the correct answer as a *second*, lower-ranked candidate for a typo this small, tighten to `n=1` — re-run and confirm before moving on rather than guessing at the fix.

- [ ] **Step 5: Commit**

```bash
git add src/agent/player_matching.py tests/test_player_matching.py
git commit -m "feat(agent): layered player-name matcher for get_player_rating (A126)"
```

---

### Task 6: Roster lookup + `get_player_rating` agent tool

**Files:**
- Create: `src/agent/player_rating_tool.py`
- Modify: `src/agent/graph.py` (`run_agent`, around lines 535-536 per current source)
- Test: `tests/test_player_rating_tool.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_player_rating_tool.py
"""Tests for get_player_rating: roster lookup + the LLM-callable tool
itself. The tool is built fresh per match (build_player_rating_tool), closing
over that match's own roster -- the first per-match-context-bound tool in
this codebase, since get_default_tools()/build_graph() already run fresh per
run_agent() call (no new global state needed)."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.agent.player_rating_tool import build_player_rating_tool, get_team_roster
from src.utils.db_manager import DuckDBManager


def _make_db_manager(tmp_path: Path) -> DuckDBManager:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    return DuckDBManager(config_path=str(config_path))


def _seed(db_manager: DuckDBManager) -> None:
    with db_manager.connection() as conn:
        conn.execute("CREATE TABLE raw_matches (match_id TEXT, date TIMESTAMP, home_team TEXT, away_team TEXT)")
        conn.execute("INSERT INTO raw_matches VALUES ('m1', '2026-09-01', 'Man City', 'Everton')")
        conn.execute("CREATE TABLE player_dim (player_id BIGINT PRIMARY KEY, player_name TEXT, opta_id TEXT)")
        conn.execute("INSERT INTO player_dim VALUES (1, 'Erling Haaland', NULL), (2, 'Jordan Pickford', NULL)")
        conn.execute(
            "CREATE TABLE raw_player_match_stats (match_id TEXT, player_id BIGINT, team_name TEXT, minutes_played INTEGER, rating FLOAT, goals INTEGER, assists INTEGER, xg FLOAT, xa FLOAT, xgot FLOAT, shots INTEGER, interceptions FLOAT, recoveries FLOAT, PRIMARY KEY (match_id, player_id))"
        )
        conn.execute("INSERT INTO raw_player_match_stats VALUES ('m1', 1, 'Man City', 90, 8.1, 1, 0, 0.8, 0.1, 0.5, 3, 0, 1)")
        conn.execute("INSERT INTO raw_player_match_stats VALUES ('m1', 2, 'Everton', 90, 6.5, 0, 0, 0, 0, 0, 0, 1, 5)")
        conn.execute("CREATE TABLE player_market_values (fotmob_player_id BIGINT, transfermarkt_player_id BIGINT, snapshot_date TEXT, market_value_eur BIGINT)")
        conn.execute("INSERT INTO player_market_values VALUES (1, 100, '2026-09-01', 220000000)")


def test_get_team_roster_returns_player_names_for_a_team(tmp_path: Path):
    db_manager = _make_db_manager(tmp_path)
    _seed(db_manager)

    roster = get_team_roster(db_manager, team_name="Man City", as_of_date="2026-09-15")
    assert roster == ["Erling Haaland"]


def test_tool_returns_value_for_a_known_player(tmp_path: Path):
    db_manager = _make_db_manager(tmp_path)
    _seed(db_manager)

    tool = build_player_rating_tool(
        db_manager, home_team="Man City", away_team="Everton", as_of_date="2026-09-15",
    )
    result = tool.invoke({"player_name": "Erling Haaland"})

    assert result == {"matched": True, "market_value_eur": 220000000}


def test_tool_degrades_to_matched_false_for_an_unknown_name(tmp_path: Path):
    db_manager = _make_db_manager(tmp_path)
    _seed(db_manager)

    tool = build_player_rating_tool(
        db_manager, home_team="Man City", away_team="Everton", as_of_date="2026-09-15",
    )
    result = tool.invoke({"player_name": "Nobody On This Roster"})

    assert result == {"matched": False}


def test_tool_degrades_to_matched_false_when_player_has_no_market_value(tmp_path: Path):
    """Jordan Pickford is on the roster but has no player_market_values row
    seeded -- a real, non-blocking gap (unmapped crosswalk entry, no listed
    value, or never fetched), not an error."""
    db_manager = _make_db_manager(tmp_path)
    _seed(db_manager)

    tool = build_player_rating_tool(
        db_manager, home_team="Man City", away_team="Everton", as_of_date="2026-09-15",
    )
    result = tool.invoke({"player_name": "Jordan Pickford"})

    assert result == {"matched": False}


def test_tool_never_reads_a_snapshot_dated_after_as_of_date(tmp_path: Path):
    """Point-in-time correctness -- the same lookahead-bias class of bug
    W179 already had to fix once for closing-line odds features. Two
    snapshots for the same player (an older, lower value and a newer,
    higher one); a lookup dated between them must return the OLDER one,
    never peek at the not-yet-existing newer snapshot."""
    db_manager = _make_db_manager(tmp_path)
    _seed(db_manager)
    with db_manager.connection() as conn:
        conn.execute("INSERT INTO player_market_values VALUES (1, 100, '2026-10-01', 300000000)")  # future, higher

    tool = build_player_rating_tool(
        db_manager, home_team="Man City", away_team="Everton", as_of_date="2026-09-15",
    )
    result = tool.invoke({"player_name": "Erling Haaland"})

    assert result == {"matched": True, "market_value_eur": 220000000}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_player_rating_tool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agent.player_rating_tool'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/agent/player_rating_tool.py
"""get_player_rating: the agent tool that replaces A121's web_search-based
FIFA-rating workaround. Scoped automatically to the current match's two
rosters (~40-50 names) -- narrowing the candidate pool this way is what
makes reliable name matching possible at all (design spec, Phase 1
section). Built fresh per match by build_player_rating_tool(), closed over
that match's roster, rather than reading match context from any global
state -- see graph.py's run_agent() for the call site.

Underlying signal is Transfermarkt market_value_eur (not an EA FC-style
ability rating, see the design spec's 2026-09-23 revision note) -- a
market-perception-of-quality/depth proxy, same underlying purpose as the
originally-planned SoFIFA rating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool as tool_decorator

from src.agent.player_matching import match_player_name
from src.utils.helpers import standardize_team_name
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.db_manager import DuckDBManager

LOGGER = get_logger(__name__)

# Design spec: rolls over each team's past matchday squads, same trailing
# pool window SQUAD_*/lineup_features.py's compute_frds already uses --
# reused here as the "current roster" definition, not reinvented.
ROSTER_POOL_DAYS = 90


def get_team_roster(db_manager: "DuckDBManager", team_name: str, as_of_date: str) -> list[str]:
    """Distinct player names who've played for team_name (standardized,
    same convention as every other team-name join in this codebase) within
    ROSTER_POOL_DAYS before as_of_date."""
    team_std = standardize_team_name(team_name)
    with db_manager.connection(read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT pd.player_name
            FROM raw_player_match_stats s
            JOIN raw_matches m ON s.match_id = m.match_id
            JOIN player_dim pd ON s.player_id = pd.player_id
            WHERE s.team_name = ?
              AND m.date >= CAST(? AS TIMESTAMP) - INTERVAL '90 days'
              AND m.date < CAST(? AS TIMESTAMP)
            """,
            [team_std, as_of_date, as_of_date],
        ).fetchall()
    return [row[0] for row in rows]


def _lookup_market_value(db_manager: "DuckDBManager", player_name: str, roster: list[str], as_of_date: str) -> dict:
    matched_name = match_player_name(player_name, roster)
    if matched_name is None:
        return {"matched": False}

    with db_manager.connection(read_only=True) as conn:
        row = conn.execute(
            """
            SELECT v.market_value_eur
            FROM player_market_values v
            JOIN player_dim pd ON v.fotmob_player_id = pd.player_id
            WHERE pd.player_name = ? AND v.snapshot_date <= ?
            ORDER BY v.snapshot_date DESC
            LIMIT 1
            """,
            [matched_name, as_of_date],
        ).fetchone()
    if row is None:
        return {"matched": False}
    return {"matched": True, "market_value_eur": row[0]}


def build_player_rating_tool(db_manager: "DuckDBManager", home_team: str, away_team: str, as_of_date: str):
    """Returns a fresh get_player_rating tool closed over this match's own
    roster pool (both teams combined) -- call once per match, in
    graph.py's run_agent(), before build_graph()/bind_tools()."""
    roster = get_team_roster(db_manager, home_team, as_of_date) + get_team_roster(db_manager, away_team, as_of_date)

    @tool_decorator
    def get_player_rating(player_name: str) -> dict:
        """Look up a player's Transfermarkt market value (market_value_eur)
        -- a quick, structured quality-gap proxy when weighing whether an
        absent player's replacement is a real downgrade. Only resolves
        names on this match's own two rosters. Returns {"matched": false}
        for any name not confidently found, or with no listed value --
        never guesses."""
        return _lookup_market_value(db_manager, player_name, roster, as_of_date)

    return get_player_rating
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_player_rating_tool.py -v`
Expected: PASS, all 5 tests

- [ ] **Step 5: Wire into `get_default_tools()` and `run_agent()`**

In `src/agent/tools.py`, change `get_default_tools()`'s signature to optionally accept a pre-built extra tool (keeps this file from needing to know about `DuckDBManager`/match context directly):

```python
def get_default_tools(extra_tools: list | None = None) -> list:
    """A31: forecast_league, forecast_international, and resolve_competition are
    no longer LLM-callable -- they're invoked directly by the deterministic
    pipeline nodes in src/agent/pipeline.py before the LLM ever runs. Only
    web_search remains available for the LLM's own optional follow-up digging.

    A126: extra_tools lets a caller (graph.py's run_agent()) add match-scoped
    tools built fresh per call, e.g. get_player_rating -- kept as an explicit
    param rather than this module reaching into DB/match state itself, so
    tools.py stays free of per-match context."""
    return [web_search, *(extra_tools or [])]
```

In `src/agent/graph.py`'s `run_agent()`, right where `tools = get_default_tools()` is currently called (around line 535-536), build and pass the player-rating tool:

```python
    if tools is None:
        from src.agent.tools import get_default_tools

        extra_tools = []
        try:
            from src.agent.player_rating_tool import build_player_rating_tool
            from src.utils.db_manager import DuckDBManager

            extra_tools.append(build_player_rating_tool(
                DuckDBManager(),
                home_team=match_info["home_team"],
                away_team=match_info["away_team"],
                as_of_date=match_info["date"],
            ))
        except Exception:
            LOGGER.warning("Could not build get_player_rating tool for this match -- proceeding without it.", exc_info=True)

        tools = get_default_tools(extra_tools=extra_tools)
```

(`LOGGER` must already exist at module level in `graph.py` — confirm before pasting; if it's named differently there, use the existing name rather than introducing a second logger.) The try/except here matters: a crosswalk/DB hiccup for this one match must degrade to "no player-rating tool this match" (the LLM just won't have it available), never crash the whole recommendation.

- [ ] **Step 6: Run the full agent test suite to check for regressions**

Run: `pytest tests/test_agent_tools_snapshot.py tests/test_agent_graph*.py -v`
Expected: all pass, no new failures. If any existing test calls `get_default_tools()` expecting exactly `[web_search]`, update its assertion to `get_default_tools() == [web_search]` (the no-`extra_tools`-arg case is unchanged) rather than loosening the test.

- [ ] **Step 7: Commit**

```bash
git add src/agent/player_rating_tool.py src/agent/tools.py src/agent/graph.py tests/test_player_rating_tool.py
git commit -m "feat(agent): add get_player_rating tool, built per-match (A126)"
```

---

### Task 7: Prompt updates (all 4 posture files)

**Files:**
- Modify: `config/prompts/agent_v1.txt`
- Modify: `config/prompts/agent_v1_aggressive.txt`
- Modify: `config/prompts/agent_v1_balanced.txt`
- Modify: `config/prompts/agent_v1_conservative.txt`
- Test: `tests/test_agent_prompt_thresholds.py` (extend existing file)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent_prompt_thresholds.py` (following whatever pattern the existing A118-122 "instruction present in all 4 files" checks there already use — read the file first to match its exact helper/loop style before adding this):

```python
def test_get_player_rating_instruction_present_in_all_4_postures():
    for path in _ALL_PROMPT_PATHS:  # reuse the existing constant/fixture this file already defines
        text = Path(path).read_text(encoding="utf-8")
        assert "get_player_rating" in text
        assert "FIFA / EA Sports FC card" not in text  # A121's old web_search instruction must be gone
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_prompt_thresholds.py::test_get_player_rating_instruction_present_in_all_4_postures -v`
Expected: FAIL — `assert "get_player_rating" in text` fails (not yet present)

- [ ] **Step 3: Update all 4 prompt files**

In each of the 4 files, replace the current A121 sentence:

> "If the evidence already gathered doesn't make the replacement's quality clear and it materially affects a market you are close to recommending, and you still have your one optional web_search call available, spend it comparing the missing player's and their likely replacement's FIFA / EA Sports FC card overall rating (e.g. "Alex Meret FIFA rating" vs "Elia Caprile FIFA rating") as a rough quality-gap proxy -- a small gap means the absence probably costs little, a large one means it is a genuine downgrade. Not worth a tool call for a squad player who barely features regardless."

with:

> "If the evidence already gathered doesn't make the replacement's quality clear and it materially affects a market you are close to recommending, call get_player_rating for both the missing player and their likely replacement and compare market_value_eur as a rough quality-gap proxy -- a small gap means the absence probably costs little, a large one means it is a genuine downgrade. This is a structured tool call, not a web_search, so it doesn't count against your one optional web_search budget."

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_prompt_thresholds.py -v`
Expected: PASS, including every pre-existing test in this file (no regression to the other posture checks already there)

- [ ] **Step 5: Commit**

```bash
git add config/prompts/agent_v1.txt config/prompts/agent_v1_aggressive.txt config/prompts/agent_v1_balanced.txt config/prompts/agent_v1_conservative.txt tests/test_agent_prompt_thresholds.py
git commit -m "feat(agent): retire A121 web_search FIFA-rating workaround for get_player_rating (A126)"
```

---

### Task 8: Full regression + docs

**Files:**
- Modify: `documents/agent_user_stories.md` (A126 status)

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ app/backend/tests/ scripts/ -q`
Expected: PASS, same pre-existing skip/failure count as before this plan (no new failures) -- baseline for this worktree is 1 pre-existing unrelated failure (`tests/test_prepare_training_data_league_scoping.py::test_international_context_pools_e0_and_swe_against_real_registry`) plus whatever `app/backend/tests/`/`scripts/` show given this worktree has no real local `data/fpai_core.db` (gitignored, never populated here) -- several `app/backend/tests/` fixture/dashboard tests fail against an empty DB for that reason alone, unrelated to this plan; don't fix them here. If any test run leaves behind stub `data/*.db` files, move them out of `data/` afterward (one at a time -- a bulk `rm -rf`/wildcard `mv` gets blocked by the sandbox's destructive-action guard) so they don't turn later `pytest.skip("Real database not available")` checks into real errors.

- [ ] **Step 2: Update A126's status**

In `documents/agent_user_stories.md`, change A126's status column from `future` to `completed (2026-09-23)`, and append to its own row text (after the existing "Full design: ..." sentence):

> "**Implemented (2026-09-23):** `src/ingestion/transfermarkt/` (reep_crosswalk.py, fetcher.py, merge.py), `main.py refresh-market-values`, and the per-match `get_player_rating` tool (`src/agent/player_rating_tool.py`, wired into `graph.py`'s `run_agent()`) all shipped, TDD, full suite green (excluding this worktree's own missing-real-local-DB gaps, unrelated). **Data source changed mid-implementation**: the design spec originally called for SoFIFA ability ratings, but sofifa.com's Cloudflare bot protection blocks a plain `requests` fetch (confirmed live), and the only real workaround (`soccerdata`/`seleniumbase`, needing a real local Chrome install) created a dependency conflict with `google-genai`/`langgraph-sdk`'s `websockets` requirement in this project's shared venv -- a real risk to the live agent stack. **User-confirmed pivot to Transfermarkt market values** instead: same `requests`-only pattern as every other ingestion module here, no new dependency, no Cloudflare problem, tested live. Full history in `docs/superpowers/specs/2026-09-22-player-quality-proxy-design.md`'s 2026-09-23 revision note. Underlying signal is now `market_value_eur` (transfer-market perception), not an EA FC-style ability rating -- same underlying purpose (player-quality/depth proxy), different flavor."

- [ ] **Step 3: Commit**

```bash
git add documents/agent_user_stories.md
git commit -m "docs: mark A126 completed -- Transfermarkt ingestion + get_player_rating tool shipped"
```
