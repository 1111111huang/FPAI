# Phase 14b Part 1: Ingestion Restructuring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `src/ingestion/` from flat per-source files into per-source subpackages (`football_data/`, `understat/`, `common/`), and namespace `data/raw/` to match, with zero behavior change — this is a pure refactor that prepares the ground for the new `fotmob/` package (Plan 2, US#92–95). Corresponds to US#91 in `documents/user_stories.md`.

**Architecture:** Source-agnostic helpers (`TeamNameMapper`, `LEAGUE_TIER_MAP`) move into `src/ingestion/common/`. Football-data.co.uk-specific files (`scraper.py`, `data_loader.py`, `match_schema.py`) move into `src/ingestion/football_data/`. Understat-specific files (`understat.py`, `understat_fetcher.py`) move into `src/ingestion/understat/`. `src/ingestion/__init__.py` re-exports stay the same names, just pointed at new internal paths, so `main.py`'s only package-level import (`from src.ingestion import CSVLoader, FootballDataScraper`) needs no change. `data/raw/*.csv` moves to `data/raw/football_data/`.

**Tech Stack:** Python, pytest, git.

---

### Task 1: Extract source-agnostic helpers into `src/ingestion/common/`

**Files:**
- Create: `src/ingestion/common/__init__.py`
- Create: `src/ingestion/common/league_tiers.py`
- Create: `src/ingestion/common/team_mapping.py`

- [ ] **Step 1: Create the common package**

```bash
mkdir -p src/ingestion/common
touch src/ingestion/common/__init__.py
```

- [ ] **Step 2: Write `league_tiers.py`** (moved verbatim from `src/ingestion/schema.py`, minus the `MatchSchema` lazy re-export which is football-data-specific)

```python
"""League tier mapping shared across ingestion sources."""

from __future__ import annotations

LEAGUE_TIER_MAP: dict[str, int] = {
    "E0": 1,  # Premier League
    "E1": 2,  # Championship
    "E2": 3,  # League One
}


def map_league_code_to_tier(league_code: str) -> int:
    """Map league code to its tier integer."""
    code = str(league_code).strip().upper()
    return LEAGUE_TIER_MAP.get(code, 4)
```

- [ ] **Step 3: Write `team_mapping.py`** (moved verbatim from `src/ingestion/understat.py` lines 20-135: `_levenshtein_distance`, `_similarity_score`, `TeamNameMapper`)

```python
"""Fuzzy team-name resolution shared across ingestion sources.

Maps a source's team-name spelling (Understat, FotMob, etc.) onto the
canonical names already used in raw_matches, via an explicit JSON mapping
file with a Levenshtein-distance fallback for names not yet mapped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.utils.helpers import standardize_team_name
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _levenshtein_distance(left: str, right: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    left = left.lower()
    right = right.lower()
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, lchar in enumerate(left, start=1):
        current = [i]
        for j, rchar in enumerate(right, start=1):
            insert_cost = previous[j] + 1
            delete_cost = current[j - 1] + 1
            replace_cost = previous[j - 1] + (lchar != rchar)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _similarity_score(left: str, right: str) -> float:
    """Convert Levenshtein distance into a 0-1 similarity score."""
    if not left and not right:
        return 1.0
    max_len = max(len(left), len(right))
    if max_len == 0:
        return 1.0
    distance = _levenshtein_distance(left, right)
    return 1.0 - (distance / max_len)


class TeamNameMapper:
    """Map a source's team names to the CSV canonical names."""

    def __init__(self, mapping_path: str = "config/team_mapping.json", min_similarity: float = 0.82) -> None:
        self.mapping_path = Path(mapping_path)
        self.min_similarity = min_similarity
        self.mapping = self._load_mapping()

    def _load_mapping(self) -> dict[str, str]:
        if not self.mapping_path.exists():
            LOGGER.warning("Team mapping file not found: %s", self.mapping_path)
            return {}
        try:
            with self.mapping_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            LOGGER.warning("Invalid JSON in team mapping file: %s", self.mapping_path)
            return {}
        if not isinstance(payload, dict):
            LOGGER.warning("Team mapping file must be a JSON object: %s", self.mapping_path)
            return {}
        return {str(key): str(value) for key, value in payload.items()}

    def map_team(self, team_name: str, candidates: Iterable[str] | None = None) -> str:
        """Map a team name using explicit mappings or a fuzzy fallback."""
        normalized = " ".join(str(team_name).strip().split())
        if not normalized:
            return normalized
        if normalized in self.mapping:
            return self.mapping[normalized]

        if candidates is None:
            LOGGER.warning(
                "Unmapped team '%s'. Add mapping to %s.",
                normalized,
                self.mapping_path,
            )
            return normalized

        suggestion, score = self.suggest(normalized, candidates)
        if suggestion is None:
            LOGGER.warning(
                "Unmapped team '%s'. Add mapping to %s.",
                normalized,
                self.mapping_path,
            )
            return normalized

        if score >= self.min_similarity:
            LOGGER.warning(
                "Unmapped team '%s'. Using fuzzy match '%s' (score=%.2f). "
                "Add mapping to %s.",
                normalized,
                suggestion,
                score,
                self.mapping_path,
            )
            return suggestion

        LOGGER.warning(
            "Unmapped team '%s'. Closest match '%s' (score=%.2f). "
            "Add mapping to %s.",
            normalized,
            suggestion,
            score,
            self.mapping_path,
        )
        return normalized

    def suggest(self, team_name: str, candidates: Iterable[str]) -> tuple[str | None, float]:
        """Suggest the closest mapping candidate for a new team name."""
        best_name: str | None = None
        best_score = -1.0
        for candidate in candidates:
            candidate_name = standardize_team_name(str(candidate))
            score = _similarity_score(team_name, candidate_name)
            if score > best_score:
                best_score = score
                best_name = candidate_name
        return best_name, best_score
```

- [ ] **Step 4: Verify both modules import cleanly**

Run: `python -c "from src.ingestion.common.team_mapping import TeamNameMapper; from src.ingestion.common.league_tiers import map_league_code_to_tier; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/common/
git commit -m "refactor: extract source-agnostic team_mapping and league_tiers helpers (US#91)"
```

---

### Task 2: Move football-data.co.uk files into `src/ingestion/football_data/`

**Files:**
- Move: `src/ingestion/scraper.py` → `src/ingestion/football_data/scraper.py`
- Move: `src/ingestion/data_loader.py` → `src/ingestion/football_data/loader.py`
- Move: `src/ingestion/match_schema.py` → `src/ingestion/football_data/match_schema.py`
- Delete: `src/ingestion/schema.py` (superseded by `src/ingestion/common/league_tiers.py`; its only importer is updated in this task)

- [ ] **Step 1: Create the subpackage and move the files**

```bash
mkdir -p src/ingestion/football_data
touch src/ingestion/football_data/__init__.py
git mv src/ingestion/scraper.py src/ingestion/football_data/scraper.py
git mv src/ingestion/data_loader.py src/ingestion/football_data/loader.py
git mv src/ingestion/match_schema.py src/ingestion/football_data/match_schema.py
git rm src/ingestion/schema.py
```

- [ ] **Step 2: Fix `match_schema.py`'s import**

In `src/ingestion/football_data/match_schema.py`, change:

```python
from src.ingestion.schema import map_league_code_to_tier
```

to:

```python
from src.ingestion.common.league_tiers import map_league_code_to_tier
```

- [ ] **Step 3: Fix `loader.py`'s import of `MatchSchema`**

In `src/ingestion/football_data/loader.py`, change:

```python
from src.ingestion.match_schema import MatchSchema
```

to:

```python
from src.ingestion.football_data.match_schema import MatchSchema
```

- [ ] **Step 4: Namespace the raw-data directory by source**

In `src/ingestion/football_data/scraper.py`, in `FootballDataScraper.__init__`, change:

```python
self.raw_data_dir = Path(settings.paths.raw_data_dir)
```

to:

```python
self.raw_data_dir = Path(settings.paths.raw_data_dir) / "football_data"
```

In `src/ingestion/football_data/loader.py`, in `CSVLoader.__init__`, change:

```python
self.raw_data_dir = Path(self.db_manager.settings.paths.raw_data_dir)
```

to:

```python
self.raw_data_dir = Path(self.db_manager.settings.paths.raw_data_dir) / "football_data"
```

- [ ] **Step 5: Move the existing CSV files on disk** (these are gitignored, not tracked by git, so use plain `mv`)

```bash
mkdir -p data/raw/football_data
mv data/raw/*.csv data/raw/football_data/
ls data/raw/football_data/ | wc -l
```

Expected: same file count as `ls data/raw/*.csv | wc -l` showed before the move (10 files, per the current repo state).

- [ ] **Step 6: Update `.gitignore`**

In `.gitignore`, change:

```
data/raw/*.csv
```

to:

```
data/raw/football_data/*.csv
```

- [ ] **Step 7: Verify the moved package imports cleanly**

Run: `python -c "from src.ingestion.football_data.loader import CSVLoader; from src.ingestion.football_data.scraper import FootballDataScraper; from src.ingestion.football_data.match_schema import MatchSchema; print('ok')"`
Expected: `ok`

- [ ] **Step 8: Commit**

```bash
git add -A src/ingestion/football_data .gitignore
git commit -m "refactor: move football-data.co.uk ingestion into src/ingestion/football_data/ (US#91)"
```

---

### Task 3: Move Understat files into `src/ingestion/understat/`

**Files:**
- Move: `src/ingestion/understat.py` → `src/ingestion/understat/merge.py`
- Move: `src/ingestion/understat_fetcher.py` → `src/ingestion/understat/fetcher.py`

- [ ] **Step 1: Create the subpackage and move the files**

```bash
mkdir -p src/ingestion/understat
git mv src/ingestion/understat.py src/ingestion/understat/merge.py
git mv src/ingestion/understat_fetcher.py src/ingestion/understat/fetcher.py
touch src/ingestion/understat/__init__.py
git add src/ingestion/understat/__init__.py
```

- [ ] **Step 2: Remove the now-duplicated `TeamNameMapper` code from `merge.py`**

In `src/ingestion/understat/merge.py`, delete lines 20-135 (the `_levenshtein_distance` function, `_similarity_score` function, and `TeamNameMapper` class — this code now lives in `src/ingestion/common/team_mapping.py` from Task 1).

Change the top of the file from:

```python
"""Helpers for integrating Understat xG data with CSV match records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import pandas as pd

from src.utils.helpers import standardize_team_name
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.db_manager import DuckDBManager

LOGGER = get_logger(__name__)


def _levenshtein_distance(left: str, right: str) -> int:
    ...
    [through the end of the TeamNameMapper class]


def _resolve_column(df: pd.DataFrame, options: Iterable[str]) -> str | None:
```

to:

```python
"""Helpers for integrating Understat xG data with CSV match records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import pandas as pd

from src.ingestion.common.team_mapping import TeamNameMapper
from src.utils.helpers import standardize_team_name
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.db_manager import DuckDBManager

LOGGER = get_logger(__name__)


def _resolve_column(df: pd.DataFrame, options: Iterable[str]) -> str | None:
```

(`json` and `Path` imports are dropped since they were only used by the now-removed `TeamNameMapper._load_mapping`; `merge_understat_data` and `update_raw_matches_xg` keep working unchanged since `TeamNameMapper` is imported with the same name.)

- [ ] **Step 3: Verify the moved package imports cleanly**

Run: `python -c "from src.ingestion.understat.merge import TeamNameMapper, merge_understat_data, update_raw_matches_xg; from src.ingestion.understat.fetcher import fetch_league_season, fetch_seasons_range; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/ingestion/understat/
git commit -m "refactor: move Understat ingestion into src/ingestion/understat/ (US#91)"
```

---

### Task 4: Update `src/ingestion/__init__.py` and downstream import sites

**Files:**
- Modify: `src/ingestion/__init__.py`
- Modify: `main.py:322-323`
- Modify: `tests/test_understat.py:15-16`
- Modify: `tests/test_ingestion.py:13`

- [ ] **Step 1: Rewrite `src/ingestion/__init__.py`**

Change:

```python
from .match_schema import MatchSchema
from .data_loader import CSVLoader
from .scraper import FootballDataScraper
from .understat import TeamNameMapper, merge_understat_data

__all__ = [
    "MatchSchema",
    "CSVLoader",
    "FootballDataScraper",
    "TeamNameMapper",
    "merge_understat_data",
]
```

to:

```python
from .football_data.match_schema import MatchSchema
from .football_data.loader import CSVLoader
from .football_data.scraper import FootballDataScraper
from .common.team_mapping import TeamNameMapper
from .understat.merge import merge_understat_data

__all__ = [
    "MatchSchema",
    "CSVLoader",
    "FootballDataScraper",
    "TeamNameMapper",
    "merge_understat_data",
]
```

- [ ] **Step 2: Update `main.py`'s two direct submodule imports**

In `main.py`, inside `run_fetch_understat` (around line 322-323), change:

```python
    from src.ingestion.understat import update_raw_matches_xg
    from src.ingestion.understat_fetcher import fetch_seasons_range
```

to:

```python
    from src.ingestion.understat.merge import update_raw_matches_xg
    from src.ingestion.understat.fetcher import fetch_seasons_range
```

(`main.py:21`'s `from src.ingestion import CSVLoader, FootballDataScraper` needs no change — it imports from the package re-export updated in Step 1.)

- [ ] **Step 3: Update `tests/test_understat.py`**

Change:

```python
from src.ingestion.understat import TeamNameMapper, update_raw_matches_xg
from src.ingestion.understat_fetcher import fetch_league_season
```

to:

```python
from src.ingestion.common.team_mapping import TeamNameMapper
from src.ingestion.understat.merge import update_raw_matches_xg
from src.ingestion.understat.fetcher import fetch_league_season
```

- [ ] **Step 4: Update `tests/test_ingestion.py`**

Change:

```python
from src.ingestion.data_loader import CSVLoader
```

to:

```python
from src.ingestion.football_data.loader import CSVLoader
```

- [ ] **Step 5: Run the full test suite**

Run: `pytest -q`
Expected: same pass count as before this refactor (no failures, no collection errors)

- [ ] **Step 6: Manual smoke test against real data**

Run: `python main.py ingest`
Expected: log line `Ingest complete | raw_matches=<N> | feature_store=<N>` with the same row counts as before the restructuring (CSVs now read from `data/raw/football_data/` via the updated `CSVLoader`)

- [ ] **Step 7: Commit**

```bash
git add src/ingestion/__init__.py main.py tests/test_understat.py tests/test_ingestion.py
git commit -m "refactor: point ingestion re-exports and call sites at new subpackages (US#91)"
```

---

### Task 5: Mark US#91 complete in documentation

**Files:**
- Modify: `documents/user_stories.md`

- [ ] **Step 1: Mark the story complete**

In `documents/user_stories.md`, change:

```
- **US#91**: Restructure `src/ingestion/` into per-source subpackages (`football_data/`, `understat/`, `fbref/`, `common/`) and namespace `data/raw/` to match; update the 3 existing import sites.
```

to:

```
- **US#91** (complete): Restructured `src/ingestion/` into per-source subpackages (`football_data/`, `understat/`, `common/`) and namespaced `data/raw/` to match. (Note: the registry now uses `fotmob/` rather than `fbref/` for the player-data source — see US#92.)
```

- [ ] **Step 2: Commit**

```bash
git add documents/user_stories.md
git commit -m "docs: mark US#91 ingestion restructuring complete"
```
