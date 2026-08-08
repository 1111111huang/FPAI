"""Regression tests for US#131: prepare_training_data must scope by league.

Found while verifying Sweden's forecasts end-to-end: `feature_count`/
`feature_completeness` in a real Sweden forecast payload looked suspiciously
low for two well-established clubs. Tracing it back further found the real
bug one layer down, in training itself, not inference: `ModelManager.
prepare_training_data()`'s SQL query joined `raw_matches`/`feature_store`
with **no league/competition filter at all**. This was invisible while only
E0 existed (there was nothing else to accidentally include), but once
Sweden's rows also existed in the same shared tables, training with
`context=SWE` silently trained on **E0 data instead**: Sweden's registry-
gated 74-feature list still includes 9 `MKT_AH_*`/`MKT_LAMBDA_*`/
`MKT_IMPLIED_OVER25` features (ungated -- MKT_* is never split by
`resolve_feature_group_tag()`) that are permanently NaN for Sweden (no O/U-
2.5 or AH odds in its source) but populated for E0. The mandatory
non-null-features dropna for non-XGBoost models (`prepare_training_data`,
`required_non_null.extend(feature_columns)`) then silently dropped every
Sweden row and kept only E0's -- training an EPL model mislabeled
`context=SWE`, with no error, no warning, and plausible-looking metrics.
Reproduced directly against the real repo's real data before writing this
test (see documents/user_stories.md's US#131 completion notes).

These tests prove: (1) the bug was real against a synthetic two-competition
fixture matching this exact failure shape, and (2) the fix (filtering by the
requesting competition's own `league_code` from the registry) resolves it,
while a `general_purpose` competition (`league_code=None`, e.g.
"international") intentionally stays unfiltered -- pooling across every
competition is that tier's actual design (US#138), not a bug.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import sys

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.logic.competition_registry import CompetitionDefinition
from src.models.base_model import XGBoostRegressorModel
from src.models.model_manager import ModelManager

_REGISTRY = {
    "E1": CompetitionDefinition(
        competition_id="E1", tier="competition_specific", league_code="E1",
        enabled_feature_groups=("OFF_GOALS", "MKT"),
    ),
    "SW1": CompetitionDefinition(
        competition_id="SW1", tier="competition_specific", league_code="SW1",
        enabled_feature_groups=("OFF_GOALS", "MKT"),
    ),
    "international": CompetitionDefinition(
        competition_id="international", tier="general_purpose", league_code=None,
        enabled_feature_groups=("MKT",),
    ),
}


def _lookup(competition_id: str, registry_path=None) -> CompetitionDefinition:
    if competition_id not in _REGISTRY:
        raise ValueError(f"Unknown competition '{competition_id}'.")
    return _REGISTRY[competition_id]


def _write_config(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    model_dir = tmp_path / "models"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path), "model_dir": str(model_dir)}}),
        encoding="utf-8",
    )
    schema_path = tmp_path / "config" / "schema.yaml"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(
        yaml.safe_dump({"training_setup": {"selected_features": ["OFF_HOME_FTHG_R5", "MKT_AH_LINE"]}}),
        encoding="utf-8",
    )
    return config_path


def _seed_two_competitions(config_path: Path) -> None:
    """E1 has real MKT_AH_LINE values (like real E0); SW1's is permanently
    NaN (like real Sweden's football-data.co.uk source) -- the exact failure
    shape that let E1 rows silently masquerade as SW1's training data."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    db_path = config["paths"]["database_path"]
    with duckdb.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE raw_matches (
                match_id TEXT PRIMARY KEY, date TIMESTAMP, odds_h FLOAT,
                fthg INTEGER, ftag INTEGER, league TEXT
            )
            """
        )
        conn.execute(
            'CREATE TABLE feature_store (match_id TEXT PRIMARY KEY, "OFF_HOME_FTHG_R5" FLOAT, "MKT_AH_LINE" FLOAT)'
        )
        for i in range(10):
            match_id = f"e1_{i}"
            conn.execute(
                "INSERT INTO raw_matches VALUES (?, ?, ?, ?, ?, ?)",
                [match_id, f"2025-08-{10 + i:02d} 15:00:00", 1.9, 1, 1, "E1"],
            )
            conn.execute("INSERT INTO feature_store VALUES (?, ?, ?)", [match_id, 1.5, 0.0])
        for i in range(10):
            match_id = f"sw1_{i}"
            conn.execute(
                "INSERT INTO raw_matches VALUES (?, ?, ?, ?, ?, ?)",
                [match_id, f"2025-09-{10 + i:02d} 15:00:00", 2.1, 2, 0, "SW1"],
            )
            conn.execute("INSERT INTO feature_store VALUES (?, ?, ?)", [match_id, 2.0, None])


def _make_manager(config_path: Path, competition_id: str) -> ModelManager:
    return ModelManager(
        model=XGBoostRegressorModel(),  # XGBoost tolerates NaN features -- isolates the league-scoping fix
        config_path=str(config_path),
        target_config={"target": "home_goals"},
        feature_subset=["OFF_HOME_FTHG_R5", "MKT_AH_LINE"],
        competition_id=competition_id,
    )


def test_unfiltered_query_would_have_trained_sw1_on_e1_data(tmp_path: Path) -> None:
    """Reproduces the bug directly at the SQL level, independent of ModelManager,
    to prove it's real: the old unfiltered join + the mandatory non-null-features
    dropna for non-XGBoost models keeps only E1's rows even when requesting SW1."""
    config_path = _write_config(tmp_path)
    _seed_two_competitions(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    with duckdb.connect(config["paths"]["database_path"]) as conn:
        df = conn.execute(
            """
            SELECT r.match_id, r.league, f."OFF_HOME_FTHG_R5", f."MKT_AH_LINE"
            FROM raw_matches r
            INNER JOIN feature_store f ON r.match_id = f.match_id
            """
        ).fetchdf()
    survivors = df.dropna(subset=["OFF_HOME_FTHG_R5", "MKT_AH_LINE"])
    assert set(survivors["league"]) == {"E1"}
    assert len(survivors) == 10


def test_prepare_training_data_scopes_to_requested_competition(tmp_path: Path) -> None:
    """The fix: requesting context=SW1 must only ever train on SW1's own rows,
    never silently substitute E1's -- even though SW1's own rows would all be
    dropped by the non-null-features requirement if E1's weren't excluded first."""
    config_path = _write_config(tmp_path)
    _seed_two_competitions(config_path)

    with patch("src.logic.competition_registry.get_competition_definition", side_effect=_lookup):
        manager = _make_manager(config_path, "SW1")
        X_train, X_val, X_test, y_train, y_val, y_test, test_meta = manager.prepare_training_data()

    total_rows = len(X_train) + len(X_val) + len(X_test)
    assert total_rows == 10  # only SW1's 10 rows, never E1's


def test_prepare_training_data_never_mixes_in_the_other_competition(tmp_path: Path) -> None:
    """Direct check that no E1 row's data leaked into SW1's training set, by
    joining the actual rows used back to raw_matches.league."""
    config_path = _write_config(tmp_path)
    _seed_two_competitions(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    with patch("src.logic.competition_registry.get_competition_definition", side_effect=_lookup):
        manager = _make_manager(config_path, "SW1")
        _, _, _, _, _, _, test_meta = manager.prepare_training_data()

    with duckdb.connect(config["paths"]["database_path"], read_only=True) as conn:
        leagues = conn.execute(
            "SELECT DISTINCT league FROM raw_matches WHERE match_id IN "
            f"({','.join('?' * len(test_meta['match_id']))})",
            list(test_meta["match_id"]),
        ).fetchall()
    assert {row[0] for row in leagues} <= {"SW1"}


def test_prepare_training_data_stays_unfiltered_for_general_purpose(tmp_path: Path) -> None:
    """A general_purpose competition (league_code=None) must keep pooling
    across every competition -- that's the intended design (US#138), not
    something this fix should restrict."""
    config_path = _write_config(tmp_path)
    _seed_two_competitions(config_path)

    with patch("src.logic.competition_registry.get_competition_definition", side_effect=_lookup):
        manager = _make_manager(config_path, "international")
        X_train, X_val, X_test, y_train, y_val, y_test, test_meta = manager.prepare_training_data()

    total_rows = len(X_train) + len(X_val) + len(X_test)
    assert total_rows == 20  # both E1's and SW1's rows pooled together


def test_prepare_training_data_falls_back_gracefully_for_unregistered_competition(tmp_path: Path) -> None:
    """If the competition isn't in the registry at all (e.g. a stale/removed
    competition_id), the league filter is skipped with a warning rather than
    crashing training outright -- same defensive posture as the existing
    feature-group-gating fallback in _load_selected_features."""
    config_path = _write_config(tmp_path)
    _seed_two_competitions(config_path)

    with patch("src.logic.competition_registry.get_competition_definition", side_effect=_lookup):
        manager = _make_manager(config_path, "NOT_REGISTERED")
        X_train, X_val, X_test, y_train, y_val, y_test, test_meta = manager.prepare_training_data()

    total_rows = len(X_train) + len(X_val) + len(X_test)
    assert total_rows == 20  # unfiltered fallback, both competitions' rows


# ---------------------------------------------------------------------------
# US#138: real-registry pooling regression, against the actual repo's
# config/competitions.yaml and the real populated data/fpai_core.db in this
# worktree -- not a synthetic fixture. Proves the acceptance criterion this
# story was scoped for ("with SWE also registered, the training set visibly
# includes both leagues' rows") is genuinely satisfied by the US#131 league
# filter, since general_purpose competitions (league_code=None, e.g.
# "international") intentionally stay unfiltered by that fix -- no separate
# pooling mechanism needed to be built; US#138 turned out to already be done.
# ---------------------------------------------------------------------------


def test_international_context_pools_e0_and_swe_against_real_registry() -> None:
    """No mocking, no tmp_path fixture -- exercises the real repo's own
    config/competitions.yaml and this worktree's real, populated
    data/fpai_core.db directly."""
    from src.models.base_model import XGBoostModel

    manager = ModelManager(
        model=XGBoostModel(),
        target_config={"target": "result_3way"},
        competition_id="international",
    )
    X_train, X_val, X_test, y_train, y_val, y_test, test_meta = manager.prepare_training_data()
    total_rows = len(X_train) + len(X_val) + len(X_test)

    with manager.db_manager.connection(read_only=True) as conn:
        placeholders = ",".join("?" * len(test_meta["match_id"]))
        leagues = conn.execute(
            f"SELECT DISTINCT league FROM raw_matches WHERE match_id IN ({placeholders})",
            list(test_meta["match_id"]),
        ).fetchall()

    # E0, SWE, and SP1 rows all genuinely present -- real pooling, not just a
    # feature-list restriction. 11,089 = 3,800 (E0) + 3,489 (SWE) + 3,800 (SP1),
    # matching the real ingested row counts from US#124/125/143. Note this pool
    # is unfiltered raw_matches (international's league_code is None -> no
    # WHERE clause at all, per prepare_training_data()), not "every registered
    # competition" specifically -- SP1's rows are pooled here even before
    # US#147 registers SP1 in config/competitions.yaml, since registration and
    # raw ingestion are decoupled events.
    assert total_rows == 11089
    assert {row[0] for row in leagues} == {"E0", "SWE", "SP1"}
