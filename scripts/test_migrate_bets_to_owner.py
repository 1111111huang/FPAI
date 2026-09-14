from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.backend.bet_tracker import BetTracker
from app.backend.users import UserStore
from scripts.migrate_bets_to_owner import migrate_bets_to_owner


def test_migrate_assigns_every_null_user_id_bet_to_the_owner(tmp_path: Path):
    bets_db = tmp_path / "bets.db"
    users_db = tmp_path / "users.db"
    tracker = BetTracker(db_path=bets_db)
    tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.1, stake=10.0,
        source="manual", recommendation_snapshot=None,  # user_id defaults to None -- pre-migration state
    )

    migrate_bets_to_owner(owner_email="owner@gmail.com", bets_db_path=bets_db, users_db_path=users_db)

    owner = UserStore(db_path=users_db).get_by_email("owner@gmail.com")
    assert owner is not None
    migrated = tracker.list_bets(user_id=owner.id)
    assert len(migrated) == 1
    assert migrated[0].match_id == "m1"


def test_migrate_is_idempotent(tmp_path: Path):
    bets_db = tmp_path / "bets.db"
    users_db = tmp_path / "users.db"
    tracker = BetTracker(db_path=bets_db)
    tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.1, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    migrate_bets_to_owner(owner_email="owner@gmail.com", bets_db_path=bets_db, users_db_path=users_db)
    migrate_bets_to_owner(owner_email="owner@gmail.com", bets_db_path=bets_db, users_db_path=users_db)  # run twice
    owner = UserStore(db_path=users_db).get_by_email("owner@gmail.com")
    assert len(tracker.list_bets(user_id=owner.id)) == 1  # not duplicated
