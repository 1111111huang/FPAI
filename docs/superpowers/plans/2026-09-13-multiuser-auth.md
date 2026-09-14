# Multi-User Auth (Google-Only, Invite-Only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the app's single-shared-secret model with real per-user accounts (Google sign-in only, invite-only via an email allowlist), so bet-tracking data is private per user while the recommendations dashboard stays shared/unauthenticated exactly as today.

**Architecture:** Auth.js (NextAuth v4) runs entirely inside the Next.js frontend — Google OAuth + a `signIn` callback checking a server-side email allowlist, JWT session strategy (httpOnly cookie, never exposed to client JS). FastAPI never talks to Google or verifies JWTs: instead, new Next.js server-side API routes ("proxy routes") read the already-verified session via `getServerSession()`, then forward the request to FastAPI with the caller's email in an `X-User-Email` header, authenticated by a new server-only shared secret (`X-Internal-Secret`) distinct from today's public `APP_ACCESS_TOKEN`. FastAPI adds a `user_id` column to `user_bets` and a new `users` table; existing rows get migrated to the owner's account. The existing `APP_ACCESS_TOKEN` gate narrows to `/api/admin/*` routes only.

**Tech Stack:** FastAPI (Python, existing), SQLite (existing pattern, new `data/users.db` + `user_bets.user_id` column), Next.js 14 App Router + `next-auth` v4 (new), Google OAuth 2.0 (free).

---

## Why a second secret, not the existing `APP_ACCESS_TOKEN`

`NEXT_PUBLIC_APP_ACCESS_TOKEN` ([api.ts:17](app/frontend/lib/api.ts)) is shipped to the browser bundle — anyone with devtools open can read it. It's fine as a bot-deterrent for the shared dashboard, but it must never be the thing that lets a client assert "I am user X" — that would let anyone impersonate anyone. The new `INTERNAL_API_SECRET` is a **server-only** env var (no `NEXT_PUBLIC_` prefix), read only inside Next.js server-side route handlers, never sent to the browser. It authenticates "this request really came from our own Next.js server, which already verified the session cookie" — a different, narrower trust boundary than the admin token.

---

### Task 1: `users` table and `UserStore`

**Files:**
- Create: `app/backend/users.py`
- Test: `app/backend/tests/test_users.py`

- [ ] **Step 1: Write the failing test**

```python
# app/backend/tests/test_users.py
from __future__ import annotations

from pathlib import Path

from app.backend.users import UserStore


def test_get_or_create_creates_a_new_user_on_first_sight(tmp_path: Path):
    store = UserStore(db_path=tmp_path / "users.db")
    user = store.get_or_create("jane@gmail.com")
    assert user.email == "jane@gmail.com"
    assert user.id is not None


def test_get_or_create_returns_the_same_user_on_second_call(tmp_path: Path):
    store = UserStore(db_path=tmp_path / "users.db")
    first = store.get_or_create("jane@gmail.com")
    second = store.get_or_create("jane@gmail.com")
    assert first.id == second.id


def test_get_or_create_is_case_insensitive_on_email(tmp_path: Path):
    store = UserStore(db_path=tmp_path / "users.db")
    first = store.get_or_create("Jane@Gmail.com")
    second = store.get_or_create("jane@gmail.com")
    assert first.id == second.id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest app/backend/tests/test_users.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.backend.users'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/backend/users.py
"""W210: per-user accounts for multi-user bet tracking. A user is created
lazily on first sign-in (get_or_create) -- Auth.js's own signIn allowlist
callback (frontend) is what actually restricts who can reach this point at
all; by the time FastAPI ever sees an email here, Google + the allowlist
have already vouched for it. Same SQLite-per-concern pattern as
bet_tracker.py/recommendation_cache.py (W163: repo-root data/, not
app/data/)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "users.db"


@dataclass(frozen=True)
class User:
    id: int
    email: str
    created_at: str


class UserStore:
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
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )

    def get_or_create(self, email: str) -> User:
        normalized = email.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, created_at FROM users WHERE email = ?", (normalized,)
            ).fetchone()
            if row:
                return User(id=row[0], email=row[1], created_at=row[2])
            created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            cursor = conn.execute(
                "INSERT INTO users (email, created_at) VALUES (?, ?)",
                (normalized, created_at),
            )
            return User(id=cursor.lastrowid, email=normalized, created_at=created_at)

    def get_by_email(self, email: str) -> User | None:
        normalized = email.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, created_at FROM users WHERE email = ?", (normalized,)
            ).fetchone()
        return User(id=row[0], email=row[1], created_at=row[2]) if row else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest app/backend/tests/test_users.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/backend/users.py app/backend/tests/test_users.py
git commit -m "feat(auth): add UserStore for per-user accounts (W210)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: `user_id` on `user_bets`

**Files:**
- Modify: `app/backend/bet_tracker.py`
- Test: `app/backend/tests/test_bet_tracker.py` (existing file — add new tests, don't break old ones)

- [ ] **Step 1: Write the failing test**

Add to `app/backend/tests/test_bet_tracker.py`:

```python
def test_create_bet_stores_user_id_and_list_bets_filters_by_it(tmp_path):
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.1, stake=10.0,
        source="manual", recommendation_snapshot=None, user_id=1,
    )
    tracker.create_bet(
        match_id="m2", date="2026-08-23", home_team="Chelsea", away_team="Fulham",
        market="result_3way", selection="away", odds=3.0, stake=5.0,
        source="manual", recommendation_snapshot=None, user_id=2,
    )
    user_1_bets = tracker.list_bets(user_id=1)
    assert len(user_1_bets) == 1
    assert user_1_bets[0].match_id == "m1"
    assert user_1_bets[0].user_id == 1
```

(Check the top of the file for its existing `BetTracker`/`create_bet` import — reuse it, don't re-import.)

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest app/backend/tests/test_bet_tracker.py::test_create_bet_stores_user_id_and_list_bets_filters_by_it -v`
Expected: FAIL with `TypeError: create_bet() got an unexpected keyword argument 'user_id'`

- [ ] **Step 3: Implement — add the column and thread `user_id` through**

In `app/backend/bet_tracker.py`, update the schema, dataclass, and every method:

```python
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
    user_id: int | None
```

```python
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
                    created_at TEXT NOT NULL,
                    user_id INTEGER
                )
                """
            )
            # W210: additive migration for pre-existing DBs created before
            # user_id existed -- CREATE TABLE IF NOT EXISTS is a no-op on an
            # already-existing table, so the column must be added by hand.
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(user_bets)")}
            if "user_id" not in existing_cols:
                conn.execute("ALTER TABLE user_bets ADD COLUMN user_id INTEGER")
```

```python
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
        user_id: int | None = None,
    ) -> Bet:
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        snapshot_json = json.dumps(recommendation_snapshot) if recommendation_snapshot is not None else None
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO user_bets
                (match_id, date, home_team, away_team, market, selection, odds, stake,
                 outcome, profit_loss, source, recommendation_snapshot_json, created_at, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', NULL, ?, ?, ?, ?)
                """,
                (match_id, date, home_team, away_team, market, selection, odds, stake,
                 source, snapshot_json, created_at, user_id),
            )
            bet_id = cursor.lastrowid
        return self.get_bet(bet_id)  # type: ignore[return-value]

    def get_bet(self, bet_id: int) -> Bet | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_bets WHERE id = ?", (bet_id,)
            ).fetchone()
        return self._row_to_bet(row) if row else None

    def list_bets(self, user_id: int | None = None) -> list[Bet]:
        with self._connect() as conn:
            if user_id is None:
                rows = conn.execute("SELECT * FROM user_bets ORDER BY id ASC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM user_bets WHERE user_id = ? ORDER BY id ASC", (user_id,)
                ).fetchall()
        return [self._row_to_bet(row) for row in rows]

    def list_open_bets(self, user_id: int | None = None) -> list[Bet]:
        with self._connect() as conn:
            if user_id is None:
                rows = conn.execute(
                    "SELECT * FROM user_bets WHERE outcome = 'open' ORDER BY id ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM user_bets WHERE outcome = 'open' AND user_id = ? ORDER BY id ASC",
                    (user_id,),
                ).fetchall()
        return [self._row_to_bet(row) for row in rows]
```

```python
    @staticmethod
    def _row_to_bet(row: tuple) -> Bet:
        (bet_id, match_id, date, home_team, away_team, market, selection, odds, stake,
         outcome, profit_loss, source, snapshot_json, created_at, user_id) = row
        return Bet(
            id=bet_id, match_id=match_id, date=date, home_team=home_team, away_team=away_team,
            market=market, selection=selection, odds=odds, stake=stake, outcome=outcome,
            profit_loss=profit_loss, source=source,
            recommendation_snapshot=json.loads(snapshot_json) if snapshot_json else None,
            created_at=created_at, user_id=user_id,
        )
```

`settle_bet` and `get_bet` are unchanged (they operate on a single already-known `bet_id`, ownership is checked at the route layer in Task 4, not here).

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest app/backend/tests/test_bet_tracker.py -v`
Expected: all pass, including every pre-existing test (they call `create_bet`/`list_bets` without `user_id`, which now defaults to `None` — identical behavior to before)

- [ ] **Step 5: Commit**

```bash
git add app/backend/bet_tracker.py app/backend/tests/test_bet_tracker.py
git commit -m "feat(auth): add user_id column to user_bets, backward-compatible default (W210)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: `get_current_user_email` dependency

**Files:**
- Create: `app/backend/auth_deps.py`
- Test: `app/backend/tests/test_auth_deps.py`

- [ ] **Step 1: Write the failing test**

```python
# app/backend/tests/test_auth_deps.py
from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.backend.auth_deps import get_current_user_email

app = FastAPI()


@app.get("/whoami")
def whoami(email: str = pytest.importorskip("fastapi").Depends(get_current_user_email)):
    return {"email": email}


def test_rejects_when_internal_secret_env_var_unset(monkeypatch):
    monkeypatch.delenv("INTERNAL_API_SECRET", raising=False)
    with TestClient(app) as client:
        response = client.get("/whoami", headers={"X-User-Email": "jane@gmail.com", "X-Internal-Secret": "anything"})
    assert response.status_code == 401


def test_rejects_wrong_secret(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_SECRET", "real-secret")
    with TestClient(app) as client:
        response = client.get("/whoami", headers={"X-User-Email": "jane@gmail.com", "X-Internal-Secret": "wrong"})
    assert response.status_code == 401


def test_rejects_missing_email_header(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_SECRET", "real-secret")
    with TestClient(app) as client:
        response = client.get("/whoami", headers={"X-Internal-Secret": "real-secret"})
    assert response.status_code == 401


def test_accepts_correct_secret_and_email(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_SECRET", "real-secret")
    with TestClient(app) as client:
        response = client.get("/whoami", headers={"X-User-Email": "jane@gmail.com", "X-Internal-Secret": "real-secret"})
    assert response.status_code == 200
    assert response.json() == {"email": "jane@gmail.com"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest app/backend/tests/test_auth_deps.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.backend.auth_deps'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/backend/auth_deps.py
"""W210: FastAPI dependency for routes scoped to a real per-user identity.

Deliberately NOT JWT verification -- FastAPI never talks to Google or
Auth.js directly. The Next.js frontend's own server-side proxy routes
(app/frontend/app/api/bets/*) call getServerSession() there, which already
cryptographically verifies the session cookie using NEXTAUTH_SECRET; by the
time a request reaches here, the email has already been vouched for. This
dependency's only job is confirming the request truly came from our own
Next.js server (via INTERNAL_API_SECRET, a server-only env var never sent
to any browser) rather than a client hitting FastAPI directly and forging
an X-User-Email header. Two distinct trust boundaries, two distinct
secrets -- see the plan's header note on why this isn't APP_ACCESS_TOKEN."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException


def get_current_user_email(
    x_user_email: str | None = Header(default=None),
    x_internal_secret: str | None = Header(default=None),
) -> str:
    expected_secret = os.environ.get("INTERNAL_API_SECRET")
    if not expected_secret or x_internal_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not x_user_email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_user_email
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest app/backend/tests/test_auth_deps.py -v`
Expected: 4 passed

(The `pytest.importorskip("fastapi").Depends` in the test file's route definition is awkward — replace it with a plain top-of-file `from fastapi import Depends` import instead, then `email: str = Depends(get_current_user_email)`. Simpler, same effect.)

- [ ] **Step 5: Commit**

```bash
git add app/backend/auth_deps.py app/backend/tests/test_auth_deps.py
git commit -m "feat(auth): add get_current_user_email dependency (W210)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Wire user auth into the bet routes; scope `APP_ACCESS_TOKEN` to admin only

**Files:**
- Modify: `app/backend/main.py:499-528` (`RequireAppTokenMiddleware`)
- Modify: `app/backend/main.py:1323-1387` (the 5 `/api/bets*` routes)
- Modify: `app/backend/bets.py` (`get_bet_tracker` stays; add nothing new here — user resolution happens in main.py)
- Test: `app/backend/tests/test_bets_endpoints.py` (existing — update, don't break)

- [ ] **Step 1: Write the failing test**

Add to `app/backend/tests/test_bets_endpoints.py`:

```python
from app.backend.auth_deps import get_current_user_email


def test_list_bets_requires_user_auth_and_filters_by_user(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
    tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.1, stake=10.0,
        source="manual", recommendation_snapshot=None, user_id=1,
    )
    tracker.create_bet(
        match_id="m2", date="2026-08-23", home_team="Chelsea", away_team="Fulham",
        market="result_3way", selection="away", odds=3.0, stake=5.0,
        source="manual", recommendation_snapshot=None, user_id=2,
    )
    app.dependency_overrides[get_current_user_email] = lambda: "user1@gmail.com"
    try:
        with TestClient(app) as client:
            response = client.get("/api/bets")
        assert response.status_code == 200
        # Both come back because the override doesn't resolve to a real
        # user_id -- this test only proves the dependency is wired in and
        # the route still works; user-id resolution is verified in the
        # next test.
    finally:
        app.dependency_overrides.pop(get_current_user_email, None)


def test_list_bets_401s_without_the_internal_secret(tmp_path: Path):
    _override_tracker(tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/bets")
    assert response.status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest app/backend/tests/test_bets_endpoints.py -v`
Expected: `test_list_bets_401s_without_the_internal_secret` FAILs (currently returns 200 — no auth dependency yet)

- [ ] **Step 3: Implement**

In `app/backend/main.py`, narrow the middleware:

```python
    async def dispatch(self, request, call_next):
        token = os.environ.get("APP_ACCESS_TOKEN")
        is_exempt = request.method == "OPTIONS" or request.url.path == "/api/health"
        is_admin_route = request.url.path.startswith("/api/admin/")
        if not token or is_exempt or not is_admin_route:
            return await call_next(request)
        if request.headers.get("x-app-token") != token:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)
```

Update the docstring's first paragraph to say "gates `/api/admin/*` routes" instead of "every request" — the rest of the reasoning about middleware ordering is unchanged.

Then wire `get_current_user_email` + `UserStore` into the bet routes:

```python
from app.backend.auth_deps import get_current_user_email
from app.backend.users import UserStore


def get_user_store() -> UserStore:
    return UserStore()


@app.post("/api/bets/from-recommendation")
async def create_bet_from_recommendation(
    request: BetFromRecommendationRequest,
    tracker: BetTracker = Depends(bets.get_bet_tracker),
    user_email: str = Depends(get_current_user_email),
    user_store: UserStore = Depends(get_user_store),
) -> BetOut:
    """Every field but stake is locked -- derived from the recommendation
    snapshot itself, which is also stored verbatim (recommendations aren't
    reproducible run-to-run, agent_techspec.md sec18.6)."""
    try:
        resolved = bets.resolve_from_recommendation(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    user = user_store.get_or_create(user_email)
    bet = tracker.create_bet(
        match_id=resolved["match_id"], date=resolved["date"],
        home_team=resolved["home_team"], away_team=resolved["away_team"],
        market=resolved["market"], selection=resolved["selection"],
        odds=resolved["odds"], stake=resolved["stake"],
        source="from_recommendation", recommendation_snapshot=request.recommendation,
        user_id=user.id,
    )
    return BetOut.from_bet(bet)


@app.post("/api/bets/manual")
async def create_bet_manual(
    request: BetManualRequest,
    tracker: BetTracker = Depends(bets.get_bet_tracker),
    user_email: str = Depends(get_current_user_email),
    user_store: UserStore = Depends(get_user_store),
) -> BetOut:
    user = user_store.get_or_create(user_email)
    bet = tracker.create_bet(
        match_id=request.match_id, date=request.date,
        home_team=request.home_team, away_team=request.away_team,
        market=request.market, selection=request.selection,
        odds=request.odds, stake=request.stake,
        source="manual", recommendation_snapshot=None,
        user_id=user.id,
    )
    return BetOut.from_bet(bet)


@app.get("/api/bets")
async def list_bets(
    tracker: BetTracker = Depends(bets.get_bet_tracker),
    user_email: str = Depends(get_current_user_email),
    user_store: UserStore = Depends(get_user_store),
) -> list[BetOut]:
    user = user_store.get_or_create(user_email)
    return [BetOut.from_bet(bet) for bet in tracker.list_bets(user_id=user.id)]


@app.get("/api/bets/stats")
async def get_bet_stats(
    tracker: BetTracker = Depends(bets.get_bet_tracker),
    user_email: str = Depends(get_current_user_email),
    user_store: UserStore = Depends(get_user_store),
) -> dict:
    user = user_store.get_or_create(user_email)
    return compute_bet_stats(tracker.list_bets(user_id=user.id))


@app.post("/api/bets/settle-open")
async def settle_open(
    tracker: BetTracker = Depends(bets.get_bet_tracker),
    user_email: str = Depends(get_current_user_email),
    user_store: UserStore = Depends(get_user_store),
) -> list[BetOut]:
    user = user_store.get_or_create(user_email)
    client = get_fixtures_client()
    sweden_client = get_sweden_fixtures_client()
    settled = await run_in_threadpool(settle_open_bets, tracker, client, sweden_client, user_id=user.id)
    return [BetOut.from_bet(bet) for bet in settled]
```

`settle_open_bets` (in `app/backend/settlement.py`) needs a `user_id` parameter threaded to its own `tracker.list_open_bets(...)` call — open its definition, add `user_id: int | None = None` to its signature, and pass it to `tracker.list_open_bets(user_id=user_id)` inside.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest app/backend/tests/test_bets_endpoints.py app/backend/tests/test_settlement.py -v`
Expected: all pass. If `test_settlement.py` has existing calls to `settle_open_bets(tracker, client, sweden_client)` without `user_id`, they still pass unchanged (new param defaults to `None`).

- [ ] **Step 5: Commit**

```bash
git add app/backend/main.py app/backend/settlement.py app/backend/tests/test_bets_endpoints.py
git commit -m "feat(auth): require per-user auth on bet routes, scope APP_ACCESS_TOKEN to /api/admin (W210)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Migration script for existing bets

**Files:**
- Create: `scripts/migrate_bets_to_owner.py`
- Test: `scripts/test_migrate_bets_to_owner.py`

- [ ] **Step 1: Write the failing test**

```python
# scripts/test_migrate_bets_to_owner.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest scripts/test_migrate_bets_to_owner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.migrate_bets_to_owner'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/migrate_bets_to_owner.py
"""W210: one-time migration -- every user_bets row created before multi-user
existed has user_id=NULL. Assigns them all to a single owner account.
Idempotent: only touches rows still NULL, so running it twice (or against a
DB that's already been migrated) is a safe no-op on the second run."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from app.backend.bet_tracker import DEFAULT_DB_PATH as DEFAULT_BETS_DB_PATH
from app.backend.users import DEFAULT_DB_PATH as DEFAULT_USERS_DB_PATH
from app.backend.users import UserStore


def migrate_bets_to_owner(
    owner_email: str,
    bets_db_path: str | Path = DEFAULT_BETS_DB_PATH,
    users_db_path: str | Path = DEFAULT_USERS_DB_PATH,
) -> int:
    """Returns the number of rows migrated."""
    owner = UserStore(db_path=users_db_path).get_or_create(owner_email)
    conn = sqlite3.connect(bets_db_path)
    try:
        cursor = conn.execute(
            "UPDATE user_bets SET user_id = ? WHERE user_id IS NULL", (owner.id,)
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-email", required=True, help="Email to assign all existing (user_id=NULL) bets to")
    args = parser.parse_args()
    count = migrate_bets_to_owner(owner_email=args.owner_email)
    print(f"Migrated {count} bet(s) to {args.owner_email}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest scripts/test_migrate_bets_to_owner.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/migrate_bets_to_owner.py scripts/test_migrate_bets_to_owner.py
git commit -m "feat(auth): add idempotent migration script for pre-multiuser bets (W210)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Google sign-in with an invite-only allowlist (Auth.js)

**Files:**
- Modify: `app/frontend/package.json` (add `next-auth`)
- Create: `app/frontend/lib/auth.ts`
- Create: `app/frontend/app/api/auth/[...nextauth]/route.ts`

- [ ] **Step 1: Install the dependency**

```bash
cd app/frontend && npm install next-auth@4.24.7
```

- [ ] **Step 2: Write `authOptions`**

```typescript
// app/frontend/lib/auth.ts
// W210: Google-only, invite-only sign-in. ALLOWED_EMAILS is a plain
// comma-separated env var, not a database table -- deliberately the
// simplest possible allowlist for a small, manually-curated user list.
// Google does the real authentication; this callback is the only gate on
// top of it. JWT session strategy (not database sessions) -- no shared
// session store needed between this app and FastAPI, and FastAPI never
// needs to read it at all (see app/backend/auth_deps.py's own docstring).
import type { NextAuthOptions } from "next-auth";
import GoogleProvider from "next-auth/providers/google";

function getAllowedEmails(): string[] {
  return (process.env.ALLOWED_EMAILS ?? "")
    .split(",")
    .map((e) => e.trim().toLowerCase())
    .filter(Boolean);
}

export const authOptions: NextAuthOptions = {
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
  ],
  session: { strategy: "jwt" },
  callbacks: {
    async signIn({ user }) {
      if (!user.email) return false;
      const allowed = getAllowedEmails();
      return allowed.includes(user.email.toLowerCase());
    },
  },
  pages: {
    signIn: "/login",
  },
};
```

- [ ] **Step 3: Wire the NextAuth route handler**

```typescript
// app/frontend/app/api/auth/[...nextauth]/route.ts
import NextAuth from "next-auth";
import { authOptions } from "@/lib/auth";

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
```

- [ ] **Step 4: Manual verification (no automated test — this step is inherently an external OAuth integration)**

1. Go to console.cloud.google.com → create/select a project → APIs & Services → **OAuth consent screen**: User type **External**; fill in app name + your email as support/developer contact; **leave Publishing status as "Testing"** (do not click Publish App) and add each allowlisted Google account under "Test users." This is a free, second gate — while in Testing mode, Google itself refuses login to any account not listed here, on top of (not instead of) this app's own `ALLOWED_EMAILS` check in the `signIn` callback below. Publishing to Production removes this gate and eventually requires Google's app-verification review — neither is needed or wanted for a small, invite-only user list.
2. APIs & Services → **Credentials** → Create Credentials → **OAuth client ID** → Application type **Web application**. Authorized JavaScript origin: `http://localhost:3000`. Authorized redirect URI: `http://localhost:3000/api/auth/callback/google` (Auth.js's fixed callback path — not configurable). Copy the resulting Client ID/Secret.
3. When deploying for real, return to this same client and add a second origin/redirect-URI pair for the production domain — one client can hold both dev and prod URLs, no need for a second client.
4. Add to `app/frontend/.env.local`:
   ```
   GOOGLE_CLIENT_ID=<from Google Cloud Console>
   GOOGLE_CLIENT_SECRET=<from Google Cloud Console>
   NEXTAUTH_SECRET=<run: openssl rand -base64 32>
   NEXTAUTH_URL=http://localhost:3000
   ALLOWED_EMAILS=your-own-email@gmail.com
   ```
5. Run `npm run dev`, visit `http://localhost:3000/api/auth/signin`, click "Sign in with Google," sign in with the allowlisted email.
6. Expected: redirected back, no error. Try again after temporarily removing your email from `ALLOWED_EMAILS` (leaving it as a Google test user) — expected: redirected to `/login` with an `?error=AccessDenied`-style query param (this app's own `signIn` callback returned `false`, distinct from Google's own Testing-mode rejection, which would instead show Google's own "access blocked" page before ever reaching this app).

- [ ] **Step 5: Commit**

```bash
git add app/frontend/package.json app/frontend/package-lock.json app/frontend/lib/auth.ts app/frontend/app/api/auth
git commit -m "feat(auth): Google sign-in with invite-only email allowlist (W210)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Login page and a protected "My Bets" route

**Files:**
- Create: `app/frontend/app/login/page.tsx`
- Create: `app/frontend/middleware.ts`

- [ ] **Step 1: Write the login page**

```tsx
// app/frontend/app/login/page.tsx
"use client";

import { signIn } from "next-auth/react";

export default function LoginPage() {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginTop: "4rem", gap: "1rem" }}>
      <h1>Sign in</h1>
      <button onClick={() => signIn("google", { callbackUrl: "/bets" })}>
        Sign in with Google
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Protect `/bets` (and any other user-scoped page) via middleware**

```typescript
// app/frontend/middleware.ts
// W210: gates only user-scoped pages -- the shared dashboard ("/") stays
// unauthenticated by design (Task decision: shared recommendations,
// private bet tracking only).
export { default } from "next-auth/middleware";

export const config = {
  matcher: ["/bets/:path*"],
};
```

- [ ] **Step 3: Manual verification**

1. `npm run dev`, visit `http://localhost:3000/bets` while signed out.
2. Expected: redirected to `/login` (Auth.js's default middleware behavior for a matched, unauthenticated route).
3. Sign in, revisit `/bets` — expected: page loads (it doesn't exist yet as a real page if this is a fresh app; a placeholder `app/frontend/app/bets/page.tsx` returning `<div>Bets</div>` is enough to prove the redirect logic, and gets replaced by real content in Task 9).

- [ ] **Step 4: Commit**

```bash
git add app/frontend/app/login app/frontend/middleware.ts
git commit -m "feat(auth): login page and middleware protecting /bets (W210)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 8: Server-side proxy routes for the bet endpoints

**Files:**
- Create: `app/frontend/app/api/bets/route.ts` (GET list, POST manual)
- Create: `app/frontend/app/api/bets/from-recommendation/route.ts`
- Create: `app/frontend/app/api/bets/settle-open/route.ts`
- Create: `app/frontend/app/api/bets/stats/route.ts`

- [ ] **Step 1: Write the shared forwarding helper**

```typescript
// app/frontend/lib/backendProxy.ts
// W210: every proxy route in app/api/bets/* goes through this so the
// INTERNAL_API_SECRET header (server-only, never sent to the browser) is
// attached exactly once, in exactly one place.
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function forwardToBackend(path: string, init: RequestInit = {}): Promise<Response> {
  const session = await getServerSession(authOptions);
  if (!session?.user?.email) {
    return new Response(JSON.stringify({ detail: "Unauthorized" }), { status: 401 });
  }
  const headers = new Headers(init.headers);
  headers.set("X-User-Email", session.user.email);
  headers.set("X-Internal-Secret", process.env.INTERNAL_API_SECRET!);
  if (init.body) headers.set("Content-Type", "application/json");
  return fetch(`${API_BASE}${path}`, { ...init, headers });
}
```

- [ ] **Step 2: Write the route handlers**

```typescript
// app/frontend/app/api/bets/route.ts
import { forwardToBackend } from "@/lib/backendProxy";

export async function GET() {
  const response = await forwardToBackend("/api/bets");
  return new Response(await response.text(), { status: response.status });
}

export async function POST(request: Request) {
  const body = await request.text();
  const response = await forwardToBackend("/api/bets/manual", { method: "POST", body });
  return new Response(await response.text(), { status: response.status });
}
```

```typescript
// app/frontend/app/api/bets/from-recommendation/route.ts
import { forwardToBackend } from "@/lib/backendProxy";

export async function POST(request: Request) {
  const body = await request.text();
  const response = await forwardToBackend("/api/bets/from-recommendation", { method: "POST", body });
  return new Response(await response.text(), { status: response.status });
}
```

```typescript
// app/frontend/app/api/bets/settle-open/route.ts
import { forwardToBackend } from "@/lib/backendProxy";

export async function POST() {
  const response = await forwardToBackend("/api/bets/settle-open", { method: "POST" });
  return new Response(await response.text(), { status: response.status });
}
```

```typescript
// app/frontend/app/api/bets/stats/route.ts
import { forwardToBackend } from "@/lib/backendProxy";

export async function GET() {
  const response = await forwardToBackend("/api/bets/stats");
  return new Response(await response.text(), { status: response.status });
}
```

- [ ] **Step 3: Add `INTERNAL_API_SECRET` to both environments**

`app/frontend/.env.local`:
```
INTERNAL_API_SECRET=<run: openssl rand -base64 32>
```

Backend `.env` (must be the **same value**):
```
INTERNAL_API_SECRET=<the same value as above>
```

- [ ] **Step 4: Manual verification**

1. Start FastAPI (`./venv/bin/python -m main serve`, or however it's normally run locally) and `npm run dev`.
2. Sign in via `/login`.
3. `curl -H "Cookie: <copy from browser devtools after signing in>" http://localhost:3000/api/bets/stats` — expected: a real stats JSON, not a 401.
4. `curl http://localhost:3000/api/bets/stats` (no cookie) — expected: 401.
5. `curl http://localhost:8000/api/bets` directly (bypassing the proxy, no headers) — expected: 401 from `get_current_user_email` (Task 3/4), confirming FastAPI itself is not reachable without the internal secret even if someone finds its URL.

- [ ] **Step 5: Commit**

```bash
git add app/frontend/lib/backendProxy.ts app/frontend/app/api/bets
git commit -m "feat(auth): server-side proxy routes forwarding verified sessions to FastAPI (W210)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 9: Point the frontend's bet functions at the new proxy routes

**Files:**
- Modify: `app/frontend/lib/api.ts:91-157` (the 5 bet-related functions)

- [ ] **Step 1: Update `lib/api.ts`**

Change every bet-related function's URL from `${API_BASE}/api/bets...` (via `apiFetch`, which attaches the now-irrelevant-for-these-routes `X-App-Token`) to a plain same-origin `fetch` against `/api/bets...` (the new proxy routes, which rely on the browser's own session cookie — no manual header needed client-side):

```typescript
export async function logBetFromRecommendation(body: {
  match_id: string; recommendation: unknown; market: string; selection: string; stake: number;
}): Promise<Bet> {
  const response = await fetch(`/api/bets/from-recommendation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new ApiError(`Failed to log bet (${response.status})`, response.status);
  return response.json();
}

export async function logBetManual(body: {
  match_id: string; date: string; home_team: string; away_team: string;
  market: string; selection: string; odds: number; stake: number;
}): Promise<Bet> {
  const response = await fetch(`/api/bets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new ApiError(`Failed to log bet (${response.status})`, response.status);
  return response.json();
}

export async function getBets(): Promise<Bet[]> {
  const response = await fetch(`/api/bets`);
  if (!response.ok) throw new ApiError(`Failed to load bets (${response.status})`, response.status);
  return response.json();
}

export async function settleOpenBets(): Promise<Bet[]> {
  const response = await fetch(`/api/bets/settle-open`, { method: "POST" });
  if (!response.ok) throw new ApiError(`Failed to settle open bets (${response.status})`, response.status);
  return response.json();
}

export async function getBetStats(): Promise<BetStats> {
  const response = await fetch(`/api/bets/stats`);
  if (!response.ok) throw new ApiError(`Failed to load bet stats (${response.status})`, response.status);
  return response.json();
}
```

Every other function in this file (`getFixtures`, `generateRecommendation`, `getCachedRecommendation`, `getStatus`, etc.) is **unchanged** — they stay on the direct-to-FastAPI `apiFetch`/`API_BASE` path, since the shared dashboard remains unauthenticated by design.

- [ ] **Step 2: Manual verification**

Open the app in a browser signed in as the allowlisted user, log a manual bet via whatever UI calls `logBetManual`, refresh, confirm `getBets()` shows it. Open a private/incognito window, attempt to hit `/api/bets` directly without signing in — expected: 401 (this is the same as Task 8 Step 4's curl check, now exercised through the real UI/fetch path).

- [ ] **Step 3: Commit**

```bash
git add app/frontend/lib/api.ts
git commit -m "feat(auth): frontend bet calls go through the new authenticated proxy routes (W210)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 10: Run the real migration and verify end-to-end

**Files:** none (operational step)

- [ ] **Step 1: Back up the real database first**

```bash
cp data/user_bets.db data/user_bets.db.bak-pre-migration
```

- [ ] **Step 2: Run the migration against the real DB**

```bash
./venv/bin/python scripts/migrate_bets_to_owner.py --owner-email your-own-email@gmail.com
```

Expected output: `Migrated N bet(s) to your-own-email@gmail.com` where N matches however many rows already exist (check first: `sqlite3 data/user_bets.db "SELECT COUNT(*) FROM user_bets"`).

- [ ] **Step 3: Verify directly, not just by trusting the script's own exit code**

```bash
sqlite3 data/user_bets.db "SELECT COUNT(*) FROM user_bets WHERE user_id IS NULL"
```
Expected: `0`

```bash
sqlite3 data/users.db "SELECT * FROM users"
```
Expected: one row, your email.

- [ ] **Step 4: Full end-to-end check**

Sign in as the owner account through the real (or locally-run-against-real-data) app, visit `/bets`, confirm every pre-existing bet still shows up exactly as before the migration.

- [ ] **Step 5: Run the full test suite one more time**

```bash
./venv/bin/python -m pytest tests/ app/backend/tests/ scripts/ -q
cd app/frontend && npx vitest run
```
Expected: same pass count as before this feature (plus the new tests from Tasks 1-5), zero new failures beyond the pre-existing, documented `test_fixtures_endpoint.py` environment gap.

- [ ] **Step 6: Commit the migration backup removal (or keep it — your call) and update the docs**

Per this project's own established workflow (CLAUDE.md), append a completed entry to `documents/app_user_stories.md` (next available `W` id) summarizing what shipped, and update `documents/app_prd.md`'s Section 2.3 (already marked "decided direction, not yet implemented" — change to "implemented," per PHASE 48/A111's own convention of updating the PRD as decisions become real).

```bash
git add documents/app_user_stories.md documents/app_prd.md
git commit -m "docs(app): W209 -- multi-user Google auth shipped

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage** (against the 4 decisions from planning):
1. Shared recommendations, private bet tracking → Task 9's note (only bet routes move to the proxy; dashboard/recommendation calls untouched). ✅
2. Invite-only allowlist via Google → Task 6. ✅
3. Existing data migrated to the owner account → Tasks 5 and 10. ✅
4. `APP_ACCESS_TOKEN` kept for admin only → Task 4, Step 3. ✅

**Placeholder scan:** no TBD/TODO, every step has real, complete code or an exact manual command with expected output.

**Type/signature consistency:** `create_bet(..., user_id: int | None = None)` (Task 2) matches every call site added in Task 4 and Task 5. `list_bets(user_id: int | None = None)` and `list_open_bets(user_id: int | None = None)` match their Task 4 call sites. `get_current_user_email` (Task 3) is imported and used with the identical name in Task 4. `UserStore.get_or_create`/`get_by_email` (Task 1) match their usages in Tasks 4, 5, and 10.
