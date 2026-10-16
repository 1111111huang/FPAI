# Batched T-30 Kickoff-Soon Push Notifications Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one batched "Kickoff soon" push per cluster of same-night fixtures whose kickoffs fall within a configurable window of each other, instead of spamming one push per fixture's T-30 refresh.

**Architecture:** Two new pure functions (`cluster_fixtures_by_kickoff`, `build_batch_notification_body`) plus a small scheduling addition to `_eod_job()` in `app/backend/scheduler_wiring.py`: after the existing per-league fetch/generate loop, the combined fixture list is clustered and one new one-shot job per cluster is scheduled via the same `scheduler.schedule_once` primitive `build_schedule_t30` already uses, timed just after the cluster's last fixture would have had its own T-30 refresh complete. The existing per-fixture T-30 *refresh* job is untouched.

**Tech Stack:** Python/FastAPI backend (existing), `RecoverableScheduler`/`schedule_once` (existing), `send_to_all_subscribers` (existing, from the PWA push work).

**Spec:** `docs/superpowers/specs/2026-10-16-t30-batched-push-design.md`

---

## Before you start

All code locations below were read directly from the current file (post-PWA-merge) — `app/backend/scheduler_wiring.py`'s `register_eod_job`/`_eod_job` already has the `push_store: PushSubscriptionStore | None = None` parameter and the nightly "New recommendations ready" push block (lines 594-601) from the prior PWA plan. This plan builds on top of that, doesn't re-add it.

---

### Task 1: `cluster_fixtures_by_kickoff`

**Files:**
- Modify: `app/backend/scheduler_wiring.py`
- Test: `app/backend/tests/test_scheduler_wiring.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `app/backend/tests/test_scheduler_wiring.py` (these use `NormalizedMatch`, already imported in this file):
```python


def _match(match_id: str, kickoff: str) -> NormalizedMatch:
    return NormalizedMatch(
        match_id=match_id, utc_date=kickoff, status="SCHEDULED",
        home_team=f"Home{match_id}", away_team=f"Away{match_id}", home_goals=None, away_goals=None,
    )


def test_cluster_fixtures_by_kickoff_empty_list_returns_empty() -> None:
    assert cluster_fixtures_by_kickoff([]) == []


def test_cluster_fixtures_by_kickoff_single_fixture_returns_one_cluster() -> None:
    fixture = _match("m1", "2026-10-17T15:00:00Z")
    assert cluster_fixtures_by_kickoff([fixture]) == [[fixture]]


def test_cluster_fixtures_by_kickoff_groups_fixtures_within_window() -> None:
    a = _match("a", "2026-10-17T15:00:00Z")
    b = _match("b", "2026-10-17T15:10:00Z")
    clusters = cluster_fixtures_by_kickoff([b, a], window_minutes=15)
    assert clusters == [[a, b]]


def test_cluster_fixtures_by_kickoff_splits_fixtures_outside_window() -> None:
    a = _match("a", "2026-10-17T15:00:00Z")
    b = _match("b", "2026-10-17T15:30:00Z")
    clusters = cluster_fixtures_by_kickoff([a, b], window_minutes=15)
    assert clusters == [[a], [b]]


def test_cluster_fixtures_by_kickoff_chains_a_cluster_beyond_the_window_end_to_end() -> None:
    """14:55, 15:05, 15:18 stay one cluster -- each CONSECUTIVE gap is <=15
    min (10, then 13) -- even though first-to-last is 23 min, over the
    window. A 15:40 fixture starts a new cluster (22 min since 15:18)."""
    a = _match("a", "2026-10-17T14:55:00Z")
    b = _match("b", "2026-10-17T15:05:00Z")
    c = _match("c", "2026-10-17T15:18:00Z")
    d = _match("d", "2026-10-17T15:40:00Z")
    clusters = cluster_fixtures_by_kickoff([a, b, c, d], window_minutes=15)
    assert clusters == [[a, b, c], [d]]
```
Add `cluster_fixtures_by_kickoff` to the existing `from app.backend.scheduler_wiring import (...)` import block at the top of the file (alphabetical, matching that block's existing style).

- [ ] **Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && pytest app/backend/tests/test_scheduler_wiring.py -k cluster_fixtures -v`
Expected: FAIL — `ImportError: cannot import name 'cluster_fixtures_by_kickoff'`.

- [ ] **Step 3: Write the function**

In `app/backend/scheduler_wiring.py`, add this right after `t30_run_at`'s definition (after its `return kickoff - timedelta(minutes=30)` line, before `build_schedule_t30`):
```python
T30_BATCH_WINDOW_MINUTES = 15


def cluster_fixtures_by_kickoff(
    fixtures: list[NormalizedMatch], window_minutes: int = T30_BATCH_WINDOW_MINUTES
) -> list[list[NormalizedMatch]]:
    """Chained clustering, not a fixed grid: sorted by kickoff, a fixture
    joins the current cluster if it's within `window_minutes` of the
    *previous* fixture in sort order -- so a cluster can span more than
    window_minutes end-to-end if kickoffs chain closely together (a run at
    14:55/15:05/15:18 stays one cluster even though the ends are 23 min
    apart; only each consecutive gap is checked against the window)."""
    if not fixtures:
        return []
    ordered = sorted(fixtures, key=lambda f: f.utc_date)
    clusters: list[list[NormalizedMatch]] = [[ordered[0]]]
    for fixture in ordered[1:]:
        previous = clusters[-1][-1]
        gap = _kickoff_time(fixture) - _kickoff_time(previous)
        if gap <= timedelta(minutes=window_minutes):
            clusters[-1].append(fixture)
        else:
            clusters.append([fixture])
    return clusters
```

Then factor the kickoff-time parse (currently inline in `t30_run_at`) into a shared helper, so both functions use one source of truth. Change:
```python
def t30_run_at(fixture: NormalizedMatch) -> datetime:
    kickoff = datetime.fromisoformat(fixture.utc_date.replace("Z", "+00:00"))
    return kickoff - timedelta(minutes=30)
```
to:
```python
def _kickoff_time(fixture: NormalizedMatch) -> datetime:
    return datetime.fromisoformat(fixture.utc_date.replace("Z", "+00:00"))


def t30_run_at(fixture: NormalizedMatch) -> datetime:
    return _kickoff_time(fixture) - timedelta(minutes=30)
```
(Move `_kickoff_time` above `t30_run_at`, then `T30_BATCH_WINDOW_MINUTES`/`cluster_fixtures_by_kickoff` after it, in that order.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_scheduler_wiring.py -k "cluster_fixtures or t30_run_at" -v`
Expected: PASS (5 new cluster tests + the pre-existing `test_t30_run_at_is_30_minutes_before_kickoff`, confirming the refactor didn't change its behavior).

- [ ] **Step 5: Run the full scheduler_wiring test file**

Run: `pytest app/backend/tests/test_scheduler_wiring.py -q`
Expected: all tests pass (the `_kickoff_time` extraction is a pure refactor, no other test should be affected).

- [ ] **Step 6: Commit**

```bash
git add app/backend/scheduler_wiring.py app/backend/tests/test_scheduler_wiring.py
git commit -m "feat(backend): add cluster_fixtures_by_kickoff for batching same-time fixtures"
```

---

### Task 2: `build_batch_notification_body`

**Files:**
- Modify: `app/backend/scheduler_wiring.py`
- Test: `app/backend/tests/test_scheduler_wiring.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `app/backend/tests/test_scheduler_wiring.py`:
```python


def test_build_batch_notification_body_lists_all_when_at_or_under_limit() -> None:
    a = _match("a", "2026-10-17T15:00:00Z")
    b = _match("b", "2026-10-17T15:05:00Z")
    body = build_batch_notification_body([a, b])
    assert body == "Homea v Awaya, Homeb v Awayb"


def test_build_batch_notification_body_truncates_with_remaining_count() -> None:
    fixtures = [_match(str(i), "2026-10-17T15:00:00Z") for i in range(5)]
    body = build_batch_notification_body(fixtures, max_shown=3)
    assert body == "Home0 v Away0, Home1 v Away1, Home2 v Away2 +2 more"
```
Add `build_batch_notification_body` to the same import block as Task 1.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/backend/tests/test_scheduler_wiring.py -k build_batch_notification_body -v`
Expected: FAIL — `ImportError: cannot import name 'build_batch_notification_body'`.

- [ ] **Step 3: Write the function**

In `app/backend/scheduler_wiring.py`, add right after `cluster_fixtures_by_kickoff`'s closing `return clusters` line:
```python
def build_batch_notification_body(cluster: list[NormalizedMatch], max_shown: int = 3) -> str:
    pairs = [f"{f.home_team} v {f.away_team}" for f in cluster[:max_shown]]
    body = ", ".join(pairs)
    remaining = len(cluster) - max_shown
    if remaining > 0:
        body += f" +{remaining} more"
    return body
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_scheduler_wiring.py -k build_batch_notification_body -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add app/backend/scheduler_wiring.py app/backend/tests/test_scheduler_wiring.py
git commit -m "feat(backend): add build_batch_notification_body for grouped T-30 push text"
```

---

### Task 3: Wire batched scheduling into `_eod_job()`

**Files:**
- Modify: `app/backend/scheduler_wiring.py`
- Test: `app/backend/tests/test_scheduler_wiring.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `app/backend/tests/test_scheduler_wiring.py`:
```python


def test_register_eod_job_schedules_one_batched_notify_per_kickoff_cluster(tmp_path: Path) -> None:
    """Two fixtures 10 minutes apart (one cluster) + a third fixture 40
    minutes later (its own cluster) should produce exactly 2 batch-notify
    pushes, not 3 (one per fixture) and not 1 (all fixtures lumped
    together)."""
    close_a = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    close_b = NormalizedMatch(
        match_id="m2", utc_date=f"{_FUTURE_DAY_STR}T15:10:00Z", status="SCHEDULED",
        home_team="Chelsea", away_team="Fulham", home_goals=None, away_goals=None,
    )
    far = NormalizedMatch(
        match_id="m3", utc_date=f"{_FUTURE_DAY_STR}T15:50:00Z", status="SCHEDULED",
        home_team="Liverpool", away_team="Brentford", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [close_a, close_b, far]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    push_store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    push_store.add(user_id=1, endpoint="https://push.example/a", p256dh="k", auth="a")

    # "now" is well after all 3 fixtures' own T-30 times AND after the
    # batch-notify jobs' (T-30 + buffer) times, so registering triggers
    # immediate catch-up for everything -- same pattern as this file's
    # other EOD/T-30 catch-up tests.
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    recommendation = {
        "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
        "overall": "no_bet", "markets": [], "explanation": "test", "confidence": "low",
        "limitations": [], "prediction_basis": "market_odds_only",
    }
    # Computed ahead of time to match _schedule_t30_batch_notify's own
    # run_at formula (max t30_run_at in the cluster + a 3-minute buffer) --
    # hardcoded as timedelta(minutes=3) here rather than importing
    # T30_NOTIFY_BUFFER_MINUTES, since that constant doesn't exist until
    # this task's Step 3 (this test is written and confirmed-failing
    # first, per TDD).
    close_cluster_run_at = t30_run_at(close_b) + timedelta(minutes=3)  # later of close_a/close_b
    far_cluster_run_at = t30_run_at(far) + timedelta(minutes=3)
    close_job_id = f"t30_notify_{_FUTURE_DAY_STR}_m1"
    far_job_id = f"t30_notify_{_FUTURE_DAY_STR}_m3"

    with patch("app.backend.recommendations.run_agent", return_value=recommendation), \
         patch("app.backend.scheduler_wiring.send_to_all_subscribers") as mock_send:
        register_eod_job(
            scheduler, fixtures_client=fixtures_client, odds_client=None,
            cache=cache, config=config, now_fn=lambda: now, push_store=push_store,
        )
        assert _wait_until(lambda: run_log.has_run(EOD_JOB_ID, _FUTURE_DAY_STR))
        # Both batch-notify jobs' run_at is already in the past relative to
        # "now" (same as T-30's own catch-up above), so registering them
        # triggers an immediate catch-up run on a background thread --
        # must wait for both to actually finish *before* the patch above
        # is torn down, or a late-firing job calls the real (unpatched)
        # send_to_all_subscribers instead of the mock (same gotcha this
        # file's other catch-up tests already guard against).
        assert _wait_until(lambda: run_log.has_run(close_job_id, close_cluster_run_at.isoformat()))
        assert _wait_until(lambda: run_log.has_run(far_job_id, far_cluster_run_at.isoformat()))

    # 1 nightly "New recommendations ready" call + 2 batch-notify calls
    # (one for [close_a, close_b], one for [far]) = 3 total.
    assert mock_send.call_count == 3
    bodies = [kwargs["body"] for _, kwargs in mock_send.call_args_list]
    assert "Tomorrow's picks" in bodies[0]
    batch_bodies = bodies[1:]
    assert any("Arsenal v Everton, Chelsea v Fulham" in b for b in batch_bodies)
    assert any("Liverpool v Brentford" in b for b in batch_bodies)
    titles = [kwargs["title"] for _, kwargs in mock_send.call_args_list]
    assert titles.count("Kickoff soon") == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest app/backend/tests/test_scheduler_wiring.py::test_register_eod_job_schedules_one_batched_notify_per_kickoff_cluster -v`
Expected: FAIL — `assert 1 == 3` (today only the nightly push fires; no batch-notify calls exist yet).

- [ ] **Step 3: Wire the scheduling**

In `app/backend/scheduler_wiring.py`'s `_eod_job()` (inside `register_eod_job`), add `all_fixtures: list[NormalizedMatch] = []` right after the `date_str = next_day_date_str(now_fn)` / `enabled = set(...)` lines, before `for league in COMPETITIONS:`.

Inside the loop, right after the existing:
```python
            if fixtures is None:
                continue  # this league isn't configured this run (e.g. no sweden_fixtures_client)
```
add:
```python
            all_fixtures.extend(fixtures)
```

Then change the existing post-loop push block from:
```python
        # None (the default, and every pre-existing caller) means push
        # notifications aren't wired for this run -- skip entirely rather
        # than constructing a store no caller asked for.
        if push_store is not None:
            send_to_all_subscribers(
                store=push_store, title="New recommendations ready",
                body=f"Tomorrow's picks ({date_str}) are in.", url="/",
            )
```
to:
```python
        # None (the default, and every pre-existing caller) means push
        # notifications aren't wired for this run -- skip entirely rather
        # than constructing a store no caller asked for.
        if push_store is not None:
            send_to_all_subscribers(
                store=push_store, title="New recommendations ready",
                body=f"Tomorrow's picks ({date_str}) are in.", url="/",
            )
            for cluster in cluster_fixtures_by_kickoff(all_fixtures):
                _schedule_t30_batch_notify(scheduler, push_store, cluster, date_str)
```

Add this new function right after `build_schedule_t30`'s closing `return _schedule` line (so it lives next to the other scheduling-closure helper):
```python
T30_NOTIFY_BUFFER_MINUTES = 3


def _schedule_t30_batch_notify(
    scheduler: RecoverableScheduler, push_store: PushSubscriptionStore,
    cluster: list[NormalizedMatch], date_str: str,
) -> None:
    """One push per kickoff cluster (not per fixture) -- scheduled a few
    minutes after the LAST fixture in the cluster would have had its own
    T-30 refresh (t30_refresh.py, unchanged) complete, so every cluster
    member's state is current by the time this reads the fixture list."""
    run_at = max(t30_run_at(fixture) for fixture in cluster) + timedelta(minutes=T30_NOTIFY_BUFFER_MINUTES)
    job_id = f"t30_notify_{date_str}_{cluster[0].match_id}"

    def _job(cluster: list[NormalizedMatch] = cluster) -> None:
        send_to_all_subscribers(
            store=push_store, title="Kickoff soon",
            body=build_batch_notification_body(cluster), url="/",
        )

    scheduler.schedule_once(job_id, _job, run_at=run_at)
```
(The `cluster: list[NormalizedMatch] = cluster` default-argument binding in `_job` is deliberate, not redundant -- it captures each loop iteration's own `cluster` value at closure-creation time, avoiding the classic Python late-binding bug where every `_job` would otherwise reference whatever `cluster` happens to be by the time any of them actually runs, same pattern `build_schedule_t30`'s own closures already use implicitly via its outer function call per fixture.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest app/backend/tests/test_scheduler_wiring.py::test_register_eod_job_schedules_one_batched_notify_per_kickoff_cluster -v`
Expected: PASS.

- [ ] **Step 5: Run the full backend suite**

Run: `pytest app/backend/tests -q`
Expected: all tests pass except the 8 known pre-existing, unrelated failures (`test_fixtures_endpoint.py`, `test_agent_performance_dashboard_endpoint.py`, `test_recommendation_outcomes_endpoints.py` — confirmed pre-existing on `main`, nothing to do with this plan).

- [ ] **Step 6: Commit**

```bash
git add app/backend/scheduler_wiring.py app/backend/tests/test_scheduler_wiring.py
git commit -m "feat(backend): schedule one batched 'Kickoff soon' push per T-30 cluster"
```

---

## After all tasks: full verification

```bash
source venv/bin/activate && pytest app/backend/tests -q
```
Expected: same pass/fail shape as Task 3 Step 5 (8 known pre-existing failures, everything else green, including every new test from this plan).
