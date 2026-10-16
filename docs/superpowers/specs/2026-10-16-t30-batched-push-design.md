# Batched T-30 "Kickoff Soon" Push Notifications

## 1. Problem

The PWA push system (`docs/superpowers/specs/2026-10-15-pwa-mobile-app-design.md`) currently fires on two events: nightly new-recommendations-ready, and bet-settled. It does not notify around the per-fixture T-30 refresh (`app/backend/t30_refresh.py`, W10) — the recommendation refresh that runs 30 minutes before each fixture's kickoff, across every fixture in all 6 enabled leagues, every day.

Naively sending one push per T-30 job would spam every subscriber: T-30 jobs are scheduled per fixture, independently, and real kickoffs cluster tightly around common slots (15:00, 17:30, 20:00). A typical Saturday can have 20-30+ fixtures T-30'ing within the same 15-minute window, which would mean 20-30+ near-simultaneous pushes to every user, most for matches they have no stake in.

## 2. Scope

**In scope:**
- Cluster same-night fixtures by kickoff-time proximity (configurable window, default 15 minutes).
- One push per cluster, sent after every fixture in that cluster has had its own T-30 refresh complete.
- Confirmed semantics (direct user decision): blast to every subscriber (not scoped to a user's own bets), fires regardless of whether the refreshed pick actually changed from its prior value. This mirrors the existing nightly "new recommendations ready" push's blast model, just event-triggered instead of once-nightly.

**Out of scope (explicitly not doing):**
- Per-user relevance filtering (only notifying users with a bet on a clustered match) — rejected in favor of the simpler blast model, consistent with the decision above.
- Only notifying on a materially changed pick — rejected; every T-30 refresh in a cluster counts, regardless of outcome.
- A UI/admin control for the clustering window — it's a named constant, not a runtime setting (this app has no admin surface for this and no per-operator need to tune it).
- Changing the existing per-fixture T-30 *refresh* job (`t30_refresh.py`) in any way — it stays exactly as-is, fault-isolated per fixture. This spec only adds a notification layer on top, reading already-refreshed cache state.

## 3. Clustering

New pure function, e.g. `cluster_fixtures_by_kickoff(fixtures: list[NormalizedMatch], window_minutes: int = 15) -> list[list[NormalizedMatch]]`, in `app/backend/scheduler_wiring.py` (or a small new module if that file is judged too large at implementation time — a call for the plan, not this spec).

Algorithm: sort fixtures by `utc_date` ascending; walk the sorted list, starting a new cluster whenever the gap between a fixture and the previous one exceeds `window_minutes`; otherwise append to the current cluster. This is chained clustering, not a fixed grid — e.g. with a 15-minute window, fixtures at 14:55, 15:05, 15:18 stay one cluster (each consecutive gap is ≤ 15 min: 10, then 13) even though the first and last are 23 minutes apart; a fixture at 15:40 would start a new cluster (gap from 15:18 is 22 min, over the window).

## 4. Scheduling

Today, `_eod_job()` (`scheduler_wiring.py`) loops over `COMPETITIONS`, fetching and generating recommendations one league at a time, scheduling each fixture's own T-30 refresh job inside that per-league flow (via `build_schedule_t30`/`run_eod_batch`). That per-league flow is untouched by this change.

Addition: accumulate every league's fetched fixtures into one combined list across the loop's iterations (bookkeeping only — the per-league generation/scheduling calls themselves don't change). After the loop finishes, run `cluster_fixtures_by_kickoff` over the combined list. For each resulting cluster, schedule one new one-shot job via the existing `scheduler.schedule_once(...)` mechanism (same primitive `build_schedule_t30` already uses), timed at `max(kickoff times in cluster) - 30 minutes + a small buffer (e.g. 3 minutes)` — i.e. after the *last* fixture in the cluster would have had its own T-30 refresh complete, so every cluster member's cache entry is current before the notification reads it.

## 5. The notify job

No `RecommendationCache` read needed: the approved message format (Section 2) surfaces only team names, not pick content, and the cluster's fixtures (with their `home_team`/`away_team`) are already fully known at EOD-schedule time, captured in the notify job's closure. For a given cluster, the job builds the push body directly from that fixture list (up to 3 "Home v Away" pairs, then `"+N more"` if the cluster has more) and calls the existing `send_to_all_subscribers` (`push_notifications.py`, unchanged) with `title="Kickoff soon"`, that body, `url="/"`. (If a future revision wants pick content in the body, that's the point where a cache read would be added -- not needed for this spec's scope.)

Reuses the same `push_store: PushSubscriptionStore | None = None`-defaults-to-skip pattern the nightly EOD push already established in `register_eod_job` — existing tests that don't pass a push store stay unaffected.

## 6. Testing

- Unit tests for `cluster_fixtures_by_kickoff`: empty list, single fixture, a tight cluster, two fixtures outside the window (two separate clusters), a chained cluster where only consecutive gaps are checked (first/last fixture more than `window_minutes` apart but still one cluster).
- Unit test for the push-body builder: ≤3 matches (no "+N more"), >3 matches (truncated with correct count).
- Integration-style test (mirroring the existing `test_register_eod_job_notifies_subscribers_once_after_the_nightly_run` pattern): registering the EOD job with a push store and multiple same-night fixtures schedules the expected number of cluster-notify jobs, and firing one sends exactly one push with the expected body.
