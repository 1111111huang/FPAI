# Log-a-Bet UX Fixes + MatchCard Quick-Log Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every inconvenience found in the log-a-bet UX audit (lost query params on sign-in, no outcome feedback, no duplicate warning, no delete, minor polish), and add a quick "log the recommended pick" control directly on match cards (Dashboard + Match Explorer), gated to `direct_bet` recommendations only.

**Architecture:** No new pages, no new backend services. Backend gets one new endpoint (`DELETE /api/bets/{id}`, user-scoped) and one new `BetTracker` method. Frontend gets a new `Match.rawRecommendation` field (threaded through the two places `Match` objects already receive a recommendation), a new proxy route, and targeted edits to `MatchUI.tsx`/`BetTracker.tsx`. The new MatchCard quick-log control reuses the existing exported `LogBetButton` component verbatim -- no new bet-logging component.

**Tech Stack:** FastAPI + SQLite (backend), Next.js 14 App Router + React + Vitest/RTL (frontend), existing Auth.js proxy-route pattern.

---

## Context for every task below

- Branch: `feature/w210-multiuser-auth` (already checked out -- do not create a new branch).
- Backend bet routes live in `app/backend/main.py` (currently lines ~1339-1470s); the `Bet`/`BetTracker` model is `app/backend/bet_tracker.py`; request/response schemas are `app/backend/bets.py`.
- Frontend bet UI is split across `app/frontend/components/MatchUI.tsx` (Dashboard/Match Explorer/match-detail-page components, including `MatchCard`, `LogBetButton`, `ProbabilityRow`, `MatchAnalysisPage`) and `app/frontend/components/BetTracker.tsx` (`/bets` page, `ManualBetForm`, `BetRow`).
- Auth: FastAPI routes use `Depends(get_current_user_email)` + `Depends(get_user_store)` (see any existing `/api/bets*` route for the pattern). Frontend calls to `/api/bets*` go through Next.js server-side proxy routes under `app/frontend/app/api/bets/*` (see `app/frontend/lib/backendProxy.ts`'s `forwardToBackend()`), never straight to FastAPI.
- Run backend tests with `./venv/bin/python -m pytest app/backend -q` from the repo root. Run frontend tests with `npx vitest run` from `app/frontend/`. Run `npx tsc --noEmit` and `npx next build` from `app/frontend/` before considering any frontend task done.

---

### Task 1: Fix sign-in redirect losing match query params (Critical)

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx` (`LogBetButton`, ~line 1730-1753)
- Test: `app/frontend/components/__tests__/LogBetButton.test.tsx`

**Problem:** `LogBetButton`'s unauthenticated state links to `` `/login?callbackUrl=${encodeURIComponent(`/matches/${matchId}`)}` ``, dropping the `home`/`away`/`date`/`league` query params `MatchAnalysisPage` requires to render at all (it shows "Missing match details" without them). A user who signs in from this link loses their place completely.

- [ ] **Step 1: Read the current test file in full**

Run: `cat app/frontend/components/__tests__/LogBetButton.test.tsx`

It currently has: a top-level `recommendation` fixture (`as never`-cast, `candidates: []`), a `mockUseSession` mock of `next-auth/react`, and one existing test asserting the OLD hardcoded href (`toHaveAttribute("href", "/login?callbackUrl=%2Fmatches%2Fm1")`) -- that assertion will need updating in Step 4 below, since it encodes the exact bug this task fixes. **No `next/navigation` mock exists in this file yet** -- add one from scratch, mirroring `app/login/__tests__/page.test.tsx`'s established pattern exactly (`mockUsePathname`/`mockUseSearchParams` as `vi.fn()`s, reset in `beforeEach`).

- [ ] **Step 2: Add the next/navigation mock, update the existing test, and add the new one**

At the top of `app/frontend/components/__tests__/LogBetButton.test.tsx`, alongside the existing `mockUseSession`/`vi.mock("next-auth/react", ...)`:

```tsx
const mockUsePathname = vi.fn();
const mockUseSearchParams = vi.fn();
vi.mock("next/navigation", () => ({
  usePathname: () => mockUsePathname(),
  useSearchParams: () => mockUseSearchParams(),
}));
```

In the `beforeEach`, add resets and a default (bare-path, no query) return value so every test not specifically about the callbackUrl keeps working unchanged:

```tsx
  beforeEach(() => {
    mockUseSession.mockReset();
    mockUsePathname.mockReset();
    mockUseSearchParams.mockReset();
    mockUsePathname.mockReturnValue("/matches/m1");
    mockUseSearchParams.mockReturnValue(new URLSearchParams());
  });
```

Update the existing test's assertion (it currently hardcodes the pre-fix href):

```tsx
  it("shows a Sign in prompt instead of the log-bet control when unauthenticated", () => {
    mockUseSession.mockReturnValue({ status: "unauthenticated" });
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" />);
    expect(screen.getByRole("link", { name: /sign in/i })).toHaveAttribute("href", "/login?callbackUrl=%2Fmatches%2Fm1");
    expect(screen.queryByRole("button", { name: /log bet/i })).not.toBeInTheDocument();
  });
```

stays passing as-is against the new `beforeEach` default (bare pathname, empty search params) -- no change needed to this test itself, only confirm it still passes after Step 4.

Add the new test:

```tsx
it("preserves the current page's full URL (query params included) in the sign-in callbackUrl", () => {
  mockUseSession.mockReturnValue({ status: "unauthenticated" });
  mockUsePathname.mockReturnValue("/matches/560572");
  mockUseSearchParams.mockReturnValue(
    new URLSearchParams("home=Crystal+Palace&away=Ipswich+Town&date=2026-09-12&league=E0")
  );

  render(<LogBetButton matchId="560572" recommendation={recommendation} market="result_3way" selection="home" />);

  const link = screen.getByRole("link", { name: /sign in to log this bet/i });
  const callbackUrl = new URL(link.getAttribute("href")!, "http://localhost").searchParams.get("callbackUrl")!;
  expect(callbackUrl).toBe("/matches/560572?home=Crystal+Palace&away=Ipswich+Town&date=2026-09-12&league=E0");
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `npx vitest run components/__tests__/LogBetButton.test.tsx` (from `app/frontend/`)
Expected: FAIL -- the current href only contains `/matches/560572`, callbackUrl won't match; also likely fails to even mount cleanly since `next/navigation` isn't imported by the component yet (mocking an unused import is harmless, so this specific failure mode may not occur, but the assertion itself fails regardless).

- [ ] **Step 4: Fix `LogBetButton` to use the real current URL**

In `app/frontend/components/MatchUI.tsx`, add a new import line (this file does not import from `next/navigation` at all today):

```tsx
import { usePathname, useSearchParams } from "next/navigation";
```

Replace:

```tsx
  const { status } = useSession();
  const [open, setOpen] = useState(false);
  const [stake, setStake] = useState("");
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "done" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");

  if (status === "unauthenticated") {
    return (
      <Link href={`/login?callbackUrl=${encodeURIComponent(`/matches/${matchId}`)}`} className="text-xs font-medium text-accent">
        Sign in to log this bet
      </Link>
    );
  }
```

with:

```tsx
  const { status } = useSession();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [open, setOpen] = useState(false);
  const [stake, setStake] = useState("");
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "done" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");

  if (status === "unauthenticated") {
    // W215: the whole current URL (query params included), not a bare
    // `/matches/${matchId}` -- MatchAnalysisPage requires home/away/date to
    // render at all (see its own "Missing match details" guard), so a
    // truncated callbackUrl stranded a signed-in user on a dead page with
    // no way back to the bet they were trying to log. Both hooks are
    // nullable outside a real router context (confirmed empirically: they
    // don't throw, they return null) -- the `?? `/matches/${matchId}`` /
    // `searchParams?.toString()` guards keep this correct in that case
    // rather than only in the browser.
    const query = searchParams?.toString();
    const currentUrl = query ? `${pathname ?? `/matches/${matchId}`}?${query}` : pathname ?? `/matches/${matchId}`;
    return (
      <Link href={`/login?callbackUrl=${encodeURIComponent(currentUrl)}`} className="text-xs font-medium text-accent">
        Sign in to log this bet
      </Link>
    );
  }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `npx vitest run components/__tests__/LogBetButton.test.tsx`
Expected: PASS

- [ ] **Step 6: Run the full frontend suite to check for regressions**

Run: `npx vitest run` (from `app/frontend/`)
Expected: all pass. `MatchUI.tsx` had no `next/navigation` import before this task, and no other test file mocks it -- confirmed empirically (a standalone probe component calling real, unmocked `usePathname`/`useSearchParams` outside a router context) that both hooks simply return `null` rather than throwing, which is exactly why Step 4's code uses `searchParams?.toString()` and `pathname ?? `/matches/${matchId}`` -- with those guards in place, every other test file that renders `<LogBetButton>`/`<MatchCard>`/`<MatchAnalysisPage>` without mocking `next/navigation` keeps working unchanged (falls back to the bare `/matches/${matchId}` path, same as before this task). If the suite still shows a failure here, it means one of those two guards was dropped during implementation -- fix the component, not the tests.

- [ ] **Step 7: Commit**

```bash
git add app/frontend/components/MatchUI.tsx app/frontend/components/__tests__/LogBetButton.test.tsx
git commit -m "fix(app): W215 -- preserve match query params in the sign-in redirect

LogBetButton's unauthenticated Link hardcoded /login?callbackUrl=/matches/\${matchId},
dropping home/away/date/league -- MatchAnalysisPage requires all three to
render at all, so signing in from this link stranded the user on its
\"Missing match details\" dead end instead of back at the bet they were
trying to log. Now builds callbackUrl from the real current pathname +
query string (usePathname/useSearchParams).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Show the real outcome (and a link to Bet Tracker) after logging from a recommendation

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx` (`LogBetButton`)
- Test: `app/frontend/components/__tests__/LogBetButton.test.tsx`

**Problem:** `logBetFromRecommendation()` already returns the full `Bet` (including a real `outcome` if W212's auto-settle-on-log just resolved it), but `LogBetButton.submit()` discards the response and always shows a flat "Logged" with no link anywhere else to see it.

- [ ] **Step 1: Add an `@/lib/api` mock to the test file (not present yet) and write the failing tests**

`LogBetButton.test.tsx` currently has no mock of `@/lib/api` at all -- its two existing tests never actually call `submit()`. Add, alongside the existing `mockUseSession`/`next-auth/react` mock (real `ApiError` subclass, matching `BetTracker.fixtureError.test.tsx`'s established factory-mock pattern for this exact need):

```tsx
vi.mock("@/lib/api", () => ({
  logBetFromRecommendation: vi.fn(),
  ApiError: class ApiError extends Error {
    status?: number;
    constructor(message: string, status?: number) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  },
}));
```

Add `import { logBetFromRecommendation } from "@/lib/api";` and `import userEvent from "@testing-library/user-event";` to this file's imports, and reset the mock in `beforeEach` alongside the others: `vi.mocked(logBetFromRecommendation).mockReset();`.

Then write the failing tests:

```tsx
it("shows the real settled outcome when the bet auto-settles immediately, with a link to Bet Tracker", async () => {
  mockUseSession.mockReturnValue({ status: "authenticated", data: { user: { email: "a@b.com" } } });
  vi.mocked(logBetFromRecommendation).mockResolvedValue({
    id: 1, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
    market: "result_3way", selection: "home", odds: 2.1, stake: 10, outcome: "won",
    profit_loss: 12.1, source: "from_recommendation", recommendation_snapshot: null, created_at: "now",
  });
  const user = userEvent.setup();
  render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" />);

  await user.click(screen.getByRole("button", { name: "Log bet" }));
  await user.type(screen.getByPlaceholderText("Stake"), "10");
  await user.click(screen.getByRole("button", { name: "Confirm" }));

  expect(await screen.findByText(/won/i)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /view in bet tracker/i })).toHaveAttribute("href", "/bets");
});

it("still shows a plain 'Logged' (no outcome yet) when the bet stays open", async () => {
  mockUseSession.mockReturnValue({ status: "authenticated", data: { user: { email: "a@b.com" } } });
  vi.mocked(logBetFromRecommendation).mockResolvedValue({
    id: 2, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
    market: "result_3way", selection: "home", odds: 2.1, stake: 10, outcome: "open",
    profit_loss: null, source: "from_recommendation", recommendation_snapshot: null, created_at: "now",
  });
  const user = userEvent.setup();
  render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" />);

  await user.click(screen.getByRole("button", { name: "Log bet" }));
  await user.type(screen.getByPlaceholderText("Stake"), "10");
  await user.click(screen.getByRole("button", { name: "Confirm" }));

  expect(await screen.findByText("Logged")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /view in bet tracker/i })).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run components/__tests__/LogBetButton.test.tsx`
Expected: FAIL -- no "won" text, no "View in Bet Tracker" link exist yet.

- [ ] **Step 3: Implement**

In `LogBetButton`, add a state var to hold the settled bet and use it in the done-state render:

```tsx
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "done" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [loggedBet, setLoggedBet] = useState<Bet | null>(null);
```

(Import `Bet` from `@/lib/types` in this file's existing type-only import block if not already imported.)

```tsx
  async function submit() {
    const parsedStake = parseFloat(stake);
    if (!parsedStake || parsedStake <= 0) {
      setSaveStatus("error");
      setErrorMsg("Enter a stake greater than 0.");
      return;
    }
    setSaveStatus("saving");
    try {
      const bet = await logBetFromRecommendation({ match_id: matchId, recommendation, market, selection, stake: parsedStake });
      setLoggedBet(bet);
      setSaveStatus("done");
    } catch (err) {
      setSaveStatus("error");
      setErrorMsg(err instanceof ApiError ? err.message : "Could not log bet.");
    }
  }

  if (saveStatus === "done") {
    // W215: logBetFromRecommendation() already returns the settled outcome
    // (W212 may have auto-settled it immediately) -- show it instead of a
    // flat "Logged" that hides real information already in hand, and give
    // a way to see it in context instead of a dead end.
    return (
      <span className="flex items-center gap-2 text-xs">
        <span className="text-good">
          {loggedBet && loggedBet.outcome !== "open" ? `Logged -- ${loggedBet.outcome}` : "Logged"}
        </span>
        <Link href="/bets" className="font-medium text-accent">
          View in Bet Tracker
        </Link>
      </span>
    );
  }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run components/__tests__/LogBetButton.test.tsx`
Expected: PASS

- [ ] **Step 5: Run full frontend suite**

Run: `npx vitest run`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/components/MatchUI.tsx app/frontend/components/__tests__/LogBetButton.test.tsx
git commit -m "fix(app): W215 -- show the real settled outcome after logging from a recommendation

logBetFromRecommendation() already returns the full Bet (including a
real outcome if W212 auto-settled it immediately) -- the response was
being discarded in favor of a flat 'Logged'. Now shows 'Logged -- won'/
'Logged -- lost' when already settled, plus a 'View in Bet Tracker'
link either way (there was previously no path back to /bets at all
from this button).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Backend -- DELETE /api/bets/{id}

**Files:**
- Modify: `app/backend/bet_tracker.py`
- Modify: `app/backend/main.py`
- Test: `app/backend/tests/test_bet_tracker.py`
- Test: `app/backend/tests/test_bets_endpoints.py`

- [ ] **Step 1: Write the failing `BetTracker.delete_bet` test**

Add to `app/backend/tests/test_bet_tracker.py`:

```python
def test_delete_bet_removes_it(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )

    deleted = tracker.delete_bet(bet.id)

    assert deleted is True
    assert tracker.get_bet(bet.id) is None


def test_delete_bet_returns_false_for_a_nonexistent_id(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    assert tracker.delete_bet(999) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest app/backend/tests/test_bet_tracker.py -k delete_bet -v`
Expected: FAIL with `AttributeError: 'BetTracker' object has no attribute 'delete_bet'`

- [ ] **Step 3: Implement `delete_bet`**

In `app/backend/bet_tracker.py`, add after `settle_bet`:

```python
    def delete_bet(self, bet_id: int) -> bool:
        """Returns True if a row was actually deleted, False if bet_id
        didn't exist. Ownership is the caller's responsibility (main.py's
        route checks bet.user_id before calling this) -- this method itself
        has no notion of "whose" bet it is."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM user_bets WHERE id = ?", (bet_id,))
        return cursor.rowcount > 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest app/backend/tests/test_bet_tracker.py -k delete_bet -v`
Expected: PASS

- [ ] **Step 5: Write the failing endpoint test**

Add to `app/backend/tests/test_bets_endpoints.py` (reuses this file's existing `_override_tracker`/`_override_user` helpers and its `_no_real_settlement_network_calls` autouse fixture):

```python
def test_delete_bet_endpoint_removes_the_callers_own_bet(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
    user = _override_user(tmp_path)
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None, user_id=user.id,
    )
    try:
        with TestClient(app) as client:
            response = client.delete(f"/api/bets/{bet.id}")
        assert response.status_code == 204
        assert tracker.get_bet(bet.id) is None
    finally:
        app.dependency_overrides.clear()


def test_delete_bet_endpoint_404s_for_a_bet_owned_by_someone_else(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
    _override_user(tmp_path)  # authenticates as this user
    other_users_bet = tracker.create_bet(
        match_id="m2", date="2026-08-23", home_team="Chelsea", away_team="Fulham",
        market="btts", selection="yes", odds=1.9, stake=5.0,
        source="manual", recommendation_snapshot=None, user_id=999,  # a different user
    )
    try:
        with TestClient(app) as client:
            response = client.delete(f"/api/bets/{other_users_bet.id}")
        assert response.status_code == 404
        # Not actually deleted -- confirms this 404s before ever calling delete_bet.
        assert tracker.get_bet(other_users_bet.id) is not None
    finally:
        app.dependency_overrides.clear()


def test_delete_bet_endpoint_404s_for_a_nonexistent_id(tmp_path: Path):
    _override_tracker(tmp_path)
    _override_user(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.delete("/api/bets/999999")
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_delete_bet_endpoint_401s_without_the_internal_secret(tmp_path: Path):
    _override_tracker(tmp_path)
    with TestClient(app) as client:
        response = client.delete("/api/bets/1")
    assert response.status_code == 401
```

- [ ] **Step 6: Run test to verify it fails**

Run: `./venv/bin/python -m pytest app/backend/tests/test_bets_endpoints.py -k delete_bet -v`
Expected: FAIL (404 Not Found for the route itself -- `DELETE /api/bets/{id}` doesn't exist yet).

- [ ] **Step 7: Implement the route**

In `app/backend/main.py`, add right after the `settle_open` route (find `@app.post("/api/bets/settle-open")` and its full function body, insert after it):

```python
@app.delete("/api/bets/{bet_id}", status_code=204)
async def delete_bet(
    bet_id: int,
    tracker: BetTracker = Depends(bets.get_bet_tracker),
    user_email: str = Depends(get_current_user_email),
    user_store: UserStore = Depends(get_user_store),
) -> None:
    """W215: direct user feedback -- a mis-logged bet (typo'd stake,
    accidental duplicate) had no way to be corrected except a direct
    database edit. 404s (not 403) for a bet owned by someone else, same as
    a nonexistent id -- doesn't confirm to a caller that a given bet_id
    exists at all if it isn't theirs."""
    user = user_store.get_or_create(user_email)
    bet = tracker.get_bet(bet_id)
    if bet is None or bet.user_id != user.id:
        raise HTTPException(status_code=404, detail="Bet not found.")
    tracker.delete_bet(bet_id)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `./venv/bin/python -m pytest app/backend/tests/test_bets_endpoints.py app/backend/tests/test_bet_tracker.py -q`
Expected: all pass.

- [ ] **Step 9: Run the full backend suite**

Run: `./venv/bin/python -m pytest app/backend -q`
Expected: same pre-existing 5 `test_fixtures_endpoint.py` failures as always (documented, unrelated), zero new failures.

- [ ] **Step 10: Commit**

```bash
git add app/backend/bet_tracker.py app/backend/main.py app/backend/tests/test_bet_tracker.py app/backend/tests/test_bets_endpoints.py
git commit -m "feat(app): W215 -- DELETE /api/bets/{id}, scoped to the owning user

A mis-logged bet (typo'd stake, accidental duplicate) had no way to be
corrected except a direct database edit. New BetTracker.delete_bet()
+ DELETE /api/bets/{id} route, 404ing (not 403) for a bet that doesn't
exist or belongs to someone else -- doesn't confirm existence to a
non-owner either way.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Frontend -- wire delete into the Bet Tracker's bet list

**Files:**
- Create: `app/frontend/app/api/bets/[id]/route.ts`
- Modify: `app/frontend/lib/api.ts`
- Modify: `app/frontend/components/BetTracker.tsx` (`BetRow`, the bets-list header, `BetTrackerPage`)
- Test: `app/frontend/components/__tests__/BetTracker.fixtureError.test.tsx`

Depends on Task 3 (the backend route must exist).

- [ ] **Step 1: Create the proxy route**

```ts
// app/frontend/app/api/bets/[id]/route.ts
import { forwardToBackend } from "@/lib/backendProxy";

export async function DELETE(request: Request, { params }: { params: { id: string } }) {
  const response = await forwardToBackend(`/api/bets/${params.id}`, { method: "DELETE" });
  return new Response(response.status === 204 ? null : await response.text(), { status: response.status });
}
```

- [ ] **Step 2: Add `deleteBet` to `lib/api.ts`**

```ts
/** W215: removes a logged bet -- 404s (via ApiError) if it isn't the
 * caller's own bet or doesn't exist. */
export async function deleteBet(id: number): Promise<void> {
  const response = await fetch(`/api/bets/${id}`, { method: "DELETE" });
  if (!response.ok) {
    throw new ApiError(`Failed to delete bet (${response.status})`, response.status);
  }
}
```

- [ ] **Step 3: Write the failing frontend test**

Add to `app/frontend/components/__tests__/BetTracker.fixtureError.test.tsx` (it already has the full `vi.mock("@/lib/api", ...)` factory with a real `ApiError` class -- add `deleteBet: vi.fn()` to that factory's returned object, and import it alongside the other named imports at the top):

```tsx
describe("BetTrackerPage lets a user delete their own logged bet (W215)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getBets).mockReset();
    vi.mocked(getBetStats).mockReset();
    vi.mocked(getStatus).mockReset();
    vi.mocked(settleOpenBets).mockReset();
    vi.mocked(deleteBet).mockReset();
    vi.mocked(getFixtures).mockResolvedValue([]);
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    vi.mocked(settleOpenBets).mockResolvedValue([]);
  });

  const bet = {
    id: 1, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
    market: "result_3way", selection: "home", odds: 2.1, stake: 10, outcome: "open" as const,
    profit_loss: null, source: "manual" as const, recommendation_snapshot: null, created_at: "now",
  };

  it("shows a two-step confirm, and removes the row on confirm", async () => {
    vi.mocked(getBets).mockResolvedValueOnce([bet]).mockResolvedValueOnce([]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 0, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 0, total_profit: 0, max_drawdown: 0, starting_bankroll: 0, current_bankroll: 0,
    });
    vi.mocked(deleteBet).mockResolvedValue(undefined);
    const user = userEvent.setup();

    render(<BetTrackerPage />);
    await screen.findByText(/Arsenal v Everton/);

    await user.click(screen.getByRole("button", { name: /delete/i }));
    expect(screen.getByText(/delete this bet\?/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^yes$/i }));

    expect(deleteBet).toHaveBeenCalledWith(1);
    await waitFor(() => expect(screen.queryByText(/Arsenal v Everton/)).not.toBeInTheDocument());
  });

  it("clicking Cancel on the confirm step leaves the bet in place", async () => {
    vi.mocked(getBets).mockResolvedValue([bet]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 0, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 0, total_profit: 0, max_drawdown: 0, starting_bankroll: 0, current_bankroll: 0,
    });
    const user = userEvent.setup();

    render(<BetTrackerPage />);
    await screen.findByText(/Arsenal v Everton/);

    await user.click(screen.getByRole("button", { name: /delete/i }));
    await user.click(screen.getByRole("button", { name: /cancel/i }));

    expect(deleteBet).not.toHaveBeenCalled();
    expect(screen.getByText(/Arsenal v Everton/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 4: Run test to verify it fails**

Run: `npx vitest run components/__tests__/BetTracker.fixtureError.test.tsx`
Expected: FAIL -- no "Delete" button exists in `BetRow` yet.

- [ ] **Step 5: Implement -- `BetRow` gains a delete affordance with a 2-step confirm**

In `app/frontend/components/BetTracker.tsx`, add `deleteBet` to the existing `@/lib/api` import line, and change `BetRow` to accept an `onDeleted` callback:

```tsx
function BetRow({ bet, onDeleted }: { bet: Bet; onDeleted: () => void }) {
  const outcomeColor = bet.outcome === "won" ? "text-good" : bet.outcome === "lost" ? "text-serious" : "text-muted";
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);

  async function confirmDelete() {
    setDeleting(true);
    try {
      await deleteBet(bet.id);
      onDeleted();
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="grid grid-cols-[1fr_auto_auto_auto_auto_auto] items-center gap-4 border-b border-border py-3 text-sm last:border-b-0">
      <span className="truncate text-ink">
        {bet.home_team} v {bet.away_team}
        <span className="ml-2 text-xs text-ink-secondary">
          {bet.market} · {bet.selection}
        </span>
      </span>
      <span className="text-right font-mono text-ink-secondary">{bet.odds.toFixed(2)}</span>
      <span className="text-right font-mono text-ink-secondary">{bet.stake.toFixed(2)}</span>
      <span className="text-right font-mono text-ink">
        {bet.profit_loss !== null ? bet.profit_loss.toFixed(2) : "—"}
      </span>
      <span className={`justify-self-end uppercase text-xs font-medium ${outcomeColor}`}>{bet.outcome}</span>
      <span className="justify-self-end text-xs">
        {confirming ? (
          <span className="flex items-center gap-1.5">
            <span className="text-ink-secondary">Delete this bet?</span>
            <button type="button" onClick={confirmDelete} disabled={deleting} className="font-medium text-serious disabled:opacity-50">
              {deleting ? "…" : "Yes"}
            </button>
            <button type="button" onClick={() => setConfirming(false)} className="text-ink-secondary">
              Cancel
            </button>
          </span>
        ) : (
          <button type="button" onClick={() => setConfirming(true)} className="text-ink-secondary hover:text-serious">
            Delete
          </button>
        )}
      </span>
    </div>
  );
}
```

Update the header row's grid to match the new 6-column layout, and pass `onDeleted={load}` through:

```tsx
            <div className="grid grid-cols-[1fr_auto_auto_auto_auto_auto] gap-4 border-b border-border pb-1.5 text-xs font-medium uppercase tracking-wide text-muted">
              <span>Match</span>
              <span className="text-right">Odds</span>
              <span className="text-right">Stake</span>
              <span className="text-right">P&amp;L</span>
              <span className="text-right">Outcome</span>
              <span />
            </div>
            {bets.map((bet) => (
              <BetRow key={bet.id} bet={bet} onDeleted={load} />
            ))}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `npx vitest run components/__tests__/BetTracker.fixtureError.test.tsx`
Expected: PASS

- [ ] **Step 7: Run full frontend suite + tsc + build**

Run (from `app/frontend/`): `npx vitest run && npx tsc --noEmit && npx next build`
Expected: all pass, 0 type errors, build exits 0.

- [ ] **Step 8: Commit**

```bash
git add app/frontend/app/api/bets/\[id\]/route.ts app/frontend/lib/api.ts app/frontend/components/BetTracker.tsx app/frontend/components/__tests__/BetTracker.fixtureError.test.tsx
git commit -m "feat(app): W215 -- delete a logged bet from the Bet Tracker

Wires the new DELETE /api/bets/{id} backend route (previous commit)
into BetRow: a 'Delete' link expands to a 'Delete this bet? Yes /
Cancel' inline confirm (matching this file's existing inline-confirm
style, not a native window.confirm), removing the row and refreshing
stats on confirm.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Add `Match.rawRecommendation` (needed by Task 6)

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx` (`Match` type, `applyRecommendation`, `MatchCard.handleExpand`)
- Test: `app/frontend/components/__tests__/MatchUI.test.tsx`

**Why:** `LogBetButton` requires the full raw `MatchRecommendationOut` snapshot, which `applyRecommendation()` currently flattens into `candidates`/`recommendationPick` and discards. Task 6's MatchCard quick-log button needs it back.

- [ ] **Step 1: Write the failing test**

Add near `MatchUI.test.tsx`'s existing `applyRecommendation`-adjacent tests (search the file for `"applyRecommendation"` or a `Match` fixture builder to match its existing style):

```tsx
it("W215: applyRecommendation carries the raw recommendation through on Match.rawRecommendation", () => {
  const rec: MatchRecommendationOut = {
    match: {}, overall: "direct_bet", candidates: [], recommendation_pick: null, explanation: [],
    confidence: "high", limitations: [], prediction_basis: "team_history_and_market",
    invalid_market_count: 0, cold_start_risk: false, feature_completeness: 0.9, unknown_team: false,
  };
  const base = fixtureToMatch({
    match_id: "m1", utc_date: "2026-08-22T15:00:00Z", status: "SCHEDULED",
    home_team: "Arsenal", away_team: "Everton", home_goals: null, away_goals: null,
  });

  const result = applyRecommendation(base, rec);

  expect(result.rawRecommendation).toBe(rec);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run components/__tests__/MatchUI.test.tsx -t "rawRecommendation"`
Expected: FAIL -- `result.rawRecommendation` is `undefined`, property doesn't exist on the type.

- [ ] **Step 3: Add the field to `Match` and populate it in both places a `Match` gets a recommendation applied**

In `Match`'s type definition, add after `recommendationPick`:

```tsx
  recommendationPick: RecommendationPick | null;
  // W215: the untouched recommendation snapshot -- needed to log a bet
  // directly from a MatchCard/LogBetButton without re-fetching it. null
  // until a recommendation has actually been applied (fixtureToMatch's
  // bare-fixture construction never sets this).
  rawRecommendation?: MatchRecommendationOut | null;
```

In `fixtureToMatch()`'s return object, add `rawRecommendation: null,` alongside the other `recommendationPick: null,` line.

In `applyRecommendation()`, add `rawRecommendation: rec,` alongside `recommendationPick: rec.recommendation_pick,`.

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run components/__tests__/MatchUI.test.tsx -t "rawRecommendation"`
Expected: PASS

- [ ] **Step 5: Run full frontend suite + tsc**

Run: `npx vitest run && npx tsc --noEmit`
Expected: all pass. Deliberately made optional (`rawRecommendation?:`), not required -- at least 6 existing test files (`DashboardRail.test.tsx`, `MatchUI.completedRedesign.test.tsx`, `MatchUI.hitMiss.test.tsx`, `MatchUI.dateboundary.test.tsx`, `MatchUI.test.tsx`, `lib/dashboardMetrics.test.ts`) construct `Match` object literals by hand without going through `fixtureToMatch`/`applyRecommendation`; a required field would force edits to all of them for a property none of them care about. `tsc --noEmit` must still be clean with zero of those files touched.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/components/MatchUI.tsx app/frontend/components/__tests__/MatchUI.test.tsx
git commit -m "feat(app): W215 -- carry the raw recommendation snapshot on Match.rawRecommendation

applyRecommendation() flattened the fetched MatchRecommendationOut into
candidates/recommendationPick and discarded the original object --
LogBetButton needs the untouched snapshot, and MatchCard (next commit)
needs to be able to log a bet without re-fetching it.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: MatchCard quick-log control (Dashboard + Match Explorer), direct_bet only

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx` (`MatchCard`)
- Test: `app/frontend/components/__tests__/MatchUI.test.tsx`

Depends on Task 5.

**Placement:** `MatchCard`'s entire face (team names, market/pick/odds/edge grid) is one big `<button onClick={handleExpand}>` -- a nested interactive control there would be invalid HTML and would also trigger the card's own expand/collapse. The quick-log control goes inside the *expanded* section instead (the `expand-rows` div, a sibling of that button, already only visible when `open` is true) -- reuses the existing `LogBetButton` component verbatim, right after the explanation bullets.

- [ ] **Step 1: Write the failing test**

Add near `MatchUI.test.tsx`'s existing `MatchCard` tests (search for `"MatchCard"` to find its describe block and existing fixture-building helpers):

```tsx
describe("MatchCard quick-log control (W215)", () => {
  it("shows a Log bet control for a direct_bet recommendation once expanded", async () => {
    const match = applyRecommendation(baseMatchFixture(), {
      match: {}, overall: "direct_bet", confidence: "high", explanation: ["test"], limitations: [],
      prediction_basis: "team_history_and_market", invalid_market_count: 0, cold_start_risk: false,
      feature_completeness: 0.9, unknown_team: false,
      recommendation_pick: { market: "result_3way", selection: "home", reason: "test" },
      candidates: [{
        market: "result_3way", selection: "home", recommendation_type: "direct_bet",
        current_odds: 2.1, min_odds: 1.5, ml_probability: 0.55, implied_probability: 0.48, value_edge: 0.07,
      }],
    });
    const user = userEvent.setup();
    render(<MatchCard match={match} onUpdate={() => {}} />);

    await user.click(screen.getByRole("button", { name: /arsenal.*everton/i }));

    expect(screen.getByRole("button", { name: "Log bet" })).toBeInTheDocument();
  });

  it("shows no quick-log control for a conditional recommendation", async () => {
    const match = applyRecommendation(baseMatchFixture(), {
      match: {}, overall: "conditional", confidence: "high", explanation: ["test"], limitations: [],
      prediction_basis: "team_history_and_market", invalid_market_count: 0, cold_start_risk: false,
      feature_completeness: 0.9, unknown_team: false,
      recommendation_pick: { market: "result_3way", selection: "home", reason: "test" },
      candidates: [{
        market: "result_3way", selection: "home", recommendation_type: "conditional",
        current_odds: 1.8, min_odds: 2.0, ml_probability: 0.55, implied_probability: 0.55, value_edge: -0.02,
        target_odds: 2.05,
      }],
    });
    const user = userEvent.setup();
    render(<MatchCard match={match} onUpdate={() => {}} />);

    await user.click(screen.getByRole("button", { name: /arsenal.*everton/i }));

    expect(screen.queryByRole("button", { name: "Log bet" })).not.toBeInTheDocument();
  });
});
```

(If this file has no `baseMatchFixture()` helper already, build the `Match` inline the same way its other `MatchCard` tests already do -- check the file first; don't introduce a second, differently-shaped fixture helper.)

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run components/__tests__/MatchUI.test.tsx -t "quick-log"`
Expected: FAIL -- no "Log bet" button renders anywhere in the expanded section yet.

- [ ] **Step 3: Implement**

In `MatchCard`, inside the `expand-rows` div, right after the `explanation` bullet list's closing `</ul>` and its `invalidMarketCount` warning block (the same spot `match.explanation.map(...)` renders), add:

```tsx
                {/* W215: quick-log the card's own resolved pick, direct_bet
                    only -- a conditional/no_bet pick isn't something the
                    agent is actually recommending you act on yet, and this
                    is a one-click shortcut, not the full multi-market
                    picker ProbabilityRow's per-row LogBetButton already
                    provides on the detail page. Reuses LogBetButton as-is. */}
                {shown?.recommendationType === "direct_bet" && match.rawRecommendation && (
                  <div className="mt-3 border-t border-border pt-3">
                    <LogBetButton
                      matchId={match.id}
                      recommendation={match.rawRecommendation}
                      market={shown.market}
                      selection={shown.selection}
                    />
                  </div>
                )}
```

Also thread `rawRecommendation` through `MatchCard.handleExpand()`'s own inline fetch (the `onUpdate(applyRecommendation(match, rec))` call already does this correctly since `applyRecommendation` was fixed in Task 5 -- no change needed here beyond confirming it, since `handleExpand` already calls `applyRecommendation`).

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run components/__tests__/MatchUI.test.tsx -t "quick-log"`
Expected: PASS

- [ ] **Step 5: Run full frontend suite + tsc + build**

Run: `npx vitest run && npx tsc --noEmit && npx next build`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/components/MatchUI.tsx app/frontend/components/__tests__/MatchUI.test.tsx
git commit -m "feat(app): W215 -- quick-log a direct_bet recommendation straight from a match card

Direct user request: a log option on match cards themselves (Dashboard
and Match Explorer both render MatchCard, so one change covers both),
not just the detail page's per-market table. Reuses LogBetButton
verbatim inside the card's existing expand-on-click section (the whole
card face is already one <button>, so a nested control has to live in
the sibling expanded area, not inside it). direct_bet only, per direct
instruction -- conditional/no_bet/insufficient_data show nothing extra
here; the full per-market picker on the detail page is unaffected and
still covers every market/recommendation type.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Duplicate-bet warning on the match detail page

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx` (`MatchAnalysisPage`, `ProbabilityRow`, `LogBetButton`)
- Test: `app/frontend/components/__tests__/MatchUI.test.tsx`

**Design:** `MatchAnalysisPage` fetches the current user's bets once (`getBets()`, already used elsewhere, returns every bet for the signed-in user -- unauthenticated visitors never see this fetch attempted, matching `LogBetButton`'s own auth-gating) and computes which `(market, selection)` pairs already have a logged bet for this exact `match_id`. `ProbabilityRow` receives that as a plain `boolean` per row and renders a small note instead of hiding the button -- logging a second bet on the same market is still allowed (e.g. a genuinely different real-world wager), just no longer silent.

**Real risk in this file specifically, read before starting:** `MatchUI.test.tsx`'s `vi.mock("@/lib/api", ...)` factory (top of file) has no `getBets` export yet -- adding this feature means adding `getBets: vi.fn()` there. This file's `useSession` mock (from `next-auth/react`, this file's own separate mock, distinct from `LogBetButton.test.tsx`'s) defaults to `"unauthenticated"`, and this new `getBets()` call only fires when `status === "authenticated"` -- but at least one existing test already overrides that (`"W210 follow-up (W115 re-enable): renders a Log bet control..."`, ~line 1013, and the `describe("LogBetButton ...")` block's own `beforeEach`, ~line 1059) and renders `MatchAnalysisPage`/`LogBetButton` without ever mocking `getBets`. An unmocked `vi.fn()` returns `undefined`, and calling `.then()` on it throws. This file's own header comment already documents this exact class of gap once before (search for "Tasks 7/8 already hit and fixed the same way" near the `getStatus`/`getSandboxStatus` mock additions) -- same fix here: after adding the mock and your new test, **run this file's full suite**, and for every test that newly fails this way, add `vi.mocked(getBets).mockResolvedValue([])` to that describe block's `beforeEach` (or inline before `render(...)` if the block has no `beforeEach`).

- [ ] **Step 1: Add `getBets` to this file's `@/lib/api` mock, and write the failing test**

Add `getBets: vi.fn(),` to the `vi.mock("@/lib/api", () => ({ ... }))` factory near the top of the file, and add `getBets` to the `import { ... } from "@/lib/api"` (actually `"../MatchUI"`'s sibling import block -- check which import statement this file uses for API mocks vs. component exports and add it to the correct one; `getCachedRecommendation`/`getFixtures` etc. are imported directly from `@/lib/api` in this file's existing mock-consumption imports).

```tsx
it("W215: shows an 'Already logged' note next to a market/selection the user already has a bet on", async () => {
  vi.mocked(useSession).mockReturnValue({ data: { user: {} }, status: "authenticated" } as never);
  vi.mocked(getBets).mockResolvedValue([
    {
      id: 1, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
      market: "result_3way", selection: "home", odds: 2.1, stake: 10, outcome: "open",
      profit_loss: null, source: "manual", recommendation_snapshot: null, created_at: "now",
    },
  ]);
  vi.mocked(getCachedRecommendation).mockResolvedValue(
    makeRecommendation({
      overall: "direct_bet",
      candidates: [
        { market: "result_3way", selection: "home", recommendation_type: "direct_bet", current_odds: 2.1, min_odds: 0, ml_probability: 0.6, implied_probability: 0.48, value_edge: 0.12 },
        { market: "result_3way", selection: "draw", recommendation_type: "no_bet", current_odds: 3.2, min_odds: 0, ml_probability: 0.25, implied_probability: 0.31, value_edge: -0.06 },
      ],
      recommendation_pick: { market: "result_3way", selection: "home" },
    })
  );

  render(<MatchAnalysisPage id="m1" home="Arsenal" away="Everton" date="2026-08-22" />);

  const homeRow = (await screen.findByText("result_3way · home")).closest("div")!;
  expect(within(homeRow).getByText(/already logged/i)).toBeInTheDocument();
  // A different market/selection on the same match is unaffected.
  const drawRow = screen.getByText("result_3way · draw").closest("div")!;
  expect(within(drawRow).queryByText(/already logged/i)).not.toBeInTheDocument();
});
```

(Use this file's existing `makeRecommendation(...)` fixture helper -- confirm its exact name/shape by reading how the `"W210 follow-up (W115 re-enable)"` test above builds its recommendation, and match it exactly rather than guessing. Add `within` to this file's `@testing-library/react` import if not already there.)

- [ ] **Step 2: Run test to verify it fails, then run the whole file and fix the fallout described above**

Run: `npx vitest run components/__tests__/MatchUI.test.tsx -t "Already logged"`
Expected: FAIL -- no such text renders yet.

Then run: `npx vitest run components/__tests__/MatchUI.test.tsx`
Expected: some pre-existing tests in this file now fail with a `getBets`/`.then` error (see the risk note above) until you add `vi.mocked(getBets).mockResolvedValue([])` wherever needed -- do that now, before continuing, and re-run until this file is green except for the one new intentionally-failing test.

- [ ] **Step 3: Implement**

In `MatchAnalysisPage`, add a `useSession()` check and a `getBets()` fetch, computing a `Set<string>` of `"${market}::${selection}"` keys already logged for this match:

```tsx
  const { status } = useSession();
  const [loggedKeys, setLoggedKeys] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (status !== "authenticated") {
      setLoggedKeys(new Set());
      return;
    }
    let cancelled = false;
    getBets()
      .then((allBets) => {
        if (cancelled) return;
        const keys = allBets.filter((b) => b.match_id === id).map((b) => `${b.market}::${b.selection}`);
        setLoggedKeys(new Set(keys));
      })
      .catch(() => {
        // Best-effort -- a failure here just means no "Already logged" note
        // shows, never blocks the page or logging a new bet.
      });
    return () => {
      cancelled = true;
    };
  }, [id, status]);
```

Import `useSession` from `next-auth/react` and `getBets` from `@/lib/api` in this file's existing import blocks if not already imported (`useSession` is already imported for `LogBetButton`; reuse that line).

Pass `alreadyLogged` down to each row:

```tsx
              match.candidates.map((m, i) => (
                <ProbabilityRow
                  key={`${m.market}-${i}`}
                  m={m}
                  matchId={id}
                  recommendation={rawRecommendation ?? undefined}
                  alreadyLogged={loggedKeys.has(`${m.market}::${m.selection}`)}
                />
              ))
```

In `ProbabilityRow`, accept and render it:

```tsx
function ProbabilityRow({
  m,
  matchId,
  recommendation,
  alreadyLogged = false,
}: {
  m: MarketRec;
  matchId?: string;
  recommendation?: MatchRecommendationOut;
  alreadyLogged?: boolean;
}) {
  ...
        {matchId && recommendation && !anomalous && (
          <span className="flex items-center gap-1.5">
            <LogBetButton matchId={matchId} recommendation={recommendation} market={m.market} selection={m.selection} />
            {alreadyLogged && <span className="text-[10px] text-muted">(already logged)</span>}
          </span>
        )}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run components/__tests__/MatchUI.test.tsx -t "Already logged"`
Expected: PASS

- [ ] **Step 5: Run full frontend suite + tsc + build**

Run: `npx vitest run && npx tsc --noEmit && npx next build`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/components/MatchUI.tsx app/frontend/components/__tests__/MatchUI.test.tsx
git commit -m "feat(app): W215 -- warn (not block) when a market/selection is already logged

MatchAnalysisPage now fetches the signed-in user's bets once (best-
effort -- a failure never blocks the page or logging) and flags any
market/selection on this match that already has a logged bet with a
small '(already logged)' note next to that row's LogBetButton.
Logging a second bet on the same market is still allowed (a genuinely
different real-world wager is plausible) -- this closes the silent
duplicate gap without adding a hard block.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 8: LogBetButton polish -- Cancel + restate terms before committing

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx` (`LogBetButton`)
- Test: `app/frontend/components/__tests__/LogBetButton.test.tsx`

- [ ] **Step 1: Write the failing tests**

```tsx
it("shows a Cancel control that collapses back to the plain Log bet link without submitting", async () => {
  mockUseSession.mockReturnValue({ status: "authenticated", data: { user: { email: "a@b.com" } } });
  const user = userEvent.setup();
  render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" />);

  await user.click(screen.getByRole("button", { name: "Log bet" }));
  expect(screen.getByPlaceholderText("Stake")).toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "Cancel" }));

  expect(screen.queryByPlaceholderText("Stake")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Log bet" })).toBeInTheDocument();
  expect(logBetFromRecommendation).not.toHaveBeenCalled();
});

it("restates the market/selection/odds being logged next to the stake input", async () => {
  mockUseSession.mockReturnValue({ status: "authenticated", data: { user: { email: "a@b.com" } } });
  const recommendationWithOdds = { ...recommendation, candidates: [
    { market: "result_3way", selection: "home", recommendation_type: "direct_bet", current_odds: 2.35 },
  ]};
  const user = userEvent.setup();
  render(<LogBetButton matchId="m1" recommendation={recommendationWithOdds} market="result_3way" selection="home" />);

  await user.click(screen.getByRole("button", { name: "Log bet" }));

  expect(screen.getByText(/home @ 2\.35/i)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run components/__tests__/LogBetButton.test.tsx -t "Cancel"`
Expected: FAIL -- no Cancel button, no restated terms.

- [ ] **Step 3: Implement**

```tsx
  if (!open) {
    return (
      <button type="button" onClick={() => setOpen(true)} className="text-xs font-medium text-accent">
        Log bet
      </button>
    );
  }

  // W215: restate exactly what's about to be logged -- the odds column is
  // several cells away in ProbabilityRow's grid (or absent entirely on
  // MatchCard's quick-log path, Task 6), so nothing here previously
  // confirmed the actual terms at the point of commitment.
  const matchedCandidate = recommendation.candidates.find((c) => c.market === market && c.selection === selection);
  const oddsLabel = matchedCandidate?.current_odds != null ? ` @ ${matchedCandidate.current_odds.toFixed(2)}` : "";

  return (
    <span className="flex flex-wrap items-center gap-1.5">
      <span className="text-xs text-ink-secondary">
        {selection}{oddsLabel}
      </span>
      <input
        value={stake}
        onChange={(e) => setStake(e.target.value)}
        placeholder="Stake"
        inputMode="decimal"
        className="w-16 rounded border border-border bg-surface px-1.5 py-0.5 text-xs text-ink outline-none focus:border-accent"
      />
      <button
        type="button"
        onClick={submit}
        disabled={saveStatus === "saving"}
        className="text-xs font-medium text-accent disabled:opacity-50"
      >
        {saveStatus === "saving" ? "…" : "Confirm"}
      </button>
      <button type="button" onClick={() => setOpen(false)} disabled={saveStatus === "saving"} className="text-xs text-ink-secondary disabled:opacity-50">
        Cancel
      </button>
      {saveStatus === "error" && <span className="text-xs text-serious">{errorMsg}</span>}
    </span>
  );
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run components/__tests__/LogBetButton.test.tsx`
Expected: PASS

- [ ] **Step 5: Run full frontend suite + tsc + build**

Run: `npx vitest run && npx tsc --noEmit && npx next build`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/components/MatchUI.tsx app/frontend/components/__tests__/LogBetButton.test.tsx
git commit -m "fix(app): W215 -- LogBetButton gets a Cancel and restates terms before committing

Two minor UX audit findings: (1) once expanded there was no way to
collapse back without submitting or leaving the page; (2) the
stake/Confirm UI never repeated the selection/odds being logged, only
inferable from the row several cells away. Both fixed in place.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 9: ManualBetForm search results show a league tag

**Files:**
- Modify: `app/frontend/components/BetTracker.tsx` (`ManualBetForm`)
- Test: `app/frontend/components/__tests__/BetTracker.fixtureError.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
it("W215: each search result shows its league, not just team names and date", async () => {
  vi.mocked(getFixtures).mockResolvedValue([
    {
      match_id: "1", utc_date: "2026-08-22T15:00:00Z", status: "SCHEDULED",
      home_team: "Arsenal", away_team: "Everton", home_goals: null, away_goals: null, competition: "E0",
    },
  ]);
  const user = userEvent.setup();
  render(<BetTrackerPage />);

  await user.type(screen.getByPlaceholderText("Search a real fixture by team name…"), "Arsenal");

  expect(await screen.findByText("E0")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run components/__tests__/BetTracker.fixtureError.test.tsx -t "league"`
Expected: FAIL -- no league text renders in a search result today.

- [ ] **Step 3: Implement**

`Fixture` already carries an optional `competition` field (used elsewhere in this codebase, e.g. `MatchUI.tsx`'s `fixtureToMatch`). In `ManualBetForm`'s results list:

```tsx
                {results.map((f) => (
                  <button
                    key={f.match_id}
                    type="button"
                    onClick={() => setSelected(f)}
                    className="flex items-center gap-2 rounded-lg border border-border p-2 text-left text-sm text-ink hover:border-border-strong"
                  >
                    <TeamBadge name={f.home_team} />
                    {f.home_team} v {f.away_team}
                    <TeamBadge name={f.away_team} />
                    {f.competition && (
                      <span className="rounded border border-border px-1.5 py-0.5 text-[10px] uppercase text-ink-secondary">
                        {f.competition}
                      </span>
                    )}
                    <span className="ml-auto text-xs text-ink-secondary">{formatDate(f.utc_date)}</span>
                  </button>
                ))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run components/__tests__/BetTracker.fixtureError.test.tsx -t "league"`
Expected: PASS

- [ ] **Step 5: Run full frontend suite + tsc + build**

Run: `npx vitest run && npx tsc --noEmit && npx next build`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/components/BetTracker.tsx app/frontend/components/__tests__/BetTracker.fixtureError.test.tsx
git commit -m "fix(app): W215 -- ManualBetForm search results show a league tag

Minor UX audit finding: same-date, same-named fixtures across leagues
were indistinguishable in the search results list. Fixture already
carries competition -- just wasn't rendered here.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review Notes (for whoever executes this plan)

- **Spec coverage:** Task 1 = critical query-param-loss finding; Task 2 = outcome-visibility finding; Tasks 3-4 = delete/edit finding; Task 5-6 = the new MatchCard quick-log feature (direct_bet only, per direct instruction); Task 7 = duplicate-bet finding; Task 8 = the two minor LogBetButton findings (Cancel, restated terms); Task 9 = the minor league-tag finding. All 9 items from the audit are covered.
- **Ordering:** Tasks 1-2 and 8 touch the same component (`LogBetButton`) -- fine to run sequentially in the order listed (each is a separate, independently-committable diff on top of the last). Task 6 depends on Task 5. Task 4 depends on Task 3. Tasks 7 and 9 are independent of everything else and can run in any order relative to the rest.
- **After all 9 tasks:** dispatch a final whole-branch code reviewer comparing the finished set of changes against this plan and the original audit findings, then use `superpowers:finishing-a-development-branch`.
