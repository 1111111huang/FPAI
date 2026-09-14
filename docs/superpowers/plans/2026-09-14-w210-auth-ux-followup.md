# W210 Auth UX Follow-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every real gap found by the systematic audit of the W210 multi-user auth feature — a visible signed-in/sign-out UI, auth-aware failure states, a real bug fix in manual bet entry, and enough UI polish that re-enabling the hidden bet-tracking UI (W106/W115) is actually safe to do.

**Architecture:** No new backend work — every gap here is frontend-only (session UI, error messaging, form validation, and small a11y/UX fixes). The one exception is Task 5, which fixes a real pre-existing data-integrity bug (freeform market/selection text in the manual bet form can produce a permanently-unsettleable bet) by constraining the form to the exact values `src/agent/market_resolution.py`'s `RESOLVABLE_MARKETS`/`market_correct()` actually understand.

**Tech Stack:** Next.js 14 App Router, `next-auth/react` (`useSession`, `signOut`, already installed from W210), Tailwind (existing conventions), Vitest + Testing Library (existing frontend test stack).

---

## Valid market/selection values (ground truth for Task 5)

Confirmed directly from `src/agent/market_resolution.py`:

```
result_3way  -> "home" | "draw" | "away"
btts         -> "yes" | "no"
total_goals  -> "over_2.5" | "under_2.5"
total_corners -> "over_9.5" | "under_9.5"
```

Any other market, or a selection outside its market's set, can never resolve — `settlement.py`'s `RESOLVABLE_MARKETS` filter or `market_correct()`'s exact-match check silently excludes it forever.

---

### Task 1: Signed-in state + sign-out UI in `AppShell`

**Files:**
- Create: `app/frontend/components/UserMenu.tsx`
- Modify: `app/frontend/components/AppShell.tsx`
- Test: `app/frontend/components/__tests__/UserMenu.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// app/frontend/components/__tests__/UserMenu.test.tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { UserMenu } from "../UserMenu";

const mockUseSession = vi.fn();
const mockSignOut = vi.fn();
vi.mock("next-auth/react", () => ({
  useSession: () => mockUseSession(),
  signOut: (...args: unknown[]) => mockSignOut(...args),
}));

describe("UserMenu", () => {
  beforeEach(() => {
    mockUseSession.mockReset();
    mockSignOut.mockReset();
  });

  it("renders a Sign in link when unauthenticated", () => {
    mockUseSession.mockReturnValue({ data: null, status: "unauthenticated" });
    render(<UserMenu />);
    expect(screen.getByRole("link", { name: /sign in/i })).toHaveAttribute("href", "/login");
  });

  it("renders nothing while the session is loading", () => {
    mockUseSession.mockReturnValue({ data: null, status: "loading" });
    const { container } = render(<UserMenu />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the signed-in email and a working Sign out button", async () => {
    mockUseSession.mockReturnValue({
      data: { user: { email: "fpai.deploy@gmail.com" } },
      status: "authenticated",
    });
    render(<UserMenu />);
    expect(screen.getByText("fpai.deploy@gmail.com")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /sign out/i }));
    expect(mockSignOut).toHaveBeenCalledWith({ callbackUrl: "/" });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app/frontend && npx vitest run components/__tests__/UserMenu.test.tsx`
Expected: FAIL — `Cannot find module '../UserMenu'`

- [ ] **Step 3: Write minimal implementation**

```tsx
// app/frontend/components/UserMenu.tsx
"use client";

// W210 follow-up: the one piece of session UI this app had zero of --
// before this, a signed-in user had no way to see which account they were
// using or to sign out short of manually hitting /api/auth/signout. Lives
// in AppShell so it's present on every page, signed-in or not.
import Link from "next/link";
import { signOut, useSession } from "next-auth/react";

export function UserMenu() {
  const { data: session, status } = useSession();

  if (status === "loading") return null;

  if (status === "unauthenticated" || !session?.user?.email) {
    return (
      <Link href="/login" className="text-xs font-medium text-accent">
        Sign in
      </Link>
    );
  }

  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="truncate text-ink-secondary" title={session.user.email}>
        {session.user.email}
      </span>
      <button
        type="button"
        onClick={() => signOut({ callbackUrl: "/" })}
        className="font-medium text-accent"
      >
        Sign out
      </button>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app/frontend && npx vitest run components/__tests__/UserMenu.test.tsx`
Expected: 3 passed

- [ ] **Step 5: Wire it into `AppShell`**

In `app/frontend/components/AppShell.tsx`, add the import:

```tsx
import { UserMenu } from "@/components/UserMenu";
```

Place it at the bottom of the desktop `<aside>` (after the `<nav>` block, inside the same `<div>` that currently only wraps `brandBlock` + `<nav>` — read the surrounding JSX first, since it needs a `mt-auto`-style push-to-bottom sibling, not a nested addition to the existing `<div>`). Concretely, change:

```tsx
      <aside className="hidden shrink-0 flex-col px-5 py-6 lg:flex lg:h-screen lg:w-56 lg:border-r lg:border-border">
        <div>
          {brandBlock}
          <nav className="mt-6 flex flex-col gap-1 text-sm">
            {/* ...existing NAV_ITEMS.map... */}
          </nav>
        </div>
      </aside>
```

to:

```tsx
      <aside className="hidden shrink-0 flex-col px-5 py-6 lg:flex lg:h-screen lg:w-56 lg:border-r lg:border-border">
        <div>
          {brandBlock}
          <nav className="mt-6 flex flex-col gap-1 text-sm">
            {/* ...existing NAV_ITEMS.map, unchanged... */}
          </nav>
        </div>
        <div className="mt-auto pt-4">
          <UserMenu />
        </div>
      </aside>
```

(`lg:flex lg:flex-col` is already on `<aside>` itself, so `mt-auto` on this new child correctly pushes it to the bottom of the flex column — confirm this after editing by checking the `<aside>` className string still has `flex-col` in it.)

Also add it to the mobile drawer, inside the drawer's own `<div className="flex items-start justify-between">` block — right after the close (`X`) button, so it's visible whenever the drawer is open:

```tsx
            <div className="flex items-start justify-between">
              {brandBlock}
              <button type="button" onClick={() => setMenuOpen(false)} aria-label="Close menu" className="text-ink-secondary">
                <X size={20} />
              </button>
            </div>
            <div className="mt-4">
              <UserMenu />
            </div>
```

- [ ] **Step 6: Run the full frontend suite**

Run: `cd app/frontend && npx vitest run`
Expected: all prior tests still pass, plus the 3 new `UserMenu` tests. If `AppShell.test.tsx` breaks (it may now need `next-auth/react` mocked too, since `AppShell` renders `UserMenu`), add the same `vi.mock("next-auth/react", ...)` block used in this task's own test file to `AppShell.test.tsx`'s existing mocks, defaulting to `{ data: null, status: "unauthenticated" }` so existing assertions about nav links are unaffected.

- [ ] **Step 7: Commit**

```bash
git add app/frontend/components/UserMenu.tsx app/frontend/components/AppShell.tsx app/frontend/components/__tests__/UserMenu.test.tsx app/frontend/components/__tests__/AppShell.test.tsx
git commit -m "feat(auth): add signed-in/sign-out UI to AppShell (W210 follow-up)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Login page — honor `callbackUrl`, show rejection errors, skip if already signed in

**Files:**
- Modify: `app/frontend/app/login/page.tsx`
- Test: `app/frontend/app/login/__tests__/page.test.tsx` (new directory — check whether `app/frontend/app/**/__tests__/` is an existing convention or whether page tests live elsewhere in this codebase first; if `app/login/page.tsx` has no sibling test precedent, put the test next to the other page tests you find, matching whatever the closest existing App Router page test does)

- [ ] **Step 1: Write the failing test**

```tsx
// app/frontend/app/login/__tests__/page.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import LoginPage from "../page";

const mockUseSession = vi.fn();
const mockSignIn = vi.fn();
const mockUseSearchParams = vi.fn();
const mockPush = vi.fn();

vi.mock("next-auth/react", () => ({
  useSession: () => mockUseSession(),
  signIn: (...args: unknown[]) => mockSignIn(...args),
}));
vi.mock("next/navigation", () => ({
  useSearchParams: () => mockUseSearchParams(),
  useRouter: () => ({ push: mockPush }),
}));

describe("LoginPage", () => {
  beforeEach(() => {
    mockUseSession.mockReset();
    mockSignIn.mockReset();
    mockUseSearchParams.mockReset();
    mockPush.mockReset();
    mockUseSearchParams.mockReturnValue(new URLSearchParams());
    mockUseSession.mockReturnValue({ status: "unauthenticated" });
  });

  it("passes the callbackUrl query param through to signIn instead of a hardcoded path", async () => {
    mockUseSearchParams.mockReturnValue(new URLSearchParams("callbackUrl=%2Fbets%2Fsomething"));
    render(<LoginPage />);
    screen.getByRole("button", { name: /sign in with google/i }).click();
    expect(mockSignIn).toHaveBeenCalledWith("google", { callbackUrl: "/bets/something" });
  });

  it("falls back to /bets when there is no callbackUrl param", () => {
    render(<LoginPage />);
    screen.getByRole("button", { name: /sign in with google/i }).click();
    expect(mockSignIn).toHaveBeenCalledWith("google", { callbackUrl: "/bets" });
  });

  it("shows an explanation when redirected back with ?error=AccessDenied", () => {
    mockUseSearchParams.mockReturnValue(new URLSearchParams("error=AccessDenied"));
    render(<LoginPage />);
    expect(screen.getByText(/not authorized|not on the allowlist|access denied/i)).toBeInTheDocument();
  });

  it("redirects to /bets immediately if already signed in", () => {
    mockUseSession.mockReturnValue({ status: "authenticated" });
    render(<LoginPage />);
    expect(mockPush).toHaveBeenCalledWith("/bets");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app/frontend && npx vitest run app/login/__tests__/page.test.tsx`
Expected: FAIL — current page ignores `callbackUrl`, has no error handling, no already-signed-in redirect.

- [ ] **Step 3: Rewrite the login page**

```tsx
// app/frontend/app/login/page.tsx
"use client";

import { useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { signIn, useSession } from "next-auth/react";

const ERROR_MESSAGES: Record<string, string> = {
  AccessDenied: "That Google account is not authorized for this app. Ask the owner to add it to the allowlist.",
};

export default function LoginPage() {
  const { status } = useSession();
  const router = useRouter();
  const searchParams = useSearchParams();

  const callbackUrl = searchParams.get("callbackUrl") ?? "/bets";
  const error = searchParams.get("error");

  useEffect(() => {
    if (status === "authenticated") router.push("/bets");
  }, [status, router]);

  if (status === "authenticated") return null;

  return (
    <div className="flex flex-col items-center mt-16 gap-4">
      <h1 className="text-lg font-medium text-ink">Sign in</h1>
      {error && (
        <p className="max-w-sm text-center text-sm text-serious">
          {ERROR_MESSAGES[error] ?? "Sign-in failed. Please try again."}
        </p>
      )}
      <button
        type="button"
        onClick={() => signIn("google", { callbackUrl })}
        className="rounded-md border border-accent px-4 py-2 text-sm font-medium text-accent hover:bg-accent/10"
      >
        Sign in with Google
      </button>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app/frontend && npx vitest run app/login/__tests__/page.test.tsx`
Expected: 4 passed

- [ ] **Step 5: Manual re-verification of the direct-redirect behavior**

This page is also reachable directly by `withAuth`'s own redirect (Task 7's fix from the prior W210 round already makes that a 1-hop redirect straight here). Re-run the same curl check used in that prior task: boot both servers, `curl -D - http://localhost:3000/bets` unauthenticated, confirm `Location: /login?callbackUrl=%2Fbets` still works end to end with this rewritten page (i.e. the page doesn't error on a real request, `npx tsc --noEmit` is clean).

- [ ] **Step 6: Commit**

```bash
git add app/frontend/app/login
git commit -m "fix(auth): login page honors callbackUrl, shows rejection errors, skips if already signed in (W210 follow-up)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: `LogBetButton` auth-awareness

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx` (the `LogBetButton` function only)
- Test: find `LogBetButton`'s existing test coverage first (grep `MatchUI.test.tsx` and any other `MatchUI.*.test.tsx` for `LogBetButton` or `"Log bet"` — there is at least one existing test from W115 asserting it does NOT render while hidden; that test's assumption doesn't change here, since the button stays commented out in `ProbabilityRow` until Task 7)

- [ ] **Step 1: Write the failing test**

Add a new test file `app/frontend/components/__tests__/LogBetButton.test.tsx` (this component isn't exported for direct testing today — check `MatchUI.tsx`'s export list; `LogBetButton` is already `export function LogBetButton`, per the codebase, so it's directly importable):

```tsx
// app/frontend/components/__tests__/LogBetButton.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { LogBetButton } from "../MatchUI";

const mockUseSession = vi.fn();
vi.mock("next-auth/react", () => ({
  useSession: () => mockUseSession(),
}));

const recommendation = {
  match: { home: "Arsenal", away: "Everton", date: "2026-08-22", league: "E0" },
  overall: "direct_bet",
  candidates: [],
  recommendation_pick: { market: "result_3way", selection: "home" },
  explanation: [],
  confidence: "medium",
  limitations: [],
  prediction_basis: "team_history_and_market",
} as never;

describe("LogBetButton auth-awareness", () => {
  beforeEach(() => {
    mockUseSession.mockReset();
  });

  it("shows a Sign in prompt instead of the log-bet control when unauthenticated", () => {
    mockUseSession.mockReturnValue({ status: "unauthenticated" });
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" />);
    expect(screen.getByRole("link", { name: /sign in/i })).toHaveAttribute("href", "/login?callbackUrl=%2Fmatches%2Fm1");
    expect(screen.queryByRole("button", { name: /log bet/i })).not.toBeInTheDocument();
  });

  it("shows the real Log bet control when authenticated", () => {
    mockUseSession.mockReturnValue({ status: "authenticated" });
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" />);
    expect(screen.getByRole("button", { name: /log bet/i })).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app/frontend && npx vitest run components/__tests__/LogBetButton.test.tsx`
Expected: FAIL — current `LogBetButton` has no session awareness.

- [ ] **Step 3: Update `LogBetButton`**

In `app/frontend/components/MatchUI.tsx`, modify the start of `LogBetButton` (around line 1717 — read the current full function first, since you're adding a branch, not replacing the whole thing):

```tsx
export function LogBetButton({
  matchId,
  recommendation,
  market,
  selection,
}: {
  matchId: string;
  recommendation: MatchRecommendationOut;
  market: string;
  selection: string;
}) {
  const { status } = useSession();
  const [open, setOpen] = useState(false);
  const [stake, setStake] = useState("");
  const [status_, setStatus] = useState<"idle" | "saving" | "done" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");

  if (status === "unauthenticated") {
    return (
      <Link href={`/login?callbackUrl=${encodeURIComponent(`/matches/${matchId}`)}`} className="text-xs font-medium text-accent">
        Sign in to log this bet
      </Link>
    );
  }
  if (status === "loading") return null;

  // ...rest of the function body is UNCHANGED from here down (submit(), the
  // stake input, etc.) -- only the top of the function and the addition of
  // the early-return branch above are new. Do not rewrite the rest.
```

(The renamed `status_`/`setStatus` local state is deliberate — `status` from `useSession()` and the existing local `status`/`setStatus` state for the save-in-progress flow collide on the same name. Rename ONLY the local one throughout the rest of the function body to `status_`/`setStatus`, keep every other reference to it — e.g. `status === "saving"`, `status === "done"` — updated to `status_ === "saving"` etc. Read the full existing function body first to find every occurrence before renaming, don't miss one.)

Add the import at the top of `MatchUI.tsx` if not already present:
```tsx
import Link from "next/link";
import { useSession } from "next-auth/react";
```
(Check whether `Link` is already imported in this file before adding a duplicate import — `MatchUI.tsx` is large and may already import it for other links.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app/frontend && npx vitest run components/__tests__/LogBetButton.test.tsx`
Expected: 2 passed

- [ ] **Step 5: Run the full frontend suite**

Run: `cd app/frontend && npx vitest run`
Expected: all pass, including the existing W115 test asserting `LogBetButton` doesn't render at all while still commented out in `ProbabilityRow` (unaffected by this task — the component itself changed, not whether it's called).

- [ ] **Step 6: Commit**

```bash
git add app/frontend/components/MatchUI.tsx app/frontend/components/__tests__/LogBetButton.test.tsx
git commit -m "fix(auth): LogBetButton shows a sign-in prompt instead of failing silently when logged out (W210 follow-up)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: `BetTrackerPage` — decouple stats/bets loading, friendlier session-expiry messaging

**Files:**
- Modify: `app/frontend/components/BetTracker.tsx`
- Test: existing `BetTracker.*.test.tsx` files — add new tests, don't break existing ones

- [ ] **Step 1: Write the failing tests**

Add to the most relevant existing `BetTracker.*.test.tsx` file (check `BetTracker.race.test.tsx`/`BetTracker.fixtureError.test.tsx` for the closest existing pattern and add alongside):

```tsx
it("still shows the bets list when only getBetStats fails", async () => {
  vi.mocked(getBets).mockResolvedValue([/* one real Bet fixture matching this file's existing shape */]);
  vi.mocked(getBetStats).mockRejectedValue(new ApiError("Failed to load bet stats (500)", 500));
  render(<BetTrackerPage />);
  await waitFor(() => expect(screen.getByText(/logged bets/i)).toBeInTheDocument());
  // the bets list itself rendered even though stats failed
});

it("shows a re-authenticate prompt on a 401, not a raw status-code message", async () => {
  vi.mocked(getBets).mockRejectedValue(new ApiError("Failed to load bets (401)", 401));
  vi.mocked(getBetStats).mockRejectedValue(new ApiError("Failed to load bet stats (401)", 401));
  render(<BetTrackerPage />);
  await waitFor(() => expect(screen.getByRole("link", { name: /sign in/i })).toBeInTheDocument());
});
```

(Use this file's existing `Bet`/mock-fixture conventions for the first test's placeholder bet object — check how other tests in the same file construct one rather than inventing a new shape.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app/frontend && npx vitest run components/__tests__/BetTracker.race.test.tsx` (or whichever file you added to)
Expected: FAIL on both new tests.

- [ ] **Step 3: Rewrite `BetTrackerPage`'s `load()`**

In `app/frontend/components/BetTracker.tsx`, replace:

```tsx
  async function load() {
    try {
      const [betList, betStats] = await Promise.all([getBets(), getBetStats()]);
      setBets(betList);
      setStats(betStats);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load bets.");
    }
  }
```

with:

```tsx
  const [needsAuth, setNeedsAuth] = useState(false);

  async function load() {
    setNeedsAuth(false);
    const [betsResult, statsResult] = await Promise.allSettled([getBets(), getBetStats()]);

    if (betsResult.status === "fulfilled") {
      setBets(betsResult.value);
    } else if (betsResult.reason instanceof ApiError && betsResult.reason.status === 401) {
      setNeedsAuth(true);
    } else {
      setError(betsResult.reason instanceof ApiError ? betsResult.reason.message : "Could not load bets.");
    }

    if (statsResult.status === "fulfilled") {
      setStats(statsResult.value);
    }
    // A stats-only failure (including a 401 already surfaced above via
    // betsResult) intentionally doesn't block the bets list from showing --
    // stats just stays null, and StatsBar's existing `{stats && (...)}`
    // guard already handles that by rendering nothing for that section.
  }
```

Then, in the render, add the sign-in prompt near the top of the returned JSX (right after the `<h1>`/description block, before `{stats && (...)}`):

```tsx
      {needsAuth && (
        <p className="mt-4 text-sm text-ink-secondary">
          Your session expired.{" "}
          <Link href="/login?callbackUrl=%2Fbets" className="font-medium text-accent">
            Sign in again
          </Link>{" "}
          to see your bets.
        </p>
      )}
```

Add `import Link from "next/link";` at the top of the file if not already present.

Apply the same `err instanceof ApiError && err.status === 401` pattern to `ManualBetForm`'s `submit()` catch block and `handleSettle()`'s catch block — both currently do `setStatus("error"); setErrorMsg(err instanceof ApiError ? err.message : "...")`. Change the error message construction in both to check for 401 first and show "Your session expired — sign in again" instead of the raw status-code string, otherwise keep the existing fallback behavior unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app/frontend && npx vitest run` (full suite, since this touches shared state used by multiple existing tests)
Expected: all pass, including the 2 new tests.

- [ ] **Step 5: Commit**

```bash
git add app/frontend/components/BetTracker.tsx app/frontend/components/__tests__/
git commit -m "fix(auth): decouple stats/bets loading, show re-auth prompt on session expiry (W210 follow-up)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Fix `ManualBetForm`'s freeform Market/Selection fields (Critical)

**Files:**
- Modify: `app/frontend/components/BetTracker.tsx` (`ManualBetForm` only)
- Test: existing `BetTracker.*.test.tsx` files

- [ ] **Step 1: Write the failing test**

```tsx
it("only offers valid market/selection combinations, not freeform text", async () => {
  render(<ManualBetForm onLogged={vi.fn()} />);
  // ... select a fixture first via this file's existing helper/pattern for
  // that step, then:
  const marketSelect = screen.getByLabelText(/market/i);
  expect(marketSelect.tagName).toBe("SELECT");
  await userEvent.selectOptions(marketSelect, "btts");
  const selectionSelect = screen.getByLabelText(/selection/i);
  expect(selectionSelect.tagName).toBe("SELECT");
  const options = Array.from(selectionSelect.querySelectorAll("option")).map((o) => o.textContent);
  expect(options).toEqual(expect.arrayContaining(["Yes", "No"]));
  expect(options).not.toEqual(expect.arrayContaining(["home", "draw", "away"]));
});
```

(Adapt the fixture-selection setup to whatever this test file's existing tests already use to get `ManualBetForm` into its post-fixture-selected state — check an existing test in the same file for that exact sequence rather than guessing.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app/frontend && npx vitest run` (targeted to the file you added this test to)
Expected: FAIL — current inputs are plain `<input>` text fields, not `<select>`.

- [ ] **Step 3: Replace the freeform inputs with constrained dropdowns**

In `app/frontend/components/BetTracker.tsx`, add this constant near the top of the file (module scope, above `ManualBetForm`):

```tsx
// W210 follow-up: the only markets/selections settlement.py's market_correct()
// can ever resolve (src/agent/market_resolution.py, RESOLVABLE_MARKETS).
// A freeform text field let a user log a bet with a market/selection that
// could never programmatically settle -- it would just sit "open" forever
// with no error anywhere. Constraining to exactly these values closes that.
const MARKET_SELECTIONS: Record<string, { value: string; label: string }[]> = {
  result_3way: [
    { value: "home", label: "Home" },
    { value: "draw", label: "Draw" },
    { value: "away", label: "Away" },
  ],
  btts: [
    { value: "yes", label: "Yes" },
    { value: "no", label: "No" },
  ],
  total_goals: [
    { value: "over_2.5", label: "Over 2.5" },
    { value: "under_2.5", label: "Under 2.5" },
  ],
  total_corners: [
    { value: "over_9.5", label: "Over 9.5" },
    { value: "under_9.5", label: "Under 9.5" },
  ],
};
const MARKET_LABELS: Record<string, string> = {
  result_3way: "Result",
  btts: "BTTS",
  total_goals: "Total goals",
  total_corners: "Total corners",
};
```

Replace the Market/Selection `<input>` pair (inside the `grid grid-cols-2 gap-3 sm:grid-cols-4` block) with:

```tsx
            <div>
              <label htmlFor="manual-bet-market" className="sr-only">Market</label>
              <select
                id="manual-bet-market"
                value={market}
                onChange={(e) => {
                  setMarket(e.target.value);
                  setSelection(""); // force a fresh, valid choice for the new market
                }}
                className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-ink outline-none focus:border-accent"
              >
                {Object.keys(MARKET_SELECTIONS).map((m) => (
                  <option key={m} value={m}>{MARKET_LABELS[m]}</option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="manual-bet-selection" className="sr-only">Selection</label>
              <select
                id="manual-bet-selection"
                value={selection}
                onChange={(e) => setSelection(e.target.value)}
                className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-ink outline-none focus:border-accent"
              >
                <option value="">Select…</option>
                {MARKET_SELECTIONS[market]?.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
```

(`market`'s initial state is already `"result_3way"`, one of the 4 valid keys — no change needed to its `useState` default. `selection`'s initial state is already `""`, which now correctly maps to the dropdown's own placeholder "Select…" option instead of an empty text field.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app/frontend && npx vitest run`
Expected: all pass, including the new test. Check whether `submit()`'s existing validation (`if (!selection.trim() || ...)`) still makes sense — a `<select>`'s value is never whitespace-only, so `.trim()` is now redundant but harmless; leave it unless it causes a type issue.

- [ ] **Step 5: Commit**

```bash
git add app/frontend/components/BetTracker.tsx app/frontend/components/__tests__/
git commit -m "fix(betting): constrain manual bet market/selection to values settlement can actually resolve (W210 follow-up, Critical)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: `BetTracker` UI polish — column headers, stats loading state, form labels, fixture-search empty state

**Files:**
- Modify: `app/frontend/components/BetTracker.tsx`
- Test: existing `BetTracker.*.test.tsx` files

- [ ] **Step 1: Write the failing tests**

```tsx
it("shows column headers above the logged bets list", async () => {
  vi.mocked(getBets).mockResolvedValue([/* one bet fixture */]);
  vi.mocked(getBetStats).mockResolvedValue(/* one stats fixture matching this file's existing shape */);
  render(<BetTrackerPage />);
  await waitFor(() => expect(screen.getByText(/logged bets/i)).toBeInTheDocument());
  expect(screen.getByText("Odds")).toBeInTheDocument();
  expect(screen.getByText("Stake")).toBeInTheDocument();
  expect(screen.getByText("P&L")).toBeInTheDocument();
  expect(screen.getByText("Outcome")).toBeInTheDocument();
});

it("shows a loading placeholder for stats while they're in flight", () => {
  vi.mocked(getBets).mockReturnValue(new Promise(() => {})); // never resolves
  vi.mocked(getBetStats).mockReturnValue(new Promise(() => {}));
  render(<BetTrackerPage />);
  expect(screen.getByText(/loading/i)).toBeInTheDocument();
});

it("shows a 'no matches' message when a fixture search returns nothing", async () => {
  vi.mocked(getFixtures).mockResolvedValue([]);
  render(<ManualBetForm onLogged={vi.fn()} />);
  await userEvent.type(screen.getByPlaceholderText(/search a real fixture/i), "zzz-no-such-team");
  await waitFor(() => expect(screen.getByText(/no matching fixtures/i)).toBeInTheDocument());
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app/frontend && npx vitest run`
Expected: all 3 new tests FAIL.

- [ ] **Step 3: Add column headers to the bets list**

In `app/frontend/components/BetTracker.tsx`, find the `<h2>Logged bets</h2>` block and add a header row directly above the `.map()`:

```tsx
      <h2 className="mt-6 text-sm font-semibold uppercase tracking-wide text-muted">Logged bets</h2>
      {bets && bets.length > 0 && (
        <div className="mt-2 grid grid-cols-[1fr_auto_auto_auto_auto] gap-4 border-b border-border pb-1.5 text-xs font-medium uppercase tracking-wide text-muted">
          <span>Match</span>
          <span className="text-right">Odds</span>
          <span className="text-right">Stake</span>
          <span className="text-right">P&amp;L</span>
          <span className="text-right">Outcome</span>
        </div>
      )}
```

(Match the existing grid column template — `grid-cols-[1fr_auto_auto_auto_auto]` — exactly to `BetRow`'s own grid so columns actually align; read `BetRow`'s current className to confirm the exact template string before copying it.)

- [ ] **Step 4: Add a stats loading placeholder**

Replace:
```tsx
      {stats && (
        <div className="mt-6">
          <StatsBar stats={stats} />
        </div>
      )}
```
with:
```tsx
      <div className="mt-6">
        {stats ? <StatsBar stats={stats} /> : !needsAuth && !error && <p className="text-sm text-ink-secondary">Loading…</p>}
      </div>
```

(Reuses the `needsAuth`/`error` state Task 4 already introduced, so the loading placeholder doesn't show forever if the load actually failed — only while genuinely in flight.)

- [ ] **Step 5: Add persistent labels to `ManualBetForm`'s Odds/Stake inputs**

The Market/Selection fields already got real `<label>`s in Task 5. Apply the same `sr-only` label pattern to the Odds and Stake `<input>` elements in the same grid:

```tsx
            <div>
              <label htmlFor="manual-bet-odds" className="sr-only">Odds</label>
              <input id="manual-bet-odds" value={odds} onChange={(e) => setOdds(e.target.value)} placeholder="Odds" inputMode="decimal" className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-ink outline-none focus:border-accent" />
            </div>
            <div>
              <label htmlFor="manual-bet-stake" className="sr-only">Stake</label>
              <input id="manual-bet-stake" value={stake} onChange={(e) => setStake(e.target.value)} placeholder="Stake" inputMode="decimal" className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-ink outline-none focus:border-accent" />
            </div>
```

(Keep `placeholder` too — it's still useful as an in-field hint even with a screen-reader-only label; only the *persistent, accessible* label was missing, not the visual hint.)

- [ ] **Step 6: Add a "no matching fixtures" empty state**

In the fixture search block, change:
```tsx
          {results.length > 0 && (
            <div className="mt-2 flex flex-col gap-1.5">
              {results.map((f) => (/* ... */))}
            </div>
          )}
```
to:
```tsx
          {query.trim().length > 0 && !fixturesError && (
            results.length > 0 ? (
              <div className="mt-2 flex flex-col gap-1.5">
                {results.map((f) => (/* ...unchanged... */))}
              </div>
            ) : (
              <p className="mt-2 text-sm text-ink-secondary">No matching fixtures.</p>
            )
          )}
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd app/frontend && npx vitest run`
Expected: all pass, including the 3 new tests from this task plus everything from Tasks 1-5.

- [ ] **Step 8: Commit**

```bash
git add app/frontend/components/BetTracker.tsx app/frontend/components/__tests__/
git commit -m "fix(betting): bet-list column headers, stats loading state, form labels, empty fixture-search state (W210 follow-up)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Re-enable the Bets nav tab and LogBetButton (W106/W115)

**Files:**
- Modify: `app/frontend/components/AppShell.tsx` (`NAV_ITEMS`)
- Modify: `app/frontend/components/MatchUI.tsx` (`ProbabilityRow`'s commented `LogBetButton` block)
- Test: existing `AppShell.test.tsx` and `MatchUI.test.tsx`

- [ ] **Step 1: Check existing tests that assert the CURRENT hidden state**

Before changing anything, find and read the two tests these hides originally added (per the plan's own reference to them):
- `AppShell.test.tsx`: a test asserting "Bets" is absent from rendered nav links (added when W106 shipped).
- `MatchUI.test.tsx`: `"W115: does not render a Log bet control -- bet tracking hidden for now"`.

Both will need to flip from asserting absence to asserting presence — this task inverts what W106/W115 did, on purpose, now that Tasks 1-6 have closed the gaps that motivated hiding it.

- [ ] **Step 2: Re-enable the nav tab**

In `app/frontend/components/AppShell.tsx`, change:
```tsx
// Bets tab hidden from nav (2026-08-13) -- feature not ready yet. The route
// (/bets, BetTracker.tsx) and its active="bets" AppShell state are left
// intact, just unlinked -- flip this back to re-surface it, no other change
// needed.
const NAV_ITEMS: {...}[] = [
  { href: "/", label: "Daily Edges", key: "dashboard", icon: House },
  { href: "/matches", label: "All Matches", key: "matches", icon: ListBullets },
];
```
to:
```tsx
// W210 follow-up (2026-09-14): re-enabled -- real per-user auth, a visible
// signed-in/sign-out UI (Task 1), and auth-aware failure states (Tasks 2-6)
// close the gaps that motivated hiding this on 2026-08-13 (W106).
const NAV_ITEMS: {...}[] = [
  { href: "/", label: "Daily Edges", key: "dashboard", icon: House },
  { href: "/matches", label: "All Matches", key: "matches", icon: ListBullets },
  { href: "/bets", label: "Bets", key: "bets", icon: /* pick an appropriate existing Phosphor icon already imported in this file's import list, or add one -- check what's already imported before adding a new icon import */ },
];
```

- [ ] **Step 3: Update the AppShell test**

Find the existing test asserting "Bets" is absent and invert it to assert presence — read the exact current test text first (it will have a name like `"does not render a Bets nav link"` or similar) and rewrite its assertion, keeping the rest of the test structure.

- [ ] **Step 4: Re-enable `LogBetButton` in `ProbabilityRow`**

In `app/frontend/components/MatchUI.tsx`, change:
```tsx
        {/* Log-bet UI hidden (2026-08-13, W115) -- bet tracking isn't built
            out enough to surface yet, same call as W106 hiding the Bets nav
            tab. LogBetButton itself and its backend path are untouched;
            uncomment below to re-enable once ready.
        {matchId && recommendation && !anomalous && (
          <LogBetButton matchId={matchId} recommendation={recommendation} market={m.market} selection={m.selection} />
        )}
        */}
```
to:
```tsx
        {/* W210 follow-up (2026-09-14): re-enabled -- see Task 3's
            auth-aware LogBetButton and AppShell's Task 1 sign-in UI. */}
        {matchId && recommendation && !anomalous && (
          <LogBetButton matchId={matchId} recommendation={recommendation} market={m.market} selection={m.selection} />
        )}
```

- [ ] **Step 5: Update the MatchUI test**

Find `"W115: does not render a Log bet control -- bet tracking hidden for now"` and invert it — rename to reflect the new behavior (e.g. `"W210: renders a Log bet control now that bet tracking is re-enabled"`), assert the button/link IS present for the same direct_bet, non-anomalous fixture case the original test used. Since `LogBetButton` now needs `next-auth/react`'s `useSession` mocked (Task 3), make sure this test file has that mock in place — check whether `MatchUI.test.tsx` already needs it for other reasons or whether you're adding it fresh; default the mock to `{ status: "authenticated" }` for this specific test so the real button (not the sign-in prompt) is what's being asserted on.

- [ ] **Step 6: Run the full frontend suite**

Run: `cd app/frontend && npx vitest run`
Expected: all pass, zero regressions, both inverted tests now assert the new (enabled) behavior.

- [ ] **Step 7: Manual end-to-end check**

This is the actual moment to do the real-browser sign-in test that's been deferred since W210's original Task 10. Boot both servers, sign in for real as `fpai.deploy@gmail.com`, confirm: the Bets nav tab is visible and works, `LogBetButton` appears on a real match's probability row and successfully logs a bet, `UserMenu` shows the signed-in email and Sign Out actually ends the session (visiting `/bets` afterward redirects to `/login` again).

- [ ] **Step 8: Run the full backend + frontend suite one final time**

```bash
cd /Users/tianqihuang/Documents/GitHub/FPAI
./venv/bin/python -m pytest tests/ app/backend/tests/ scripts/ -q
cd app/frontend && npx vitest run
```
Expected: same pass counts as the end of the original W210 branch (backend: pre-existing `test_fixtures_endpoint.py` gap only; frontend: all passing, count up by however many tests Tasks 1-7 added).

- [ ] **Step 9: Update the docs**

Append a completion note to W210's existing entry in `documents/app_user_stories.md` (same row, don't create a new one — matches this project's own established pattern) summarizing this follow-up round: the 12 audit findings, which were fixed, and that the Bets nav tab + `LogBetButton` are now genuinely re-enabled (not just technically wired).

- [ ] **Step 10: Commit**

```bash
git add app/frontend/components/AppShell.tsx app/frontend/components/MatchUI.tsx app/frontend/components/__tests__/ documents/app_user_stories.md
git commit -m "feat(betting): re-enable Bets nav tab and LogBetButton now that auth UX gaps are closed (W106/W115 follow-up)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage** against the 12 audit findings:
1. No visible signed-in state → Task 1. ✅
2. `LogBetButton` no auth-awareness → Task 3. ✅
3. Critical: freeform market/selection → Task 5. ✅
4. Raw 401 messages → Task 4. ✅
5. Stats failure blanks bets list → Task 4. ✅
6. `callbackUrl` hardcoded → Task 2. ✅
7. No `?error=AccessDenied` handling → Task 2. ✅
8. No column headers → Task 6. ✅
9. Already-signed-in visitor not redirected from `/login` → Task 2. ✅
10. No stats loading placeholder → Task 6. ✅
11. No persistent form labels → Task 5 (market/selection) + Task 6 (odds/stake). ✅
12. No "no results" fixture-search state → Task 6. ✅

Plus the actual re-enable (W106/W115) → Task 7, sequenced last since it depends on every other fix being in place first.

**Placeholder scan**: no TBD/TODO; every step has real code or an exact command. Two steps (Task 1 Step 5's exact `<aside>` insertion point, Task 3 Step 3's full function body, Task 7 Step 2's icon choice) explicitly instruct reading the current file first rather than guessing at surrounding context that could have drifted — this is a deliberate "read first" instruction, not a placeholder for missing logic.

**Type consistency**: `needsAuth` (Task 4) is referenced again in Task 6 Step 4's loading-placeholder logic — same name, same component, consistent. `MARKET_SELECTIONS`/`MARKET_LABELS` (Task 5) are used only within that task. `UserMenu` (Task 1) is imported identically in Task 1 Step 5 — no signature drift across tasks.
