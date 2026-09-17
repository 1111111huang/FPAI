# Unified Log-Bet Modal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every inline-expand bet-logging UI (`LogBetButton`'s stake row, `ManualBetForm`'s field grid) with one shared, centered modal dialog (`LogBetModal`) that opens as an overlay on top of the current page -- matching the direct user mockup. Entering from a match card/detail-page market row pre-populates and locks Market/Pick/Odds (only Stake is editable); entering from the Bets page's manual fixture search leaves Market/Pick/Odds editable.

**Architecture:** One new presentational+stateful component, `components/LogBetModal.tsx` -- owns the modal shell (backdrop, centered card, close button), the Market/Pick/Odds/Stake fields (locked-display or editable, driven by a `locked` prop), the live "Returns $X.XX if it hits" figure, and Cancel/Confirm. It does NOT call the API itself -- callers pass an `onSubmit(fields) => Promise<void>` so `LogBetButton` can call `logBetFromRecommendation` and `ManualBetForm` can call `logBetManual` through the exact same modal UI. This is the first true centered-overlay modal in the app (existing overlays -- `AppShell`'s mobile menu, `MatchCard`'s mobile insights rail -- are side-drawers); its backdrop/close pattern is adapted from those, not copied verbatim (drawer positioning doesn't apply to a centered dialog).

**Tech Stack:** React/Next.js 14, Tailwind, Vitest/RTL, `@phosphor-icons/react`.

---

## Context for every task below

- Branch: `feature/w210-multiuser-auth` (already checked out -- do not create a new branch).
- `LogBetButton` and `ProbabilityRow` live in `app/frontend/components/MatchUI.tsx` (currently ~line 1783 and ~1937); `MatchCard`'s quick-log call site is ~line 1230; `MatchAnalysisPage` is ~line 2096.
- `ManualBetForm` lives in `app/frontend/components/BetTracker.tsx` (currently ~line 48), including `MARKET_SELECTIONS` (module-scope constant, already the closed set of valid market/selection pairs) and `formatDateLong` (~line 22).
- API functions: `logBetFromRecommendation`, `logBetManual` in `app/frontend/lib/api.ts` -- signatures already exist and are unchanged by this plan.
- Run frontend tests with `npx vitest run` from `app/frontend/`. Run `npx tsc --noEmit` and `npx next build` from `app/frontend/` before considering any task done.
- The dev servers currently running locally are in **sandbox mode** (SANDBOX_DATE=2026-09-14) -- do not stop/restart them as part of this plan; restarting them (to pick up code changes) is a normal, expected step and should use the same sandbox launch (`./venv/bin/python scripts/launch_sandbox.py 2026-09-14` from the repo root, backend :8000/frontend :3000) rather than a plain non-sandbox launch, so the pinned "today" isn't lost.

---

### Task 1: Build `LogBetModal` + the shared `matchStatusLabel` helper

**Files:**
- Create: `app/frontend/components/LogBetModal.tsx`
- Modify: `app/frontend/components/MatchUI.tsx` (export `formatDay`, `formatKickoff`; add and export `matchStatusLabel`)
- Test: `app/frontend/components/__tests__/LogBetModal.test.tsx` (new)

- [ ] **Step 1: Export `formatDay`/`formatKickoff` and add `matchStatusLabel`**

In `app/frontend/components/MatchUI.tsx`, change:
```tsx
function formatKickoff(iso: string): string {
```
to:
```tsx
export function formatKickoff(iso: string): string {
```
and change:
```tsx
function formatDay(iso: string, asOf: Date, sandboxMode: boolean): string {
```
to:
```tsx
export function formatDay(iso: string, asOf: Date, sandboxMode: boolean): string {
```

Immediately after `formatDay`'s closing `}`, add:
```tsx
// W218: shared "Today · Full Time" / "Today · 3:00 PM" style label for
// LogBetModal's fixture header -- the same day/time convention every list
// page already uses (MatchCard's closing row), reused instead of a fourth
// copy of this exact ternary.
export function matchStatusLabel(kickoffIso: string, isCompleted: boolean, asOf: Date, sandboxMode: boolean): string {
  const day = formatDay(kickoffIso, asOf, sandboxMode);
  return `${day} · ${isCompleted ? "Full Time" : formatKickoff(kickoffIso)}`;
}
```

- [ ] **Step 2: Write the failing tests for `LogBetModal`**

Create `app/frontend/components/__tests__/LogBetModal.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LogBetModal } from "../LogBetModal";

const baseProps = {
  open: true,
  onClose: vi.fn(),
  homeTeam: "Villarreal",
  awayTeam: "Real Betis",
  statusLabel: "Today · Full Time",
  onSubmit: vi.fn(),
};

describe("LogBetModal", () => {
  it("renders nothing when closed", () => {
    render(<LogBetModal {...baseProps} open={false} locked market="result_3way" selection="draw" odds={4.0} />);
    expect(screen.queryByText("Log bet")).not.toBeInTheDocument();
  });

  it("locked mode: shows Market/Pick/Odds as fixed text, not editable controls", () => {
    render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} />);

    expect(screen.getByText("Villarreal v Real Betis · via The Odds API")).toBeInTheDocument();
    expect(screen.getByText("3-Way Result")).toBeInTheDocument();
    expect(screen.getByText("Draw")).toBeInTheDocument();
    expect(screen.getByText("4.00")).toBeInTheDocument();
    expect(screen.queryByLabelText(/^market$/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/^pick$/i)).not.toBeInTheDocument();
  });

  it("editable mode: Market/Selection are real dropdowns, Odds is a real input", () => {
    render(
      <LogBetModal {...baseProps} locked={false} market="result_3way" selection="" odds={null} />
    );

    expect(screen.getByLabelText(/^market$/i).tagName).toBe("SELECT");
    expect(screen.getByLabelText(/^pick$/i).tagName).toBe("SELECT");
    expect(screen.getByLabelText(/^odds$/i)).toHaveAttribute("placeholder", "0.00");
  });

  it("editable mode: changing market resets the selection", async () => {
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked={false} market="result_3way" selection="home" odds={null} />);

    await user.selectOptions(screen.getByLabelText(/^market$/i), "btts");

    expect((screen.getByLabelText(/^pick$/i) as HTMLSelectElement).value).toBe("");
  });

  it("shows a live 'Returns $X.XX if it hits' as stake is typed, using the locked odds", async () => {
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} />);

    expect(screen.getByText(/returns \$0\.00 if it hits/i)).toBeInTheDocument();

    await user.type(screen.getByLabelText(/^stake$/i), "10");

    expect(screen.getByText(/returns \$40\.00 if it hits/i)).toBeInTheDocument();
  });

  it("Cancel calls onClose without calling onSubmit", async () => {
    const onClose = vi.fn();
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} onClose={onClose} onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: /cancel/i }));

    expect(onClose).toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("the X button also calls onClose", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} onClose={onClose} />);

    await user.click(screen.getByRole("button", { name: /close/i }));

    expect(onClose).toHaveBeenCalled();
  });

  it("clicking the backdrop calls onClose", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    const { container } = render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} onClose={onClose} />);

    await user.click(container.querySelector('[aria-hidden="true"]')!);

    expect(onClose).toHaveBeenCalled();
  });

  it("Confirm bet calls onSubmit with the parsed fields (locked mode)", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} onSubmit={onSubmit} />);

    await user.type(screen.getByLabelText(/^stake$/i), "10");
    await user.click(screen.getByRole("button", { name: /confirm bet/i }));

    expect(onSubmit).toHaveBeenCalledWith({ market: "result_3way", selection: "draw", odds: 4.0, stake: 10 });
  });

  it("Confirm bet calls onSubmit with the user-picked fields (editable mode)", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked={false} market="result_3way" selection="" odds={null} onSubmit={onSubmit} />);

    await user.selectOptions(screen.getByLabelText(/^market$/i), "btts");
    await user.selectOptions(screen.getByLabelText(/^pick$/i), "yes");
    await user.type(screen.getByLabelText(/^odds$/i), "1.9");
    await user.type(screen.getByLabelText(/^stake$/i), "5");
    await user.click(screen.getByRole("button", { name: /confirm bet/i }));

    expect(onSubmit).toHaveBeenCalledWith({ market: "btts", selection: "yes", odds: 1.9, stake: 5 });
  });

  it("rejects submit with no stake / a zero odds / an unselected pick, without calling onSubmit", async () => {
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked={false} market="result_3way" selection="" odds={null} onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: /confirm bet/i }));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByText(/fill in/i)).toBeInTheDocument();
  });

  it("shows an inline error and keeps the modal open when onSubmit rejects", async () => {
    const onSubmit = vi.fn().mockRejectedValue(new Error("Could not log bet."));
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} onSubmit={onSubmit} />);

    await user.type(screen.getByLabelText(/^stake$/i), "10");
    await user.click(screen.getByRole("button", { name: /confirm bet/i }));

    expect(await screen.findByText("Could not log bet.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /confirm bet/i })).toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `npx vitest run components/__tests__/LogBetModal.test.tsx` (from `app/frontend/`)
Expected: FAIL -- `../LogBetModal` doesn't exist yet.

- [ ] **Step 4: Implement `LogBetModal`**

Create `app/frontend/components/LogBetModal.tsx`:

```tsx
"use client";

import { useState } from "react";
import { Plus, WarningCircle, X } from "@phosphor-icons/react";
import { TeamBadge, marketLabel } from "./MatchUI";

// W218: the one closed set of valid market/selection pairs -- duplicated
// from BetTracker.tsx's MARKET_SELECTIONS (that file imports this one, not
// the other way around, to avoid a circular import between the two
// components) is exactly the trap this plan avoids: exported here as the
// single source, BetTracker.tsx's copy is deleted in Task 3.
export const MARKET_SELECTIONS: Record<string, { value: string; label: string }[]> = {
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

export type LogBetFields = { market: string; selection: string; odds: number; stake: number };

export function LogBetModal({
  open,
  onClose,
  homeTeam,
  awayTeam,
  statusLabel,
  locked,
  market: initialMarket,
  selection: initialSelection,
  odds: initialOdds,
  onSubmit,
}: {
  open: boolean;
  onClose: () => void;
  homeTeam: string;
  awayTeam: string;
  statusLabel: string;
  // W218: true for the from-a-recommendation paths (MatchCard's quick-log,
  // ProbabilityRow's per-market picker) -- Market/Pick/Odds are already
  // decided by the recommendation being logged and shown as fixed text, not
  // controls. false for ManualBetForm's fixture-search path, where none of
  // the three are known yet.
  locked: boolean;
  market: string;
  selection: string;
  odds: number | null;
  onSubmit: (fields: LogBetFields) => Promise<void>;
}) {
  const [market, setMarket] = useState(initialMarket);
  const [selection, setSelection] = useState(initialSelection);
  const [odds, setOdds] = useState(initialOdds != null ? String(initialOdds) : "");
  const [stake, setStake] = useState("");
  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  if (!open) return null;

  const parsedOdds = parseFloat(odds);
  const parsedStake = parseFloat(stake);
  const returns = parsedOdds > 0 && parsedStake > 0 ? (parsedOdds * parsedStake).toFixed(2) : "0.00";

  async function handleConfirm() {
    if (!selection.trim() || !parsedOdds || parsedOdds <= 1 || !parsedStake || parsedStake <= 0) {
      setErrorMsg("Fill in a pick, a valid odds (>1), and a stake (>0).");
      return;
    }
    setSaving(true);
    setErrorMsg(null);
    try {
      await onSubmit({ market, selection, odds: parsedOdds, stake: parsedStake });
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Could not log bet.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <div className="fixed inset-0 z-40 bg-page/70 backdrop-blur-sm" onClick={onClose} aria-hidden="true" />
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Log bet"
        className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-border bg-surface p-6 shadow-2xl"
      >
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold text-ink">Log bet</h2>
            <p className="mt-1 text-sm text-ink-secondary">
              {homeTeam} v {awayTeam} · via The Odds API
            </p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close" className="text-ink-secondary hover:text-ink">
            <X size={20} />
          </button>
        </div>

        <div className="mt-4 flex items-center gap-3 rounded-xl border border-border bg-page/60 p-3">
          <span className="flex -space-x-2">
            <TeamBadge name={homeTeam} size="lg" />
            <TeamBadge name={awayTeam} size="lg" />
          </span>
          <div>
            <div className="text-base font-semibold text-ink">{homeTeam} v {awayTeam}</div>
            <div className="text-xs text-ink-secondary">{statusLabel}</div>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3">
          <div>
            <div className="mb-1 text-xs text-ink-secondary">Market</div>
            {locked ? (
              <div className="rounded-lg border border-border bg-page/60 px-3 py-2 text-sm font-semibold text-ink">
                {marketLabel(market).label}
              </div>
            ) : (
              <>
                <label htmlFor="log-bet-market" className="sr-only">Market</label>
                <select
                  id="log-bet-market"
                  value={market}
                  onChange={(e) => {
                    setMarket(e.target.value);
                    setSelection(""); // force a fresh, valid choice for the new market
                  }}
                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-ink outline-none focus:border-accent"
                >
                  {Object.keys(MARKET_SELECTIONS).map((m) => (
                    <option key={m} value={m}>{marketLabel(m).label}</option>
                  ))}
                </select>
              </>
            )}
          </div>
          <div>
            <div className="mb-1 text-xs text-ink-secondary">Pick</div>
            {locked ? (
              <div className="rounded-lg border border-border bg-page/60 px-3 py-2 text-sm font-semibold text-ink">
                {MARKET_SELECTIONS[market]?.find((o) => o.value === selection)?.label ?? selection}
              </div>
            ) : (
              <>
                <label htmlFor="log-bet-selection" className="sr-only">Pick</label>
                <select
                  id="log-bet-selection"
                  value={selection}
                  onChange={(e) => setSelection(e.target.value)}
                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-ink outline-none focus:border-accent"
                >
                  <option value="">Select…</option>
                  {MARKET_SELECTIONS[market]?.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </>
            )}
          </div>
          <div>
            <div className="mb-1 text-xs text-ink-secondary">Odds</div>
            {locked ? (
              <div className="rounded-lg border border-border bg-page/60 px-3 py-2 text-sm font-semibold text-ink">
                {parsedOdds > 0 ? parsedOdds.toFixed(2) : "—"}
              </div>
            ) : (
              <>
                <label htmlFor="log-bet-odds" className="sr-only">Odds</label>
                <input
                  id="log-bet-odds"
                  value={odds}
                  onChange={(e) => setOdds(e.target.value)}
                  placeholder="0.00"
                  inputMode="decimal"
                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-ink outline-none focus:border-accent"
                />
              </>
            )}
          </div>
          <div>
            <label htmlFor="log-bet-stake" className="mb-1 block text-xs text-ink-secondary">Stake</label>
            <div className="relative">
              <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-sm text-ink-secondary">$</span>
              <input
                id="log-bet-stake"
                type="number"
                min="0"
                step="0.01"
                value={stake}
                onChange={(e) => setStake(e.target.value)}
                placeholder="0.00"
                autoFocus
                className="w-full rounded-lg border border-border bg-surface py-2 pl-6 pr-3 text-sm text-ink outline-none focus:border-accent"
              />
            </div>
          </div>
        </div>

        <p className="mt-3 text-sm text-ink-secondary">
          Returns <span className="font-mono font-medium text-good">${returns}</span> if it hits
        </p>

        {errorMsg && (
          <p className="mt-2 flex items-center gap-1.5 text-xs text-serious">
            <WarningCircle weight="fill" size={13} />
            {errorMsg}
          </p>
        )}

        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={saving}
            className="rounded-lg border border-border px-4 py-2 text-sm font-medium text-ink-secondary disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            disabled={saving}
            className="flex items-center gap-1.5 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white transition hover:bg-accent/90 disabled:opacity-50"
          >
            <Plus size={14} weight="bold" />
            {saving ? "Logging…" : "Confirm bet"}
          </button>
        </div>
      </div>
    </>
  );
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx vitest run components/__tests__/LogBetModal.test.tsx`
Expected: PASS (all 13 tests)

- [ ] **Step 6: Run `npx tsc --noEmit` and confirm `marketLabel`/`TeamBadge` are exported from `MatchUI.tsx`**

Both already are (`marketLabel` since W174, `TeamBadge` since the original DraftUI port) -- this step is just confirming the import resolves, not adding new exports.

- [ ] **Step 7: Commit**

```bash
git add app/frontend/components/LogBetModal.tsx app/frontend/components/__tests__/LogBetModal.test.tsx app/frontend/components/MatchUI.tsx
git commit -m "feat(app): W218 -- new unified LogBetModal component

Direct user mockup: one shared centered-overlay modal for logging a
bet, replacing every inline-expand bet-logging UI. Not wired into any
caller yet (Tasks 2-3) -- this task only builds and tests the
component itself: a backdrop + centered dialog, Market/Pick/Odds shown
as locked text or real editable controls (driven by a \`locked\` prop),
a live \"Returns \$X.XX if it hits\" figure, Cancel/X/backdrop-click all
close without submitting, and \`onSubmit\` is a plain async callback so
callers stay in charge of which API function actually gets called.

Also exports formatDay/formatKickoff (previously module-private) plus
a new matchStatusLabel() helper, both needed to build the fixture
header's \"Today · Full Time\" label consistently across every caller.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Rewire `LogBetButton` to open the modal (locked mode)

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx` (`LogBetButton`, `MatchCard`'s call site, `ProbabilityRow`, `MatchAnalysisPage`)
- Test: `app/frontend/components/__tests__/LogBetButton.test.tsx`, `app/frontend/components/__tests__/MatchUI.test.tsx`

Depends on Task 1.

**Design:** `LogBetButton` keeps its trigger (`link`/`pill` variant, unauthenticated Sign-in link, `done` state) exactly as-is -- only the `open` (stake-entry) state changes: instead of rendering the inline stake/Confirm/Cancel row, it renders `<LogBetModal open locked .../>`. `ProbabilityRow` and `MatchCard` both need to pass `statusLabel`, `homeTeam`, and `awayTeam` down to `LogBetButton` -- `homeTeam`/`awayTeam` as new explicit props, NOT pulled from `recommendation.match` (that field is typed `Record<string, unknown>` in `lib/types.ts`, so `.home`/`.away` on it doesn't type-check; every caller already has real typed `home`/`away` strings of its own, which is the actual reason to add these as props rather than reach into the untyped snapshot). `MatchAnalysisPage` needs its own `asOf`/`sandboxMode` (via `useSandboxAsOf()`, not currently called there) to compute `statusLabel` the same way `MatchCard` already does.

- [ ] **Step 1: Read the current state of both test files in full**

Run: `cat app/frontend/components/__tests__/LogBetButton.test.tsx app/frontend/components/__tests__/MatchUI.test.tsx | head -100` and search each file for every test that currently asserts on the OLD inline stake row (`getByPlaceholderText("Stake")`, `getByRole("button", { name: "Confirm" })`, `getByRole("button", { name: "Cancel" })` scoped to LogBetButton, the restated-terms text `/home @ 2\.35/i`-style assertions) -- these all need rewriting in this task to instead assert on `LogBetModal`'s own elements (`getByLabelText(/^stake$/i)`, `getByRole("button", { name: /confirm bet/i })`, the modal's own Cancel button, the locked Market/Pick/Odds display text). Do this file-by-file, test-by-test -- do not skip any; a test still asserting on the deleted inline UI will fail, not silently pass. Every existing `<LogBetButton .../>` construction in these test files will also need new `homeTeam`/`awayTeam`/`statusLabel` props added (Step 2 makes them required) -- catalog those construction sites here too.

- [ ] **Step 2: Add `statusLabel`/`homeTeam`/`awayTeam` props to `LogBetButton`, and switch the `open` branch to render `LogBetModal`**

In `app/frontend/components/MatchUI.tsx`, add `import { LogBetModal } from "./LogBetModal";` near the top (with the other local imports).

Change `LogBetButton`'s props to add `statusLabel: string`, `homeTeam: string`, `awayTeam: string`, and replace the whole `if (!open) { ... }` trigger block + everything after it (the inline stake row, the `matchedCandidate`/`oddsLabel` computation) with:

```tsx
  if (!open) {
    return (
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setOpen(true);
        }}
        className={
          variant === "pill"
            ? "flex shrink-0 items-center gap-1 rounded-full bg-accent px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-accent/90"
            : "text-xs font-medium text-accent"
        }
      >
        {variant === "pill" && <Plus size={13} weight="bold" />}
        Log bet
      </button>
    );
  }

  const matchedCandidate = recommendation.candidates.find((c) => c.market === market && c.selection === selection);

  return (
    <LogBetModal
      open
      onClose={() => setOpen(false)}
      homeTeam={homeTeam}
      awayTeam={awayTeam}
      statusLabel={statusLabel}
      locked
      market={market}
      selection={selection}
      odds={matchedCandidate?.current_odds ?? null}
      onSubmit={async ({ stake }) => {
        const bet = await logBetFromRecommendation({ match_id: matchId, recommendation, market, selection, stake });
        setLoggedBet(bet);
        setSaveStatus("done");
        setOpen(false);
      }}
    />
  );
```

Delete the now-unused `submit()` function and the `stake`/`errorMsg` state (the modal owns stake entry and its own error display now) -- keep `saveStatus`/`loggedBet` (still needed for the `done` state) and `open`. `onClick={(e) => e.stopPropagation())}` on the `done`-state `<span>` and the unauthenticated `<Link>` stay unchanged (still needed so those clicks don't bubble to MatchCard's own expand handler).

- [ ] **Step 3: Thread `statusLabel`/`homeTeam`/`awayTeam` from both call sites**

In `MatchCard`'s call site (~line 1232), add the props, computed the same way the closing row already does (`match.home`/`match.away` are already real typed strings on `Match`):

```tsx
                <LogBetButton
                  matchId={match.id}
                  recommendation={match.rawRecommendation}
                  market={shown.market}
                  selection={shown.selection}
                  variant="pill"
                  homeTeam={match.home}
                  awayTeam={match.away}
                  statusLabel={`${day} · ${isCompleted ? "Full Time" : formatKickoff(match.kickoffIso)}`}
                />
```

(`day` and `isCompleted` are already in scope in `MatchCard` -- this is the exact formula `matchStatusLabel()` encapsulates; using it directly here, `matchStatusLabel(match.kickoffIso, isCompleted, asOf, sandboxMode)`, is equally correct and preferred if it reads cleaner -- either is fine, pick one and don't leave both forms in the codebase.)

In `MatchAnalysisPage` (which already receives `home`/`away` as its own real typed string props from the route), add `const { asOf, sandboxMode } = useSandboxAsOf();` (import `useSandboxAsOf` from `@/lib/useSandboxAsOf` if not already imported in this file) and pass `homeTeam={home}`, `awayTeam={away}`, `statusLabel={matchStatusLabel(match.kickoffIso, match.status === "completed", asOf, sandboxMode)}` down through to `ProbabilityRow` (its own prop signature gains all three: `statusLabel: string`, `homeTeam: string`, `awayTeam: string`), which passes all three straight through to `LogBetButton`.

- [ ] **Step 4: Update every test found in Step 1 to interact with the modal instead of the inline row**

For each: replace `getByPlaceholderText("Stake")` with `getByLabelText(/^stake$/i)`; replace `getByRole("button", { name: "Confirm" })` with `getByRole("button", { name: /confirm bet/i })`; the restated-terms assertion (`/home @ 2\.35/i`) becomes an assertion on the modal's locked Odds display text (`screen.getByText("2.35")` scoped appropriately, or the Pick display showing the selection's label). Tests exercising the Cancel-clears-stale-error behavior (Task 8, W215) still apply conceptually but now against the modal's own Cancel button and its own `errorMsg` state -- rewrite them the same way, verifying against `LogBetModal`'s actual behavior (already covered generically by Task 1's own test suite, but LogBetButton's integration of it -- does closing and reopening `LogBetButton`'s modal show a fresh, unmounted `LogBetModal` each time, given `{open && <LogBetModal .../>}`'s conditional-render nature naturally resets all of `LogBetModal`'s internal state on every close/reopen, unlike the old single-component inline-expand which needed an explicit `setSaveStatus("idle")` reset -- confirm this is genuinely true (a new mount always starts clean) and note it in a comment if the old explicit-reset test no longer has anything to prove, rather than deleting coverage silently).

Every test in the `"MatchCard quick-log control (W215/W217)"` describe block in `MatchUI.test.tsx` that asserts `getByPlaceholderText("Stake")` (proving the click opened something) should instead assert `screen.getByLabelText(/^stake$/i)` is present (the modal opened) -- the "does NOT toggle the card's own expand/collapse" test (checking `.expand-rows`'s `is-open` class) is unaffected by this change and needs no rewrite beyond the placeholder-text swap.

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx vitest run components/__tests__/LogBetButton.test.tsx components/__tests__/MatchUI.test.tsx`
Expected: PASS, every test updated in Step 4 accounted for.

- [ ] **Step 6: Run the full frontend suite, tsc, and build**

Run: `npx vitest run && npx tsc --noEmit && npx next build` (from `app/frontend/`)
Expected: all pass except the one pre-existing, unrelated flaky failure (`components/__tests__/BetTracker.race.test.tsx`) -- confirm no NEW failures.

- [ ] **Step 7: Manually verify against the running sandbox instance**

The local servers are already running in sandbox mode (SANDBOX_DATE=2026-09-14) -- restart the frontend only (`lsof -ti:3000 | xargs kill -9`, then relaunch via `./venv/bin/python scripts/launch_sandbox.py 2026-09-14` from the repo root, which restarts both -- or just the frontend dev process directly if you can target it without touching the backend) to pick up this change, then `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:3000` to confirm it's back up. This step is a sanity check, not a substitute for the automated tests above.

- [ ] **Step 8: Commit**

```bash
git add app/frontend/components/MatchUI.tsx app/frontend/components/__tests__/LogBetButton.test.tsx app/frontend/components/__tests__/MatchUI.test.tsx
git commit -m "feat(app): W218 -- LogBetButton opens the unified modal instead of expanding inline

Direct user mockup + request: every entry point through LogBetButton
(MatchCard's quick-log pill, ProbabilityRow's per-market picker on the
detail page) now opens LogBetModal in locked mode -- Market/Pick/Odds
shown as fixed text (already decided by the recommendation being
logged), only Stake editable. The trigger button/pill, the
unauthenticated sign-in link, and the settled-outcome 'done' state are
all unchanged.

MatchAnalysisPage gained its own useSandboxAsOf() call (previously
absent) so ProbabilityRow's LogBetButton gets the same real
'Today · Full Time'-style statusLabel MatchCard's own quick-log button
already computes, via the new shared matchStatusLabel() helper.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Rewire `ManualBetForm` to open the modal (editable mode)

**Files:**
- Modify: `app/frontend/components/BetTracker.tsx` (`ManualBetForm`)
- Test: `app/frontend/components/__tests__/BetTracker.fixtureError.test.tsx`

Depends on Task 1.

**Design:** The fixture SEARCH step (search box + results list) is unchanged. Once a fixture is `selected`, instead of rendering the inline field-grid + "Potential return" + filled button (all added in W217), `ManualBetForm` opens `<LogBetModal locked={false} .../>`. Closing the modal (Cancel/X/backdrop) returns to the search step with `selected` cleared (matching the existing "Change fixture" behavior), or -- simpler and arguably better -- keeps `selected` set and just closes the modal, letting the user reopen it without re-searching; pick whichever reads better once you see it rendered, but do not leave the user on a dead screen with no way to either log the bet or search again.

- [ ] **Step 1: Read the current state of `BetTracker.fixtureError.test.tsx` in full**

Search for every test that renders `ManualBetForm` through `BetTrackerPage` and interacts with the OLD inline field grid after selecting a fixture (`getByLabelText(/market/i)`, `getByLabelText(/outcome/i)`, the Odds/Stake plain inputs, the "Log bet" filled button, the "Potential return" text) -- catalog every one before changing any code, the same way Task 2's Step 1 does for the other file.

- [ ] **Step 2: Replace the selected-fixture branch with `LogBetModal`**

In `app/frontend/components/BetTracker.tsx`:
- Add `import { LogBetModal } from "./LogBetModal";` and remove the now-redundant local `MARKET_SELECTIONS` constant (delete it -- `LogBetModal.tsx` exports the one true copy now); update any other reference to `MARKET_SELECTIONS` in this file (`BetRow`'s edit form, W216) to `import { MARKET_SELECTIONS } from "./LogBetModal";` instead of using its own local copy.
- Delete `formatDateLong` from this file if `matchStatusLabel`/`formatDay`/`formatKickoff` (imported from `./MatchUI`) fully replace its one use site below; otherwise keep it if still needed elsewhere in this file (check before deleting).
- Replace the whole `) : (` branch (the selected-fixture header + field grid + Potential return + Log bet button, added in W217) with:

```tsx
      ) : (
        <LogBetModal
          open
          onClose={() => setSelected(null)}
          homeTeam={selected.home_team}
          awayTeam={selected.away_team}
          statusLabel={matchStatusLabel(selected.utc_date, selected.status === "FINISHED", asOf, sandboxMode)}
          locked={false}
          market="result_3way"
          selection=""
          odds={null}
          onSubmit={async ({ market, selection, odds, stake }) => {
            await logBetManual({
              match_id: selected.match_id, date: formatDate(selected.utc_date),
              home_team: selected.home_team, away_team: selected.away_team,
              market, selection, odds, stake,
            });
            setSelected(null);
            setQuery("");
            onLogged();
          }}
        />
      )}
```

(Import `matchStatusLabel` from `./MatchUI` alongside the file's existing MatchUI imports.) This replaces `submit()`, and the `market`/`selection`/`odds`/`stake`/`status`/`errorMsg` state in `ManualBetForm` -- delete all of those (the modal owns them now); keep `query`/`fixtures`/`selected`/`fixturesError`/`retryTick` (still needed for the search step) and the `onSessionExpired` prop threading: `LogBetModal`'s `onSubmit` throwing an `ApiError` with `status === 401` needs the same `onSessionExpired()` call the old `submit()` had -- catch it in the `onSubmit` callback here (not inside `LogBetModal`, which has no knowledge of `ApiError`/401 semantics) before re-throwing or handling it:

```tsx
          onSubmit={async ({ market, selection, odds, stake }) => {
            try {
              await logBetManual({
                match_id: selected.match_id, date: formatDate(selected.utc_date),
                home_team: selected.home_team, away_team: selected.away_team,
                market, selection, odds, stake,
              });
              setSelected(null);
              setQuery("");
              onLogged();
            } catch (err) {
              if (err instanceof ApiError && err.status === 401) {
                onSessionExpired();
                return;
              }
              throw err; // LogBetModal's own catch shows this inline
            }
          }}
```

- [ ] **Step 3: Update every test found in Step 1**

Same shape as Task 2 Step 4: swap `getByLabelText(/market/i)` → `getByLabelText(/^market$/i)` (LogBetModal's label, exact-match now that "Edit market" from `BetRow`'s own edit form -- unrelated, W216 -- could otherwise ambiguously match a loose `/market/i`), `getByLabelText(/outcome/i)` → `getByLabelText(/^pick$/i)` (label text changed from "Outcome" to "Pick" to match the mockup), Odds/Stake similarly. The "shows a boxed selected-fixture header" test (if any, from W217) now describes `LogBetModal`'s own header -- either delete it if fully covered by Task 1's `LogBetModal.test.tsx`, or keep one integration-level test here proving `ManualBetForm` actually wires the right `homeTeam`/`awayTeam`/`statusLabel` through (prefer keeping one such test -- it's the only thing Task 1's isolated component tests can't prove on their own).

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run components/__tests__/BetTracker.fixtureError.test.tsx`
Expected: PASS.

- [ ] **Step 5: Run the full frontend suite, tsc, and build**

Run: `npx vitest run && npx tsc --noEmit && npx next build` (from `app/frontend/`)
Expected: all pass except the one pre-existing, unrelated flaky failure (`BetTracker.race.test.tsx`).

- [ ] **Step 6: Commit**

```bash
git add app/frontend/components/BetTracker.tsx app/frontend/components/__tests__/BetTracker.fixtureError.test.tsx
git commit -m "feat(app): W218 -- ManualBetForm opens the unified modal after picking a fixture

Direct user mockup + request: the fixture search step is unchanged,
but selecting a result now opens LogBetModal in editable mode (Market/
Pick/Odds are all real controls, none pre-known) instead of W217's
inline field grid. MARKET_SELECTIONS' one true copy now lives in
LogBetModal.tsx (this file's own copy deleted, BetRow's edit form
(W216) now imports it from there too). A 401 from logBetManual still
routes to the page's existing needsAuth/onSessionExpired flow -- caught
in ManualBetForm's own onSubmit callback (LogBetModal has no ApiError/
401 awareness of its own) before re-throwing anything else for the
modal's own inline error display to show.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Final whole-branch review

- [ ] **Step 1: Read the combined diff of all 3 commits from this plan**

Run: `git log --oneline -4` to see them, `git diff <task-1-sha>~1..HEAD --stat` for the combined shape.

- [ ] **Step 2: Confirm no leftover dead code**

Grep the whole frontend for any remaining reference to the deleted inline-expand markup/state this plan removed (`getByPlaceholderText("Stake")` outside `LogBetModal.test.tsx`'s own new tests and any test file NOT touched by this plan that has its own unrelated reason to query a "Stake" placeholder -- check `app/frontend/components/__tests__/BetTracker.fixtureError.test.tsx`'s `BetRow`-edit-form tests, W216, which have their OWN separate Stake input inside `BetRow`'s inline edit form -- unrelated to this plan, must NOT be touched or broken).

- [ ] **Step 3: Run the full suite one final time as a whole**

Run: `npx vitest run && npx tsc --noEmit && npx next build` (from `app/frontend/`)
Expected: all pass except the one documented pre-existing flake.

- [ ] **Step 4: Manual smoke test against the running sandbox instance**

Restart the sandbox-mode servers (`./venv/bin/python scripts/launch_sandbox.py 2026-09-14` from the repo root) and verify via curl that both `:3000` and `:8000/api/sandbox/status` (expect `{"sandbox_mode":true,"as_of":"2026-09-14"}`) respond, confirming the branch's changes are live for the user to click through themselves.

---

## Self-Review Notes (for whoever executes this plan)

- **Spec coverage:** "All UX flows should take user to a small window on top of the page" -- Task 1 builds it, Tasks 2-3 wire every existing entry point (MatchCard quick-log, ProbabilityRow per-market picker, ManualBetForm) into it, none left on the old inline-expand UI. "When entering from a match card, the market, odds and pick will be pre populated" -- `locked` mode, Task 2.
- **Type consistency:** `LogBetFields` (`{ market, selection, odds, stake }`) is the one shape every `onSubmit` caller receives and must match -- `logBetFromRecommendation`'s call in Task 2 and `logBetManual`'s call in Task 3 both destructure from it identically.
- **Scope boundary:** `BetRow`'s own inline edit form (W216, editing an already-logged bet) is explicitly NOT part of this plan -- it's a different feature (editing history, not logging a new bet) and keeps its own existing inline UI; only its `MARKET_SELECTIONS` import source changes (still the same values, just re-exported from `LogBetModal.tsx` instead of locally defined in `BetTracker.tsx`).
