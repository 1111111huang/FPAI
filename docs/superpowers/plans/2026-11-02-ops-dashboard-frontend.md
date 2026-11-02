# Ops Dashboard Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new `/ops` page in the existing Next.js frontend — four tabs (failures & circuit breakers, odds availability, feature health, recommendation performance), one shared date-range + league filter bar above them, behind the same NextAuth login gate `/bets` already uses. The fourth tab links the existing, currently-orphaned `agent-performance` page into real navigation instead of rebuilding it.

**Architecture:** One page (`app/ops/page.tsx`) rendering one client component (`OpsDashboard.tsx`) that owns the shared filter state and switches between four tab components, each a thin fetch-and-render wrapper calling its own `lib/api.ts` function (`apiFetch`'s existing `X-App-Token` attachment, same pattern `getAgentPerformanceDashboard` already uses — no new auth code). Route-level auth is one line in `middleware.ts`'s existing `matcher` array, not a new auth system.

**Tech Stack:** Next.js (App Router), React, Tailwind (existing design tokens: `bg-surface`, `border-border`, `text-ink`, `text-muted`, `text-good`/`text-serious`), `@phosphor-icons/react`.

---

### Task 1: Gate `/ops` behind the existing login middleware

**Files:**
- Modify: `app/frontend/middleware.ts`

- [ ] **Step 1: Read the current matcher**

Run: `cat app/frontend/middleware.ts`

Confirm the current `config.matcher` is `["/bets/:path*"]` — `withAuth`'s own `authorized({ token }) { return !!token; }` callback (any signed-in, allowlisted Google account) is reused as-is; no new auth logic needed, only a wider matcher.

- [ ] **Step 2: Add `/ops` to the matcher**

```typescript
export const config = {
  matcher: ["/bets/:path*", "/ops/:path*"],
};
```

- [ ] **Step 3: Commit**

```bash
git add app/frontend/middleware.ts
git commit -m "feat: gate /ops behind the existing login middleware"
```

(No automated test for Next.js middleware matching in this codebase's existing suite — verified manually in Task 6's smoke check instead.)

---

### Task 2: `lib/api.ts` — three new fetch functions + types

**Files:**
- Modify: `app/frontend/lib/api.ts`
- Modify: `app/frontend/lib/types.ts`
- Test: `app/frontend/lib/__tests__/api.test.ts` (check this file's existing structure first: `grep -n "getAgentPerformanceDashboard\|describe(" app/frontend/lib/__tests__/api.test.ts`)

- [ ] **Step 1: Write the failing tests**

Add to `app/frontend/lib/__tests__/api.test.ts`, matching its existing `fetch` mock pattern (check the file for the exact mock-setup helper it already uses before writing these — likely a `beforeEach` that stubs global `fetch`):

```typescript
describe("getOpsFailures", () => {
  it("fetches failures and circuit breaker status", async () => {
    const mockResponse = { failures: [], circuit_breakers: { the_odds_api: { state: "closed" }, oddspapi: { state: "closed" } } };
    global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => mockResponse });

    const result = await getOpsFailures({ league: "I1" });

    expect(result).toEqual(mockResponse);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/admin/ops/failures?league=I1"),
      expect.any(Object)
    );
  });
});

describe("getOpsOddsAvailability", () => {
  it("fetches odds availability for a date range", async () => {
    const mockResponse = { fixtures: [] };
    global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => mockResponse });

    const result = await getOpsOddsAvailability({ dateFrom: "2026-11-01", dateTo: "2026-11-02", league: "E0" });

    expect(result).toEqual(mockResponse);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/admin/ops/odds-availability?date_from=2026-11-01&date_to=2026-11-02&league=E0"),
      expect.any(Object)
    );
  });
});

describe("getOpsFeatureHealth", () => {
  it("fetches feature health for a date range", async () => {
    const mockResponse = { by_league: {} };
    global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => mockResponse });

    const result = await getOpsFeatureHealth({ dateFrom: "2026-11-01", dateTo: "2026-11-02" });

    expect(result).toEqual(mockResponse);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd app/frontend && npx jest lib/__tests__/api.test.ts -t "getOps"`
Expected: FAIL — `getOpsFailures is not defined`

- [ ] **Step 3: Add the types**

Append to `app/frontend/lib/types.ts`:

```typescript
export interface FailureRow {
  id: number;
  ts: string;
  subsystem: string;
  unit_id: string;
  league: string | null;
  exc_type: string;
  message: string;
  traceback: string;
  retryable: boolean | null;
  alerted: boolean;
}

export interface CircuitBreakerStatus {
  state: "closed" | "open" | "half_open";
}

export interface OpsFailuresResponse {
  failures: FailureRow[];
  circuit_breakers: {
    the_odds_api: CircuitBreakerStatus;
    oddspapi: CircuitBreakerStatus;
  };
}

export interface OddsAvailabilityFixture {
  match_id: string;
  date: string;
  league: string | null;
  home_team: string | null;
  away_team: string | null;
  missing_markets: string[];
  likely_cause: "circuit_open" | "not_posted";
}

export interface OpsOddsAvailabilityResponse {
  fixtures: OddsAvailabilityFixture[];
}

export interface FeatureHealthLeagueStats {
  sample_size: number;
  cold_start_risk_rate: number;
  unknown_team_rate: number;
  avg_feature_completeness: number | null;
}

export interface OpsFeatureHealthResponse {
  by_league: Record<string, FeatureHealthLeagueStats>;
}
```

- [ ] **Step 4: Implement the fetch functions**

Append to `app/frontend/lib/api.ts`:

```typescript
import type {
  OpsFailuresResponse, OpsOddsAvailabilityResponse, OpsFeatureHealthResponse,
} from "./types"; // merge into this file's existing type import if one already exists

export async function getOpsFailures(opts: { since?: string; league?: string; subsystem?: string } = {}): Promise<OpsFailuresResponse> {
  const params = new URLSearchParams();
  if (opts.since) params.set("since", opts.since);
  if (opts.league) params.set("league", opts.league);
  if (opts.subsystem) params.set("subsystem", opts.subsystem);
  const query = params.toString();
  const response = await apiFetch(`/api/admin/ops/failures${query ? `?${query}` : ""}`);
  if (!response.ok) {
    throw new ApiError(`Failed to load ops failures (${response.status})`, response.status);
  }
  return response.json();
}

export async function getOpsOddsAvailability(opts: { dateFrom: string; dateTo: string; league?: string }): Promise<OpsOddsAvailabilityResponse> {
  const params = new URLSearchParams({ date_from: opts.dateFrom, date_to: opts.dateTo });
  if (opts.league) params.set("league", opts.league);
  const response = await apiFetch(`/api/admin/ops/odds-availability?${params.toString()}`);
  if (!response.ok) {
    throw new ApiError(`Failed to load odds availability (${response.status})`, response.status);
  }
  return response.json();
}

export async function getOpsFeatureHealth(opts: { dateFrom: string; dateTo: string; league?: string }): Promise<OpsFeatureHealthResponse> {
  const params = new URLSearchParams({ date_from: opts.dateFrom, date_to: opts.dateTo });
  if (opts.league) params.set("league", opts.league);
  const response = await apiFetch(`/api/admin/ops/feature-health?${params.toString()}`);
  if (!response.ok) {
    throw new ApiError(`Failed to load feature health (${response.status})`, response.status);
  }
  return response.json();
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd app/frontend && npx jest lib/__tests__/api.test.ts -t "getOps"`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add app/frontend/lib/api.ts app/frontend/lib/types.ts app/frontend/lib/__tests__/api.test.ts
git commit -m "feat: add lib/api.ts functions for the three new ops endpoints"
```

---

### Task 3: Shared filter bar + `OpsDashboard` shell

**Files:**
- Create: `app/frontend/components/OpsDashboard.tsx`
- Create: `app/frontend/app/ops/page.tsx`
- Test: `app/frontend/components/__tests__/OpsDashboard.test.tsx`

- [ ] **Step 1: Write the failing test**

Check this codebase's existing component test pattern first:

Run: `grep -n "render(\|screen\.\|describe(" app/frontend/components/__tests__/AgentPerformanceDashboard.test.tsx 2>/dev/null | head -10`

(If no such test file exists yet for `AgentPerformanceDashboard`, use `app/frontend/components/__tests__/MatchUI.test.tsx`'s own `render`/`screen` pattern instead.)

```typescript
import { render, screen, fireEvent } from "@testing-library/react";
import { OpsDashboard } from "../OpsDashboard";

jest.mock("@/lib/api", () => ({
  getOpsFailures: jest.fn().mockResolvedValue({ failures: [], circuit_breakers: { the_odds_api: { state: "closed" }, oddspapi: { state: "closed" } } }),
  getOpsOddsAvailability: jest.fn().mockResolvedValue({ fixtures: [] }),
  getOpsFeatureHealth: jest.fn().mockResolvedValue({ by_league: {} }),
  getAgentPerformanceDashboard: jest.fn().mockResolvedValue({ main_metrics: {}, by_market_metrics: [], by_market_selection_metrics: [], by_league_metrics: [], top_bets: [], bottom_bets: [] }),
}));

describe("OpsDashboard", () => {
  it("renders all four tab labels and defaults to the failures tab", async () => {
    render(<OpsDashboard />);
    expect(await screen.findByText(/failures/i)).toBeInTheDocument();
    expect(screen.getByText(/odds availability/i)).toBeInTheDocument();
    expect(screen.getByText(/feature health/i)).toBeInTheDocument();
    expect(screen.getByText(/performance/i)).toBeInTheDocument();
  });

  it("switches tabs on click", async () => {
    render(<OpsDashboard />);
    await screen.findByText(/failures/i);
    fireEvent.click(screen.getByRole("tab", { name: /odds availability/i }));
    expect(screen.getByRole("tab", { name: /odds availability/i })).toHaveAttribute("aria-selected", "true");
  });

  it("renders a league filter that defaults to all leagues", async () => {
    render(<OpsDashboard />);
    await screen.findByText(/failures/i);
    expect(screen.getByLabelText(/league/i)).toHaveValue("");
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd app/frontend && npx jest components/__tests__/OpsDashboard.test.tsx`
Expected: FAIL — `Cannot find module '../OpsDashboard'`

- [ ] **Step 3: Implement `OpsDashboard.tsx`**

```tsx
"use client";

/** 2026-11-02: /ops -- failures & circuit breakers, odds availability,
 * feature health, and recommendation performance (the existing, previously
 * unlinked agent-performance page, folded in here instead of rebuilt).
 * One shared date-range + league filter bar above all four tabs, so they
 * can't drift out of sync with each other.
 * Design: docs/superpowers/specs/2026-11-02-background-job-safety-net-design.md */

import { useState } from "react";

import { AppShell } from "./AppShell";
import { FailuresTab } from "./ops/FailuresTab";
import { OddsAvailabilityTab } from "./ops/OddsAvailabilityTab";
import { FeatureHealthTab } from "./ops/FeatureHealthTab";
import { AgentPerformancePage } from "./AgentPerformanceDashboard";

type TabKey = "failures" | "odds" | "features" | "performance";

const TABS: { key: TabKey; label: string }[] = [
  { key: "failures", label: "Failures" },
  { key: "odds", label: "Odds Availability" },
  { key: "features", label: "Feature Health" },
  { key: "performance", label: "Performance" },
];

const LEAGUES = ["E0", "SP1", "I1", "D1", "F1"];

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}
function daysAgoIso(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
}

export function OpsDashboard() {
  const [tab, setTab] = useState<TabKey>("failures");
  const [league, setLeague] = useState<string>("");
  const [dateFrom, setDateFrom] = useState<string>(daysAgoIso(7));
  const [dateTo, setDateTo] = useState<string>(todayIso());

  return (
    <AppShell active="ops">
      <div className="mx-auto max-w-5xl p-4">
        <h1 className="text-xl font-semibold text-ink">Ops Dashboard</h1>

        <div className="mt-4 flex flex-wrap items-center gap-3 rounded-lg border border-border bg-surface p-3 text-sm">
          <label className="flex items-center gap-2">
            <span className="text-muted">From</span>
            <input
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              className="rounded border border-border bg-bg px-2 py-1 text-ink"
            />
          </label>
          <label className="flex items-center gap-2">
            <span className="text-muted">To</span>
            <input
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              className="rounded border border-border bg-bg px-2 py-1 text-ink"
            />
          </label>
          <label className="flex items-center gap-2">
            <span className="text-muted">League</span>
            <select
              aria-label="League"
              value={league}
              onChange={(e) => setLeague(e.target.value)}
              className="rounded border border-border bg-bg px-2 py-1 text-ink"
            >
              <option value="">All</option>
              {LEAGUES.map((l) => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>
          </label>
        </div>

        <div role="tablist" className="mt-4 flex gap-1 border-b border-border">
          {TABS.map((t) => (
            <button
              key={t.key}
              role="tab"
              aria-selected={tab === t.key}
              onClick={() => setTab(t.key)}
              className={`px-3 py-2 text-sm ${
                tab === t.key ? "border-b-2 border-ink font-medium text-ink" : "text-muted hover:text-ink"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div className="mt-4">
          {tab === "failures" && <FailuresTab league={league || undefined} />}
          {tab === "odds" && <OddsAvailabilityTab dateFrom={dateFrom} dateTo={dateTo} league={league || undefined} />}
          {tab === "features" && <FeatureHealthTab dateFrom={dateFrom} dateTo={dateTo} league={league || undefined} />}
          {tab === "performance" && <AgentPerformancePage />}
        </div>
      </div>
    </AppShell>
  );
}
```

- [ ] **Step 4: Create the page route**

```tsx
import { OpsDashboard } from "@/components/OpsDashboard";

export default function Page() {
  return <OpsDashboard />;
}
```

Save as `app/frontend/app/ops/page.tsx`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd app/frontend && npx jest components/__tests__/OpsDashboard.test.tsx`
Expected: still FAIL at this point — `./ops/FailuresTab`, `./ops/OddsAvailabilityTab`, `./ops/FeatureHealthTab` don't exist yet (Tasks 4-6 create them). Confirm the failure is specifically those three missing imports, not something else, before moving on.

- [ ] **Step 6: Commit** (deferred to the end of Task 6, once the three tab components it imports actually exist and the test can pass for real)

---

### Task 4: `FailuresTab`

**Files:**
- Create: `app/frontend/components/ops/FailuresTab.tsx`
- Test: `app/frontend/components/ops/__tests__/FailuresTab.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
import { render, screen } from "@testing-library/react";
import { FailuresTab } from "../FailuresTab";
import { getOpsFailures } from "@/lib/api";

jest.mock("@/lib/api");

describe("FailuresTab", () => {
  it("renders failure rows and circuit breaker state", async () => {
    (getOpsFailures as jest.Mock).mockResolvedValue({
      failures: [{
        id: 1, ts: "2026-10-31T11:07:54Z", subsystem: "pregenerate", unit_id: "league=I1",
        league: "I1", exc_type: "JSONDecodeError", message: "Expecting value: line 1 column 1 (char 0)",
        traceback: "...", retryable: null, alerted: true,
      }],
      circuit_breakers: { the_odds_api: { state: "open" }, oddspapi: { state: "closed" } },
    });

    render(<FailuresTab />);

    expect(await screen.findByText(/JSONDecodeError/)).toBeInTheDocument();
    expect(screen.getByText(/league=I1/)).toBeInTheDocument();
    expect(screen.getByText(/the_odds_api/i)).toBeInTheDocument();
    expect(screen.getByText(/open/i)).toBeInTheDocument();
  });

  it("renders an empty state with no failures", async () => {
    (getOpsFailures as jest.Mock).mockResolvedValue({
      failures: [], circuit_breakers: { the_odds_api: { state: "closed" }, oddspapi: { state: "closed" } },
    });

    render(<FailuresTab />);

    expect(await screen.findByText(/no failures/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd app/frontend && npx jest components/ops/__tests__/FailuresTab.test.tsx`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `FailuresTab.tsx`**

```tsx
"use client";

import { useEffect, useState } from "react";

import { getOpsFailures } from "@/lib/api";
import type { OpsFailuresResponse } from "@/lib/types";

function BreakerPill({ vendor, state }: { vendor: string; state: string }) {
  const color = state === "open" ? "bg-serious/10 text-serious" : state === "half_open" ? "bg-amber-500/10 text-amber-600" : "bg-good/10 text-good";
  return (
    <div className={`rounded-full px-3 py-1 text-xs font-mono font-semibold ${color}`}>
      {vendor}: {state}
    </div>
  );
}

export function FailuresTab({ league }: { league?: string }) {
  const [data, setData] = useState<OpsFailuresResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getOpsFailures({ league })
      .then((result) => { if (!cancelled) setData(result); })
      .catch((err) => { if (!cancelled) setError(err.message ?? "Failed to load failures"); });
    return () => { cancelled = true; };
  }, [league]);

  if (error) return <div className="text-sm text-serious">{error}</div>;
  if (!data) return <div className="text-sm text-muted">Loading...</div>;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-2">
        <BreakerPill vendor="the_odds_api" state={data.circuit_breakers.the_odds_api.state} />
        <BreakerPill vendor="oddspapi" state={data.circuit_breakers.oddspapi.state} />
      </div>

      {data.failures.length === 0 ? (
        <div className="rounded-lg border border-border bg-surface p-4 text-sm text-muted">No failures in this window.</div>
      ) : (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-muted">
              <th className="pb-2">Time</th>
              <th className="pb-2">Subsystem</th>
              <th className="pb-2">Unit</th>
              <th className="pb-2">League</th>
              <th className="pb-2">Error</th>
            </tr>
          </thead>
          <tbody>
            {data.failures.map((f) => (
              <tr key={f.id} className="border-t border-border">
                <td className="py-2 font-mono">{f.ts}</td>
                <td className="py-2">{f.subsystem}</td>
                <td className="py-2 font-mono">{f.unit_id}</td>
                <td className="py-2">{f.league ?? "—"}</td>
                <td className="py-2 font-mono text-serious">{f.exc_type}: {f.message}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd app/frontend && npx jest components/ops/__tests__/FailuresTab.test.tsx`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/frontend/components/ops/FailuresTab.tsx app/frontend/components/ops/__tests__/FailuresTab.test.tsx
git commit -m "feat: add FailuresTab component"
```

---

### Task 5: `OddsAvailabilityTab`

**Files:**
- Create: `app/frontend/components/ops/OddsAvailabilityTab.tsx`
- Test: `app/frontend/components/ops/__tests__/OddsAvailabilityTab.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
import { render, screen } from "@testing-library/react";
import { OddsAvailabilityTab } from "../OddsAvailabilityTab";
import { getOpsOddsAvailability } from "@/lib/api";

jest.mock("@/lib/api");

describe("OddsAvailabilityTab", () => {
  it("renders fixtures with missing markets and cause", async () => {
    (getOpsOddsAvailability as jest.Mock).mockResolvedValue({
      fixtures: [{
        match_id: "558542", date: "2026-11-02", league: "I1",
        home_team: "Udinese", away_team: "Roma",
        missing_markets: ["over/under 2.5"], likely_cause: "circuit_open",
      }],
    });

    render(<OddsAvailabilityTab dateFrom="2026-11-01" dateTo="2026-11-02" />);

    expect(await screen.findByText(/Udinese/)).toBeInTheDocument();
    expect(screen.getByText(/over\/under 2.5/)).toBeInTheDocument();
    expect(screen.getByText(/circuit_open/)).toBeInTheDocument();
  });

  it("renders an empty state with nothing missing", async () => {
    (getOpsOddsAvailability as jest.Mock).mockResolvedValue({ fixtures: [] });

    render(<OddsAvailabilityTab dateFrom="2026-11-01" dateTo="2026-11-02" />);

    expect(await screen.findByText(/no missing markets/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd app/frontend && npx jest components/ops/__tests__/OddsAvailabilityTab.test.tsx`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `OddsAvailabilityTab.tsx`**

```tsx
"use client";

import { useEffect, useState } from "react";

import { getOpsOddsAvailability } from "@/lib/api";
import type { OpsOddsAvailabilityResponse } from "@/lib/types";

export function OddsAvailabilityTab({ dateFrom, dateTo, league }: { dateFrom: string; dateTo: string; league?: string }) {
  const [data, setData] = useState<OpsOddsAvailabilityResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getOpsOddsAvailability({ dateFrom, dateTo, league })
      .then((result) => { if (!cancelled) setData(result); })
      .catch((err) => { if (!cancelled) setError(err.message ?? "Failed to load odds availability"); });
    return () => { cancelled = true; };
  }, [dateFrom, dateTo, league]);

  if (error) return <div className="text-sm text-serious">{error}</div>;
  if (!data) return <div className="text-sm text-muted">Loading...</div>;
  if (data.fixtures.length === 0) {
    return <div className="rounded-lg border border-border bg-surface p-4 text-sm text-muted">No missing markets in this window.</div>;
  }

  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-left text-muted">
          <th className="pb-2">Match</th>
          <th className="pb-2">League</th>
          <th className="pb-2">Missing</th>
          <th className="pb-2">Likely cause</th>
        </tr>
      </thead>
      <tbody>
        {data.fixtures.map((f) => (
          <tr key={f.match_id} className="border-t border-border">
            <td className="py-2">{f.home_team} v {f.away_team}</td>
            <td className="py-2">{f.league ?? "—"}</td>
            <td className="py-2">{f.missing_markets.join(", ")}</td>
            <td className={`py-2 font-mono ${f.likely_cause === "circuit_open" ? "text-serious" : "text-muted"}`}>
              {f.likely_cause}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd app/frontend && npx jest components/ops/__tests__/OddsAvailabilityTab.test.tsx`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/frontend/components/ops/OddsAvailabilityTab.tsx app/frontend/components/ops/__tests__/OddsAvailabilityTab.test.tsx
git commit -m "feat: add OddsAvailabilityTab component"
```

---

### Task 6: `FeatureHealthTab`

**Files:**
- Create: `app/frontend/components/ops/FeatureHealthTab.tsx`
- Test: `app/frontend/components/ops/__tests__/FeatureHealthTab.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
import { render, screen } from "@testing-library/react";
import { FeatureHealthTab } from "../FeatureHealthTab";
import { getOpsFeatureHealth } from "@/lib/api";

jest.mock("@/lib/api");

describe("FeatureHealthTab", () => {
  it("renders per-league feature health stats", async () => {
    (getOpsFeatureHealth as jest.Mock).mockResolvedValue({
      by_league: { I1: { sample_size: 12, cold_start_risk_rate: 0.25, unknown_team_rate: 0.0, avg_feature_completeness: 0.82 } },
    });

    render(<FeatureHealthTab dateFrom="2026-11-01" dateTo="2026-11-02" />);

    expect(await screen.findByText("I1")).toBeInTheDocument();
    expect(screen.getByText(/82/)).toBeInTheDocument();
    expect(screen.getByText(/25/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd app/frontend && npx jest components/ops/__tests__/FeatureHealthTab.test.tsx`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `FeatureHealthTab.tsx`**

```tsx
"use client";

import { useEffect, useState } from "react";

import { getOpsFeatureHealth } from "@/lib/api";
import type { OpsFeatureHealthResponse } from "@/lib/types";

function formatPct(v: number): string {
  return `${(v * 100).toFixed(0)}%`;
}

export function FeatureHealthTab({ dateFrom, dateTo, league }: { dateFrom: string; dateTo: string; league?: string }) {
  const [data, setData] = useState<OpsFeatureHealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getOpsFeatureHealth({ dateFrom, dateTo, league })
      .then((result) => { if (!cancelled) setData(result); })
      .catch((err) => { if (!cancelled) setError(err.message ?? "Failed to load feature health"); });
    return () => { cancelled = true; };
  }, [dateFrom, dateTo, league]);

  if (error) return <div className="text-sm text-serious">{error}</div>;
  if (!data) return <div className="text-sm text-muted">Loading...</div>;

  const rows = Object.entries(data.by_league);
  if (rows.length === 0) {
    return <div className="rounded-lg border border-border bg-surface p-4 text-sm text-muted">No recommendations in this window.</div>;
  }

  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-left text-muted">
          <th className="pb-2">League</th>
          <th className="pb-2 text-right">Sample</th>
          <th className="pb-2 text-right">Avg completeness</th>
          <th className="pb-2 text-right">Cold-start rate</th>
          <th className="pb-2 text-right">Unknown-team rate</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([leagueCode, stats]) => (
          <tr key={leagueCode} className="border-t border-border">
            <td className="py-2 font-semibold">{leagueCode}</td>
            <td className="py-2 text-right font-mono">{stats.sample_size}</td>
            <td className="py-2 text-right font-mono">{stats.avg_feature_completeness !== null ? formatPct(stats.avg_feature_completeness) : "—"}</td>
            <td className="py-2 text-right font-mono">{formatPct(stats.cold_start_risk_rate)}</td>
            <td className="py-2 text-right font-mono">{formatPct(stats.unknown_team_rate)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 4: Run every ops test together, including `OpsDashboard`'s own (deferred from Task 3)**

Run: `cd app/frontend && npx jest components/ops components/__tests__/OpsDashboard.test.tsx`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/frontend/components/ops/FeatureHealthTab.tsx app/frontend/components/ops/__tests__/FeatureHealthTab.test.tsx app/frontend/components/OpsDashboard.tsx app/frontend/app/ops/page.tsx app/frontend/components/__tests__/OpsDashboard.test.tsx
git commit -m "feat: add FeatureHealthTab and wire OpsDashboard's four tabs together"
```

---

### Task 7: Add `/ops` to `AppShell`'s nav

**Files:**
- Modify: `app/frontend/components/AppShell.tsx`

This is the actual fix for the `agent-performance` page's own W174 problem — linking it (via the Performance tab inside `/ops`) into real navigation instead of leaving it reachable only by a direct URL nobody remembers.

- [ ] **Step 1: Read the current `NAV_ITEMS` and `key` type**

Run: `grep -n "key: \"dashboard\"\|NAV_ITEMS" app/frontend/components/AppShell.tsx`

- [ ] **Step 2: Add the `ops` nav entry**

Widen the `key` union type:

```typescript
key: "dashboard" | "matches" | "bets" | "ops";
```

Add to `NAV_ITEMS` (confirm `Gauge` exists in the installed `@phosphor-icons/react` version — `grep -n "export.*Gauge" app/frontend/node_modules/@phosphor-icons/react/dist/index.d.ts`; if it doesn't, substitute any other already-imported-elsewhere icon rather than guessing a new one):

```typescript
import { Gauge, House, List, ListBullets, MagnifyingGlass, Ticket, X, type Icon } from "@phosphor-icons/react";

// ...

{ href: "/ops", label: "Ops", key: "ops", icon: Gauge },
```

- [ ] **Step 3: Run the existing AppShell tests to confirm no regressions**

Run: `cd app/frontend && npx jest components/__tests__/AppShell.test.tsx`
Expected: all PASS (adding a nav item shouldn't break any existing nav-item-count or label assertions — if one does assert an exact count, update it to account for the new item)

- [ ] **Step 4: Commit**

```bash
git add app/frontend/components/AppShell.tsx
git commit -m "feat: link /ops into AppShell navigation"
```

---

### Task 8: Full regression pass

**Files:** none (verification only)

- [ ] **Step 1: Run the full frontend test suite**

Run: `cd app/frontend && npx jest`
Expected: no new failures

- [ ] **Step 2: Manual smoke check**

Run the frontend locally (`npm run dev` in `app/frontend`), sign in, navigate to `/ops`, confirm: all four tabs render, switching tabs works, the date/league filter bar updates the failures/odds/feature-health tabs' data on change, and the Performance tab shows the same content `/agent-performance` already does.

---

## Self-review notes

- **Spec coverage**: implements all of §7 (the four-tab dashboard, shared filter bar, access model) plus explicitly closes the W174 "built but never linked" gap flagged as the risk to avoid repeating.
- **Dependency on prior plans**: requires Plan 3's three new backend endpoints to exist; Task 5 of Plan 3 is read before writing this plan's filter-bar-to-performance-tab wiring, since that endpoint takes `days` (a count), not `date_from`/`date_to` like the other three — `AgentPerformancePage` is rendered here with its own internal filters unchanged (not fed by this page's shared filter bar) rather than silently passing a mismatched prop shape into it.
- **Type consistency**: `FailureRow`/`OddsAvailabilityFixture`/`FeatureHealthLeagueStats` (Task 2) match the exact JSON shapes Plan 3's endpoints return field-for-field.
- **Known gap, not silently dropped**: the Performance tab's filters are independent of the shared date/league bar above it (noted above) — unifying them would mean either changing the existing, working `/api/recommendations/performance-dashboard` endpoint's parameter shape (out of scope, risks regressing `agent-performance`'s existing behavior) or a client-side `days`-from-`dateFrom` translation layer. Left as a follow-up, not bundled into this plan.
