# FPAI Web App — Design Discussions & User Stories

This document tracks the third FPAI component: a bettor-facing web app (`app/`) built on top of the ML forecasting engine (`FPAI_PRD.md`) and the LangGraph betting agent (`agent_prd.md`). Stories are prefixed `W##` to distinguish them from the ML forecasting engine (`US#`) and agent (`A##`) stories.

**Status:** functionality is still being worked out. Unlike the other two story documents, this one leads with an open discussion log. Nothing in the Open Design Discussions section is a decision — items only graduate into the Stories table once we've actually agreed on them. `documents/app_prd.md` and `documents/app_techspec.md` are drafts and will be filled in as these discussions resolve.

---

## Confirmed So Far

These came out of initial brainstorming and are treated as settled unless something below reopens them:

- **Primary user:** an individual bettor (not an internal analyst dashboard) — same user described in `agent_prd.md` Section 2.1.
- **Tech stack:** Next.js/React + TypeScript frontend, FastAPI backend wrapping the existing Python agent/forecast code.
- **Auth:** single-user, no accounts, for the MVP.
- **Bet tracker:** logs bets the user actually placed (not automatic hypothetical tracking).
- **"Why" explanation:** expandable static detail from the already-generated payload — no new on-demand LLM call.
- **Directory:** new top-level `app/` (vs. `frontend/`, `webapp/`, `dashboard/`).

---

## Open Design Discussions

### D1: Fixture discovery — how does the app know which upcoming matches to list?
The agent today only accepts a single `--home`/`--away`/`--date` triple via `agent-recommend`; there's no "list upcoming fixtures for a league" capability anywhere in the codebase. Before we can design a match-list screen, we need to decide how fixtures get discovered.

Options raised so far:
- (a) No fixture list in MVP — user manually enters home/away/date/league to request a recommendation.
- (b) New backend fixture-discovery step reusing the agent's existing web-search tool.
- (c) A fixtures table populated ahead of time from a data source.

**Status: open.**

### D2: Recommendation generation latency & caching
A live agent run (LLM + Tavily search) takes roughly 10–30s per match today. Need to decide: generate on-demand per page view, pre-generate on a schedule, or cache aggressively per match+date. This affects both backend architecture and frontend loading-state UX.

**Status: open.**

### D3: Bet tracker lifecycle & data model
Need to nail down: exact fields, status values (open/won/lost/void/push?), how/when auto-settlement against `raw_matches` runs, whether a logged bet must reference a generated recommendation or can be entered freeform, whether bets are editable/deletable after logging, and whether starting bankroll is fixed or configurable.

**Status: open.**

### D4: "Why" panel scope
Confirmed it's static (no new LLM call), but not yet decided exactly which fields render — explanation text alone, or explanation + top_features + confidence + limitations together; per-market only, or an overall summary too.

**Status: open.**

### D5: Backend/business-logic boundary
Should new glue logic (recommendation caching, fixture discovery) live in `app/backend` as app-specific code, or in `src/agent`/`src/forecast` as reusable engine capability the CLI could also use later? Where's the line between "the app" and "the engine"?

**Status: open.**

### D6: Visual design / component library
No UI language decided yet — component library (e.g. Tailwind/shadcn), layout, mobile vs. desktop priority. A browser-based visual companion tool is available for mockups/comparisons once we get here, if useful.

**Status: open.**

### D7: Local dev / deployment shape
docker-compose vs. documented manual run steps for FastAPI + Next.js + the existing DuckDB file; how secrets (`TAVILY_API_KEY`, model provider keys) are handled in a web-server context vs. the current CLI-only usage.

**Status: open.**

---

## Story Dependencies & Execution Order

Not yet defined — phases and dependencies will be added here once enough of the Open Design Discussions above resolve into concrete, buildable decisions.

---

## Stories

No stories yet. Discussion items above move here once resolved, following the `agent_user_stories.md` format (ID, status, description with acceptance criteria, size/milestone/dependency comments).

| ID | Status | Description | Comments |
|---|---|---|---|
| — | — | — | — |
