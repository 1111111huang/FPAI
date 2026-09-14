# Product Requirements Document - FPAI Web App

> **DRAFT — functionality under active discussion.** This document only records what has actually been agreed. See `documents/app_user_stories.md` for the open design-discussion log; sections below will be filled in as those discussions resolve.

## 1. Product Objective

A bettor-facing web app that surfaces the FPAI betting agent's (`agent_prd.md`) per-match recommendations and lets the user track bets they actually place against those recommendations. Scope beyond this is not yet settled — see `documents/app_user_stories.md`.

## 2. Product Positioning

### 2.1 Primary User
An individual bettor — the same user described in `agent_prd.md` Section 2.1 — using the web app directly instead of the CLI.

### 2.2 Relationship to Other Components
This app is a consumer of the forecasting engine (`FPAI_PRD.md`) and betting agent (`agent_prd.md`). It does not change their product contracts.

### 2.3 Auth Model
**Implemented 2026-09-14 (W210).** Real multi-user accounts: each user gets their own Google login and their own bet-tracking data (bets, settlement, stats scoped by user, not global) — the shared recommendations dashboard itself stays unauthenticated, unchanged, and free of any new per-user API-cost exposure. Google OAuth only, invite-only via a `signIn` callback email allowlist (`ALLOWED_EMAILS`), not open signup. Auth.js (NextAuth v4) runs entirely in the Next.js frontend; FastAPI never talks to Google or verifies JWTs — server-side proxy routes call `getServerSession()` and forward the caller's email to FastAPI via an `X-User-Email` header authenticated by a separate `INTERNAL_API_SECRET`. `APP_ACCESS_TOKEN` (W97) is narrowed to `/api/admin/*` only; every bet route requires real per-user auth. Pre-existing single-user `user_bets` rows migrate to one owner account via `scripts/migrate_bets_to_owner.py` (idempotent, run for real 2026-09-14 — see `documents/app_user_stories.md` W210 completion notes for the full implementation and verification record). Supersedes the original single-user/no-accounts decision below, kept for history: ~~Single-user, no accounts, for the initial version. Revisit if/when multi-user support is needed.~~

## 3. Core Capabilities

TBD — pending resolution of Open Design Discussions D1–D4 in `documents/app_user_stories.md` (fixture discovery, recommendation caching, bet tracker data model, "why" panel scope).

## 4. Output Requirements

TBD.

## 5. Non-Goals

Provisional, subject to revision:

- Payment processing.
- Native mobile app.
- Live odds feed beyond what the agent's existing web-search tool provides.
- Production hosting/deployment (local/dev use only for now).

## 6. Roadmap

TBD — will mirror the phase structure once it's established in `documents/app_user_stories.md`.
