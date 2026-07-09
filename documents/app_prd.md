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
Single-user, no accounts, for the initial version. Revisit if/when multi-user support is needed.

## 3. Core Capabilities

TBD — pending resolution of Open Design Discussions D1–D4 in `documents/app_user_stories.md` (fixture discovery, recommendation caching, bet tracker data model, "why" panel scope).

## 4. Output Requirements

TBD.

## 5. Non-Goals

Provisional, subject to revision:

- Multi-user accounts or authentication.
- Payment processing.
- Native mobile app.
- Live odds feed beyond what the agent's existing web-search tool provides.
- Production hosting/deployment (local/dev use only for now).

## 6. Roadmap

TBD — will mirror the phase structure once it's established in `documents/app_user_stories.md`.
