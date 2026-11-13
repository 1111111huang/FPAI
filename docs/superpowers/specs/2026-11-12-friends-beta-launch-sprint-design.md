# Friends-Beta Launch Sprint — Design

**Date:** 2026-11-12
**Status:** Approved. Tracked as `app_user_stories.md` PHASE 63 (W255-W261).

## Problem

User wants to share FPAI with friends as a beta within 2-3 weeks. Five known issues named: live match status lag, under-tested bet logging, missing odds, limited league coverage with no use during international breaks, and unintuitive match-browsing UX — plus a request to borrow ideas from competitors (Score AI) to enrich the product.

Five issues plus competitor research is too much to design individually in one pass. This doc scopes it as a prioritized sprint backlog, not five feature specs.

## Investigation findings that shaped the scope

- **In-flight work overlaps heavily with the named issues.** The last ~10 days of commits/plans already target: a vendor circuit breaker (odds reliability, implemented), an ops dashboard (odds/live-status/feature-health visibility, working but uncommitted), a shared-lock fix for fixture-cache staleness (BUG-095, pushed but unverified in prod), an OddsPapi parser bug fix (BUG-105, fixed locally, not deployed), and UCL/UEL + domestic cup ingestion (mostly landed). The sprint should finish/verify this work rather than re-litigate it from scratch.
- **The beta's actual access path is broken right now.** The 5 most recent commits on `develop` are mid-debugging the staging Basic Auth proxy's CORS/OPTIONS handling; an uncommitted temp diagnostic probe is still in `nginx.conf.template`. This is a hard blocker independent of the 5 product issues.
- **We are inside a real FIFA international break as of this writing** (2026-11-11 through 2026-11-20, confirmed live against production: zero matches across all 5 supported leagues). This is the literal case of issue #4. It is not fixable within scope — no club competition (including UCL/UEL/cups) plays during an international window; covering it for real would mean onboarding international/national-team fixtures, a from-scratch league-addition effort. Real matches resume 2026-11-21, inside the sprint window.
- **Bet-logging's real-world verification gap is old and still open.** `W26` (opened 2026-07-12) flagged two bet-logging paths that have never been observed against real data: a same-day live-match settlement, and the T-30 "odds changed" refresh path. Both blockers it cited (timing, no real Odds API key) no longer apply.

## Approach chosen

Two combined axes, confirmed with the user via `AskUserQuestion`:

1. **Sequenced by dependency ("A"):** Week 1 unblocks access (staging proxy) and lands visibility (ops dashboard committed) — both prerequisites for doing Week 2/3 work with real evidence instead of guesses. Week 2 spends that visibility on the user's stated top-priority issue, odds. Week 3 covers bet-logging trust and browsing UX, plus optional league-variety enrichment.
2. **Scoped to a solid league subset ("C"):** the beta targets only the 5 fully-built domestic leagues (E0/SP1/I1/D1/F1). UCL/UEL/cups are bonus enrichment, not gating. International-fixture coverage is explicitly out of scope; the sprint's answer to "no use during international breaks" is an honest empty state (W260) plus launch timing (invite friends on/after 2026-11-21), not new league engineering.

Alternatives considered and rejected: running infra and product work as fully parallel tracks (risks both ending up half-finished for a single-person effort); polishing UI before fixing underlying data trust issues (risks a prettier app that's still not credible).

## Explicitly out of scope for this sprint

- International/national-team fixture coverage.
- Multi-book odds comparison / line-movement history (`app_user_stories.md` D10) — needs a bigger data budget, not a sprint fix.
- Live scores/standings as a standalone surface (D9) — reopens `app_prd.md`'s non-goals, needs a product-scope decision first.

## Resulting backlog

See `app_user_stories.md` PHASE 63, stories W255-W261, for the scoped, acceptance-criteria-bearing tasks.
