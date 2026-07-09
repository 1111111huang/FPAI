# Technical Specification — FPAI Web App

> **DRAFT — functionality under active discussion.** Only settled decisions are recorded here. See `documents/app_user_stories.md` for the open design-discussion log (particularly D2 caching, D5 backend/engine boundary, D7 deployment shape), all of which affect this document.

## 1. Stack

- Frontend: Next.js + React + TypeScript, under `app/frontend`.
- Backend: FastAPI, under `app/backend`, wrapping the existing `src/agent` and `src/forecast` Python code rather than reimplementing it.
- Storage: the existing project DuckDB file — no new database introduced. A `user_bets` table is anticipated for the bet tracker (see D3 in `documents/app_user_stories.md`), exact schema TBD.

## 2. Module Structure (provisional)

```
app/
  README.md
  backend/     # FastAPI app; endpoints TBD pending D1-D5
  frontend/    # Next.js app; pages TBD pending D1, D4, D6
```

## 3. Open Items

Everything else — API surface, request/response schemas, caching strategy, fixture discovery mechanism, bet settlement flow, deployment shape — is intentionally undecided. Do not build against this document until the corresponding discussion in `documents/app_user_stories.md` is resolved and reflected here.
