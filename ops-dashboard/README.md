# Ops Dashboard (standalone)

An ad-hoc, on-demand viewer for the real production backend's
`/api/admin/ops/*` endpoints — Failures, Odds Availability, Feature Health,
Performance. Not deployed anywhere, not part of the Oddsey frontend, no
login screen. Launched locally whenever you want to look, closed when done.

## Run it

```bash
python3 ops-dashboard/serve.py        # defaults to port 8765
# or: python3 ops-dashboard/serve.py 9000
```

Then open the printed URL (e.g. `http://localhost:8765`).

## How it works

- `index.html` / `app.js` — static page, vanilla JS, no build step, no
  framework. Fetches real data via same-origin `/api/admin/ops/*` calls.
- `serve.py` — stdlib-only Python server. Serves the static files, and
  proxies any `/api/*` request to the real production backend
  (`FPAI_PROD_URL`, from the repo root's `.env`), attaching the admin
  `x-app-token` header itself. The token never reaches browser-visible JS,
  and the browser only ever talks to `localhost` — no CORS config needed
  on the real backend.

## Requirements

`FPAI_PROD_URL` and `FPAI_PROD_TOKEN` set in the repo root's `.env`
(already required for the existing `prod-regen-and-rate-limits` workflow).
