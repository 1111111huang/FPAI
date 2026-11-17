# Staging Auth Proxy — CORS Preflight Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the staging Basic Auth proxy (`deploy/staging-auth-proxy/`) actually let a credentialed cross-origin CORS preflight through, so the Vercel-hosted staging frontend can reach the Railway-hosted staging backend. This is W255 — the hard blocker for the entire friends-beta launch sprint (`documents/app_user_stories.md` PHASE 63).

**Architecture:** Two prior attempts at the same mechanism (overriding `auth_basic` directly inside an `if`, then toggling `auth_basic`'s value via a `$auth_basic_realm` variable set inside an `if`) have both failed live — the second attempt's own last commit says OPTIONS still 401s, with temp diagnostic headers added but not yet read. Rather than trying a third variant of "toggle auth_basic for OPTIONS," this plan switches to a fundamentally more robust mechanism: answer the OPTIONS preflight **locally**, inside nginx, with `return 204` — one of only two directives (`return`/`rewrite`, alongside `set`) nginx's own "if is evil" guidance calls safe, because `return` ends request processing during the rewrite phase, so the access-phase `auth_basic` check never runs for that request at all. No variable-visibility or phase-ordering ambiguity is left to get wrong. Real (non-OPTIONS) requests are completely unaffected — still proxied through with `auth_basic` enforced exactly as before. The preflight response reflects the request's own `Origin` header rather than hardcoding one, which is safe (a preflight carries no credentials and no sensitive data) and keeps real origin enforcement where it already lives: the upstream FastAPI `CORSMiddleware`'s `CORS_ALLOWED_ORIGINS` allow-list, which still gates whether the browser can read the *real* request's response.

**Tech Stack:** nginx (Alpine image, `nginx:1.27-alpine`), bash + curl for verification (no new dependencies — this is infra config, not application code, so "tests" here are a reproducible curl script rather than pytest).

---

### Task 1: Write the failing reproduction script

**Files:**
- Create: `deploy/staging-auth-proxy/verify_cors_preflight.sh`

This script is the "test" for this plan — a real curl-based check against the live staging URL, since nginx config has no unit-testable boundary. Run it now, before any fix, to confirm it fails exactly the way the last commit described.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Verifies the staging auth proxy actually lets a credentialed cross-origin
# CORS preflight through (W255, documents/app_user_stories.md PHASE 63).
#
# Usage:
#   STAGING_URL=https://<staging-proxy-host> \
#   FRONTEND_ORIGIN=https://<staging-frontend-host> \
#   [BASIC_AUTH_USER=... BASIC_AUTH_PASS=...] \
#   ./deploy/staging-auth-proxy/verify_cors_preflight.sh
set -euo pipefail

: "${STAGING_URL:?Set STAGING_URL to the staging proxy's public URL, e.g. https://fpai-staging.up.railway.app}"
: "${FRONTEND_ORIGIN:?Set FRONTEND_ORIGIN to the staging frontend's origin, e.g. https://fpai-staging.vercel.app}"

fail=0

echo "== 1. OPTIONS preflight (no credentials attached -- browsers never send them on a preflight) =="
preflight_headers=$(curl -s -i --max-time 10 \
  -X OPTIONS "$STAGING_URL/api/status" \
  -H "Origin: $FRONTEND_ORIGIN" \
  -H "Access-Control-Request-Method: GET" \
  -H "Access-Control-Request-Headers: content-type" \
  -H "Cache-Control: no-cache")
status_line=$(echo "$preflight_headers" | head -n 1)
echo "$preflight_headers" | tr -d '\r'
if ! echo "$status_line" | grep -qE "HTTP/[0-9.]+ 204"; then
  echo "FAIL: expected 204 for the OPTIONS preflight, got: $status_line"
  fail=1
elif ! echo "$preflight_headers" | grep -qi "Access-Control-Allow-Origin: $FRONTEND_ORIGIN"; then
  echo "FAIL: missing/incorrect Access-Control-Allow-Origin on the preflight response"
  fail=1
else
  echo "PASS: preflight returned 204 with Access-Control-Allow-Origin: $FRONTEND_ORIGIN"
fi

echo
echo "== 2. Real GET with no credentials still requires auth (unauthenticated real requests stay gated) =="
real_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$STAGING_URL/api/status")
if [[ "$real_status" == "401" ]]; then
  echo "PASS: unauthenticated GET got 401 as expected"
else
  echo "FAIL: expected 401 for an unauthenticated real GET, got: $real_status"
  fail=1
fi

if [[ -n "${BASIC_AUTH_USER:-}" && -n "${BASIC_AUTH_PASS:-}" ]]; then
  echo
  echo "== 3. Real GET with correct Basic Auth credentials reaches the upstream =="
  authed_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
    -u "$BASIC_AUTH_USER:$BASIC_AUTH_PASS" "$STAGING_URL/api/status")
  if [[ "$authed_status" == "401" ]]; then
    echo "FAIL: correct Basic Auth credentials still got 401 -- got: $authed_status"
    fail=1
  else
    echo "PASS: authenticated GET reached the upstream, got: $authed_status"
  fi
else
  echo
  echo "(skipping step 3 -- set BASIC_AUTH_USER/BASIC_AUTH_PASS to also verify the authenticated path)"
fi

exit $fail
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x deploy/staging-auth-proxy/verify_cors_preflight.sh
```

- [ ] **Step 3: Run it against the live staging proxy to confirm it currently fails**

```bash
STAGING_URL=<the real staging proxy URL> \
FRONTEND_ORIGIN=<the real staging frontend origin> \
./deploy/staging-auth-proxy/verify_cors_preflight.sh
```

Expected (RED): check 1 fails — the preflight comes back `401`, not `204` (matching the last commit's "`auth_basic` still 401s OPTIONS"). Note the exact status line and whether `X-Debug-Request-Method`/`X-Debug-Auth-Basic-Realm` (the already-deployed temp diagnostic headers) appear in the output.

**If the debug headers are completely absent from the response** (not even `X-Debug-Request-Method`), the currently-deployed container isn't running the committed config at all — stop here and force a clean redeploy (Railway dashboard → the staging-auth-proxy service → Deploy → clear build cache, or `railway up --service <staging-proxy-service>` if the CLI is authenticated) before proceeding to Task 2, then re-run Step 3.

- [ ] **Step 4: Commit the verification script**

```bash
git add deploy/staging-auth-proxy/verify_cors_preflight.sh
git commit -m "test(deploy): add CORS preflight verification script for the staging auth proxy"
```

---

### Task 2: Replace the auth-toggle mechanism with a local OPTIONS short-circuit

**Files:**
- Modify: `deploy/staging-auth-proxy/nginx.conf.template`

- [ ] **Step 1: Replace the file header comment and the `location /` block**

Replace lines 1-99 of `deploy/staging-auth-proxy/nginx.conf.template` with:

```nginx
# This is loaded by the official nginx image from /etc/nginx/conf.d/*
# The docker entrypoint below will envsubst this template to /etc/nginx/conf.d/reverse.conf
#
# Forked from biw/railway-nginx-password-auth (the Railway "Basic Auth Proxy"
# template) with one change: a CORS preflight (OPTIONS) request is answered
# locally with a 204 instead of being gated by auth_basic. Browsers never
# attach credentials to a preflight -- that's the Fetch spec, not a bug to
# work around upstream -- so gating OPTIONS the same as every other method
# meant a credentialed cross-origin request could never succeed through this
# proxy: the preflight always 401'd and the browser gave up before ever
# sending the real request.
#
# Mechanism history (two prior attempts, both confirmed live not to work,
# before this one): (1) `if ($request_method = OPTIONS) { auth_basic off; }`
# -- nginx cannot reliably override auth_basic across two different config
# contexts (include vs. if), a known nginx limitation (trac.nginx.org/nginx/
# ticket/2013). (2) Toggling a variable (`$auth_basic_realm`) that
# `auth_basic $auth_basic_realm;` reads later -- a textbook-documented
# pattern that should work per nginx's phase ordering (rewrite phase, where
# `if`/`set` run, precedes the access phase, where `auth_basic` is checked),
# but still 401'd OPTIONS live.
#
# This third approach sidesteps that whole mechanism family: `return` inside
# `if` (alongside `set`) is one of only two directives nginx's own
# "IfIsEvil" guidance calls safe, specifically because it ends request
# processing immediately during the rewrite phase -- the access-phase
# auth_basic check never runs at all for an OPTIONS request, regardless of
# any variable's value or phase-ordering subtlety. Real (non-OPTIONS)
# requests are unaffected, still proxied through with auth_basic enforced.
#
# Reflects the request's own Origin header on the preflight response rather
# than hardcoding one: safe because a preflight carries no credentials and
# no sensitive data, and real origin enforcement still lives downstream --
# the upstream FastAPI CORSMiddleware's CORS_ALLOWED_ORIGINS allow-list only
# attaches Access-Control-Allow-Origin to the REAL request's response for an
# actually-allowed origin, so a disallowed origin's preflight "succeeding"
# here still can't read the real response.

server {
    listen 80;
    listen [::]:80;

    # Add DNS resolver - dynamically configured by entrypoint script
    resolver ${DNS_RESOLVER} valid=300s;
    resolver_timeout 5s;

    # Upstream is supplied via env var ORIGIN (e.g., http://app:3000)
    set $upstream "${ORIGIN}";

    # Basic hardening / sane defaults
    client_max_body_size 50m;

    # Proxy headers
    location / {
        # Default: no auth, so a deploy with no basic_auth.conf content at
        # all (ALLOW_NO_AUTH=true) still has $auth_basic_realm declared --
        # nginx needs at least one `set` for a custom variable to parse it.
        set $auth_basic_realm off;

        # May override $auth_basic_realm to the configured realm (and set
        # auth_basic_user_file) -- see docker-entrypoint.sh. Empty when
        # BASIC_AUTH isn't configured, leaving the default above in place.
        include /etc/nginx/snippets/basic_auth.conf;

        # See file header: answer a CORS preflight locally and return
        # immediately, before auth_basic's access-phase check ever runs.
        if ($request_method = OPTIONS) {
            add_header Access-Control-Allow-Origin $http_origin always;
            add_header Access-Control-Allow-Credentials true always;
            add_header Access-Control-Allow-Methods "GET, POST, PUT, PATCH, DELETE, OPTIONS" always;
            add_header Access-Control-Allow-Headers "Content-Type, Authorization, X-App-Token" always;
            return 204;
        }

        auth_basic $auth_basic_realm;

        proxy_http_version 1.1;

        # For HTTPS upstreams, we need to send the upstream host for SSL cert validation
        # Extract host from upstream URL for proper SSL handshake
        proxy_ssl_server_name on;
        proxy_ssl_name $proxy_host;

        proxy_set_header Host $proxy_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;

        # WebSocket support
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;

        proxy_pass $upstream;
    }
}

# Map for Connection header that works for both WS and regular HTTP
map $http_upgrade $connection_upgrade {
    default upgrade;
    '' close;
}
```

This removes the now-unnecessary `if ($request_method = OPTIONS) { set $auth_basic_realm off; }` block (the preflight no longer reaches `auth_basic` at all) and the two temp `X-Debug-*` `add_header` lines and the `/__debug_conf__` location (root cause is now known — no need to keep probing for it).

- [ ] **Step 2: Validate the nginx config syntax locally before deploying**

```bash
docker run --rm \
  -e ORIGIN=http://localhost:3000 \
  -e DNS_RESOLVER="127.0.0.11" \
  -v "$(pwd)/deploy/staging-auth-proxy/nginx.conf.template:/etc/nginx/templates/reverse.conf.template:ro" \
  nginx:1.27-alpine \
  sh -c 'envsubst "\$ORIGIN \$DNS_RESOLVER" < /etc/nginx/templates/reverse.conf.template > /tmp/reverse.conf && nginx -t -c /tmp/reverse.conf' 2>&1 || true
```

Expected: no syntax errors reported (ignore any `resolver`/`upstream` warnings that only matter at real runtime inside the full container — this just checks the config parses).

- [ ] **Step 3: Commit**

```bash
git add deploy/staging-auth-proxy/nginx.conf.template
git commit -m "fix(deploy): answer CORS preflights locally in the staging proxy instead of toggling auth_basic"
```

---

### Task 3: Deploy and verify green

**Files:** none (deploy-only task)

- [ ] **Step 1: Deploy the staging-auth-proxy service**

Push to whatever branch/remote triggers the staging-auth-proxy service's Railway deploy (per `documents/app_user_stories.md`'s branching model: `develop` → Railway staging). Wait for the deploy to finish.

- [ ] **Step 2: Re-run the verification script from Task 1**

```bash
STAGING_URL=<the real staging proxy URL> \
FRONTEND_ORIGIN=<the real staging frontend origin> \
BASIC_AUTH_USER=<the real staging BASIC_AUTH username> \
BASIC_AUTH_PASS=<the real staging BASIC_AUTH password> \
./deploy/staging-auth-proxy/verify_cors_preflight.sh
```

Expected (GREEN): all 3 checks PASS — preflight returns `204` with a matching `Access-Control-Allow-Origin`, an unauthenticated real GET still returns `401`, and an authenticated real GET reaches the upstream.

**If check 1 still fails after confirming the deploy picked up this exact commit** (compare the deployed git SHA, e.g. via a `/api/status`-style endpoint or Railway's deploy log, against `git rev-parse HEAD`), the failure is no longer this `location` block — `return` inside `if` leaves nothing else in this file that could cause a 401 on OPTIONS. At that point the cause is operational, outside this file, and the next places to check are: (a) whether the tested `STAGING_URL` is actually the staging-auth-proxy service and not some other Railway service/domain, (b) whether Railway's own project-level settings have a separate access-control feature enabled in front of this service, (c) whether a CDN/proxy in front of the `STAGING_URL` domain (e.g. Cloudflare, if that domain's DNS is proxied through it) is caching a pre-fix response — bust it with a cache-busting query param (`?cachebust=$(date +%s)`) appended to the curl request in the script and re-run.

- [ ] **Step 3: Manually verify from an actual browser**, since curl can't fully simulate every browser preflight quirk. Open the staging frontend, open DevTools → Network, trigger an API-backed page (e.g. the Dashboard), and confirm: the OPTIONS request (if the browser sends one, which it will for most of this app's API calls since they include a custom `X-App-Token`/`Content-Type` header, both of which trigger preflighting) shows `204`, and the follow-up real request shows real data, not a CORS error in the console.

---

### Task 4: Confirm cleanup and close out the story

**Files:**
- Modify: `documents/app_user_stories.md`

- [ ] **Step 1:** Confirm `deploy/staging-auth-proxy/nginx.conf.template` no longer contains `__debug_conf__` or `X-Debug-` (already removed in Task 2 — this step is just the final check before marking done).

```bash
grep -n "__debug_conf__\|X-Debug-" deploy/staging-auth-proxy/nginx.conf.template || echo "clean"
```

Expected: `clean`.

- [ ] **Step 2:** Mark W255 completed in `documents/app_user_stories.md` (PHASE 63 table) with a completion note summarizing the actual mechanism shipped (local `return 204` for OPTIONS, not a third variant of the auth-toggle approach) and the verification evidence from Task 3.

- [ ] **Step 3: Commit**

```bash
git add documents/app_user_stories.md
git commit -m "docs: mark W255 completed -- staging proxy CORS preflight fix verified live"
```

---

## Self-Review Notes

- **Spec coverage:** W255's acceptance criteria (`documents/app_user_stories.md` PHASE 63) are: a real OPTIONS preflight succeeds, a credentialed cross-origin request succeeds, and the temp diagnostic probe is removed. Tasks 1-3 cover the first two (reproduction + fix + live verification); Task 2 Step 1 + Task 4 Step 1 cover the third.
- **Placeholder scan:** no TBD/TODO; the "if check 1 still fails" branch in Task 3 gives concrete, runnable next steps rather than a vague escalation, since the live root cause can't be fully confirmed without network access to the real staging URL from within this plan.
- **Type/naming consistency:** `verify_cors_preflight.sh`'s env vars (`STAGING_URL`, `FRONTEND_ORIGIN`, `BASIC_AUTH_USER`, `BASIC_AUTH_PASS`) are used identically across Task 1 and Task 3. The CORS header values added in Task 2 (`Content-Type, Authorization, X-App-Token`, methods `GET, POST, PUT, PATCH, DELETE, OPTIONS`) were checked against `app/frontend/lib/api.ts`'s actual usage (`X-App-Token`, `Content-Type: application/json`, `POST`/`PATCH` calls) rather than guessed.
