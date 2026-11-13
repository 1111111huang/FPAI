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
  -H "Cache-Control: no-cache" || true)
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
real_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$STAGING_URL/api/status" || true)
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
    -u "$BASIC_AUTH_USER:$BASIC_AUTH_PASS" "$STAGING_URL/api/status" || true)
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
