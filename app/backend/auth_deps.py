"""W210: FastAPI dependency for routes scoped to a real per-user identity.

Deliberately NOT JWT verification -- FastAPI never talks to Google or
Auth.js directly. The Next.js frontend's own server-side proxy routes
(app/frontend/app/api/bets/*) call getServerSession() there, which already
cryptographically verifies the session cookie using NEXTAUTH_SECRET; by the
time a request reaches here, the email has already been vouched for. This
dependency's only job is confirming the request truly came from our own
Next.js server (via INTERNAL_API_SECRET, a server-only env var never sent
to any browser) rather than a client hitting FastAPI directly and forging
an X-User-Email header. Two distinct trust boundaries, two distinct
secrets -- see the plan's header note on why this isn't APP_ACCESS_TOKEN."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException


def get_current_user_email(
    x_user_email: str | None = Header(default=None),
    x_internal_secret: str | None = Header(default=None),
) -> str:
    expected_secret = os.environ.get("INTERNAL_API_SECRET")
    if not expected_secret or x_internal_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not x_user_email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_user_email
