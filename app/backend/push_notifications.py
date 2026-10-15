"""Sends Web Push notifications via VAPID (pywebpush) -- no third-party
push service needed at this user count. A subscription is pruned here (not
just via the unsubscribe endpoint) since a browser can silently invalidate
its own endpoint -- permission revoked, or the push service expiring it --
without ever calling back to unsubscribe."""

from __future__ import annotations

import json
import os

from pywebpush import WebPushException, webpush

from app.backend.push_subscriptions import PushSubscriptionStore
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _vapid_claims() -> dict:
    contact_email = os.environ.get("VAPID_CONTACT_EMAIL", "admin@example.com")
    return {"sub": f"mailto:{contact_email}"}


def send_to_user(store: PushSubscriptionStore, user_id: int, title: str, body: str, url: str = "/") -> None:
    private_key = os.environ.get("VAPID_PRIVATE_KEY")
    if not private_key:
        LOGGER.warning("Push notification skipped -- VAPID_PRIVATE_KEY not configured.")
        return
    payload = json.dumps({"title": title, "body": body, "url": url})
    for sub in store.list_for_user(user_id):
        try:
            webpush(
                subscription_info={
                    "endpoint": sub.endpoint,
                    "keys": {"p256dh": sub.p256dh, "auth": sub.auth},
                },
                data=payload,
                vapid_private_key=private_key,
                vapid_claims=_vapid_claims(),
            )
        except WebPushException as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code in (404, 410):
                store.remove(sub.endpoint)
            else:
                LOGGER.warning("Push send failed for endpoint=%s: %s", sub.endpoint, exc)


def send_to_all_subscribers(store: PushSubscriptionStore, title: str, body: str, url: str = "/") -> None:
    for user_id in store.list_subscribed_user_ids():
        send_to_user(store, user_id, title, body, url)
