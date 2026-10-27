"""Shared Slack webhook alerting (BUG-091 path, generalized for BUG-096).

Originally lived only inside scheduler_wiring.py's _alert_all_keys_exhausted.
Moved here so eod_batch.py/t30_refresh.py can alert too: scheduler_wiring
imports both of those modules, so they can't import from it back.

Same contract as before:
- No-op unless SLACK_ALERT_WEBHOOK_URL is set ("off unless configured").
- Debounced per key: one message per cooldown window per key, since the
  conditions alerted on tend to recur on every call while they last.
- Never raises: a failed alert POST must never break the work it reports on.
"""

from __future__ import annotations

import os
import time

import requests

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

SLACK_ALERT_COOLDOWN_SECONDS = 3600.0

# debounce key -> time.monotonic() of the last alert sent for it.
_last_slack_alert_at: dict[str, float] = {}


def post_slack_alert(text: str, debounce_key: str, cooldown_seconds: float = SLACK_ALERT_COOLDOWN_SECONDS) -> bool:
    """Posts `text` to SLACK_ALERT_WEBHOOK_URL. Returns True only when a
    POST was actually attempted (webhook set, not inside the cooldown)."""
    webhook_url = os.environ.get("SLACK_ALERT_WEBHOOK_URL")
    if not webhook_url:
        return False
    now = time.monotonic()
    last = _last_slack_alert_at.get(debounce_key)
    if last is not None and now - last < cooldown_seconds:
        return False
    _last_slack_alert_at[debounce_key] = now
    try:
        requests.post(webhook_url, json={"text": text}, timeout=5)
    except requests.RequestException:
        LOGGER.warning("Slack alert POST failed (%s)", debounce_key, exc_info=True)
    return True
