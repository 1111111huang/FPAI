# Installable Mobile App (PWA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing Next.js web app (`app/frontend`) into an installable PWA that works on both Android and iOS: home-screen install, offline fallback, Web Push notifications (new daily recs, bet settlement), and native-feeling polish (safe-area insets, haptics, tab transitions).

**Architecture:** No new codebase. A manifest + hand-written service worker makes the existing site installable; a new `push_subscriptions` SQLite table (same per-concern pattern as `users.db`/`user_bets.db`) plus a small `pywebpush`-based sender hook into the two places recommendations/settlements already happen (`scheduler_wiring.py`'s nightly EOD job, `main.py`'s bet-settlement call sites).

**Tech Stack:** Next.js 14.2 App Router (existing), vanilla service worker (no Workbox/next-pwa), Web Push API + `pywebpush`/VAPID (no third-party push SaaS), FastAPI + sqlite3 (existing patterns).

**Spec:** `docs/superpowers/specs/2026-10-15-pwa-mobile-app-design.md`

---

## Before you start

Two things were verified while writing this plan (not guesses — actually run):

1. **`pywebpush` version pin matters.** The latest `pywebpush` (2.5.0) requires `cryptography>=47.0.0`, which conflicts with this repo's `mlflow==3.10.1` (`requires cryptography<47,>=43.0.0` — one shared `requirements.txt`/venv for the whole project). `pywebpush==1.14.1` only requires `cryptography>=2.6.1` and has no conflict. Task 8 pins this exact version — do not `pip install pywebpush` unpinned.
2. **Icon source asset.** `app/frontend/public/oddsey-logo.png` is a real 1033×1033 PNG — `sips` (built into macOS, already confirmed present) resizes it directly, no image library needed.

---

### Task 1: PWA icons + manifest

**Files:**
- Create: `app/frontend/public/icons/icon-192.png`
- Create: `app/frontend/public/icons/icon-512.png`
- Create: `app/frontend/public/manifest.json`

- [ ] **Step 1: Generate the two icon sizes from the existing logo**

Run:
```bash
mkdir -p app/frontend/public/icons
sips -z 192 192 app/frontend/public/oddsey-logo.png --out app/frontend/public/icons/icon-192.png
sips -z 512 512 app/frontend/public/oddsey-logo.png --out app/frontend/public/icons/icon-512.png
```
Expected: both commands print `/Users/.../icon-192.png` / `icon-512.png` with no error, and both files exist (`ls app/frontend/public/icons`).

- [ ] **Step 2: Write the manifest**

```json
{
  "name": "Oddsey",
  "short_name": "Oddsey",
  "description": "Oddsey betting agent web app.",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#0a0a0d",
  "theme_color": "#0a0a0d",
  "icons": [
    { "src": "/icons/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icons/icon-512.png", "sizes": "512x512", "type": "image/png" }
  ]
}
```
Save as `app/frontend/public/manifest.json`. Colors match `app/frontend/app/globals.css`'s `--bg-app` (the app is locked to one dark theme, D6 — no light variant to also account for).

- [ ] **Step 3: Commit**

```bash
git add app/frontend/public/icons app/frontend/public/manifest.json
git commit -m "feat(frontend): add PWA manifest and home-screen icons"
```

---

### Task 2: Wire the manifest into the app's metadata

**Files:**
- Modify: `app/frontend/app/layout.tsx`

- [ ] **Step 1: Add manifest/icon/theme-color metadata**

Replace the top of `app/frontend/app/layout.tsx` (the `metadata` export) with:

```tsx
import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "Oddsey",
  description: "Oddsey betting agent web app.",
  manifest: "/manifest.json",
  icons: {
    icon: "/icons/icon-192.png",
    apple: "/icons/icon-192.png",
  },
};

export const viewport: Viewport = {
  themeColor: "#0a0a0d",
};
```

(Everything below the `viewport` export — the font setup and `RootLayout` function — stays unchanged.)

- [ ] **Step 2: Verify the app still builds**

Run: `cd app/frontend && npm run build`
Expected: build succeeds (exit code 0); no "Unsupported metadata" warnings for `manifest`/`icons`/`themeColor` in the output.

- [ ] **Step 3: Commit**

```bash
git add app/frontend/app/layout.tsx
git commit -m "feat(frontend): wire PWA manifest and theme-color into app metadata"
```

---

### Task 3: Service worker + registration

**Files:**
- Create: `app/frontend/public/sw.js`
- Create: `app/frontend/components/ServiceWorkerRegistration.tsx`
- Test: `app/frontend/components/__tests__/ServiceWorkerRegistration.test.tsx`
- Modify: `app/frontend/app/providers.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
import { render } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ServiceWorkerRegistration } from "../ServiceWorkerRegistration";

describe("ServiceWorkerRegistration", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("registers the service worker on mount when supported", () => {
    const register = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(window.navigator, "serviceWorker", {
      value: { register },
      configurable: true,
    });
    render(<ServiceWorkerRegistration />);
    expect(register).toHaveBeenCalledWith("/sw.js");
  });

  it("does nothing when serviceWorker is unsupported", () => {
    Object.defineProperty(window.navigator, "serviceWorker", {
      value: undefined,
      configurable: true,
    });
    expect(() => render(<ServiceWorkerRegistration />)).not.toThrow();
  });
});
```
Save as `app/frontend/components/__tests__/ServiceWorkerRegistration.test.tsx`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app/frontend && npx vitest run components/__tests__/ServiceWorkerRegistration.test.tsx`
Expected: FAIL — `Cannot find module '../ServiceWorkerRegistration'`.

- [ ] **Step 3: Write the service worker**

```js
// app/frontend/public/sw.js
const CACHE_NAME = "oddsey-shell-v1";

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(["/", "/offline"])));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  event.respondWith(fetch(event.request).catch(() => caches.match("/offline")));
});

self.addEventListener("push", (event) => {
  if (!event.data) return;
  const payload = event.data.json();
  event.waitUntil(
    self.registration.showNotification(payload.title, {
      body: payload.body,
      data: { url: payload.url || "/" },
    })
  );
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const url = (event.notification.data && event.notification.data.url) || "/";
  event.waitUntil(self.clients.openWindow(url));
});
```
Save as `app/frontend/public/sw.js`.

- [ ] **Step 4: Write the registration component**

```tsx
"use client";

import { useEffect } from "react";

export function ServiceWorkerRegistration() {
  useEffect(() => {
    if (!("serviceWorker" in navigator)) return;
    navigator.serviceWorker.register("/sw.js").catch(() => {
      // ponytail: best-effort -- an unsupported/blocked browser just never
      // gets install/offline/push; nothing else in the app depends on it.
    });
  }, []);
  return null;
}
```
Save as `app/frontend/components/ServiceWorkerRegistration.tsx`.

- [ ] **Step 5: Mount it once, in `Providers`**

```tsx
"use client";

import { SessionProvider } from "next-auth/react";
import { ServiceWorkerRegistration } from "@/components/ServiceWorkerRegistration";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider>
      <ServiceWorkerRegistration />
      {children}
    </SessionProvider>
  );
}
```
Replace the full contents of `app/frontend/app/providers.tsx` with this.

- [ ] **Step 6: Run test to verify it passes**

Run: `cd app/frontend && npx vitest run components/__tests__/ServiceWorkerRegistration.test.tsx`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add app/frontend/public/sw.js app/frontend/components/ServiceWorkerRegistration.tsx \
        app/frontend/components/__tests__/ServiceWorkerRegistration.test.tsx app/frontend/app/providers.tsx
git commit -m "feat(frontend): add PWA service worker (offline shell + push) and register it"
```

---

### Task 4: Offline fallback page

**Files:**
- Create: `app/frontend/app/offline/page.tsx`

- [ ] **Step 1: Write the page**

```tsx
export default function OfflinePage() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-3 bg-page px-6 text-center">
      <h1 className="text-xl font-semibold text-ink">You&apos;re offline</h1>
      <p className="max-w-sm text-sm text-ink-secondary">
        Oddsey needs a connection to load fixtures and recommendations. Reconnect and try again.
      </p>
    </div>
  );
}
```
Save as `app/frontend/app/offline/page.tsx`. Deliberately not wrapped in `AppShell` — it must render from the service worker's cache alone with no API calls (`AppShell` fetches fixtures on search focus and calls the sandbox-status endpoint on mount).

- [ ] **Step 2: Verify it builds and renders**

Run: `cd app/frontend && npm run build`
Expected: build succeeds; `/offline` listed as a static route in the build output.

- [ ] **Step 3: Commit**

```bash
git add app/frontend/app/offline/page.tsx
git commit -m "feat(frontend): add offline fallback page for the PWA service worker"
```

---

### Task 5: Install guidance banner (Android + iOS)

**Files:**
- Create: `app/frontend/components/InstallAppBanner.tsx`
- Test: `app/frontend/components/__tests__/InstallAppBanner.test.tsx`
- Modify: `app/frontend/components/AppShell.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, beforeEach, vi } from "vitest";
import { InstallAppBanner } from "../InstallAppBanner";

function mockMatchMedia(matches: boolean) {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    configurable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches,
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })),
  });
}

describe("InstallAppBanner", () => {
  beforeEach(() => {
    localStorage.clear();
    mockMatchMedia(false);
    Object.defineProperty(window.navigator, "standalone", { value: false, configurable: true });
  });

  it("shows iOS Add-to-Home-Screen instructions on an iOS user agent", () => {
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue(
      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"
    );
    render(<InstallAppBanner />);
    expect(screen.getByText(/add to home screen/i)).toBeInTheDocument();
  });

  it("does not render when already running standalone", () => {
    mockMatchMedia(true);
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue(
      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"
    );
    render(<InstallAppBanner />);
    expect(screen.queryByText(/install oddsey/i)).not.toBeInTheDocument();
  });

  it("does not render when previously dismissed", () => {
    localStorage.setItem("oddsey-install-banner-dismissed", "1");
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue(
      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"
    );
    render(<InstallAppBanner />);
    expect(screen.queryByText(/add to home screen/i)).not.toBeInTheDocument();
  });

  it("shows an Install button on Android and calls prompt() on click", async () => {
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue(
      "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/120.0 Mobile Safari/537.36"
    );
    render(<InstallAppBanner />);
    const promptFn = vi.fn();
    const event = new Event("beforeinstallprompt", { cancelable: true }) as Event & { prompt: () => void };
    event.prompt = promptFn;
    window.dispatchEvent(event);
    const button = await screen.findByRole("button", { name: /^install$/i });
    fireEvent.click(button);
    expect(promptFn).toHaveBeenCalled();
  });
});
```
Save as `app/frontend/components/__tests__/InstallAppBanner.test.tsx`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app/frontend && npx vitest run components/__tests__/InstallAppBanner.test.tsx`
Expected: FAIL — `Cannot find module '../InstallAppBanner'`.

- [ ] **Step 3: Write the component**

```tsx
"use client";

import { useEffect, useState } from "react";

const DISMISSED_KEY = "oddsey-install-banner-dismissed";

type InstallPromptEvent = Event & { prompt: () => void };

function isIos(): boolean {
  return /iphone|ipad|ipod/i.test(window.navigator.userAgent);
}

function isStandalone(): boolean {
  return (
    window.matchMedia("(display-mode: standalone)").matches ||
    (window.navigator as Navigator & { standalone?: boolean }).standalone === true
  );
}

export function InstallAppBanner() {
  const [visible, setVisible] = useState(false);
  const [ios, setIos] = useState(false);
  const [deferredPrompt, setDeferredPrompt] = useState<InstallPromptEvent | null>(null);

  useEffect(() => {
    if (isStandalone()) return;
    if (localStorage.getItem(DISMISSED_KEY)) return;

    if (isIos()) {
      setIos(true);
      setVisible(true);
      return;
    }

    function onBeforeInstallPrompt(event: Event) {
      event.preventDefault();
      setDeferredPrompt(event as InstallPromptEvent);
      setVisible(true);
    }
    window.addEventListener("beforeinstallprompt", onBeforeInstallPrompt);
    return () => window.removeEventListener("beforeinstallprompt", onBeforeInstallPrompt);
  }, []);

  function dismiss() {
    localStorage.setItem(DISMISSED_KEY, "1");
    setVisible(false);
  }

  function install() {
    deferredPrompt?.prompt();
    setVisible(false);
  }

  if (!visible) return null;

  return (
    <div className="flex items-center justify-between gap-3 border-b border-border bg-surface px-4 py-2 text-sm text-ink">
      {ios ? (
        <span>Install Oddsey: tap Share, then &quot;Add to Home Screen&quot;.</span>
      ) : (
        <span>Install Oddsey for the full app experience.</span>
      )}
      <div className="flex shrink-0 items-center gap-3">
        {!ios && (
          <button type="button" onClick={install} className="font-medium text-accent">
            Install
          </button>
        )}
        <button type="button" onClick={dismiss} aria-label="Dismiss" className="text-ink-secondary">
          Dismiss
        </button>
      </div>
    </div>
  );
}
```
Save as `app/frontend/components/InstallAppBanner.tsx`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app/frontend && npx vitest run components/__tests__/InstallAppBanner.test.tsx`
Expected: PASS (4 tests).

- [ ] **Step 5: Mount the banner in `AppShell`**

In `app/frontend/components/AppShell.tsx`, add the import near the top:

```tsx
import { InstallAppBanner } from "@/components/InstallAppBanner";
```

Then, immediately above the `<main className="flex-1 px-4 pb-20 ...">` line (inside the `<div className="flex min-w-0 flex-1 flex-col lg:min-h-0">` wrapper, right after the search-bar `<div>` block closes), add:

```tsx
        <InstallAppBanner />
```

- [ ] **Step 6: Run the existing AppShell test to confirm nothing broke**

Run: `cd app/frontend && npx vitest run components/__tests__/AppShell.test.tsx`
Expected: PASS (no new failures — `InstallAppBanner` renders `null` by default since `localStorage`/`matchMedia` in jsdom give `isStandalone() === false` but neither mocked UA is iOS nor is `beforeinstallprompt` ever dispatched in that test, so `visible` stays `false`).

- [ ] **Step 7: Commit**

```bash
git add app/frontend/components/InstallAppBanner.tsx app/frontend/components/__tests__/InstallAppBanner.test.tsx \
        app/frontend/components/AppShell.tsx
git commit -m "feat(frontend): add platform-aware install guidance banner (Android + iOS)"
```

---

### Task 6: Native-feeling polish in AppShell (safe-area insets, haptics, tab transitions)

**Files:**
- Modify: `app/frontend/components/AppShell.tsx`

- [ ] **Step 1: Safe-area insets on the mobile header and bottom tab bar**

In `AppShell.tsx`, find the mobile-only compact header div:
```tsx
      <div className="flex items-center justify-between border-b border-border bg-surface px-4 py-3 lg:hidden">
```
Change `py-3` to account for the top notch/status-bar inset:
```tsx
      <div className="flex items-center justify-between border-b border-border bg-surface px-4 pb-3 pt-[calc(0.75rem+env(safe-area-inset-top))] lg:hidden">
```

Find the fixed bottom tab bar:
```tsx
      <nav className="fixed inset-x-0 bottom-0 z-20 flex items-center justify-around border-t border-border bg-page px-2 py-1.5 lg:hidden">
```
Change it to account for the home-indicator inset:
```tsx
      <nav className="fixed inset-x-0 bottom-0 z-20 flex items-center justify-around border-t border-border bg-page px-2 pb-[calc(0.375rem+env(safe-area-inset-bottom))] pt-1.5 lg:hidden">
```

- [ ] **Step 2: Haptic tap feedback + view-transition on the bottom tab bar**

Add `useRouter` to the imports at the top of `AppShell.tsx`:
```tsx
import { useRouter } from "next/navigation";
```

Inside the `AppShell` function body, add (near the other `useState` calls):
```tsx
  const router = useRouter();

  function handleTabClick(event: React.MouseEvent<HTMLAnchorElement>, href: string) {
    navigator.vibrate?.(10);
    if (typeof document === "undefined" || !("startViewTransition" in document)) return;
    if (href === window.location.pathname) return;
    event.preventDefault();
    (document as Document & { startViewTransition: (cb: () => void) => void }).startViewTransition(() => {
      router.push(href);
    });
  }
```

In the bottom tab bar's `<Link>` (inside the `NAV_ITEMS.map` in the `<nav className="fixed inset-x-0 bottom-0 ...">` block), add an `onClick`:
```tsx
              onClick={(event) => handleTabClick(event, item.href)}
```
right next to the existing `prefetch={...}` prop on that `<Link>`.

- [ ] **Step 3: Mark the scrolling main region as a view-transition target**

Find:
```tsx
        <main className="flex-1 px-4 pb-20 sm:px-6 lg:min-h-0 lg:overflow-y-auto lg:pb-8">
```
Add the Tailwind arbitrary-property class (unsupported browsers simply ignore the unknown CSS property, no `@supports` needed):
```tsx
        <main className="flex-1 px-4 pb-20 [view-transition-name:main-content] sm:px-6 lg:min-h-0 lg:overflow-y-auto lg:pb-8">
```

- [ ] **Step 4: Run the existing AppShell test**

Run: `cd app/frontend && npx vitest run components/__tests__/AppShell.test.tsx`
Expected: PASS — `navigator.vibrate` is `undefined` in jsdom, so `navigator.vibrate?.(10)` is a no-op; `"startViewTransition" in document` is `false` in jsdom, so `handleTabClick` returns before calling `event.preventDefault()`, and existing `Link` navigation assertions are unaffected.

- [ ] **Step 5: Manual check (not automatable in jsdom)**

On a real phone (or Chrome DevTools device toolbar with "Emulate CSS media feature" off), confirm the bottom tab bar no longer sits flush against the screen edge on a device with a home indicator, and that tapping a tab produces a brief vibration on Android.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/components/AppShell.tsx
git commit -m "feat(frontend): add safe-area insets, tap haptics, and tab view-transitions"
```

---

### Task 7: Backend — `PushSubscriptionStore`

**Files:**
- Create: `app/backend/push_subscriptions.py`
- Test: `app/backend/tests/test_push_subscriptions.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.push_subscriptions import PushSubscriptionStore


def test_add_creates_a_new_subscription(tmp_path: Path):
    store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    sub = store.add(user_id=1, endpoint="https://push.example/abc", p256dh="key1", auth="auth1")
    assert sub.user_id == 1
    assert sub.endpoint == "https://push.example/abc"


def test_add_is_idempotent_for_the_same_endpoint(tmp_path: Path):
    store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    store.add(user_id=1, endpoint="https://push.example/abc", p256dh="key1", auth="auth1")
    store.add(user_id=1, endpoint="https://push.example/abc", p256dh="key2", auth="auth2")
    subs = store.list_for_user(1)
    assert len(subs) == 1
    assert subs[0].p256dh == "key2"


def test_remove_deletes_the_subscription(tmp_path: Path):
    store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    store.add(user_id=1, endpoint="https://push.example/abc", p256dh="key1", auth="auth1")
    store.remove("https://push.example/abc")
    assert store.list_for_user(1) == []


def test_list_subscribed_user_ids_returns_distinct_users(tmp_path: Path):
    store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    store.add(user_id=1, endpoint="https://push.example/a", p256dh="k", auth="a")
    store.add(user_id=1, endpoint="https://push.example/b", p256dh="k", auth="a")
    store.add(user_id=2, endpoint="https://push.example/c", p256dh="k", auth="a")
    assert sorted(store.list_subscribed_user_ids()) == [1, 2]
```
Save as `app/backend/tests/test_push_subscriptions.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest app/backend/tests/test_push_subscriptions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.backend.push_subscriptions'`.

- [ ] **Step 3: Write the store**

```python
"""Web Push subscriptions (one row per installed PWA instance a user has
granted notification permission to) -- same SQLite-per-concern pattern as
users.py/bet_tracker.py. A user may have multiple rows (phone + desktop);
`endpoint` is the natural unique key Web Push already assigns per device."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "push_subscriptions.db"


@dataclass(frozen=True)
class PushSubscription:
    id: int
    user_id: int
    endpoint: str
    p256dh: str
    auth: str
    created_at: str


class PushSubscriptionStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS push_subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    endpoint TEXT NOT NULL UNIQUE,
                    p256dh TEXT NOT NULL,
                    auth TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def add(self, user_id: int, endpoint: str, p256dh: str, auth: str) -> PushSubscription:
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with self._connect() as conn:
            # ON CONFLICT ... DO UPDATE: re-subscribing the same endpoint
            # (the browser renews it) refreshes its keys/owner instead of
            # erroring on the UNIQUE constraint.
            conn.execute(
                """
                INSERT INTO push_subscriptions (user_id, endpoint, p256dh, auth, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(endpoint) DO UPDATE SET
                    user_id = excluded.user_id, p256dh = excluded.p256dh,
                    auth = excluded.auth, created_at = excluded.created_at
                """,
                (user_id, endpoint, p256dh, auth, created_at),
            )
            row = conn.execute(
                "SELECT id, user_id, endpoint, p256dh, auth, created_at FROM push_subscriptions WHERE endpoint = ?",
                (endpoint,),
            ).fetchone()
            return PushSubscription(*row)

    def remove(self, endpoint: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))

    def list_for_user(self, user_id: int) -> list[PushSubscription]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, user_id, endpoint, p256dh, auth, created_at FROM push_subscriptions WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            return [PushSubscription(*row) for row in rows]

    def list_subscribed_user_ids(self) -> list[int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT user_id FROM push_subscriptions").fetchall()
            return [row[0] for row in rows]
```
Save as `app/backend/push_subscriptions.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest app/backend/tests/test_push_subscriptions.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add app/backend/push_subscriptions.py app/backend/tests/test_push_subscriptions.py
git commit -m "feat(backend): add PushSubscriptionStore for Web Push device registrations"
```

---

### Task 8: Backend — sending Web Push (`pywebpush`)

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Create: `app/backend/push_notifications.py`
- Test: `app/backend/tests/test_push_notifications.py`

- [ ] **Step 1: Pin and install `pywebpush`**

Add this line to `requirements.txt`, in the "Web app backend (app/backend)" section (after `httpx==0.28.1`):
```
pywebpush==1.14.1  # pinned -- 2.x requires cryptography>=47, which conflicts
                   # with mlflow==3.10.1's own cryptography<47 pin (one shared
                   # venv/requirements.txt for the whole project). 1.14.1 only
                   # requires cryptography>=2.6.1, verified conflict-free.
```

Run: `source venv/bin/activate && pip install pywebpush==1.14.1 && pip check`
Expected: `pip check` prints `No broken requirements found.`

- [ ] **Step 2: Document the new env vars**

Add to `.env.example`, in a new section after the `ENABLE_SCHEDULER` block:
```
# --- Backend: Web Push notifications ---
# VAPID keypair for Web Push (new recs / bet settlement alerts). Generate
# once with (py-vapid ships with pywebpush, installed above):
#   mkdir /tmp/vapid_keys && cd /tmp/vapid_keys && vapid --gen
#   vapid --applicationServerKey   # -> NEXT_PUBLIC_VAPID_PUBLIC_KEY (frontend .env.example)
#   python3 -c "
#   import base64
#   from cryptography.hazmat.primitives import serialization
#   with open('private_key.pem', 'rb') as f:
#       key = serialization.load_pem_private_key(f.read(), password=None)
#   raw = key.private_numbers().private_value.to_bytes(32, 'big')
#   print(base64.urlsafe_b64encode(raw).rstrip(b'=').decode())
#   "                               # -> VAPID_PRIVATE_KEY (below)
#   cd / && rm -rf /tmp/vapid_keys  # keys are now captured above, delete the PEM files
VAPID_PRIVATE_KEY=
# Contact address required by the Web Push spec's VAPID claims (shown to
# push services, never to end users). Any real address you control.
VAPID_CONTACT_EMAIL=
```

Add to `app/frontend/.env.example`, after `INTERNAL_API_SECRET`:
```

# Public half of the same VAPID keypair (repo-root .env.example's
# VAPID_PRIVATE_KEY) -- safe to expose to the browser, it's the public key.
NEXT_PUBLIC_VAPID_PUBLIC_KEY=
```

- [ ] **Step 3: Write the failing test**

```python
from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from pywebpush import WebPushException

from app.backend.push_notifications import send_to_all_subscribers, send_to_user
from app.backend.push_subscriptions import PushSubscriptionStore


def _store_with_one_subscription(tmp_path: Path, user_id: int = 1) -> PushSubscriptionStore:
    store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    store.add(user_id=user_id, endpoint="https://push.example/abc", p256dh="key", auth="auth")
    return store


def test_send_to_user_calls_webpush_for_each_subscription(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "test-private-key")
    store = _store_with_one_subscription(tmp_path)
    with patch("app.backend.push_notifications.webpush") as mock_webpush:
        send_to_user(store, user_id=1, title="Hi", body="Body", url="/bets")
    assert mock_webpush.call_count == 1
    _, kwargs = mock_webpush.call_args
    assert kwargs["subscription_info"]["endpoint"] == "https://push.example/abc"


def test_send_to_user_skips_silently_when_vapid_key_unset(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("VAPID_PRIVATE_KEY", raising=False)
    store = _store_with_one_subscription(tmp_path)
    with patch("app.backend.push_notifications.webpush") as mock_webpush:
        send_to_user(store, user_id=1, title="Hi", body="Body")
    mock_webpush.assert_not_called()


def test_send_to_user_prunes_an_expired_subscription(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "test-private-key")
    store = _store_with_one_subscription(tmp_path)

    class _Response:
        status_code = 410

    with patch(
        "app.backend.push_notifications.webpush",
        side_effect=WebPushException("gone", response=_Response()),
    ):
        send_to_user(store, user_id=1, title="Hi", body="Body")
    assert store.list_for_user(1) == []


def test_send_to_all_subscribers_sends_to_every_subscribed_user(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "test-private-key")
    store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    store.add(user_id=1, endpoint="https://push.example/a", p256dh="k", auth="a")
    store.add(user_id=2, endpoint="https://push.example/b", p256dh="k", auth="a")
    with patch("app.backend.push_notifications.webpush") as mock_webpush:
        send_to_all_subscribers(store, title="Hi", body="Body")
    assert mock_webpush.call_count == 2
```
Save as `app/backend/tests/test_push_notifications.py`.

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest app/backend/tests/test_push_notifications.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.backend.push_notifications'`.

- [ ] **Step 5: Write the module**

```python
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
```
Save as `app/backend/push_notifications.py`.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest app/backend/tests/test_push_notifications.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .env.example app/frontend/.env.example \
        app/backend/push_notifications.py app/backend/tests/test_push_notifications.py
git commit -m "feat(backend): add pywebpush-based Web Push sender with endpoint pruning"
```

---

### Task 9: Backend — subscribe/unsubscribe endpoints

**Files:**
- Modify: `app/backend/main.py`
- Modify: `app/backend/tests/conftest.py`
- Test: `app/backend/tests/test_push_endpoints.py`

- [ ] **Step 1: Add the autouse test override (before writing the endpoints)**

`conftest.py` currently has no `sys.path` bootstrap of its own (every individual test file adds one, but conftest.py loads before any of them run, so it needs its own). Replace its top `from __future__ import annotations` / `import os` / `import socket` / `import pytest` block with:
```python
from __future__ import annotations

from pathlib import Path
import os
import socket
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

import pytest

from app.backend.main import app, get_push_store
from app.backend.push_subscriptions import PushSubscriptionStore
```
(Verified live: without the `sys.path.append` line, that import raises `ModuleNotFoundError: No module named 'app'`, not the "doesn't exist yet" error Step 3 below expects — every test file in this suite relies on exactly this append, conftest.py is the one file that never needed it before now.)

And add this new fixture at the end of the file:
```python


@pytest.fixture(autouse=True)
def _default_push_store(tmp_path, request):
    """Every test in this suite gets a tmp_path-scoped PushSubscriptionStore
    by default (mirrors this file's own _block_real_network: a safety net so
    no test has to know push exists, not a replacement for a test that
    specifically wants to assert push behavior -- that test overrides
    get_push_store again with its own store, inside its own body, after
    this fixture has already run)."""
    if request.node.get_closest_marker("live"):
        yield
        return
    store = PushSubscriptionStore(db_path=tmp_path / "push_subscriptions.db")
    app.dependency_overrides[get_push_store] = lambda: store
    yield
    app.dependency_overrides.pop(get_push_store, None)
```
This references `get_push_store`, which doesn't exist yet — that's expected; it's added in Step 3 below. (This ordering — test infra before the code it guards — is deliberate: Step 2's test run is the thing that proves the guard is needed.)

- [ ] **Step 2: Write the failing test**

```python
from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend.auth_deps import get_current_user_email
from app.backend.main import app, get_push_store, get_user_store
from app.backend.push_subscriptions import PushSubscriptionStore
from app.backend.users import UserStore


def _override_user(tmp_path: Path, email: str = "test-user@example.com"):
    store = UserStore(db_path=tmp_path / "users.db")
    app.dependency_overrides[get_current_user_email] = lambda: email
    app.dependency_overrides[get_user_store] = lambda: store
    return store.get_or_create(email)


def test_push_subscribe_stores_the_subscription(tmp_path: Path):
    user = _override_user(tmp_path)
    push_store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    app.dependency_overrides[get_push_store] = lambda: push_store
    with TestClient(app) as client:
        response = client.post(
            "/api/push/subscribe",
            json={"endpoint": "https://push.example/abc", "p256dh": "key", "auth": "auth"},
        )
    assert response.status_code == 204
    subs = push_store.list_for_user(user.id)
    assert len(subs) == 1
    assert subs[0].endpoint == "https://push.example/abc"


def test_push_unsubscribe_removes_the_subscription(tmp_path: Path):
    _override_user(tmp_path)
    push_store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    app.dependency_overrides[get_push_store] = lambda: push_store
    with TestClient(app) as client:
        client.post(
            "/api/push/subscribe",
            json={"endpoint": "https://push.example/abc", "p256dh": "key", "auth": "auth"},
        )
        response = client.request(
            "DELETE", "/api/push/subscribe", json={"endpoint": "https://push.example/abc"}
        )
    assert response.status_code == 204
    assert push_store.list_subscribed_user_ids() == []
```
Save as `app/backend/tests/test_push_endpoints.py`.

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest app/backend/tests/test_push_endpoints.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_push_store' from 'app.backend.main'`.

- [ ] **Step 4: Add the endpoints**

In `app/backend/main.py`, add this import near the other `app.backend` imports (next to `from app.backend.users import UserStore`):
```python
from app.backend.push_notifications import send_to_user as send_push_to_user
from app.backend.push_subscriptions import PushSubscriptionStore
```

Add this right after `get_user_store()`'s definition (after the `return _user_store_singleton` line, before `_settle_if_already_decided`):
```python
_push_store_singleton: PushSubscriptionStore | None = None


def get_push_store() -> PushSubscriptionStore:
    """FastAPI dependency -- overridden in tests via conftest.py's
    autouse _default_push_store fixture, matching get_user_store's own
    singleton pattern."""
    global _push_store_singleton
    if _push_store_singleton is None:
        _push_store_singleton = PushSubscriptionStore()
    return _push_store_singleton


class PushSubscribeRequest(BaseModel):
    endpoint: str
    p256dh: str
    auth: str


class PushUnsubscribeRequest(BaseModel):
    endpoint: str


@app.post("/api/push/subscribe", status_code=204)
async def push_subscribe(
    request: PushSubscribeRequest,
    user_email: str = Depends(get_current_user_email),
    user_store: UserStore = Depends(get_user_store),
    push_store: PushSubscriptionStore = Depends(get_push_store),
) -> None:
    user = user_store.get_or_create(user_email)
    push_store.add(user.id, request.endpoint, request.p256dh, request.auth)


@app.delete("/api/push/subscribe", status_code=204)
async def push_unsubscribe(
    request: PushUnsubscribeRequest,
    push_store: PushSubscriptionStore = Depends(get_push_store),
) -> None:
    push_store.remove(request.endpoint)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest app/backend/tests/test_push_endpoints.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Run the full backend suite to confirm the autouse fixture didn't break anything**

Run: `pytest app/backend/tests -q`
Expected: all tests pass (same count as before this task, plus the 2 new ones).

- [ ] **Step 7: Commit**

```bash
git add app/backend/main.py app/backend/tests/conftest.py app/backend/tests/test_push_endpoints.py
git commit -m "feat(backend): add POST/DELETE /api/push/subscribe endpoints"
```

---

### Task 10: Wire the "new recommendations ready" push trigger

**Files:**
- Modify: `app/backend/scheduler_wiring.py`
- Modify: `app/backend/main.py` (the `register_eod_job` call site)
- Test: `app/backend/tests/test_scheduler_wiring.py` (append one test)

- [ ] **Step 1: Write the failing test**

Append to the end of `app/backend/tests/test_scheduler_wiring.py`:
```python


def test_register_eod_job_notifies_subscribers_once_after_the_nightly_run(tmp_path: Path) -> None:
    """New in this task: register_eod_job accepts an optional push_store --
    when supplied, every existing per-league test above (which omit it) is
    completely unaffected (push_store defaults to None -> notification step
    skipped entirely, no PushSubscriptionStore ever constructed for those
    tests). Only a test that explicitly opts in, like this one, exercises
    the notify path."""
    fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [fixture]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    push_store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    push_store.add(user_id=1, endpoint="https://push.example/a", p256dh="k", auth="a")

    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    recommendation = {
        "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
        "overall": "no_bet", "markets": [], "explanation": "test", "confidence": "low",
        "limitations": [], "prediction_basis": "market_odds_only",
    }
    with patch("app.backend.recommendations.run_agent", return_value=recommendation), \
         patch("app.backend.scheduler_wiring.send_to_all_subscribers") as mock_send:
        register_eod_job(
            scheduler, fixtures_client=fixtures_client, odds_client=None,
            cache=cache, config=config, now_fn=lambda: now, push_store=push_store,
        )
        assert _wait_until(lambda: run_log.has_run(EOD_JOB_ID, _FUTURE_DAY_STR))

    assert mock_send.call_count == 1
    _, kwargs = mock_send.call_args
    assert kwargs["store"] is push_store
```
Add the new imports this test needs at the top of the file (next to the other `app.backend` imports): `from app.backend.push_subscriptions import PushSubscriptionStore`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest app/backend/tests/test_scheduler_wiring.py::test_register_eod_job_notifies_subscribers_once_after_the_nightly_run -v`
Expected: FAIL — `TypeError: register_eod_job() got an unexpected keyword argument 'push_store'`.

- [ ] **Step 3: Add the `push_store` parameter and the post-loop notification call**

In `app/backend/scheduler_wiring.py`, add this import near the top (with the other `app.backend` imports):
```python
from app.backend.push_notifications import send_to_all_subscribers
from app.backend.push_subscriptions import PushSubscriptionStore
```

Change `register_eod_job`'s signature (add the new parameter at the end, default `None` so every existing call site — including all 7 pre-existing tests — is unaffected):
```python
def register_eod_job(
    scheduler: RecoverableScheduler,
    fixtures_client: FootballDataClient,
    odds_client: OddsAPIClient | None,
    cache: RecommendationCache,
    config: AgentConfig,
    now_fn: Callable[[], datetime] = lambda: sandbox_now(NY_TZ),
    sweden_fixtures_client: SwedenFixturesClient | None = None,
    la_liga_fixtures_client: FootballDataClient | None = None,
    serie_a_fixtures_client: FootballDataClient | None = None,
    bundesliga_fixtures_client: FootballDataClient | None = None,
    ligue1_fixtures_client: FootballDataClient | None = None,
    oddspapi_client: OddsPapiClient | None = None,
    push_store: PushSubscriptionStore | None = None,
) -> None:
```

Inside `_eod_job()`, after the `for league in COMPETITIONS:` loop finishes (i.e. right before the `scheduler.schedule_daily(EOD_JOB_ID, _eod_job, hour=EOD_HOUR, minute=EOD_MINUTE)` line, still inside `_eod_job`'s own body — so indented one level in, as the loop's final sibling statement), add:
```python
        # None (the default, and every pre-existing caller) means push
        # notifications aren't wired for this run -- skip entirely rather
        # than constructing a store no caller asked for.
        if push_store is not None:
            send_to_all_subscribers(
                store=push_store, title="New recommendations ready",
                body=f"Tomorrow's picks ({date_str}) are in.", url="/",
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest app/backend/tests/test_scheduler_wiring.py -v`
Expected: PASS (all tests, including the new one — the 7 pre-existing ones are untouched since they never pass `push_store`).

- [ ] **Step 5: Wire it at the real call site**

In `app/backend/main.py`, find the `register_eod_job(...)` call (around line 360) and add `push_store=get_push_store(),` as one more keyword argument (anywhere in the call, e.g. right after `config=config,`).

- [ ] **Step 6: Commit**

```bash
git add app/backend/scheduler_wiring.py app/backend/main.py app/backend/tests/test_scheduler_wiring.py
git commit -m "feat(backend): push-notify subscribers once the nightly EOD batch completes"
```

---

### Task 11: Wire the "bet settled" push trigger

**Files:**
- Modify: `app/backend/main.py`
- Test: `app/backend/tests/test_bet_settlement_push_notification.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend import bets
from app.backend.auth_deps import get_current_user_email
from app.backend.bet_tracker import BetTracker
from app.backend.football_data_client import NormalizedMatch
from app.backend.main import app, get_push_store, get_user_store
from app.backend.push_subscriptions import PushSubscriptionStore
from app.backend.users import UserStore


def _override_tracker(tmp_path: Path) -> BetTracker:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    app.dependency_overrides[bets.get_bet_tracker] = lambda: tracker
    return tracker


def _override_user(tmp_path: Path, email: str = "test-user@example.com"):
    store = UserStore(db_path=tmp_path / "users.db")
    app.dependency_overrides[get_current_user_email] = lambda: email
    app.dependency_overrides[get_user_store] = lambda: store
    return store.get_or_create(email)


def test_settle_open_sends_a_push_notification_for_each_settled_bet(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
    user = _override_user(tmp_path)
    push_store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    app.dependency_overrides[get_push_store] = lambda: push_store
    tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None, user_id=user.id,
    )
    with patch("app.backend.main.get_sweden_fixtures_client") as mock_sweden:
        mock_sweden.return_value.get_results.return_value = []
        with patch("app.backend.main.get_fixtures_client") as mock_client:
            mock_client.return_value.get_results.return_value = [
                NormalizedMatch(
                    match_id="m1", utc_date="2026-08-22T15:00:00Z", status="FINISHED",
                    home_team="Arsenal", away_team="Everton", home_goals=2, away_goals=1,
                ),
            ]
            with patch("app.backend.main.send_push_to_user") as mock_send:
                with TestClient(app) as client:
                    response = client.post("/api/bets/settle-open")
    assert response.status_code == 200
    assert mock_send.call_count == 1
    args, kwargs = mock_send.call_args
    assert args[1] == user.id or kwargs.get("user_id") == user.id
```
Save as `app/backend/tests/test_bet_settlement_push_notification.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest app/backend/tests/test_bet_settlement_push_notification.py -v`
Expected: FAIL — `AssertionError: assert 0 == 1` (no push sent yet — `send_push_to_user` is never called by `/api/bets/settle-open` today).

- [ ] **Step 3: Add the shared notify helper and call it from both settlement call sites**

In `app/backend/main.py`, add this function right after `_settle_if_already_decided`'s closing (after its `return tracker.get_bet(bet.id) or bet` line, before `@app.post("/api/bets/from-recommendation")`):
```python
def _notify_bet_settled(push_store: PushSubscriptionStore, bet: Bet) -> None:
    """Shared by both settlement call sites below -- auto-settle-on-log and
    the on-demand Settle open bets button -- so a bet settling through
    either path notifies its owner exactly once, from one place."""
    if bet.user_id is None or bet.outcome == "open":
        return
    result_word = "won" if bet.outcome == "won" else "lost"
    send_push_to_user(
        push_store, bet.user_id,
        title=f"Bet settled: {result_word}",
        body=f"{bet.home_team} vs {bet.away_team} -- {bet.market} {bet.selection}",
        url="/bets",
    )
```

Update `_settle_if_already_decided`'s signature to accept the store and notify each settled bet. Change:
```python
async def _settle_if_already_decided(tracker: BetTracker, bet: Bet, user_id: int) -> Bet:
```
to:
```python
async def _settle_if_already_decided(
    tracker: BetTracker, bet: Bet, user_id: int, push_store: PushSubscriptionStore
) -> Bet:
```
and change its body from:
```python
    try:
        client = get_fixtures_client()
        sweden_client = get_sweden_fixtures_client()
        await run_in_threadpool(settle_open_bets, tracker, client, sweden_client, user_id=user_id, blocking=False)
    except Exception:
        LOGGER.warning("Auto-settle-on-log failed for bet %s -- left open.", bet.id, exc_info=True)
        return bet
    return tracker.get_bet(bet.id) or bet
```
to:
```python
    try:
        client = get_fixtures_client()
        sweden_client = get_sweden_fixtures_client()
        settled = await run_in_threadpool(
            settle_open_bets, tracker, client, sweden_client, user_id=user_id, blocking=False
        )
        for settled_bet in settled:
            _notify_bet_settled(push_store, settled_bet)
    except Exception:
        LOGGER.warning("Auto-settle-on-log failed for bet %s -- left open.", bet.id, exc_info=True)
        return bet
    return tracker.get_bet(bet.id) or bet
```

Update its two callers (`create_bet_from_recommendation` and `create_bet_manual`) to pass the store. Each currently ends with a line like:
```python
    bet = await _settle_if_already_decided(tracker, bet, user.id)
```
Change both to:
```python
    bet = await _settle_if_already_decided(tracker, bet, user.id, push_store)
```
and add `push_store: PushSubscriptionStore = Depends(get_push_store),` to both route functions' parameter lists (alongside their existing `user_store: UserStore = Depends(get_user_store),` line).

Finally, update the `/api/bets/settle-open` endpoint itself. Change:
```python
@app.post("/api/bets/settle-open")
async def settle_open(
    blocking: bool = True,
    tracker: BetTracker = Depends(bets.get_bet_tracker),
    user_email: str = Depends(get_current_user_email),
    user_store: UserStore = Depends(get_user_store),
) -> list[BetOut]:
    ...
    user = user_store.get_or_create(user_email)
    client = get_fixtures_client()
    sweden_client = get_sweden_fixtures_client()
    settled = await run_in_threadpool(
        settle_open_bets, tracker, client, sweden_client, user_id=user.id, blocking=blocking
    )
    return [BetOut.from_bet(bet) for bet in settled]
```
to add `push_store: PushSubscriptionStore = Depends(get_push_store),` to the parameter list and notify after settling:
```python
@app.post("/api/bets/settle-open")
async def settle_open(
    blocking: bool = True,
    tracker: BetTracker = Depends(bets.get_bet_tracker),
    user_email: str = Depends(get_current_user_email),
    user_store: UserStore = Depends(get_user_store),
    push_store: PushSubscriptionStore = Depends(get_push_store),
) -> list[BetOut]:
    ...
    user = user_store.get_or_create(user_email)
    client = get_fixtures_client()
    sweden_client = get_sweden_fixtures_client()
    settled = await run_in_threadpool(
        settle_open_bets, tracker, client, sweden_client, user_id=user.id, blocking=blocking
    )
    for settled_bet in settled:
        _notify_bet_settled(push_store, settled_bet)
    return [BetOut.from_bet(bet) for bet in settled]
```
(The `...` above is the existing unchanged docstring -- leave it as-is.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest app/backend/tests/test_bet_settlement_push_notification.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full backend suite**

Run: `pytest app/backend/tests -q`
Expected: all tests pass, including `test_settlement_endpoint.py` and `test_bets_endpoints.py` (unaffected: `send_push_to_user` always finds zero subscriptions in their tmp_path-scoped autouse push store, so it's a real-but-empty-and-harmless call in those tests).

- [ ] **Step 6: Commit**

```bash
git add app/backend/main.py app/backend/tests/test_bet_settlement_push_notification.py
git commit -m "feat(backend): push-notify a bet's owner when it settles (either settlement path)"
```

---

### Task 12: Frontend — push subscribe/unsubscribe proxy + client helper

**Files:**
- Create: `app/frontend/app/api/push/route.ts`
- Create: `app/frontend/lib/push.ts`
- Test: `app/frontend/lib/__tests__/push.test.ts`

- [ ] **Step 1: Write the proxy route**

```ts
import { forwardToBackend } from "@/lib/backendProxy";

export async function POST(request: Request) {
  const body = await request.text();
  const response = await forwardToBackend("/api/push/subscribe", { method: "POST", body });
  return new Response(response.status === 204 ? null : await response.text(), { status: response.status });
}

export async function DELETE(request: Request) {
  const body = await request.text();
  const response = await forwardToBackend("/api/push/subscribe", { method: "DELETE", body });
  return new Response(response.status === 204 ? null : await response.text(), { status: response.status });
}
```
Save as `app/frontend/app/api/push/route.ts`.

- [ ] **Step 2: Write the failing test for the client helper**

```ts
import { describe, expect, it, vi, beforeEach } from "vitest";
import { isPushSubscribed, subscribeToPush, unsubscribeFromPush } from "../push";

const mockSubscribe = vi.fn();
const mockGetSubscription = vi.fn();
const mockUnsubscribe = vi.fn();
const fakeFetch = vi.fn();

beforeEach(() => {
  mockSubscribe.mockReset();
  mockGetSubscription.mockReset();
  mockUnsubscribe.mockReset();
  fakeFetch.mockReset().mockResolvedValue({ ok: true });
  global.fetch = fakeFetch as unknown as typeof fetch;
  Object.defineProperty(window.navigator, "serviceWorker", {
    configurable: true,
    value: {
      ready: Promise.resolve({
        pushManager: { subscribe: mockSubscribe, getSubscription: mockGetSubscription },
      }),
    },
  });
});

describe("isPushSubscribed", () => {
  it("returns false when there is no active subscription", async () => {
    mockGetSubscription.mockResolvedValue(null);
    expect(await isPushSubscribed()).toBe(false);
  });

  it("returns true when a subscription exists", async () => {
    mockGetSubscription.mockResolvedValue({});
    expect(await isPushSubscribed()).toBe(true);
  });
});

describe("subscribeToPush", () => {
  it("subscribes via PushManager and posts the subscription to /api/push", async () => {
    mockSubscribe.mockResolvedValue({
      endpoint: "https://push.example/abc",
      toJSON: () => ({ keys: { p256dh: "key", auth: "auth" } }),
    });
    await subscribeToPush();
    expect(mockSubscribe).toHaveBeenCalledWith(expect.objectContaining({ userVisibleOnly: true }));
    expect(fakeFetch).toHaveBeenCalledWith(
      "/api/push",
      expect.objectContaining({ method: "POST" })
    );
    const body = JSON.parse(fakeFetch.mock.calls[0][1].body as string);
    expect(body).toEqual({ endpoint: "https://push.example/abc", p256dh: "key", auth: "auth" });
  });
});

describe("unsubscribeFromPush", () => {
  it("deletes the server-side subscription and unsubscribes locally", async () => {
    mockGetSubscription.mockResolvedValue({
      endpoint: "https://push.example/abc",
      unsubscribe: mockUnsubscribe,
    });
    await unsubscribeFromPush();
    expect(fakeFetch).toHaveBeenCalledWith(
      "/api/push",
      expect.objectContaining({ method: "DELETE" })
    );
    expect(mockUnsubscribe).toHaveBeenCalled();
  });

  it("does nothing when there is no active subscription", async () => {
    mockGetSubscription.mockResolvedValue(null);
    await unsubscribeFromPush();
    expect(fakeFetch).not.toHaveBeenCalled();
  });
});
```
Save as `app/frontend/lib/__tests__/push.test.ts`.

- [ ] **Step 3: Run test to verify it fails**

Run: `cd app/frontend && npx vitest run lib/__tests__/push.test.ts`
Expected: FAIL — `Cannot find module '../push'`.

- [ ] **Step 4: Write the client helper**

```ts
const PUBLIC_KEY = process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY;

function urlBase64ToUint8Array(base64String: string): Uint8Array {
  const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
  const rawData = atob(base64);
  return Uint8Array.from([...rawData].map((char) => char.charCodeAt(0)));
}

export async function isPushSubscribed(): Promise<boolean> {
  if (!("serviceWorker" in navigator)) return false;
  const registration = await navigator.serviceWorker.ready;
  const subscription = await registration.pushManager.getSubscription();
  return subscription !== null;
}

export async function subscribeToPush(): Promise<void> {
  if (!PUBLIC_KEY) throw new Error("Push notifications are not configured.");
  const registration = await navigator.serviceWorker.ready;
  const subscription = await registration.pushManager.subscribe({
    userVisibleOnly: true,
    applicationServerKey: urlBase64ToUint8Array(PUBLIC_KEY),
  });
  const json = subscription.toJSON();
  await fetch("/api/push", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      endpoint: subscription.endpoint,
      p256dh: json.keys?.p256dh,
      auth: json.keys?.auth,
    }),
  });
}

export async function unsubscribeFromPush(): Promise<void> {
  const registration = await navigator.serviceWorker.ready;
  const subscription = await registration.pushManager.getSubscription();
  if (!subscription) return;
  await fetch("/api/push", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ endpoint: subscription.endpoint }),
  });
  await subscription.unsubscribe();
}
```
Save as `app/frontend/lib/push.ts`.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd app/frontend && npx vitest run lib/__tests__/push.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add app/frontend/app/api/push/route.ts app/frontend/lib/push.ts app/frontend/lib/__tests__/push.test.ts
git commit -m "feat(frontend): add push subscribe/unsubscribe proxy route and client helper"
```

---

### Task 13: Frontend — notification toggle in `UserMenu`

**Files:**
- Modify: `app/frontend/components/UserMenu.tsx`
- Modify: `app/frontend/components/__tests__/UserMenu.test.tsx`

- [ ] **Step 1: Add the failing tests**

Add this mock near the top of `app/frontend/components/__tests__/UserMenu.test.tsx` (alongside the existing `vi.mock("next-auth/react", ...)`):
```tsx
const mockIsPushSubscribed = vi.fn();
const mockSubscribeToPush = vi.fn();
const mockUnsubscribeFromPush = vi.fn();
vi.mock("@/lib/push", () => ({
  isPushSubscribed: (...args: unknown[]) => mockIsPushSubscribed(...args),
  subscribeToPush: (...args: unknown[]) => mockSubscribeToPush(...args),
  unsubscribeFromPush: (...args: unknown[]) => mockUnsubscribeFromPush(...args),
}));
```
Add `mockIsPushSubscribed.mockReset().mockResolvedValue(false);`, `mockSubscribeToPush.mockReset().mockResolvedValue(undefined);`, and `mockUnsubscribeFromPush.mockReset().mockResolvedValue(undefined);` to the existing `beforeEach`.

Append these tests inside the existing `describe("UserMenu", ...)` block:
```tsx
  it("shows a notifications toggle for a signed-in user when push is supported", async () => {
    Object.defineProperty(window.navigator, "serviceWorker", { value: {}, configurable: true });
    Object.defineProperty(window, "PushManager", { value: function () {}, configurable: true });
    mockUseSession.mockReturnValue({
      data: { user: { email: "fpai.deploy@gmail.com" } },
      status: "authenticated",
    });
    render(<UserMenu />);
    expect(await screen.findByRole("button", { name: /turn on notifications/i })).toBeInTheDocument();
  });

  it("subscribes when the toggle is clicked while off", async () => {
    Object.defineProperty(window.navigator, "serviceWorker", { value: {}, configurable: true });
    Object.defineProperty(window, "PushManager", { value: function () {}, configurable: true });
    mockUseSession.mockReturnValue({
      data: { user: { email: "fpai.deploy@gmail.com" } },
      status: "authenticated",
    });
    render(<UserMenu />);
    const toggle = await screen.findByRole("button", { name: /turn on notifications/i });
    await userEvent.click(toggle);
    expect(mockSubscribeToPush).toHaveBeenCalled();
  });

  it("hides the toggle when the browser has no PushManager support", () => {
    Object.defineProperty(window.navigator, "serviceWorker", { value: undefined, configurable: true });
    Object.defineProperty(window, "PushManager", { value: undefined, configurable: true });
    mockUseSession.mockReturnValue({
      data: { user: { email: "fpai.deploy@gmail.com" } },
      status: "authenticated",
    });
    render(<UserMenu />);
    expect(screen.queryByText(/notifications/i)).not.toBeInTheDocument();
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app/frontend && npx vitest run components/__tests__/UserMenu.test.tsx`
Expected: FAIL — the three new tests fail (no notifications toggle rendered yet).

- [ ] **Step 3: Update `UserMenu`**

Replace the full contents of `app/frontend/components/UserMenu.tsx` with:
```tsx
"use client";

// W210 follow-up: the one piece of session UI this app had zero of --
// before this, a signed-in user had no way to see which account they were
// using or to sign out short of manually hitting /api/auth/signout. Lives
// in AppShell so it's present on every page, signed-in or not.
import Link from "next/link";
import { useEffect, useState } from "react";
import { signOut, useSession } from "next-auth/react";
import { isPushSubscribed, subscribeToPush, unsubscribeFromPush } from "@/lib/push";

function canUsePush(): boolean {
  if (typeof window === "undefined") return false;
  if (!("serviceWorker" in navigator) || !("PushManager" in window)) return false;
  const ios = /iphone|ipad|ipod/i.test(window.navigator.userAgent);
  if (!ios) return true;
  // iOS only fires push for a PWA actually added to the home screen -- a
  // plain Safari tab's permission prompt would be a guaranteed no-op there.
  return (
    window.matchMedia("(display-mode: standalone)").matches ||
    (window.navigator as Navigator & { standalone?: boolean }).standalone === true
  );
}

export function UserMenu() {
  const { data: session, status } = useSession();
  const [pushAvailable, setPushAvailable] = useState(false);
  const [pushSubscribed, setPushSubscribed] = useState(false);

  useEffect(() => {
    if (status !== "authenticated") return;
    const available = canUsePush();
    setPushAvailable(available);
    if (available) {
      isPushSubscribed().then(setPushSubscribed);
    }
  }, [status]);

  async function togglePush() {
    if (pushSubscribed) {
      await unsubscribeFromPush();
      setPushSubscribed(false);
    } else {
      await subscribeToPush();
      setPushSubscribed(true);
    }
  }

  if (status === "loading") return null;

  if (status === "unauthenticated" || !session?.user?.email) {
    return (
      <Link href="/login" className="text-xs font-medium text-accent">
        Sign in
      </Link>
    );
  }

  return (
    <div className="flex flex-col gap-2 text-xs">
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-ink-secondary" title={session.user.email}>
          {session.user.email}
        </span>
        <button
          type="button"
          onClick={() => signOut({ callbackUrl: "/" })}
          className="font-medium text-accent"
        >
          Sign out
        </button>
      </div>
      {pushAvailable && (
        <button type="button" onClick={togglePush} className="text-left text-ink-secondary">
          {pushSubscribed ? "Turn off notifications" : "Turn on notifications"}
        </button>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app/frontend && npx vitest run components/__tests__/UserMenu.test.tsx`
Expected: PASS (all tests, old and new).

- [ ] **Step 5: Run the full frontend suite**

Run: `cd app/frontend && npx vitest run`
Expected: all tests pass (nothing else imports `UserMenu` in a way `@/lib/push`'s new import would break — it's mocked globally the same way `next-auth/react` already is for every other test via `vitest.setup.ts`... actually `@/lib/push` is NOT globally mocked, only within `UserMenu.test.tsx` — confirm no other test file renders `UserMenu`/`AppShell` without going through jsdom's `serviceWorker`-undefined default, which makes `canUsePush()` return `false` and the toggle simply not render, same as today).

- [ ] **Step 6: Commit**

```bash
git add app/frontend/components/UserMenu.tsx app/frontend/components/__tests__/UserMenu.test.tsx
git commit -m "feat(frontend): add a push-notification toggle to UserMenu"
```

---

### Task 14: Documentation updates

**Files:**
- Modify: `documents/app_prd.md`
- Modify: `documents/app_techspec.md`
- Modify: `documents/app_user_stories.md`

- [ ] **Step 1: Update the PRD's non-goal**

In `documents/app_prd.md`, Section 5 ("Non-Goals"), change:
```
- Native mobile app.
```
to:
```
- A native/app-store-distributed mobile app (Capacitor, React Native, etc.) -- superseded in part by the installable PWA below, which covers push/offline/home-screen-install without a native wrapper or store distribution. Revisit only if deep native device APIs or store presence become a real requirement.
```

Add a new subsection at the end of Section 2 ("Product Positioning"), after 2.3:
```

### 2.4 Mobile Platform
**Implemented <date this task actually ships>.** The web app is also an installable Progressive Web App on both Android and iOS: home-screen install (platform-aware -- a native prompt on Android, manual "Add to Home Screen" instructions on iOS, since no install-prompt API exists there), an offline fallback page, Web Push notifications (new daily recommendations, bet settlement), and native-feeling polish (safe-area insets, tap haptics on Android, tab-switch transitions). See `documents/app_techspec.md` for the implementation and `docs/superpowers/specs/2026-10-15-pwa-mobile-app-design.md` for the full design rationale.
```
(Replace `<date this task actually ships>` with the real date when you do this step -- don't leave the placeholder in.)

- [ ] **Step 2: Add a techspec entry**

`documents/app_techspec.md` numbers feature sections sequentially under section 15 as `### 15.N Title (W###, date)` (see 15.24/15.25 for the most recent examples) and tags each with this project's own ticket number. Look up the next free `W###` (one past whatever the most recent entry in the file uses) and today's real date, then append this section at the end of the file, filling both in:

```markdown
### 15.26 Installable Mobile App (PWA): Install, Offline, Push, Polish (W###, <date>)

Converts the existing responsive web app into an installable PWA, working on both Android and iOS, with no second codebase and no app-store distribution -- full rationale in `docs/superpowers/specs/2026-10-15-pwa-mobile-app-design.md`.

**Install:** `app/frontend/public/manifest.json` + two generated icon sizes (`sips`-resized from the existing `oddsey-logo.png`) + `display: "standalone"` wired into `layout.tsx`'s `metadata`/`viewport` exports. Android gets a native `beforeinstallprompt` flow; iOS has no install-prompt API at all, so `InstallAppBanner.tsx` shows manual "Add to Home Screen" instructions there instead, gated on `navigator.standalone`/`matchMedia("(display-mode: standalone)")` so it never shows once already installed.

**Offline:** a hand-written service worker (`public/sw.js`, no Workbox/next-pwa dependency) caches the app shell on install and serves `/offline` (`app/offline/page.tsx`, deliberately not wrapped in `AppShell` since it must render from cache alone) on any failed `fetch`.

**Push notifications:** Web Push/VAPID via `pywebpush` -- pinned to `1.14.1`, not latest, since 2.x's `cryptography>=47` requirement conflicts with `mlflow`'s own `<47` pin in this repo's one shared `requirements.txt`. Subscriptions live in a new `push_subscriptions` SQLite table (`app/backend/push_subscriptions.py`, same per-concern pattern as `users.db`), registered/unregistered via `POST`/`DELETE /api/push/subscribe` (proxied through `app/frontend/app/api/push/route.ts` the same way `/api/bets/*` already is) and toggled from `UserMenu.tsx`. `push_notifications.py`'s `send_to_user`/`send_to_all_subscribers` prune a subscription automatically on a `404`/`410` from the push service (a browser can silently invalidate its own endpoint without ever calling the unsubscribe endpoint). Two trigger points, both reusing existing jobs rather than scheduling anything new: `scheduler_wiring.py`'s nightly EOD job notifies every subscriber once after it finishes (an optional `push_store` parameter, defaulting to `None` -- meaning "not wired," not "construct one" -- so none of the function's pre-existing tests needed updating); `main.py`'s `_notify_bet_settled` is called from both places a bet can settle (`_settle_if_already_decided`'s auto-settle-on-log path and the explicit `/api/bets/settle-open` endpoint), one shared function instead of duplicated per caller.

**Polish:** safe-area insets (`env(safe-area-inset-*)`) on the mobile header/bottom tab bar, `navigator.vibrate` tap feedback on the bottom tab bar (Android only -- no-op on iOS, which has no vibration API), and a `view-transition-name` on the main content region paired with a `document.startViewTransition()`-wrapped tab click handler (Chrome/Edge only, plain instant navigation everywhere else).

**Deploy prerequisite:** a VAPID keypair must be generated once (`requirements.txt`'s `pywebpush` entry documents the exact commands) and set as `VAPID_PRIVATE_KEY`/`VAPID_CONTACT_EMAIL` (backend) and `NEXT_PUBLIC_VAPID_PUBLIC_KEY` (frontend) before push notifications work in any deployed environment -- everything else in this feature works without it (push sends just no-op with a logged warning).
```

- [ ] **Step 3: Mark the user story complete**

`documents/app_user_stories.md` tracks each story as a table row: `| W### | status | Description | Size/Depends/Completion notes |` (see the W232 row for the exact shape). Using the same `W###` and date chosen in Step 2, append a row:

```markdown
| W### | completed | **Installable Mobile App (PWA).** The web app is now installable on both Android and iOS: home-screen install (platform-aware -- native prompt on Android, manual instructions on iOS), an offline fallback page, Web Push notifications (new daily recommendations, bet settlement), and native-feeling polish (safe-area insets, tap haptics, tab-switch transitions). No native wrapper, no app-store distribution -- supersedes `app_prd.md`'s old "Native mobile app" non-goal. | Size L · Design: `docs/superpowers/specs/2026-10-15-pwa-mobile-app-design.md` · Plan: `docs/superpowers/plans/2026-10-15-pwa-mobile-app.md`. **Completion notes (<date>):** see `app_techspec.md` §15.26 for the full implementation. One-time deploy prerequisite before push notifications work live: generate a VAPID keypair and set `VAPID_PRIVATE_KEY`/`VAPID_CONTACT_EMAIL` (backend) + `NEXT_PUBLIC_VAPID_PUBLIC_KEY` (frontend) -- see `requirements.txt`'s `pywebpush` entry for the exact one-time commands. Everything else in this feature works even if that step is skipped (push sends just no-op with a logged warning). |
```

- [ ] **Step 4: Commit**

```bash
git add documents/app_prd.md documents/app_techspec.md documents/app_user_stories.md
git commit -m "docs: document the installable PWA mobile app (CLAUDE.md workflow)"
```

---

## After all tasks: full verification

Run the complete test suites once more, end to end:

```bash
pytest app/backend/tests -q
cd app/frontend && npx vitest run && npm run build
```
Expected: everything green, and the production build still succeeds with the new metadata/manifest/service-worker files in place.

Then, manually (can't be automated in CI): deploy to a staging/dev URL, install the PWA on one real Android phone and one real iPhone, kill the network and confirm the offline page, and trigger one end-to-end push notification on each (e.g. via the settle-open button against a bet you've logged on a finished match).
