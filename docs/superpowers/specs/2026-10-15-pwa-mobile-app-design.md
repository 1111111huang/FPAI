# Installable Mobile App (PWA) for Oddsey

## 1. Problem

`documents/app_prd.md` currently lists "Native mobile app" as a non-goal. The web app (`app/frontend`, Next.js 14 App Router) already has a responsive mobile layout — including a bottom tab bar for small screens (`AppShell.tsx:270-290`, added W114) — but it only runs in a browser tab: no home-screen install, no push notifications, no offline handling, and no native-feeling touches (safe-area insets, haptics, transitions).

This spec turns the existing web app into an installable Progressive Web App (PWA) that works on both Android and iOS, without a second codebase.

## 2. Scope

**In scope:**
- PWA manifest + icons + standalone display mode (installable, no browser chrome once installed)
- Platform-aware install guidance (Android native install prompt vs. iOS manual "Add to Home Screen" instructions)
- A minimal hand-written service worker: offline-fallback page + app-shell caching
- Push notifications (new daily recommendations ready; a tracked bet settles), via Web Push / VAPID, no third-party push service
- Native-feeling polish: safe-area insets, tap haptics (Android only), tab-switch transitions

**Out of scope (explicitly not doing):**
- A native wrapper (Capacitor/React Native) — straightforward future upgrade if ever needed, since it would wrap this same web app; not needed at this scale (invite-only, allowlisted users, no app-store distribution goal)
- Offline data sync / background sync queue — the app is read-mostly against live data; an offline fallback page is enough
- Redesigning the existing bottom-tab navigation structure — confirmed during design discussion that the nav itself (W114) is fine; the gap was installability + polish, not information architecture
- Changing `documents/app_prd.md`'s non-goal line itself — tracked as a side-effect update in Section 8, not a scope item here

## 3. Platform differences (why both, not just one)

| | Android (Chrome/Edge) | iOS (Safari 16.4+) |
|---|---|---|
| Install | Automatic `beforeinstallprompt` event → native "Install app" prompt | No install-prompt API exists. User must manually use Share → "Add to Home Screen"; the app must build its own instructional UI for this. |
| Push notifications | Work as soon as permission is granted, even from a regular tab | Supported since 16.4, but **only fire once the app is added to the home screen** — a no-op from a plain Safari tab |
| `navigator.vibrate` (haptics) | Supported | Not supported at all — graceful-enhancement only |
| Service worker / cache durability | Full support, no eviction quirks | Installed home-screen apps are treated durably; a page never added to home screen is more aggressive about evicting its cache after inactivity |

Design consequence: the app must detect which situation it's in (`navigator.standalone` on iOS / `matchMedia('(display-mode: standalone)')` cross-platform) and branch the install UI and the push-permission gate accordingly (Section 5).

## 4. PWA shell

- `app/frontend/public/manifest.json`: name "Oddsey", icons generated from the existing `public/oddsey-logo.png` (192/512px), `display: "standalone"`, theme/background colors matching the existing `bg-page`/`bg-surface` tokens.
- `layout.tsx`: add `<link rel="manifest">`, `apple-touch-icon`, and `theme-color` meta tags.
- `app/frontend/public/sw.js`: a small hand-written service worker (no `next-pwa`/Workbox dependency) that:
  - On `install`, caches the app shell (root route + static assets)
  - On `fetch` failure (no network), serves a static `/offline` route instead of the browser's default error screen
  - On `push`, shows a notification (Section 6)
- A small client component (mounted once in `providers.tsx` or `layout.tsx`) registers `sw.js` via `navigator.serviceWorker.register`.

## 5. Install guidance UI

A small banner/card component, shown once per session when the app isn't already running standalone:

- **Android/Chrome:** listens for `beforeinstallprompt`, stores the event, renders an "Install app" button that calls `event.prompt()` when clicked.
- **iOS Safari:** no event to listen for — detect iOS + not-standalone via UA/`navigator.standalone`, render static instructions ("Tap Share, then Add to Home Screen") instead of a button.
- Dismissed state persisted in `localStorage` so it doesn't nag every visit.

## 6. Push notifications

**Receiving (frontend):** standard Push API. `sw.js`'s `push` event handler calls `self.registration.showNotification(...)` with title/body/url from the push payload; a `notificationclick` handler focuses/opens the relevant page (`/` for new recs, `/bets` for a settlement).

**Subscribing (frontend):** a toggle in `UserMenu.tsx` that calls `PushManager.subscribe()` (VAPID public key baked in at build time) and POSTs the resulting subscription to a new backend endpoint. On iOS, this control is hidden/disabled until the standalone check (Section 3) passes — asking for permission before that point is a guaranteed no-op there. On Android, no such gate.

**Storage (backend):** new `push_subscriptions` table in `data/users.db`, following the existing `UserStore` sqlite-per-file pattern (`users.py`) — keyed by user email (one-to-many: a user may have multiple installed devices), storing `endpoint`, `p256dh`, `auth`.

**Sending (backend):** `pywebpush` (new dependency — VAPID-based Web Push, no SaaS account needed at this scale). A small `push_notifications.py` module wrapping "send to all of a user's subscriptions, prune ones that come back expired/410."

**Triggers (hook into existing jobs, nothing new to schedule):**
- After the nightly EOD batch (`eod_batch.py`) finishes producing next-day recommendations → notify all subscribed users once.
- Whenever a tracked bet transitions to settled (win/loss) via the existing settle-open path (`bets.py`/`bet_tracker.py`) → notify that bet's owner.

## 7. Native-feeling polish

- Safe-area insets: `env(safe-area-inset-top/bottom)` padding added to the fixed mobile header and bottom tab bar in `AppShell.tsx`, so they clear notches/home-indicators once installed full-screen.
- Tap haptics: `navigator.vibrate(10)` on bottom-tab-bar taps, feature-detected (no-op where unsupported, i.e. all of iOS).
- Tab-switch transitions: CSS `view-transition-name` on the main content region for native-feeling cross-fades in browsers that support the View Transitions API; no-op fallback (today's instant swap) everywhere else. No animation library.

## 8. Side effects

- `documents/app_prd.md` Section 5 ("Non-Goals") currently lists "Native mobile app." Update to reflect that a PWA (not a native/store-distributed app) is now in scope, once this ships.
- `documents/app_techspec.md` and `documents/app_user_stories.md` get the usual new-functionality entry + user story per `CLAUDE.md`'s workflow rules, written at plan/implementation time, not in this design doc.

## 9. Testing

- Vitest: service-worker registration logic, push-subscribe flow (mocked `PushManager`/`ServiceWorkerRegistration`), install-prompt-banner branching (iOS vs Android UA/standalone detection).
- Backend: unit test for `push_notifications.py`'s send/prune logic (mocked `pywebpush`), and for the two trigger call-sites (EOD batch completion, bet settlement).
- Manual (not automatable): install on a real Android phone and a real iPhone, confirm offline fallback with network killed, confirm one end-to-end push on each platform.

## 10. Open items for implementation time

- VAPID keypair generation and where the private key is stored (Railway env var, matching the existing `INTERNAL_API_SECRET`/`APP_ACCESS_TOKEN` pattern).
- Icon asset generation (192/512px PNGs) from the existing logo — a one-time asset task, not a design decision.
