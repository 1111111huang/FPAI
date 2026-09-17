"use client";

import { Suspense, useEffect, useState } from "react";
import Image from "next/image";
import { useSearchParams } from "next/navigation";
import { signIn, useSession } from "next-auth/react";

const ERROR_MESSAGES: Record<string, string> = {
  AccessDenied: "That Google account is not authorized for this app. Ask the owner to add it to the allowlist.",
};

// Loop guard (2026-09-15): if the client believes it's signed in but
// middleware keeps disagreeing (found live -- a real, if now-fixed,
// production bug), the hard-navigation redirect below would otherwise
// bounce here, redirect, land back here, forever -- a fast, blinking
// full-page-reload loop with no visible error, the worst version of this
// failure. sessionStorage (not state -- must survive the hard reload
// itself) counts attempts within a short window; past the limit, stop
// redirecting and show a real message instead of blinking silently.
const LOOP_GUARD_KEY = "login-redirect-attempts";
const LOOP_GUARD_MAX_ATTEMPTS = 3;
const LOOP_GUARD_WINDOW_MS = 10_000;

function recordRedirectAttempt(): boolean {
  try {
    const raw = sessionStorage.getItem(LOOP_GUARD_KEY);
    const now = Date.now();
    const prev = raw ? (JSON.parse(raw) as { count: number; firstAt: number }) : null;
    const fresh = !prev || now - prev.firstAt > LOOP_GUARD_WINDOW_MS;
    const next = fresh ? { count: 1, firstAt: now } : { count: prev.count + 1, firstAt: prev.firstAt };
    sessionStorage.setItem(LOOP_GUARD_KEY, JSON.stringify(next));
    return next.count <= LOOP_GUARD_MAX_ATTEMPTS;
  } catch {
    // Storage unavailable (private-mode edge case) -- fail open rather than
    // ever blocking a legitimate sign-in over this.
    return true;
  }
}

// Standard 4-color Google "G" mark -- inlined rather than fetched (no
// external-image dependency for a single small icon).
function GoogleIcon() {
  return (
    <svg viewBox="0 0 48 48" width="20" height="20" aria-hidden="true">
      <path fill="#FFC107" d="M43.611,20.083H42V20H24v8h11.303c-1.649,4.657-6.08,8-11.303,8c-6.627,0-12-5.373-12-12c0-6.627,5.373-12,12-12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C12.955,4,4,12.955,4,24c0,11.045,8.955,20,20,20c11.045,0,20-8.955,20-20C44,22.659,43.862,21.35,43.611,20.083z" />
      <path fill="#FF3D00" d="M6.306,14.691l6.571,4.819C14.655,15.108,18.961,12,24,12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C16.318,4,9.656,8.337,6.306,14.691z" />
      <path fill="#4CAF50" d="M24,44c5.166,0,9.86-1.977,13.409-5.192l-6.19-5.238C29.211,35.091,26.715,36,24,36c-5.202,0-9.619-3.317-11.283-7.946l-6.522,5.025C9.505,39.556,16.227,44,24,44z" />
      <path fill="#1976D2" d="M43.611,20.083H42V20H24v8h11.303c-0.792,2.237-2.231,4.166-4.087,5.571c0.001-0.001,0.002-0.001,0.003-0.002l6.19,5.238C36.971,39.205,44,34,44,24C44,22.659,43.862,21.35,43.611,20.083z" />
    </svg>
  );
}

function LoginPageInner() {
  const { status } = useSession();
  const searchParams = useSearchParams();
  const [loopDetected, setLoopDetected] = useState(false);

  const callbackUrl = searchParams.get("callbackUrl") ?? "/bets";
  const error = searchParams.get("error");

  useEffect(() => {
    if (status !== "authenticated") return;
    // Production bug (2026-09-15): a plain router.push() here is a soft
    // client-side navigation, which can replay a *stale* middleware
    // redirect that Next.js's router cache captured from before sign-in
    // (e.g. an auto-prefetch of this same callbackUrl fired the moment
    // AppShell mounted, pre-login, with no session cookie yet) -- the user
    // is stuck bouncing back to /login forever even though they're really
    // signed in, confirmed live via a temp middleware debug log showing
    // req.cookies missing the session cookie entirely on that soft nav. A
    // hard navigation bypasses the router cache and asks the server fresh,
    // matching the one thing that reliably worked: manually re-entering
    // the URL.
    if (recordRedirectAttempt()) {
      window.location.href = callbackUrl;
    } else {
      setLoopDetected(true);
    }
  }, [status, callbackUrl]);

  if (loopDetected) {
    return (
      <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4">
        <div className="w-full max-w-sm rounded-2xl border border-border bg-surface p-8 text-center shadow-[0_0_60px_-15px_var(--blue)]">
          <h1 className="text-xl font-semibold text-ink">Having trouble signing in</h1>
          <p className="mt-2 text-sm text-ink-secondary">
            You're signed in, but this page couldn't get you to {callbackUrl}. Try a full page reload, or come back
            in a moment.
          </p>
        </div>
      </div>
    );
  }

  if (status === "authenticated") return null;

  return (
    <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4">
      {/* W217: redesigned as a self-contained centered card, matching direct
          user feedback ("a focused window ... not a bare page") -- there's
          no live page left to blur behind it by the time this route
          renders (a real modal-over-the-app would mean replacing the
          middleware page-redirect with a client-side dialog, a bigger
          architecture change than asked for here), so this is the page
          itself redesigned to read as that focused window on its own. */}
      <div className="w-full max-w-sm rounded-2xl border border-border bg-surface p-8 text-center shadow-[0_0_60px_-15px_var(--blue)]">
        <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full border-2 border-accent">
          <Image src="/oddsey-logo.png" width={40} height={40} alt="" priority />
        </div>
        <h1 className="mt-5 text-xl font-semibold text-ink">Sign in to Oddsey</h1>
        <p className="mt-2 text-sm text-ink-secondary">Track your edges and log your bets across sessions.</p>
        {error && (
          <p className="mt-4 text-sm text-serious">
            {ERROR_MESSAGES[error] ?? "Sign-in failed. Please try again."}
          </p>
        )}
        <button
          type="button"
          onClick={() => signIn("google", { callbackUrl })}
          className="mt-6 flex w-full items-center justify-center gap-3 rounded-full bg-white px-4 py-3 text-sm font-semibold text-black shadow-lg transition hover:bg-white/90"
        >
          <GoogleIcon />
          Continue with Google
        </button>
        <p className="mt-4 text-xs text-ink-secondary">
          By continuing you agree to Oddsey&apos;s <span className="font-medium text-ink">Terms</span>
        </p>
      </div>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={null}>
      <LoginPageInner />
    </Suspense>
  );
}
