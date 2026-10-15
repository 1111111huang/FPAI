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
  // Truthiness, not `"x" in y` -- matches ServiceWorkerRegistration.tsx's
  // own feature check (and its test's `value: undefined` convention for
  // simulating an unsupported browser without deleting the property).
  if (!navigator.serviceWorker || !(window as Window & { PushManager?: unknown }).PushManager) return false;
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
