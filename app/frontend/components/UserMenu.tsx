"use client";

// W210 follow-up: the one piece of session UI this app had zero of --
// before this, a signed-in user had no way to see which account they were
// using or to sign out short of manually hitting /api/auth/signout. Lives
// in AppShell so it's present on every page, signed-in or not.
import Link from "next/link";
import { signOut, useSession } from "next-auth/react";

export function UserMenu() {
  const { data: session, status } = useSession();

  if (status === "loading") return null;

  if (status === "unauthenticated" || !session?.user?.email) {
    return (
      <Link href="/login" className="text-xs font-medium text-accent">
        Sign in
      </Link>
    );
  }

  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="truncate text-ink-secondary" title={session.user.email}>
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
  );
}
