"use client";

import { Suspense, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { signIn, useSession } from "next-auth/react";

const ERROR_MESSAGES: Record<string, string> = {
  AccessDenied: "That Google account is not authorized for this app. Ask the owner to add it to the allowlist.",
};

function LoginPageInner() {
  const { status } = useSession();
  const router = useRouter();
  const searchParams = useSearchParams();

  const callbackUrl = searchParams.get("callbackUrl") ?? "/bets";
  const error = searchParams.get("error");

  useEffect(() => {
    if (status === "authenticated") router.push("/bets");
  }, [status, router]);

  if (status === "authenticated") return null;

  return (
    <div className="flex flex-col items-center mt-16 gap-4">
      <h1 className="text-lg font-medium text-ink">Sign in</h1>
      {error && (
        <p className="max-w-sm text-center text-sm text-serious">
          {ERROR_MESSAGES[error] ?? "Sign-in failed. Please try again."}
        </p>
      )}
      <button
        type="button"
        onClick={() => signIn("google", { callbackUrl })}
        className="rounded-md border border-accent px-4 py-2 text-sm font-medium text-accent hover:bg-accent/10"
      >
        Sign in with Google
      </button>
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
