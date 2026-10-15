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
