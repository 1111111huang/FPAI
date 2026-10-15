"use client";

import { useEffect } from "react";

export function ServiceWorkerRegistration() {
  useEffect(() => {
    if (!navigator.serviceWorker) return;
    navigator.serviceWorker.register("/sw.js").catch(() => {
      // ponytail: best-effort -- an unsupported/blocked browser just never
      // gets install/offline/push; nothing else in the app depends on it.
    });
  }, []);
  return null;
}
