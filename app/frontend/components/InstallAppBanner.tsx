"use client";

import { useEffect, useState } from "react";

const DISMISSED_KEY = "oddsey-install-banner-dismissed";

type InstallPromptEvent = Event & { prompt: () => void };

function isIos(): boolean {
  return /iphone|ipad|ipod/i.test(window.navigator.userAgent);
}

function isStandalone(): boolean {
  return (
    (typeof window.matchMedia === "function" && window.matchMedia("(display-mode: standalone)").matches) ||
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
