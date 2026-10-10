"use client";

import { useEffect } from "react";
import { createPortal } from "react-dom";

// Mirrors LogBetModal's dialog conventions (portal to document.body so
// clicks/cursor can't leak into whatever clickable row/card triggered it --
// see LogBetModal's W218 fix comment for why that matters -- plus
// Escape/backdrop-to-cancel) trimmed down for a plain two-button confirm, no
// form fields so no focus trap needed.
export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Delete",
  cancelLabel = "Cancel",
  confirmAriaLabel,
  cancelAriaLabel,
  danger = true,
  busy = false,
  error,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  confirmAriaLabel?: string;
  cancelAriaLabel?: string;
  danger?: boolean;
  busy?: boolean;
  error?: string | null;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  useEffect(() => {
    if (!open) return;
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape" && !busy) onCancel();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open, busy, onCancel]);

  if (!open) return null;

  return createPortal(
    <>
      <div
        className="fixed inset-0 z-40 bg-page/70 backdrop-blur-sm"
        onClick={() => {
          if (!busy) onCancel();
        }}
        aria-hidden="true"
      />
      <div
        role="alertdialog"
        aria-modal="true"
        aria-label={title}
        className="fixed left-1/2 top-1/2 z-50 w-full max-w-sm -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-border bg-surface p-6 shadow-2xl"
      >
        <h2 className="text-base font-semibold text-ink">{title}</h2>
        <p className="mt-2 text-sm text-ink-secondary">{message}</p>
        {error && <p className="mt-2 text-sm text-serious">{error}</p>}
        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            aria-label={cancelAriaLabel ?? cancelLabel}
            className="rounded-lg border border-border px-4 py-2 text-sm font-medium text-ink-secondary disabled:opacity-50"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={busy}
            aria-label={confirmAriaLabel ?? confirmLabel}
            className={`rounded-full px-4 py-2 text-sm font-semibold text-white transition disabled:opacity-50 ${
              danger ? "bg-serious hover:bg-serious/90" : "bg-accent hover:bg-accent/90"
            }`}
          >
            {busy ? "…" : confirmLabel}
          </button>
        </div>
      </div>
    </>,
    document.body
  );
}
