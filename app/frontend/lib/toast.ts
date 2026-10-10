// Plain-DOM toast, deliberately not a React component: LogBetModal unmounts
// itself (parent sets open=false) the instant onSubmit resolves, so a toast
// rendered inside it would vanish before anyone saw it. A node appended
// straight to document.body survives that unmount.
let activeToast: HTMLDivElement | null = null;

export function showToast(message: string, durationMs = 2500) {
  if (typeof document === "undefined") return;
  activeToast?.remove();

  const el = document.createElement("div");
  el.textContent = message;
  el.className =
    "fixed top-4 left-1/2 z-[60] -translate-x-1/2 rounded-xl border border-border bg-surface px-4 py-2.5 text-sm font-medium text-ink shadow-2xl transition-opacity duration-300";
  el.style.opacity = "0";
  document.body.appendChild(el);
  activeToast = el;

  requestAnimationFrame(() => {
    el.style.opacity = "1";
  });

  setTimeout(() => {
    el.style.opacity = "0";
    setTimeout(() => {
      el.remove();
      if (activeToast === el) activeToast = null;
    }, 300);
  }, durationMs);
}
