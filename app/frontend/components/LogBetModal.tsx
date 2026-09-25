"use client";

import { useEffect, useRef, useState } from "react";
import { Plus, WarningCircle, X } from "@phosphor-icons/react";
import { TeamBadge, marketLabel } from "./MatchUI";

// W218: the one closed set of valid market/selection pairs -- duplicated
// from BetTracker.tsx's MARKET_SELECTIONS (that file imports this one, not
// the other way around, to avoid a circular import between the two
// components) is exactly the trap this plan avoids: exported here as the
// single source, BetTracker.tsx's copy is deleted in Task 3.
export const MARKET_SELECTIONS: Record<string, { value: string; label: string }[]> = {
  result_3way: [
    { value: "home", label: "Home" },
    { value: "draw", label: "Draw" },
    { value: "away", label: "Away" },
  ],
  btts: [
    { value: "yes", label: "Yes" },
    { value: "no", label: "No" },
  ],
  total_goals: [
    { value: "over_2.5", label: "Over 2.5" },
    { value: "under_2.5", label: "Under 2.5" },
  ],
  total_corners: [
    { value: "over_9.5", label: "Over 9.5" },
    { value: "under_9.5", label: "Under 9.5" },
  ],
};

export type LogBetFields = { market: string; selection: string; odds: number; stake: number };

export function LogBetModal({
  open,
  onClose,
  homeTeam,
  awayTeam,
  statusLabel,
  locked,
  market: initialMarket,
  selection: initialSelection,
  odds: initialOdds,
  onSubmit,
}: {
  open: boolean;
  onClose: () => void;
  homeTeam: string;
  awayTeam: string;
  statusLabel: string;
  // W218: true for the from-a-recommendation paths (MatchCard's quick-log,
  // ProbabilityRow's per-market picker) -- Market/Pick/Odds are already
  // decided by the recommendation being logged and shown as fixed text, not
  // controls. false for ManualBetForm's fixture-search path, where none of
  // the three are known yet.
  locked: boolean;
  market: string;
  selection: string;
  odds: number | null;
  onSubmit: (fields: LogBetFields) => Promise<void>;
}) {
  const [market, setMarket] = useState(initialMarket);
  const [selection, setSelection] = useState(initialSelection);
  const [odds, setOdds] = useState(initialOdds != null ? String(initialOdds) : "");
  const [stake, setStake] = useState("");
  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  // W218 code review follow-up: the Escape handler lives inside a
  // useEffect keyed on [open, onClose] (see below) -- it must NOT also key
  // on `saving`, since that would tear down and rebuild the whole effect
  // (including the body-scroll-lock and focus-trigger-capture logic) every
  // time `saving` toggles, incorrectly re-capturing document.activeElement
  // as something inside the dialog instead of the original trigger. A ref,
  // kept in sync separately, lets the handler read the live value without
  // that effect re-running.
  const savingRef = useRef(saving);
  useEffect(() => {
    savingRef.current = saving;
  }, [saving]);

  // W218 code review follow-up: this is the app's first true centered
  // dialog (role="dialog" aria-modal="true", not a side-drawer like
  // AppShell's mobile menu or MatchCard's insights rail, neither of which
  // claim the ARIA modal contract) -- Escape-to-close, a body-scroll lock,
  // a Tab focus trap, and returning focus to whatever triggered the modal
  // on close are basic requirements of that contract, not polish.
  useEffect(() => {
    if (!open) return;
    const triggerEl = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        // W218 code review follow-up: don't let Escape (or the backdrop
        // click below) close the modal mid-submit -- the in-flight
        // onSubmit() would then land in an unmounted component, silently
        // dropping a real success confirmation or a real error. Matches
        // the Cancel button's own `disabled={saving}` guard.
        if (!savingRef.current) onClose();
        return;
      }
      if (e.key !== "Tab" || !dialogRef.current) return;
      const focusable = dialogRef.current.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      if (triggerEl instanceof HTMLElement) triggerEl.focus();
    };
  }, [open, onClose]);

  if (!open) return null;

  const parsedOdds = parseFloat(odds);
  const parsedStake = parseFloat(stake);
  const returns = parsedOdds > 0 && parsedStake > 0 ? (parsedOdds * parsedStake).toFixed(2) : "0.00";

  async function handleConfirm() {
    // W218 code review follow-up: in locked mode, selection/odds come
    // straight from the recommendation candidate being logged -- already
    // real (a candidate is never listed with no odds or an empty
    // selection) and not user-editable here, so re-validating them would
    // be a dead end with no control to fix whatever it complained about.
    // Only stake -- the one thing locked mode actually lets you type --
    // needs checking there; editable mode still validates all three.
    const fieldsInvalid = locked
      ? false
      : !selection.trim() || !parsedOdds || parsedOdds <= 1;
    if (fieldsInvalid || !parsedStake || parsedStake <= 0) {
      setErrorMsg(
        locked ? "Enter a stake greater than 0." : "Fill in a pick, a valid odds (>1), and a stake (>0)."
      );
      return;
    }
    setSaving(true);
    setErrorMsg(null);
    try {
      await onSubmit({ market, selection, odds: parsedOdds, stake: parsedStake });
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Could not log bet.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <div
        className="fixed inset-0 z-40 bg-page/70 backdrop-blur-sm"
        onClick={() => {
          if (!saving) onClose();
        }}
        aria-hidden="true"
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label="Log bet"
        className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-border bg-surface p-6 shadow-2xl"
      >
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold text-ink">Log bet</h2>
            <p className="mt-1 text-sm text-ink-secondary">
              {homeTeam} v {awayTeam} · via The Odds API
            </p>
            {/* W109: the only other place this framing exists is /bets' page
                subtitle (BetTracker.tsx), invisible from this modal -- shared
                by MatchCard, ProbabilityRow, and ManualBetForm, so one line
                here covers all three click points at once. */}
            <p className="mt-1 text-xs text-ink-secondary">
              This logs a bet you've actually placed — not automatic hypothetical tracking.
            </p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close" className="text-ink-secondary hover:text-ink">
            <X size={20} />
          </button>
        </div>

        <div className="mt-4 flex items-center gap-3 rounded-xl border border-border bg-page/60 p-3">
          <span className="flex -space-x-2">
            <TeamBadge name={homeTeam} size="lg" />
            <TeamBadge name={awayTeam} size="lg" />
          </span>
          <div>
            <div className="text-base font-semibold text-ink">{homeTeam} v {awayTeam}</div>
            <div className="text-xs text-ink-secondary">{statusLabel}</div>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3">
          <div>
            <div className="mb-1 text-xs text-ink-secondary">Market</div>
            {locked ? (
              <div className="rounded-lg border border-border bg-page/60 px-3 py-2 text-sm font-semibold text-ink">
                {marketLabel(market).label}
              </div>
            ) : (
              <>
                <label htmlFor="log-bet-market" className="sr-only">Market</label>
                <select
                  id="log-bet-market"
                  value={market}
                  onChange={(e) => {
                    setMarket(e.target.value);
                    setSelection(""); // force a fresh, valid choice for the new market
                  }}
                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-ink outline-none focus:border-accent"
                >
                  {Object.keys(MARKET_SELECTIONS).map((m) => (
                    <option key={m} value={m}>{marketLabel(m).label}</option>
                  ))}
                </select>
              </>
            )}
          </div>
          <div>
            <div className="mb-1 text-xs text-ink-secondary">Pick</div>
            {locked ? (
              <div className="rounded-lg border border-border bg-page/60 px-3 py-2 text-sm font-semibold text-ink">
                {MARKET_SELECTIONS[market]?.find((o) => o.value === selection)?.label ?? selection}
              </div>
            ) : (
              <>
                <label htmlFor="log-bet-selection" className="sr-only">Pick</label>
                <select
                  id="log-bet-selection"
                  value={selection}
                  onChange={(e) => setSelection(e.target.value)}
                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-ink outline-none focus:border-accent"
                >
                  <option value="">Select…</option>
                  {MARKET_SELECTIONS[market]?.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </>
            )}
          </div>
          <div>
            <div className="mb-1 text-xs text-ink-secondary">Odds</div>
            {locked ? (
              <div className="rounded-lg border border-border bg-page/60 px-3 py-2 text-sm font-semibold text-ink">
                {parsedOdds > 0 ? parsedOdds.toFixed(2) : "—"}
              </div>
            ) : (
              <>
                <label htmlFor="log-bet-odds" className="sr-only">Odds</label>
                <input
                  id="log-bet-odds"
                  value={odds}
                  onChange={(e) => setOdds(e.target.value)}
                  placeholder="0.00"
                  inputMode="decimal"
                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-ink outline-none focus:border-accent"
                />
              </>
            )}
          </div>
          <div>
            <label htmlFor="log-bet-stake" className="mb-1 block text-xs text-ink-secondary">Stake</label>
            <div className="relative">
              <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-sm text-ink-secondary">$</span>
              <input
                id="log-bet-stake"
                type="number"
                min="0"
                step="0.01"
                value={stake}
                onChange={(e) => setStake(e.target.value)}
                placeholder="0.00"
                autoFocus
                className="w-full rounded-lg border border-border bg-surface py-2 pl-6 pr-3 text-sm text-ink outline-none focus:border-accent"
              />
            </div>
          </div>
        </div>

        <p className="mt-3 text-sm text-ink-secondary">
          Returns <span className="font-mono font-medium text-good">${returns}</span> if it hits
        </p>

        {errorMsg && (
          <p className="mt-2 flex items-center gap-1.5 text-xs text-serious">
            <WarningCircle weight="fill" size={13} />
            {errorMsg}
          </p>
        )}

        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={saving}
            className="rounded-lg border border-border px-4 py-2 text-sm font-medium text-ink-secondary disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            disabled={saving}
            className="flex items-center gap-1.5 rounded-full bg-accent px-4 py-2 text-sm font-semibold text-white transition hover:bg-accent/90 disabled:opacity-50"
          >
            <Plus size={14} weight="bold" />
            {saving ? "Logging…" : "Confirm bet"}
          </button>
        </div>
      </div>
    </>
  );
}
