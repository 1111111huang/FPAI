"use client";

/** W12: bet tracker page -- lists logged bets and provides the manual
 * logging path (D3a). match_id is always a resolved Fixture from the
 * search results below, never free-typed team names, so auto-settlement
 * (W13) can still find the real fixture later. */

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { MagnifyingGlass, WarningCircle } from "@phosphor-icons/react";

import { ApiError, deleteBet, getBetStats, getBets, getFixtures, logBetManual, settleOpenBets } from "@/lib/api";
import type { Bet, BetStats, Fixture } from "@/lib/types";
import { useSandboxAsOf } from "@/lib/useSandboxAsOf";
import { AppShell } from "./AppShell";
import { addDays, dateString, ErrorState, TeamBadge, marketLabel } from "./MatchUI";

function formatDate(iso: string): string {
  return iso.slice(0, 10);
}

// W210 follow-up: the only markets/selections settlement.py's market_correct()
// can ever resolve (src/agent/market_resolution.py, RESOLVABLE_MARKETS).
// A freeform text field let a user log a bet with a market/selection that
// could never programmatically settle -- it would just sit "open" forever
// with no error anywhere. Constraining to exactly these values closes that.
const MARKET_SELECTIONS: Record<string, { value: string; label: string }[]> = {
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

function ManualBetForm({ onLogged, onSessionExpired }: { onLogged: () => void; onSessionExpired: () => void }) {
  const { asOf, sandboxMode } = useSandboxAsOf();
  const [query, setQuery] = useState("");
  const [fixtures, setFixtures] = useState<Fixture[] | null>(null);
  const [selected, setSelected] = useState<Fixture | null>(null);
  const [market, setMarket] = useState("result_3way");
  const [selection, setSelection] = useState("");
  const [odds, setOdds] = useState("");
  const [stake, setStake] = useState("");
  const [status, setStatus] = useState<"idle" | "saving" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");
  // W52: distinct from `errorMsg` (submit-path errors) -- this tracks the
  // fixture *search* fetch itself failing (e.g. the football-data.org
  // rate-limit degrading to a backend 503). Previously
  // `.catch(() => setFixtures([]))` silently swallowed this into a plain
  // empty fixture list, indistinguishable from a genuine "no matches"
  // result -- same ApiError-vs-generic-fallback pattern MatchExplorerPage
  // already uses (MatchUI.tsx).
  const [fixturesError, setFixturesError] = useState<string | null>(null);
  // W52 code review follow-up: matches DashboardPage/MatchExplorerPage's
  // established retry pattern (MatchUI.tsx) -- bumped by the Retry button
  // to force a fresh fetch through this same effect, rather than calling
  // getFixtures() imperatively from outside it (which would have no
  // cancellation guard against a stale in-flight request from a previous
  // run).
  const [retryTick, setRetryTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    // W211: direct user feedback -- this search was forward-only, so a bet
    // couldn't be manually logged against a match that already kicked off.
    // Reuses MatchExplorerPage's exact window (30 days back, 90 forward)
    // and its addDays/dateString helpers, which already branch correctly
    // on sandboxMode -- the previous unconditional `setUTCDate` here was
    // only correct in sandbox mode (see MatchExplorerPage's own comment).
    const from = dateString(addDays(asOf, -30, sandboxMode), sandboxMode);
    const to = dateString(addDays(asOf, 90, sandboxMode), sandboxMode);
    setFixturesError(null);
    getFixtures(from, to)
      .then((result) => {
        if (!cancelled) setFixtures(result);
      })
      .catch((err) => {
        if (cancelled) return;
        setFixtures([]);
        if (err instanceof ApiError && err.status === 401) {
          onSessionExpired();
          return;
        }
        setFixturesError(err instanceof ApiError ? err.message : "Could not load fixtures.");
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [asOf, retryTick]);

  const results = useMemo(() => {
    if (!fixtures) return [];
    const q = query.trim().toLowerCase();
    if (q.length === 0) return [];
    return fixtures
      .filter((f) => f.home_team.toLowerCase().includes(q) || f.away_team.toLowerCase().includes(q))
      .slice(0, 8);
  }, [fixtures, query]);

  async function submit() {
    if (!selected) return;
    const parsedOdds = parseFloat(odds);
    const parsedStake = parseFloat(stake);
    if (!selection.trim() || !parsedOdds || parsedOdds <= 1 || !parsedStake || parsedStake <= 0) {
      setStatus("error");
      setErrorMsg("Fill in selection, a valid odds (>1), and a stake (>0).");
      return;
    }
    setStatus("saving");
    try {
      await logBetManual({
        match_id: selected.match_id, date: formatDate(selected.utc_date),
        home_team: selected.home_team, away_team: selected.away_team,
        market, selection, odds: parsedOdds, stake: parsedStake,
      });
      setSelected(null);
      setQuery("");
      setSelection("");
      setOdds("");
      setStake("");
      setStatus("idle");
      onLogged();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setStatus("idle");
        onSessionExpired();
        return;
      }
      setStatus("error");
      setErrorMsg(err instanceof ApiError ? err.message : "Could not log bet.");
    }
  }

  return (
    <div className="rounded-lg border border-border p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">Log a bet manually</h2>

      {!selected ? (
        <>
          <div className="relative mt-3">
            <MagnifyingGlass size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search a real fixture by team name…"
              className="w-full rounded-lg border border-border bg-surface py-2 pl-9 pr-3 text-sm text-ink outline-none placeholder:text-muted focus:border-accent"
            />
          </div>
          {fixturesError && (
            <div className="mt-2">
              <ErrorState message={fixturesError} onRetry={() => setRetryTick((t) => t + 1)} />
            </div>
          )}
          {query.trim().length > 0 && !fixturesError && (
            results.length > 0 ? (
              <div className="mt-2 flex flex-col gap-1.5">
                {results.map((f) => (
                  <button
                    key={f.match_id}
                    type="button"
                    onClick={() => setSelected(f)}
                    className="flex items-center gap-2 rounded-lg border border-border p-2 text-left text-sm text-ink hover:border-border-strong"
                  >
                    <TeamBadge name={f.home_team} />
                    {f.home_team} v {f.away_team}
                    <TeamBadge name={f.away_team} />
                    <span className="ml-auto text-xs text-ink-secondary">{formatDate(f.utc_date)}</span>
                  </button>
                ))}
              </div>
            ) : (
              <p className="mt-2 text-sm text-ink-secondary">No matching fixtures.</p>
            )
          )}
        </>
      ) : (
        <div className="mt-3 flex flex-col gap-3">
          <div className="flex items-center justify-between text-sm text-ink">
            <span>
              {selected.home_team} v {selected.away_team} · {formatDate(selected.utc_date)}
            </span>
            <button type="button" onClick={() => setSelected(null)} className="text-xs text-accent">
              Change fixture
            </button>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div>
              <label htmlFor="manual-bet-market" className="sr-only">Market</label>
              <select
                id="manual-bet-market"
                value={market}
                onChange={(e) => {
                  setMarket(e.target.value);
                  setSelection(""); // force a fresh, valid choice for the new market
                }}
                className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-ink outline-none focus:border-accent"
              >
                {Object.keys(MARKET_SELECTIONS).map((m) => (
                  <option key={m} value={m}>{marketLabel(m).label}</option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="manual-bet-selection" className="sr-only">Selection</label>
              <select
                id="manual-bet-selection"
                value={selection}
                onChange={(e) => setSelection(e.target.value)}
                className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-ink outline-none focus:border-accent"
              >
                <option value="">Select…</option>
                {MARKET_SELECTIONS[market]?.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="manual-bet-odds" className="sr-only">Odds</label>
              <input
                id="manual-bet-odds"
                value={odds}
                onChange={(e) => setOdds(e.target.value)}
                placeholder="Odds"
                inputMode="decimal"
                className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-ink outline-none focus:border-accent"
              />
            </div>
            <div>
              <label htmlFor="manual-bet-stake" className="sr-only">Stake</label>
              <input
                id="manual-bet-stake"
                value={stake}
                onChange={(e) => setStake(e.target.value)}
                placeholder="Stake"
                inputMode="decimal"
                className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-ink outline-none focus:border-accent"
              />
            </div>
          </div>
          {status === "error" && (
            <p className="flex items-center gap-1.5 text-xs text-serious">
              <WarningCircle weight="fill" size={13} />
              {errorMsg}
            </p>
          )}
          <button
            type="button"
            onClick={submit}
            disabled={status === "saving"}
            className="self-start rounded-md border border-accent px-3 py-1.5 text-sm font-medium text-accent disabled:opacity-50"
          >
            {status === "saving" ? "Logging…" : "Log bet"}
          </button>
        </div>
      )}
    </div>
  );
}

function BetRow({
  bet,
  onDeleted,
  onSessionExpired,
}: {
  bet: Bet;
  onDeleted: () => void;
  onSessionExpired: () => void;
}) {
  const outcomeColor = bet.outcome === "won" ? "text-good" : bet.outcome === "lost" ? "text-serious" : "text-muted";
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  // Matches ManualBetForm's `cancelled` convention -- this row can be
  // removed from the list (a delete elsewhere, a reload) while its own
  // delete is still in flight; a ref (not state) survives the finally
  // block running after unmount without itself triggering a re-render.
  const mountedRef = useRef(true);
  useEffect(() => {
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const label = `${bet.home_team} v ${bet.away_team}`;

  async function confirmDelete() {
    setDeleting(true);
    setDeleteError(null);
    try {
      await deleteBet(bet.id);
      onDeleted();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        onSessionExpired();
      } else {
        setDeleteError(err instanceof ApiError ? err.message : "Could not delete bet.");
      }
    } finally {
      if (mountedRef.current) setDeleting(false);
    }
  }

  return (
    <div className="grid grid-cols-[1fr_auto_auto_auto_auto_auto] items-center gap-4 border-b border-border py-3 text-sm last:border-b-0">
      <span className="truncate text-ink">
        {bet.home_team} v {bet.away_team}
        <span className="ml-2 text-xs text-ink-secondary">
          {bet.market} · {bet.selection}
        </span>
      </span>
      <span className="text-right font-mono text-ink-secondary">{bet.odds.toFixed(2)}</span>
      <span className="text-right font-mono text-ink-secondary">{bet.stake.toFixed(2)}</span>
      <span className="text-right font-mono text-ink">
        {bet.profit_loss !== null ? bet.profit_loss.toFixed(2) : "—"}
      </span>
      <span className={`justify-self-end uppercase text-xs font-medium ${outcomeColor}`}>{bet.outcome}</span>
      <span className="justify-self-end text-xs">
        {confirming ? (
          <span className="flex items-center gap-1.5">
            <span className="text-ink-secondary">Delete this bet?</span>
            <button
              type="button"
              onClick={confirmDelete}
              disabled={deleting}
              aria-label={`Confirm delete: ${label}`}
              className="font-medium text-serious disabled:opacity-50"
            >
              {deleting ? "…" : "Yes"}
            </button>
            <button
              type="button"
              onClick={() => {
                setConfirming(false);
                setDeleteError(null);
              }}
              disabled={deleting}
              aria-label={`Cancel delete: ${label}`}
              className="text-ink-secondary disabled:opacity-50"
            >
              Cancel
            </button>
            {deleteError && <span className="text-serious">{deleteError}</span>}
          </span>
        ) : (
          <button
            type="button"
            onClick={() => setConfirming(true)}
            aria-label={`Delete bet: ${label}`}
            className="text-ink-secondary hover:text-serious"
          >
            Delete
          </button>
        )}
      </span>
    </div>
  );
}

function StatBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border p-3">
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1 font-mono text-lg text-ink">{value}</div>
    </div>
  );
}

function StatsBar({ stats }: { stats: BetStats }) {
  const roiColor = stats.roi > 0 ? "text-good" : stats.roi < 0 ? "text-serious" : "text-ink";
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <StatBox label="Bankroll" value={stats.current_bankroll.toFixed(2)} />
      <StatBox label="ROI" value={`${(stats.roi * 100).toFixed(1)}%`} />
      <StatBox label="Hit rate" value={stats.bets_settled > 0 ? `${(stats.hit_rate * 100).toFixed(0)}%` : "—"} />
      <StatBox label="Max drawdown" value={`${(stats.max_drawdown * 100).toFixed(1)}%`} />
      <div className="col-span-2 text-xs text-ink-secondary sm:col-span-4">
        {stats.bets_settled} settled ({stats.bets_won} won) · {stats.bets_open} open ·{" "}
        <span className={roiColor}>{stats.total_profit >= 0 ? "+" : ""}{stats.total_profit.toFixed(2)} profit</span>
      </div>
    </div>
  );
}

export function BetTrackerPage() {
  const [bets, setBets] = useState<Bet[] | null>(null);
  const [stats, setStats] = useState<BetStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [settling, setSettling] = useState(false);
  const [settleMsg, setSettleMsg] = useState<string | null>(null);

  const [needsAuth, setNeedsAuth] = useState(false);
  // W210 follow-up: a non-401 stats failure (e.g. a 500) previously tracked
  // nothing -- stats stayed null forever with no signal it had failed, so
  // the "Loading…" placeholder below showed forever too. A 401 already has
  // its own signal (needsAuth, set via the bets side); this covers every
  // other rejection.
  const [statsFailed, setStatsFailed] = useState(false);

  async function load() {
    setNeedsAuth(false);
    setError(null);
    setStatsFailed(false);
    const [betsResult, statsResult] = await Promise.allSettled([getBets(), getBetStats()]);

    if (betsResult.status === "fulfilled") {
      setBets(betsResult.value);
    } else if (betsResult.reason instanceof ApiError && betsResult.reason.status === 401) {
      // Deliberately leaves a prior successful load's `bets` list in place --
      // it reads as "last known good data" under the banner rather than
      // flashing to empty, and a real reload after sign-in replaces it anyway.
      setNeedsAuth(true);
    } else {
      setError(betsResult.reason instanceof ApiError ? betsResult.reason.message : "Could not load bets.");
    }

    if (statsResult.status === "fulfilled") {
      setStats(statsResult.value);
    } else {
      // A stats-only failure (401 included -- already surfaced above via
      // betsResult/needsAuth) intentionally doesn't block the bets list from
      // showing. statsFailed just retires the loading placeholder below;
      // there's no separate error message for it, matching this section's
      // existing "degrade quietly" behavior.
      setStatsFailed(true);
    }
  }

  async function handleSettle() {
    setSettling(true);
    setSettleMsg(null);
    try {
      const settled = await settleOpenBets();
      setSettleMsg(
        settled.length === 0
          ? "No open bets have a finished, scorable result yet."
          : `Settled ${settled.length} bet${settled.length === 1 ? "" : "s"}.`
      );
      await load();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setNeedsAuth(true);
      } else {
        setSettleMsg(err instanceof ApiError ? err.message : "Could not settle open bets.");
      }
    } finally {
      setSettling(false);
    }
  }

  useEffect(() => {
    // W214: direct user feedback -- settlement should be automatic, the
    // same way a match's completed status just appears whenever the page
    // is loaded, not behind a manual action. A best-effort settle attempt
    // right before the normal load, silent unless it hits a 401 (load()
    // below still runs either way and surfaces its own errors normally --
    // this never blocks the page on a settlement failure). The manual
    // "Settle open bets" button stays for a re-check without leaving the
    // page (a match can finish while already viewing it, unlike this
    // mount-time check).
    async function settleThenLoad() {
      try {
        await settleOpenBets();
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) setNeedsAuth(true);
      }
      await load();
    }
    settleThenLoad();
  }, []);

  return (
    <AppShell active="bets">
      <h1 className="text-xl font-semibold tracking-tight text-ink">Bet Tracker</h1>
      <p className="mt-1 text-sm text-ink-secondary">Bets you've actually placed -- not automatic hypothetical tracking.</p>

      {needsAuth && (
        <p className="mt-4 text-sm text-ink-secondary">
          Your session expired.{" "}
          <Link href="/login?callbackUrl=%2Fbets" className="font-medium text-accent">
            Sign in again
          </Link>{" "}
          to see your bets.
        </p>
      )}

      <div className="mt-6">
        {stats ? <StatsBar stats={stats} /> : !needsAuth && !error && !statsFailed && <p className="text-sm text-ink-secondary">Loading…</p>}
      </div>

      <div className="mt-6">
        <ManualBetForm onLogged={load} onSessionExpired={() => setNeedsAuth(true)} />
      </div>

      <div className="mt-8">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">Logged bets</h2>
          <button
            type="button"
            onClick={handleSettle}
            disabled={settling}
            className="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-ink-secondary hover:border-border-strong disabled:opacity-50"
          >
            {settling ? "Checking results…" : "Settle open bets"}
          </button>
        </div>
        {settleMsg && <p className="mt-2 text-xs text-ink-secondary">{settleMsg}</p>}
        {error && <p className="mt-2 text-sm text-serious">{error}</p>}
        {!error && !needsAuth && bets === null && <p className="mt-2 text-sm text-ink-secondary">Loading…</p>}
        {!error && bets && bets.length === 0 && (
          <p className="mt-2 text-sm text-ink-secondary">No bets logged yet.</p>
        )}
        {!error && bets && bets.length > 0 && (
          <div className="mt-2">
            <div className="grid grid-cols-[1fr_auto_auto_auto_auto_auto] gap-4 border-b border-border pb-1.5 text-xs font-medium uppercase tracking-wide text-muted">
              <span>Match</span>
              <span className="text-right">Odds</span>
              <span className="text-right">Stake</span>
              <span className="text-right">P&amp;L</span>
              <span className="text-right">Outcome</span>
              <span />
            </div>
            {bets.map((bet) => (
              <BetRow key={bet.id} bet={bet} onDeleted={load} onSessionExpired={() => setNeedsAuth(true)} />
            ))}
          </div>
        )}
      </div>
    </AppShell>
  );
}
