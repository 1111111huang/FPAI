"use client";

/** W12: bet tracker page -- lists logged bets and provides the manual
 * logging path (D3a). match_id is always a resolved Fixture from the
 * search results below, never free-typed team names, so auto-settlement
 * (W13) can still find the real fixture later. */

import { useEffect, useMemo, useState } from "react";
import { MagnifyingGlass, WarningCircle } from "@phosphor-icons/react";

import { ApiError, getBetStats, getBets, getFixtures, logBetManual, settleOpenBets } from "@/lib/api";
import type { Bet, BetStats, Fixture } from "@/lib/types";
import { DraftNav, TeamBadge } from "./MatchUI";

function formatDate(iso: string): string {
  return iso.slice(0, 10);
}

function ManualBetForm({ onLogged }: { onLogged: () => void }) {
  const [query, setQuery] = useState("");
  const [fixtures, setFixtures] = useState<Fixture[] | null>(null);
  const [selected, setSelected] = useState<Fixture | null>(null);
  const [market, setMarket] = useState("result_3way");
  const [selection, setSelection] = useState("");
  const [odds, setOdds] = useState("");
  const [stake, setStake] = useState("");
  const [status, setStatus] = useState<"idle" | "saving" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    const from = new Date();
    const to = new Date();
    to.setDate(to.getDate() + 90);
    getFixtures(from.toISOString().slice(0, 10), to.toISOString().slice(0, 10))
      .then(setFixtures)
      .catch(() => setFixtures([]));
  }, []);

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
          {results.length > 0 && (
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
            <input
              value={market}
              onChange={(e) => setMarket(e.target.value)}
              placeholder="Market"
              className="rounded border border-border bg-surface px-2 py-1.5 text-sm text-ink outline-none focus:border-accent"
            />
            <input
              value={selection}
              onChange={(e) => setSelection(e.target.value)}
              placeholder="Selection"
              className="rounded border border-border bg-surface px-2 py-1.5 text-sm text-ink outline-none focus:border-accent"
            />
            <input
              value={odds}
              onChange={(e) => setOdds(e.target.value)}
              placeholder="Odds"
              inputMode="decimal"
              className="rounded border border-border bg-surface px-2 py-1.5 text-sm text-ink outline-none focus:border-accent"
            />
            <input
              value={stake}
              onChange={(e) => setStake(e.target.value)}
              placeholder="Stake"
              inputMode="decimal"
              className="rounded border border-border bg-surface px-2 py-1.5 text-sm text-ink outline-none focus:border-accent"
            />
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

function BetRow({ bet }: { bet: Bet }) {
  const outcomeColor = bet.outcome === "won" ? "text-good" : bet.outcome === "lost" ? "text-serious" : "text-muted";
  return (
    <div className="grid grid-cols-[1fr_auto_auto_auto_auto] items-center gap-4 border-b border-border py-3 text-sm last:border-b-0">
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

  async function load() {
    try {
      const [betList, betStats] = await Promise.all([getBets(), getBetStats()]);
      setBets(betList);
      setStats(betStats);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load bets.");
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
      setSettleMsg(err instanceof ApiError ? err.message : "Could not settle open bets.");
    } finally {
      setSettling(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <main className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      <DraftNav active="bets" />

      <h1 className="text-xl font-semibold tracking-tight text-ink">Bet Tracker</h1>
      <p className="mt-1 text-sm text-ink-secondary">Bets you've actually placed -- not automatic hypothetical tracking.</p>

      {stats && (
        <div className="mt-6">
          <StatsBar stats={stats} />
        </div>
      )}

      <div className="mt-6">
        <ManualBetForm onLogged={load} />
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
        {!error && bets === null && <p className="mt-2 text-sm text-ink-secondary">Loading…</p>}
        {!error && bets && bets.length === 0 && (
          <p className="mt-2 text-sm text-ink-secondary">No bets logged yet.</p>
        )}
        {!error && bets && bets.length > 0 && (
          <div className="mt-2">
            {bets.map((bet) => (
              <BetRow key={bet.id} bet={bet} />
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
