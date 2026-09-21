import type {
  AgentPerformanceDashboard,
  Bet,
  BetStats,
  Fixture,
  MatchRecommendationOut,
  SandboxStatus,
  StatusResponse,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

// W97: only set once the app is deployed somewhere public -- mirrors the
// backend's APP_ACCESS_TOKEN (see app/backend/main.py's RequireAppTokenMiddleware),
// which itself only enforces the check when that env var is set, so local
// dev is completely unaffected either way.
const APP_TOKEN = process.env.NEXT_PUBLIC_APP_ACCESS_TOKEN;

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

/** Every request goes through here so the shared-secret header (when
 * configured) is attached exactly once, instead of repeated at each of the
 * call sites below. */
function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  if (APP_TOKEN) headers.set("X-App-Token", APP_TOKEN);
  return fetch(`${API_BASE}${path}`, { ...init, headers });
}

// Direct user report: switching between pages feels laggy. Every page mount
// re-fetches fixtures/recommendations/status from scratch with no reuse --
// AppShell, DashboardPage and MatchExplorerPage each call getFixtures() on
// their own, and navigating back to a page you just left re-fetches
// everything again. A short-TTL in-memory cache, keyed by request path, lets
// a page revisited within a few seconds render instantly from the last
// response instead of blocking on a fresh round trip -- 15s balances that
// against odds/predictions actually moving (a T-30 refresh or regen), and
// is short enough that no page ever shows meaningfully stale data.
const READ_CACHE_TTL_MS = 15_000;
const readCache = new Map<string, { data: unknown; expires: number }>();

async function cachedGet<T>(path: string, errorMessage: string): Promise<T> {
  const cached = readCache.get(path);
  if (cached && cached.expires > Date.now()) return cached.data as T;

  const response = await apiFetch(path);
  if (!response.ok) {
    throw new ApiError(`${errorMessage} (${response.status})`, response.status);
  }
  const data = (await response.json()) as T;
  readCache.set(path, { data, expires: Date.now() + READ_CACHE_TTL_MS });
  return data;
}

export async function getFixtures(dateFrom?: string, dateTo?: string): Promise<Fixture[]> {
  const params = new URLSearchParams();
  if (dateFrom) params.set("date_from", dateFrom);
  if (dateTo) params.set("date_to", dateTo);
  const query = params.toString();

  return cachedGet<Fixture[]>(`/api/fixtures${query ? `?${query}` : ""}`, "Failed to load fixtures");
}

export type RecommendationRequestBody = {
  home_team: string;
  away_team: string;
  date: string;
  league?: string;
  match_id?: string;
  odds?: { home: number; draw: number; away: number };
};

/** The explicit "regenerate now" call (W11) -- always invokes the real agent. */
export async function generateRecommendation(
  body: RecommendationRequestBody
): Promise<MatchRecommendationOut> {
  const response = await apiFetch(`/api/recommendations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new ApiError(`Failed to generate recommendation (${response.status})`, response.status);
  }
  // A regenerate-now must be reflected immediately, not shadowed by whatever
  // getCachedRecommendation() cached up to READ_CACHE_TTL_MS ago for this
  // same match/date.
  if (body.match_id) {
    readCache.delete(`/api/recommendations/${encodeURIComponent(body.match_id)}?date=${encodeURIComponent(body.date)}`);
  }
  return response.json();
}

/** Cache-only read (W11) -- never triggers a live agent call. Returns null on
 * a 404 (nothing generated yet for this match/date), throws on any other
 * failure. Doesn't go through cachedGet(): a 404 is a valid "nothing yet"
 * result here, not an error to throw, so it needs its own read-cache
 * handling (still the same shared `readCache` map/TTL) rather than
 * cachedGet()'s throw-on-!ok contract. */
export async function getCachedRecommendation(
  matchId: string,
  date: string
): Promise<MatchRecommendationOut | null> {
  const path = `/api/recommendations/${encodeURIComponent(matchId)}?date=${encodeURIComponent(date)}`;
  const cached = readCache.get(path);
  if (cached && cached.expires > Date.now()) return cached.data as MatchRecommendationOut | null;

  const response = await apiFetch(path);
  if (response.status === 404) {
    readCache.set(path, { data: null, expires: Date.now() + READ_CACHE_TTL_MS });
    return null;
  }
  if (!response.ok) {
    throw new ApiError(`Failed to load cached recommendation (${response.status})`, response.status);
  }
  const data: MatchRecommendationOut = await response.json();
  readCache.set(path, { data, expires: Date.now() + READ_CACHE_TTL_MS });
  return data;
}

/** W12: logs a bet with every field but stake locked to the given
 * recommendation snapshot. */
export async function logBetFromRecommendation(body: {
  match_id: string;
  recommendation: MatchRecommendationOut;
  market: string;
  selection: string;
  stake: number;
}): Promise<Bet> {
  const response = await fetch(`/api/bets/from-recommendation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new ApiError(`Failed to log bet (${response.status})`, response.status);
  }
  return response.json();
}

/** W12: manual entry -- match_id must reference a real, resolved fixture. */
export async function logBetManual(body: {
  match_id: string;
  date: string;
  home_team: string;
  away_team: string;
  market: string;
  selection: string;
  odds: number;
  stake: number;
}): Promise<Bet> {
  const response = await fetch(`/api/bets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new ApiError(`Failed to log bet (${response.status})`, response.status);
  }
  return response.json();
}

export async function getBets(): Promise<Bet[]> {
  const response = await fetch(`/api/bets`);
  if (!response.ok) {
    throw new ApiError(`Failed to load bets (${response.status})`, response.status);
  }
  return response.json();
}

/** W215: removes a logged bet -- 404s (via ApiError) if it isn't the
 * caller's own bet or doesn't exist. */
export async function deleteBet(id: number): Promise<void> {
  const response = await fetch(`/api/bets/${id}`, { method: "DELETE" });
  if (!response.ok) {
    throw new ApiError(`Failed to delete bet (${response.status})`, response.status);
  }
}

/** W216: partial edit -- only the fields provided are changed, the rest of
 * the bet (including its outcome/profit_loss recomputation if already
 * settled) is handled server-side. */
export async function updateBet(
  id: number,
  fields: { market?: string; selection?: string; odds?: number; stake?: number }
): Promise<Bet> {
  const response = await fetch(`/api/bets/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(fields),
  });
  if (!response.ok) {
    throw new ApiError(`Failed to update bet (${response.status})`, response.status);
  }
  return response.json();
}

/** W13: on-demand settlement trigger -- no scheduler (W08/W09 deferred).
 * Returns the bets that were actually settled by this call (won/lost);
 * corners bets and not-yet-finished matches are never included.
 *
 * `blocking` (default true, matching every prior caller unchanged): false
 * -- used only by BetTracker.tsx's automatic settle-on-page-load (W214) --
 * tells the backend to skip (not wait through) any competition/date whose
 * football-data.org rate-limit budget is exhausted right now, rather than
 * blocking this call for up to a minute. Found live, 2026-09-15: the
 * automatic on-load settle attempt was blocking the whole Bets page behind
 * "Loading…" for that long once W213's ResultsCache fix made same-day
 * results always hit the live API. The explicit "Settle open bets" button
 * keeps the default -- a real user-triggered wait is fine there. */
export async function settleOpenBets(options?: { blocking?: boolean }): Promise<Bet[]> {
  const query = options?.blocking === false ? "?blocking=false" : "";
  const response = await fetch(`/api/bets/settle-open${query}`, { method: "POST" });
  if (!response.ok) {
    throw new ApiError(`Failed to settle open bets (${response.status})`, response.status);
  }
  return response.json();
}

/** W14: ROI/hit-rate/bankroll summary, computed only over settled bets. */
export async function getBetStats(): Promise<BetStats> {
  const response = await fetch(`/api/bets/stats`);
  if (!response.ok) {
    throw new ApiError(`Failed to load bet stats (${response.status})`, response.status);
  }
  return response.json();
}

/** W17: data staleness + current model selections. */
export async function getStatus(): Promise<StatusResponse> {
  return cachedGet<StatusResponse>(`/api/status`, "Failed to load status");
}

/** W27: introspects whether sandbox mode is active and, if so, the as-of date. */
export async function getSandboxStatus(): Promise<SandboxStatus> {
  return cachedGet<SandboxStatus>(`/api/sandbox/status`, "Failed to load sandbox status");
}

/** W172: local-only diagnostics dashboard -- not called from any nav-linked
 * page, only app/agent-performance/page.tsx (W174, itself unlinked). */
export async function getAgentPerformanceDashboard(
  days?: number,
  topN?: number
): Promise<AgentPerformanceDashboard> {
  const params = new URLSearchParams();
  if (days !== undefined) params.set("days", String(days));
  if (topN !== undefined) params.set("top_n", String(topN));
  const query = params.toString();
  const response = await apiFetch(`/api/recommendations/performance-dashboard${query ? `?${query}` : ""}`);
  if (!response.ok) {
    throw new ApiError(`Failed to load agent performance dashboard (${response.status})`, response.status);
  }
  return response.json();
}
