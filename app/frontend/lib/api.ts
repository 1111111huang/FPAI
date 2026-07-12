import type { Fixture, MatchRecommendationOut } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

export async function getFixtures(dateFrom?: string, dateTo?: string): Promise<Fixture[]> {
  const params = new URLSearchParams();
  if (dateFrom) params.set("date_from", dateFrom);
  if (dateTo) params.set("date_to", dateTo);
  const query = params.toString();

  const response = await fetch(`${API_BASE}/api/fixtures${query ? `?${query}` : ""}`);
  if (!response.ok) {
    throw new ApiError(`Failed to load fixtures (${response.status})`, response.status);
  }
  return response.json();
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
  const response = await fetch(`${API_BASE}/api/recommendations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new ApiError(`Failed to generate recommendation (${response.status})`, response.status);
  }
  return response.json();
}

/** Cache-only read (W11) -- never triggers a live agent call. Returns null on
 * a 404 (nothing generated yet for this match/date), throws on any other
 * failure. */
export async function getCachedRecommendation(
  matchId: string,
  date: string
): Promise<MatchRecommendationOut | null> {
  const response = await fetch(
    `${API_BASE}/api/recommendations/${encodeURIComponent(matchId)}?date=${encodeURIComponent(date)}`
  );
  if (response.status === 404) return null;
  if (!response.ok) {
    throw new ApiError(`Failed to load cached recommendation (${response.status})`, response.status);
  }
  return response.json();
}
