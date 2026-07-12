// Mirrors the real backend response shapes exactly (app/backend/football_data_client.py
// NormalizedMatch, app/backend/recommendations.py MatchRecommendationOut) -- these are
// the wire types, distinct from the UI-facing Match type in components/MatchUI.tsx.

export type Fixture = {
  match_id: string;
  utc_date: string;
  status: string;
  home_team: string;
  away_team: string;
  home_goals: number | null;
  away_goals: number | null;
};

export type MarketRecommendationOut = {
  market: string;
  selection: string;
  recommendation_type: "direct_bet" | "conditional" | "no_bet";
  current_odds: number | null;
  min_odds: number;
  ml_probability: number;
  implied_probability: number;
  value_edge: number;
};

export type MatchRecommendationOut = {
  match: Record<string, unknown>;
  overall: "direct_bet" | "conditional" | "no_bet" | "insufficient_data";
  markets: MarketRecommendationOut[];
  explanation: string;
  confidence: "low" | "medium" | "high" | string;
  limitations: string[];
  prediction_basis: string;
  invalid_market_count: number;
};
