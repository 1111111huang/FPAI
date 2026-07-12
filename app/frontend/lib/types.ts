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
  // W15: cold_start_risk/unknown_team are first-class trust signals --
  // treat them as authoritative even when prediction_basis itself claims
  // team_history_and_market (see agent_techspec.md / US#108).
  cold_start_risk: boolean;
  feature_completeness: number | null;
  unknown_team: boolean;
};

// W12: mirrors app/backend/bets.py BetOut.
export type Bet = {
  id: number;
  match_id: string;
  date: string;
  home_team: string;
  away_team: string;
  market: string;
  selection: string;
  odds: number;
  stake: number;
  outcome: "open" | "won" | "lost";
  profit_loss: number | null;
  source: "from_recommendation" | "manual";
  recommendation_snapshot: Record<string, unknown> | null;
  created_at: string;
};

// W17: mirrors app/backend/main.py's GET /api/status response, itself a
// thin pass-through of src/tools/data_tools.get_data_freshness() and
// src/tools/model_tools.get_model_status().
export type DataFreshness = {
  latest_match_date: string | null;
  days_since_update: number | null;
  match_count: number;
  is_stale: boolean;
};

export type ModelStatusEntry = {
  model_type: string | null;
  primary_metric_value: number | null;
  metric_name: string | null;
  selected_at: string | null;
};

export type ModelStatus = {
  league: Record<string, ModelStatusEntry>;
  international: Record<string, ModelStatusEntry>;
};

export type StatusResponse = {
  data_freshness: DataFreshness;
  model_status: ModelStatus;
};

// W14: mirrors app/backend/bet_stats.py's compute_bet_stats() return shape.
export type BetStats = {
  bets_settled: number;
  bets_open: number;
  bets_won: number;
  roi: number;
  hit_rate: number;
  total_staked: number;
  total_profit: number;
  max_drawdown: number;
  starting_bankroll: number;
  current_bankroll: number;
};
