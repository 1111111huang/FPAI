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
  // W64: "E0" or "SWE". Optional (not `competition: string`) so every
  // existing hand-built Fixture literal across the test suite -- there are
  // several, none of which set this field -- keeps type-checking without
  // modification; fixtureToMatch() (Task 3) defaults a missing value to
  // "E0", mirroring the backend's own default.
  competition?: string;
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
  // W84 (app_user_stories.md), agent-side A52 (agent_user_stories.md): the
  // price a "conditional" market would need to reach to clear
  // min_value_edge, code-computed server-side -- null when not applicable
  // (not conditional, no current_odds, or no such target exists) or absent
  // entirely on a pre-A52 cached row, so optional rather than required.
  target_odds?: number | null;
};

export type MatchRecommendationOut = {
  match: Record<string, unknown>;
  overall: "direct_bet" | "conditional" | "no_bet" | "insufficient_data";
  markets: MarketRecommendationOut[];
  // One bullet per aspect (value edge, team news, form, market caveats, ...)
  // instead of one narrative paragraph -- direct user request.
  explanation: string[];
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
  // A82 (agent_user_stories.md): Kelly-derived stake-sizing suggestion for
  // this recommendation's actual pick, as a multiple of an abstract "Unit
  // Bet" -- not a dollar figure. null when there's no priced pick, or
  // absent entirely on a pre-A82 cached row.
  unit_bet_multiplier?: number | null;
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
export type LeagueFreshness = {
  latest_match_date: string | null;
  days_since_update: number | null;
  match_count: number;
  is_stale: boolean;
};

// W74: by_league (US#136, engine-side) was already passed through
// unmodified by GET /api/status -- optional here so an older mock/response
// without it still type-checks the same way it always ran.
export type DataFreshness = LeagueFreshness & {
  by_league?: Record<string, LeagueFreshness>;
};

export type ModelStatusEntry = {
  model_type: string | null;
  primary_metric_value: number | null;
  metric_name: string | null;
  selected_at: string | null;
};

// US#110 (engine-side): keyed dynamically by competition_id (e.g. "E0",
// "SWE") for competition_specific competitions, plus a shared
// "international" bucket -- no more fixed "league" key.
export type ModelStatus = Record<string, Record<string, ModelStatusEntry>>;

export type StatusResponse = {
  data_freshness: DataFreshness;
  model_status: ModelStatus;
};

// W27: reflects the backend's sandbox_clock module -- as_of is null when
// sandbox_mode is false.
export type SandboxStatus = {
  sandbox_mode: boolean;
  as_of: string | null;
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

// W170/W171: app/backend/agent_performance_dashboard.py's return shape.
export type HitRateSummary = {
  sample_size: number;
  correct: number;
  hit_rate: number;
};

// Mirrors src/agent/evaluation.py's build_evaluation_report() return shape,
// denominated in UB (an abstract Unit Bet, not dollars -- same convention
// as the rest of this feature).
export type SegmentMetrics = {
  matches_evaluated: number;
  bets_placed: number;
  bets_won: number;
  roi: number;
  hit_rate: number;
  bet_frequency: number;
  max_drawdown: number;
  insufficient_data_rate: number;
  starting_bankroll: number;
  ending_bankroll: number;
  total_staked: number;
  total_profit: number;
};

export type StakedBet = {
  match_id: string;
  market: string;
  selection: string;
  odds: number;
  stake: number;
  won: boolean;
  payout: number;
};

// StakedBet plus match context, only populated for top_winners/top_losers
// (W171) -- home_team/away_team are null on a cache miss, degraded, not absent.
export type TopBet = StakedBet & {
  date: string | null;
  competition: string | null;
  home_team: string | null;
  away_team: string | null;
};

export type AgentPerformanceDashboard = {
  overall: HitRateSummary;
  by_market: Record<string, HitRateSummary>;
  by_competition: Record<string, HitRateSummary>;
  by_confidence: Record<string, HitRateSummary>;
  kelly_roi_simulation: SegmentMetrics;
  by_market_metrics: Record<string, SegmentMetrics>;
  by_market_selection_metrics: Record<string, SegmentMetrics>;
  by_league_metrics: Record<string, SegmentMetrics>;
  staked_bets: StakedBet[];
  top_winners: TopBet[];
  top_losers: TopBet[];
};
