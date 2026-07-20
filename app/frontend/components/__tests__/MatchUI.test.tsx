/**
 * W22: frontend test strategy. Component-level tests for MatchCard/
 * TeamBadge/StatusBadge rendering across all four `overall` states
 * (including the degraded ones from W16), plus an interaction test for
 * the bet-logging modal's locked-except-stake behavior. Runs headless via
 * Vitest + React Testing Library -- no live backend; @/lib/api is mocked.
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";

import { LogBetButton, MatchAnalysisPage, MatchCard, StatusBadge, TeamBadge, type Match, type Overall } from "../MatchUI";
import type { MatchRecommendationOut } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  generateRecommendation: vi.fn(),
  getCachedRecommendation: vi.fn(),
  getFixtures: vi.fn(),
  logBetFromRecommendation: vi.fn(),
  ApiError: class ApiError extends Error {},
}));

import { generateRecommendation, getCachedRecommendation, logBetFromRecommendation } from "@/lib/api";

const ALL_OVERALL_STATES: { overall: Overall; label: string }[] = [
  { overall: "direct_bet", label: "Direct Bet" },
  { overall: "conditional", label: "Conditional" },
  { overall: "no_bet", label: "No Bet" },
  { overall: "insufficient_data", label: "Insufficient Data" },
];

function baseMatch(overrides: Partial<Match> = {}): Match {
  return {
    id: "m1",
    league: "E0",
    tier: "competition_specific",
    kickoffIso: "2026-08-22T15:00:00Z",
    home: "Arsenal",
    away: "Everton",
    status: "upcoming",
    hasRecommendation: true,
    overall: "direct_bet",
    confidence: "medium",
    markets: [],
    explanation: "test explanation",
    limitations: [],
    predictionBasis: "team_history_and_market",
    coldStartRisk: false,
    featureCompleteness: 0.9,
    unknownTeam: false,
    invalidMarketCount: 0,
    ...overrides,
  };
}

describe("TeamBadge", () => {
  it("renders initials for a team name", () => {
    render(<TeamBadge name="Arsenal" />);
    expect(screen.getByText("ARS")).toBeInTheDocument();
  });
});

describe("StatusBadge", () => {
  it.each(ALL_OVERALL_STATES)("renders the correct label for overall=$overall", ({ overall, label }) => {
    render(<StatusBadge status={overall} />);
    expect(screen.getByText(label)).toBeInTheDocument();
  });
});

describe("MatchCard", () => {
  it.each(ALL_OVERALL_STATES)("renders without crashing for overall=$overall", ({ overall, label }) => {
    const match = baseMatch({ overall });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Arsenal")).toBeInTheDocument();
    expect(screen.getByText("Everton")).toBeInTheDocument();
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it("renders 'Not yet generated' when hasRecommendation is false, without crashing", () => {
    const match = baseMatch({ hasRecommendation: false });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Not yet generated")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// W47: MatchCard.handleExpand / MatchAnalysisPage.load must check the
// precomputed cache (getCachedRecommendation) before falling back to the
// live "regenerate now" call (generateRecommendation).
// ---------------------------------------------------------------------------

function makeRecommendation(overrides: Partial<MatchRecommendationOut> = {}): MatchRecommendationOut {
  return {
    match: { home: "Arsenal", away: "Everton", date: "2026-08-22", league: "E0" },
    overall: "direct_bet",
    markets: [],
    explanation: "test explanation",
    confidence: "medium",
    limitations: [],
    prediction_basis: "team_history_and_market",
    invalid_market_count: 0,
    cold_start_risk: false,
    feature_completeness: 0.9,
    unknown_team: false,
    ...overrides,
  };
}

describe("MatchCard -- cache-first expand (W47)", () => {
  beforeEach(() => {
    vi.mocked(getCachedRecommendation).mockReset();
    vi.mocked(generateRecommendation).mockReset();
  });

  it("on a cache hit, applies the cached recommendation and never calls generateRecommendation", async () => {
    const cachedRec = makeRecommendation({ explanation: "cached explanation" });
    vi.mocked(getCachedRecommendation).mockResolvedValue(cachedRec);
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    const match = baseMatch({ hasRecommendation: false });

    render(<MatchCard match={match} onUpdate={onUpdate} />);
    await user.click(screen.getByText("Not yet generated"));

    await waitFor(() => expect(onUpdate).toHaveBeenCalled());
    expect(getCachedRecommendation).toHaveBeenCalledWith("m1", "2026-08-22");
    expect(generateRecommendation).not.toHaveBeenCalled();
    expect(onUpdate).toHaveBeenCalledWith(
      expect.objectContaining({ hasRecommendation: true, explanation: "cached explanation" })
    );
  });

  it("on a cache miss (null), falls back to generateRecommendation and applies its result unchanged", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(null);
    const liveRec = makeRecommendation({ explanation: "live explanation" });
    vi.mocked(generateRecommendation).mockResolvedValue(liveRec);
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    const match = baseMatch({ hasRecommendation: false });

    render(<MatchCard match={match} onUpdate={onUpdate} />);
    await user.click(screen.getByText("Not yet generated"));

    await waitFor(() => expect(onUpdate).toHaveBeenCalled());
    expect(getCachedRecommendation).toHaveBeenCalledWith("m1", "2026-08-22");
    expect(generateRecommendation).toHaveBeenCalledWith({
      home_team: "Arsenal",
      away_team: "Everton",
      date: "2026-08-22",
      league: "E0",
      match_id: "m1",
    });
    expect(onUpdate).toHaveBeenCalledWith(
      expect.objectContaining({ hasRecommendation: true, explanation: "live explanation" })
    );
  });
});

describe("MatchAnalysisPage -- cache-first load (W47)", () => {
  beforeEach(() => {
    vi.mocked(getCachedRecommendation).mockReset();
    vi.mocked(generateRecommendation).mockReset();
  });

  it("on a cache hit, renders the cached recommendation and never calls generateRecommendation", async () => {
    const cachedRec = makeRecommendation({ explanation: "cached explanation" });
    vi.mocked(getCachedRecommendation).mockResolvedValue(cachedRec);

    render(<MatchAnalysisPage id="m1" home="Arsenal" away="Everton" date="2026-08-22" />);

    await waitFor(() => expect(screen.getByText("cached explanation")).toBeInTheDocument());
    expect(getCachedRecommendation).toHaveBeenCalledWith("m1", "2026-08-22");
    expect(generateRecommendation).not.toHaveBeenCalled();
  });

  it("on a cache miss (null), falls back to generateRecommendation and renders its result unchanged", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(null);
    const liveRec = makeRecommendation({ explanation: "live explanation" });
    vi.mocked(generateRecommendation).mockResolvedValue(liveRec);

    render(<MatchAnalysisPage id="m1" home="Arsenal" away="Everton" date="2026-08-22" />);

    await waitFor(() => expect(screen.getByText("live explanation")).toBeInTheDocument());
    expect(getCachedRecommendation).toHaveBeenCalledWith("m1", "2026-08-22");
    expect(generateRecommendation).toHaveBeenCalledWith({
      home_team: "Arsenal",
      away_team: "Everton",
      date: "2026-08-22",
      league: "E0",
      match_id: "m1",
    });
  });
});

describe("LogBetButton (bet-logging locked-except-stake behavior)", () => {
  const recommendation: MatchRecommendationOut = {
    match: { home: "Arsenal", away: "Everton", date: "2026-08-22", league: "E0" },
    overall: "direct_bet",
    markets: [
      { market: "result_3way", selection: "home", recommendation_type: "direct_bet", current_odds: 2.1, min_odds: 1.5, ml_probability: 0.5, implied_probability: 0.47, value_edge: 0.03 },
    ],
    explanation: "test",
    confidence: "medium",
    limitations: [],
    prediction_basis: "team_history_and_market",
    invalid_market_count: 0,
    cold_start_risk: false,
    feature_completeness: 0.9,
    unknown_team: false,
  };

  it("initially shows only a 'Log bet' trigger -- nothing editable yet", () => {
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" />);
    expect(screen.getByText("Log bet")).toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Stake")).not.toBeInTheDocument();
  });

  it("opening it reveals only a stake input -- no home/away/odds/market/selection fields", async () => {
    const user = userEvent.setup();
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" />);

    await user.click(screen.getByText("Log bet"));

    expect(screen.getByPlaceholderText("Stake")).toBeInTheDocument();
    // The only text input rendered is the stake field -- confirms nothing
    // else (home/away/odds/market/selection) is exposed as editable.
    expect(screen.getAllByRole("textbox")).toHaveLength(1);
  });

  it("rejects a zero/invalid stake without calling the API", async () => {
    const user = userEvent.setup();
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" />);

    await user.click(screen.getByText("Log bet"));
    await user.click(screen.getByText("Confirm"));

    expect(screen.getByText("Enter a stake greater than 0.")).toBeInTheDocument();
    expect(logBetFromRecommendation).not.toHaveBeenCalled();
  });

  it("confirming a valid stake locks match_id/recommendation/market/selection from props, only stake is user input", async () => {
    vi.mocked(logBetFromRecommendation).mockResolvedValue({
      id: 1, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
      market: "result_3way", selection: "home", odds: 2.1, stake: 10, outcome: "open",
      profit_loss: null, source: "from_recommendation", recommendation_snapshot: null, created_at: "now",
    });
    const user = userEvent.setup();
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" />);

    await user.click(screen.getByText("Log bet"));
    await user.type(screen.getByPlaceholderText("Stake"), "10");
    await user.click(screen.getByText("Confirm"));

    expect(logBetFromRecommendation).toHaveBeenCalledWith({
      match_id: "m1", recommendation, market: "result_3way", selection: "home", stake: 10,
    });
    expect(await screen.findByText("Logged")).toBeInTheDocument();
  });
});
