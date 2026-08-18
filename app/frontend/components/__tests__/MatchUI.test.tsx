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

import {
  fixtureToMatch,
  LogBetButton,
  MatchAnalysisPage,
  MatchCard,
  StatusBadge,
  TeamBadge,
  type Match,
  type Overall,
} from "../MatchUI";
import type { Fixture, MatchRecommendationOut } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  generateRecommendation: vi.fn(),
  getCachedRecommendation: vi.fn(),
  getFixtures: vi.fn(),
  logBetFromRecommendation: vi.fn(),
  // W64: MatchAnalysisPage now renders inside AppShell (Task 9), which
  // independently calls getStatus() (unconditionally, on mount) and
  // useSandboxAsOf() -> getSandboxStatus() -- neither existed as a
  // dependency of this test file before that wiring. Both need mocks here
  // or AppShell throws "No export is defined on the mock" the moment it
  // mounts, same class of pre-existing-mock gap Tasks 7/8 already hit and
  // fixed the same way (see AppShell.test.tsx, MatchUI.race.test.tsx, etc).
  getStatus: vi.fn(),
  getSandboxStatus: vi.fn(),
  ApiError: class ApiError extends Error {},
}));

import {
  generateRecommendation,
  getCachedRecommendation,
  getSandboxStatus,
  getStatus,
  logBetFromRecommendation,
} from "@/lib/api";

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
    explanation: ["test explanation"],
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

describe("TeamBadge -- Allsvenskan club colors (W61)", () => {
  // Keys are the exact spelling The Odds API returns for these fixtures
  // (confirmed live, W55/W59) -- not the ML engine's internal canonical
  // short name, since that's only used for odds-matching, never rendered.
  it.each([
    ["Malmo FF", "#6CACE4"],
    ["AIK", "#000000"],
    ["Djurgardens IF", "#003D7A"],
    ["Hammarby IF", "#046A38"],
  ])("renders %s with its real club color, not the generic hash-based fallback", (name, primary) => {
    const { container } = render(<TeamBadge name={name} />);
    const badge = container.querySelector("span");
    expect(badge).toHaveStyle({ background: primary });
  });

  it("still falls back to the generic hash-based badge for an unmapped Allsvenskan club", () => {
    // No regression: a club not explicitly added (e.g. a smaller/promoted
    // side) must keep rendering via the existing fallback, not crash or
    // render blank.
    const { container } = render(<TeamBadge name="Sirius" />);
    const badge = container.querySelector("span");
    expect(badge).toHaveAttribute("style");
    expect(badge?.getAttribute("style")).not.toBe("");
  });
});

describe("TeamBadge -- La Liga club colors (W80)", () => {
  // Keys are the exact `shortName` football-data.org returns for these
  // fixtures (confirmed live, W74/W76) -- not the ML engine's internal
  // canonical short name, since that's only used for odds/corpus
  // matching, never rendered.
  it.each([
    ["Real Madrid", "#FFFFFF"],
    ["Barça", "#A50044"],
    ["Atleti", "#CB3524"],
    ["Sevilla FC", "#D00027"],
  ])("renders %s with its real club color, not the generic hash-based fallback", (name, primary) => {
    const { container } = render(<TeamBadge name={name} />);
    const badge = container.querySelector("span");
    expect(badge).toHaveStyle({ background: primary });
  });

  it("still falls back to the generic hash-based badge for an unmapped La Liga club", () => {
    // No regression: a club not explicitly added (e.g. Santander, W78's
    // documented cold-start case) must keep rendering via the existing
    // fallback, not crash or render blank.
    const { container } = render(<TeamBadge name="Santander" />);
    const badge = container.querySelector("span");
    expect(badge).toHaveAttribute("style");
    expect(badge?.getAttribute("style")).not.toBe("");
  });
});

describe("StatusBadge", () => {
  it.each(ALL_OVERALL_STATES)("renders the correct label for overall=$overall", ({ overall, label }) => {
    render(<StatusBadge status={overall} />);
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it.each(ALL_OVERALL_STATES)("W107: has a plain-language explanation for overall=$overall", ({ overall, label }) => {
    render(<StatusBadge status={overall} />);
    expect(screen.getByText(label)).toHaveAttribute("title");
    expect(screen.getByText(label).getAttribute("title")?.length).toBeGreaterThan(10);
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

  it("W64: shows 'Modeled' (not the old league-specific 'EPL') for a competition_specific match, since this tag now renders on both E0 and SWE cards", () => {
    const match = baseMatch({ tier: "competition_specific" });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Modeled")).toBeInTheDocument();
    expect(screen.queryByText("EPL")).not.toBeInTheDocument();
  });

  // A "Conditional" verdict names a state ("wait") but previously never said
  // what it's waiting for -- W84/A52: targetOdds (code-computed server-side,
  // src/agent/schema.py _compute_target_odds) is the price this market
  // would need to reach to clear min_value_edge. Shown in the ODDS box
  // itself (replacing the current price with "Wait ≥"/target_odds, warning-
  // colored) rather than as plain text, so it's visually distinct from a
  // normal live price.
  it("shows the wait-condition threshold (target_odds), warning-colored, for a conditional market", () => {
    const match = baseMatch({
      overall: "conditional",
      markets: [
        {
          market: "result_3way", selection: "home", recommendationType: "conditional",
          currentOdds: 1.15, minOdds: 0, mlProbability: 0.6, impliedProbability: 0.87, valueEdge: -0.27,
          targetOdds: 1.85,
        },
      ],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Wait ≥")).toBeInTheDocument();
    const value = screen.getByText("1.85");
    expect(value).toBeInTheDocument();
    expect(value).toHaveClass("text-warning");
    // The "Odds" *label* still steps aside for "Wait ≥" (still one box, one
    // headline number) -- but direct user feedback reversed the earlier
    // call to hide the current price entirely: a bare target with no
    // reference point doesn't say how far off the market actually is, so
    // the live current_odds is now shown underneath as a small "now X.XX".
    expect(screen.queryByText("Odds")).not.toBeInTheDocument();
    expect(screen.getByText("now 1.15")).toBeInTheDocument();
  });

  it("shows the plain Odds box (no Wait ≥) for a direct_bet market -- there's nothing to wait for", () => {
    const match = baseMatch({
      overall: "direct_bet",
      markets: [
        {
          market: "result_3way", selection: "home", recommendationType: "direct_bet",
          currentOdds: 1.8, minOdds: 0, mlProbability: 0.6, impliedProbability: 0.56, valueEdge: 0.04,
          targetOdds: null,
        },
      ],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.queryByText("Wait ≥")).not.toBeInTheDocument();
    expect(screen.getByText("Odds")).toBeInTheDocument();
    expect(screen.getByText("1.80")).toBeInTheDocument();
  });

  it("shows the plain Odds box when target_odds is null (not applicable, or no such target exists)", () => {
    const match = baseMatch({
      overall: "conditional",
      markets: [
        {
          market: "result_3way", selection: "home", recommendationType: "conditional",
          currentOdds: 1.8, minOdds: 0, mlProbability: 0.55, impliedProbability: 0.56, valueEdge: -0.01,
          targetOdds: null,
        },
      ],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.queryByText("Wait ≥")).not.toBeInTheDocument();
    expect(screen.getByText("Odds")).toBeInTheDocument();
  });

  it("shows the plain Odds box when target_odds is absent (pre-A52 cached data)", () => {
    const match = baseMatch({
      overall: "conditional",
      markets: [
        {
          market: "result_3way", selection: "home", recommendationType: "conditional",
          currentOdds: 1.8, minOdds: 0, mlProbability: 0.55, impliedProbability: 0.56, valueEdge: -0.01,
        },
      ],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.queryByText("Wait ≥")).not.toBeInTheDocument();
    expect(screen.getByText("Odds")).toBeInTheDocument();
  });
});

describe("MatchCard -- W153: the top badge describes the shown market, not match.overall", () => {
  // Direct user report: a live card showed a "Direct Bet" badge over a
  // market that was rendering as "WAIT ≥" (conditional) right below it.
  // A65 (agent_user_stories.md) fixes this at the source for new
  // generations, but the badge must be correct even against an
  // already-stale/inconsistent cached row -- this is the frontend-side
  // guarantee, independent of whatever match.overall happens to say.
  it("shows Conditional, not the stale Direct Bet overall, when the single shown market is conditional", () => {
    const match = baseMatch({
      overall: "direct_bet", // stale -- as if cached before A65 shipped
      markets: [
        {
          market: "home_corners", selection: "over_2.5", recommendationType: "conditional",
          currentOdds: 1.13, minOdds: 0, mlProbability: 0.92, impliedProbability: 0.885, valueEdge: 0.035,
          targetOdds: 1.2,
        },
      ],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Conditional")).toBeInTheDocument();
    expect(screen.queryByText("Direct Bet")).not.toBeInTheDocument();
  });

  it("shows the higher-edge conditional market's badge, not a lower-edge direct_bet market's, when overall reports the latter", () => {
    // The general multi-market case A65's backend cap alone doesn't fully
    // cover: overall="direct_bet" is legitimately true of *a* market
    // (result_3way), but bestMarket() picks the higher-edge one
    // (home_corners, conditional) to actually display -- the badge must
    // follow what's shown, not the match-wide aggregate.
    const match = baseMatch({
      overall: "direct_bet",
      markets: [
        {
          market: "result_3way", selection: "home", recommendationType: "direct_bet",
          currentOdds: 2.1, minOdds: 1.8, mlProbability: 0.55, impliedProbability: 0.48, valueEdge: 0.03,
          targetOdds: null,
        },
        {
          market: "home_corners", selection: "over_2.5", recommendationType: "conditional",
          currentOdds: 1.13, minOdds: 0, mlProbability: 0.92, impliedProbability: 0.885, valueEdge: 0.15,
          targetOdds: 1.2,
        },
      ],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Conditional")).toBeInTheDocument();
    expect(screen.queryByText("Direct Bet")).not.toBeInTheDocument();
  });
});

describe("MatchCard -- live match display", () => {
  it("shows a LIVE badge for a live match", () => {
    const match = baseMatch({ status: "live", result: { home: 1, away: 0 } });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("LIVE")).toBeInTheDocument();
  });

  it("shows the current in-progress score for a live match", () => {
    const match = baseMatch({ status: "live", result: { home: 1, away: 0 } });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("0")).toBeInTheDocument();
  });

  it("does not show a LIVE badge or score for an upcoming match", () => {
    const match = baseMatch({ status: "upcoming" });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.queryByText("LIVE")).not.toBeInTheDocument();
  });

  it("does not show a LIVE badge for a completed match", () => {
    const match = baseMatch({ status: "completed", result: { home: 2, away: 1 } });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.queryByText("LIVE")).not.toBeInTheDocument();
  });

  it("the recommendation badge still shows for a live match with a recommendation -- live doesn't replace it", () => {
    const match = baseMatch({ status: "live", result: { home: 1, away: 0 }, overall: "direct_bet" });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("LIVE")).toBeInTheDocument();
    expect(screen.getByText("Direct Bet")).toBeInTheDocument();
  });

  it("the Market/Pick/Odds/Edge row is unchanged for a live match -- still shows the original recommendation and odds, not relabeled 'live'", () => {
    const match = baseMatch({
      status: "live",
      result: { home: 1, away: 0 },
      markets: [
        {
          market: "btts", selection: "no", recommendationType: "direct_bet",
          currentOdds: 1.66, minOdds: 0, mlProbability: 0.6, impliedProbability: 0.56, valueEdge: 0.074,
        },
      ],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Odds")).toBeInTheDocument();
    expect(screen.getByText("1.66")).toBeInTheDocument();
    expect(screen.queryByText("Live Odds")).not.toBeInTheDocument();
  });
});

describe("MatchCard -- Market/Pick/Odds/Edge grid redesign (2026-08-13, direct mockup)", () => {
  it("renders Market/Pick/Odds/Edge with a real team name and direction caption for a home pick", () => {
    const match = baseMatch({
      home: "Arsenal", away: "Everton",
      overall: "direct_bet",
      markets: [
        { market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 1.8, minOdds: 0, mlProbability: 0.6, impliedProbability: 0.56, valueEdge: 0.04 },
      ],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    // W121 follow-up: market names are humanized ("3-Way Result" / "Full
    // Time"), not the raw "result_3way" backend string.
    expect(screen.getByText("3-Way Result")).toBeInTheDocument();
    expect(screen.getByText("Full Time")).toBeInTheDocument();
    // "Arsenal" renders twice by design -- once as the home team, once as
    // the pick's team-named label (pickLabel reuses selectionLabel).
    expect(screen.getAllByText("Arsenal")).toHaveLength(2);
    expect(screen.getByText("To Win")).toBeInTheDocument();
    expect(screen.getByText("1.80")).toBeInTheDocument();
    expect(screen.getByText("Decimal")).toBeInTheDocument();
    expect(screen.getByText("+4.0%")).toBeInTheDocument();
    expect(screen.getByText("Positive Edge")).toBeInTheDocument();
  });

  it("shows an 'Over'-captioned pick for a totals-market selection, and no caption/arrow for a draw pick", () => {
    // total_goals is the real backend market for an over/under line
    // (src/agent/schema.py) -- "over_under_2.5" (this test's original
    // market string) was never a real one.
    const overUnder = baseMatch({
      markets: [
        { market: "total_goals", selection: "over_2.5", recommendationType: "direct_bet", currentOdds: 1.85, minOdds: 0, mlProbability: 0.55, impliedProbability: 0.54, valueEdge: 0.01 },
      ],
    });
    render(<MatchCard match={overUnder} onUpdate={vi.fn()} />);
    expect(screen.getByText("Over 2.5")).toBeInTheDocument();
    expect(screen.getByText("Over")).toBeInTheDocument();

    const draw = baseMatch({
      markets: [
        { market: "result_3way", selection: "draw", recommendationType: "direct_bet", currentOdds: 3.2, minOdds: 0, mlProbability: 0.35, impliedProbability: 0.31, valueEdge: 0.04 },
      ],
    });
    render(<MatchCard match={draw} onUpdate={vi.fn()} />);
    expect(screen.getByText("Draw")).toBeInTheDocument();
    expect(screen.queryByText("To Win")).not.toBeInTheDocument();
  });

  it("shows dashes for Market/Pick/Odds/Edge when there's no recommendation yet", () => {
    const match = baseMatch({ hasRecommendation: false, markets: [] });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(3); // Market, Pick, Odds, Edge
  });

  it("mockup points 1/4: team row is one horizontal line, and day/time are bullet-separated", () => {
    const match = baseMatch({ home: "Arsenal", away: "Everton" });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    // Home and away sit in the same row -- both team names' closest
    // ancestor <button> row (not stacked in separate divs) shares one
    // parent flex container.
    const homeEl = screen.getByText("Arsenal");
    const awayEl = screen.getByText("Everton");
    expect(homeEl.parentElement).toBe(awayEl.parentElement);
    expect(screen.getByText("v")).toBeInTheDocument();
    expect(screen.getByText("•")).toBeInTheDocument();
  });

  it("does not show the Positive Edge tag for a no_bet market with a numerically positive edge", () => {
    const match = baseMatch({
      overall: "no_bet",
      markets: [
        { market: "result_3way", selection: "away", recommendationType: "no_bet", currentOdds: 5.0, minOdds: 0, mlProbability: 0.23, impliedProbability: 0.2, valueEdge: 0.032 },
      ],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.queryByText("Positive Edge")).not.toBeInTheDocument();
  });

  it("keeps the days-until-kickoff label independently findable", () => {
    // UTC noon kickoff + sandboxMode, matching this file's own
    // matchWithKickoff/dateboundary convention -- sidesteps the local-vs-UTC
    // getter distinction dayDiff/formatDay deliberately branch on.
    const match = baseMatch({ kickoffIso: "2026-08-25T12:00:00Z" });
    render(
      <MatchCard match={match} onUpdate={vi.fn()} asOf={new Date("2026-08-22T00:00:00Z")} sandboxMode={true} />
    );
    expect(screen.getByText("in 3 days")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Direct user report (2026-08-08): a "No Bet" card showed a prominent
// positive "+3.2% EDGE" -- a real, positive-but-below-threshold edge on a
// no_bet market (a legitimate state, e.g. A54's ineligible-market downgrade,
// or simply below min_value_edge) was picked as bestMarket() purely because
// it had the numerically highest value_edge, with no regard for whether
// that market was actually being recommended. A green-colored positive edge
// on a "No Bet" card reads as a good bet that isn't actually being offered.
// ---------------------------------------------------------------------------
describe("MatchCard -- bestMarket prefers an actionable market over a higher-edge no_bet one", () => {
  it("shows the direct_bet market's odds/edge, not a no_bet market with a numerically higher edge", () => {
    const match = baseMatch({
      overall: "direct_bet",
      markets: [
        // Higher edge, but not actionable -- must not be the one shown.
        {
          market: "result_3way", selection: "away", recommendationType: "no_bet",
          currentOdds: 5.0, minOdds: 0, mlProbability: 0.23, impliedProbability: 0.2, valueEdge: 0.08,
          targetOdds: null,
        },
        // Lower edge, but this is the actual recommendation.
        {
          market: "result_3way", selection: "home", recommendationType: "direct_bet",
          currentOdds: 1.8, minOdds: 0, mlProbability: 0.6, impliedProbability: 0.56, valueEdge: 0.04,
          targetOdds: null,
        },
      ],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("1.80")).toBeInTheDocument();
    expect(screen.getByText("+4.0%")).toBeInTheDocument();
    expect(screen.queryByText("5.00")).not.toBeInTheDocument();
    expect(screen.queryByText("+8.0%")).not.toBeInTheDocument();
  });

  it("does not color a no_bet market's positive edge as 'good' when nothing is actionable", () => {
    const match = baseMatch({
      overall: "no_bet",
      markets: [
        {
          market: "result_3way", selection: "away", recommendationType: "no_bet",
          currentOdds: 5.0, minOdds: 0, mlProbability: 0.23, impliedProbability: 0.2, valueEdge: 0.032,
          targetOdds: null,
        },
      ],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    const edge = screen.getByText("+3.2%");
    expect(edge).toBeInTheDocument();
    expect(edge).not.toHaveClass("text-good");
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
    explanation: ["test explanation"],
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

function baseFixture(overrides: Partial<Fixture> = {}): Fixture {
  return {
    match_id: "m1",
    utc_date: "2026-08-22T15:00:00Z",
    status: "SCHEDULED",
    home_team: "Arsenal",
    away_team: "Everton",
    home_goals: null,
    away_goals: null,
    ...overrides,
  };
}

describe("fixtureToMatch -- W64 real competition, not a hardcoded E0", () => {
  it("uses the fixture's real competition when present", () => {
    const fixture = baseFixture({ competition: "SWE" });
    expect(fixtureToMatch(fixture).league).toBe("SWE");
  });

  it("falls back to E0 when competition is genuinely absent", () => {
    // baseFixture()'s default never sets `competition` at all (not merely
    // `undefined`) -- proving the `?? "E0"` fallback works for a field
    // that's truly missing, not just re-testing the SWE case above.
    const fixtureWithoutCompetition = baseFixture();
    expect(fixtureToMatch(fixtureWithoutCompetition).league).toBe("E0");
  });
});

describe("fixtureToMatch -- live status (a match currently being played)", () => {
  it("maps an IN_PLAY fixture to status 'live'", () => {
    const fixture = baseFixture({ status: "IN_PLAY", home_goals: 1, away_goals: 0 });
    expect(fixtureToMatch(fixture).status).toBe("live");
  });

  it("maps a PAUSED fixture (half-time) to status 'live' too", () => {
    const fixture = baseFixture({ status: "PAUSED", home_goals: 1, away_goals: 0 });
    expect(fixtureToMatch(fixture).status).toBe("live");
  });

  it("populates result with the current in-progress score for a live fixture, not just a completed one", () => {
    const fixture = baseFixture({ status: "IN_PLAY", home_goals: 1, away_goals: 0 });
    expect(fixtureToMatch(fixture).result).toEqual({ home: 1, away: 0 });
  });

  it("a live fixture with no goals yet gets a 0-0 result, not undefined", () => {
    const fixture = baseFixture({ status: "IN_PLAY", home_goals: 0, away_goals: 0 });
    expect(fixtureToMatch(fixture).result).toEqual({ home: 0, away: 0 });
  });

  it("a genuinely SCHEDULED fixture (unchanged) still maps to 'upcoming', not 'live'", () => {
    const fixture = baseFixture({ status: "SCHEDULED" });
    expect(fixtureToMatch(fixture).status).toBe("upcoming");
  });
});

describe("MatchCard -- cache-first expand (W47)", () => {
  beforeEach(() => {
    vi.mocked(getCachedRecommendation).mockReset();
    vi.mocked(generateRecommendation).mockReset();
  });

  it("on a cache hit, applies the cached recommendation and never calls generateRecommendation", async () => {
    const cachedRec = makeRecommendation({ explanation: ["cached explanation"] });
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
      expect.objectContaining({ hasRecommendation: true, explanation: ["cached explanation"] })
    );
  });

  it("W64: requests a recommendation with the fixture's real competition, not a hardcoded E0", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(null);
    vi.mocked(generateRecommendation).mockResolvedValue(makeRecommendation());
    const user = userEvent.setup();
    const match = baseMatch({ hasRecommendation: false, league: "SWE" });

    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    await user.click(screen.getByText("Not yet generated"));

    await waitFor(() => expect(generateRecommendation).toHaveBeenCalled());
    expect(generateRecommendation).toHaveBeenCalledWith(expect.objectContaining({ league: "SWE" }));
  });

  it("on a cache miss (null), falls back to generateRecommendation and applies its result unchanged", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(null);
    const liveRec = makeRecommendation({ explanation: ["live explanation"] });
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
      expect.objectContaining({ hasRecommendation: true, explanation: ["live explanation"] })
    );
  });
});

describe("MatchAnalysisPage -- cache-first load (W47)", () => {
  beforeEach(() => {
    vi.mocked(getCachedRecommendation).mockReset();
    vi.mocked(generateRecommendation).mockReset();
    // AppShell (wrapping MatchAnalysisPage as of Task 9) calls these
    // unconditionally on mount -- give them harmless defaults so each
    // test's real assertions aren't drowned out by an unrelated mock
    // rejection/undefined-return from AppShell's own chrome. Rejecting
    // getStatus matches AppShell.test.tsx's own precedent (AppShell
    // catches the rejection and just shows "--" in its sidebar footer).
    vi.mocked(getStatus).mockReset().mockRejectedValue(new Error("no backend"));
    vi.mocked(getSandboxStatus).mockReset().mockResolvedValue({ sandbox_mode: false, as_of: null });
  });

  it("on a cache hit, renders the cached recommendation and never calls generateRecommendation", async () => {
    const cachedRec = makeRecommendation({ explanation: ["cached explanation"] });
    vi.mocked(getCachedRecommendation).mockResolvedValue(cachedRec);

    render(<MatchAnalysisPage id="m1" home="Arsenal" away="Everton" date="2026-08-22" />);

    await waitFor(() => expect(screen.getByText("cached explanation")).toBeInTheDocument());
    expect(getCachedRecommendation).toHaveBeenCalledWith("m1", "2026-08-22");
    expect(generateRecommendation).not.toHaveBeenCalled();
  });

  it("on a cache miss (null), falls back to generateRecommendation and renders its result unchanged", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(null);
    const liveRec = makeRecommendation({ explanation: ["live explanation"] });
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

  it("W64: requests a recommendation with the passed league, not a hardcoded E0", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(null);
    vi.mocked(generateRecommendation).mockResolvedValue(makeRecommendation());

    render(<MatchAnalysisPage id="m1" home="Malmo FF" away="AIK" date="2026-08-22" league="SWE" />);

    await waitFor(() => expect(generateRecommendation).toHaveBeenCalled());
    expect(generateRecommendation).toHaveBeenCalledWith(expect.objectContaining({ league: "SWE" }));
    // W110: the header shows the full competition name, not the raw code.
    expect(screen.getByText("Allsvenskan")).toBeInTheDocument();
    expect(screen.queryByText("SWE")).not.toBeInTheDocument();
  });

  it("W110: falls back to the raw code for a competition with no known full name", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(makeRecommendation());

    render(<MatchAnalysisPage id="m1" home="Arsenal" away="Everton" date="2026-08-22" league="XYZ" />);

    expect(await screen.findByText("XYZ")).toBeInTheDocument();
  });

  it("W112: does not render the Squad Intelligence stub", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(makeRecommendation());

    render(<MatchAnalysisPage id="m1" home="Arsenal" away="Everton" date="2026-08-22" />);

    await screen.findByText("Agent Reasoning");
    expect(screen.queryByText("Squad Intelligence")).not.toBeInTheDocument();
  });

  it("W111: shows a plain-language summary sentence naming the actual team, not just 'home'", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(
      makeRecommendation({
        overall: "direct_bet",
        confidence: "high",
        markets: [
          {
            market: "result_3way", selection: "home", recommendation_type: "direct_bet",
            current_odds: 1.8, min_odds: 0, ml_probability: 0.6, implied_probability: 0.56, value_edge: 0.04,
          },
        ],
      })
    );

    render(<MatchAnalysisPage id="m1" home="Arsenal" away="Everton" date="2026-08-22" />);

    expect(
      // Rebrand: FPAI -> Oddsey.
      await screen.findByText("Oddsey recommends betting on Arsenal (result_3way), with high confidence.")
    ).toBeInTheDocument();
  });

  it("W111: falls back to a plain no-data sentence when overall is insufficient_data", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(makeRecommendation({ overall: "insufficient_data" }));

    render(<MatchAnalysisPage id="m1" home="Arsenal" away="Everton" date="2026-08-22" />);

    expect(
      await screen.findByText("Oddsey doesn't have enough data yet for a confident read on this match.")
    ).toBeInTheDocument();
  });

  it("W84: shows the wait-condition (target_odds) on a conditional market's own row", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(
      makeRecommendation({
        overall: "conditional",
        markets: [
          {
            market: "result_3way", selection: "home", recommendation_type: "conditional",
            current_odds: 1.15, min_odds: 0, ml_probability: 0.6, implied_probability: 0.87, value_edge: -0.27,
            target_odds: 1.85,
          },
        ],
      })
    );

    render(<MatchAnalysisPage id="m1" home="Arsenal" away="Everton" date="2026-08-22" />);

    const condition = await screen.findByText(/Needs 1\.85\+ to clear edge/);
    expect(condition).toBeInTheDocument();
    expect(condition).toHaveClass("text-warning");
  });

  it("W84: shows no wait-condition line when target_odds is null", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(
      makeRecommendation({
        overall: "conditional",
        markets: [
          {
            market: "result_3way", selection: "home", recommendation_type: "conditional",
            current_odds: 1.8, min_odds: 0, ml_probability: 0.55, implied_probability: 0.56, value_edge: -0.01,
            target_odds: null,
          },
        ],
      })
    );

    render(<MatchAnalysisPage id="m1" home="Arsenal" away="Everton" date="2026-08-22" />);

    // W111: a substring match on "result_3way" is now ambiguous -- the new
    // plain-language summary sentence also mentions the market name -- so
    // wait on the ProbabilityRow's own unique combined text instead.
    await screen.findByText("result_3way · home");
    expect(screen.queryByText(/to clear edge/)).not.toBeInTheDocument();
  });

  it("W84: shows no wait-condition line for a direct_bet market", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(
      makeRecommendation({
        overall: "direct_bet",
        markets: [
          {
            market: "result_3way", selection: "home", recommendation_type: "direct_bet",
            current_odds: 2.1, min_odds: 0, ml_probability: 0.6, implied_probability: 0.48, value_edge: 0.12,
            target_odds: null,
          },
        ],
      })
    );

    render(<MatchAnalysisPage id="m1" home="Arsenal" away="Everton" date="2026-08-22" />);

    // W111: a substring match on "result_3way" is now ambiguous -- the new
    // plain-language summary sentence also mentions the market name -- so
    // wait on the ProbabilityRow's own unique combined text instead.
    await screen.findByText("result_3way · home");
    expect(screen.queryByText(/to clear edge/)).not.toBeInTheDocument();
  });

  it("W115: does not render a Log bet control -- bet tracking hidden for now", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(
      makeRecommendation({
        overall: "direct_bet",
        markets: [
          {
            market: "result_3way", selection: "home", recommendation_type: "direct_bet",
            current_odds: 2.1, min_odds: 0, ml_probability: 0.6, implied_probability: 0.48, value_edge: 0.12,
          },
        ],
      })
    );

    render(<MatchAnalysisPage id="m1" home="Arsenal" away="Everton" date="2026-08-22" />);

    // W111: a substring match on "result_3way" is now ambiguous -- the new
    // plain-language summary sentence also mentions the market name -- so
    // wait on the ProbabilityRow's own unique combined text instead.
    await screen.findByText("result_3way · home");
    expect(screen.queryByText("Log bet")).not.toBeInTheDocument();
  });
});

describe("LogBetButton (bet-logging locked-except-stake behavior)", () => {
  const recommendation: MatchRecommendationOut = {
    match: { home: "Arsenal", away: "Everton", date: "2026-08-22", league: "E0" },
    overall: "direct_bet",
    markets: [
      { market: "result_3way", selection: "home", recommendation_type: "direct_bet", current_odds: 2.1, min_odds: 1.5, ml_probability: 0.5, implied_probability: 0.47, value_edge: 0.03 },
    ],
    explanation: ["test"],
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
