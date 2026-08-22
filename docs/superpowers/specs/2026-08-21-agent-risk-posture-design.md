# Agent Risk Posture: Conservative / Balanced / Aggressive Presets

**Date:** 2026-08-21
**Status:** Approved, pre-implementation
**Covers:** new story to be appended to `documents/agent_user_stories.md`

## Motivation

Investigated (direct user report) why SP1's full-season backtest (2026-08-07, 322 matches) placed 218 bets (67.7% frequency) while today's identical league, same underlying model swapped back in for a direct test, places ~1 bet across an entire season (~0.03/matchday). Root-caused: the numeric thresholds (`min_value_edge=0.05`, odds bounds `1.2`-`11.0`) are unchanged since 08-07; the actual cause is prompt text added alongside A66/A67 (~08-17/18) that explicitly instructs the LLM not to trust a qualifying `value_edge` without further justification. Confirmed directly: 70/299 SP1 markets numerically qualify (edge ≥5%, odds in range) today, yet only 1 gets recommended — the code-level downgrade passes fire on only 0-2/299, so the gap is almost entirely the LLM's own prompt-internalized caution, not a downstream filter.

This was deliberate, real hardening work (A65/A66/A67 fixed genuine incoherent-bet bugs), but it leaves no way to dial the resulting posture — today's behavior is the only option, and it's now far more conservative than useful for experimentation. The user wants a way to compare postures empirically (does a more permissive posture actually perform better or worse on ROI/hit-rate?) rather than being stuck with one fixed, very conservative behavior.

## Goals

- Three selectable postures — conservative, balanced, aggressive — usable via the existing `--config` flag on `agent-backtest`/`agent-train` (no new CLI flag).
- Calibrated against an observable target: (`direct_bet` + `conditional`) recommendations per matchday (~9-10 matches, one league), measured per league:
  - conservative: < 3/matchday
  - balanced: ~5/matchday
  - aggressive: > 7/matchday
- Each posture is a full, independently-readable `AgentConfig` (its own yaml + its own prompt file) — not a runtime-computed variant — so any posture's exact behavior can be inspected/diffed directly, matching this repo's existing `agent_config_deepseek.yaml` precedent.
- Prompt language is the primary lever (confirmed the dominant effect); `min_value_edge` is a secondary, smaller adjustment per posture. Odds bounds (`min_odds_threshold`/`max_odds_threshold`/`min_conditional_odds_threshold`) stay fixed across all three — those are safety rails against degenerate prices, not an aggressiveness dial.
- Calibration is empirical: draft each posture's prompt, test against a small real sample (1-2 matchdays, not a full season) per posture, measure the actual rate, iterate wording before a final, larger validation run.

## Non-goals

- No new `AgentConfig` field or code path for "posture" — reusing `system_prompt_version` + a per-posture config file needs zero new code.
- Not a live, user-facing setting in the app (explicitly deferred — this is for backtesting/experimentation only, per the user's own scoping).
- Not changing `min_odds_threshold`/`max_odds_threshold`/`min_conditional_odds_threshold` per posture — those stay fixed as safety rails.
- Not touching the A65/A66/A67 downgrade code itself — those coherence checks (null odds, out-of-range odds, ineligible-market conditional, sub-floor value-edge, conditional-below-floor) apply identically regardless of posture; a posture changes how often the LLM *proposes* a bet, not what's structurally allowed to survive as one.
- Not retroactively re-validating E0/D1/I1/F1's own historical backtest numbers under the new postures — this story is scoped to building and calibrating the mechanism, using SP1 as the calibration corpus since it's what surfaced the investigation.

## Architecture

### 1. Three new config files (`config/agent_config_{conservative,balanced,aggressive}.yaml`)

Each is a copy of today's `config/agent_config.yaml` with two fields changed:

```yaml
system_prompt_version: "v1_conservative"   # or v1_balanced / v1_aggressive
min_value_edge: 0.06                        # starting guess; see per-posture table below
```

Everything else (`model`, `provider`, `temperature`, odds bounds, `markets`) stays identical to today's default, so a posture comparison isolates exactly the two variables above.

### 2. Three new prompt files (`config/prompts/agent_v1_{conservative,balanced,aggressive}.txt`)

Each starts from today's `agent_v1.txt` verbatim, with the "Value Calculation" section's edge-coherence language replaced per posture. First-draft text (starting point for calibration, not final):

**`agent_v1_balanced.txt`** (target ~5/matchday) — replace today's coherence paragraph with:

```
- Never recommend direct_bet unless this specific market's own value_edge is actually >= 0.05 —
  recommendation_type='direct_bet' with a value_edge below that (including negative) is incoherent
  and will be downgraded to "no_bet" automatically.
- If value_edge clears the threshold and current_odds is a real, current price in the allowed range,
  recommend it. Only decline a qualifying edge when you have a SPECIFIC, concrete reason tied to
  this exact match (e.g. a confirmed absence of a key starter, verified conflicting recorded
  evidence, or a stated data-quality problem with the forecast itself) — not generic caution like
  "the model may lack full context" or "uncertainty is high" on its own. A qualifying number is the
  default signal to act on, not a suggestion to second-guess.
```

**`agent_v1_conservative.txt`** (target <3/matchday) — a middle ground between today's prompt and balanced above:

```
- Never recommend direct_bet unless this specific market's own value_edge is actually >= 0.05 —
  recommendation_type='direct_bet' with a value_edge below that (including negative) is incoherent
  and will be downgraded to "no_bet" automatically.
- A qualifying value_edge is necessary but not sufficient. Before recommending, check the forecast's
  own uncertainty (high entropy/cold_start_risk) and the available team news for anything that
  meaningfully contradicts the model's assumption. Decline if either raises real doubt, even without
  a single decisive red flag — err toward no_bet when genuinely unsure.
```

**`agent_v1_aggressive.txt`** (target >7/matchday):

```
- Never recommend direct_bet unless this specific market's own value_edge is actually >= 0.05 —
  recommendation_type='direct_bet' with a value_edge below that (including negative) is incoherent
  and will be downgraded to "no_bet" automatically.
- Trust a qualifying value_edge by default. Decline only if there is a specific, named, current fact
  that directly contradicts the recommendation (e.g. a confirmed missing key player, or evidence the
  match itself may not occur as scheduled) — general model uncertainty, entropy, or "the model might
  be wrong" reasoning alone is not sufficient grounds to decline a market that already clears the
  threshold.
```

Every other prompt section (output format, stop rules, confidence guidelines, etc.) is untouched across all three.

### 3. Starting `min_value_edge` per posture (secondary lever, subject to calibration)

| Posture | `min_value_edge` (starting guess) | Target rate |
|---|---|---|
| conservative | 0.06 | < 3/matchday |
| balanced | 0.05 (unchanged from today's default) | ~5/matchday |
| aggressive | 0.04 | > 7/matchday |

## Calibration procedure

1. For each posture, run `agent-backtest --league SP1 --split train --sample <N>` (or a fixed small date range covering 1-2 real matchdays, ~9-20 matches) with that posture's config.
2. Count (`direct_bet` + `conditional`) recommendations, normalize to a per-matchday rate (matches / ~10).
3. If the rate misses its target bracket, adjust that posture's prompt wording (not the threshold first — confirmed prompt language is the dominant lever) and re-test on a fresh small sample (avoid re-testing the exact same sample repeatedly, to not overfit wording to one specific set of matches).
4. Once all three postures land in their brackets on the small sample, run one larger validation pass (e.g. the full SP1 train split, ~299 matches) per posture to confirm the rate holds at scale and to get real ROI/hit-rate numbers for comparison.
5. Record the final prompt text, threshold values, and both the small-sample and full-validation rates in the completion notes — same standard of honest, real-numbers reporting this project already uses throughout `agent_user_stories.md`.

## Testing

- No unit test can verify "does this prompt produce ~5 recommendations per matchday" — that's an empirical, LLM-behavior question, validated via the calibration procedure above, not pytest.
- Do add a small regression test confirming `AgentConfig.from_yaml()` loads all three new config files without error and that each resolves to its own distinct `system_prompt_version` — a cheap check that the three files exist, parse, and are wired correctly, before spending real API calls calibrating them.
