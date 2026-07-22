"""Sub-tag classification for competition-registry feature gating (US#133).

`config/competitions.yaml`'s `enabled_feature_groups` originally gated whole
feature families (`OFF`, `DEF`, `OPP_ADJ`, `STRENGTH`, `INTERACTION`, ...).
Several of those families mix raw-data dependencies that not every
competition's data source can supply: e.g. Sweden's Allsvenskan (football-
data.co.uk's "New Leagues" CSV format) has goals and 1X2 market odds only --
no shots, shots-on-target, or corners columns at all.

A bare `OFF`/`DEF`/`OPP_ADJ` tag can't express "goals yes, shots no", so a
competition that naively enabled the whole family would get its shot/corner
sub-features cold-start-imputed with another competition's column mean
(silently misleading, not just "reduced signal" -- see US#133/US#134).

This module classifies each `config/schema.yaml` feature into the specific
sub-tag that gates it, based on which raw columns it actually depends on
(verified against `src/features/feature_factory.py`'s computation, not
name-pattern guessing):

- `OFF`/`DEF`: split into `_GOALS` (fthg/ftag/xg/xga/luck), `_SHOTS`
  (hs/as/hst/ast, shot_accuracy, save_rate), and `_CORNERS` (hc/ac).
- `OPP_ADJ`: split into `_GOALS`, `_SHOTS` (SOT only), and `_CORNERS`.
- `STRENGTH`/`INTERACTION`: split into `_GOALS` and `_SHOTS` (SOT only) --
  verified against feature_factory.py, both families are goals-only except
  for one SOT-based feature each (STRENGTH_SoT_Diff, INTERACTION_ATTACK_SOT_DIFF_R5).
- `DIS_*` (cards: hy/ay/hr/ar): gated as a whole family under `DIS`. US#133's
  docstring claimed "DIS already has its own group tag" but this was never
  actually true -- no code anywhere checked for a bare `"DIS"` tag, so
  `DIS_*` features passed through unconditionally regardless of
  `enabled_feature_groups` content. Found and fixed in US#127, which needed
  `DIS` to be a *real*, enforceable exclusion for Sweden's Allsvenskan
  (goals + 1X2 odds only, no cards data at all).
- `CTX_HOME_CORNERS_STD_R5`/`CTX_AWAY_CORNERS_STD_R5` (hc/ac-derived rolling
  std): gated under `CTX_CORNERS`. The rest of the `CTX_*` family (rest days,
  cumulative points, PPG, goals/conceded std, streaks) is goals/date-only and
  remains ungated (`None` -- always passes), same as before.
- `H2H_CORNERS_R5` (hc/ac-derived rolling H2H mean): gated under `H2H_CORNERS`.
  `H2H_TOTAL_GOALS_R5`/`H2H_HOME_WIN_RATE_R5` are goals-only and remain
  ungated.

`CTX_CORNERS`/`H2H_CORNERS` close the residual gap US#133 flagged and
explicitly left as a follow-up (neither `CTX` nor `H2H` had a pre-existing
bare tag to split, so those two features passed through ungated into any
competition enabling `CTX`/`H2H` -- including a goals-only competition like
Sweden -- and would show up as feature_completeness degradation from
cold-start-imputed corner values with no real corner data behind them).

Families still not classified here (`MKT`, `EFFICIENCY`, and the existing
`SQUAD`-gated prefixes `SQUAD_`/`LUCK_`/`XOC_`/`FRDS_`/`DEF_ANCHOR_`) are
unaffected by this module -- `EFFICIENCY_*` was verified against
feature_factory.py to be entirely goals-ratio-based (no shots/corners
dependency), so it does not need splitting. `MKT_*` was audited in US#127
too: `MKT_IMPLIED_HOME/DRAW/AWAY` and `MKT_OVERROUND` depend only on
avgh/avgd/avga (Sweden has these), but `MKT_IMPLIED_OVER25`, `MKT_AH_LINE`,
`MKT_AH_HOME_ODDS`, `MKT_AH_AWAY_ODDS`, `MKT_LAMBDA_TOTAL`, `MKT_LAMBDA_HOME`,
`MKT_LAMBDA_AWAY`, `MKT_POISSON_BTTS_PROB`, and `MKT_LAMBDA_AH_DIFF` depend on
over25_odds/ah_line/ah_home_odds/ah_away_odds, which Sweden's source (closing
1X2 odds only) never populates. Deliberately left unsplit: unlike the
CTX/H2H-corners case, `_apply_cold_start_imputation` already excludes
`MKT_*` from mean-imputation ("their NaN encodes genuine missing odds
data"), so these columns correctly stay `NaN` rather than being silently
backfilled with another competition's values -- the same accepted behavior
pre-2020 EPL rows already rely on for the same columns. Left as a known,
accepted, permanent gap for Sweden (9 of the ~17 MKT_* columns), not a
correctness bug requiring a new sub-tag.
"""

from __future__ import annotations


def resolve_feature_group_tag(feature_name: str) -> str | None:
    """Return the `enabled_feature_groups` sub-tag that gates `feature_name`.

    Returns a concrete tag for `DIS_*` (-> `"DIS"`), the two corner-dependent
    `CTX_*_CORNERS_STD_R5` features (-> `"CTX_CORNERS"`), and `H2H_CORNERS_R5`
    (-> `"H2H_CORNERS"`), in addition to the OFF/DEF/OPP_ADJ/STRENGTH/
    INTERACTION sub-tags.

    Returns `None` for features not governed by this mechanism at all (e.g.
    the rest of `CTX_*`, `MKT_*`, `EFFICIENCY_*`, the rest of `H2H_*`,
    `SQUAD_*` and friends) -- callers should treat `None` as "not gated
    here", i.e. pass the feature through exactly like before this mechanism
    existed.
    """
    name = feature_name
    upper = name.upper()
    tokens = set(name.split("_"))

    # DEF_ANCHOR_* (Phase 15 defensive anchor, US#104) is a SQUAD-managed
    # prefix, not the DEF rolling-stat family, even though it also starts
    # with "DEF_" -- must not be swept into DEF_GOALS/DEF_SHOTS/DEF_CORNERS.
    if name.startswith("DEF_ANCHOR_"):
        return None

    if name.startswith("OPP_ADJ_"):
        if "SOT" in upper:
            return "OPP_ADJ_SHOTS"
        if "CORNER" in upper:
            return "OPP_ADJ_CORNERS"
        if "GOAL" in upper:
            return "OPP_ADJ_GOALS"
        return None

    if name.startswith("STRENGTH_"):
        return "STRENGTH_SHOTS" if "SOT" in upper else "STRENGTH_GOALS"

    if name.startswith("INTERACTION_"):
        return "INTERACTION_SHOTS" if "SOT" in upper else "INTERACTION_GOALS"

    if name.startswith("OFF_") or name.startswith("DEF_"):
        prefix = "OFF" if name.startswith("OFF_") else "DEF"
        if tokens & {"HC", "AC"}:
            return f"{prefix}_CORNERS"
        if tokens & {"HS", "AS", "HST", "AST"} or "SHOT_ACCURACY" in upper or "SAVE_RATE" in upper:
            return f"{prefix}_SHOTS"
        return f"{prefix}_GOALS"

    if name.startswith("DIS_"):
        return "DIS"

    if name.startswith("CTX_") and "CORNER" in upper:
        return "CTX_CORNERS"

    if name.startswith("H2H_") and "CORNER" in upper:
        return "H2H_CORNERS"

    return None
