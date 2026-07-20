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

Families not classified here (`DIS`, `CTX`, `MKT`, `EFFICIENCY`, `H2H`, and
the existing `SQUAD`-gated prefixes `SQUAD_`/`LUCK_`/`XOC_`/`FRDS_`/
`DEF_ANCHOR_`) are unaffected by this module -- `EFFICIENCY_*` was verified
against feature_factory.py to be entirely goals-ratio-based (no shots/corners
dependency), so it does not need splitting. `DIS` (cards) already has its own
group tag. `CTX`'s `*_CORNERS_STD_R5` features and `H2H_CORNERS_R5` are a
known residual gap (pre-existing: neither was ever gated by
`enabled_feature_groups` before this change) -- out of scope for US#133,
which is bounded to OFF/DEF/OPP_ADJ (plus the STRENGTH/INTERACTION mixes
this surfaced); left as a follow-up.
"""

from __future__ import annotations


def resolve_feature_group_tag(feature_name: str) -> str | None:
    """Return the `enabled_feature_groups` sub-tag that gates `feature_name`.

    Returns `None` for features not governed by this split-family mechanism
    (e.g. `DIS_*`, `CTX_*`, `MKT_*`, `EFFICIENCY_*`, `H2H_*`, `SQUAD_*` and
    friends) -- callers should treat `None` as "not gated here", i.e. pass
    the feature through exactly like before this mechanism existed.
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

    return None
