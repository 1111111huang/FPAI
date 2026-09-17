/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      // Direct user spec (2026-09-17): Inter (font-sans, the Tailwind
      // default -- so every existing unqualified text class keeps working
      // unchanged) and IBM Plex Mono (font-mono -- already used pervasively
      // throughout the app for numeric content: odds/stakes/percentages/
      // edge values, see MatchUI.tsx/BetTracker.tsx/AgentPerformanceDashboard.tsx
      // /DashboardRail.tsx, so this one config change propagates the real
      // font everywhere that convention is already followed, no per-usage
      // changes needed). Variables set on <html> by app/layout.tsx's
      // next/font/google calls.
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["var(--font-plex-mono)", "ui-monospace", "monospace"],
      },
      // Direct user spec (2026-09-17), standardized color audit: Tailwind
      // KEY names mostly kept from before (page/surface/ink/muted/accent/
      // good/warning/serious), just repointed to the new CSS variables --
      // deliberate, not an oversight: those names already matched their
      // new role 1:1 (e.g. "accent" already meant "primary action, links,"
      // exactly --blue's new role), so keeping them avoids an otherwise-
      // pure-rename diff across every one of their 100+ existing call
      // sites for zero visual benefit. Only genuinely NEW concepts get new
      // key names: surface-raised/inset (the old single "surface" didn't
      // distinguish card/modal/inset depth), gold/purple/slate (colors
      // with no old equivalent), and each `-dim` pill-background variant.
      // "critical" removed entirely -- it and "serious" were two
      // different reds with no real distinction; LiveBadge (its only
      // caller) now uses "serious" like every other red, one true red.
      colors: {
        page: "var(--bg-app)",
        surface: "var(--bg-card)",
        "surface-raised": "var(--bg-card-raised)",
        inset: "var(--bg-inset)",
        ink: "var(--text-1)",
        "ink-secondary": "var(--text-2)",
        muted: "var(--text-3)",
        hairline: "var(--border-soft)",
        border: "var(--border)",
        "border-soft": "var(--border-soft)",
        accent: "var(--blue)",
        gold: "var(--gold)",
        "gold-dim": "var(--gold-dim)",
        purple: "var(--purple)",
        "purple-dim": "var(--purple-dim)",
        slate: "var(--slate)",
        good: "var(--green)",
        "good-dim": "var(--green-dim)",
        // "warning" keeps its name -- still the right word for the
        // caution states it covers (a conditional pick waiting on price,
        // cold-start/unknown-team risk, insufficient_data) -- only its
        // value moves from the old shared gold/amber hex to the new,
        // dedicated --orange (freed up by the serious/critical merge
        // below; brand/model-data uses now pull from --gold instead).
        warning: "var(--orange)",
        "warning-dim": "var(--orange-dim)",
        serious: "var(--red)",
        "serious-dim": "var(--red-dim)",
      },
      borderColor: {
        DEFAULT: "var(--border)",
      },
    },
  },
  plugins: [],
};
