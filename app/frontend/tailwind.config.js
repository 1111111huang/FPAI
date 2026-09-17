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
      colors: {
        page: "var(--page-plane)",
        surface: "var(--surface-1)",
        ink: "var(--text-primary)",
        "ink-secondary": "var(--text-secondary)",
        muted: "var(--text-muted)",
        hairline: "var(--gridline)",
        border: "var(--border-hairline)",
        "border-strong": "var(--border-hairline-strong)",
        accent: "var(--accent)",
        good: "var(--status-good)",
        warning: "var(--status-warning)",
        serious: "var(--status-serious)",
        critical: "var(--status-critical)",
      },
      borderColor: {
        DEFAULT: "var(--border-hairline)",
      },
    },
  },
  plugins: [],
};
