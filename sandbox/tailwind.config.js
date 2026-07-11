/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
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
