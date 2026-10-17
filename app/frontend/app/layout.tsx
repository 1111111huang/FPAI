import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "Oddsey",
  description: "Oddsey betting agent web app.",
  manifest: "/manifest.json",
  icons: {
    icon: "/icons/icon-192.png",
    apple: "/icons/icon-192.png",
  },
};

export const viewport: Viewport = {
  themeColor: "#0a0a0d",
  // Found live (2026-10-17): missing entirely, so iOS never rendered the
  // page edge-to-edge in standalone PWA mode -- every env(safe-area-inset-*)
  // used throughout the app (e.g. AppShell.tsx's bottom tab bar) resolved
  // to 0 instead of the real notch/home-indicator inset, and the viewport
  // WebKit laid out against didn't actually match the real screen. The
  // bottom tab bar's fixed positioning visibly glitching mid-scroll
  // (detaching and floating over mid-page content instead of staying
  // pinned to the true viewport bottom) traces back to this same gap.
  viewportFit: "cover",
};

// Direct user spec (2026-09-17): Inter (400-800, the default for
// everything -- headings, body, labels, buttons, the wordmark) and IBM
// Plex Mono (400-600, numbers only -- scores/odds/stakes/percentages/
// edge values/probability bars). next/font/google self-hosts both at
// build time (no runtime CDN request, no FOUT) -- same mechanism AppShell's
// wordmark already used for Montserrat before this (removed there, see
// AppShell.tsx; Inter is now the global default so a separate per-
// component font call for the wordmark is no longer needed). Inter is a
// variable font (no `weight` prop needed -- the full variable axis is
// loaded and comfortably covers 400-800); IBM Plex Mono ships only static
// weights, so `weight` is required and explicitly scoped to 400-600.
const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const plexMono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-plex-mono" });

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${plexMono.variable}`}>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
