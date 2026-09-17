import type { Metadata } from "next";
import { IBM_Plex_Mono, Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "Oddsey",
  description: "Oddsey betting agent web app.",
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
