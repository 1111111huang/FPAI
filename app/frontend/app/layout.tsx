import type { Metadata } from "next";
import "./globals.css";
import { StatusFooter } from "@/components/StatusFooter";

export const metadata: Metadata = {
  title: "FPAI",
  description: "FPAI betting agent web app.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        {children}
        <StatusFooter />
      </body>
    </html>
  );
}
