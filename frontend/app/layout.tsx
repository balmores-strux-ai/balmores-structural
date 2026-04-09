import "./globals.css";
import type { Metadata, Viewport } from "next";

export const metadata: Metadata = {
  title: "BALMORES STRUCTURAL",
  description:
    "Parametric 3D building frame — PyNite finite element analysis in the browser, plus an optional neural assistant.",
  openGraph: {
    title: "BALMORES STRUCTURAL",
    description:
      "PyNite FEA for regular grids — spans, loads, steel sections — with 3D preview and force envelopes.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "BALMORES STRUCTURAL",
    description:
      "PyNite FEA for parametric frames — forces, deflections, reactions — with optional AI assistant.",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#080a0f",
  colorScheme: "dark",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
