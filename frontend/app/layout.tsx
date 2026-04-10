import "./globals.css";
import type { Metadata, Viewport } from "next";

export const metadata: Metadata = {
  title: "BALMORES STRUCTURAL",
  description:
    "Natural-language structural brief to PyNite 3D frame analysis — reactions, beam and column envelopes, drift, P-Δ.",
  openGraph: {
    title: "BALMORES STRUCTURAL",
    description:
      "Describe spans and loads in plain English; get PyNite FEM results, storey drift, and support reactions.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "BALMORES STRUCTURAL",
    description:
      "PyNite 3D FEA from a design brief — irregular grids, DL/LL, wind, simplified seismic, P-Δ.",
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
