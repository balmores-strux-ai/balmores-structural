import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "BALMORES STRUCTURAL",
  description: "5000-model structural analysis workspace — natural language to 3D frame, instant predictions, ETABS-trained brain.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
