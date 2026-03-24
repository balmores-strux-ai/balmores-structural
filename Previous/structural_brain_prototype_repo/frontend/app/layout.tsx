import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Structural Brain Prototype',
  description: 'Chat-first structural analysis workspace prototype'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
