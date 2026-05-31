"use client";

import Link from "next/link";
import { useState } from "react";
import BalmoresLogo from "@/components/BalmoresLogo";

const NAV_LINKS = [
  { href: "/", label: "Home" },
  { href: "/about", label: "About" },
  { href: "/about/sandra-agcaoili", label: "Sandra" },
  { href: "/cv", label: "CV" },
  { href: "/research", label: "Research" },
] as const;

type SiteNavProps = {
  current?: (typeof NAV_LINKS)[number]["href"];
};

export default function SiteNav({ current }: SiteNavProps) {
  const [open, setOpen] = useState(false);

  return (
    <nav
      className="site-nav"
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "16px 24px",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        backdropFilter: "blur(8px)",
        background: "rgba(8,10,15,0.6)",
        position: "sticky",
        top: 0,
        zIndex: 10,
      }}
      aria-label="Primary"
    >
      <div
        className="site-nav-inner"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          width: "100%",
          gap: 12,
        }}
      >
        <BalmoresLogo variant="nav" />
        <button
          type="button"
          className="site-nav-toggle"
          aria-expanded={open}
          aria-controls="site-nav-menu"
          onClick={() => setOpen((v) => !v)}
        >
          {open ? "Close" : "Menu"}
        </button>
        <div id="site-nav-menu" className={`site-nav-links${open ? " is-open" : ""}`}>
          {NAV_LINKS.map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              className="site-nav-link"
              style={{
                color: "#93c5fd",
                textDecoration: "none",
                fontSize: 14,
                marginLeft: 20,
              }}
              aria-current={current === href ? "page" : undefined}
              onClick={() => setOpen(false)}
            >
              <span style={current === href ? { color: "#fff", fontWeight: 600 } : undefined}>
                {label}
              </span>
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
