import Link from "next/link";

const NAV_LINKS = [
  { href: "/", label: "Home" },
  { href: "/about", label: "About" },
  { href: "/cv", label: "CV" },
  { href: "/research", label: "Research" },
] as const;

type SiteNavProps = {
  current?: (typeof NAV_LINKS)[number]["href"];
};

const navStyle: React.CSSProperties = {
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
};

const brandStyle: React.CSSProperties = {
  fontWeight: 700,
  letterSpacing: "0.08em",
  fontSize: 14,
  color: "#cbd5e1",
};

const linkStyle: React.CSSProperties = {
  color: "#93c5fd",
  textDecoration: "none",
  fontSize: 14,
  marginLeft: 20,
};

export default function SiteNav({ current }: SiteNavProps) {
  return (
    <nav style={navStyle} aria-label="Primary">
      <Link href="/" style={{ textDecoration: "none", color: "inherit" }}>
        <span style={brandStyle}>BALMORES - LAB</span>
      </Link>
      <div>
        {NAV_LINKS.map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            style={{
              ...linkStyle,
              ...(current === href ? { color: "#fff", fontWeight: 600 } : {}),
            }}
            aria-current={current === href ? "page" : undefined}
          >
            {label}
          </Link>
        ))}
      </div>
    </nav>
  );
}
