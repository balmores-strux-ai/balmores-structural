import Link from "next/link";

import {
  SANDRA_AGCAOILI,
  SANDRA_LINKEDIN_URL,
  SANDRA_ORCID_ID,
  SANDRA_ORCID_URL,
  SANDRA_PROFILE_URL,
  SANDRA_RESEARCHGATE_URL,
  SANDRA_ROCKETREACH_URL,
} from "@/lib/research-team";

type SandraProfileBadgesProps = {
  align?: "left" | "center";
  compact?: boolean;
  showOrcidPill?: boolean;
};

const BADGES = [
  {
    id: "profile",
    label: "Official profile",
    href: SANDRA_PROFILE_URL,
    external: false,
  },
  {
    id: "linkedin",
    label: "LinkedIn",
    href: SANDRA_LINKEDIN_URL,
    external: true,
  },
  {
    id: "researchgate",
    label: "ResearchGate",
    href: SANDRA_RESEARCHGATE_URL,
    external: true,
  },
  {
    id: "rocketreach",
    label: "RocketReach",
    href: SANDRA_ROCKETREACH_URL,
    external: true,
  },
  {
    id: "upd",
    label: "UP Diliman",
    href: "https://upd.edu.ph",
    external: true,
  },
  {
    id: "research",
    label: "Balmores Lab research",
    href: "/research",
    external: false,
  },
] as const;

export default function SandraProfileBadges({
  align = "center",
  compact = false,
  showOrcidPill = true,
}: SandraProfileBadgesProps) {
  const rowJustify = align === "center" ? "center" : "flex-start";

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: align === "center" ? "center" : "flex-start",
        gap: compact ? 10 : 14,
      }}
      aria-label={`Verified profiles for ${SANDRA_AGCAOILI.name}`}
    >
      {showOrcidPill ? (
        <a
          href={SANDRA_ORCID_URL}
          target="_blank"
          rel="me noopener noreferrer"
          itemProp="sameAs"
          className="orcid-pill"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            padding: compact ? "5px 10px" : "6px 12px",
            borderRadius: 999,
            border: "1px solid rgba(165, 180, 252, 0.35)",
            background: "rgba(255,255,255,0.04)",
            color: "#e2e8f0",
            fontSize: compact ? 12 : 13,
            textDecoration: "none",
            lineHeight: 1,
            whiteSpace: "nowrap",
          }}
          aria-label={`ORCID iD ${SANDRA_ORCID_ID}`}
          title="Open ORCID record"
        >
          <img
            src="https://orcid.org/sites/default/files/images/orcid_16x16.png"
            alt="ORCID iD icon"
            width={16}
            height={16}
            style={{ display: "block" }}
          />
          <span style={{ color: "#a5b4fc", fontWeight: 600, letterSpacing: "0.02em" }}>
            ORCID
          </span>
          <span className="orcid-url-full" style={{ color: "#cbd5e1" }}>
            {SANDRA_ORCID_URL}
          </span>
          <span className="orcid-url-short" style={{ color: "#cbd5e1" }}>
            {SANDRA_ORCID_ID}
          </span>
        </a>
      ) : null}

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: compact ? 6 : 10,
          justifyContent: rowJustify,
        }}
      >
        {BADGES.map((badge) => (
          <Link
            key={badge.id}
            href={badge.href}
            target={badge.external ? "_blank" : undefined}
            rel={badge.external ? "me noopener noreferrer" : undefined}
            itemProp="sameAs"
            style={{
              display: "inline-flex",
              alignItems: "center",
              padding: compact ? "5px 10px" : "6px 12px",
              borderRadius: 999,
              border: "1px solid rgba(165, 180, 252, 0.35)",
              background: "rgba(255,255,255,0.04)",
              color: "#e2e8f0",
              fontSize: compact ? 12 : 13,
              textDecoration: "none",
            }}
          >
            {badge.label}
          </Link>
        ))}
      </div>
    </div>
  );
}
