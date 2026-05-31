import Link from "next/link";

import { SANDRA_AGCAOILI, SANDRA_PROFILE_URL } from "@/lib/research-team";

type SandraProfileBadgesProps = {
  align?: "left" | "center";
  compact?: boolean;
};

const BADGES = [
  {
    id: "profile",
    label: "Official profile",
    href: SANDRA_PROFILE_URL,
    external: false,
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
}: SandraProfileBadgesProps) {
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: compact ? 6 : 10,
        justifyContent: align === "center" ? "center" : "flex-start",
      }}
      aria-label={`Verified profiles for ${SANDRA_AGCAOILI.name}`}
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
  );
}
