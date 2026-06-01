import Link from "next/link";
import SandraIdentityLinks from "@/components/SandraIdentityLinks";
import { formatSandraSocieties, SANDRA_AGCAOILI } from "@/lib/research-team";

const cardStyle: React.CSSProperties = {
  padding: 20,
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 14,
  background: "rgba(255,255,255,0.02)",
};

const badgeStyle: React.CSSProperties = {
  display: "inline-block",
  padding: "4px 10px",
  fontSize: 11,
  letterSpacing: "0.1em",
  textTransform: "uppercase",
  borderRadius: 999,
  border: "1px solid rgba(14,165,233,0.4)",
  background: "rgba(14,165,233,0.08)",
  color: "#7dd3fc",
  marginRight: 8,
  marginBottom: 8,
};

type ResearchPartnerCardProps = {
  compact?: boolean;
};

export default function ResearchPartnerCard({ compact = false }: ResearchPartnerCardProps) {
  const p = SANDRA_AGCAOILI;

  return (
    <div
      style={cardStyle}
      itemScope
      itemType="https://schema.org/Person"
      itemID={p.id}
    >
      <meta itemProp="givenName" content={p.givenName} />
      <meta itemProp="familyName" content={p.familyName} />
      <span
        itemProp="homeLocation"
        itemScope
        itemType="https://schema.org/Place"
        style={{ display: "none" }}
      >
        <meta itemProp="name" content={p.location} />
      </span>

      <div style={{ color: "#7dd3fc", fontSize: 12, letterSpacing: "0.12em", textTransform: "uppercase" }}>
        Research Partner
      </div>
      <h3
        style={{
          margin: "8px 0 4px",
          fontSize: compact ? 18 : 22,
          fontWeight: 600,
          color: "#f1f5f9",
        }}
      >
        <Link
          href={p.profilePath}
          style={{ color: "inherit", textDecoration: "none" }}
          itemProp="url"
        >
          <span itemProp="name">{p.name}</span>
        </Link>
      </h3>
      <SandraIdentityLinks compact={compact} />
      <p style={{ margin: "8px 0 12px", color: "#a5b4fc", fontSize: compact ? 15 : 17 }} itemProp="jobTitle">
        {p.jobTitle}
      </p>

      <div style={{ marginBottom: compact ? 10 : 14 }}>
        <span style={badgeStyle}>UP Diliman · PhD AI</span>
        <span style={badgeStyle}>AAP Member</span>
        <span style={badgeStyle}>Based in Singapore</span>
      </div>

      <p
        style={{
          color: "#cbd5e1",
          lineHeight: 1.65,
          fontSize: compact ? 14 : 16,
          margin: "0 0 12px",
        }}
        itemProp="description"
      >
        {compact ? p.shortBio : p.bio}
      </p>

      {!compact ? (
        <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "160px 1fr",
              gap: 10,
              padding: "8px 0",
              borderTop: "1px dashed rgba(255,255,255,0.06)",
              fontSize: 14,
            }}
          >
            <div style={{ color: "#94a3b8" }}>Education</div>
            <div style={{ color: "#e2e8f0" }} itemProp="alumniOf" itemScope itemType="https://schema.org/EducationalOrganization">
              <span itemProp="name">{p.credential.institution}</span>
              <br />
              <span itemProp="department">{p.credential.name}</span>
              <br />
              <time dateTime={p.credential.startDate}>Sep 2023</time> – Present
            </div>
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "160px 1fr",
              gap: 10,
              padding: "8px 0",
              borderTop: "1px dashed rgba(255,255,255,0.06)",
              fontSize: 14,
            }}
          >
            <div style={{ color: "#94a3b8" }}>Societies</div>
            <div style={{ color: "#e2e8f0" }}>{formatSandraSocieties(p.societies)}</div>
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "160px 1fr",
              gap: 10,
              padding: "8px 0",
              borderTop: "1px dashed rgba(255,255,255,0.06)",
              fontSize: 14,
            }}
          >
            <div style={{ color: "#94a3b8" }}>Collaboration</div>
            <div style={{ color: "#e2e8f0" }}>{p.collaborationFocus}</div>
          </div>
        </>
      ) : null}

      {!compact ? (
        <p style={{ margin: "12px 0 0", fontSize: 14 }}>
          <Link
            href={p.profilePath}
            style={{ color: "#93c5fd", textDecoration: "underline", textUnderlineOffset: 3 }}
          >
            View full profile →
          </Link>
        </p>
      ) : (
        <p style={{ margin: "8px 0 0", fontSize: 13 }}>
          <Link href={p.profilePath} style={{ color: "#93c5fd" }}>
            Full profile →
          </Link>
        </p>
      )}
    </div>
  );
}

export function ResearchAuthorsLine({ linkStyle }: { linkStyle: React.CSSProperties }) {
  return (
    <>
      <span itemProp="author" itemScope itemType="https://schema.org/Person">
        <span itemProp="name">Louie Doniego Balmores</span>
      </span>
      {" & "}
      <span itemProp="author" itemScope itemType="https://schema.org/Person">
        <span itemProp="name">{SANDRA_AGCAOILI.name}</span>
      </span>
    </>
  );
}
