import {
  SANDRA_AGCAOILI,
  SANDRA_LINKEDIN_URL,
  SANDRA_ORCID_ID,
  SANDRA_ORCID_URL,
  SANDRA_RESEARCHGATE_URL,
} from "@/lib/research-team";

type SandraIdentityLinksProps = {
  compact?: boolean;
  inline?: boolean;
  style?: React.CSSProperties;
};

const linkStyle: React.CSSProperties = {
  color: "#93c5fd",
  textDecoration: "underline",
  textUnderlineOffset: 3,
};

export default function SandraIdentityLinks({
  compact = false,
  inline = false,
  style,
}: SandraIdentityLinksProps) {
  const Tag = inline ? "span" : "p";

  return (
    <Tag
      style={{
        margin: inline ? 0 : compact ? "4px 0 0" : "6px 0 0",
        fontSize: compact ? 13 : 14,
        color: "#94a3b8",
        display: inline ? "inline" : undefined,
        ...style,
      }}
      aria-label={inline ? undefined : `Professional profiles for ${SANDRA_AGCAOILI.name}`}
    >
      <a
        href={SANDRA_LINKEDIN_URL}
        target="_blank"
        rel="me noopener noreferrer"
        itemProp="sameAs"
        style={linkStyle}
      >
        LinkedIn
      </a>
      {" · "}
      <a
        href={SANDRA_ORCID_URL}
        target="_blank"
        rel="me noopener noreferrer"
        itemProp="sameAs"
        style={linkStyle}
      >
        ORCID
      </a>
      {" · "}
      <a
        href={SANDRA_RESEARCHGATE_URL}
        target="_blank"
        rel="me noopener noreferrer"
        itemProp="sameAs"
        style={linkStyle}
      >
        ResearchGate
      </a>
      {!compact && !inline ? (
        <span style={{ color: "#64748b" }}> ({SANDRA_ORCID_ID})</span>
      ) : null}
    </Tag>
  );
}
