import React from "react";

const ORCID_ID = "0009-0008-5479-4033";
const ORCID_URL = `https://orcid.org/${ORCID_ID}`;

type Profile = {
  id: string;
  label: string;
  href: string;
  hoverColor: string;
  svg: React.ReactNode;
  /**
   * When true the badge is kept in the DOM (so search engines still see the
   * rel=me / schema.org sameAs link) but is visually hidden from users and
   * screen readers. Used for identity-graph platforms whose public badge is
   * off-topic for visitors.
   */
  hidden?: boolean;
};

const Ico = {
  linkedin: (
    <svg viewBox="0 0 24 24" aria-hidden="true" width="18" height="18">
      <path
        fill="currentColor"
        d="M4.98 3.5C4.98 4.88 3.87 6 2.5 6S0 4.88 0 3.5 1.12 1 2.5 1s2.48 1.12 2.48 2.5zM.2 8h4.6v14H.2V8zm7.4 0h4.4v2h.1c.6-1.1 2.1-2.3 4.3-2.3 4.6 0 5.5 3 5.5 6.9V22h-4.6v-6.4c0-1.5-.03-3.5-2.12-3.5-2.13 0-2.46 1.7-2.46 3.4V22H7.6V8z"
      />
    </svg>
  ),
  x: (
    <svg viewBox="0 0 24 24" aria-hidden="true" width="16" height="16">
      <path
        fill="currentColor"
        d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"
      />
    </svg>
  ),
  github: (
    <svg viewBox="0 0 24 24" aria-hidden="true" width="18" height="18">
      <path
        fill="currentColor"
        d="M12 .5C5.65.5.5 5.65.5 12.02c0 5.1 3.29 9.42 7.86 10.95.58.1.8-.25.8-.56v-1.96c-3.2.7-3.87-1.37-3.87-1.37-.52-1.33-1.28-1.68-1.28-1.68-1.05-.72.08-.71.08-.71 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.74.4-1.26.73-1.55-2.55-.29-5.23-1.28-5.23-5.68 0-1.25.44-2.28 1.17-3.08-.12-.29-.51-1.46.11-3.04 0 0 .96-.31 3.15 1.18a10.86 10.86 0 0 1 5.73 0c2.19-1.49 3.15-1.18 3.15-1.18.62 1.58.23 2.75.11 3.04.73.8 1.17 1.83 1.17 3.08 0 4.41-2.69 5.38-5.25 5.67.41.36.78 1.07.78 2.16v3.2c0 .31.21.67.8.56A11.53 11.53 0 0 0 23.5 12.02C23.5 5.65 18.35.5 12 .5z"
      />
    </svg>
  ),
  aboutme: (
    <svg viewBox="0 0 24 24" aria-hidden="true" width="18" height="18">
      <circle cx="12" cy="8" r="3.4" fill="currentColor" />
      <path fill="currentColor" d="M4.5 20c0-4.15 3.36-7 7.5-7s7.5 2.85 7.5 7v1H4.5v-1z" />
    </svg>
  ),
  chess: (
    <svg viewBox="0 0 24 24" aria-hidden="true" width="18" height="18">
      <path
        fill="currentColor"
        d="M8 2h8v2h-2v3h2l2 5h-3v6h2v2H7v-2h2v-6H6l2-5h2V4H8V2zm2 6l-1 3h6l-1-3h-4z"
      />
    </svg>
  ),
  wikidata: (
    <svg viewBox="0 0 24 24" aria-hidden="true" width="22" height="14">
      <rect x="1" y="6" width="2" height="12" fill="currentColor" />
      <rect x="5" y="6" width="2" height="12" fill="currentColor" />
      <rect x="9" y="6" width="2" height="12" fill="currentColor" />
      <rect x="13" y="6" width="2" height="12" fill="currentColor" />
      <rect x="17" y="6" width="2" height="12" fill="currentColor" />
      <rect x="21" y="6" width="2" height="12" fill="currentColor" />
    </svg>
  ),
  prc: (
    <svg viewBox="0 0 24 24" aria-hidden="true" width="18" height="18">
      <path
        fill="currentColor"
        d="M12 2 3 6v6c0 5 3.8 9.4 9 10 5.2-.6 9-5 9-10V6l-9-4zm-1 14-4-4 1.4-1.4L11 13.2l5.6-5.6L18 9l-7 7z"
      />
    </svg>
  ),
  rss: (
    <svg viewBox="0 0 24 24" aria-hidden="true" width="18" height="18">
      <path
        fill="currentColor"
        d="M6.18 15.64a2.18 2.18 0 1 1 0 4.36 2.18 2.18 0 0 1 0-4.36zM4 4.44v3.22c7.69 0 13.9 6.21 13.9 13.9h3.22c0-9.47-7.65-17.12-17.12-17.12zm0 5.87v3.23c4.44 0 8.03 3.59 8.03 8.03h3.23c0-6.22-5.04-11.26-11.26-11.26z"
      />
    </svg>
  ),
};

const PROFILES: Profile[] = [
  {
    id: "linkedin",
    label: "LinkedIn",
    href: "https://www.linkedin.com/in/louiebalmores/",
    hoverColor: "#0a66c2",
    svg: Ico.linkedin,
  },
  {
    id: "x",
    label: "X (Twitter)",
    href: "https://x.com/louiedbalmores",
    hoverColor: "#ffffff",
    svg: Ico.x,
  },
  {
    id: "github",
    label: "GitHub",
    href: "https://github.com/balmores-strux-ai/balmores-structural",
    hoverColor: "#ffffff",
    svg: Ico.github,
  },
  {
    id: "aboutme",
    label: "about.me",
    href: "https://about.me/louiebalmoresdesign/",
    hoverColor: "#22d3ee",
    svg: Ico.aboutme,
  },
  {
    id: "chess",
    label: "WorldChess",
    href: "https://worldchess.com/profile/422673",
    hoverColor: "#d4d4d4",
    svg: Ico.chess,
    hidden: true,
  },
  {
    id: "wikidata",
    label: "Wikidata",
    href: "https://www.wikidata.org/wiki/Q139544451",
    hoverColor: "#c0c6cc",
    svg: Ico.wikidata,
  },
  {
    id: "prc",
    label: "PRC (Philippines)",
    href: "https://www.prc.gov.ph/uploaded/documents/CE1113se.pdf",
    hoverColor: "#60a5fa",
    svg: Ico.prc,
  },
  {
    id: "rss",
    label: "RSS feed",
    href: "/feed.xml",
    hoverColor: "#f59e0b",
    svg: Ico.rss,
  },
];

type ProfileBadgesProps = {
  align?: "left" | "center";
  showOrcidPill?: boolean;
  compact?: boolean;
};

export default function ProfileBadges({
  align = "center",
  showOrcidPill = true,
  compact = false,
}: ProfileBadgesProps) {
  const rowJustify =
    align === "center" ? "center" : "flex-start";

  return (
    <div
      className="profile-badges"
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: align === "center" ? "center" : "flex-start",
        gap: compact ? 10 : 14,
        padding: compact ? "0" : "4px 0 2px",
      }}
    >
      {showOrcidPill ? (
        <a
          id="cy-effective-orcid-url"
          href={ORCID_URL}
          target="orcid.widget"
          rel="me noopener noreferrer"
          itemProp="sameAs"
          className="orcid-pill"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            padding: "6px 12px",
            borderRadius: 999,
            border: "1px solid rgba(165, 180, 252, 0.35)",
            background: "rgba(255,255,255,0.04)",
            color: "#e2e8f0",
            fontSize: 13,
            textDecoration: "none",
            lineHeight: 1,
            whiteSpace: "nowrap",
          }}
          aria-label={`ORCID iD ${ORCID_ID}`}
          title="Open ORCID record"
        >
          {}
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
            {ORCID_URL}
          </span>
          <span className="orcid-url-short" style={{ color: "#cbd5e1" }}>
            {ORCID_ID}
          </span>
        </a>
      ) : null}

      <ul
        className="profile-badges__row"
        style={{
          listStyle: "none",
          margin: 0,
          padding: 0,
          display: "flex",
          flexWrap: "wrap",
          justifyContent: rowJustify,
          gap: compact ? 6 : 10,
        }}
      >
        {PROFILES.filter((p) => !p.hidden).map((p) => (
          <li key={p.id} style={{ display: "inline-flex" }}>
            <a
              className="profile-badges__link"
              href={p.href}
              target={p.id === "rss" ? undefined : "_blank"}
              rel={p.id === "rss" ? undefined : "me noopener noreferrer"}
              itemProp={p.id === "rss" || p.id === "prc" ? undefined : "sameAs"}
              aria-label={p.label}
              title={p.label}
              style={{
                ["--pb-hover" as unknown as string]: p.hoverColor,
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                width: compact ? 30 : 34,
                height: compact ? 30 : 34,
                borderRadius: 10,
                border: "1px solid rgba(255,255,255,0.10)",
                background: "rgba(255,255,255,0.03)",
                color: "#cbd5e1",
                textDecoration: "none",
                transition:
                  "transform 0.15s ease, color 0.15s ease, border-color 0.15s ease, background 0.15s ease",
              } as React.CSSProperties}
            >
              {p.svg}
            </a>
          </li>
        ))}
      </ul>

      {/* Identity-graph links kept in the DOM for crawlers (schema.org sameAs
          + rel=me) but visually hidden so they don't clutter the UI. Google,
          Bing and Mastodon verifiers still read these. */}
      {PROFILES.some((p) => p.hidden) ? (
        <ul className="profile-badges__seo" aria-hidden="true">
          {PROFILES.filter((p) => p.hidden).map((p) => (
            <li key={p.id}>
              <a
                href={p.href}
                rel="me noopener noreferrer"
                itemProp="sameAs"
                tabIndex={-1}
              >
                {p.label}
              </a>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
