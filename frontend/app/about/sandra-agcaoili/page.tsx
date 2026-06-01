import type { Metadata } from "next";
import Link from "next/link";
import SiteNav from "@/components/SiteNav";
import SandraIdentityLinks from "@/components/SandraIdentityLinks";
import SandraProfileBadges from "@/components/SandraProfileBadges";
import { RESEARCH_ARTICLES } from "@/lib/research-articles";
import {
  buildSandraFaqLd,
  buildSandraGraph,
  SANDRA_AGCAOILI,
  SANDRA_FAQ,
  SANDRA_PROFILE_PATH,
  SANDRA_PROFILE_URL,
  SANDRA_PERSON_ID,
} from "@/lib/research-team";
import { breadcrumbLd, PERSON_NAME, SITE_URL } from "@/lib/seo";

export const metadata: Metadata = {
  title: "Sandra Agcaoili - AI Researcher | Balmores Lab",
  description: SANDRA_AGCAOILI.bio,
  alternates: { canonical: SANDRA_PROFILE_PATH },
  keywords: [
    "Sandra Agcaoili",
    "Sandra Agcaili",
    "Sandra Agcaoli",
    "Sandra Agcaoili AI researcher",
    "Sandra Agcaoili Balmores Lab",
    "Sandra Agcaoili balmoreslab.com",
    "Sandra Agcaoili UP Diliman",
    "AI researcher Singapore",
    "PhD Artificial Intelligence Philippines",
    "Analytics and Artificial Intelligence Association of the Philippines",
    "Balmores Lab Sandra Agcaoili",
    "Sandra Agcaoili Singapore",
    "Sandra Agcaoili animal nutrition",
    "Sandra Agcaoili University of the Philippines",
    "Sandra Agcaoili Zagro",
    "Sandra Agcaoili agriculture",
    "Sandra Agcaoili licensed agriculturist",
    "Sandra Agcaoili PRC",
    "Licensed Agriculturist Philippines",
  ],
  authors: [{ name: SANDRA_AGCAOILI.name, url: SANDRA_PROFILE_URL }],
  openGraph: {
    type: "profile",
    url: SANDRA_PROFILE_URL,
    title: `${SANDRA_AGCAOILI.name} - ${SANDRA_AGCAOILI.jobTitle}`,
    description: SANDRA_AGCAOILI.shortBio,
    firstName: SANDRA_AGCAOILI.givenName,
    lastName: SANDRA_AGCAOILI.familyName,
  } as Metadata["openGraph"],
  twitter: {
    title: `${SANDRA_AGCAOILI.name} - AI Researcher`,
    description: SANDRA_AGCAOILI.shortBio,
  },
  robots: { index: true, follow: true },
};

const S = {
  page: {
    minHeight: "100vh",
    background:
      "radial-gradient(1200px 600px at 90% -10%, rgba(14,165,233,0.12), transparent 60%), radial-gradient(1000px 500px at 0% 100%, rgba(99,102,241,0.10), transparent 60%), #080a0f",
    color: "#e6edf3",
    fontFamily:
      'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  } as React.CSSProperties,
  container: { maxWidth: 960, margin: "0 auto", padding: "40px 24px 80px" } as React.CSSProperties,
  eyebrow: {
    color: "#94a3b8",
    letterSpacing: "0.18em",
    fontSize: 12,
    textTransform: "uppercase",
    marginBottom: 12,
  } as React.CSSProperties,
  h1: { fontSize: 40, lineHeight: 1.1, fontWeight: 700, margin: "0 0 8px", letterSpacing: "-0.02em" } as React.CSSProperties,
  title: { fontSize: 20, color: "#7dd3fc", margin: "0 0 16px", fontWeight: 500 } as React.CSSProperties,
  lede: { fontSize: 17, color: "#cbd5e1", maxWidth: 720, lineHeight: 1.6, margin: "0 0 20px" } as React.CSSProperties,
  section: { padding: "36px 0", borderBottom: "1px solid rgba(255,255,255,0.06)" } as React.CSSProperties,
  h2: {
    fontSize: 13,
    letterSpacing: "0.22em",
    textTransform: "uppercase",
    color: "#60a5fa",
    margin: "0 0 20px",
    fontWeight: 600,
  } as React.CSSProperties,
  h3: { fontSize: 22, margin: "0 0 10px", fontWeight: 600, color: "#f1f5f9" } as React.CSSProperties,
  p: { color: "#cbd5e1", lineHeight: 1.65, fontSize: 16, margin: "0 0 14px" } as React.CSSProperties,
  card: {
    padding: 20,
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 14,
    background: "rgba(255,255,255,0.02)",
    marginBottom: 12,
  } as React.CSSProperties,
  badge: {
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
  } as React.CSSProperties,
  kvRow: {
    display: "grid",
    gridTemplateColumns: "180px 1fr",
    gap: 12,
    padding: "8px 0",
    borderTop: "1px dashed rgba(255,255,255,0.06)",
    fontSize: 15,
  } as React.CSSProperties,
  kvKey: { color: "#94a3b8" } as React.CSSProperties,
  kvVal: { color: "#e2e8f0" } as React.CSSProperties,
  link: { color: "#93c5fd", textDecoration: "underline", textUnderlineOffset: 4 } as React.CSSProperties,
  chip: {
    display: "inline-block",
    padding: "4px 10px",
    margin: "4px 4px 0 0",
    fontSize: 12,
    border: "1px solid rgba(99,102,241,0.45)",
    background: "rgba(99,102,241,0.08)",
    borderRadius: 999,
    color: "#c7d2fe",
  } as React.CSSProperties,
  footer: { padding: "24px 0 0", color: "#64748b", fontSize: 13, textAlign: "center" } as React.CSSProperties,
};

export default function SandraAgcaoiliProfilePage() {
  const crumbs = breadcrumbLd([
    { name: "Home", path: "/" },
    { name: "About", path: "/about" },
    { name: SANDRA_AGCAOILI.name, path: SANDRA_PROFILE_PATH },
  ]);

  return (
    <main style={S.page} itemScope itemType="https://schema.org/ProfilePage">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(buildSandraGraph()) }} />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(buildSandraFaqLd()) }} />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(crumbs) }} />

      <SiteNav current="/about/sandra-agcaoili" />

      <article
        style={S.container}
        className="site-page-shell"
        itemScope
        itemType="https://schema.org/Person"
        itemID={SANDRA_PERSON_ID}
      >
        <meta itemProp="url" content={SANDRA_PROFILE_URL} />
        <meta itemProp="givenName" content={SANDRA_AGCAOILI.givenName} />
        <meta itemProp="familyName" content={SANDRA_AGCAOILI.familyName} />
        <meta itemProp="nationality" content={SANDRA_AGCAOILI.nationality} />

        <p style={{ margin: "0 0 20px", fontSize: 14 }}>
          <Link href="/about" style={S.link}>← About Balmores Lab</Link>
        </p>

        <header style={{ paddingBottom: 20, borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
          <div style={S.eyebrow}>Official Profile · balmoreslab.com</div>
          <h1 style={S.h1}>
            <span itemProp="name">{SANDRA_AGCAOILI.name}</span>
          </h1>
          <SandraIdentityLinks />
          <p style={S.title} itemProp="jobTitle">
            {SANDRA_AGCAOILI.jobTitle} · {SANDRA_AGCAOILI.role} · {SANDRA_AGCAOILI.professionalTitle}
          </p>
          <p className="name-aliases small-muted" style={{ margin: "0 0 12px", fontSize: 14 }}>
            Canonical name: <strong>Sandra Agcaoili</strong> · connected to{" "}
            <a href={SITE_URL} style={S.link}>balmoreslab.com</a>
          </p>
          <p style={S.lede} itemProp="description">
            {SANDRA_AGCAOILI.bio}
          </p>
          <div>
            <span style={S.badge}>AI Research · Balmores Lab</span>
            <span style={S.badge}>UP Diliman · PhD AI</span>
            <span style={S.badge}>AAP Member</span>
            <span style={S.badge}>Singapore</span>
            <span style={S.badge}>Licensed Agriculturist · PRC</span>
            <span style={S.badge}>UP · BS Agriculture</span>
          </div>
          <div style={{ marginTop: 18 }}>
            <SandraProfileBadges align="left" />
          </div>
        </header>

        <section style={S.section} aria-labelledby="glance-h">
          <h2 id="glance-h" style={S.h2}>At a Glance</h2>
          <div style={S.card}>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Full Name</div>
              <div style={S.kvVal}>{SANDRA_AGCAOILI.name}</div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Profession</div>
              <div style={S.kvVal}>
                {SANDRA_AGCAOILI.jobTitle} · {SANDRA_AGCAOILI.role}
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Regulated profession</div>
              <div style={S.kvVal}>{SANDRA_AGCAOILI.professionalTitle}</div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Professional License</div>
              <div style={S.kvVal} itemProp="hasCredential" itemScope itemType="https://schema.org/EducationalOccupationalCredential">
                <span itemProp="name">{SANDRA_AGCAOILI.prcLicense.title}</span>
                {" — "}
                <a
                  href={SANDRA_AGCAOILI.prcLicense.issuerUrl}
                  style={S.link}
                  target="_blank"
                  rel="noopener external"
                  itemProp="recognizedBy"
                  itemScope
                  itemType="https://schema.org/GovernmentOrganization"
                >
                  <span itemProp="name">{SANDRA_AGCAOILI.prcLicense.issuer}</span>
                </a>
                , {SANDRA_AGCAOILI.prcLicense.country}
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Nationality</div>
              <div style={S.kvVal}>Filipino</div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Currently Based</div>
              <div style={S.kvVal} itemProp="homeLocation">
                Singapore
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Research Partner</div>
              <div style={S.kvVal}>
                <Link href="/about" style={S.link}>{PERSON_NAME}</Link>
                {" · "}
                <Link href="/" style={S.link}>Balmores Lab</Link>
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Official URL</div>
              <div style={S.kvVal}>
                <a href={SANDRA_PROFILE_URL} style={S.link}>
                  balmoreslab.com/about/sandra-agcaoili
                </a>
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>LinkedIn</div>
              <div style={S.kvVal}>
                <a
                  href={SANDRA_AGCAOILI.linkedinUrl}
                  style={S.link}
                  target="_blank"
                  rel="me noopener noreferrer"
                  itemProp="sameAs"
                >
                  linkedin.com/in/sandra-agcaoili-a059a2152
                </a>
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>ORCID</div>
              <div style={S.kvVal}>
                <a
                  href={SANDRA_AGCAOILI.orcidUrl}
                  style={S.link}
                  target="_blank"
                  rel="me noopener noreferrer"
                  itemProp="sameAs"
                >
                  {SANDRA_AGCAOILI.orcidId}
                </a>
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>ResearchGate</div>
              <div style={S.kvVal}>
                <a
                  href={SANDRA_AGCAOILI.researchgateUrl}
                  style={S.link}
                  target="_blank"
                  rel="me noopener noreferrer"
                  itemProp="sameAs"
                >
                  researchgate.net/profile/Sandra-Agcaoili
                </a>
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>RocketReach</div>
              <div style={S.kvVal}>
                <a
                  href={SANDRA_AGCAOILI.rocketreachUrl}
                  style={S.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  itemProp="sameAs"
                >
                  rocketreach.co/sandra-agcaoili
                </a>
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>ContactOut</div>
              <div style={S.kvVal}>
                <a
                  href={SANDRA_AGCAOILI.contactoutUrl}
                  style={S.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  itemProp="sameAs"
                >
                  contactout.com/sandra-agcaoili-10976
                </a>
              </div>
            </div>
          </div>
        </section>

        <section style={S.section} aria-labelledby="credentials-h">
          <h2 id="credentials-h" style={S.h2}>Licenses &amp; Credentials</h2>
          <div style={S.card} itemProp="hasCredential" itemScope itemType="https://schema.org/EducationalOccupationalCredential">
            <div style={{ color: "#7dd3fc", fontSize: 12, letterSpacing: "0.12em", textTransform: "uppercase" }}>
              Professional license
            </div>
            <h3 style={S.h3}>
              <span itemProp="name">{SANDRA_AGCAOILI.prcLicense.title}</span>
            </h3>
            <p style={S.p}>{SANDRA_AGCAOILI.prcLicense.description}</p>
            <p style={S.p}>
              Issued by{" "}
              <a href={SANDRA_AGCAOILI.prcLicense.issuerUrl} style={S.link} target="_blank" rel="noopener external">
                {SANDRA_AGCAOILI.prcLicense.issuer}
              </a>{" "}
              (PRC), Republic of the Philippines.
            </p>
          </div>
        </section>

        <section style={S.section} aria-labelledby="skills-h">
          <h2 id="skills-h" style={S.h2}>Core Competencies</h2>
          <div style={{ marginTop: 8 }}>
            {SANDRA_AGCAOILI.skills.map((skill) => (
              <span key={skill} style={S.chip} itemProp="knowsAbout">
                {skill}
              </span>
            ))}
          </div>
        </section>

        <section style={S.section} aria-labelledby="summary-h">
          <h2 id="summary-h" style={S.h2}>Career Summary</h2>
          <p style={S.p}>{SANDRA_AGCAOILI.careerSummary}</p>
        </section>

        <section style={S.section} aria-labelledby="experience-h">
          <h2 id="experience-h" style={S.h2}>Professional Experience</h2>
          {SANDRA_AGCAOILI.workHistory.map((role) => (
            <div
              key={`${role.organization}-${role.title}`}
              style={S.card}
              itemProp="hasOccupation"
              itemScope
              itemType="https://schema.org/Occupation"
            >
              <div style={{ color: "#7dd3fc", fontSize: 12, letterSpacing: "0.12em", textTransform: "uppercase" }}>
                {role.current ? "Current role" : "Previous role"}
              </div>
              <h3 style={S.h3}>
                <span itemProp="name">{role.title}</span>
              </h3>
              <p style={S.p}>
                <span itemProp="occupationLocation" itemScope itemType="https://schema.org/Place">
                  <span itemProp="name">{role.location}</span>
                </span>
                {" · "}
                <span itemProp="worksFor" itemScope itemType="https://schema.org/Organization">
                  <span itemProp="name">{role.organization}</span>
                </span>
              </p>
              <div style={S.kvRow}>
                <div style={S.kvKey}>Period</div>
                <div style={S.kvVal}>
                  {role.current
                    ? "Present — Research Partner, Balmores Lab"
                    : `${role.startYear} – ${role.endYear}`}
                </div>
              </div>
            </div>
          ))}
        </section>

        <section style={S.section} aria-labelledby="education-h">
          <h2 id="education-h" style={S.h2}>Education</h2>
          {SANDRA_AGCAOILI.education.map((edu) => (
            <div
              key={`${edu.institution}-${edu.degree}`}
              style={S.card}
              itemProp="alumniOf"
              itemScope
              itemType="https://schema.org/EducationalOrganization"
            >
              <div style={{ color: "#7dd3fc", fontSize: 12, letterSpacing: "0.12em", textTransform: "uppercase" }}>
                {edu.status}
              </div>
              <h3 style={S.h3}>
                <span itemProp="name">
                  {edu.degree} — {edu.field}
                </span>
              </h3>
              <p style={S.p}>
                {edu.institutionUrl ? (
                  <a href={edu.institutionUrl} style={S.link} rel="noopener external" target="_blank" itemProp="url">
                    {edu.institution}
                  </a>
                ) : (
                  <span itemProp="name">{edu.institution}</span>
                )}
              </p>
              <div style={S.kvRow}>
                <div style={S.kvKey}>Years</div>
                <div style={S.kvVal}>
                  {edu.endYear ? `${edu.startYear} – ${edu.endYear}` : `${edu.startYear} – Present`}
                </div>
              </div>
              {edu.status === "In progress" ? (
                <div style={S.kvRow}>
                  <div style={S.kvKey}>Activities &amp; Societies</div>
                  <div style={S.kvVal}>
                    Analytics and Artificial Intelligence Association of the Philippines (AAP)
                  </div>
                </div>
              ) : null}
            </div>
          ))}
        </section>

        <section style={S.section} aria-labelledby="research-h">
          <h2 id="research-h" style={S.h2}>Research at Balmores Lab</h2>
          <p style={S.p}>{SANDRA_AGCAOILI.collaborationFocus}</p>
          <h3 style={{ ...S.h3, fontSize: 16, marginTop: 20 }}>Focus areas</h3>
          <div style={{ marginTop: 8 }}>
            {SANDRA_AGCAOILI.researchAreas.map((area) => (
              <span key={area} style={S.chip} itemProp="knowsAbout">
                {area}
              </span>
            ))}
          </div>
        </section>

        <section style={S.section} aria-labelledby="papers-h">
          <h2 id="papers-h" style={S.h2}>Co-Authored Working Papers</h2>
          {RESEARCH_ARTICLES.slice(0, 4).map((a) => (
            <div key={a.slug} style={S.card}>
              <Link href={`/research/${a.slug}`} style={{ ...S.link, fontWeight: 600, fontSize: 15 }}>
                {a.headline}
              </Link>
              <p style={{ ...S.p, marginTop: 8, fontSize: 14, color: "#94a3b8" }}>
                with {PERSON_NAME} ·{" "}
                <time dateTime={a.datePublished}>
                  {new Date(a.datePublished).toLocaleDateString("en-US", { year: "numeric", month: "long" })}
                </time>
              </p>
            </div>
          ))}
          <p style={S.p}>
            <Link href="/research" style={S.link}>View full research programme →</Link>
          </p>
        </section>

        <section style={S.section} aria-labelledby="faq-h">
          <h2 id="faq-h" style={S.h2}>Frequently Asked Questions</h2>
          {SANDRA_FAQ.map((item) => (
            <details key={item.question} style={{ ...S.card, marginBottom: 12 }}>
              <summary style={{ ...S.h3, cursor: "pointer", fontSize: 17 }}>{item.question}</summary>
              <p style={{ ...S.p, marginTop: 12 }}>{item.answer}</p>
            </details>
          ))}
        </section>

        <footer style={S.footer}>
          (c) {new Date().getFullYear()} {SANDRA_AGCAOILI.name} · Research Partner,{" "}
          <Link href="/" style={S.link}>Balmores Lab</Link>
        </footer>
      </article>
    </main>
  );
}
