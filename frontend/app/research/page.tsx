import type { Metadata } from "next";
import Link from "next/link";
import ProfileBadges from "@/components/ProfileBadges";
import ResearchPartnerCard from "@/components/ResearchPartnerCard";
import SiteNav from "@/components/SiteNav";
import { RESEARCH_ARTICLES } from "@/lib/research-articles";
import { SANDRA_AGCAOILI } from "@/lib/research-team";
import { breadcrumbLd, JOB_TITLE, PERSON_NAME, SITE_URL } from "@/lib/seo";

export const metadata: Metadata = {
  title: "Research - AI-Driven Structural Engineering",
  description:
    "Research programme at Balmores Lab by Louie Doniego Balmores and Sandra Agcaoili: AI for structural integrity, computational design, material efficiency, physics-informed neural surrogates, and privacy-preserving on-device FEM loops.",
  alternates: { canonical: "/research" },
  keywords: [
    "structural AI research",
    "Louie Balmores research",
    "Sandra Agcaoili AI researcher",
    "PyNite",
    "physics-informed neural networks",
    "structural optimization",
    "computational design",
  ],
  authors: [
    { name: PERSON_NAME, url: `${SITE_URL}/about` },
    { name: SANDRA_AGCAOILI.name },
  ],
  openGraph: {
    type: "website",
    url: `${SITE_URL}/research`,
    title: "Research - Balmores Lab | Louie Doniego Balmores",
    description:
      "AI-driven structural optimization, NL-to-FEM pipelines, and physics-informed surrogates.",
  },
  twitter: {
    title: "Research - Balmores Lab",
    description:
      "AI-driven structural engineering research by Louie Doniego Balmores.",
  },
};

const S = {
  page: {
    minHeight: "100vh",
    background: "#080a0f",
    color: "#e6edf3",
    fontFamily:
      'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  } as React.CSSProperties,
  container: { maxWidth: 940, margin: "0 auto", padding: "40px 28px 80px" } as React.CSSProperties,
  h1: { fontSize: 40, margin: "0 0 8px", letterSpacing: "-0.02em", fontWeight: 700 } as React.CSSProperties,
  sub: { color: "#a5b4fc", margin: 0, fontSize: 18 } as React.CSSProperties,
  lede: { color: "#cbd5e1", fontSize: 16, lineHeight: 1.6, marginTop: 14, maxWidth: 720 } as React.CSSProperties,
  h2: {
    fontSize: 13,
    letterSpacing: "0.22em",
    textTransform: "uppercase",
    color: "#60a5fa",
    margin: "40px 0 14px",
    fontWeight: 600,
  } as React.CSSProperties,
  article: {
    padding: 22,
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 14,
    background: "rgba(255,255,255,0.02)",
    marginBottom: 16,
  } as React.CSSProperties,
  headline: { margin: 0, fontSize: 20, color: "#f1f5f9", fontWeight: 600, lineHeight: 1.3 } as React.CSSProperties,
  meta: { color: "#94a3b8", fontSize: 13, marginTop: 8 } as React.CSSProperties,
  abstract: { color: "#cbd5e1", lineHeight: 1.65, fontSize: 15, margin: "12px 0 0" } as React.CSSProperties,
  keywords: { marginTop: 12, display: "flex", flexWrap: "wrap", gap: 6 } as React.CSSProperties,
  chip: {
    padding: "3px 10px",
    fontSize: 11,
    border: "1px solid rgba(99,102,241,0.45)",
    background: "rgba(99,102,241,0.08)",
    borderRadius: 999,
    color: "#c7d2fe",
  } as React.CSSProperties,
  link: { color: "#93c5fd", textDecoration: "underline", textUnderlineOffset: 3 } as React.CSSProperties,
};

export default function ResearchPage() {
  const crumbs = breadcrumbLd([
    { name: "Home", path: "/" },
    { name: "Research", path: "/research" },
  ]);

  return (
    <main style={S.page}>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@graph": [
              {
                "@type": "CollectionPage",
                "@id": `${SITE_URL}/research#page`,
                url: `${SITE_URL}/research`,
                name: "Research - Balmores Lab",
                mainEntity: { "@id": `${SITE_URL}/#person` },
                isPartOf: { "@id": `${SITE_URL}/#website` },
                author: { "@id": `${SITE_URL}/#person` },
              },
              ...RESEARCH_ARTICLES.map((a) => ({
                "@type": "ScholarlyArticle",
                "@id": `${SITE_URL}/research/${a.slug}#article`,
                headline: a.headline,
                url: `${SITE_URL}/research/${a.slug}`,
              })),
            ],
          }),
        }}
      />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(crumbs) }} />

      <SiteNav current="/research" />

      <article style={S.container}>
        <h1 style={S.h1}>Research</h1>
        <p style={S.sub}>AI-driven structural engineering — Balmores Lab</p>
        <p style={S.lede}>
          Directed by{" "}
          <Link href="/about" style={S.link}>
            {PERSON_NAME}
          </Link>
          , {JOB_TITLE}          , in collaboration with{" "}
          <Link href={SANDRA_AGCAOILI.profilePath} style={S.link}>
            <strong>{SANDRA_AGCAOILI.name}</strong>
          </Link>
          , AI Researcher and research
          partner (PhD in Artificial Intelligence, University of the Philippines
          Diliman · based in Singapore). Focus areas: AI models for structural
          integrity, computational design, and material efficiency — validated
          against PyNite and ETABS, not black-box guesses.
        </p>
        <p style={{ ...S.lede, marginTop: 12 }}>
          See also:{" "}
          <Link href="/about" style={S.link}>official profile</Link>
          {" · "}
          <Link href="/cv" style={S.link}>curriculum vitae</Link>
          {" · "}
          <Link href="/" style={S.link}>Balmores Strux AI demo</Link>
        </p>

        <h2 style={S.h2}>Research Team</h2>
        <ResearchPartnerCard />

        <h2 style={S.h2}>Working Papers</h2>
        {RESEARCH_ARTICLES.map((a) => (
          <section
            key={a.slug}
            id={a.slug}
            style={S.article}
            itemScope
            itemType="https://schema.org/ScholarlyArticle"
          >
            <h3 style={S.headline} itemProp="headline">
              <Link href={`/research/${a.slug}`} style={{ color: "inherit", textDecoration: "none" }}>
                {a.headline}
              </Link>
            </h3>
            <div style={S.meta}>
              by{" "}
              <span itemProp="author" itemScope itemType="https://schema.org/Person">
                <Link href="/about" itemProp="name" style={S.link}>
                  {PERSON_NAME}
                </Link>
              </span>
              {" & "}
              <span itemProp="author" itemScope itemType="https://schema.org/Person">
                <Link href={SANDRA_AGCAOILI.profilePath} itemProp="name" style={S.link}>
                  {SANDRA_AGCAOILI.name}
                </Link>
              </span>
              {" · "}
              <time itemProp="datePublished" dateTime={a.datePublished}>
                {new Date(a.datePublished).toLocaleDateString("en-US", {
                  year: "numeric",
                  month: "long",
                  day: "numeric",
                })}
              </time>
              {" · "}
              <Link href={`/research/${a.slug}`} style={S.link}>
                Read full abstract →
              </Link>
            </div>
            <p style={S.abstract} itemProp="abstract">
              {a.abstract.length > 280 ? `${a.abstract.slice(0, 280)}…` : a.abstract}
            </p>
            <div style={S.keywords}>
              {a.keywords.map((k) => (
                <span key={k} style={S.chip}>
                  {k}
                </span>
              ))}
            </div>
            <meta itemProp="inLanguage" content="en" />
            <meta itemProp="publisher" content="Balmores Lab" />
            <link itemProp="url" href={`${SITE_URL}/research/${a.slug}`} />
          </section>
        ))}

        <h2 style={S.h2}>The Closed Loop (system under study)</h2>
        <div style={S.article}>
          <p style={{ ...S.abstract, marginTop: 0 }}>
            The live demo on this site is also the primary research artifact: a
            fully on-device pipeline where the engineer&apos;s intent is
            interpreted, solved, and reviewed without any data leaving the
            machine.
          </p>
          <ol style={{ color: "#cbd5e1", lineHeight: 1.7, fontSize: 15, paddingLeft: 20, margin: "12px 0 0" }}>
            <li>
              <strong>Interpret.</strong> A locally-hosted reasoning LLM
              (DeepSeek-R1 via Ollama, loopback only) canonicalises a
              plain-English or shorthand structural brief into a strict,
              parseable form.
            </li>
            <li>
              <strong>Solve.</strong> The deterministic PyNite finite-element
              kernel runs the analysis — beams, 2D frames, and 3D buildings
              with P-Δ, drift, and base reactions.
            </li>
            <li>
              <strong>Review.</strong> The same local LLM reads the
              authoritative numeric result and writes an executive summary,
              recommendations, and a conclusion — grounded only in the FEM
              output.
            </li>
            <li>
              <strong>Return.</strong> The commentary streams back into the
              chat, token by token. A deterministic engineering summary is
              substituted if the model is unavailable, so the system never
              fails closed.
            </li>
          </ol>
        </div>

        <h2 style={S.h2}>Research Roadmap</h2>
        <div style={S.article}>
          <ul style={{ color: "#cbd5e1", lineHeight: 1.7, fontSize: 15, paddingLeft: 20, margin: 0 }}>
            <li>
              <strong>MSc (Computer Science) — now.</strong> Physics-informed
              neural surrogates that reproduce PyNite/ETABS envelopes in under
              a second with a physics-residual regulariser.
            </li>
            <li>
              <strong>DIT (Information Technology) — next.</strong> Productionising
              the privacy-preserving on-device loop: uncertainty quantification,
              an immutable audit trail back to a verifying FEM solve, and
              secure local deployment for engineering practices.
            </li>
            <li>
              <strong>Future.</strong> Multi-objective generative design with
              embodied-carbon as a first-class objective; expansion from
              Philippine NSCP 2015 to multi-code (ASCE 7, Eurocode) support;
              and on-device fine-tuning so the assistant learns a firm&apos;s
              detailing preferences without sharing data.
            </li>
          </ul>
        </div>

        <h2 style={S.h2}>Cite this work</h2>
        <div style={{ ...S.article, fontFamily: "ui-monospace, monospace", fontSize: 13, color: "#cbd5e1" }}>
          Balmores, L. D., &amp; Agcaoili, S. (2026). <em>Balmores Lab research programme in
          AI-driven structural engineering.</em> Balmores Lab.
          Available at: https://www.balmoreslab.com/research
        </div>

        <h2 style={S.h2}>Author &amp; Verified Profiles</h2>
        <div style={{ marginTop: 4 }}>
          <ProfileBadges align="left" showOrcidPill />
        </div>
      </article>
    </main>
  );
}
