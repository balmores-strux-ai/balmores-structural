import type { Metadata } from "next";
import Link from "next/link";
import Script from "next/script";

const SITE_URL = "https://www.balmoreslab.com";

export const metadata: Metadata = {
  title: "Curriculum Vitae — Louie Doniego Balmores",
  description:
    "Public CV of Louie Doniego Balmores, Registered Civil Engineer (PRC Philippines, 2013, Seq. 350). 10+ years structural design · AI-driven structural optimization research.",
  alternates: { canonical: "/cv" },
  openGraph: {
    type: "profile",
    url: `${SITE_URL}/cv`,
    title: "CV — Louie Doniego Balmores",
    description:
      "Structural Engineer (PRC PH) and AI researcher. CV / resume · Balmores Laboratory.",
  },
};

const resumeLd = {
  "@context": "https://schema.org",
  "@type": ["Person", "ProfilePage"],
  "@id": `${SITE_URL}/cv#resume`,
  mainEntity: { "@id": `${SITE_URL}/#person` },
  url: `${SITE_URL}/cv`,
  name: "Louie Doniego Balmores",
  jobTitle: "Structural Engineer & AI Researcher",
  worksFor: {
    "@type": "Organization",
    name: "Balmores Laboratory",
    url: SITE_URL,
  },
  alumniOf: [
    {
      "@type": "EducationalOrganization",
      name: "Civil Engineering — Philippines",
    },
  ],
  hasCredential: [
    {
      "@type": "EducationalOccupationalCredential",
      name: "Registered Civil Engineer",
      credentialCategory: "Professional License",
      recognizedBy: {
        "@type": "GovernmentOrganization",
        name: "Professional Regulation Commission (PRC)",
        url: "https://prc.gov.ph",
      },
      datePublished: "2013-11-27",
      identifier: "Nov 2013 CE Licensure Examination — Sequence No. 350",
    },
  ],
  knowsAbout: [
    "Structural Engineering",
    "Reinforced Concrete Design",
    "Steel Design",
    "Seismic Analysis",
    "Finite Element Analysis",
    "PyNite",
    "ETABS",
    "Artificial Intelligence",
    "Deep Learning for Engineering",
    "Computational Design",
    "Structural Optimization",
  ],
};

const S = {
  page: {
    minHeight: "100vh",
    background: "#080a0f",
    color: "#e6edf3",
    fontFamily:
      'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  } as React.CSSProperties,
  container: {
    maxWidth: 880,
    margin: "0 auto",
    padding: "56px 28px 80px",
  } as React.CSSProperties,
  h1: { fontSize: 36, margin: "0 0 6px", letterSpacing: "-0.02em", fontWeight: 700 } as React.CSSProperties,
  sub: { color: "#a5b4fc", margin: 0, fontSize: 18 } as React.CSSProperties,
  lede: { color: "#cbd5e1", fontSize: 16, lineHeight: 1.6, marginTop: 14 } as React.CSSProperties,
  h2: {
    fontSize: 13,
    letterSpacing: "0.22em",
    textTransform: "uppercase",
    color: "#60a5fa",
    margin: "36px 0 14px",
    fontWeight: 600,
  } as React.CSSProperties,
  card: {
    padding: 18,
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 12,
    background: "rgba(255,255,255,0.02)",
    marginBottom: 12,
  } as React.CSSProperties,
  role: { color: "#f1f5f9", fontSize: 17, fontWeight: 600, margin: 0 } as React.CSSProperties,
  period: { color: "#94a3b8", fontSize: 13, marginTop: 4 } as React.CSSProperties,
  p: { color: "#cbd5e1", lineHeight: 1.65, fontSize: 15, margin: "10px 0 0" } as React.CSSProperties,
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
  link: { color: "#93c5fd", textDecoration: "underline", textUnderlineOffset: 3 } as React.CSSProperties,
};

export default function CVPage() {
  return (
    <main style={S.page}>
      <Script
        id="ld-json-resume"
        type="application/ld+json"
        strategy="beforeInteractive"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(resumeLd) }}
      />

      <article
        style={S.container}
        className="h-card"
        itemScope
        itemType="https://schema.org/Person"
        itemID={`${SITE_URL}/#person`}
      >
        <header>
          <h1 style={S.h1} className="p-name" itemProp="name">
            Louie Doniego Balmores
          </h1>
          <p style={S.sub}>
            <span className="p-job-title" itemProp="jobTitle">Structural Engineer &amp; AI Researcher</span>
          </p>
          <p style={S.lede} className="p-note" itemProp="description">
            Registered Civil Engineer (PRC Philippines, 2013, Seq. 350). 10+
            years of high-performance structural design. Founder of{" "}
            <a className="u-url" style={S.link} href={SITE_URL} itemProp="url">
              Balmores Laboratory
            </a>{" "}
            — research initiative on AI-driven structural optimization.
          </p>
        </header>

        <h2 style={S.h2}>Credentials</h2>
        <div style={S.card}>
          <p style={S.role}>Registered Civil Engineer — PRC Philippines</p>
          <p style={S.period}>November 2013 · Sequence No. 350 ·{" "}
            <a style={S.link} href="https://prc.gov.ph" rel="noopener external" target="_blank">
              prc.gov.ph
            </a>
          </p>
        </div>
        <div style={S.card}>
          <p style={S.role}>P.Eng License Candidate</p>
          <p style={S.period}>Professional Engineers Ontario · target 2027</p>
        </div>
        <div style={S.card}>
          <p style={S.role}>PE License Candidate</p>
          <p style={S.period}>NCEES (USA) · target 2028</p>
        </div>

        <h2 style={S.h2}>Experience</h2>
        <div style={S.card}>
          <p style={S.role}>Founder · Balmores Laboratory</p>
          <p style={S.period}>2023 – present · balmoreslab.com</p>
          <p style={S.p}>
            Research on AI-driven structural optimization. Building
            Balmores Strux AI — natural-language-to-PyNite 3D FEM
            pipeline with PyTorch surrogate models trained on parametric
            ETABS datasets.
          </p>
        </div>
        <div style={S.card}>
          <p style={S.role}>Structural Engineer (consulting, various)</p>
          <p style={S.period}>2013 – present · Philippines</p>
          <p style={S.p}>
            10+ years of practice in reinforced-concrete and steel design
            for mid- and high-rise buildings. Seismic analysis, P-Δ
            effects, drift control, foundation design.
          </p>
        </div>

        <h2 style={S.h2}>Skills</h2>
        <div>
          {[
            "Reinforced concrete design",
            "Steel design",
            "Seismic analysis",
            "Finite Element Analysis",
            "PyNite",
            "ETABS",
            "P-Delta analysis",
            "PyTorch",
            "Deep learning",
            "Next.js / React",
            "FastAPI",
            "Parametric design",
            "Computational design",
          ].map((s) => (
            <span key={s} style={S.chip}>
              {s}
            </span>
          ))}
        </div>

        <h2 style={S.h2}>Selected Projects</h2>
        <div style={S.card}>
          <p style={S.role}>Balmores Strux AI (2024–present)</p>
          <p style={S.p}>
            Open-source structural-AI playground. Chat with a PyNite FEM
            backend in plain English to produce 3D frame models with
            reactions, storey drift, member envelopes, and P-Δ. PyTorch
            surrogate trained on ~5000 parametric ETABS models.{" "}
            <Link href="/" style={S.link}>View demo →</Link>
          </p>
        </div>

        <h2 style={S.h2}>Contact</h2>
        <p style={S.p}>
          Website:{" "}
          <a className="u-url" style={S.link} href={SITE_URL}>
            balmoreslab.com
          </a>{" "}
          · Profile:{" "}
          <Link style={S.link} href="/about">/about</Link>{" "}
          · LinkedIn:{" "}
          <a
            className="u-url"
            style={S.link}
            href="https://www.linkedin.com/in/louiebalmores/"
            rel="me noopener external"
            target="_blank"
          >
            louiebalmores
          </a>
          {" · "}
          X:{" "}
          <a
            className="u-url"
            style={S.link}
            href="https://x.com/louiedbalmores"
            rel="me noopener external"
            target="_blank"
          >
            @louiedbalmores
          </a>
        </p>
      </article>
    </main>
  );
}
