import type { Metadata } from "next";
import Link from "next/link";
import Script from "next/script";

const SITE_URL = "https://www.balmoreslab.com";

export const metadata: Metadata = {
  title: "Research — AI-Driven Structural Engineering",
  description:
    "Research programme at Balmores Laboratory: AI for structural integrity, computational design, and material efficiency. Led by Louie Doniego Balmores.",
  alternates: { canonical: "/research" },
  openGraph: {
    type: "website",
    url: `${SITE_URL}/research`,
    title: "Research — Balmores Laboratory",
    description:
      "AI-driven structural optimization, computational design, and material efficiency.",
  },
};

const articles = [
  {
    slug: "ai-driven-structural-optimization",
    headline:
      "AI-Driven Structural Optimization: Neural Surrogates on Parametric ETABS Datasets",
    abstract:
      "Training deep-learning surrogate models on 5,000+ parametric ETABS models to predict member demand and preliminary sizing in seconds. Early results show strong correlation for RC-frame envelopes with material-efficiency gains up to 12% vs. engineer-only baselines.",
    keywords: [
      "structural optimization",
      "deep learning",
      "ETABS",
      "reinforced concrete",
      "surrogate modeling",
    ],
    datePublished: "2025-11-01",
  },
  {
    slug: "natural-language-to-pynite",
    headline:
      "Natural-Language-to-FEM: A Prompt-Driven PyNite Pipeline for 3D Irregular Frames",
    abstract:
      "A parser + LLM-assisted pipeline that converts plain-English building briefs into validated PyNite 3D models. Handles irregular grids, asymmetric bays, storey heights, DL/LL loadings, wind, and simplified seismic. Produces reactions, storey drift, P-Δ, and member envelopes.",
    keywords: [
      "natural language processing",
      "PyNite",
      "finite element analysis",
      "computational design",
    ],
    datePublished: "2025-12-15",
  },
  {
    slug: "material-efficiency-generative-design",
    headline:
      "Material Efficiency by Generative Structural Design with Embodied-Carbon as a First-Class Objective",
    abstract:
      "Multi-objective optimization combining structural compliance, cost, and embodied-carbon. Demonstrates Pareto fronts for common mid-rise typologies in the Philippine context.",
    keywords: [
      "material efficiency",
      "embodied carbon",
      "generative design",
      "structural engineering",
    ],
    datePublished: "2026-02-10",
  },
];

const researchLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "CollectionPage",
      "@id": `${SITE_URL}/research#page`,
      url: `${SITE_URL}/research`,
      name: "Research — Balmores Laboratory",
      mainEntity: { "@id": `${SITE_URL}/#person` },
      isPartOf: { "@id": `${SITE_URL}/#website` },
    },
    ...articles.map((a) => ({
      "@type": "ScholarlyArticle",
      "@id": `${SITE_URL}/research#${a.slug}`,
      headline: a.headline,
      abstract: a.abstract,
      keywords: a.keywords.join(", "),
      datePublished: a.datePublished,
      author: { "@id": `${SITE_URL}/#person` },
      publisher: { "@id": `${SITE_URL}/#organization` },
      isPartOf: { "@id": `${SITE_URL}/research#page` },
      mainEntityOfPage: `${SITE_URL}/research`,
      url: `${SITE_URL}/research`,
      inLanguage: "en",
    })),
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
  container: { maxWidth: 940, margin: "0 auto", padding: "56px 28px 80px" } as React.CSSProperties,
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
};

export default function ResearchPage() {
  return (
    <main style={S.page}>
      <Script
        id="ld-json-research"
        type="application/ld+json"
        strategy="beforeInteractive"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(researchLd) }}
      />

      <article style={S.container}>
        <h1 style={S.h1}>Research</h1>
        <p style={S.sub}>AI-driven structural engineering · Balmores Laboratory</p>
        <p style={S.lede}>
          Directed by{" "}
          <Link href="/about" style={{ color: "#93c5fd" }}>
            Louie Doniego Balmores
          </Link>
          . Focus areas: AI models for structural integrity, computational
          design, and material efficiency. The goal is to convert traditional
          structural analysis into automated, intelligent workflows.
        </p>

        <h2 style={S.h2}>Working Papers</h2>
        {articles.map((a) => (
          <section
            key={a.slug}
            id={a.slug}
            style={S.article}
            itemScope
            itemType="https://schema.org/ScholarlyArticle"
          >
            <h3 style={S.headline} itemProp="headline">
              {a.headline}
            </h3>
            <div style={S.meta}>
              by{" "}
              <span itemProp="author" itemScope itemType="https://schema.org/Person">
                <span itemProp="name">Louie Doniego Balmores</span>
              </span>
              {" · "}
              <time itemProp="datePublished" dateTime={a.datePublished}>
                {new Date(a.datePublished).toLocaleDateString("en-US", {
                  year: "numeric",
                  month: "long",
                  day: "numeric",
                })}
              </time>
              {" · Balmores Laboratory"}
            </div>
            <p style={S.abstract} itemProp="abstract">
              {a.abstract}
            </p>
            <div style={S.keywords}>
              {a.keywords.map((k) => (
                <span key={k} style={S.chip}>
                  {k}
                </span>
              ))}
            </div>
            <meta itemProp="inLanguage" content="en" />
            <meta itemProp="publisher" content="Balmores Laboratory" />
          </section>
        ))}

        <h2 style={S.h2}>Cite this work</h2>
        <div style={{ ...S.article, fontFamily: "ui-monospace, monospace", fontSize: 13, color: "#cbd5e1" }}>
          Balmores, L. D. (2026). <em>Balmores Laboratory research programme in
          AI-driven structural engineering.</em> Balmores Laboratory.
          Available at: https://www.balmoreslab.com/research
        </div>
      </article>
    </main>
  );
}
