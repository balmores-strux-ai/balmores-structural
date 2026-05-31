import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import ProfileBadges from "@/components/ProfileBadges";
import SiteNav from "@/components/SiteNav";
import {
  buildArticleLd,
  getResearchArticle,
  RESEARCH_ARTICLES,
} from "@/lib/research-articles";
import { breadcrumbLd, PERSON_NAME, SITE_URL } from "@/lib/seo";
import { SANDRA_AGCAOILI } from "@/lib/research-team";

type Props = { params: { slug: string } };

export function generateStaticParams() {
  return RESEARCH_ARTICLES.map((a) => ({ slug: a.slug }));
}

export function generateMetadata({ params }: Props): Metadata {
  const article = getResearchArticle(params.slug);
  if (!article) return {};

  const url = `${SITE_URL}/research/${article.slug}`;
  return {
    title: article.headline,
    description: article.abstract,
    alternates: { canonical: `/research/${article.slug}` },
    keywords: article.keywords,
    authors: [
      { name: PERSON_NAME, url: `${SITE_URL}/about` },
      { name: SANDRA_AGCAOILI.name },
    ],
    openGraph: {
      type: "article",
      url,
      title: article.headline,
      description: article.abstract,
      publishedTime: article.datePublished,
      authors: [PERSON_NAME, SANDRA_AGCAOILI.name],
    },
    twitter: {
      title: article.headline,
      description: article.abstract,
    },
  };
}

const S = {
  page: {
    minHeight: "100vh",
    background: "#080a0f",
    color: "#e6edf3",
    fontFamily:
      'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  } as React.CSSProperties,
  container: { maxWidth: 820, margin: "0 auto", padding: "40px 28px 80px" } as React.CSSProperties,
  h1: { fontSize: 34, margin: "0 0 12px", lineHeight: 1.2, fontWeight: 700 } as React.CSSProperties,
  meta: { color: "#94a3b8", fontSize: 14, marginBottom: 24 } as React.CSSProperties,
  abstract: { color: "#cbd5e1", lineHeight: 1.7, fontSize: 17, margin: "0 0 24px" } as React.CSSProperties,
  chip: {
    padding: "4px 12px",
    fontSize: 12,
    border: "1px solid rgba(99,102,241,0.45)",
    background: "rgba(99,102,241,0.08)",
    borderRadius: 999,
    color: "#c7d2fe",
    marginRight: 6,
    marginBottom: 6,
    display: "inline-block",
  } as React.CSSProperties,
  link: { color: "#93c5fd", textDecoration: "underline", textUnderlineOffset: 3 } as React.CSSProperties,
};

export default function ResearchArticlePage({ params }: Props) {
  const article = getResearchArticle(params.slug);
  if (!article) notFound();

  const crumbs = breadcrumbLd([
    { name: "Home", path: "/" },
    { name: "Research", path: "/research" },
    { name: article.headline.slice(0, 60), path: `/research/${article.slug}` },
  ]);

  return (
    <main style={S.page}>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(buildArticleLd(article)) }} />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(crumbs) }} />

      <SiteNav current="/research" />

      <article style={S.container} className="site-page-shell" itemScope itemType="https://schema.org/ScholarlyArticle">
        <p style={{ margin: "0 0 20px", fontSize: 14 }}>
          <Link href="/research" style={S.link}>
            ← All research
          </Link>
        </p>

        <h1 style={S.h1} itemProp="headline">
          {article.headline}
        </h1>
        <div style={S.meta}>
          by{" "}
          <Link href="/about" style={S.link}>
            <span itemProp="author" itemScope itemType="https://schema.org/Person">
              <span itemProp="name">{PERSON_NAME}</span>
            </span>
          </Link>
          {" & "}
          <Link href={SANDRA_AGCAOILI.profilePath} style={S.link}>
            <span itemProp="author" itemScope itemType="https://schema.org/Person">
              <span itemProp="name">{SANDRA_AGCAOILI.name}</span>
            </span>
          </Link>
          {" · "}
          <time itemProp="datePublished" dateTime={article.datePublished}>
            {new Date(article.datePublished).toLocaleDateString("en-US", {
              year: "numeric",
              month: "long",
              day: "numeric",
            })}
          </time>
          {" · "}
          <span itemProp="publisher">Balmores Lab</span>
        </div>

        <p style={S.abstract} itemProp="abstract">
          {article.abstract}
        </p>

        <div style={{ marginBottom: 32 }}>
          {article.keywords.map((k) => (
            <span key={k} style={S.chip}>
              {k}
            </span>
          ))}
        </div>

        <p style={{ color: "#94a3b8", fontSize: 15, lineHeight: 1.6 }}>
          This is a working paper from the{" "}
          <Link href="/research" style={S.link}>
            Balmores Lab research programme
          </Link>
          . For credentials and background, see the{" "}
          <Link href="/about" style={S.link}>
            official profile
          </Link>{" "}
          and{" "}
          <Link href="/cv" style={S.link}>
            CV
          </Link>
          .
        </p>

        <div style={{ marginTop: 36 }}>
          <ProfileBadges align="left" showOrcidPill />
        </div>
      </article>
    </main>
  );
}
