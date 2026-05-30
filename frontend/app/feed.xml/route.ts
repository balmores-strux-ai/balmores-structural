export const runtime = "edge";
export const dynamic = "force-dynamic";

import { RESEARCH_ARTICLES } from "@/lib/research-articles";

const SITE_URL = "https://www.balmoreslab.com";

const staticItems = [
  {
    id: `${SITE_URL}/about`,
    title: "About Louie Doniego Balmores - Structural Engineer & AI Researcher",
    link: `${SITE_URL}/about`,
    summary:
      "Official profile. Registered Civil Engineer (PRC Philippines, Nov 2013, Seq. 350). 10+ years high-performance structural design. Founder, Balmores Lab.",
    published: "2026-05-29T00:00:00Z",
  },
  {
    id: `${SITE_URL}/cv`,
    title: "Curriculum Vitae - Louie Doniego Balmores",
    link: `${SITE_URL}/cv`,
    summary:
      "Public CV. Credentials, education, experience, skills, selected projects, and research publications.",
    published: "2026-05-29T00:00:00Z",
  },
  {
    id: `${SITE_URL}/research`,
    title: "Research programme - AI-Driven Structural Engineering",
    link: `${SITE_URL}/research`,
    summary:
      "AI for structural integrity, computational design, and material efficiency. Working papers on neural surrogates, NL-to-FEM, and physics-informed ML.",
    published: "2026-05-29T00:00:00Z",
  },
  {
    id: `${SITE_URL}/`,
    title: "Balmores Lab - AI-driven structural engineering",
    link: `${SITE_URL}/`,
    summary:
      "Official site of Louie Doniego Balmores. Balmores Strux AI playground — PyNite 3D FEM from a design brief.",
    published: "2026-05-29T00:00:00Z",
  },
];

const articleItems = RESEARCH_ARTICLES.map((a) => ({
  id: `${SITE_URL}/research/${a.slug}`,
  title: a.headline,
  link: `${SITE_URL}/research/${a.slug}`,
  summary: a.abstract,
  published: `${a.datePublished}T00:00:00Z`,
}));

const items = [...staticItems, ...articleItems];

function xml(s: string) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

export function GET() {
  const now = new Date().toISOString();
  const entries = items
    .map(
      (it) => `
  <entry>
    <id>${xml(it.id)}</id>
    <title>${xml(it.title)}</title>
    <link href="${xml(it.link)}" rel="alternate" type="text/html"/>
    <published>${xml(it.published)}</published>
    <updated>${xml(it.published)}</updated>
    <author>
      <name>Louie Doniego Balmores</name>
      <uri>${SITE_URL}/about</uri>
    </author>
    <summary>${xml(it.summary)}</summary>
  </entry>`,
    )
    .join("");

  const body = `<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Balmores Lab</title>
  <subtitle>Louie Doniego Balmores - Structural Engineer &amp; AI Researcher</subtitle>
  <link href="${SITE_URL}/feed.xml" rel="self"/>
  <link href="${SITE_URL}/" rel="alternate" type="text/html"/>
  <id>${SITE_URL}/</id>
  <updated>${now}</updated>
  <author>
    <name>Louie Doniego Balmores</name>
    <uri>${SITE_URL}/about</uri>
  </author>
  <rights>(c) ${new Date().getFullYear()} Louie Doniego Balmores</rights>
  ${entries}
</feed>`;

  return new Response(body, {
    headers: {
      "content-type": "application/atom+xml; charset=utf-8",
      "cache-control": "public, max-age=3600",
    },
  });
}
