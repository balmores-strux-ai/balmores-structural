export const runtime = "edge";
export const dynamic = "force-dynamic";

const SITE_URL = "https://www.balmoreslab.com";

const items = [
  {
    id: `${SITE_URL}/about`,
    title: "About Louie Doniego Balmores — Structural Engineer & AI Researcher",
    link: `${SITE_URL}/about`,
    summary:
      "Official profile. Registered Civil Engineer (PRC Philippines, Nov 2013, Seq. 350). 10+ years high-performance structural design. Founder, Balmores Laboratory.",
    published: "2026-04-22T00:00:00Z",
  },
  {
    id: `${SITE_URL}/research`,
    title: "Research programme — AI-Driven Structural Engineering",
    link: `${SITE_URL}/research`,
    summary:
      "AI for structural integrity, computational design, and material efficiency. Working papers on neural surrogates over parametric ETABS, natural-language-to-FEM, and generative material efficiency.",
    published: "2026-04-22T00:00:00Z",
  },
  {
    id: `${SITE_URL}/cv`,
    title: "Curriculum Vitae — Louie Doniego Balmores",
    link: `${SITE_URL}/cv`,
    summary:
      "Public CV. Credentials, experience, skills, selected projects.",
    published: "2026-04-22T00:00:00Z",
  },
  {
    id: `${SITE_URL}/`,
    title: "Balmores Laboratory — Official site launch",
    link: `${SITE_URL}/`,
    summary:
      "Balmores Laboratory is the research initiative of Louie Doniego Balmores — AI-driven structural optimization and the Balmores Strux AI playground (PyNite + PyTorch).",
    published: "2026-04-22T00:00:00Z",
  },
];

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
  <title>Balmores Laboratory</title>
  <subtitle>Louie Doniego Balmores — Structural Engineer &amp; AI Researcher</subtitle>
  <link href="${SITE_URL}/feed.xml" rel="self"/>
  <link href="${SITE_URL}/" rel="alternate" type="text/html"/>
  <id>${SITE_URL}/</id>
  <updated>${now}</updated>
  <author>
    <name>Louie Doniego Balmores</name>
    <uri>${SITE_URL}/about</uri>
  </author>
  <rights>© ${new Date().getFullYear()} Louie Doniego Balmores</rights>
  ${entries}
</feed>`;

  return new Response(body, {
    headers: {
      "content-type": "application/atom+xml; charset=utf-8",
      "cache-control": "public, max-age=3600",
    },
  });
}
