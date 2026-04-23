import { NextRequest } from "next/server";

// WebFinger (RFC 7033) — Fediverse / Mastodon / ActivityPub use this to
// resolve an identity string like acct:louie@balmoreslab.com. Even if
// you don't run an ActivityPub actor, returning a correct WebFinger
// response makes the domain discoverable to Mastodon clients and is
// increasingly ingested by AI crawlers for identity resolution.
export const runtime = "edge";

const SITE_URL = "https://www.balmoreslab.com";

export function GET(req: NextRequest) {
  const resource = req.nextUrl.searchParams.get("resource") ?? "";
  const profile = {
    subject: resource || "acct:louie@balmoreslab.com",
    aliases: [
      SITE_URL,
      `${SITE_URL}/about`,
      "https://www.linkedin.com/in/louiebalmores/",
      "https://x.com/louiedbalmores",
    ],
    properties: {
      "http://schema.org/name": "Louie Doniego Balmores",
      "http://schema.org/jobTitle": "Structural Engineer & AI Researcher",
      "http://schema.org/url": SITE_URL,
    },
    links: [
      { rel: "http://webfinger.net/rel/profile-page", type: "text/html", href: `${SITE_URL}/about` },
      { rel: "http://webfinger.net/rel/avatar", type: "image/png", href: `${SITE_URL}/opengraph-image` },
      { rel: "canonical", href: `${SITE_URL}/#person` },
      { rel: "describedby", type: "application/ld+json", href: `${SITE_URL}/seo-schema.json` },
    ],
  };
  return new Response(JSON.stringify(profile, null, 2), {
    headers: {
      "content-type": "application/jrd+json; charset=utf-8",
      "cache-control": "public, max-age=3600",
      "access-control-allow-origin": "*",
    },
  });
}
