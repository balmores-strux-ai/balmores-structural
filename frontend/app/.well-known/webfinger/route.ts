import { NextRequest } from "next/server";

import { SITE_URL } from "@/lib/seo";
import {
  SANDRA_AGCAOILI,
  SANDRA_LINKEDIN_URL,
  SANDRA_ORCID_URL,
  SANDRA_PROFILE_URL,
  SANDRA_RESEARCHGATE_URL,
  SANDRA_ROCKETREACH_URL,
  SANDRA_CONTACTOUT_URL,
} from "@/lib/research-team";

export const runtime = "edge";

const LOUIE_RESOURCE = "acct:louie@balmoreslab.com";
const SANDRA_RESOURCE = "acct:sandra@balmoreslab.com";

function louieProfile(resource: string) {
  return {
    subject: resource || LOUIE_RESOURCE,
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
      { rel: "describedby", type: "application/rdf+xml", href: `${SITE_URL}/foaf.rdf` },
    ],
  };
}

function sandraProfile(resource: string) {
  return {
    subject: resource || SANDRA_RESOURCE,
    aliases: [
      SITE_URL,
      SANDRA_PROFILE_URL,
      `${SITE_URL}/research`,
      `${SITE_URL}/about`,
      SANDRA_LINKEDIN_URL,
      SANDRA_ORCID_URL,
      SANDRA_RESEARCHGATE_URL,
      SANDRA_ROCKETREACH_URL,
      SANDRA_CONTACTOUT_URL,
    ],
    properties: {
      "http://schema.org/name": SANDRA_AGCAOILI.name,
      "http://schema.org/alternateName": "Sandra Agcaili; Sandra Agcaoli; Sandra A. Agcaoili",
      "http://schema.org/jobTitle": SANDRA_AGCAOILI.jobTitle,
      "http://schema.org/url": SANDRA_PROFILE_URL,
      "http://schema.org/worksFor": SITE_URL,
    },
    links: [
      {
        rel: "http://webfinger.net/rel/profile-page",
        type: "text/html",
        href: SANDRA_PROFILE_URL,
      },
      {
        rel: "http://webfinger.net/rel/avatar",
        type: "image/png",
        href: `${SANDRA_PROFILE_URL}/opengraph-image`,
      },
      { rel: "canonical", href: `${SANDRA_PROFILE_URL}#person` },
      {
        rel: "describedby",
        type: "application/ld+json",
        href: `${SITE_URL}/sandra-agcaoili-schema.json`,
      },
      { rel: "describedby", type: "application/rdf+xml", href: `${SITE_URL}/sandra-agcaoili.foaf.rdf` },
    ],
  };
}

export function GET(req: NextRequest) {
  const resource = (req.nextUrl.searchParams.get("resource") ?? "").trim().toLowerCase();
  const profile =
    resource.includes("sandra") || resource.includes("sandra@balmoreslab.com")
      ? sandraProfile(resource || SANDRA_RESOURCE)
      : louieProfile(resource || LOUIE_RESOURCE);

  return new Response(JSON.stringify(profile, null, 2), {
    headers: {
      "content-type": "application/jrd+json; charset=utf-8",
      "cache-control": "public, max-age=3600",
      "access-control-allow-origin": "*",
    },
  });
}
