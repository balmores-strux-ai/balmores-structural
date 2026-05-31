import type { MetadataRoute } from "next";
import { RESEARCH_ARTICLES } from "@/lib/research-articles";
import { SANDRA_PROFILE_URL } from "@/lib/research-team";

const SITE_URL = "https://www.balmoreslab.com";

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date();
  const researchPages = RESEARCH_ARTICLES.map((a) => ({
    url: `${SITE_URL}/research/${a.slug}`,
    lastModified,
    changeFrequency: "monthly" as const,
    priority: 0.75,
  }));

  return [
    { url: `${SITE_URL}/`, lastModified, changeFrequency: "weekly", priority: 1.0 },
    { url: `${SITE_URL}/about`, lastModified, changeFrequency: "monthly", priority: 0.9 },
    { url: SANDRA_PROFILE_URL, lastModified, changeFrequency: "monthly", priority: 0.88 },
    { url: `${SITE_URL}/cv`, lastModified, changeFrequency: "monthly", priority: 0.85 },
    { url: `${SITE_URL}/research`, lastModified, changeFrequency: "monthly", priority: 0.85 },
    ...researchPages,
    { url: `${SITE_URL}/feed.xml`, lastModified, changeFrequency: "weekly", priority: 0.6 },
    { url: `${SITE_URL}/seo-schema.json`, lastModified, changeFrequency: "monthly", priority: 0.5 },
    { url: `${SITE_URL}/sandra-agcaoili-schema.json`, lastModified, changeFrequency: "monthly", priority: 0.55 },
    { url: `${SITE_URL}/humans.txt`, lastModified, changeFrequency: "yearly", priority: 0.3 },
    { url: `${SITE_URL}/llms.txt`, lastModified, changeFrequency: "monthly", priority: 0.5 },
    { url: `${SITE_URL}/foaf.rdf`, lastModified, changeFrequency: "yearly", priority: 0.35 },
    { url: `${SITE_URL}/sandra-agcaoili.foaf.rdf`, lastModified, changeFrequency: "yearly", priority: 0.35 },
    { url: `${SITE_URL}/.well-known/webfinger?resource=acct:sandra@balmoreslab.com`, lastModified, changeFrequency: "yearly", priority: 0.3 },
  ];
}
