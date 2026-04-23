import type { MetadataRoute } from "next";

const SITE_URL = "https://www.balmoreslab.com";

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date();
  return [
    { url: `${SITE_URL}/`, lastModified, changeFrequency: "weekly", priority: 1.0 },
    { url: `${SITE_URL}/about`, lastModified, changeFrequency: "monthly", priority: 0.9 },
    { url: `${SITE_URL}/cv`, lastModified, changeFrequency: "monthly", priority: 0.8 },
    { url: `${SITE_URL}/research`, lastModified, changeFrequency: "monthly", priority: 0.8 },
    { url: `${SITE_URL}/seo-schema.json`, lastModified, changeFrequency: "monthly", priority: 0.5 },
    { url: `${SITE_URL}/humans.txt`, lastModified, changeFrequency: "yearly", priority: 0.3 },
    { url: `${SITE_URL}/llms.txt`, lastModified, changeFrequency: "monthly", priority: 0.5 },
    { url: `${SITE_URL}/.well-known/webfinger`, lastModified, changeFrequency: "yearly", priority: 0.3 },
  ];
}
