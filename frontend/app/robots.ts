import type { MetadataRoute } from "next";

const SITE_URL = "https://www.balmoreslab.com";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      { userAgent: "*", allow: "/", disallow: ["/api/"] },

      // Classic search engines — explicit allow to remove any ambiguity.
      { userAgent: ["Googlebot", "Googlebot-Image", "Googlebot-News", "Google-InspectionTool"], allow: "/" },
      { userAgent: ["Bingbot", "AdIdxBot"], allow: "/" },
      { userAgent: ["DuckDuckBot", "Slurp", "Baiduspider", "YandexBot"], allow: "/" },

      // AI answer engines / LLM crawlers. Allowing these is a *strategic
      // win* for Knowledge-Panel goals: ChatGPT/Claude/Perplexity answers
      // that cite balmoreslab.com generate indirect link signals back to
      // Google, and entity recognition in AI answers has started feeding
      // Google's own entity graph.
      { userAgent: ["GPTBot", "ChatGPT-User", "OAI-SearchBot"], allow: "/" },
      { userAgent: ["ClaudeBot", "Claude-Web", "anthropic-ai"], allow: "/" },
      { userAgent: ["PerplexityBot", "Perplexity-User"], allow: "/" },
      { userAgent: ["Google-Extended"], allow: "/" },
      { userAgent: ["Applebot", "Applebot-Extended"], allow: "/" },
      { userAgent: ["CCBot"], allow: "/" },
      { userAgent: ["Amazonbot", "FacebookBot", "Meta-ExternalAgent"], allow: "/" },
      { userAgent: ["Bytespider"], allow: "/" },
      { userAgent: ["cohere-ai", "cohere-training-data-crawler"], allow: "/" },
      { userAgent: ["YouBot"], allow: "/" },
      { userAgent: ["DiffBot"], allow: "/" },
    ],
    sitemap: [`${SITE_URL}/sitemap.xml`],
    host: SITE_URL,
  };
}
