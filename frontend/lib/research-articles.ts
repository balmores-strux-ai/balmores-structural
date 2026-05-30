import { PERSON_ID, SITE_URL } from "./seo";
import { SANDRA_AGCAOILI } from "./research-team";

const ARTICLE_AUTHORS = [
  { "@id": PERSON_ID },
  { "@id": SANDRA_AGCAOILI.id },
];

export type ResearchArticle = {
  slug: string;
  headline: string;
  abstract: string;
  keywords: string[];
  datePublished: string;
  status: "working-paper" | "in-progress" | "doctoral-programme";
};

export const RESEARCH_ARTICLES: ResearchArticle[] = [
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
    status: "working-paper",
  },
  {
    slug: "natural-language-to-pynite",
    headline:
      "Natural-Language-to-FEM: A Prompt-Driven PyNite Pipeline for 3D Irregular Frames",
    abstract:
      "A parser and LLM-assisted pipeline that converts plain-English building briefs into validated PyNite 3D models. Handles irregular grids, asymmetric bays, storey heights, DL/LL loadings, wind, and simplified seismic. Produces reactions, storey drift, P-Delta, and member envelopes.",
    keywords: [
      "natural language processing",
      "PyNite",
      "finite element analysis",
      "computational design",
    ],
    datePublished: "2025-12-15",
    status: "working-paper",
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
    status: "working-paper",
  },
  {
    slug: "privacy-preserving-on-device-llm-fem-loop",
    headline:
      "A Privacy-Preserving On-Device Loop: Local LLM Interpretation → FEM Solve → LLM Review for Structural Briefs",
    abstract:
      "A closed loop that keeps every prompt on the engineer's own machine. A locally-hosted reasoning LLM (DeepSeek-R1 on Ollama, loopback only) canonicalises a plain-English structural brief; the deterministic PyNite finite-element kernel solves it; the same local LLM then reviews the authoritative numeric result and writes recommendations and a conclusion. No prompt, model, or result ever leaves the device. We characterise latency, reasoning-trace scrubbing, loopback security gates, and a deterministic fallback that guarantees the system never fails closed.",
    keywords: [
      "privacy-preserving AI",
      "on-device inference",
      "DeepSeek-R1",
      "retrieval-augmented engineering",
      "PyNite",
      "human-in-the-loop",
    ],
    datePublished: "2026-04-05",
    status: "working-paper",
  },
  {
    slug: "physics-informed-surrogates-doctoral",
    headline:
      "Physics-Informed Neural Surrogates for Real-Time RC/Steel Frame Analysis under NSCP 2015 / ASCE 7 (Doctoral Programme)",
    abstract:
      "Doctoral research direction extending the MSc thesis: a unified surrogate trained on parametric PyNite/ETABS solves with an explicit physics-residual loss against the governing stiffness equations, so sub-second predictions remain code-consistent rather than merely statistically plausible. Targets uncertainty-quantified member envelopes, storey drift, and base reactions, with an audit trail back to a verifying FEM solve.",
    keywords: [
      "physics-informed neural networks",
      "surrogate modeling",
      "uncertainty quantification",
      "Information Technology",
      "scientific machine learning",
      "ETABS",
    ],
    datePublished: "2026-05-20",
    status: "doctoral-programme",
  },
];

export function getResearchArticle(slug: string): ResearchArticle | undefined {
  return RESEARCH_ARTICLES.find((a) => a.slug === slug);
}

export function buildResearchGraph() {
  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "CollectionPage",
        "@id": `${SITE_URL}/research#page`,
        url: `${SITE_URL}/research`,
        name: "Research - Balmores Lab",
        mainEntity: { "@id": PERSON_ID },
        isPartOf: { "@id": `${SITE_URL}/#website` },
      },
      ...RESEARCH_ARTICLES.map((a) => ({
        "@type": "ScholarlyArticle",
        "@id": `${SITE_URL}/research/${a.slug}#article`,
        headline: a.headline,
        abstract: a.abstract,
        keywords: a.keywords.join(", "),
        datePublished: a.datePublished,
        author: ARTICLE_AUTHORS,
        publisher: { "@id": `${SITE_URL}/#organization` },
        isPartOf: { "@id": `${SITE_URL}/research#page` },
        mainEntityOfPage: `${SITE_URL}/research/${a.slug}`,
        url: `${SITE_URL}/research/${a.slug}`,
        inLanguage: "en",
        creativeWorkStatus: a.status === "doctoral-programme" ? "InProgress" : "Draft",
      })),
    ],
  };
}

export function buildArticleLd(article: ResearchArticle) {
  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "ScholarlyArticle",
        "@id": `${SITE_URL}/research/${article.slug}#article`,
        headline: article.headline,
        abstract: article.abstract,
        keywords: article.keywords.join(", "),
        datePublished: article.datePublished,
        author: ARTICLE_AUTHORS,
        publisher: { "@id": `${SITE_URL}/#organization` },
        mainEntityOfPage: `${SITE_URL}/research/${article.slug}`,
        url: `${SITE_URL}/research/${article.slug}`,
        inLanguage: "en",
      },
      {
        "@type": "WebPage",
        "@id": `${SITE_URL}/research/${article.slug}#webpage`,
        url: `${SITE_URL}/research/${article.slug}`,
        name: article.headline,
        isPartOf: { "@id": `${SITE_URL}/#website` },
        about: { "@id": PERSON_ID },
      },
    ],
  };
}
