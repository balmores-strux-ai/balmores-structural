/**
 * Canonical SEO / Knowledge Graph constants for balmoreslab.com.
 * Keep identity strings identical across layout JSON-LD, pages, llms.txt, and bios.
 */

import {
  buildSandraPersonNode,
  buildSandraProfilePageNode,
  SANDRA_AGCAOILI,
  SANDRA_PERSON_ID,
} from "./research-team";

export const SITE_URL = "https://www.balmoreslab.com";
export const SITE_NAME = "Balmores Lab";
export const PERSON_NAME = "Louie Doniego Balmores";
export const JOB_TITLE = "Structural Engineer & AI Researcher";
export const PERSON_SLUG = "louie-doniego-balmores";

export const PERSON_ID = `${SITE_URL}/#person`;
export const ORG_ID = `${SITE_URL}/#organization`;
export const SITE_ID = `${SITE_URL}/#website`;
export const PROFILE_PAGE_ID = `${SITE_URL}/about#profilepage`;

export const ORCID_ID = "0009-0008-5479-4033";
export const ORCID_URL = `https://orcid.org/${ORCID_ID}`;
export const WIKIDATA_ID = "Q139544451";
export const WIKIDATA_URL = `https://www.wikidata.org/wiki/${WIKIDATA_ID}`;

export const DEFAULT_DESCRIPTION =
  "Louie Doniego Balmores is a licensed Civil Engineer (PRC Philippines, 2013) with over 10 years of structural engineering experience and an AI researcher pioneering AI-driven structural optimization at Balmores Lab (balmoreslab.com).";

export const SHORT_DESCRIPTION =
  "Licensed Civil Engineer (PRC Philippines) with 10+ years of structural design experience, researching AI-driven structural optimization at Balmores Lab.";

/** Identity URLs published as schema.org sameAs + rel=me across the site. */
export const BASE_SAME_AS = [
  "https://www.linkedin.com/in/louiebalmores/",
  "https://x.com/louiedbalmores",
  "https://twitter.com/louiedbalmores",
  ORCID_URL,
  "https://about.me/louiebalmoresdesign/",
  WIKIDATA_URL,
  "https://github.com/balmores-strux-ai/balmores-structural",
  "https://worldchess.com/profile/422673",
  "https://prc.gov.ph",
  `${SITE_URL}/about`,
  `${SITE_URL}/cv`,
  `${SITE_URL}/research`,
];

/** Optional sameAs from Render env vars — Wikidata/ORCID also hardcoded above. */
export function getExtraSameAs(): string[] {
  const env = (typeof process !== "undefined" ? process.env : {}) as Record<string, string | undefined>;
  const urls: string[] = [];
  const push = (u: string | undefined) => {
    if (u) urls.push(u);
  };
  push(env.NEXT_PUBLIC_GITHUB_URL);
  push(env.NEXT_PUBLIC_KEYBASE_URL);
  push(env.NEXT_PUBLIC_MASTODON_URL);
  push(env.NEXT_PUBLIC_SCHOLAR_URL);
  push(env.NEXT_PUBLIC_GRAVATAR_URL);
  push(env.NEXT_PUBLIC_RESEARCHGATE_URL);
  push(env.NEXT_PUBLIC_YOUTUBE_URL);
  push(env.NEXT_PUBLIC_DEVTO_URL);
  push(env.NEXT_PUBLIC_STACKOVERFLOW_URL);
  const wikidata = env.NEXT_PUBLIC_WIKIDATA_ID;
  if (wikidata && /^Q\d+$/.test(wikidata)) urls.push(`https://www.wikidata.org/wiki/${wikidata}`);
  const orcid = env.NEXT_PUBLIC_ORCID_ID;
  if (orcid && /^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$/.test(orcid)) urls.push(`https://orcid.org/${orcid}`);
  return urls;
}

export function getAllSameAs(): string[] {
  return Array.from(new Set([...BASE_SAME_AS, ...getExtraSameAs()]));
}

export function breadcrumbLd(items: { name: string; path: string }[]) {
  return {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: items.map((item, i) => ({
      "@type": "ListItem",
      position: i + 1,
      name: item.name,
      item: item.path.startsWith("http") ? item.path : `${SITE_URL}${item.path}`,
    })),
  };
}

export const PERSON_KNOWS_ABOUT = [
  "Structural Engineering",
  "Artificial Intelligence",
  "Machine Learning for Engineering",
  "Building Design",
  "High-Performance Buildings",
  "Computational Design",
  "Structural Optimization",
  "Finite Element Analysis",
  "PyNite",
  "Reinforced Concrete Design",
  "Steel Design",
  "Seismic Design",
  "Generative Design",
  "Parametric Structural Modeling",
  "Physics-Informed Neural Networks",
  "Natural Language Processing for Engineering",
];

export function buildRootStructuredData() {
  const ALL_SAME_AS = getAllSameAs();
  const today = new Date().toISOString().split("T")[0];

  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Person",
        "@id": PERSON_ID,
        name: PERSON_NAME,
        givenName: "Louie",
        additionalName: "Doniego",
        familyName: "Balmores",
        alternateName: ["Louie Balmores", "Engr. Louie Balmores", "Louie D. Balmores"],
        honorificSuffix: "P.Eng (Candidate), PE (Candidate)",
        jobTitle: JOB_TITLE,
        description: DEFAULT_DESCRIPTION,
        url: SITE_URL,
        mainEntityOfPage: { "@id": PROFILE_PAGE_ID },
        image: {
          "@type": "ImageObject",
          url: `${SITE_URL}/opengraph-image`,
          caption: `${PERSON_NAME} - ${JOB_TITLE}`,
        },
        gender: "Male",
        birthDate: "1991-06-26",
        birthPlace: {
          "@type": "Place",
          name: "Tuguegarao City, Cagayan Valley, Philippines",
          address: {
            "@type": "PostalAddress",
            addressLocality: "Tuguegarao City",
            addressRegion: "Cagayan Valley (Region II)",
            addressCountry: "PH",
          },
        },
        homeLocation: {
          "@type": "Place",
          name: "Ontario, Canada",
          address: {
            "@type": "PostalAddress",
            addressRegion: "Ontario",
            addressCountry: "CA",
          },
        },
        workLocation: {
          "@type": "Place",
          name: "Toronto, Ontario, Canada",
          address: {
            "@type": "PostalAddress",
            addressLocality: "Toronto",
            addressRegion: "Ontario",
            addressCountry: "CA",
          },
        },
        address: {
          "@type": "PostalAddress",
          addressLocality: "Toronto",
          addressRegion: "Ontario",
          addressCountry: "CA",
        },
        nationality: { "@type": "Country", name: "Philippines" },
        worksFor: { "@id": ORG_ID },
        founder: { "@id": ORG_ID },
        alumniOf: [
          { "@type": "EducationalOrganization", name: "Master of Science in Computer Science (in progress)" },
          { "@type": "EducationalOrganization", name: "Doctor of Information Technology (in progress)" },
          { "@type": "EducationalOrganization", name: "Bachelor of Science in Civil Engineering" },
        ],
        knowsLanguage: ["English", "Filipino"],
        knowsAbout: PERSON_KNOWS_ABOUT,
        hasOccupation: [
          {
            "@type": "Occupation",
            name: "Structural Engineer",
            occupationLocation: { "@type": "Country", name: "Philippines" },
            skills:
              "Reinforced concrete design, steel design, seismic analysis, P-Delta, finite element modeling, ETABS, PyNite",
          },
          {
            "@type": "Occupation",
            name: "AI Researcher",
            skills:
              "Deep learning for structural optimization, surrogate FEM models, generative structural design, computational design automation",
          },
        ],
        hasCredential: [
          {
            "@type": "EducationalOccupationalCredential",
            credentialCategory: "Professional License",
            name: "Registered Civil Engineer",
            recognizedBy: {
              "@type": "GovernmentOrganization",
              name: "Professional Regulation Commission (PRC)",
              url: "https://prc.gov.ph",
              address: {
                "@type": "PostalAddress",
                addressCountry: "PH",
                addressRegion: "Metro Manila",
              },
            },
            datePublished: "2013-11-27",
            identifier: "Nov 2013 Civil Engineer Licensure Examination - Sequence No. 350",
          },
          {
            "@type": "EducationalOccupationalCredential",
            credentialCategory: "Professional License (Candidate)",
            name: "PEng Candidate",
            recognizedBy: {
              "@type": "Organization",
              name: "Professional Engineers Ontario (PEO)",
              url: "https://www.peo.on.ca",
            },
          },
          {
            "@type": "EducationalOccupationalCredential",
            credentialCategory: "Professional License (Candidate)",
            name: "US PE Candidate",
            recognizedBy: {
              "@type": "Organization",
              name: "NCEES - National Council of Examiners for Engineering and Surveying",
              url: "https://ncees.org",
            },
          },
        ],
        sameAs: ALL_SAME_AS,
        colleague: { "@id": SANDRA_PERSON_ID },
      },
      buildSandraPersonNode(),
      buildSandraProfilePageNode(),
      {
        "@type": "Organization",
        "@id": ORG_ID,
        name: SITE_NAME,
        alternateName: ["Balmores Strux AI", "Balmores Structural", "Balmores Laboratory"],
        url: SITE_URL,
        logo: `${SITE_URL}/opengraph-image`,
        founder: { "@id": PERSON_ID },
        employee: [{ "@id": PERSON_ID }, { "@id": SANDRA_PERSON_ID }],
        sameAs: ALL_SAME_AS.filter((u) => !u.endsWith("/cv") && !u.endsWith("/research")),
        description:
          "Balmores Lab researches AI-driven structural engineering — converting traditional structural analysis workflows into automated, intelligent systems.",
        knowsAbout: [
          "Structural AI",
          "Computational Structural Design",
          "Finite Element Analysis",
          "Structural Optimization",
          "PyNite",
          "Physics-Informed Neural Networks",
        ],
      },
      {
        "@type": "WebSite",
        "@id": SITE_ID,
        url: SITE_URL,
        name: SITE_NAME,
        description: `Official website of ${PERSON_NAME}, ${SANDRA_AGCAOILI.name}, and ${SITE_NAME} — AI-driven structural engineering.`,
        publisher: { "@id": PERSON_ID },
        author: { "@id": PERSON_ID },
        inLanguage: "en",
      },
      {
        "@type": "ProfilePage",
        "@id": PROFILE_PAGE_ID,
        url: `${SITE_URL}/about`,
        name: `About ${PERSON_NAME}`,
        about: { "@id": PERSON_ID },
        mainEntity: { "@id": PERSON_ID },
        isPartOf: { "@id": SITE_ID },
        dateModified: today,
      },
    ],
  };
}
