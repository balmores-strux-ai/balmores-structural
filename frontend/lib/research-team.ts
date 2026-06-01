const SITE_URL = "https://www.balmoreslab.com";
const PERSON_ID = `${SITE_URL}/#person`;
const ORG_ID = `${SITE_URL}/#organization`;

export const SANDRA_SLUG = "sandra-agcaoili";
export const SANDRA_PROFILE_PATH = `/about/${SANDRA_SLUG}`;
export const SANDRA_PROFILE_URL = `${SITE_URL}${SANDRA_PROFILE_PATH}`;
export const SANDRA_PERSON_ID = `${SANDRA_PROFILE_URL}#person`;
export const SANDRA_PROFILE_PAGE_ID = `${SANDRA_PROFILE_URL}#profilepage`;
export const SANDRA_FAQ_ID = `${SANDRA_PROFILE_URL}#faq`;

/** @deprecated Use SANDRA_PERSON_ID — kept for fragment backward-compat in microdata */
export const SANDRA_LEGACY_FRAGMENT_ID = `${SITE_URL}/about#sandra-agcaoili`;

const SITE_ROOT = SITE_URL;

/** Correct short forms — canonical name remains Sandra Agcaoili. */
export const SANDRA_ALIASES = [
  "Sandra A. Agcaoili",
  "Sandra Agcaoili AI researcher",
  "Sandra Agcaoili Balmores Lab",
  "Sandra Agcaoili UP Diliman",
  "Sandra Agcaoili Singapore",
  "Sandra Agcaoili Licensed Agriculturist",
  "Sandra Agcaoili PRC",
] as const;

/** Common misspellings — same person; maps to balmoreslab.com profile. */
export const SANDRA_TYPOS = [
  "Sandra Agcaili",
  "Sandra Agcaoli",
  "Sandra Agcaoilli",
  "Sandra Agcaoilii",
  "Sandra Agcaolli",
  "Sandra Agcaoili Balmores",
  "Sandra Agcaili Balmores Lab",
  "Sandra Agcaoli AI researcher",
] as const;

export function getSandraAlternateNames(): string[] {
  return Array.from(new Set([...SANDRA_ALIASES, ...SANDRA_TYPOS]));
}

export const SANDRA_ORCID_ID = "0009-0006-4000-0031";
export const SANDRA_ORCID_URL = `https://orcid.org/${SANDRA_ORCID_ID}`;
export const SANDRA_LINKEDIN_URL =
  "https://www.linkedin.com/in/sandra-agcaoili-a059a2152/";
export const SANDRA_RESEARCHGATE_URL =
  "https://www.researchgate.net/profile/Sandra-Agcaoili";
export const SANDRA_ROCKETREACH_URL =
  "https://rocketreach.co/sandra-agcaoili-email_72113673";
export const SANDRA_CONTACTOUT_URL =
  "https://contactout.com/sandra-agcaoili-10976";
export const SANDRA_PRC_URL = "https://prc.gov.ph";

export const SANDRA_PRC_LICENSE = {
  title: "Licensed Agriculturist",
  credentialCategory: "Professional License",
  issuer: "Professional Regulation Commission (PRC)",
  issuerUrl: SANDRA_PRC_URL,
  country: "Philippines",
  description:
    "Registered professional license to practise as an agriculturist in the Republic of the Philippines, under the Professional Regulation Commission (PRC).",
} as const;

export const SANDRA_SKILLS = [
  "Animal Nutrition",
  "Feed Formulation",
  "Technical Sales Support",
  "Product Development",
  "Artificial Intelligence",
  "Machine Learning",
  "Analytics",
] as const;

export type SandraEducationEntry = {
  degree: string;
  field: string;
  institution: string;
  institutionUrl?: string;
  startYear: number;
  endYear: number | null;
  status: string;
};

export type SandraWorkEntry = {
  title: string;
  organization: string;
  startYear: number | null;
  endYear: number | null;
  location: string;
  current?: boolean;
};

export const SANDRA_EDUCATION: readonly SandraEducationEntry[] = [
  {
    degree: "Doctor of Philosophy (PhD)",
    field: "Artificial Intelligence",
    institution: "University of the Philippines Diliman",
    institutionUrl: "https://upd.edu.ph",
    startYear: 2023,
    endYear: null,
    status: "In progress",
  },
  {
    degree: "Bachelor of Science (BS)",
    field: "Agriculture",
    institution: "University of the Philippines",
    startYear: 2005,
    endYear: 2009,
    status: "Completed",
  },
];

export const SANDRA_WORK_HISTORY: readonly SandraWorkEntry[] = [
  {
    title: "Research Partner",
    organization: "Balmores Lab",
    startYear: null,
    endYear: null,
    location: "Singapore",
    current: true,
  },
  {
    title: "Technical Specialist (Animal Nutrition)",
    organization: "Zagro Singapore Pte Ltd",
    startYear: 2014,
    endYear: 2018,
    location: "Singapore",
  },
  {
    title: "Animal Nutritionist",
    organization: "Alpha Multitrade (Feedconcept)",
    startYear: 2012,
    endYear: 2014,
    location: "Singapore",
  },
  {
    title: "Technical Services Associate",
    organization: "Glenwood Technologies International, Inc.",
    startYear: 2010,
    endYear: 2011,
    location: "Singapore",
  },
];

export const SANDRA_CAREER_SUMMARY =
  "Sandra Agcaoili is a Filipino Licensed Agriculturist registered with the Professional Regulation Commission (PRC) of the Philippines and is based in Singapore. She earned a Bachelor of Science in Agriculture from the University of the Philippines (2005–2009) and is pursuing a Doctor of Philosophy in Artificial Intelligence at the University of the Philippines Diliman (2023–present). Her earlier career spans animal nutrition and technical sales in Singapore — Technical Specialist (Animal Nutrition) at Zagro Singapore Pte Ltd (2014–2018), Animal Nutritionist at Alpha Multitrade (Feedconcept) (2012–2014), and Technical Services Associate at Glenwood Technologies International, Inc. (2010–2011).";

/** Canonical long-form biography (meta, JSON-LD, profile pages). */
export const SANDRA_BIO =
  `${SANDRA_CAREER_SUMMARY} She is Research Partner at Balmores Lab (balmoreslab.com), where she collaborates with Louie Doniego Balmores on machine learning, analytics, and trustworthy AI for engineering — including neural surrogate models, physics-informed deep learning, privacy-preserving on-device inference, and reproducible evaluation frameworks that connect open-source finite-element analysis (PyNite) with production structural workflows. She is a member of the Analytics and Artificial Intelligence Association of the Philippines (AAP). Official profile: ${SANDRA_PROFILE_URL}.`;

export const SANDRA_SHORT_BIO =
  "AI Researcher · Licensed Agriculturist (PRC Philippines) · Singapore · PhD AI (UP Diliman) · BS Agriculture (UP) · Research Partner, Balmores Lab";

export const SANDRA_AGCAOILI = {
  slug: SANDRA_SLUG,
  profilePath: SANDRA_PROFILE_PATH,
  profileUrl: SANDRA_PROFILE_URL,
  id: SANDRA_PERSON_ID,
  profilePageId: SANDRA_PROFILE_PAGE_ID,
  name: "Sandra Agcaoili",
  givenName: "Sandra",
  familyName: "Agcaoili",
  alternateName: getSandraAlternateNames(),
  jobTitle: "AI Researcher",
  professionalTitle: "Licensed Agriculturist (PRC Philippines)",
  role: "Research Partner, Balmores Lab",
  nationality: "Philippines",
  location: "Singapore",
  homeLocation: {
    "@type": "Place" as const,
    name: "Singapore",
    address: {
      "@type": "PostalAddress" as const,
      addressLocality: "Singapore",
      addressCountry: "SG",
    },
  },
  workLocation: {
    "@type": "Place" as const,
    name: "Singapore",
    address: {
      "@type": "PostalAddress" as const,
      addressLocality: "Singapore",
      addressCountry: "SG",
    },
  },
  alumniOf: {
    "@type": "EducationalOrganization" as const,
    name: "University of the Philippines Diliman",
    alternateName: "UP Diliman",
    url: "https://upd.edu.ph",
    department: "Doctor of Philosophy (PhD) in Artificial Intelligence",
  },
  credential: {
    name: "Doctor of Philosophy (PhD) in Artificial Intelligence",
    startDate: "2023-09",
    status: "In progress",
    institution: "University of the Philippines Diliman",
  },
  societies: [
    {
      name: "Analytics and Artificial Intelligence Association of the Philippines",
      alternateName: "AAP",
    },
  ],
  knowsAbout: [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Analytics",
    "AI for Engineering Applications",
    "Structural AI Research",
    "Physics-Informed Neural Networks",
    "Privacy-Preserving Machine Learning",
    "Natural Language Processing",
    "Computational Design",
    "Animal Nutrition",
    "Agriculture",
    "Licensed Agriculturist",
    "Professional Regulation Commission",
    "Feed Formulation",
    "Technical Sales Support",
    "Product Development",
    "Singapore",
  ],
  prcLicense: SANDRA_PRC_LICENSE,
  skills: SANDRA_SKILLS,
  careerSummary: SANDRA_CAREER_SUMMARY,
  education: SANDRA_EDUCATION,
  workHistory: SANDRA_WORK_HISTORY,
  hasOccupation: {
    name: "AI Researcher",
    skills:
      "Deep learning, neural surrogate models, AI pipeline evaluation, privacy-preserving on-device inference, analytics for engineering applications",
  },
  bio: SANDRA_BIO,
  shortBio: SANDRA_SHORT_BIO,
  collaborationFocus:
    "Leads co-development of neural surrogate architectures, reproducible evaluation protocols, and privacy-aware AI pipelines for Balmores Lab’s natural-language-to-FEM and structural-optimization programme, in collaboration with Louie Doniego Balmores.",
  linkedinUrl: SANDRA_LINKEDIN_URL,
  orcidId: SANDRA_ORCID_ID,
  orcidUrl: SANDRA_ORCID_URL,
  researchgateUrl: SANDRA_RESEARCHGATE_URL,
  rocketreachUrl: SANDRA_ROCKETREACH_URL,
  contactoutUrl: SANDRA_CONTACTOUT_URL,
  researchAreas: [
    "Neural surrogate models for structural FEM",
    "Privacy-preserving on-device AI for engineering",
    "Physics-informed deep learning",
    "AI pipeline evaluation and benchmarking",
    "Analytics for scientific and structural computing",
  ],
};

export type SandraSociety = { name: string; alternateName?: string };

export function formatSandraYearRange(
  start: number | null,
  end: number | null,
  current?: boolean,
): string {
  if (current) return "Present";
  if (start != null && end != null) return `${start} – ${end}`;
  if (start != null) return `${start} – Present`;
  return "";
}

export function formatSandraSocieties(
  societies: readonly SandraSociety[] = SANDRA_AGCAOILI.societies,
): string {
  return societies
    .map((s) => (s.alternateName ? `${s.name} (${s.alternateName})` : s.name))
    .join(" · ");
}

export function getSandraExtraSameAs(): string[] {
  const env = (typeof process !== "undefined" ? process.env : {}) as Record<string, string | undefined>;
  const urls: string[] = [
    SANDRA_LINKEDIN_URL,
    SANDRA_ORCID_URL,
    SANDRA_RESEARCHGATE_URL,
    SANDRA_ROCKETREACH_URL,
    SANDRA_CONTACTOUT_URL,
  ];
  const push = (u: string | undefined) => {
    if (u) urls.push(u);
  };
  push(env.NEXT_PUBLIC_SANDRA_LINKEDIN_URL);
  push(env.NEXT_PUBLIC_SANDRA_ORCID_URL);
  push(env.NEXT_PUBLIC_SANDRA_SCHOLAR_URL);
  push(env.NEXT_PUBLIC_SANDRA_RESEARCHGATE_URL);
  const orcid = env.NEXT_PUBLIC_SANDRA_ORCID_ID;
  if (orcid && /^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$/.test(orcid)) {
    urls.push(`https://orcid.org/${orcid}`);
  }
  return Array.from(new Set(urls));
}

export function getSandraSameAs(): string[] {
  return Array.from(
    new Set([
      SITE_ROOT,
      SANDRA_PROFILE_URL,
      `${SITE_URL}/research`,
      `${SITE_URL}/about`,
      `${SITE_URL}/sandra-agcaoili-schema.json`,
      `${SITE_URL}/sandra-agcaoili.foaf.rdf`,
      "https://upd.edu.ph",
      SANDRA_PRC_URL,
      ...getSandraExtraSameAs(),
    ]),
  );
}

export function buildSandraHasCredentials() {
  const prc = {
    "@type": "EducationalOccupationalCredential" as const,
    credentialCategory: SANDRA_PRC_LICENSE.credentialCategory,
    name: SANDRA_PRC_LICENSE.title,
    recognizedBy: {
      "@type": "GovernmentOrganization" as const,
      name: SANDRA_PRC_LICENSE.issuer,
      url: SANDRA_PRC_LICENSE.issuerUrl,
      address: {
        "@type": "PostalAddress" as const,
        addressCountry: "PH",
      },
    },
  };
  const degrees = SANDRA_EDUCATION.map((edu) => ({
    "@type": "EducationalOccupationalCredential" as const,
    credentialCategory: edu.status === "In progress" ? "Doctoral degree (in progress)" : "Bachelor's degree",
    name: `${edu.degree} (${edu.field})`,
    dateCreated: edu.startYear.toString(),
    ...(edu.endYear ? { validUntil: edu.endYear.toString() } : {}),
    recognizedBy: {
      "@type": "EducationalOrganization" as const,
      name: edu.institution,
      url: edu.institutionUrl,
    },
  }));
  return [prc, ...degrees];
}

export function buildSandraPersonNode() {
  return {
    "@type": "Person",
    "@id": SANDRA_PERSON_ID,
    name: SANDRA_AGCAOILI.name,
    givenName: SANDRA_AGCAOILI.givenName,
    familyName: SANDRA_AGCAOILI.familyName,
    alternateName: getSandraAlternateNames(),
    jobTitle: [SANDRA_AGCAOILI.jobTitle, SANDRA_AGCAOILI.professionalTitle],
    description: SANDRA_AGCAOILI.bio,
    honorificCredential: SANDRA_PRC_LICENSE.title,
    url: SANDRA_PROFILE_URL,
    mainEntityOfPage: { "@id": SANDRA_PROFILE_PAGE_ID },
    identifier: [
      {
        "@type": "PropertyValue",
        propertyID: "balmoreslab-profile",
        value: SANDRA_PROFILE_URL,
      },
      {
        "@type": "PropertyValue",
        propertyID: "ORCID",
        value: SANDRA_ORCID_ID,
        url: SANDRA_ORCID_URL,
      },
    ],
    image: {
      "@type": "ImageObject",
      url: `${SANDRA_PROFILE_URL}/opengraph-image`,
      caption: `${SANDRA_AGCAOILI.name} - ${SANDRA_AGCAOILI.jobTitle}`,
    },
    nationality: { "@type": "Country", name: SANDRA_AGCAOILI.nationality },
    homeLocation: SANDRA_AGCAOILI.homeLocation,
    workLocation: SANDRA_AGCAOILI.workLocation,
    alumniOf: SANDRA_EDUCATION.map((edu) => ({
      "@type": "EducationalOrganization",
      name: edu.institution,
      url: edu.institutionUrl,
      department: `${edu.degree} — ${edu.field}`,
    })),
    knowsAbout: SANDRA_AGCAOILI.knowsAbout,
    knowsLanguage: ["English", "Filipino"],
    hasOccupation: [
      {
        "@type": "Occupation",
        name: SANDRA_AGCAOILI.hasOccupation.name,
        skills: SANDRA_AGCAOILI.hasOccupation.skills,
        occupationLocation: SANDRA_AGCAOILI.workLocation,
      },
      ...SANDRA_WORK_HISTORY.filter((w) => !w.current).map((w) => ({
        "@type": "Occupation",
        name: w.title,
        occupationLocation: {
          "@type": "Place",
          name: w.location,
        },
      })),
    ],
    hasCredential: buildSandraHasCredentials(),
    memberOf: {
      "@type": "Organization",
      name: SANDRA_AGCAOILI.societies[0].name,
      alternateName: SANDRA_AGCAOILI.societies[0].alternateName,
    },
    colleague: { "@id": PERSON_ID },
    worksFor: {
      "@id": ORG_ID,
      name: "Balmores Lab",
      url: SITE_ROOT,
      sameAs: SITE_ROOT,
    },
    affiliation: { "@id": ORG_ID },
    sameAs: getSandraSameAs(),
    subjectOf: { "@id": SANDRA_PROFILE_PAGE_ID },
  };
}

export function buildSandraProfilePageNode() {
  return {
    "@type": "ProfilePage",
    "@id": SANDRA_PROFILE_PAGE_ID,
    url: SANDRA_PROFILE_URL,
    name: `About ${SANDRA_AGCAOILI.name} - AI Researcher`,
    about: { "@id": SANDRA_PERSON_ID },
    mainEntity: { "@id": SANDRA_PERSON_ID },
    isPartOf: { "@id": `${SITE_URL}/#website` },
    dateModified: new Date().toISOString().split("T")[0],
  };
}

export const SANDRA_FAQ = [
  {
    question: "What is Sandra Agcaoili's official website?",
    answer:
      "Sandra Agcaoili's official web presence is published through Balmores Lab at https://www.balmoreslab.com. Her canonical person profile for search engines and Knowledge Panel is https://www.balmoreslab.com/about/sandra-agcaoili. Machine-readable identity: https://www.balmoreslab.com/sandra-agcaoili-schema.json.",
  },
  {
    question: "Who is Sandra Agcaoili?",
    answer: `${SANDRA_BIO}`,
  },
  {
    question: "Is Sandra Agcaoili a licensed agriculturist?",
    answer:
      "Yes. Sandra Agcaoili is a Licensed Agriculturist in the Philippines, registered with the Professional Regulation Commission (PRC) at https://prc.gov.ph. This credential complements her Bachelor of Science in Agriculture (University of the Philippines, 2005–2009) and her doctoral research in Artificial Intelligence at UP Diliman. Official profile: https://www.balmoreslab.com/about/sandra-agcaoili.",
  },
  {
    question: "Where is Sandra Agcaoili located?",
    answer:
      "Sandra Agcaoili is based in Singapore. Her canonical profile for search engines is published at https://www.balmoreslab.com/about/sandra-agcaoili on https://www.balmoreslab.com.",
  },
  {
    question: "What is Sandra Agcaoili's education?",
    answer:
      "Sandra Agcaoili holds a Bachelor of Science in Agriculture from the University of the Philippines (2005–2009) and is a doctoral candidate in Artificial Intelligence at the University of the Philippines Diliman (2023–present).",
  },
  {
    question: "Where does Sandra Agcaoili study?",
    answer:
      "Sandra Agcaoili is a doctoral candidate in Artificial Intelligence at the University of the Philippines Diliman (UP Diliman), enrolled from September 2023. She conducts collaborative research with Balmores Lab while based in Singapore.",
  },
  {
    question: "What was Sandra Agcaoili's work before AI research?",
    answer:
      "In Singapore she worked in animal nutrition: Technical Specialist (Animal Nutrition) at Zagro Singapore Pte Ltd (2014–2018), Animal Nutritionist at Alpha Multitrade (Feedconcept) (2012–2014), and Technical Services Associate at Glenwood Technologies International, Inc. (2010–2011), before her Research Partner role at Balmores Lab.",
  },
  {
    question: "What is Sandra Agcaoili's role at Balmores Lab?",
    answer:
      "She serves as Research Partner at Balmores Lab, leading methodology for neural surrogates, physics-informed models, privacy-preserving on-device inference, and evaluation frameworks for AI-driven structural engineering, alongside Louie Doniego Balmores.",
  },
  {
    question: "What organizations is Sandra Agcaoili affiliated with?",
    answer:
      "She is a Licensed Agriculturist under the Professional Regulation Commission (PRC) of the Philippines, a member of the Analytics and Artificial Intelligence Association of the Philippines (AAP), a doctoral researcher at the University of the Philippines Diliman, and Research Partner at Balmores Lab. Profile: https://www.balmoreslab.com/about/sandra-agcaoili.",
  },
  {
    question: "What does Sandra Agcaoili research?",
    answer:
      "Her work spans neural surrogate models for finite-element analysis, physics-informed deep learning, privacy-preserving on-device AI for engineering workflows, and analytics for structural computing — published through the Balmores Lab programme at https://www.balmoreslab.com/research.",
  },
];

export function buildSandraFaqLd() {
  return {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    "@id": SANDRA_FAQ_ID,
    mainEntity: SANDRA_FAQ.map((item) => ({
      "@type": "Question",
      name: item.question,
      acceptedAnswer: {
        "@type": "Answer",
        text: item.answer,
      },
    })),
  };
}

/** Full @graph for Sandra's dedicated profile page */
export function buildSandraGraph() {
  return {
    "@context": "https://schema.org",
    "@graph": [
      buildSandraPersonNode(),
      buildSandraProfilePageNode(),
      {
        "@type": "AboutPage",
        "@id": `${SANDRA_PROFILE_URL}#aboutpage`,
        url: SANDRA_PROFILE_URL,
        name: `About ${SANDRA_AGCAOILI.name}`,
        mainEntity: { "@id": SANDRA_PERSON_ID },
        about: { "@id": SANDRA_PERSON_ID },
        isPartOf: { "@id": `${SITE_URL}/#website` },
      },
      {
        "@type": "WebSite",
        "@id": `${SITE_URL}/#website`,
        url: SITE_ROOT,
        name: "Balmores Lab",
        publisher: { "@id": PERSON_ID },
        about: [{ "@id": PERSON_ID }, { "@id": SANDRA_PERSON_ID }],
      },
      {
        "@type": "Organization",
        "@id": ORG_ID,
        name: "Balmores Lab",
        url: SITE_ROOT,
        employee: [{ "@id": SANDRA_PERSON_ID }],
      },
    ],
  };
}

/** Single Person block — used on pages that mention Sandra inline */
export function sandraPersonLd() {
  return buildSandraPersonNode();
}

export const RESEARCH_TEAM = [SANDRA_AGCAOILI];

export function buildSandraSchemaJson() {
  return {
    "@context": "https://schema.org",
    ...buildSandraPersonNode(),
  };
}
