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

export const SANDRA_AGCAOILI = {
  slug: SANDRA_SLUG,
  profilePath: SANDRA_PROFILE_PATH,
  profileUrl: SANDRA_PROFILE_URL,
  id: SANDRA_PERSON_ID,
  profilePageId: SANDRA_PROFILE_PAGE_ID,
  name: "Sandra Agcaoili",
  givenName: "Sandra",
  familyName: "Agcaoili",
  alternateName: ["Sandra A. Agcaoili"],
  jobTitle: "AI Researcher",
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
  ],
  hasOccupation: {
    name: "AI Researcher",
    skills:
      "Deep learning, neural surrogate models, AI pipeline evaluation, privacy-preserving on-device inference, analytics for engineering applications",
  },
  bio:
    "Sandra Agcaoili is a Filipino AI researcher and research partner at Balmores Lab. She is pursuing a Doctor of Philosophy (PhD) in Artificial Intelligence at the University of the Philippines Diliman (September 2023 – present) and is an active member of the Analytics and Artificial Intelligence Association of the Philippines (AAP). Based in Singapore, she collaborates with Louie Doniego Balmores on AI-driven structural engineering — neural surrogates, privacy-preserving on-device inference, physics-informed models, and evaluation frameworks that connect classical finite-element workflows with modern deep learning.",
  shortBio:
    "AI Researcher · PhD (Artificial Intelligence), UP Diliman · Research Partner, Balmores Lab · Based in Singapore",
  collaborationFocus:
    "Co-developing surrogate-model architectures, evaluation protocols, and AI pipeline design for Balmores Lab's structural optimization and natural-language-to-FEM research programme.",
  researchAreas: [
    "Neural surrogate models for structural FEM",
    "Privacy-preserving on-device AI for engineering",
    "Physics-informed deep learning",
    "AI pipeline evaluation and benchmarking",
    "Analytics for scientific and structural computing",
  ],
};

export function getSandraExtraSameAs(): string[] {
  const env = (typeof process !== "undefined" ? process.env : {}) as Record<string, string | undefined>;
  const urls: string[] = [];
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
  return urls;
}

export function getSandraSameAs(): string[] {
  return Array.from(
    new Set([
      SANDRA_PROFILE_URL,
      `${SITE_URL}/research`,
      `${SITE_URL}/about`,
      "https://upd.edu.ph",
      ...getSandraExtraSameAs(),
    ]),
  );
}

export function buildSandraPersonNode() {
  const today = new Date().toISOString().split("T")[0];
  return {
    "@type": "Person",
    "@id": SANDRA_PERSON_ID,
    name: SANDRA_AGCAOILI.name,
    givenName: SANDRA_AGCAOILI.givenName,
    familyName: SANDRA_AGCAOILI.familyName,
    alternateName: SANDRA_AGCAOILI.alternateName,
    jobTitle: SANDRA_AGCAOILI.jobTitle,
    description: SANDRA_AGCAOILI.bio,
    url: SANDRA_PROFILE_URL,
    mainEntityOfPage: { "@id": SANDRA_PROFILE_PAGE_ID },
    image: {
      "@type": "ImageObject",
      url: `${SANDRA_PROFILE_URL}/opengraph-image`,
      caption: `${SANDRA_AGCAOILI.name} - ${SANDRA_AGCAOILI.jobTitle}`,
    },
    nationality: { "@type": "Country", name: SANDRA_AGCAOILI.nationality },
    homeLocation: SANDRA_AGCAOILI.homeLocation,
    workLocation: SANDRA_AGCAOILI.workLocation,
    alumniOf: SANDRA_AGCAOILI.alumniOf,
    knowsAbout: SANDRA_AGCAOILI.knowsAbout,
    knowsLanguage: ["English", "Filipino"],
    hasOccupation: {
      "@type": "Occupation",
      name: SANDRA_AGCAOILI.hasOccupation.name,
      skills: SANDRA_AGCAOILI.hasOccupation.skills,
    },
    hasCredential: {
      "@type": "EducationalOccupationalCredential",
      credentialCategory: "Doctoral degree (in progress)",
      name: SANDRA_AGCAOILI.credential.name,
      dateCreated: SANDRA_AGCAOILI.credential.startDate,
      recognizedBy: SANDRA_AGCAOILI.alumniOf,
    },
    memberOf: {
      "@type": "Organization",
      name: SANDRA_AGCAOILI.societies[0].name,
      alternateName: SANDRA_AGCAOILI.societies[0].alternateName,
    },
    colleague: { "@id": PERSON_ID },
    worksFor: { "@id": ORG_ID },
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
    question: "Who is Sandra Agcaoili?",
    answer:
      "Sandra Agcaoili is a Filipino AI researcher and research partner at Balmores Lab (balmoreslab.com). She specializes in artificial intelligence for engineering applications, including neural surrogates and privacy-preserving on-device inference for structural analysis.",
  },
  {
    question: "Where does Sandra Agcaoili study?",
    answer:
      "Sandra Agcaoili is pursuing a Doctor of Philosophy (PhD) in Artificial Intelligence at the University of the Philippines Diliman, having started in September 2023. She is currently based in Singapore while conducting collaborative research with Balmores Lab.",
  },
  {
    question: "What is Sandra Agcaoili's role at Balmores Lab?",
    answer:
      "She is a research partner at Balmores Lab, co-developing AI pipeline architecture, surrogate-model evaluation protocols, and deep-learning methods for AI-driven structural engineering alongside Louie Doniego Balmores.",
  },
  {
    question: "What organizations is Sandra Agcaoili affiliated with?",
    answer:
      "She is a member of the Analytics and Artificial Intelligence Association of the Philippines (AAP) and a doctoral researcher at the University of the Philippines Diliman. Her official profile is published at https://www.balmoreslab.com/about/sandra-agcaoili.",
  },
  {
    question: "What does Sandra Agcaoili research?",
    answer:
      "Her research spans neural surrogate models for finite-element analysis, physics-informed deep learning, privacy-preserving on-device AI for engineering workflows, and analytics for structural computing — applied through the Balmores Lab research programme at balmoreslab.com/research.",
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
