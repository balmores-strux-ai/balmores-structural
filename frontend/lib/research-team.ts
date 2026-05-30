import { SITE_URL } from "./seo";

export const SANDRA_AGCAOILI = {
  id: `${SITE_URL}/about#sandra-agcaoili`,
  name: "Sandra Agcaoili",
  givenName: "Sandra",
  familyName: "Agcaoili",
  jobTitle: "AI Researcher",
  role: "Research Partner, Balmores Lab",
  location: "Singapore",
  homeLocation: {
    "@type": "Place" as const,
    name: "Singapore",
    address: {
      "@type": "PostalAddress" as const,
      addressCountry: "SG",
    },
  },
  alumniOf: {
    "@type": "EducationalOrganization" as const,
    name: "University of the Philippines Diliman",
    url: "https://upd.edu.ph",
    department: "Artificial Intelligence (PhD programme)",
  },
  credential: {
    name: "Doctor of Philosophy (PhD) in Artificial Intelligence",
    startDate: "2023-09",
    status: "In progress",
    institution: "University of the Philippines Diliman",
  },
  societies: ["Analytics and Artificial Intelligence Association of the Philippines (AAP)"],
  knowsAbout: [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Analytics",
    "AI for Engineering Applications",
    "Structural AI Research",
  ],
  bio:
    "Sandra Agcaoili is an AI researcher and research partner at Balmores Lab. She is pursuing a Doctor of Philosophy (PhD) in Artificial Intelligence at the University of the Philippines Diliman (September 2023 – present) and is an active member of the Analytics and Artificial Intelligence Association of the Philippines (AAP). Based in Singapore, she collaborates with Louie Doniego Balmores on AI-driven structural engineering — neural surrogates, privacy-preserving on-device inference, and physics-informed models that bridge classical FEM workflows with modern deep learning.",
  shortBio:
    "AI Researcher · PhD (Artificial Intelligence), UP Diliman · Research Partner, Balmores Lab · Based in Singapore",
  collaborationFocus:
    "Co-developing surrogate-model architectures, evaluation protocols, and AI pipeline design for Balmores Lab's structural optimization and NL-to-FEM research programme.",
};

export function sandraPersonLd() {
  return {
    "@type": "Person",
    "@id": SANDRA_AGCAOILI.id,
    name: SANDRA_AGCAOILI.name,
    givenName: SANDRA_AGCAOILI.givenName,
    familyName: SANDRA_AGCAOILI.familyName,
    jobTitle: SANDRA_AGCAOILI.jobTitle,
    description: SANDRA_AGCAOILI.bio,
    homeLocation: SANDRA_AGCAOILI.homeLocation,
    alumniOf: SANDRA_AGCAOILI.alumniOf,
    knowsAbout: SANDRA_AGCAOILI.knowsAbout,
    colleague: { "@id": `${SITE_URL}/#person` },
    worksFor: { "@id": `${SITE_URL}/#organization` },
    memberOf: {
      "@type": "Organization",
      name: "Analytics and Artificial Intelligence Association of the Philippines (AAP)",
    },
    hasCredential: {
      "@type": "EducationalOccupationalCredential",
      credentialCategory: "Doctoral degree (in progress)",
      name: SANDRA_AGCAOILI.credential.name,
      recognizedBy: SANDRA_AGCAOILI.alumniOf,
    },
  };
}

export const RESEARCH_TEAM = [SANDRA_AGCAOILI];
