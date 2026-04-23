import "./globals.css";
import type { Metadata, Viewport } from "next";

const SITE_URL = "https://www.balmoreslab.com";
const SITE_NAME = "Balmores Laboratory";
const PERSON_NAME = "Louie Doniego Balmores";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "Louie Doniego Balmores — Structural Engineer & AI Researcher | Balmores Laboratory",
    template: "%s | Balmores Laboratory",
  },
  description:
    "Louie Doniego Balmores is a licensed Civil Engineer (PRC Philippines, 2013) with over 10 years of structural engineering experience and an AI researcher pioneering AI-driven structural optimization at Balmores Laboratory (balmoreslab.com).",
  applicationName: SITE_NAME,
  authors: [{ name: PERSON_NAME, url: SITE_URL }],
  creator: PERSON_NAME,
  publisher: PERSON_NAME,
  generator: "Next.js",
  referrer: "origin-when-cross-origin",
  keywords: [
    "Louie Balmores",
    "Louie Doniego Balmores",
    "Balmores Structural",
    "Balmores Laboratory",
    "balmoreslab",
    "Structural Engineer",
    "AI Researcher",
    "Civil Engineer Philippines",
    "PRC Civil Engineer",
    "Structural AI",
    "PyNite",
    "Computational Structural Design",
    "Structural Optimization AI",
    "Professional Engineers Ontario",
    "P.Eng Candidate",
  ],
  category: "Structural Engineering & AI Research",
  alternates: {
    canonical: "/",
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: SITE_URL,
    siteName: SITE_NAME,
    title: "Louie Doniego Balmores — Structural Engineer & AI Researcher",
    description:
      "Licensed Civil Engineer (PRC Philippines) with 10+ years of structural design experience, researching AI-driven structural optimization at Balmores Laboratory.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Louie Doniego Balmores — Structural Engineer & AI Researcher",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    site: "@louiedbalmores",
    creator: "@louiedbalmores",
    title: "Louie Doniego Balmores — Structural Engineer & AI Researcher",
    description:
      "Licensed Civil Engineer researching AI-driven structural optimization. PyNite 3D FEA from a design brief.",
    images: ["/og-image.png"],
  },
  robots: {
    index: true,
    follow: true,
    nocache: false,
    googleBot: {
      index: true,
      follow: true,
      "max-snippet": -1,
      "max-image-preview": "large",
      "max-video-preview": -1,
    },
  },
  verification: {
    // Owner can plug a real Search Console token here without breaking the build.
    google: process.env.NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION || undefined,
  },
  icons: {
    icon: "/favicon.ico",
    shortcut: "/favicon.ico",
    apple: "/apple-touch-icon.png",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#080a0f",
  colorScheme: "dark",
};

// --- Structured data (JSON-LD) --------------------------------------------
// Google's Knowledge Graph pipelines prefer a small number of well-linked
// entities. We publish four linked nodes in a single @graph so crawlers read
// the Person, the Website that hosts them, the Organization they run, and the
// ProfilePage as one coherent entity. All nodes cross-reference via @id.

const personId = `${SITE_URL}/#person`;
const orgId = `${SITE_URL}/#organization`;
const siteId = `${SITE_URL}/#website`;
const profilePageId = `${SITE_URL}/about#profilepage`;

// Extra identity-graph refs injected only when the corresponding env var
// is set on the host (Render). New account? Add its env var, redeploy —
// no code change needed. Empty values are filtered out.
const EXTRA_SAME_AS = (() => {
  const env = (typeof process !== "undefined" ? process.env : {}) as Record<string, string | undefined>;
  const urls: string[] = [];
  const wikidata = env.NEXT_PUBLIC_WIKIDATA_ID;
  if (wikidata && /^Q\d+$/.test(wikidata)) urls.push(`https://www.wikidata.org/wiki/${wikidata}`);
  const orcid = env.NEXT_PUBLIC_ORCID_ID;
  if (orcid && /^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$/.test(orcid)) urls.push(`https://orcid.org/${orcid}`);
  const github = env.NEXT_PUBLIC_GITHUB_URL;
  if (github) urls.push(github);
  const keybase = env.NEXT_PUBLIC_KEYBASE_URL;
  if (keybase) urls.push(keybase);
  const mastodon = env.NEXT_PUBLIC_MASTODON_URL;
  if (mastodon) urls.push(mastodon);
  const scholar = env.NEXT_PUBLIC_SCHOLAR_URL;
  if (scholar) urls.push(scholar);
  const gravatar = env.NEXT_PUBLIC_GRAVATAR_URL;
  if (gravatar) urls.push(gravatar);
  const researchgate = env.NEXT_PUBLIC_RESEARCHGATE_URL;
  if (researchgate) urls.push(researchgate);
  const youtube = env.NEXT_PUBLIC_YOUTUBE_URL;
  if (youtube) urls.push(youtube);
  const devto = env.NEXT_PUBLIC_DEVTO_URL;
  if (devto) urls.push(devto);
  const stackoverflow = env.NEXT_PUBLIC_STACKOVERFLOW_URL;
  if (stackoverflow) urls.push(stackoverflow);
  return urls;
})();

const BASE_SAME_AS = [
  "https://www.linkedin.com/in/louiebalmores/",
  "https://x.com/louiedbalmores",
  "https://twitter.com/louiedbalmores",
  "https://about.me/louiebalmoresdesign/",
  "https://worldchess.com/profile/422673",
  "https://prc.gov.ph",
];

const ALL_SAME_AS = Array.from(new Set([...BASE_SAME_AS, ...EXTRA_SAME_AS]));

const structuredData = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "Person",
      "@id": personId,
      name: "Louie Doniego Balmores",
      givenName: "Louie",
      additionalName: "Doniego",
      familyName: "Balmores",
      alternateName: ["Louie Balmores", "Engr. Louie Balmores", "Louie D. Balmores"],
      honorificSuffix: "P.Eng (Candidate), PE (Candidate)",
      jobTitle: "Structural Engineer & AI Researcher",
      description:
        "Licensed Civil Engineer (PRC Philippines, 2013) with over 10 years of experience in high-performance structural design, currently pioneering AI-driven structural optimization, computational design, and material efficiency research at Balmores Laboratory.",
      url: SITE_URL,
      mainEntityOfPage: { "@id": profilePageId },
      image: {
        "@type": "ImageObject",
        url: `${SITE_URL}/louie-balmores.jpg`,
        caption: "Louie Doniego Balmores — Structural Engineer & AI Researcher",
      },
      gender: "Male",
      nationality: { "@type": "Country", name: "Philippines" },
      worksFor: { "@id": orgId },
      founder: { "@id": orgId },
      knowsLanguage: ["English", "Filipino"],
      knowsAbout: [
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
      ],
      hasOccupation: [
        {
          "@type": "Occupation",
          name: "Structural Engineer",
          occupationLocation: { "@type": "Country", name: "Philippines" },
          estimatedSalary: undefined,
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
          identifier: "Nov 2013 Civil Engineer Licensure Examination — Sequence No. 350",
        },
        {
          "@type": "EducationalOccupationalCredential",
          credentialCategory: "Professional License (Candidate)",
          name: "P.Eng License Candidate",
          recognizedBy: {
            "@type": "Organization",
            name: "Professional Engineers Ontario (PEO)",
            url: "https://www.peo.on.ca",
          },
          validFrom: "2027-01-01",
        },
        {
          "@type": "EducationalOccupationalCredential",
          credentialCategory: "Professional License (Candidate)",
          name: "PE License Candidate",
          recognizedBy: {
            "@type": "Organization",
            name: "NCEES — National Council of Examiners for Engineering and Surveying",
            url: "https://ncees.org",
          },
          validFrom: "2028-01-01",
        },
      ],
      sameAs: ALL_SAME_AS,
    },
    {
      "@type": "Organization",
      "@id": orgId,
      name: "Balmores Laboratory",
      alternateName: ["Balmores Strux AI", "Balmores Structural"],
      url: SITE_URL,
      logo: `${SITE_URL}/logo.png`,
      founder: { "@id": personId },
      sameAs: [
        "https://www.linkedin.com/in/louiebalmores/",
        "https://x.com/louiedbalmores",
      ],
      description:
        "Balmores Laboratory researches AI-driven structural engineering — converting traditional structural analysis workflows into automated, intelligent systems.",
      knowsAbout: [
        "Structural AI",
        "Computational Structural Design",
        "Finite Element Analysis",
        "Structural Optimization",
      ],
    },
    {
      "@type": "WebSite",
      "@id": siteId,
      url: SITE_URL,
      name: SITE_NAME,
      description:
        "Official website of Louie Doniego Balmores and Balmores Laboratory — AI-driven structural engineering.",
      publisher: { "@id": personId },
      inLanguage: "en",
      potentialAction: {
        "@type": "SearchAction",
        target: {
          "@type": "EntryPoint",
          urlTemplate: `${SITE_URL}/?q={search_term_string}`,
        },
        "query-input": "required name=search_term_string",
      },
    },
    {
      "@type": "ProfilePage",
      "@id": profilePageId,
      url: `${SITE_URL}/about`,
      name: "About Louie Doniego Balmores",
      about: { "@id": personId },
      mainEntity: { "@id": personId },
      isPartOf: { "@id": siteId },
      dateModified: new Date().toISOString().split("T")[0],
    },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        {/* IndieWeb / rel=me links: some crawlers and verification services use
            these to build identity graphs across platforms. */}
        <link rel="me" href="https://www.linkedin.com/in/louiebalmores/" />
        <link rel="me" href="https://x.com/louiedbalmores" />
        <link rel="me" href="https://about.me/louiebalmoresdesign/" />
        <link rel="me" href="https://worldchess.com/profile/422673" />
        {EXTRA_SAME_AS.map((href) => (
          <link key={href} rel="me" href={href} />
        ))}
        <link rel="author" href={`${SITE_URL}/about`} />
        <link rel="publisher" href={SITE_URL} />

        {/* Dublin Core metadata — indexed by some academic / semantic crawlers. */}
        <meta name="DC.creator" content={PERSON_NAME} />
        <meta name="DC.publisher" content={SITE_NAME} />
        <meta name="DC.subject" content="Structural Engineering, Artificial Intelligence, Computational Design" />

        {/* Plain author meta — still read by Google for attribution heuristics. */}
        <meta name="author" content={PERSON_NAME} />
        <meta name="copyright" content={`© ${new Date().getFullYear()} ${PERSON_NAME}`} />

        {/* Hint Google Knowledge Graph about the canonical identity URL. */}
        <meta property="profile:first_name" content="Louie" />
        <meta property="profile:last_name" content="Balmores" />
        <meta property="profile:username" content="louiedbalmores" />

        {/* IndexNow — instant crawl protocol (Bing, Yandex, Seznam, and now piloted by Google).
            The keyed file at /ykjm52si9r4gfvwhul8ob7cd3nqxpe01.txt proves ownership. */}
        <meta name="indexnow" content="ykjm52si9r4gfvwhul8ob7cd3nqxpe01" />

        {/* Expose the raw JSON-LD as a discoverable alternate — helps AI crawlers and
            linked-data indexers ingest the entity graph directly. */}
        <link rel="alternate" type="application/ld+json" href="/seo-schema.json" />
        <link rel="alternate" type="application/atom+xml" title="Balmores Laboratory feed" href="/feed.xml" />
        <link rel="alternate" type="application/rdf+xml" title="FOAF profile" href="/foaf.rdf" />

        {/* Pre-warm connections to the profile platforms referenced in sameAs,
            so bots that follow links don't stall on DNS/TLS handshakes. */}
        <link rel="dns-prefetch" href="//www.linkedin.com" />
        <link rel="dns-prefetch" href="//x.com" />
        <link rel="dns-prefetch" href="//prc.gov.ph" />

        {/* Plain <script> tag so the JSON-LD lands inside the initial static
            HTML that every crawler sees (including Bing's first pass, AI
            entity-extractors, and non-JS-executing indexers). Using next/script
            with strategy="beforeInteractive" would defer injection via the RSC
            streaming payload, which breaks static entity discovery. */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
