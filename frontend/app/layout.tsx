import "./globals.css";
import type { Metadata, Viewport } from "next";
import {
  buildRootStructuredData,
  DEFAULT_DESCRIPTION,
  getExtraSameAs,
  JOB_TITLE,
  PERSON_NAME,
  SHORT_DESCRIPTION,
  SITE_NAME,
  SITE_URL,
} from "@/lib/seo";
import { getSandraExtraSameAs, SANDRA_AGCAOILI, SANDRA_PROFILE_URL } from "@/lib/research-team";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "Louie Doniego Balmores - Structural Engineer & AI Researcher | Balmores Lab",
    template: "%s | Balmores Lab",
  },
  description: DEFAULT_DESCRIPTION,
  applicationName: SITE_NAME,
  authors: [
    { name: PERSON_NAME, url: `${SITE_URL}/about` },
    { name: SANDRA_AGCAOILI.name, url: SANDRA_PROFILE_URL },
  ],
  creator: PERSON_NAME,
  publisher: PERSON_NAME,
  generator: "Next.js",
  referrer: "origin-when-cross-origin",
  keywords: [
    "Louie Balmores",
    "Louie Doniego Balmores",
    "Luis Balmores",
    "Lui Balmores",
    "Loui Balmores",
    "Lui Doniego Balmores",
    "Loui Doniego Balmores",
    "Balmores",
    "Doniego Balmores",
    "Balmores Lab",
    "Balmores structural engineer",
    "Sandra Agcaoili AI researcher",
    "Balmores Structural",
    "Balmores Lab",
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
    "Physics-Informed Neural Networks",
    "Natural Language to FEM",
  ],
  category: "Structural Engineering & AI Research",
  alternates: {
    canonical: "/",
    types: {
      "application/atom+xml": `${SITE_URL}/feed.xml`,
    },
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: SITE_URL,
    siteName: SITE_NAME,
    title: `${PERSON_NAME} - ${JOB_TITLE}`,
    description: SHORT_DESCRIPTION,
  },
  twitter: {
    card: "summary_large_image",
    site: "@louiedbalmores",
    creator: "@louiedbalmores",
    title: `${PERSON_NAME} - ${JOB_TITLE}`,
    description:
      "Licensed Civil Engineer researching AI-driven structural optimization. PyNite 3D FEA from a design brief.",
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
    icon: [
      { url: "/logo.svg", type: "image/svg+xml" },
      { url: "/favicon.ico" },
    ],
    shortcut: "/logo.svg",
    apple: "/logo.svg",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#080a0f",
  colorScheme: "dark",
};

// Structured data built from shared seo.ts (Person, Organization, WebSite, ProfilePage @graph)
const structuredData = buildRootStructuredData();

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        {/* IndieWeb / rel=me links: some crawlers and verification services use
            these to build identity graphs across platforms. */}
        <link rel="me" href="https://www.linkedin.com/in/louiebalmores/" />
        <link rel="me" href="https://x.com/louiedbalmores" />
        <link rel="me" href="https://about.me/louiebalmoresdesign/" />
        <link rel="me" href="https://orcid.org/0009-0008-5479-4033" />
        <link rel="me" href="https://www.wikidata.org/wiki/Q139544451" />
        <link rel="me" href="https://github.com/balmores-strux-ai/balmores-structural" />
        <link rel="me" href="https://worldchess.com/profile/422673" />
        {getExtraSameAs().map((href) => (
          <link key={href} rel="me" href={href} />
        ))}
        {getSandraExtraSameAs().map((href) => (
          <link key={`sandra-${href}`} rel="me author" href={href} />
        ))}
        <link rel="author" href={`${SITE_URL}/about`} />
        <link rel="author" href={SANDRA_PROFILE_URL} />
        <link rel="publisher" href={SITE_URL} />

        {/* Dublin Core metadata - indexed by some academic / semantic crawlers. */}
        <meta name="DC.creator" content={`${PERSON_NAME}; ${SANDRA_AGCAOILI.name}`} />
        <meta name="DC.publisher" content={SITE_NAME} />
        <meta name="DC.subject" content="Structural Engineering, Artificial Intelligence, Computational Design" />
        <meta name="DC.identifier" content={SITE_URL} />

        {/* Plain author meta - still read by Google for attribution heuristics. */}
        <meta name="author" content={`${PERSON_NAME}, ${SANDRA_AGCAOILI.name}`} />
        <meta name="copyright" content={`(c) ${new Date().getFullYear()} ${PERSON_NAME}`} />

        {/* Hint Google Knowledge Graph about the canonical identity URL. */}
        <meta property="profile:first_name" content="Louie" />
        <meta property="profile:last_name" content="Balmores" />
        <meta property="profile:username" content="louiedbalmores" />

        {/* IndexNow - instant crawl protocol (Bing, Yandex, Seznam, and now piloted by Google).
            The keyed file at /ykjm52si9r4gfvwhul8ob7cd3nqxpe01.txt proves ownership. */}
        <meta name="indexnow" content="ykjm52si9r4gfvwhul8ob7cd3nqxpe01" />

        {/* Expose the raw JSON-LD as a discoverable alternate - helps AI crawlers and
            linked-data indexers ingest the entity graph directly. */}
        <link rel="alternate" type="application/ld+json" href="/seo-schema.json" />
        <link rel="alternate" type="application/ld+json" href="/sandra-agcaoili-schema.json" title="Sandra Agcaoili profile" />
        <link rel="alternate" type="application/atom+xml" title="Balmores Lab feed" href="/feed.xml" />
        <link rel="alternate" type="application/rdf+xml" title="FOAF profile" href="/foaf.rdf" />
        <link
          rel="alternate"
          type="application/rdf+xml"
          title="Sandra Agcaoili FOAF profile"
          href="/sandra-agcaoili.foaf.rdf"
        />

        {/* Pre-warm connections to the profile platforms referenced in sameAs,
            so bots that follow links don't stall on DNS/TLS handshakes. */}
        <link rel="dns-prefetch" href="//www.linkedin.com" />
        <link rel="dns-prefetch" href="//x.com" />
        <link rel="dns-prefetch" href="//prc.gov.ph" />
        <link rel="dns-prefetch" href="//orcid.org" />
        <link rel="dns-prefetch" href="//www.wikidata.org" />

        <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }} />
      </head>
      <body>{children}</body>
    </html>
  );
}

