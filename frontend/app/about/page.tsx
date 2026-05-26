import type { Metadata } from "next";
import Link from "next/link";
import ProfileBadges from "@/components/ProfileBadges";

const SITE_URL = "https://www.balmoreslab.com";

export const metadata: Metadata = {
  title: "About Louie Doniego Balmores - Structural Engineer & AI Researcher",
  description:
    "Louie Doniego Balmores, P.Eng (Candidate), PE (Candidate) - Registered Civil Engineer (PRC Philippines, Nov 2013, Sequence No. 350) with 10+ years in high-performance building design and active AI research in structural optimization.",
  alternates: { canonical: "/about" },
  openGraph: {
    type: "profile",
    url: `${SITE_URL}/about`,
    title: "About Louie Doniego Balmores - Structural Engineer & AI Researcher",
    description:
      "Registered Civil Engineer (PRC Philippines, 2013) with 10+ years in structural design, researching AI-driven structural optimization.",
    firstName: "Louie",
    lastName: "Balmores",
    username: "louiedbalmores",
    gender: "male",
  } as Metadata["openGraph"],
};

const aboutPageLd = {
  "@context": "https://schema.org",
  "@type": "AboutPage",
  "@id": `${SITE_URL}/about#aboutpage`,
  url: `${SITE_URL}/about`,
  name: "About Louie Doniego Balmores",
  description:
    "Profile of Louie Doniego Balmores - born June 26, 1991 in Tuguegarao City, Cagayan Valley, Philippines; Registered Civil Engineer (PRC Philippines, 2013), Master's in Computer Science, Doctor of Information Technology, and AI researcher based in Ontario, Canada.",
  mainEntity: { "@id": `${SITE_URL}/#person` },
  about: {
    "@type": "Person",
    "@id": `${SITE_URL}/#person`,
    name: "Louie Doniego Balmores",
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
  },
  citation: [
    {
      "@type": "CreativeWork",
      name: "PRC (Philippines) - November 2013 Civil Engineer Licensure Examination results (PDF)",
      url: "https://www.prc.gov.ph/uploaded/documents/CE1113se.pdf",
    },
    {
      "@type": "CreativeWork",
      name: "Civil Engineers - November 2013 (mirror)",
      url: "https://www.scribd.com/doc/187567272/Civil-Engineers-November-2013",
    },
  ],
  breadcrumb: {
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "Home", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "About", item: `${SITE_URL}/about` },
    ],
  },
};

// FAQPage schema - this is the single highest-converting schema for
// Knowledge-Panel adjacency, because it feeds Google's "People Also Ask"
// module. When PAA questions mention the canonical entity, Google treats
// that as strong co-occurrence signal.
const faqLd = {
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "@id": `${SITE_URL}/about#faq`,
  mainEntity: [
    {
      "@type": "Question",
      name: "Who is Louie Balmores?",
      acceptedAnswer: {
        "@type": "Answer",
        text:
          "Louie Doniego Balmores is a Filipino structural engineer and AI researcher. He is a Registered Civil Engineer (Professional Regulation Commission of the Philippines, November 2013, Sequence No. 350) with over 10 years of professional practice, and is the founder of Balmores Lab - a research initiative on AI-driven structural optimization.",
      },
    },
    {
      "@type": "Question",
      name: "Is Louie Balmores a licensed engineer?",
      acceptedAnswer: {
        "@type": "Answer",
        text:
          "Yes. Louie Balmores is a Registered Civil Engineer licensed by the Professional Regulation Commission (PRC) of the Philippines. He passed the November 2013 Civil Engineer Licensure Examination with Sequence No. 350. He is also a PEng Candidate with Professional Engineers Ontario and a US PE Candidate in the United States.",
      },
    },
    {
      "@type": "Question",
      name: "What does Louie Balmores research?",
      acceptedAnswer: {
        "@type": "Answer",
        text:
          "His research focuses on AI models for structural integrity, computational design, and material efficiency. At Balmores Lab he develops deep-learning surrogate models on top of PyNite finite-element analysis, converting plain-English structural briefs into validated 3D frame models with reactions, drift, and member envelopes.",
      },
    },
    {
      "@type": "Question",
      name: "What is Balmores Lab?",
      acceptedAnswer: {
        "@type": "Answer",
        text:
          "Balmores Lab (balmoreslab.com) is an independent research initiative founded by Louie Doniego Balmores. Its mission is to transition traditional structural engineering workflows into automated, intelligent systems - combining classical finite-element methods with neural surrogate models.",
      },
    },
    {
      "@type": "Question",
      name: "Where can I verify Louie Balmores's civil engineer license?",
      acceptedAnswer: {
        "@type": "Answer",
        text:
          "His Philippine Civil Engineer license is officially registered with the Professional Regulation Commission (PRC) at prc.gov.ph. He appears in the PRC's November 2013 Civil Engineer Licensure Examination results as Sequence No. 350.",
      },
    },
  ],
};

const S = {
  page: {
    minHeight: "100vh",
    background:
      "radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,0.12), transparent 60%), radial-gradient(1000px 500px at 100% 0%, rgba(14,165,233,0.10), transparent 60%), #080a0f",
    color: "#e6edf3",
    fontFamily:
      'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  } as React.CSSProperties,
  container: {
    maxWidth: 960,
    margin: "0 auto",
    padding: "48px 24px 80px",
  } as React.CSSProperties,
  topnav: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "16px 24px",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
    backdropFilter: "blur(8px)",
    background: "rgba(8,10,15,0.6)",
    position: "sticky",
    top: 0,
    zIndex: 10,
  } as React.CSSProperties,
  brandTitle: { fontWeight: 700, letterSpacing: "0.08em", fontSize: 14, color: "#cbd5e1" } as React.CSSProperties,
  link: {
    color: "#93c5fd",
    textDecoration: "none",
    fontSize: 14,
    marginLeft: 20,
  } as React.CSSProperties,
  hero: {
    padding: "40px 0 20px",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
  } as React.CSSProperties,
  eyebrow: {
    color: "#94a3b8",
    letterSpacing: "0.18em",
    fontSize: 12,
    textTransform: "uppercase",
    marginBottom: 12,
  } as React.CSSProperties,
  h1: {
    fontSize: 40,
    lineHeight: 1.1,
    fontWeight: 700,
    margin: "0 0 8px",
    letterSpacing: "-0.02em",
  } as React.CSSProperties,
  title: {
    fontSize: 20,
    color: "#a5b4fc",
    margin: "0 0 16px",
    fontWeight: 500,
  } as React.CSSProperties,
  lede: {
    fontSize: 17,
    color: "#cbd5e1",
    maxWidth: 720,
    lineHeight: 1.6,
    margin: "0 0 20px",
  } as React.CSSProperties,
  section: {
    padding: "36px 0",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
  } as React.CSSProperties,
  h2: {
    fontSize: 13,
    letterSpacing: "0.22em",
    textTransform: "uppercase",
    color: "#60a5fa",
    margin: "0 0 20px",
    fontWeight: 600,
  } as React.CSSProperties,
  h3: {
    fontSize: 22,
    margin: "0 0 10px",
    fontWeight: 600,
    color: "#f1f5f9",
  } as React.CSSProperties,
  p: {
    color: "#cbd5e1",
    lineHeight: 1.65,
    fontSize: 16,
    margin: "0 0 14px",
  } as React.CSSProperties,
  grid2: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
    gap: 20,
  } as React.CSSProperties,
  card: {
    padding: 20,
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 14,
    background: "rgba(255,255,255,0.02)",
  } as React.CSSProperties,
  badge: {
    display: "inline-block",
    padding: "4px 10px",
    fontSize: 11,
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    borderRadius: 999,
    border: "1px solid rgba(99,102,241,0.4)",
    background: "rgba(99,102,241,0.08)",
    color: "#c7d2fe",
    marginRight: 8,
  } as React.CSSProperties,
  kvRow: {
    display: "grid",
    gridTemplateColumns: "180px 1fr",
    gap: 12,
    padding: "8px 0",
    borderTop: "1px dashed rgba(255,255,255,0.06)",
    fontSize: 15,
  } as React.CSSProperties,
  kvKey: { color: "#94a3b8" } as React.CSSProperties,
  kvVal: { color: "#e2e8f0" } as React.CSSProperties,
  extLink: {
    color: "#93c5fd",
    textDecoration: "underline",
    textUnderlineOffset: 4,
  } as React.CSSProperties,
  socials: {
    display: "flex",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 14,
  } as React.CSSProperties,
  chip: {
    display: "inline-flex",
    alignItems: "center",
    padding: "8px 14px",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 999,
    background: "rgba(255,255,255,0.03)",
    color: "#e2e8f0",
    fontSize: 13,
    textDecoration: "none",
  } as React.CSSProperties,
  footer: {
    padding: "24px 24px 40px",
    color: "#64748b",
    fontSize: 13,
    textAlign: "center",
  } as React.CSSProperties,
};

export default function AboutPage() {
  return (
    <main style={S.page} itemScope itemType="https://schema.org/ProfilePage">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(aboutPageLd) }} />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(faqLd) }} />

      <nav style={S.topnav} aria-label="Primary">
        <Link href="/" style={{ textDecoration: "none", color: "inherit" }}>
          <span style={S.brandTitle}>BALMORES - LAB</span>
        </Link>
        <div>
          <Link href="/" style={S.link}>Home</Link>
          <Link href="/about" style={{ ...S.link, color: "#fff" }} aria-current="page">About</Link>
        </div>
      </nav>

      <article
        style={S.container}
        itemScope
        itemType="https://schema.org/Person"
        itemID={`${SITE_URL}/#person`}
      >
        <meta itemProp="url" content={SITE_URL} />
        <meta itemProp="givenName" content="Louie" />
        <meta itemProp="additionalName" content="Doniego" />
        <meta itemProp="familyName" content="Balmores" />
        <meta itemProp="birthDate" content="1991-06-26" />
        <meta itemProp="gender" content="Male" />
        <meta itemProp="nationality" content="Philippines" />
        <span
          itemProp="birthPlace"
          itemScope
          itemType="https://schema.org/Place"
          style={{ display: "none" }}
        >
          <meta itemProp="name" content="Tuguegarao City, Cagayan Valley, Philippines" />
        </span>
        <span
          itemProp="homeLocation"
          itemScope
          itemType="https://schema.org/Place"
          style={{ display: "none" }}
        >
          <meta itemProp="name" content="Ontario, Canada" />
        </span>

        {/* HERO */}
        <header style={S.hero}>
          <div style={S.eyebrow}>Official Profile - balmoreslab.com</div>
          <h1 style={S.h1}>
            <span itemProp="name">Louie Doniego Balmores</span>
            <span style={{ color: "#94a3b8", fontWeight: 500, fontSize: 22 }}>
              {" "}- <span itemProp="honorificSuffix">P.Eng (Candidate), PE (Candidate)</span>
            </span>
          </h1>
          <p style={S.title} itemProp="jobTitle">
            Structural Engineer &amp; AI Researcher
          </p>
          <p style={S.lede} itemProp="description">
            Over <strong>10 years</strong> of structural engineering experience in
            high-performance building design, currently pursuing a{" "}
            <strong>Master&apos;s degree in Computer Science</strong> and a{" "}
            <strong>Doctor of Information Technology</strong> with a focus on{" "}
            <strong>AI-driven structural optimization</strong> — transforming
            traditional structural analysis into automated, intelligent workflows at{" "}
            <Link href="/" style={S.extLink}>Balmores Lab</Link>.
          </p>
          <div>
            <span style={S.badge}>Civil Engineer, PRC PH</span>
            <span style={S.badge}>MS Computer Science</span>
            <span style={S.badge}>Doctor of Information Technology</span>
            <span style={S.badge}>AI - Structural Optimization</span>
            <span style={S.badge}>10+ Years Experience</span>
          </div>
        </header>

        {/* AT A GLANCE */}
        <section style={S.section} aria-labelledby="glance-h">
          <h2 id="glance-h" style={S.h2}>At a Glance</h2>
          <div style={S.card}>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Full Name</div>
              <div style={S.kvVal}>Louie Doniego Balmores</div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Born</div>
              <div style={S.kvVal}>
                <time dateTime="1991-06-26">June 26, 1991</time>
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Birthplace</div>
              <div style={S.kvVal}>
                Tuguegarao City, Cagayan Valley (Region II), Philippines
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Nationality</div>
              <div style={S.kvVal}>Filipino</div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Currently Based</div>
              <div style={S.kvVal}>Toronto, Ontario, Canada</div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Profession</div>
              <div style={S.kvVal}>Structural Engineer &amp; AI Researcher</div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Founder of</div>
              <div style={S.kvVal}>
                <Link href="/" style={S.extLink}>Balmores Lab</Link>
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Languages</div>
              <div style={S.kvVal}>English, Filipino (Tagalog)</div>
            </div>
          </div>
        </section>

        {/* PROFESSIONAL LICENSURE */}
        <section style={S.section} aria-labelledby="licensure-h">
          <h2 id="licensure-h" style={S.h2}>Professional Licensure</h2>

          <div
            style={S.card}
            itemProp="hasCredential"
            itemScope
            itemType="https://schema.org/EducationalOccupationalCredential"
          >
            <h3 style={S.h3}>
              <span itemProp="name">Registered Civil Engineer</span>
              <span style={{ color: "#94a3b8", fontWeight: 400, fontSize: 16 }}> - Philippines</span>
            </h3>
            <meta itemProp="credentialCategory" content="Professional License" />
            <p style={S.p}>
              Issued by the{" "}
              <strong>Professional Regulation Commission (PRC)</strong> of the
              Republic of the Philippines - the official government body that
              regulates licensed engineering practice.
            </p>

            <div style={S.kvRow}>
              <div style={S.kvKey}>Examination</div>
              <div style={S.kvVal}>November 2013 Civil Engineer Licensure Examination</div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Sequence No.</div>
              <div style={S.kvVal}><strong>350</strong></div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Date Published</div>
              <div style={S.kvVal}>
                <time itemProp="datePublished" dateTime="2013-11-27">November 27, 2013</time>
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Issuing Body</div>
              <div style={S.kvVal}>
                <span
                  itemProp="recognizedBy"
                  itemScope
                  itemType="https://schema.org/GovernmentOrganization"
                >
                  <span itemProp="name">Professional Regulation Commission (PRC)</span>
                  {" - "}
                  <a
                    href="https://prc.gov.ph"
                    style={S.extLink}
                    rel="noopener external"
                    target="_blank"
                    itemProp="url"
                  >
                    PRC Official List of Passers
                  </a>
                </span>
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Verification Documents</div>
              <div style={S.kvVal}>
                <a
                  href="https://www.prc.gov.ph/uploaded/documents/CE1113se.pdf"
                  style={S.extLink}
                  rel="noopener external"
                  target="_blank"
                >
                  PRC PDF (CE1113se)
                </a>
                {" - "}
                <a
                  href="https://www.scribd.com/doc/187567272/Civil-Engineers-November-2013"
                  style={S.extLink}
                  rel="noopener external"
                  target="_blank"
                >
                  Scribd mirror
                </a>
              </div>
            </div>
          </div>

          <h3 style={{ ...S.h3, marginTop: 28, fontSize: 18 }}>Ongoing Credentials</h3>
          <div style={{ ...S.grid2 }}>
            {/* Graduate education block — see the dedicated Education section
                below for the full thesis details. Kept here as a quick chip so
                the credentials timeline is complete at a glance. */}
            <div
              style={S.card}
              itemProp="hasCredential"
              itemScope
              itemType="https://schema.org/EducationalOccupationalCredential"
            >
              <div style={{ color: "#a5b4fc", fontSize: 12, letterSpacing: "0.12em", textTransform: "uppercase" }}>
                Graduate Studies
              </div>
              <h4 style={{ margin: "6px 0 6px", fontSize: 17, color: "#f1f5f9" }} itemProp="name">
                MS in Computer Science
              </h4>
              <meta itemProp="credentialCategory" content="Graduate degree (in progress)" />
              <p style={{ ...S.p, margin: 0 }}>
                Specialisation: Artificial Intelligence &amp; Computational
                Engineering. Thesis on neural surrogate models for finite-element
                structural analysis (see <a href="#education" style={S.extLink}>Education</a>).
              </p>
            </div>
          </div>
          <h3 style={{ ...S.h3, marginTop: 28, fontSize: 18 }}>Professional Candidacies</h3>
          <div style={S.grid2}>
            <div
              style={S.card}
              itemProp="hasCredential"
              itemScope
              itemType="https://schema.org/EducationalOccupationalCredential"
            >
              <div style={{ color: "#a5b4fc", fontSize: 12, letterSpacing: "0.12em", textTransform: "uppercase" }}>
                PEng Candidate
              </div>
              <h4 style={{ margin: "6px 0 6px", fontSize: 17, color: "#f1f5f9" }} itemProp="name">
                PEng Candidate
              </h4>
              <meta itemProp="credentialCategory" content="Professional License (Candidate)" />
              <p style={{ ...S.p, margin: 0 }}>
                <span
                  itemProp="recognizedBy"
                  itemScope
                  itemType="https://schema.org/Organization"
                >
                  <span itemProp="name">Professional Engineers Ontario (PEO)</span>
                </span>
                {" - Canada"}
              </p>
            </div>
            <div
              style={S.card}
              itemProp="hasCredential"
              itemScope
              itemType="https://schema.org/EducationalOccupationalCredential"
            >
              <div style={{ color: "#a5b4fc", fontSize: 12, letterSpacing: "0.12em", textTransform: "uppercase" }}>
                US PE Candidate
              </div>
              <h4 style={{ margin: "6px 0 6px", fontSize: 17, color: "#f1f5f9" }} itemProp="name">
                US PE Candidate
              </h4>
              <meta itemProp="credentialCategory" content="Professional License (Candidate)" />
              <p style={{ ...S.p, margin: 0 }}>
                <span
                  itemProp="recognizedBy"
                  itemScope
                  itemType="https://schema.org/Organization"
                >
                  <span itemProp="name">NCEES (USA)</span>
                </span>
                {" - United States"}
              </p>
            </div>
          </div>
        </section>

        {/* EDUCATION — added per profile update: MS in CS (on-going) with a
            specific thesis title and supervisor blurb. Visible HTML mirrors
            the Person/alumniOf JSON-LD so Google's Knowledge Panel picks it
            up as a verified school affiliation. */}
        <section style={S.section} aria-labelledby="education-h" id="education">
          <h2 id="education-h" style={S.h2}>Education</h2>

          <div
            style={S.card}
            itemProp="alumniOf"
            itemScope
            itemType="https://schema.org/EducationalOrganization"
          >
            <div style={{ color: "#a5b4fc", fontSize: 12, letterSpacing: "0.12em", textTransform: "uppercase" }}>
              Master of Science · in progress
            </div>
            <h3 style={S.h3}>
              <span itemProp="name">MS in Computer Science</span>
            </h3>
            <p style={S.p}>
              Concentration in <strong>Artificial Intelligence</strong> and{" "}
              <strong>Computational Engineering</strong>. Coursework spans deep
              learning, numerical methods for PDEs, graph neural networks, and
              distributed scientific computing — chosen to support the thesis
              programme on AI-accelerated structural analysis.
            </p>

            <div style={S.kvRow}>
              <div style={S.kvKey}>Thesis (working title)</div>
              <div style={S.kvVal}>
                <em>
                  &ldquo;Physics-Informed Neural Surrogates for Real-Time
                  Finite-Element Analysis of Reinforced-Concrete Frames under
                  NSCP&#x2009;2015 / ASCE&#x2009;7 Load Combinations&rdquo;
                </em>
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Thesis statement</div>
              <div style={S.kvVal}>
                A unified neural architecture is trained on parametric PyNite
                and ETABS solves so that an engineer&apos;s plain-English brief
                returns ULS member envelopes, storey drift, and base reactions
                in <strong>under one second</strong> — with explicit physics-loss
                regularisation against the underlying FEM, so predictions stay
                code-compliant rather than statistically plausible.
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Research outputs</div>
              <div style={S.kvVal}>
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  <li>
                    Open-source kernel:{" "}
                    <Link href="/" style={S.extLink}>
                      Balmores Strux AI
                    </Link>{" "}
                    — natural-language → PyNite FEA, validated against ETABS.
                  </li>
                  <li>
                    Local-first LLM bridge: prompts never leave the user&apos;s
                    PC (DeepSeek-R1 on loopback). See{" "}
                    <code style={{ color: "#a5b4fc" }}>LOCAL_AI_SETUP.md</code> in the repo.
                  </li>
                  <li>
                    Dataset generator:{" "}
                    <code style={{ color: "#a5b4fc" }}>etabs_brain_full.py</code>{" "}
                    — 1,000+ parametric ETABS archetypes for surrogate
                    training (concrete + steel).
                  </li>
                </ul>
              </div>
            </div>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Status</div>
              <div style={S.kvVal}>
                Coursework complete · thesis writing &amp; experimental
                validation in progress.
              </div>
            </div>
          </div>

          <div
            style={{ ...S.card, marginTop: 16 }}
            itemProp="alumniOf"
            itemScope
            itemType="https://schema.org/EducationalOrganization"
          >
            <div style={{ color: "#a5b4fc", fontSize: 12, letterSpacing: "0.12em", textTransform: "uppercase" }}>
              Doctoral Studies · in progress
            </div>
            <h3 style={S.h3}>
              <span itemProp="name">Doctor of Information Technology</span>
            </h3>
            <p style={S.p}>
              Research direction: <strong>AI-augmented engineering systems</strong> —
              applying advanced deep-learning, privacy-preserving computation
              and intelligent infrastructure to scientific and structural
              computing workloads, extending the MSc thesis on physics-informed
              neural surrogates into a full doctoral programme.
            </p>
            <div style={S.kvRow}>
              <div style={S.kvKey}>Status</div>
              <div style={S.kvVal}>Currently pursuing.</div>
            </div>
          </div>

          <div style={{ ...S.card, marginTop: 16 }}>
            <h4 style={{ margin: 0, color: "#f1f5f9", fontSize: 16 }}>
              BS in Civil Engineering
            </h4>
            <p style={{ ...S.p, marginTop: 8, marginBottom: 0 }}>
              Foundation for the November 2013 PRC Civil Engineer Licensure
              Examination (Sequence No.&nbsp;350) — see Professional Licensure
              above.
            </p>
          </div>
        </section>

        {/* RESEARCH */}
        <section style={S.section} aria-labelledby="research-h">
          <h2 id="research-h" style={S.h2}>Research</h2>
          <h3 style={S.h3}>AI-Driven Structural Engineering</h3>
          <p style={S.p}>
            <strong>Focus.</strong> Developing AI models for{" "}
            <span itemProp="knowsAbout">structural integrity</span>,{" "}
            <span itemProp="knowsAbout">computational design</span>, and{" "}
            <span itemProp="knowsAbout">material efficiency</span> - pairing
            modern deep-learning surrogate models with classical finite element
            analysis (PyNite, ETABS) to compress design-cycle time without
            compromising code compliance.
          </p>
          <p style={S.p}>
            <strong>Goal.</strong> Transitioning traditional structural analysis
            into automated, intelligent workflows: a design brief in plain
            English becomes a validated 3D frame model with reactions, drift,
            envelopes, and optimization hints - in seconds, not days.
          </p>

          <div style={{ ...S.grid2, marginTop: 20 }}>
            <div style={S.card}>
              <h4 style={{ margin: 0, color: "#f1f5f9", fontSize: 16 }}>Structural Optimization</h4>
              <p style={{ ...S.p, marginTop: 8 }}>
                Neural surrogate models trained on parametric ETABS datasets to
                predict member demand and preliminary sizing.
              </p>
            </div>
            <div style={S.card}>
              <h4 style={{ margin: 0, color: "#f1f5f9", fontSize: 16 }}>Computational Design</h4>
              <p style={{ ...S.p, marginTop: 8 }}>
                Natural-language-to-FEM pipelines powering{" "}
                <Link href="/" style={S.extLink}>Balmores Strux AI</Link>{" "}
                - open-source PyNite 3D analysis from a chat prompt.
              </p>
            </div>
            <div style={S.card}>
              <h4 style={{ margin: 0, color: "#f1f5f9", fontSize: 16 }}>Material Efficiency</h4>
              <p style={{ ...S.p, marginTop: 8 }}>
                Generative optimization for concrete and steel quantities, with
                embodied-carbon awareness as a first-class objective.
              </p>
            </div>
          </div>
        </section>

        {/* FAQ (visible) - mirrors the FAQ JSON-LD above. Must stay in
            sync with the schema, otherwise Google treats it as hidden
            structured-data spam. */}
        <section style={S.section} aria-labelledby="faq-h">
          <h2 id="faq-h" style={S.h2}>Frequently Asked Questions</h2>

          <details style={{ ...S.card, marginBottom: 12 }}>
            <summary style={{ ...S.h3, cursor: "pointer" }}>Who is Louie Balmores?</summary>
            <p style={{ ...S.p, marginTop: 12 }}>
              Louie Doniego Balmores is a Filipino structural engineer and
              AI researcher. He is a Registered Civil Engineer
              (Professional Regulation Commission of the Philippines,
              November 2013, Sequence No. 350) with over 10 years of
              professional practice, and is the founder of Balmores
              Laboratory - a research initiative on AI-driven structural
              optimization.
            </p>
          </details>

          <details style={{ ...S.card, marginBottom: 12 }}>
            <summary style={{ ...S.h3, cursor: "pointer" }}>Is Louie Balmores a licensed engineer?</summary>
            <p style={{ ...S.p, marginTop: 12 }}>
              Yes. Louie Balmores is a Registered Civil Engineer licensed
              by the Professional Regulation Commission (PRC) of the
              Philippines. He passed the November 2013 Civil Engineer
              Licensure Examination with Sequence No. 350. He is also a
              PEng Candidate with Professional Engineers Ontario
              and a US PE Candidate in the United States
             .
            </p>
          </details>

          <details style={{ ...S.card, marginBottom: 12 }}>
            <summary style={{ ...S.h3, cursor: "pointer" }}>What does Louie Balmores research?</summary>
            <p style={{ ...S.p, marginTop: 12 }}>
              His research focuses on AI models for structural integrity,
              computational design, and material efficiency. At Balmores
              Laboratory he develops deep-learning surrogate models on top
              of PyNite finite-element analysis, converting plain-English
              structural briefs into validated 3D frame models with
              reactions, drift, and member envelopes.
            </p>
          </details>

          <details style={{ ...S.card, marginBottom: 12 }}>
            <summary style={{ ...S.h3, cursor: "pointer" }}>What is Balmores Lab?</summary>
            <p style={{ ...S.p, marginTop: 12 }}>
              Balmores Lab (balmoreslab.com) is an independent
              research initiative founded by Louie Doniego Balmores. Its
              mission is to transition traditional structural engineering
              workflows into automated, intelligent systems - combining
              classical finite-element methods with neural surrogate
              models.
            </p>
          </details>

          <details style={S.card}>
            <summary style={{ ...S.h3, cursor: "pointer" }}>Where can I verify his civil engineer license?</summary>
            <p style={{ ...S.p, marginTop: 12 }}>
              His Philippine Civil Engineer license is officially
              registered with the Professional Regulation Commission
              (PRC) at <a href="https://prc.gov.ph" style={S.extLink} rel="noopener external" target="_blank">prc.gov.ph</a>.
              He appears in the PRC&apos;s November 2013 Civil Engineer
              Licensure Examination results as Sequence No. 350.
            </p>
          </details>
        </section>

        {/* SOCIAL / sameAs links, also visible to non-JSON crawlers */}
        <section style={S.section} aria-labelledby="connect-h">
          <h2 id="connect-h" style={S.h2}>Verified Profiles</h2>
          <p style={S.p}>
            Cross-platform identity graph. Each link is also published as a
            <code style={{ margin: "0 6px", color: "#a5b4fc" }}>sameAs</code>
            entry in the page&apos;s JSON-LD.
          </p>
          <div style={{ padding: "12px 0 4px" }}>
            <ProfileBadges align="left" showOrcidPill />
          </div>
        </section>
      </article>

      <footer style={S.footer}>
        (c) {new Date().getFullYear()} Louie Doniego Balmores - Balmores Lab -{" "}
        <Link href="/" style={S.extLink}>balmoreslab.com</Link>
      </footer>
    </main>
  );
}

