# Louie Doniego Balmores — Knowledge Panel Kit

A prioritized, minimal-clicks playbook for making Google recognize
`Louie Balmores` as a distinct, notable entity and mount a Knowledge
Panel on the SERP.

> Why a kit and not "just do it for you"? Every platform below requires
> human ToS acceptance, an email you own, and a CAPTCHA. The signal must
> come from **you**, not an automation — Google explicitly downranks
> entities whose cross-platform graph looks bot-generated. This kit gives
> you the exact text to paste so each account still takes <2 minutes.

---

## The three signals Google actually uses

1. **A canonical URL that declares the entity.**
   ✅ Done: `https://www.balmoreslab.com/` with JSON-LD Person + Organization
   graph and microdata.

2. **Independent cross-references to that URL (`sameAs`).**
   ✅ 6 live (LinkedIn, X, about.me, WorldChess, PRC, Twitter legacy).
   🎯 Target: ≥12 authoritative refs. The ones below each add a measurable
   lift.

3. **Consistency of name + role + bio + birthplace across refs.**
   Use the bios in `louie-balmores-bios.md` *verbatim* everywhere. Do not
   paraphrase between platforms — exact-string matches are what entity
   reconciliation keys on.

---

## Tier 1 — Highest KP leverage (do these first)

### 1.1 Wikidata  ★★★★★
The single biggest KP lever that exists. Google's Knowledge Graph
literally ingests Wikidata nightly.

- Sign up: <https://www.wikidata.org/wiki/Special:CreateAccount>
- After login, go to QuickStatements:
  <https://quickstatements.toolforge.org/#/batch>
- Click **New batch** → **Version 1 commands**.
- Open `seo/wikidata-louie-balmores.quickstatements.tsv` in this repo,
  copy the whole file, paste into the QuickStatements text area, click
  **Import TSV commands** → **Run**.
- It will create a new Q-item with: occupation, nationality, license,
  official website, LinkedIn ID, Twitter ID, and description — all the
  identity facets Google looks for.
- Caveat: Wikidata has notability standards. Your ETABS AI research + PRC
  license + publicly indexable projects give you a case. If the item gets
  flagged, add citations (a news article, a published paper, a conference
  talk). Even one indexable reference usually suffices.

### 1.2 ORCID  ★★★★★
Authoritative persistent researcher ID. Google Scholar cross-references
it, and adding it to `sameAs` is a KP gold signal.

- Register: <https://orcid.org/register>
- Use the bio text: copy "Medium bio" from `louie-balmores-bios.md`.
- Fill: Employment → "Balmores Laboratory, 2015–present, Structural
  Engineer & AI Researcher". Education → your CE degree. Works → any
  papers/presentations (optional).
- Copy your ORCID iD (format `0000-0000-0000-0000`) into
  `.env.production` on Render as `NEXT_PUBLIC_ORCID_ID`. The site will
  auto-include it in `sameAs`.

### 1.3 GitHub profile (special repo)  ★★★★☆
A GitHub user profile named after your username renders a README on
`github.com/<username>`. Google indexes it and follows `sameAs`.

- If you don't yet have a personal GitHub account (only
  `balmores-strux-ai` org exists), create one — `louiebalmores` or
  `louiedbalmores` to match the X handle.
- Create a repo with the **exact same name as your username** (e.g.
  `louiedbalmores/louiedbalmores`).
- Paste `seo/github-profile-README.md` from this kit as the repo
  README.md. It contains the Person JSON-LD in a code block plus links
  back to balmoreslab.com.
- Add your GitHub URL to `NEXT_PUBLIC_GITHUB_URL` in Render env.

### 1.4 Keybase  ★★★★☆
Keybase provides *cryptographic proofs* linking Twitter, GitHub, DNS,
and a website to one identity. Google doesn't directly ingest Keybase,
but the proofs propagate — and the public profile is a clean `sameAs`.

- Sign up: <https://keybase.io/signup>
- Run identity proofs for: Twitter (`louiedbalmores`), GitHub, DNS
  (`balmoreslab.com` TXT record), website
  (place the key file at `frontend/public/.well-known/keybase.txt`).
- Add `https://keybase.io/louiedbalmores` to env as
  `NEXT_PUBLIC_KEYBASE_URL`.

### 1.5 Google Search Console (required)  ★★★★★
Not a `sameAs`, but without this Google won't honor sitemap submission or
"Request Indexing" — both fired by `scripts/ping-crawlers.ps1`.

- <https://search.google.com/search-console>
- Add **Domain property**: `balmoreslab.com` (not URL-prefix).
- Verify via DNS TXT record on your registrar.
- Copy the HTML-tag token into Render env as
  `NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION`. The layout already reads it.

---

## Tier 2 — Supporting identity graph (do within a week)

### 2.1 Mastodon with `rel=me` back-link  ★★★☆☆
- Any instance works; `mastodon.social` or `fosstodon.org` (tech-focused)
  are good.
- In profile, add `https://www.balmoreslab.com` as a link — Mastodon
  emits it with `rel="me"`. Our site *already* emits `rel="me"` to your
  Mastodon in layout.tsx — create the account, then add its URL to
  `NEXT_PUBLIC_MASTODON_URL`.

### 2.2 Gravatar  ★★★☆☆
Gravatar is crawled heavily and shows up in Google People Cards.
- <https://gravatar.com> → add photo, bio (short bio from bios.md), link
  to balmoreslab.com.

### 2.3 GitHub organization page  ★★★☆☆
- On the `balmores-strux-ai` org, fill description, website, location,
  email, verified domain. Even orgs count as entity-graph nodes.

### 2.4 Stack Overflow / Stack Exchange  ★★☆☆☆
- <https://stackoverflow.com> profile → About me (medium bio), website
  link, Twitter, GitHub. Even a single answered question starts indexing.

### 2.5 DEV Community (dev.to)  ★★☆☆☆
- <https://dev.to> → profile with website + social links. Each post
  ranks, and the profile page itself is a `sameAs`-worthy page.

### 2.6 ResearchGate (for AI-structural papers)  ★★★☆☆
- <https://www.researchgate.net> → Author profile. If you upload a
  preprint of your ETABS AI work, it gets a ResearchGate URL which is
  another strong node.

### 2.7 Academia.edu  ★★☆☆☆
- Similar to ResearchGate. Low effort, additional node.

---

## Tier 3 — Engineering-community specific

### 3.1 PhilippineCE / Filipino Engineer directories  ★★★☆☆
Any community directory of Philippine civil engineers that allows a
profile with external link. Even one such ref adds geographic-entity
specificity.

### 3.2 LinkedIn Articles  ★★★★☆
Publish 2–3 long-form LinkedIn articles from your existing profile,
titled unambiguously:
- "AI-Driven Structural Optimization: Early Results from Balmores
  Laboratory"
- "PyNite + PyTorch: A Practical FEM-to-AI Pipeline for Mid-Rise RC
  Frames"
Each article is indexed under `linkedin.com/pulse/...` — new `sameAs`
candidates and independent content citing *Louie Doniego Balmores*.

### 3.3 YouTube channel  ★★★★☆
A channel named `Louie Balmores` (or `Balmores Lab`) with even 3–5
short videos (screen-capture demos of the PyNite app) becomes a huge
KP signal because Google owns YouTube and the entity-link is trivial.

### 3.4 Product Hunt launch for Balmores Strux AI  ★★☆☆☆
One-time launch → permanent product page with external references.

---

## After creating each account

1. In the new platform's "Website" field, paste **exactly**
   `https://www.balmoreslab.com` (with `www.`, with `https://`, no
   trailing slash). Exact-string matching matters.
2. In the bio, paste one of the 3 bios from `louie-balmores-bios.md`
   **verbatim**. Resist the urge to edit — consistency is the signal.
3. Add a profile photo. Use the **same photo** on every platform. Google
   runs image-similarity hashing for entity reconciliation.
4. After 3+ new accounts exist, add them to Render env vars and re-ship
   with `.\SHIP-SEO.ps1`. The site auto-includes them in `sameAs`.

## Timing expectations (realistic)

| Stage | Time |
|---|---|
| Sitemap discovered by Google | hours after ping |
| `/` and `/about` indexed | 1–24 h after *Request Indexing* |
| Wikidata item crawled by Google | 12–72 h after creation |
| Entity candidate formed in Google's Knowledge Graph | ~1–3 weeks after 10+ consistent refs exist |
| Knowledge Panel appears on SERP for "Louie Balmores" | typically **2–8 weeks** after entity formation |

## The single biggest accelerator

If only one thing gets done this week: **create the Wikidata item**.
Tier-1.1 alone is worth the rest of Tier 2 combined.
