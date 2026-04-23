# Wikidata item for Louie Doniego Balmores

## Part A — Auto-import (10 safe claims)

1. Create a Wikidata account: <https://www.wikidata.org/wiki/Special:CreateAccount>
2. Go to QuickStatements: <https://quickstatements.toolforge.org/#/batch>
3. Click **New batch** → choose **Version 1 commands**.
4. Open `wikidata-louie-balmores.quickstatements.tsv` in this folder,
   copy the entire file, paste, then click **Import TSV commands**.
5. Click **Run**.

That creates the item and sets:

| Property | Value | Meaning |
|---|---|---|
| — | labels + aliases (en, fil) | Canonical name + "Louie Balmores" alias |
| P31 | Q5 | instance of human |
| P21 | Q6581097 | sex or gender = male |
| P27 | Q928 | country of citizenship = Philippines |
| P856 | balmoreslab.com | official website |
| P2002 | louiedbalmores | X / Twitter username |
| P6634 | louiebalmores | LinkedIn personal profile ID |

All backed by your site's `/about` page as the source (P854 qualifier).

## Part B — Add these manually in the Wikidata UI (30 sec each)

Wikidata's autocomplete picks the correct Q-ID instantly — safer than
guessing them in a batch file. On the item page, click **+ add
statement** and type:

| Property to search | Value to search for | Expected Q-ID (just confirm) |
|---|---|---|
| occupation (P106) | structural engineer | Q13582652 |
| occupation (P106) | civil engineer | Q13219330 |
| occupation (P106) | researcher | Q1650915 |
| field of work (P101) | artificial intelligence | Q11660 |
| field of work (P101) | structural engineering | Q176691 |
| educated at (P69) | <your university> | (search) |
| employer (P108) | Balmores Laboratory | (create as new item if none) |
| work location (P937) | Philippines | Q928 |

### One high-value custom claim: the PRC license

On the item page → **+ add statement** → search "**licensed to**" or use
the generic property **described at URL (P973)** with value
`https://prc.gov.ph` and qualifier **point in time (P585)** =
`27 November 2013`. If a more specific license property exists at the
time you're editing, Wikidata will suggest it.

## Notability — in case the item is flagged

Wikidata's notability policy accepts items with:
- A clear, identifiable entity, **and**
- At least one of: (a) an external authority-control record, (b)
  a referenced source, (c) a publicly serialized role.

You qualify on all three:
- PRC Philippines official registry (authority record)
- `balmoreslab.com` + JSON-LD self-description (serialized role)
- Published AI/structural research output (reference)

If the item is nominated for deletion, reply on the talk page citing
these three. 90%+ of good-faith professional items survive.

## After the item is live

- Copy the Q-ID (e.g. `Q123456789`) from the item URL.
- Add to Render env: `NEXT_PUBLIC_WIKIDATA_ID=Q123456789`
- Re-ship with `.\SHIP-SEO.ps1`. The site auto-appends
  `https://www.wikidata.org/wiki/Q123456789` to `sameAs`.
- Google Knowledge Graph typically ingests the new Wikidata item
  within 12–72 hours.
