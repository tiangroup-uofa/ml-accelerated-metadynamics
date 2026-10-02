---
name: research-map
description: Use when editing the Quarto site in site/ — adding or updating a pillar page, a study page, the research map graph, or an overview summary. Covers the pillar/study taxonomy, the research-map data flow, and the evidence rules that overview text must follow.
---

# Maintaining the research site and research map

The site has two layers. **Pillar pages** (`p1`–`p4`) are the scientific narrative;
**study pages** (`01`–`08`) are the source of truth for methodology, full tables,
controls, and troubleshooting. Overview text summarizes; study pages retain depth.

## Taxonomy

Four pillars, in the order the science actually went:

| Pillar | Page | Question |
|---|---|---|
| 1 · Validation | `p1-validation.qmd` | Does off-the-shelf MACE reproduce xTB at equilibrium? |
| 2 · Reactive | `p2-reactive.qmd` | Does that survive bond forming? |
| 3 · Correction | `p3-correction.qmd` | Can a lightweight correction close the gap? |
| 4 · Iterative | `p4-iterative.qmd` | Does iterating that correction keep improving it? |

Pillars 2 and 3 are the flagships and carry more visual weight (`.pillar-card.flagship`
on the overview). Pillar 4 is deliberately shorter.

Every pillar page follows the same five-part spine: **Scientific question** (in a
`.question` block) → **Why we tested it** → **Key evidence** → **What we learned** →
**Related studies** (a `.studylinks` chip list, with the primary study marked
`.primary`). A reader should finish a pillar in 2–4 minutes.

## The research map

```
site/research-map.json  →  python3 site/build_research_map.py  →  site/research-map.svg
```

`research-map.qmd` includes the SVG verbatim. There is no runtime dependency, no
JavaScript, and no build step beyond that one script. **Never hand-edit
`research-map.svg`** — it is generated.

Node `kind`: `core` (one), `pillar` (four), `study` (a detailed study page), `result`
(a major finding, usually deep-linked to a pillar anchor), `future` (an open question,
drawn dashed, no `href`), `wip` (work in progress with its own page but **no results
yet**, drawn amber and dashed, linked; attached to `core` with a `pillar` edge).

Edge `rel`: `pillar` (core → pillar), `contains` (pillar → node), `motivates` (red
dashed — one result made the next experiment necessary), `validates` (green dashed — an
independent method confirmed a result). Keep `motivates`/`validates` **sparse**; they
carry the argument and lose meaning if overused.

Layout is a deterministic `col`/`row` grid — no physics, so the diagram never reflows
unpredictably. To add a node: append it with a free `row` in its pillar's band, add a
`contains` edge from the pillar, and re-run the script. If a band gets crowded, shift
the rows below it and move the pillar's own `row` to the band's centre. Curved
`motivates` edges take an optional `bow` (pixels) to avoid collisions.

Keep the map at **20–25 nodes**. It shows how studies relate, not every script or
dataset. If a node is not something a reader would want to click through to, it does
not belong.

## Evidence rules for overview text

These are not style preferences — earlier work was corrected for violating them.

1. **Every pillar claim traces to a study page or a `FINDINGS_*.md`.** If a number is
   not in a generated result file, it does not go on a pillar page.
2. **Never promote a hypothesis into a demonstrated result.** Keep the three tiers
   explicit and distinct: *directly demonstrated* / *supported interpretation* /
   *not demonstrated*. "Consistent with a correction-capacity limitation" is a
   hypothesis; "the barrier degraded from 9.87 to 13.91" is a result.
3. **xTB is the chosen reference, not ground truth.** For the Diels–Alder benchmark it
   *under*-estimates the barrier relative to DFT/experiment. Matching it is
   reference-matching, never "improved accuracy".
4. **Do not claim active learning.** Pillar 4 used a hand-fixed query rule with no
   acquisition metric or uncertainty estimate. A negative result there says nothing
   about active learning in general.
5. **Do not silently replace a published number.** When a better estimator supersedes
   an older one, show both and explain the difference — as with the constrained-scan
   barrier (9.9 kcal/mol) versus the certified saddle (6.33) on the same model.
6. **Report negative results plainly.** Pillar 4 is a negative result and is presented
   as one.

## Stability constraints

- **Study URLs `01`–`08` must not change.** External references and the group's own
  links depend on them. Restructure page *content* freely; keep the filename.
- Section anchors referenced from pillar pages or the map are explicit (`{#sec-...}`),
  not auto-generated from headings, so that rewording a heading does not break a link.
- After any change: re-run `build_research_map.py`, render with
  `quarto render` from `site/`, and check that every `assets/*.png` referenced by a page
  exists and that no internal link 404s.
