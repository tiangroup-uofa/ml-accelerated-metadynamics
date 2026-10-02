# Results site (Quarto)

Source for the project results site — a plain [Quarto](https://quarto.org)
website that renders to `../docs/` and is served via GitHub Pages at
<https://tiangroup-uofa.github.io/ml-accelerated-metadynamics/>.

## Edit / render

```bash
# needs the quarto CLI; the site is rendered with Quarto 1.10.18
cd site
python3 build_research_map.py      # research-map.json -> research-map.svg
python3 build_barrier_ladder.py    # result files -> assets/barrier_ladder.png (needs matplotlib)
quarto render                      # -> writes ../docs/
```

`_quarto.yml` sets `output-dir: ../docs`. After rendering, `../docs/.nojekyll`
must exist (GitHub Pages needs it so Quarto's `site_libs/` is served) — it is
kept in the repo; re-`touch ../docs/.nojekyll` if a clean render removes it.

## Structure

- `index.qmd` — overview: six-answer summary, pipeline, findings, key numbers.
- `p1`–`p4-*.qmd` — the four findings pillars (the scientific narrative).
- `current-work.qmd` — the Sn-β benchmark, labelled work in progress (no results).
- `methods.qmd` — estimator glossary, transition-state standard, evidence rules,
  provenance. **Read before adding any barrier to a page.**
- `0X-*.qmd` — one page per study (URLs `01`–`08` are stable; `09` added for the
  reference-handoff work); each links to its full `FINDINGS_*.md` in the repo.
- `research-map.json` / `build_research_map.py` — the research map (never hand-edit
  `research-map.svg`).
- `build_barrier_ladder.py` — the only figure generated here; it reads result
  files and writes `assets/barrier_ladder.png`.
- `assets/` — figures copied from the study directories (update these when the
  underlying figures change, then re-render).

Conventions for adding pages, map nodes, and overview text (including the evidence
rules) are in `../.claude/skills/research-map/SKILL.md`.
