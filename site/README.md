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
python3 build_sn_beta_figures.py   # Sn-β structures + audit summary -> assets/sn_beta_*.png (needs ASE)
quarto render                      # -> writes ../docs/
```

`_quarto.yml` sets `output-dir: ../docs`. After rendering, `../docs/.nojekyll`
must exist (GitHub Pages needs it so Quarto's `site_libs/` is served) — it is
kept in the repo; re-`touch ../docs/.nojekyll` if a clean render removes it.

## Structure

- `index.qmd` — overview: six-answer summary, pipeline, findings, key numbers.
- `p1`–`p4-*.qmd` — the four findings pillars (the scientific narrative).
- `current-work.qmd` — the Sn-β benchmark, labelled work in progress: an audit of a
  reconstructed model and a collaborator's saved MACE-POLAR run (both reference
  conventions), no benchmark result yet.
- `methods.qmd` — estimator glossary, transition-state standard, evidence rules,
  provenance. **Read before adding any barrier to a page.**
- `0X-*.qmd` — one page per study (URLs `01`–`08` are stable; `09` added for the
  reference-handoff work); each links to its full `FINDINGS_*.md` in the repo.
- `research-map.json` / `build_research_map.py` — the research map (never hand-edit
  `research-map.svg`).
- `build_barrier_ladder.py` — reads result files and writes `assets/barrier_ladder.png`.
- `build_sn_beta_figures.py` — reads the reconstructed Sn-β structures and
  `sweep/sn_beta_hydride/audit/tian_tt_test/tian_polar_summary.json` and writes the
  four `assets/sn_beta_*.png` figures of the Current work page (hero, stacked
  hero for phones, two-reference energy profile, x/y/z structure views). Every
  plotted energy difference is checked against the summary's labelled energetics.
- `assets/` — figures copied from the study directories (update these when the
  underlying figures change, then re-render).

Conventions for adding pages, map nodes, and overview text (including the evidence
rules) are in `../.claude/skills/research-map/SKILL.md`.
