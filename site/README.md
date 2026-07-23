# Results site (Quarto)

Source for the project results site — a plain [Quarto](https://quarto.org)
website that renders to `../docs/` and is served via GitHub Pages at
<https://tiangroup-uofa.github.io/xtb_metadynamics/>.

## Edit / render

```bash
# needs the quarto CLI (https://quarto.org/docs/get-started/)
cd site
quarto render          # -> writes ../docs/
```

`_quarto.yml` sets `output-dir: ../docs`. After rendering, `../docs/.nojekyll`
must exist (GitHub Pages needs it so Quarto's `site_libs/` is served) — it is
kept in the repo; re-`touch ../docs/.nojekyll` if a clean render removes it.

## Structure

- `index.qmd` — overview + results-at-a-glance.
- `0X-*.qmd` — one page per study; each links to its full `FINDINGS_*.md` in the repo.
- `assets/` — figures copied from the study directories (update these when the
  underlying figures change, then re-render).
