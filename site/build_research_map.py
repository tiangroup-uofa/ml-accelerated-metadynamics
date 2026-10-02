#!/usr/bin/env python3
"""
build_research_map.py — render research-map.json to a self-contained inline SVG.

Deterministic grid layout (no physics, no jitter, no JavaScript). Nodes with an
`href` become real <a> elements so they are clickable and keyboard-reachable;
`tip` becomes a native <title> tooltip. Output is included verbatim by
research-map.qmd.

    python3 build_research_map.py            # writes research-map.svg
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "research-map.json"
OUT = HERE / "research-map.svg"

STYLE = {
    "core":   dict(fill="#1f2933", stroke="#1f2933", text="#ffffff", weight="650", dash=None),
    "pillar": dict(fill="#2e6f95", stroke="#2e6f95", text="#ffffff", weight="620", dash=None),
    "study":  dict(fill="#ffffff", stroke="#2e6f95", text="#1f2933", weight="500", dash=None),
    "result": dict(fill="#f4f7f9", stroke="#a9b7c2", text="#1f2933", weight="400", dash=None),
    "future": dict(fill="#ffffff", stroke="#b8c3cd", text="#66727e", weight="400", dash="4 3"),
    # work in progress: infrastructure built, no scientific result yet (linked page)
    "wip":    dict(fill="#fdf8ec", stroke="#a86b00", text="#5c3b00", weight="550", dash="4 3"),
}
EDGE = {
    "pillar":    dict(stroke="#8fa3b3", width=1.6, dash=None),
    "contains":  dict(stroke="#cfd8e0", width=1.2, dash=None),
    "motivates": dict(stroke="#c1121f", width=1.4, dash="5 4"),
    "validates": dict(stroke="#5b8c5a", width=1.4, dash="5 4"),
}
NODE_H = 30
CHAR_W = 6.15          # approximate advance width at 11.5px DejaVu Sans
FONT = ("system-ui, -apple-system, 'Segoe UI', Roboto, "
        "'Helvetica Neue', Arial, sans-serif")


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def fit(label: str, width: int) -> str:
    """Truncate a label to the box width (labels are authored to fit; this is a guard)."""
    limit = int((width - 18) / CHAR_W)
    return label if len(label) <= limit else label[: limit - 1] + "…"


def main() -> None:
    spec = json.loads(SRC.read_text())
    lay = spec["layout"]
    col_x, col_w, row_h, top = lay["col_x"], lay["col_w"], lay["row_h"], lay["top"]

    geo = {}
    for n in spec["nodes"]:
        c = n["col"]
        x, w = col_x[c], col_w[c]
        y = top + n["row"] * row_h
        geo[n["id"]] = dict(x=x, y=y, w=w, h=NODE_H, cx=x + w / 2, cy=y + NODE_H / 2)

    width = col_x[-1] + col_w[-1] + 20
    height = max(g["y"] + g["h"] for g in geo.values()) + 26

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Research map: how the individual studies connect '
        f'into one research programme" font-family="{FONT}">',
        "<defs>",
    ]
    for name, e in EDGE.items():
        out.append(
            f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{e["stroke"]}"/></marker>'
        )
    out.append("</defs>")

    # ---- edges (drawn first so nodes sit on top) --------------------------- #
    for ed in spec["edges"]:
        a, b = geo[ed["from"]], geo[ed["to"]]
        st = EDGE[ed["rel"]]
        dash = f' stroke-dasharray="{st["dash"]}"' if st["dash"] else ""

        if ed["rel"] in ("motivates", "validates"):
            # vertical link bowing out to the left of the column
            bow = ed.get("bow", 34)
            path = (f'M {a["x"]} {a["cy"]} C {a["x"] - bow} {a["cy"]}, '
                    f'{b["x"] - bow} {b["cy"]}, {b["x"]} {b["cy"]}')
            out.append(f'<path d="{path}" fill="none" stroke="{st["stroke"]}" '
                       f'stroke-width="{st["width"]}"{dash} '
                       f'marker-end="url(#ah-{ed["rel"]})"/>')
            if ed.get("label"):
                mx = min(a["x"], b["x"]) - bow * 0.72
                my = (a["cy"] + b["cy"]) / 2
                out.append(
                    f'<text x="{mx:.1f}" y="{my:.1f}" font-size="9.5" '
                    f'fill="{st["stroke"]}" text-anchor="middle" '
                    f'dominant-baseline="middle" stroke="#ffffff" '
                    f'stroke-width="3" paint-order="stroke">'
                    f'<tspan dy="0">{esc(ed["label"])}</tspan></text>'
                )
        else:
            # left-to-right hierarchy link
            x1, y1 = a["x"] + a["w"], a["cy"]
            x2, y2 = b["x"], b["cy"]
            mid = (x1 + x2) / 2
            path = f"M {x1} {y1} C {mid} {y1}, {mid} {y2}, {x2} {y2}"
            out.append(f'<path d="{path}" fill="none" stroke="{st["stroke"]}" '
                       f'stroke-width="{st["width"]}"{dash}/>')

    # ---- nodes ------------------------------------------------------------- #
    for n in spec["nodes"]:
        g = geo[n["id"]]
        s = STYLE[n["kind"]]
        dash = f' stroke-dasharray="{s["dash"]}"' if s["dash"] else ""
        label = fit(n["label"], g["w"])
        body = [
            f'<rect class="node-box" x="{g["x"]}" y="{g["y"]:.1f}" width="{g["w"]}" '
            f'height="{g["h"]}" rx="5" fill="{s["fill"]}" stroke="{s["stroke"]}" '
            f'stroke-width="1.3"{dash}/>',
            f'<text class="node-label" x="{g["cx"]:.1f}" y="{g["cy"] + 4:.1f}" '
            f'font-size="11.5" font-weight="{s["weight"]}" fill="{s["text"]}" '
            f'text-anchor="middle">{esc(label)}</text>',
        ]
        if n.get("tip"):
            body.insert(0, f'<title>{esc(n["tip"])}</title>')
        inner = "".join(body)
        if n.get("href"):
            out.append(f'<a href="{esc(n["href"])}">{inner}</a>')
        else:
            out.append(f"<g>{inner}</g>")

    out.append("</svg>")
    OUT.write_text("\n".join(out) + "\n")

    kinds = {}
    for n in spec["nodes"]:
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    linked = sum(1 for n in spec["nodes"] if n.get("href"))
    print(f"wrote {OUT.name}: {len(spec['nodes'])} nodes "
          f"({', '.join(f'{k} {v}' for k, v in kinds.items())}), "
          f"{len(spec['edges'])} edges, {linked} clickable, "
          f"viewBox {width}×{height}")


if __name__ == "__main__":
    main()
