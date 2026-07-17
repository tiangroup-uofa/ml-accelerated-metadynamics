#!/usr/bin/env python3
"""
make_movie.py
=============
Render a metadynamics trajectory to a PNG stack AND an animated GIF, with
BONDS drawn, using ASE's built-in renderer (covalent-radii bonds) + PIL.

This is the "easy ASE visualization" the task asks for: a short movie / PNG
stack to eyeball what actually happens (cluster staying together, a bond
breaking, molecules drifting apart).

For each selected frame we call ase.io.write(..., format='png'), which draws
atoms as spheres and bonds between atoms within covalent contact. Frames are
then combined into movie_<tag>.gif with PIL.

Usage:
    python make_movie.py --trj runs/n3/low/xtb.trj --tag n3_low --stride 4
    python make_movie.py --trj runs/n1/high/xtb.trj --tag n1_high --stride 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"


def render(trj: Path, tag: str, stride: int, max_frames: int,
           rotation: str, clip_A: float):
    frames = read(str(trj), index=":", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    sel = frames[::stride][:max_frames]

    png_dir = FIG / f"movie_{tag}_frames"
    png_dir.mkdir(parents=True, exist_ok=True)

    # A fixed view scale keeps the movie stable. If molecules fly very far
    # apart we optionally recentre on the centroid and clip the view so the
    # cluster stays visible (drifting fragments leave the frame -- which is
    # itself informative).
    png_paths = []
    for i, atoms in enumerate(sel):
        a = atoms.copy()
        a.positions -= a.get_center_of_mass()
        p = png_dir / f"frame_{i:04d}.png"
        # bbox clips the view to +/- clip_A around the centroid
        write(str(p), a, format="png", rotation=rotation,
              bbox=(-clip_A, -clip_A, clip_A, clip_A), scale=20)
        png_paths.append(p)
    print(f"  rendered {len(png_paths)} PNG frames -> {png_dir}")
    return png_paths


def make_gif(png_paths, tag: str, fps: int):
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in png_paths]
    if not imgs:
        print("  no frames to animate")
        return
    gif = FIG / f"movie_{tag}.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    print(f"  wrote {gif}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trj", required=True, type=Path)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--stride", type=int, default=4,
                    help="use every Nth frame (default 4)")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--rotation", default="-70x, 10y, 0z",
                    help="ASE view rotation string")
    ap.add_argument("--clip-A", type=float, default=6.0,
                    help="half-width of the view box in Angstrom")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()
    tag = args.tag or args.trj.parent.name
    FIG.mkdir(parents=True, exist_ok=True)
    pngs = render(args.trj, tag, args.stride, args.max_frames,
                  args.rotation, args.clip_A)
    make_gif(pngs, tag, args.fps)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
