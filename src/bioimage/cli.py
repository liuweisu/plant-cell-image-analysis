"""Command line: ``bioimage leaf photos/*.jpg`` and ``bioimage nuclei images/*.tif``."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__, leaf, nuclei


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _load_gray(path: Path) -> np.ndarray:
    img = np.asarray(Image.open(path))
    return img if img.ndim == 2 else img[..., :3].mean(axis=2)


def _write_csv(rows: list[dict], out: Path | None) -> None:
    if not rows:
        print("no objects found", file=sys.stderr)
        return
    fh = open(out, "w", newline="") if out else sys.stdout
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    if out:
        fh.close()
        print(f"wrote {len(rows)} rows to {out}", file=sys.stderr)


def cmd_leaf(args) -> None:
    rows = []
    for path in map(Path, args.images):
        rgb = _load_rgb(path)
        mask, lesions, measurements = leaf.analyse_leaf_image(rgb, min_size=args.min_size)
        rows += [{"image": path.name, **m.as_dict()} for m in measurements]
        if args.overlays:
            Path(args.overlays).mkdir(parents=True, exist_ok=True)
            Image.fromarray(leaf.overlay(rgb, mask, lesions)).save(Path(args.overlays) / f"{path.stem}_overlay.png")
    _write_csv(rows, Path(args.out) if args.out else None)


def cmd_nuclei(args) -> None:
    rows = []
    for path in map(Path, args.images):
        img = _load_gray(path)
        labels = nuclei.segment_nuclei(img, min_distance=args.min_distance, watershed=not args.no_watershed)
        meas = nuclei.measure_nuclei(img, labels)
        flagged = set(nuclei.bright_compact_nuclei(meas))
        rows += [{"image": path.name, **m.as_dict(), "bright_flag": m.label in flagged} for m in meas]
        print(f"{path.name}: {labels.max()} nuclei, {len(flagged)} unusually bright", file=sys.stderr)
        if args.overlays:
            Path(args.overlays).mkdir(parents=True, exist_ok=True)
            Image.fromarray(nuclei.outlines(img, labels)).save(Path(args.overlays) / f"{path.stem}_nuclei.png")
    _write_csv(rows, Path(args.out) if args.out else None)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="bioimage")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(required=True)

    a = sub.add_parser("leaf", help="leaf area, shape and disease severity from RGB photos")
    a.add_argument("images", nargs="+")
    a.add_argument("--out", help="CSV file (default: stdout)")
    a.add_argument("--overlays", help="folder for overlay images to check the segmentation")
    a.add_argument("--min-size", type=int, default=500, help="ignore green objects smaller than this (px)")
    a.set_defaults(func=cmd_leaf)

    b = sub.add_parser("nuclei", help="count and measure nuclei in fluorescence images")
    b.add_argument("images", nargs="+")
    b.add_argument("--out")
    b.add_argument("--overlays")
    b.add_argument("--min-distance", type=int, default=5, help="minimum distance between nucleus centres (px)")
    b.add_argument("--no-watershed", action="store_true", help="plain threshold, to see what the watershed adds")
    b.set_defaults(func=cmd_nuclei)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
