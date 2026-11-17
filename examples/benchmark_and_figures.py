"""Benchmark on synthetic images with ground truth, plus figures for the README.

Run: python examples/benchmark_and_figures.py
The real microscopy example downloads one image through scikit-image (needs `pooch`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from bioimage import leaf, metrics, nuclei, synthetic

DOCS = Path(__file__).resolve().parent.parent / "docs"


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 6 * len(text), 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(im)


def leaf_benchmark(n: int = 30) -> None:
    print(f"## Leaves: segmentation and disease severity (synthetic, n={n} per background)\n")
    print("| background | leaf IoU | lesion IoU | severity error (mean abs, % points) | without margin-lesion recovery |")
    print("| --- | --- | --- | --- | --- |")
    for bg in ("paper", "bench", "soil"):
        rows = {True: [], False: []}
        for s in range(n):
            L = synthetic.leaf_image(seed=1000 + s, background=bg)
            for rec in rows:
                m = leaf.segment_leaf(L.rgb, recover_edge_lesions=rec)
                les = leaf.lesion_mask(L.rgb, m)
                sev = 100 * les.sum() / max(1, m.sum())
                rows[rec].append((metrics.iou(m, L.leaf_mask), metrics.iou(les, L.lesion_mask), abs(sev - L.severity)))
        a, b = np.mean(rows[True], axis=0), np.mean(rows[False], axis=0)
        print(f"| {bg} | {a[0]:.3f} | {a[1]:.3f} | {a[2]:.2f} | {b[2]:.2f} |")

    panels = []
    for seed, bg in ((2, "paper"), (5, "bench"), (8, "soil")):
        L = synthetic.leaf_image(size=256, seed=seed, background=bg, n_lesions=7)
        m, les, meas = leaf.analyse_leaf_image(L.rgb)
        truth = L.rgb.copy()
        truth[L.lesion_mask] = (230, 40, 200)
        panels.append(np.hstack([label(L.rgb, f"input ({bg})"), label(truth, f"truth {L.severity:.1f}%"),
                                 label(leaf.overlay(L.rgb, m, les), f"found {meas[0].severity_pct:.1f}%")]))
        Image.fromarray(L.rgb).save(DOCS / "samples" / f"leaf_{bg}.png")
    Image.fromarray(np.vstack(panels)).save(DOCS / "leaf_results.png")


def nuclei_benchmark(n: int = 10) -> None:
    print(f"\n## Nuclei: counting touching nuclei (synthetic, n={n}, 40 nuclei each, IoU >= 0.5 matching)\n")
    print("| method | precision | recall | F1 | count error |")
    print("| --- | --- | --- | --- | --- |")
    for ws, name in ((False, "threshold + connected components"), (True, "distance-transform watershed")):
        scores, count_err = [], []
        for s in range(n):
            N = synthetic.nuclei_image(seed=500 + s)
            lab = nuclei.segment_nuclei(N.image, watershed=ws)
            scores.append(metrics.match_objects(lab, N.labels))
            count_err.append(abs(int(lab.max()) - N.count))
        p = np.mean([x.precision for x in scores]); r = np.mean([x.recall for x in scores])
        f = np.mean([x.f1 for x in scores])
        print(f"| {name} | {p:.2f} | {r:.2f} | {f:.2f} | {np.mean(count_err):.1f} |")

    N = synthetic.nuclei_image(seed=501)
    a = nuclei.outlines(N.image, nuclei.segment_nuclei(N.image, watershed=False))
    b = nuclei.outlines(N.image, nuclei.segment_nuclei(N.image))
    Image.fromarray(np.hstack([label(a, "threshold only"), label(b, "watershed")])).save(DOCS / "nuclei_synthetic.png")


def real_mitosis() -> None:
    try:
        from skimage import data
        img = data.human_mitosis()
    except Exception as exc:  # no network or pooch missing
        print(f"\n(skipping real image: {exc})")
        return
    lab = nuclei.segment_nuclei(img, min_distance=4)
    meas = nuclei.measure_nuclei(img, lab)
    flagged = set(nuclei.bright_compact_nuclei(meas))
    rgb = nuclei.outlines(img, lab)
    for m in meas:
        if m.label in flagged:
            y, x = int(m.centroid_y), int(m.centroid_x)
            rgb[max(0, y - 9):y + 10, max(0, x - 9):x + 10][[0, -1], :] = (255, 60, 60)
            rgb[max(0, y - 9):y + 10, max(0, x - 9):x + 10][:, [0, -1]] = (255, 60, 60)
    Image.fromarray(rgb).save(DOCS / "nuclei_mitosis.png")
    print(f"\n## Real image: scikit-image `human_mitosis` ({img.shape[1]}x{img.shape[0]} px)\n")
    print(f"{lab.max()} nuclei segmented; {len(flagged)} flagged as unusually bright (candidate mitoses, red boxes).")


if __name__ == "__main__":
    (DOCS / "samples").mkdir(parents=True, exist_ok=True)
    leaf_benchmark()
    nuclei_benchmark()
    real_mitosis()
