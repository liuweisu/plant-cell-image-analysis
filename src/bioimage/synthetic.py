"""Synthetic images with exact ground truth.

Hand-annotated masks are expensive and themselves imperfect, so every method
here is first measured on generated images where the true leaf outline,
lesion pixels and nucleus labels are known exactly. The real-image examples
then show the same code on actual data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import draw


@dataclass
class LeafImage:
    rgb: np.ndarray  # H x W x 3 uint8
    leaf_mask: np.ndarray  # bool, leaf including lesions
    lesion_mask: np.ndarray  # bool, diseased tissue

    @property
    def severity(self) -> float:
        """Percentage of leaf area that is diseased."""
        return 100.0 * self.lesion_mask.sum() / max(1, self.leaf_mask.sum())


def _smooth_noise(shape, sigma, rng):
    n = ndi.gaussian_filter(rng.standard_normal(shape), sigma)
    return n / (np.abs(n).max() + 1e-9)


def leaf_shape(size: int = 256, rng: np.random.Generator | None = None, angle: float | None = None) -> np.ndarray:
    """A simple dicot leaf: lanceolate-to-ovate blade with a pointed tip and a petiole."""
    rng = rng or np.random.default_rng()
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    cx, cy = size / 2, size / 2
    angle = rng.uniform(-0.6, 0.6) if angle is None else angle
    # rotate coordinates
    x = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
    y = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
    length = size * rng.uniform(0.34, 0.42)
    width = length * rng.uniform(0.35, 0.6)
    t = np.clip((y + length) / (2 * length), 0, 1)  # 0 at base, 1 at tip
    # widest a bit below the middle, pointed at the tip
    profile = width * np.sin(np.pi * t ** rng.uniform(0.75, 0.95)) ** 0.8
    wobble = 1 + 0.04 * _smooth_noise((size, size), size / 25, rng)
    blade = (np.abs(x) <= profile * wobble) & (np.abs(y) <= length)
    petiole = (np.abs(x) <= size * 0.012) & (y > length - 2) & (y < length + size * 0.12)
    return blade | petiole


def leaf_image(
    size: int = 256,
    n_lesions: int = 6,
    seed: int | None = None,
    background: str = "paper",
) -> LeafImage:
    """Render a leaf photographed on a background, with necrotic lesions and chlorotic halos."""
    rng = np.random.default_rng(seed)
    leaf = leaf_shape(size, rng)
    yy, xx = np.mgrid[0:size, 0:size]

    # background: white paper, grey bench or brown soil, with an illumination gradient
    bg_colors = {"paper": (232, 230, 222), "bench": (150, 152, 156), "soil": (110, 84, 62)}
    base = np.array(bg_colors[background], float)
    grad = 1 + 0.12 * ((xx / size) - 0.5)[..., None]
    img = base * grad + rng.normal(0, 4, (size, size, 3))
    if background == "soil":
        img += 25 * _smooth_noise((size, size), 3, rng)[..., None]

    # leaf colour with veins and texture
    green = np.array([58, 118, 46], float) * rng.uniform(0.85, 1.15)
    texture = 10 * _smooth_noise((size, size), 4, rng)[..., None]
    leaf_rgb = green + texture
    ys, xs = np.nonzero(leaf)
    cy_, cx_ = ys.mean(), xs.mean()
    # midrib approximated as the principal axis of the leaf pixels
    cov = np.cov(np.vstack([xs - cx_, ys - cy_]))
    vx, vy = np.linalg.eigh(cov)[1][:, -1]
    dist_mid = np.abs((xx - cx_) * vy - (yy - cy_) * vx)
    vein = np.exp(-(dist_mid**2) / 2.0)
    leaf_rgb = leaf_rgb + vein[..., None] * np.array([40, 45, 25])

    # lesions: brown necrotic centre, yellow chlorotic halo (halo counts as healthy-but-yellow)
    lesion = np.zeros((size, size), bool)
    halo_img = np.zeros((size, size))
    for _ in range(n_lesions):
        k = rng.integers(len(ys))
        r = rng.uniform(size * 0.015, size * 0.05)
        rr, cc = draw.ellipse(ys[k], xs[k], r, r * rng.uniform(0.7, 1.3), shape=(size, size),
                              rotation=rng.uniform(0, np.pi))
        spot = np.zeros((size, size), bool)
        spot[rr, cc] = True
        spot &= leaf
        lesion |= spot
        halo_img = np.maximum(halo_img, ndi.gaussian_filter(spot.astype(float), r * 0.35))
    halo = np.clip(halo_img * 2.0, 0, 1)[..., None] * (~lesion)[..., None]
    leaf_rgb = leaf_rgb * (1 - 0.6 * halo) + np.array([190, 175, 60]) * 0.6 * halo
    necrotic = np.array([95, 62, 35], float) + 12 * _smooth_noise((size, size), 2, rng)[..., None]
    leaf_rgb = np.where(lesion[..., None], necrotic, leaf_rgb)

    # soft anti-aliased edge and a faint shadow
    shadow = ndi.gaussian_filter(np.roll(leaf, (4, 4), (0, 1)).astype(float), 3)[..., None]
    img = img * (1 - 0.25 * shadow)
    alpha = ndi.gaussian_filter(leaf.astype(float), 0.7)[..., None]
    img = img * (1 - alpha) + leaf_rgb * alpha
    img += rng.normal(0, 3, img.shape)
    return LeafImage(np.clip(img, 0, 255).astype(np.uint8), leaf, lesion & leaf)


@dataclass
class NucleiImage:
    image: np.ndarray  # H x W float in [0, 1]
    labels: np.ndarray  # int, 0 = background, 1..n = nucleus id

    @property
    def count(self) -> int:
        return int(self.labels.max())


def nuclei_image(size: int = 256, n: int = 40, seed: int | None = None, touching: float = 0.4) -> NucleiImage:
    """Fluorescence-like image of DAPI-stained nuclei, a fraction of them touching.

    Nuclei are ellipses with uneven intensity, blurred by the optics, on a
    background with uneven illumination and Poisson-like noise.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros((size, size), np.int32)
    centers: list[tuple[float, float, float]] = []
    attempts = 0
    while len(centers) < n and attempts < n * 200:
        attempts += 1
        r = rng.uniform(size * 0.025, size * 0.045)
        if centers and rng.random() < touching:
            # place next to an existing nucleus so the two touch
            py, px, pr = centers[rng.integers(len(centers))]
            ang = rng.uniform(0, 2 * np.pi)
            cy, cx = py + np.sin(ang) * (pr + r) * 0.9, px + np.cos(ang) * (pr + r) * 0.9
        else:
            cy, cx = rng.uniform(r, size - r), rng.uniform(r, size - r)
        if not (r < cy < size - r and r < cx < size - r):
            continue
        rr, cc = draw.ellipse(cy, cx, r, r * rng.uniform(0.75, 1.0), shape=labels.shape,
                              rotation=rng.uniform(0, np.pi))
        if (labels[rr, cc] > 0).mean() > 0.05:  # do not let nuclei overlap much
            continue
        free = labels[rr, cc] == 0
        labels[rr[free], cc[free]] = len(centers) + 1
        centers.append((cy, cx, r))

    intensity = np.zeros((size, size))
    for i in range(1, labels.max() + 1):
        intensity[labels == i] = rng.uniform(0.45, 1.0)
    chromatin = 1 + 0.25 * _smooth_noise((size, size), 1.5, rng)
    img = ndi.gaussian_filter(intensity * chromatin, 1.2)
    yy, xx = np.mgrid[0:size, 0:size] / size
    img += 0.06 + 0.05 * xx  # uneven illumination
    img = rng.poisson(img * 200) / 200.0
    return NucleiImage(np.clip(img / img.max(), 0, 1), labels)
