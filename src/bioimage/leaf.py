"""Leaf segmentation, disease severity and leaf shape from RGB photos.

Pipeline (classical, no training data needed):

1. **Vegetation index.** Excess green, ExG = 2g - r - b on chromaticity
   coordinates (Woebbecke et al. 1995), separates plant from soil, paper or bench
   and is insensitive to overall brightness.
2. **Threshold** ExG with Otsu's method, fill holes (lesions are not green, so they
   would otherwise be holes), remove small objects.
3. **Lesions** are leaf pixels whose colour has left the green range: in CIELAB,
   healthy tissue has a strongly negative a* (green); necrotic tissue moves to
   a* > 0 (brown/red), chlorosis to high b* with weak a* (yellow).
4. **Severity** = lesion area / leaf area, the standard plant-pathology measure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import color, filters, measure

from ._util import disk, erosion, opening, remove_small


def excess_green(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(float)
    total = rgb.sum(axis=2) + 1e-6
    r, g, b = (rgb[..., i] / total for i in range(3))
    return 2 * g - r - b


def _border_background(rgb: np.ndarray, width: int = 4) -> np.ndarray:
    """Median CIELAB colour of the image border, assumed to be background."""
    lab = color.rgb2lab(rgb)
    border = np.concatenate([lab[:width].reshape(-1, 3), lab[-width:].reshape(-1, 3),
                             lab[:, :width].reshape(-1, 3), lab[:, -width:].reshape(-1, 3)])
    return np.median(border, axis=0)


def segment_leaf(
    rgb: np.ndarray,
    min_size: int = 500,
    keep_largest: bool = False,
    recover_edge_lesions: bool = True,
    background_delta_e: float = 20.0,
    closing_fraction: float = 0.1,
) -> np.ndarray:
    """Boolean mask of leaf tissue (including diseased spots inside the leaf).

    Lesions are not green, so ExG alone cuts them out of the leaf. Lesions in
    the middle of the blade come back with hole filling, but lesions on the
    leaf margin are open to the background and stay missing. With
    ``recover_edge_lesions`` the green mask is closed with a disk (radius
    ``closing_fraction`` of the image size), and pixels
    added by the closing are kept only if their colour differs from the
    background (CIELAB distance to the median border colour).
    """
    exg = excess_green(rgb)
    thresh = filters.threshold_otsu(exg)
    green = opening(exg > thresh, 1)
    green = remove_small(green, min_size)
    mask = ndi.binary_fill_holes(green)
    if recover_edge_lesions and mask.any():
        radius = max(3, round(closing_fraction * min(rgb.shape[:2])))
        closed = ndi.binary_fill_holes(ndi.binary_closing(mask, structure=disk(radius), iterations=1))
        lab = color.rgb2lab(rgb)
        delta_e = np.linalg.norm(lab - _border_background(rgb), axis=2)
        mask = mask | (closed & (delta_e > background_delta_e))
        mask = ndi.binary_fill_holes(opening(mask, 1))
    mask = remove_small(mask, min_size)
    if keep_largest and mask.any():
        lab_img = measure.label(mask)
        largest = np.argmax(np.bincount(lab_img.ravel())[1:]) + 1
        mask = lab_img == largest
    return mask


def lesion_mask(rgb: np.ndarray, leaf: np.ndarray, a_star_threshold: float = -2.0, min_size: int = 10) -> np.ndarray:
    """Diseased (necrotic) pixels inside the leaf.

    The threshold on a* sits between healthy green (a* around -20 to -30) and
    brown lesions (a* > 0). Chlorotic yellow halos stay just below it, so they
    are counted as leaf but not as lesion, which matches how severity is
    usually scored visually.
    """
    lab = color.rgb2lab(rgb)
    lesion = (lab[..., 1] > a_star_threshold) & leaf
    lesion = opening(lesion, 1)
    return remove_small(lesion, min_size)


@dataclass(frozen=True)
class LeafMeasurement:
    leaf_id: int
    area_px: int
    perimeter_px: float
    length_px: float  # major axis of the fitted ellipse
    width_px: float  # minor axis
    aspect_ratio: float
    solidity: float  # area / convex hull area; lower for lobed or serrated leaves
    circularity: float  # 4*pi*area / perimeter^2; 1 for a circle
    lesion_count: int
    lesion_area_px: int
    severity_pct: float
    mean_exg: float  # greenness of the healthy part

    def as_dict(self) -> dict:
        return asdict(self)


def measure_leaves(rgb: np.ndarray, leaf: np.ndarray, lesions: np.ndarray, pixel_size: float | None = None) -> list[LeafMeasurement]:
    """One measurement per connected leaf. ``pixel_size`` (mm/px) is reserved for calibrated images."""
    del pixel_size
    labels = measure.label(leaf)
    exg = excess_green(rgb)
    out: list[LeafMeasurement] = []
    for region in measure.regionprops(labels):
        sl = labels == region.label
        les = lesions & sl
        n_lesions = int(measure.label(les).max())
        healthy = sl & ~les
        perim = region.perimeter or 1.0
        out.append(
            LeafMeasurement(
                leaf_id=region.label,
                area_px=int(region.area),
                perimeter_px=round(float(perim), 1),
                length_px=round(float(region.axis_major_length), 1),
                width_px=round(float(region.axis_minor_length), 1),
                aspect_ratio=round(float(region.axis_major_length / max(region.axis_minor_length, 1e-6)), 3),
                solidity=round(float(region.solidity), 4),
                circularity=round(float(4 * np.pi * region.area / perim**2), 4),
                lesion_count=n_lesions,
                lesion_area_px=int(les.sum()),
                severity_pct=round(100.0 * les.sum() / region.area, 2),
                mean_exg=round(float(exg[healthy].mean()) if healthy.any() else 0.0, 4),
            )
        )
    return out


def analyse_leaf_image(rgb: np.ndarray, min_size: int = 500) -> tuple[np.ndarray, np.ndarray, list[LeafMeasurement]]:
    leaf = segment_leaf(rgb, min_size=min_size)
    lesions = lesion_mask(rgb, leaf)
    return leaf, lesions, measure_leaves(rgb, leaf, lesions)


def overlay(rgb: np.ndarray, leaf: np.ndarray, lesions: np.ndarray) -> np.ndarray:
    """Leaf outline in white, lesions tinted magenta: easy to check by eye."""
    out = rgb.astype(float).copy()
    out[lesions] = 0.45 * out[lesions] + 0.55 * np.array([230, 40, 200])
    edge = leaf ^ erosion(leaf, 1)
    out[edge] = (255, 255, 255)
    return out.astype(np.uint8)
