"""Nucleus segmentation and counting in fluorescence microscopy (e.g. DAPI/Hoechst).

Counting nuclei is the first step of most cell-based assays: proliferation,
drug toxicity screens, mitotic index. The hard part is nuclei that touch;
a plain threshold merges them into one object. The standard classical fix is
a marker-controlled watershed on the distance transform:

1. Smooth, subtract the background, threshold with Otsu.
2. Distance transform: every foreground pixel gets its distance to the background,
   so each nucleus becomes a hill with its peak near the centre.
3. Local maxima of the distance map (at least ``min_distance`` apart) are markers,
   one per nucleus.
4. Watershed floods from the markers; the basins meet along the necks between
   touching nuclei.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, measure, segmentation

from ._util import remove_small


def foreground(image: np.ndarray, sigma: float = 1.0, background_radius: int = 25) -> np.ndarray:
    img = image.astype(float)
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    smooth = filters.gaussian(img, sigma)
    # white top-hat style background removal: subtract a heavily smoothed copy
    background = ndi.uniform_filter(ndi.grey_opening(smooth, size=background_radius), background_radius)
    corrected = np.clip(smooth - background, 0, None)
    mask = corrected > filters.threshold_otsu(corrected)
    mask = remove_small(mask, 20)
    return ndi.binary_fill_holes(mask)


def segment_nuclei(image: np.ndarray, min_distance: int = 5, sigma: float = 1.0, watershed: bool = True) -> np.ndarray:
    """Label image: 0 is background, 1..n are nuclei."""
    mask = foreground(image, sigma=sigma)
    if not watershed:
        return measure.label(mask)
    distance = ndi.distance_transform_edt(mask)
    distance = filters.gaussian(distance, 1.0)  # avoids several peaks on one irregular nucleus
    peaks = feature.peak_local_max(distance, min_distance=min_distance, labels=measure.label(mask), exclude_border=False)
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = segmentation.watershed(-distance, markers, mask=mask)
    return remove_small(labels, 20)


@dataclass(frozen=True)
class NucleusMeasurement:
    label: int
    area_px: int
    mean_intensity: float
    integrated_intensity: float  # proportional to DNA content
    eccentricity: float
    solidity: float
    centroid_y: float
    centroid_x: float

    def as_dict(self) -> dict:
        return asdict(self)


def measure_nuclei(image: np.ndarray, labels: np.ndarray) -> list[NucleusMeasurement]:
    out = []
    for r in measure.regionprops(labels, intensity_image=image.astype(float)):
        out.append(
            NucleusMeasurement(
                label=r.label,
                area_px=int(r.area),
                mean_intensity=round(float(r.intensity_mean), 4),
                integrated_intensity=round(float(r.intensity_mean * r.area), 2),
                eccentricity=round(float(r.eccentricity), 4),
                solidity=round(float(r.solidity), 4),
                centroid_y=round(float(r.centroid[0]), 1),
                centroid_x=round(float(r.centroid[1]), 1),
            )
        )
    return out


def bright_compact_nuclei(measurements: list[NucleusMeasurement], z: float = 2.0) -> list[int]:
    """Labels of nuclei much brighter than average.

    Condensed mitotic chromosomes stain brighter than interphase nuclei, so this
    is a crude mitosis flag (a real screen would train a classifier on labelled
    examples).
    """
    if not measurements:
        return []
    vals = np.array([m.mean_intensity for m in measurements])
    cut = vals.mean() + z * vals.std()
    return [m.label for m in measurements if m.mean_intensity > cut]


def outlines(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """RGB image with nucleus boundaries drawn in yellow."""
    img = image.astype(float)
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    rgb = np.dstack([img] * 3)
    rgb[segmentation.find_boundaries(labels, mode="outer")] = (1.0, 0.85, 0.0)
    return (rgb * 255).astype(np.uint8)
