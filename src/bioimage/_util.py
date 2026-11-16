"""Small helpers that behave the same across scikit-image versions."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def remove_small(mask_or_labels: np.ndarray, min_size: int) -> np.ndarray:
    """Drop objects with fewer than ``min_size`` pixels (works on bool masks and label images)."""
    if mask_or_labels.dtype == bool:
        labels, _ = ndi.label(mask_or_labels)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_size
        keep[0] = False
        return keep[labels]
    labels = mask_or_labels.copy()
    sizes = np.bincount(labels.ravel())
    small = sizes < min_size
    small[0] = False
    labels[small[labels]] = 0
    return labels


def disk(radius: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return xx**2 + yy**2 <= radius**2


def opening(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return ndi.binary_opening(mask, structure=disk(radius))


def erosion(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return ndi.binary_erosion(mask, structure=disk(radius))
