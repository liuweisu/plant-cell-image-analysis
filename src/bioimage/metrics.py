"""Segmentation quality metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    pred, truth = pred.astype(bool), truth.astype(bool)
    union = (pred | truth).sum()
    return float((pred & truth).sum() / union) if union else 1.0


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    pred, truth = pred.astype(bool), truth.astype(bool)
    denom = pred.sum() + truth.sum()
    return float(2 * (pred & truth).sum() / denom) if denom else 1.0


@dataclass(frozen=True)
class ObjectScore:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match_objects(pred: np.ndarray, truth: np.ndarray, threshold: float = 0.5) -> ObjectScore:
    """Instance-level matching: a predicted object is correct if IoU with a true object >= threshold.

    With threshold >= 0.5 each true object can match at most one prediction,
    so a greedy match is exact. This is the metric used in the Data Science
    Bowl 2018 nucleus segmentation challenge (there averaged over thresholds).
    """
    n_pred, n_true = int(pred.max()), int(truth.max())
    if n_pred == 0 or n_true == 0:
        return ObjectScore(0, n_pred, n_true)
    # contingency table of pixel overlaps
    overlap = np.zeros((n_true + 1, n_pred + 1), np.int64)
    np.add.at(overlap, (truth.ravel(), pred.ravel()), 1)
    area_t = overlap.sum(axis=1)
    area_p = overlap.sum(axis=0)
    inter = overlap[1:, 1:]
    union = area_t[1:, None] + area_p[None, 1:] - inter
    ious = inter / np.maximum(union, 1)
    tp = int((ious >= threshold).any(axis=1).sum())
    return ObjectScore(tp, n_pred - tp, n_true - tp)
