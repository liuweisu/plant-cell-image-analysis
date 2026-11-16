import numpy as np
import pytest

from bioimage import _util, leaf, metrics, nuclei, synthetic


@pytest.fixture(scope="module")
def leaves():
    return [synthetic.leaf_image(seed=s, background=bg) for s in range(4) for bg in ("paper", "bench")]


def test_excess_green_separates_leaf_from_paper():
    img = synthetic.leaf_image(seed=0)
    exg = leaf.excess_green(img.rgb)
    healthy = img.leaf_mask & ~img.lesion_mask
    assert exg[healthy].mean() > exg[~img.leaf_mask].mean() + 0.1


def test_leaf_segmentation_iou(leaves):
    ious = [metrics.iou(leaf.segment_leaf(L.rgb), L.leaf_mask) for L in leaves]
    assert np.mean(ious) > 0.95


def test_edge_lesions_are_kept_in_the_leaf(leaves):
    def missed(recover):
        return sum((L.lesion_mask & ~leaf.segment_leaf(L.rgb, recover_edge_lesions=recover)).sum() for L in leaves)
    assert missed(True) < 0.5 * missed(False)


def test_severity_error_small(leaves):
    errors = []
    for L in leaves:
        _, _, meas = leaf.analyse_leaf_image(L.rgb)
        assert len(meas) == 1
        errors.append(abs(meas[0].severity_pct - L.severity))
    assert np.mean(errors) < 1.5


def test_healthy_leaf_has_no_lesions():
    L = synthetic.leaf_image(seed=3, n_lesions=0)
    _, lesions, meas = leaf.analyse_leaf_image(L.rgb)
    assert meas[0].severity_pct < 0.5


def test_leaf_shape_measurements():
    L = synthetic.leaf_image(seed=1, n_lesions=0)
    m = leaf.analyse_leaf_image(L.rgb)[2][0]
    assert m.aspect_ratio > 1.3  # leaves are longer than wide
    assert 0.8 < m.solidity <= 1.0


def test_watershed_splits_touching_nuclei():
    scores = {True: [], False: []}
    for s in range(3):
        N = synthetic.nuclei_image(seed=s, touching=0.5)
        for ws in scores:
            scores[ws].append(metrics.match_objects(nuclei.segment_nuclei(N.image, watershed=ws), N.labels).f1)
    assert np.mean(scores[True]) > 0.9
    assert np.mean(scores[True]) > np.mean(scores[False]) + 0.2


def test_nucleus_measurements():
    N = synthetic.nuclei_image(seed=0)
    labels = nuclei.segment_nuclei(N.image)
    meas = nuclei.measure_nuclei(N.image, labels)
    assert len(meas) == labels.max()
    assert all(m.area_px > 20 for m in meas)


def test_iou_and_dice():
    a = np.zeros((10, 10), bool); a[:5] = True
    b = np.zeros((10, 10), bool); b[:5, :5] = True
    assert metrics.iou(a, b) == pytest.approx(0.5)
    assert metrics.dice(a, b) == pytest.approx(2 * 25 / 75)


def test_object_matching():
    truth = np.zeros((20, 20), int); truth[2:8, 2:8] = 1; truth[12:18, 12:18] = 2
    pred = np.zeros((20, 20), int); pred[2:8, 2:8] = 1; pred[0:2, 15:20] = 2
    s = metrics.match_objects(pred, truth)
    assert (s.true_positives, s.false_positives, s.false_negatives) == (1, 1, 1)


def test_remove_small_on_labels_and_masks():
    lab = np.zeros((10, 10), int); lab[0, 0] = 1; lab[5:9, 5:9] = 2
    out = _util.remove_small(lab, 3)
    assert set(np.unique(out)) == {0, 2}
    assert _util.remove_small(lab > 0, 3).sum() == 16
