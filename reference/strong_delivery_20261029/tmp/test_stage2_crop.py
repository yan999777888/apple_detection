"""Correctness tests for the second-stage crop verifier helpers.

The important one is the prefix sweep: the reported precision/recall must be
identical to a direct one-to-one greedy matching at the same threshold, because
the whole experiment is judged on that number.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tmp'))

import stage2_crop as s2  # noqa: E402


def _record(rng, n_boxes, n_gt):
    gt = rng.uniform(0, 200, size=(n_gt, 2))
    gt = np.concatenate([gt, gt + rng.uniform(20, 60, size=(n_gt, 2))], 1)
    boxes = gt[rng.randint(0, max(n_gt, 1), size=n_boxes)] + rng.normal(0, 6, size=(n_boxes, 4))
    boxes = np.concatenate([boxes[:, :2], boxes[:, :2] + rng.uniform(20, 60, size=(n_boxes, 2))], 1)
    stray = rng.uniform(0, 200, size=(max(n_boxes // 3, 1), 4))
    stray = np.concatenate([stray[:, :2], stray[:, :2] + 30], 1)
    boxes = np.concatenate([boxes, stray])[:n_boxes]
    return dict(boxes=boxes.astype('float32'), gt=gt.astype('float32'),
                scores=rng.uniform(0, 1, size=len(boxes)).astype('float32'))


def test_prefix_sweep_matches_direct_greedy_matching():
    rng = np.random.RandomState(5)
    records = [_record(rng, 40, 5) for _ in range(9)]
    from stage2_cli import _prefix, _sweep_fast
    prefixes = [_prefix(record, record['scores']) for record in records]
    thresholds = [0., .2, .5, .8, .95, .999]
    swept = _sweep_fast(prefixes, thresholds)
    direct = s2.sweep(records, 'scores', thresholds)
    assert swept['threshold'] == direct['threshold']
    assert (swept['tp'], swept['fp'], swept['fn']) == (direct['tp'], direct['fp'], direct['fn'])


def test_prefix_sweep_is_monotone_in_true_positives():
    rng = np.random.RandomState(9)
    records = [_record(rng, 30, 4) for _ in range(5)]
    from stage2_cli import _prefix, _sweep_fast
    prefixes = [_prefix(record, record['scores']) for record in records]
    thresholds = np.linspace(.99, .01, 60)
    rows = [_sweep_fast(prefixes, [t]) for t in thresholds]
    tp = [row['tp'] for row in rows]
    assert tp == sorted(tp)


def test_crop_and_scalars_are_scale_consistent():
    from PIL import Image
    image = Image.new('RGB', (400, 300), (10, 200, 30))
    tight, context = s2.pair_tensor(image, np.array([100., 80., 180., 160.]))
    assert tight.shape == (3, s2.CROP, s2.CROP) and context.shape == tight.shape
    assert torch.isfinite(tight).all() and torch.isfinite(context).all()
    scalars = s2.box_scalars(np.array([100., 80., 180., 160.]), 400., 300.)
    assert len(scalars) == s2.SCALARS
    # A box scaled with the image must produce identical scalars and crops.
    scaled = s2.box_scalars(np.array([50., 40., 90., 80.]), 200., 150.)
    assert scalars == pytest.approx(scaled, abs=1e-6)


def test_verifier_uses_both_crops_and_the_scalars():
    torch.manual_seed(3)
    model = s2.CropVerifier(pretrained=None)
    tight, context = torch.randn(2, 3, s2.CROP, s2.CROP), torch.randn(2, 3, s2.CROP, s2.CROP)
    scalars = torch.zeros(2, s2.SCALARS)
    base = model(tight, context, scalars)
    assert base.shape == (2,)
    changed = model(tight.clone(), torch.randn(2, 3, s2.CROP, s2.CROP), scalars.clone())
    assert not torch.allclose(base, changed, atol=1e-4)
    moved = model(tight.clone(), context.clone(), scalars + 1.)
    assert not torch.allclose(base, moved, atol=1e-4)


def test_focal_bce_ignores_nothing_and_handles_weights():
    logits = torch.tensor([2., -2., 0.])
    labels = torch.tensor([1., 0., 1.])
    plain = s2.focal_bce(logits, labels, torch.ones(3))
    heavy = s2.focal_bce(logits, labels, torch.tensor([3., 1., 1.]))
    assert bool(torch.isfinite(plain)) and float(heavy) > 0.
    assert float(heavy) != pytest.approx(float(plain))
