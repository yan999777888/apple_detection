"""Small CPU semantic checks against the actual frozen NMS/fusion functions.

No checkpoint, detector construction, GPU, modified NMS, or approximate mask
IoU is used. The score cutoff invariant is restricted to single-view final
scores; the last test demonstrates why it does not extend to box fusion.
"""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "strong_runtime"


def load_frozen(name, relative, expected_sha):
    path = FROZEN / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


roi = load_frozen("fast_test_frozen_od_roi", "src/apple_seg/od_roi.py",
    "401afe64a12f539830ad1d3f7d1b6741032dc8b9d21034105180616632621a27")
sys.path.insert(0, str(FROZEN / "src"))
fusion = load_frozen("fast_test_frozen_flip_tta", "tmp/flip_tta.py",
    "db2ad2578b0b1aa900e0f44c048349caf6f5110dd716d90a4e6aa235f734c131")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def case(name):
    """Named physical mask arrangements with explicit survivor expectations."""
    size = 8
    left = torch.zeros(size, size)
    left[:, :4] = 1
    right = 1 - left
    full = torch.ones(size, size)
    empty = torch.zeros(size, size)
    if name == "A_overlap_V_separate":
        # A overlap alone must not suppress independently visible fragments.
        amodal, visible = [full, full, full, full], [left, right, left, empty]
        scores, expected = [.90, .80, .74, .75], [0, 1, 3]
    elif name == "V_overlap_A_separate":
        # Likewise V overlap alone is insufficient for dual-mask suppression.
        amodal, visible = [left, right, left], [full, full, full]
        scores, expected = [.91, .85, .72], [0, 1]
    elif name == "stable_ties_at_threshold":
        # At exactly .75, stable input order chooses index 0 over duplicate 1.
        amodal, visible = [full, full, empty, full], [full, full, empty, full]
        scores, expected = [.75, .75, .75, .749999], [0, 2]
    elif name == "all_empty_masks":
        # Empty/empty IoU is zero; empty masks are never mutually suppressed.
        amodal, visible = [empty] * 5, [empty] * 5
        scores, expected = [.2, .75, .99, .74, .8], [2, 4, 1, 3, 0]
    elif name == "unsorted_duplicate_scores":
        amodal, visible = [full, full, empty, full], [full, full, empty, full]
        scores, expected = [.76, .95, .8, .1], [1, 2]
    elif name == "all_scores_below_threshold":
        amodal, visible = [full, right, empty], [full, right, empty]
        # A IoU is exactly .5, so the strict >.5 suppression condition fails.
        scores, expected = [.74, .2, .001], [0, 1, 2]
    elif name == "no_candidates":
        return torch.empty(0, size, size), torch.empty(0, size, size), torch.empty(0), []
    else:
        raise AssertionError(name)
    return torch.stack(amodal), torch.stack(visible), torch.tensor(scores), expected


CASES = ["A_overlap_V_separate", "V_overlap_A_separate", "stable_ties_at_threshold",
         "all_empty_masks", "unsorted_duplicate_scores", "all_scores_below_threshold", "no_candidates"]


@pytest.mark.parametrize("name", CASES)
def test_actual_frozen_dual_mask_semantics_have_known_survivors(name):
    amodal, visible, scores, expected = case(name)
    boxes = torch.tensor([[0., 0., 32., 32.]] * len(scores)).reshape(-1, 4)
    selected = roi.dual_mask_nms(amodal, visible, scores, boxes=boxes,
                                 image_shape=(32, 32), paste_chunk_pixels=256)
    assert selected.tolist() == expected


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("with_native_paste", [False, True])
@pytest.mark.parametrize("cap", [1, 2, 300])
def test_final_score_prefilter_preserves_original_above_threshold_after_cap(name, with_native_paste, cap):
    amodal, visible, scores, _ = case(name)
    boxes = torch.tensor([[0., 0., 32., 32.]] * len(scores)).reshape(-1, 4)
    kwargs = {"paste_chunk_pixels": 256, "mask_chunk_size": 2}
    if with_native_paste:
        kwargs.update(boxes=boxes, image_shape=(32, 32))
    original = roi.dual_mask_nms(amodal, visible, scores, **kwargs)[:cap]
    expected = original[scores[original] >= .75]
    keep = torch.where(scores >= .75)[0]
    if with_native_paste:
        kwargs["boxes"] = boxes[keep]
    actual = keep[roi.dual_mask_nms(amodal[keep], visible[keep], scores[keep], **kwargs)[:cap]]
    assert torch.equal(actual, expected)
    assert actual.device.type == "cpu"


def test_original_stable_pretopk_and_global_index_mapping_are_retained():
    amodal, visible, scores, _ = case("all_empty_masks")
    # Simulate native ROI forward's score>score_thresh and stable pre-NMS top-k.
    valid = torch.where(scores > .001)[0]
    prekeep = valid[torch.argsort(scores[valid], descending=True, stable=True)[:4]]
    a, v, s = amodal[prekeep], visible[prekeep], scores[prekeep]
    boxes = torch.tensor([[1., 1., 31., 31.]] * len(s))
    kwargs = dict(boxes=boxes, image_shape=(32, 32), paste_chunk_pixels=256)
    original_local = roi.dual_mask_nms(a, v, s, **kwargs)[:3]
    original_global = prekeep[original_local[s[original_local] >= .75]]
    after = torch.where(s >= .75)[0]
    filtered_local = roi.dual_mask_nms(a[after], v[after], s[after],
        boxes=boxes[after], image_shape=(32, 32), paste_chunk_pixels=256)[:3]
    restored_global = prekeep[after[filtered_local]]
    assert restored_global.tolist() == original_global.tolist() == [2, 4, 1]


def test_native_paste_with_different_roi_supports_retains_threshold_invariant():
    # Similar ROI-local masks can represent different native pixels. Exercise
    # the true tiled paste path rather than treating local grids as aligned.
    masks = torch.ones(4, 8, 8)
    boxes = torch.tensor([[0., 0., 24., 24.], [32., 32., 56., 56.],
                          [1., 1., 25., 25.], [8., 8., 32., 32.]])
    scores = torch.tensor([.8, .9, .74, .75])
    original = roi.dual_mask_nms(masks, masks, scores, boxes=boxes,
        image_shape=(64, 64), paste_chunk_pixels=256)
    keep = torch.where(scores >= .75)[0]
    filtered = keep[roi.dual_mask_nms(masks[keep], masks[keep], scores[keep],
        boxes=boxes[keep], image_shape=(64, 64), paste_chunk_pixels=256)]
    assert torch.equal(filtered, original[scores[original] >= .75])
    assert 0 in original.tolist() and 1 in original.tolist()


def test_low_score_fusion_member_changes_high_score_cluster_box_score_and_decision():
    # This is a constructed counterexample, not an observation on user data.
    # The .10 Small member overlaps a cluster formed by three strong members.
    boxes = [[0., 0., 20., 20.], [0., 0., 20., 20.],
             [1., 0., 21., 20.], [0., 1., 20., 21.]]
    scores = [.85, .84, .10, .80]
    entries = [{"pred": {"boxes": [box], "scores": [score]}}
               for box, score in zip(boxes, scores)]
    settings = dict(iou_threshold=.45, score_mode="mean", box_mode="score_weighted",
                    coverage_floor=1., view_weights=[.65, .65, .35, .35],
                    consensus_groups=[0, 0, 1, 1], consensus_bonus=1.10,
                    isolation_penalty=.90)
    metadata = []
    original_boxes, original_scores = fusion.fuse_entries(entries,
        cluster_metadata=metadata, **settings)
    assert metadata[0]["member_views"] == [0, 1, 3, 2]
    assert min(metadata[0]["member_scores"]) == .10 < .35
    filtered_entries = deepcopy(entries)
    filtered_entries[2]["pred"] = {"boxes": [], "scores": []}
    filtered_boxes, filtered_scores = fusion.fuse_entries(filtered_entries, **settings)
    assert len(original_boxes) == len(filtered_boxes) == 1
    assert not np.allclose(original_boxes, filtered_boxes, rtol=0, atol=1e-12)
    assert original_scores[0] < .75 < filtered_scores[0]
    assert original_scores[0] != filtered_scores[0]
