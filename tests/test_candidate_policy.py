"""CPU-only semantic checks for independent mask-before selection rules."""
from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment.candidate_policy import policy_description, select_candidates


def boxes(values):
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 4)


def scores(values):
    return torch.tensor(values, dtype=torch.float32)


def sample():
    # Rows 0 and 2 are duplicates; rows 1,3 are independent instances.
    return boxes([[0, 0, 10, 10], [20, 0, 30, 10], [0, 0, 10, 10], [40, 0, 50, 10]]), scores([.7, .9, .8, .8])


def test_disabled_keeps_original_order_and_inputs():
    b, s = sample()
    before_b, before_s = b.clone(), s.clone()
    selected = select_candidates(b, s)
    assert selected.tolist() == [0, 1, 2, 3]
    assert selected.dtype == torch.int64 and selected.device == b.device
    assert torch.equal(b, before_b) and torch.equal(s, before_s)


def test_topk_only_orders_scores_stably_without_nms():
    b, s = sample()
    assert select_candidates(b, s, topk=3).tolist() == [1, 2, 3]
    assert select_candidates(b, s, topk=10).tolist() == [1, 2, 3, 0]


def test_score_only_includes_float32_boundary_and_preserves_duplicates():
    cutoff = torch.tensor(.03, dtype=torch.float32)
    s = torch.stack((torch.nextafter(cutoff, torch.tensor(0.)), cutoff,
                     torch.nextafter(cutoff, torch.tensor(1.)), torch.tensor(.9)))
    b = boxes([[0, 0, 10, 10]] * 4)
    assert select_candidates(b, s, score_threshold=.03).tolist() == [3, 2, 1]
    assert policy_description(score_threshold=.03)["score_threshold"] == cutoff.item()


def test_nms_only_preserves_highest_scoring_duplicate_and_score_order():
    b, s = sample()
    assert select_candidates(b, s, box_nms_iou=.8).tolist() == [1, 2, 3]


def test_nms_same_score_tie_uses_original_index():
    b = boxes([[0, 0, 10, 10], [0, 0, 10, 10], [20, 0, 30, 10]])
    s = scores([.5, .5, .5])
    assert select_candidates(b, s, box_nms_iou=.8).tolist() == [0, 2]


def test_nms_ties_are_resolved_before_backend(monkeypatch):
    import deployment.candidate_policy as module
    seen = {}
    def capture(b, rank, iou):
        seen.update(boxes=b.clone(), rank=rank.clone(), iou=iou)
        return torch.arange(len(rank), dtype=torch.int64)
    monkeypatch.setattr(module, "nms", capture)
    b, s = sample()
    out = select_candidates(b.double(), s.double(), box_nms_iou=.8)
    assert out.tolist() == [1, 2, 3, 0]
    assert seen["boxes"].dtype == torch.float32 and seen["rank"].dtype == torch.float32
    assert seen["rank"].tolist() == [4., 3., 2., 1.]
    assert torch.equal(seen["boxes"], b[[1, 2, 3, 0]])


def test_combination_nms_precedes_topk_and_does_not_spend_budget_on_duplicates():
    b = boxes([[0, 0, 10, 10], [0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]])
    s = scores([.9, .8, .7, .02])
    assert select_candidates(b, s, score_threshold=.03, box_nms_iou=.8, topk=2).tolist() == [0, 2]


def test_nms_strict_iou_boundary_and_continuous_box_area():
    # Two 3x2 boxes shifted by one pixel have intersection=4, union=8, IoU=.5.
    b = boxes([[0, 0, 3, 2], [1, 0, 4, 2]])
    s = scores([.8, .7])
    assert select_candidates(b, s, box_nms_iou=.5).tolist() == [0, 1]
    assert select_candidates(b, s, box_nms_iou=.49).tolist() == [0]


def test_touching_boxes_not_suppressed_at_zero_iou_threshold():
    b = boxes([[0, 0, 2, 2], [2, 0, 4, 2]])
    assert select_candidates(b, scores([.8, .7]), box_nms_iou=0.).tolist() == [0, 1]


def test_identical_boxes_not_suppressed_at_one_iou_threshold():
    b = boxes([[0, 0, 2, 2]] * 2)
    assert select_candidates(b, scores([.8, .7]), box_nms_iou=1.).tolist() == [0, 1]


@pytest.mark.parametrize("options", [{}, {"score_threshold": .03}, {"topk": 64},
                                      {"box_nms_iou": .8},
                                      {"score_threshold": .03, "box_nms_iou": .8, "topk": 64}])
def test_empty_candidates_are_legal(options):
    out = select_candidates(boxes([]), scores([]), **options)
    assert out.shape == (0,) and out.dtype == torch.int64 and out.device.type == "cpu"


def test_all_filtered_is_legal():
    b, s = sample()
    assert select_candidates(b, s, score_threshold=1., box_nms_iou=.8, topk=64).numel() == 0


def test_zero_area_boxes_legal_without_automatic_deletion():
    b = boxes([[0, 0, 0, 0], [0, 0, 0, 0]])
    assert select_candidates(b, scores([.3, .2]), box_nms_iou=.8).tolist() == [0, 1]


@pytest.mark.parametrize("key,value", [
    ("score_threshold", -.01), ("score_threshold", 1.01), ("score_threshold", float("nan")),
    ("score_threshold", True), ("score_threshold", ".03"),
    ("box_nms_iou", -.1), ("box_nms_iou", 1.1), ("box_nms_iou", float("inf")),
    ("box_nms_iou", False), ("topk", 0), ("topk", -1), ("topk", 1.5), ("topk", True),
])
def test_invalid_options_rejected(key, value):
    with pytest.raises(ValueError):
        select_candidates(boxes([]), scores([]), **{key: value})


@pytest.mark.parametrize("bad_boxes,bad_scores", [
    (torch.zeros(4), scores([.5])), (torch.zeros(1, 5), scores([.5])),
    (boxes([[0, 0, 1, 1]]), torch.zeros(1, 1)),
    (boxes([[0, 0, 1, 1]]), scores([])),
    (torch.zeros((1, 4), dtype=torch.int64), scores([.5])),
    (boxes([[0, 0, 1, 1]]), torch.ones(1, dtype=torch.int64)),
    (boxes([[float("nan"), 0, 1, 1]]), scores([.5])),
    (boxes([[0, 0, 1, 1]]), scores([float("inf")])),
    (boxes([[0, 0, 1, 1]]), scores([1.1])),
    (boxes([[0, 0, 1, 1]]), scores([-.1])),
    (boxes([[1, 0, 0, 1]]), scores([.5])),
])
def test_invalid_tensor_inputs_rejected(bad_boxes, bad_scores):
    with pytest.raises(ValueError):
        select_candidates(bad_boxes, bad_scores, topk=64)


def test_no_hidden_numpy_or_gt_input_path():
    with pytest.raises(TypeError):
        select_candidates([[0., 0., 1., 1.]], scores([.5]))
    with pytest.raises(TypeError):
        select_candidates(boxes([[0, 0, 1, 1]]), scores([.5]), gt_boxes=[])
