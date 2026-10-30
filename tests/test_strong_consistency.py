"""Geometry/gradient tests of the actual C1 helper, without models or GPU."""
from copy import deepcopy

import pytest
import torch

from apple_strong.consistency import gt_anchored_consistency, horizontal_flip_boxes


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def view(ids=("a",), *, boxes=None, size=8, requires_grad=True, value=None):
    n = len(ids)
    boxes = torch.tensor([[20., 10., 80., 90.]] * n) if boxes is None else torch.tensor(boxes, dtype=torch.float32)
    boxes = boxes.reshape(n, 4).requires_grad_(requires_grad)
    if value is None:
        values = torch.linspace(-2, 2, size * size).reshape(1, size, size).repeat(n, 1, 1)
    else:
        values = torch.full((n, size, size), value)
    return {"instance_ids": list(ids), "boxes": boxes,
            "roi_boxes": boxes.detach().clone().requires_grad_(requires_grad),
            "visible_logits": values.clone().requires_grad_(requires_grad),
            "amodal_logits": (values + .2).clone().requires_grad_(requires_grad)}


def flipped(v, width=100.):
    return {"instance_ids": list(v["instance_ids"]),
            "boxes": horizontal_flip_boxes(v["boxes"].detach(), width).requires_grad_(True),
            "roi_boxes": horizontal_flip_boxes(v["roi_boxes"].detach(), width).requires_grad_(True),
            "visible_logits": v["visible_logits"].detach().flip(-1).requires_grad_(True),
            "amodal_logits": v["amodal_logits"].detach().flip(-1).requires_grad_(True)}


def total(losses):
    assert set(losses) == {"loss_consistency_box", "loss_consistency_visible", "loss_consistency_amodal"}
    assert all(v.ndim == 0 and v.dtype == torch.float32 and torch.isfinite(v) for v in losses.values())
    return sum(losses.values())


def test_explicit_box_flip_is_involution_at_image_edges_and_preserves_gradient():
    boxes = torch.tensor([[0., 0., 100., 100.], [0., 4., 1., 7.], [99., 2., 100., 9.]], requires_grad=True)
    mirrored = horizontal_flip_boxes(boxes, 100.)
    assert torch.equal(mirrored[1], torch.tensor([99., 4., 100., 7.]))
    assert torch.equal(horizontal_flip_boxes(mirrored, 100.), boxes)
    mirrored.sum().backward()
    assert torch.equal(boxes.grad, torch.tensor([[-1., 1., -1., 1.]] * 3))


def test_correctly_flip_back_identical_predictions_have_zero_loss():
    reference = view()
    moving = flipped(reference)
    result = gt_anchored_consistency(moving, reference, {"a": [20, 10, 80, 90]}, image_hw=(100, 100))
    assert total(result).item() < 1e-12
    total(result).backward()
    assert all(torch.isfinite(moving[k].grad).all() for k in ("boxes", "roi_boxes", "visible_logits", "amodal_logits"))
    assert all(reference[k].grad is None for k in ("boxes", "roi_boxes", "visible_logits", "amodal_logits"))


def test_missing_flip_back_is_detected_by_asymmetric_geometry():
    reference = view(boxes=[[5., 15., 40., 70.]])
    moving = flipped(reference)
    gt = {"a": [5, 15, 40, 70]}
    correct = total(gt_anchored_consistency(moving, reference, gt, image_hw=(100, 100)))
    wrong = total(gt_anchored_consistency(moving, reference, gt, image_hw=(100, 100), moving_flipped=False))
    assert correct.item() < 1e-12
    assert wrong.item() > .05


def test_nontrivial_loss_has_finite_moving_gradients_without_reference_edges():
    reference = view(value=-.7)
    moving = view(boxes=[[22, 13, 82, 94]], value=.8)
    result = gt_anchored_consistency(moving, reference, {"a": [20, 10, 80, 90]}, image_hw=(100, 100), moving_flipped=False)
    assert all(v.item() > 0 for v in result.values())
    total(result).backward()
    for key in ("boxes", "visible_logits", "amodal_logits"):
        assert torch.isfinite(moving[key].grad).all() and moving[key].grad.abs().sum() > 0
        assert reference[key].grad is None
    assert reference["roi_boxes"].grad is None


def test_same_raw_roi_cells_on_shifted_supports_are_not_same_image_mask():
    reference = view(boxes=[[10, 10, 70, 90]])
    moving = view(boxes=[[10, 10, 70, 90]])
    moving["roi_boxes"] = torch.tensor([[30., 10., 90., 90.]], requires_grad=True)
    gt = {"a": [10, 10, 90, 90]}
    result = gt_anchored_consistency(moving, reference, gt, image_hw=(100, 100), moving_flipped=False)
    assert result["loss_consistency_box"].item() == 0
    assert result["loss_consistency_visible"].item() > .001
    total(result).backward()
    assert moving["roi_boxes"].grad.abs().sum() > 0


def test_different_roi_supports_render_same_analytic_image_field():
    def analytic(support):
        # Probabilities vary linearly with image x; bilinear reconstruction is
        # exact away from ROI edges, even though local cells differ.
        size = 64
        box = torch.tensor([support], dtype=torch.float32)
        x = support[0] + (torch.arange(size) + .5) / size * (support[2] - support[0])
        probability = (.1 + .7 * x / 100)[None, None, :].expand(1, size, -1)
        logits = torch.logit(probability).clone().requires_grad_(True)
        return {"instance_ids": ["a"], "boxes": torch.tensor([[30., 30., 60., 60.]], requires_grad=True),
                "roi_boxes": box.requires_grad_(True), "visible_logits": logits,
                "amodal_logits": logits.clone()}
    moving, reference = analytic([0, 0, 100, 100]), analytic([10, 10, 80, 80])
    result = gt_anchored_consistency(moving, reference, {"a": [30, 30, 60, 60]}, image_hw=(100, 100), moving_flipped=False)
    assert result["loss_consistency_visible"].item() < 1e-12
    assert result["loss_consistency_amodal"].item() < 1e-12


def test_disjoint_roi_supports_zero_only_mask_terms():
    reference = view(boxes=[[0, 0, 30, 100]])
    moving = view(boxes=[[70, 0, 100, 100]])
    result = gt_anchored_consistency(moving, reference, {"a": [0, 0, 100, 100]}, image_hw=(100, 100), moving_flipped=False)
    assert result["loss_consistency_box"].item() > 0
    assert result["loss_consistency_visible"].item() == 0
    assert result["loss_consistency_amodal"].item() == 0
    total(result).backward()
    assert torch.isfinite(moving["visible_logits"].grad).all()


@pytest.mark.parametrize("empty_side", ["moving", "reference", "both"])
def test_empty_or_missing_pairs_return_moving_connected_zero(empty_side):
    moving = view(ids=() if empty_side in ("moving", "both") else ("a",))
    reference = view(ids=() if empty_side in ("reference", "both") else ("a",))
    result = gt_anchored_consistency(moving, reference, {"a": [20, 10, 80, 90]}, image_hw=(100, 100))
    assert total(result).item() == 0
    total(result).backward()
    assert moving["visible_logits"].grad is not None
    assert reference["visible_logits"].grad is None


def test_id_alignment_uses_gt_identity_not_input_order():
    reference = view(ids=("b", "a"), boxes=[[60, 10, 90, 50], [10, 10, 40, 50]], value=.3)
    moving = view(ids=("a", "b"), boxes=[[10, 10, 40, 50], [60, 10, 90, 50]], value=.3)
    result = gt_anchored_consistency(moving, reference, {"a": [10, 10, 40, 50], "b": [60, 10, 90, 50]}, image_hw=(100, 100), moving_flipped=False)
    assert total(result).item() < 1e-12


@pytest.mark.parametrize("bad", ["unknown", "duplicate"])
def test_unknown_or_duplicate_gt_identity_rejected(bad):
    moving = view(ids=("x",) if bad == "unknown" else ("a", "a"))
    with pytest.raises(ValueError):
        gt_anchored_consistency(moving, view(), {"a": [20, 10, 80, 90]}, image_hw=(100, 100))


def test_pixel_and_normalized_coordinates_produce_identical_result():
    reference = view(value=-1.)
    moving = view(boxes=[[22, 10, 82, 90]], value=1.)
    pixel = gt_anchored_consistency(moving, reference, {"a": [20, 10, 80, 90]}, image_hw=(100, 100), moving_flipped=False)
    norm_moving, norm_reference = deepcopy(moving), deepcopy(reference)
    for v in (norm_moving, norm_reference):
        v["boxes"] = v["boxes"] / 100
        v["roi_boxes"] = v["roi_boxes"] / 100
    normalized = gt_anchored_consistency(norm_moving, norm_reference, {"a": [.2, .1, .8, .9]}, image_hw=(100, 100), moving_flipped=False, coordinate_space="normalized")
    assert all(torch.allclose(pixel[k], normalized[k], atol=1e-8, rtol=0) for k in pixel)


def test_invalid_extent_and_nonfinite_logits_rejected_without_repair():
    moving = view()
    moving["visible_logits"] = torch.full_like(moving["visible_logits"], float("nan"))
    with pytest.raises(ValueError):
        gt_anchored_consistency(moving, view(), {"a": [20, 10, 80, 90]}, image_hw=(100, 100))
    with pytest.raises(ValueError):
        gt_anchored_consistency(view(), view(), {"a": [20, 10, 130, 90]}, image_hw=(100, 100))


def test_outer_autocast_does_not_change_loss_math_dtype():
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = gt_anchored_consistency(view(value=.7), view(value=-.4), {"a": [20, 10, 80, 90]}, image_hw=(100, 100), moving_flipped=False)
    assert total(result).dtype == torch.float32
    total(result).backward()
