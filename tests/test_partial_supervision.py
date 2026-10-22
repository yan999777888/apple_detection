"""CPU-only behavioral checks for real partial and paired mask supervision.

No model, checkpoint, image or CUDA runtime is needed; all inputs are synthetic
tensors that exercise the actual Hungarian, sampling and autograd code.
"""
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from apple_seg.losses import AnnotationAwareHungarianCriterion, sample_cpu_target_points


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def criterion(**overrides):
    return AnnotationAwareHungarianCriterion(match_points=128, loss_points=96,
                                             target_sample_chunk_size=1, **overrides)


def generator(seed=17):
    return torch.Generator(device="cpu").manual_seed(seed)


def outputs(seed=3, height=8, width=8):
    rng = generator(seed)
    return {"class_queries_logits": torch.randn((1, 100, 2), generator=rng, requires_grad=True),
            "visible_masks_queries_logits": torch.randn((1, 100, height, width), generator=rng, requires_grad=True),
            "amodal_masks_queries_logits": torch.randn((1, 100, height, width), generator=rng, requires_grad=True)}


def target(visible, amodal=None, *, dataset="MinneApple"):
    return {"labels": torch.zeros(len(visible), dtype=torch.int64),
            "visible_masks": visible, "amodal_masks": amodal,
            "annotation_validity": {"visible": True, "amodal": dataset == "PApple"},
            "source": {"dataset": dataset, "split": "train", "image_id": "synthetic_001"},
            "instance_ids": [str(index) for index in range(len(visible))],
            "original_hw": list(visible.shape[-2:])}


def foreground_masks():
    visible = torch.zeros((2, 11, 13), dtype=torch.uint8)
    visible[0, 1:7, 1:6] = 1
    visible[1, 5:10, 7:12] = 1
    return visible


def snapshot_target(value):
    return {key: tensor.clone() if isinstance(tensor, torch.Tensor) else deepcopy(tensor)
            for key, tensor in value.items()}


def assert_target_unchanged(value, saved):
    for key, expected in saved.items():
        if isinstance(expected, torch.Tensor):
            assert value[key].dtype == expected.dtype
            assert value[key].device == expected.device
            assert torch.equal(value[key], expected)
        else:
            assert value[key] == expected


def test_visible_only_never_supervises_amodal_even_with_optimizer_weight_decay():
    truth = target(foreground_masks())
    saved = snapshot_target(truth)
    prediction = outputs()
    parameters = [torch.nn.Parameter(value.detach().clone()) for value in prediction.values()]
    prediction = dict(zip(prediction, parameters))
    original_a = prediction["amodal_masks_queries_logits"].detach().clone()
    loss_fn = criterion()
    losses = loss_fn(prediction, [truth], generator=generator())
    assert set(losses) == {"loss_class", "loss_visible_bce", "loss_visible_dice"}
    total = sum(losses.values())
    total.backward()
    assert prediction["amodal_masks_queries_logits"].grad is None
    for key in ("class_queries_logits", "visible_masks_queries_logits"):
        assert prediction[key].grad is not None
        assert torch.isfinite(prediction[key].grad).all()
        assert prediction[key].grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0.2)
    optimizer.step()
    assert torch.equal(original_a, prediction["amodal_masks_queries_logits"].detach())
    assert_target_unchanged(truth, saved)


def test_unknown_amodal_values_cannot_change_assignment_or_loss():
    truth = target(foreground_masks())
    first, second = outputs(), outputs()
    second["amodal_masks_queries_logits"] = torch.full_like(
        first["amodal_masks_queries_logits"], 100, requires_grad=True)
    left, right = criterion(), criterion()
    loss_a = left(first, [truth], generator=generator(41))
    loss_b = right(second, [truth], generator=generator(41))
    assert left.last_assignment == right.last_assignment
    assert all(torch.equal(loss_a[key], loss_b[key]) for key in loss_a)


def test_visible_only_receives_full_weight_and_paired_visible_receives_half():
    visible = foreground_masks()
    prediction = outputs()
    prediction["amodal_masks_queries_logits"] = prediction["visible_masks_queries_logits"].detach().clone().requires_grad_()
    partial_fn, paired_fn = criterion(), criterion()
    partial = partial_fn(prediction, [target(visible)], generator=generator(27))
    paired = paired_fn(prediction, [target(visible, visible.clone(), dataset="PApple")], generator=generator(27))
    assert partial_fn.last_assignment == paired_fn.last_assignment
    torch.testing.assert_close(partial["loss_class"], paired["loss_class"], rtol=0, atol=0)
    for key in ("loss_visible_bce", "loss_visible_dice"):
        torch.testing.assert_close(partial[key], 2 * paired[key], rtol=0, atol=0)


def test_changing_real_visible_gt_changes_visible_and_class_gradients():
    # Queries 0 and 1 represent opposite sides. The real GT side must decide
    # which class query is positive and which spatial mask gets supervision.
    template = torch.full((1, 100, 8, 8), -7.0)
    template[0, 0, :, :4] = 7
    template[0, 1, :, 4:] = 7
    gradients, assignments = [], []
    for side in (slice(0, 4), slice(4, 8)):
        mask = torch.zeros((1, 8, 8), dtype=torch.uint8)
        mask[0, :, side] = 1
        prediction = {"class_queries_logits": torch.zeros((1, 100, 2), requires_grad=True),
                      "visible_masks_queries_logits": template.clone().requires_grad_(),
                      "amodal_masks_queries_logits": torch.randn((1, 100, 8, 8), requires_grad=True)}
        fn = criterion()
        sum(fn(prediction, [target(mask)], generator=generator(13)).values()).backward()
        assignments.append(fn.last_assignment["query_indices"])
        gradients.append((prediction["class_queries_logits"].grad.clone(),
                          prediction["visible_masks_queries_logits"].grad.clone()))
        assert prediction["amodal_masks_queries_logits"].grad is None
    assert assignments == [[0], [1]]
    assert gradients[0][0][0, 0, 0] < 0 < gradients[1][0][0, 0, 0]
    assert gradients[1][0][0, 1, 0] < 0 < gradients[0][0][0, 1, 0]
    assert not torch.equal(gradients[0][1], gradients[1][1])


def test_real_paired_gt_supervises_both_heads_with_one_shared_assignment():
    visible = foreground_masks()
    # Deliberate real-schema containment anomaly: V is not a subset of A.
    amodal = visible.flip(-1).clone()
    assert bool((visible.bool() & ~amodal.bool()).any())
    truth = target(visible, amodal, dataset="PApple")
    saved = snapshot_target(truth)
    prediction, fn = outputs(), criterion()
    losses = fn(prediction, [truth], generator=generator())
    assert set(losses) == {"loss_class", "loss_visible_bce", "loss_visible_dice",
                           "loss_amodal_bce", "loss_amodal_dice"}
    sum(losses.values()).backward()
    selected = set(fn.last_assignment["query_indices"])
    assert len(selected) == 2 and set(fn.last_assignment["target_indices"]) == {0, 1}
    for key in ("visible_masks_queries_logits", "amodal_masks_queries_logits"):
        grad = prediction[key].grad
        assert grad is not None and torch.isfinite(grad).all()
        actual = set(torch.nonzero(grad.abs().flatten(2).sum(-1)[0] > 0).flatten().tolist())
        assert actual == selected
    assert prediction["class_queries_logits"].grad.abs().sum() > 0
    assert_target_unchanged(truth, saved)


@pytest.mark.parametrize("dataset", ["MinneApple", "PApple"])
def test_empty_gt_keeps_classification_and_known_mask_losses_zero(dataset):
    masks = torch.zeros((0, 9, 7), dtype=torch.uint8)
    truth = target(masks, masks.clone() if dataset == "PApple" else None, dataset=dataset)
    saved = snapshot_target(truth)
    prediction, fn = outputs(), criterion()
    losses = fn(prediction, [truth], generator=generator())
    assert losses["loss_class"] > 0
    assert all(value.item() == 0 for key, value in losses.items() if key != "loss_class")
    assert fn.last_assignment == {"query_indices": [], "target_indices": []}
    sum(losses.values()).backward()
    assert prediction["class_queries_logits"].grad.abs().sum() > 0
    assert torch.count_nonzero(prediction["visible_masks_queries_logits"].grad) == 0
    amodal_grad = prediction["amodal_masks_queries_logits"].grad
    assert amodal_grad is None if dataset == "MinneApple" else torch.count_nonzero(amodal_grad) == 0
    assert_target_unchanged(truth, saved)


def test_capacity_rejects_101_gt_without_changing_inventory():
    truth = target(torch.ones((101, 2, 2), dtype=torch.uint8))
    saved = snapshot_target(truth)
    with pytest.raises(ValueError, match="100-query capacity"):
        criterion()(outputs(), [truth], generator=generator())
    assert_target_unchanged(truth, saved)


def test_unknown_amodal_rejects_fabricated_zeros_and_known_amodal_requires_annotation():
    visible = foreground_masks()
    missing = target(visible, torch.zeros_like(visible))
    with pytest.raises(ValueError, match="Unknown amodal annotation must be None"):
        criterion()(outputs(), [missing])
    missing = target(visible, dataset="PApple")
    with pytest.raises(TypeError, match="Known GT masks"):
        criterion()(outputs(), [missing])


def test_chunked_cpu_sampling_matches_grid_sample_oracle_and_preserves_original():
    rng = generator(93)
    masks = torch.randint(0, 2, (3, 39, 47), dtype=torch.uint8, generator=rng)
    saved = masks.clone()
    indices = torch.tensor([2, 0, 1, 2], dtype=torch.int64)
    coordinates = torch.rand((4, 75, 2), generator=rng)
    coordinates[:, :4] = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]])
    expected = F.grid_sample(masks[indices, None].float(), coordinates[:, None] * 2 - 1,
                             align_corners=False, mode="bilinear", padding_mode="zeros")[:, 0, 0]
    actual = sample_cpu_target_points(masks, indices, coordinates, device="cpu", chunk_size=1)
    assert actual.shape == (4, 75)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
    assert torch.equal(masks, saved) and masks.dtype == torch.uint8
    assert masks.device.type == "cpu"


def test_extreme_finite_logits_are_stable_and_repeated_seed_is_deterministic():
    prediction = outputs()
    with torch.no_grad():
        prediction["visible_masks_queries_logits"].mul_(100)
        prediction["amodal_masks_queries_logits"].mul_(100)
        prediction["class_queries_logits"].mul_(100)
    truth = target(foreground_masks(), foreground_masks().clone(), dataset="PApple")
    fn = criterion()
    first = fn(prediction, [truth], generator=generator(911))
    first_assignment = deepcopy(fn.last_assignment)
    second = fn(prediction, [truth], generator=generator(911))
    assert fn.last_assignment == first_assignment
    assert all(torch.equal(first[key], second[key]) and torch.isfinite(first[key]) for key in first)
    sum(first.values()).backward()
    assert all(torch.isfinite(tensor.grad).all() for tensor in prediction.values())


def test_defaults_preserve_requested_sampling_and_loss_weights():
    fn = AnnotationAwareHungarianCriterion()
    cfg = fn.configuration
    assert (cfg["match_points"], cfg["loss_points"], cfg["oversample_ratio"], cfg["importance_ratio"]) == (12544, 12544, 3.0, 0.75)
    assert (cfg["class_weight"], cfg["no_object_weight"], cfg["mask_weight"], cfg["dice_weight"]) == (2.0, 0.1, 5.0, 5.0)
    prediction = outputs(height=4, width=4)
    losses = fn(prediction, [target(foreground_masks()[:1])], generator=generator(105))
    sum(losses.values()).backward()
    assert prediction["visible_masks_queries_logits"].grad.abs().sum() > 0
    assert prediction["amodal_masks_queries_logits"].grad is None
