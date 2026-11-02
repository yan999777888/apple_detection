"""CPU semantic tests of the pre-mask adapter using the real frozen forward.

Heavy learned operations are tiny deterministic stubs. The topology guard is
explicitly bypassed only for these fixtures; checkpoint/model tests own it.
No CUDA context, dataset or checkpoint is involved.
"""
from collections import OrderedDict
from pathlib import Path
import sys

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "strong_runtime/src")]
from apple_seg.od_model import ODRoIHeads
from deployment.pre_mask import PreMaskDeploymentAdapter
from deployment.strong_fast import FastDeploymentAdapter


class TinyPool(nn.Module):
    def forward(self, features, proposals, shapes):
        return torch.cat(proposals)


class TinyPredictor(nn.Module):
    def __init__(self, initial):
        super().__init__()
        p = torch.as_tensor(initial, dtype=torch.float32)
        self.register_buffer("logits", torch.stack((torch.zeros_like(p), torch.logit(p)), 1))

    def forward(self, features):
        identifiers = (features[:, 0] / 10).long()
        return self.logits[identifiers], torch.zeros((len(features), 8))


class TinyHead(nn.Module):
    # Reuse the actual frozen implementation instead of mirroring its logic.
    forward = ODRoIHeads.forward

    def __init__(self, initial=(.10, .95, .40, .80, .60), final=(.95, .20, .90, .85, .75),
                 valid=(0, 2, 3, 4)):
        super().__init__()
        self.box_roi_pool = TinyPool()
        self.mask_roi_pool = TinyPool()
        self.box_head = nn.Identity()
        self.box_predictor = TinyPredictor(initial)
        self.register_buffer("final_probabilities", torch.tensor(final))
        self.valid_indices = torch.tensor(valid, dtype=torch.long)
        self.score_thresh = .15
        self.pre_mask_nms_topk = 3
        self.detections_per_img = 2
        self.roi_chunk_size = 8
        self.classifier_only = False
        self.mask_inputs = []

    def _decoded_boxes(self, regression, proposals, shape):
        return proposals[0][self.valid_indices].clone(), self.valid_indices.clone()

    def _roi_predictions(self, features, proposals, image_shapes, base_logits):
        boxes = proposals[0]
        identifiers = (boxes[:, 0] / 10).long()
        self.mask_inputs.append({"boxes": boxes.clone(), "base_logits": base_logits.clone()})
        probabilities = self.final_probabilities[identifiers]
        logits = torch.stack((torch.zeros_like(probabilities), torch.logit(probabilities)), 1)
        # Every ROI has an identifiable mask payload so index mistakes are visible.
        masks = identifiers.float()[:, None, None, None].expand(-1, 1, 2, 2).clone()
        return {"visible_logits": masks, "hidden_logits": masks + 1,
                "amodal_logits": masks + 2}, logits

    def _mask_predictions(self, pooled):
        raise AssertionError("Fixture uses its deterministic ROI predictor")

    def _final_nms(self, masks, boxes, scores, shape):
        # Deterministic sparse survivor indices exercise the complete mapping.
        return torch.arange(0, len(scores), 2, dtype=torch.long)


class TinyBackbone(nn.Module):
    def forward(self, images):
        return OrderedDict(p2=images[:, :1])


class TinyRPN(nn.Module):
    def __init__(self, proposals):
        super().__init__()
        self.register_buffer("proposals", proposals)

    def forward(self, images, features, targets):
        return [self.proposals.clone() for _ in images.image_sizes], {}


class TinyModel(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.roi_heads = head
        self.backbone = TinyBackbone()
        count = len(head.final_probabilities)
        self.proposals = torch.tensor([[i * 10., 0., i * 10. + 4., 4.] for i in range(count)]).reshape(-1, 4)
        self.rpn = TinyRPN(self.proposals)
        self.eval()


def make_adapter(model, *, cls=PreMaskDeploymentAdapter, **kwargs):
    adapter = cls(model, roi_chunk_size=64, roi_precision="autocast", **kwargs)
    adapter._validate_model = lambda: None  # Explicit tiny fixture, never production.
    return adapter


def assert_output_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        else:
            assert left[key] == right[key]


def call_head(model):
    return model.roi_heads({"p2": torch.zeros(1, 1, 64, 64)}, [model.proposals], [(64, 64)])[0][0]


def test_disabled_path_matches_actual_frozen_forward_and_parent_fast_path():
    model = TinyModel(TinyHead())
    expected = call_head(model)
    with make_adapter(model, cls=FastDeploymentAdapter):
        fast_expected = call_head(model)
    with make_adapter(model, capture_candidates=True) as adapter:
        actual = call_head(model)
        assert_output_equal(actual, expected)
        assert_output_equal(actual, fast_expected)
        trace = adapter.traces[0]
        assert trace["mask_indices"].tolist() == [0, 1, 2, 3]
        assert trace["output_indices"].tolist() == [0, 2]
        torch.testing.assert_close(trace["decoded_boxes"][trace["output_indices"]], actual["boxes"])
        assert model.roi_heads.roi_chunk_size == 64
    assert model.roi_heads.roi_chunk_size == 8
    assert "forward" not in model.roi_heads.__dict__
    assert "_roi_predictions" not in model.roi_heads.__dict__
    assert "forward" not in model.roi_heads.box_roi_pool.__dict__
    assert "_strong_fast_adapter_active" not in model.roi_heads.__dict__
    assert_output_equal(call_head(model), expected)


def test_valid_decoded_indices_then_topk_keep_logits_masks_and_trace_aligned():
    model = TinyModel(TinyHead())
    with make_adapter(model, candidate_policy={"topk": 2}, capture_candidates=True) as adapter:
        actual = call_head(model)
        trace = adapter.traces[0]
        # RPN id 1 has the highest score but decoding dropped it. The remaining
        # id 3/id 4 are decoded indices 2/3 and must retain their original logits.
        assert trace["mask_indices"].tolist() == [2, 3]
        assert trace["output_indices"].tolist() == [2]
        passed = model.roi_heads.mask_inputs[-1]
        assert passed["boxes"][:, 0].tolist() == [30., 40.]
        torch.testing.assert_close(passed["base_logits"], model.roi_heads.box_predictor.logits[[3, 4]])
        torch.testing.assert_close(actual["roi_visible_masks"], torch.full((1, 1, 2, 2), 3.).sigmoid())
        torch.testing.assert_close(trace["decoded_boxes"][trace["output_indices"]], actual["boxes"])
        assert adapter.last_stats["pre_mask"][0]["mask_candidates"] == 2


@pytest.mark.parametrize("kind", ["no_proposals", "no_valid_boxes", "all_scores_filtered"])
def test_empty_paths_return_well_shaped_outputs_and_empty_trace(kind):
    if kind == "no_proposals":
        head = TinyHead(initial=(), final=(), valid=())
    elif kind == "no_valid_boxes":
        head = TinyHead(valid=())
    else:
        head = TinyHead()
    model = TinyModel(head)
    chosen = {"score_threshold": 1.} if kind == "all_scores_filtered" else {}
    with make_adapter(model, candidate_policy=chosen, capture_candidates=True) as adapter:
        output = call_head(model)
        assert output["boxes"].shape == (0, 4)
        assert output["scores"].shape == (0,)
        assert output["roi_visible_masks"].shape == (0, 1, 2, 2)
        assert adapter.traces[0]["output_indices"].numel() == 0
        assert adapter.traces[0]["mask_indices"].numel() == 0


@pytest.mark.parametrize("bad", ["targets", "training", "two_roi_views"])
def test_eval_only_contract_rejects_gt_training_and_multiple_roi_views(bad):
    model = TinyModel(TinyHead())
    with make_adapter(model):
        if bad == "training":
            model.roi_heads.train()
        with pytest.raises(ValueError, match="eval-only"):
            model.roi_heads({"p2": torch.zeros(1, 1, 64, 64)},
                            [model.proposals] * (2 if bad == "two_roi_views" else 1),
                            [(64, 64)], [] if bad == "targets" else None)
    assert "forward" not in model.roi_heads.__dict__


def test_exception_restores_all_methods_and_existing_override_is_rejected():
    model = TinyModel(TinyHead())
    original = model.roi_heads.forward.__func__
    with pytest.raises(RuntimeError, match="inside"):
        with make_adapter(model):
            raise RuntimeError("inside")
    assert model.roi_heads.forward.__func__ is original
    assert model.roi_heads.roi_chunk_size == 8
    model.roi_heads.forward = lambda *a, **k: None
    with pytest.raises(ValueError, match="already overridden"):
        with make_adapter(model):
            pass
    assert "_strong_fast_adapter_active" not in model.roi_heads.__dict__


def test_predict_views_resets_trace_per_call_and_keeps_both_views_independent():
    model = TinyModel(TinyHead())
    with make_adapter(model, capture_candidates=True) as adapter:
        pair = adapter.predict_views(torch.zeros(2, 3, 64, 64))
        assert len(pair) == 2 and len(adapter.traces) == 2
        assert_output_equal(pair[0], pair[1])
        one = adapter.predict_views(torch.zeros(1, 3, 64, 64))
        assert len(one) == 1 and len(adapter.traces) == 1
        assert len(adapter.last_stats["pre_mask"]) == 1
