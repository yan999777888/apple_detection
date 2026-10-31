"""CPU-only checks for deployment evaluation boundaries and flip geometry."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("fast_eval_under_test", ROOT / "scripts/evaluate_fast_deployment.py")
fast = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fast)
METRICS_SPEC = importlib.util.spec_from_file_location("frozen_fast_metrics", ROOT / "strong_runtime/reference/local_amodal/scripts/metrics.py")
metrics = importlib.util.module_from_spec(METRICS_SPEC)
METRICS_SPEC.loader.exec_module(metrics)


def test_fixed_cutoff_never_calibrates_a_truncated_pr_curve():
    class FixedOnlyMetrics:
        def evaluate_at_threshold(self, records, confidence, iou):
            assert confidence == .75 and iou == .5
            return {"precision": 1., "recall": .5}

        def calibrate_threshold(self, *args, **kwargs):
            raise AssertionError("Fixed-point-only data must not claim a complete PR curve")

    summary, curve = fast.summarize([], FixedOnlyMetrics(), fast.policy(.75))
    assert curve is None and summary["at_recall90"] is None
    assert summary["evaluation_policy"]["fixed_only"]


def test_frozen_calibration_reports_r90_and_fixed_point_separately():
    records = [{"image_id": "fruit", "gt": {"boxes": [[0, 0, 10, 10]]},
                "pred": {"boxes": [[0, 0, 10, 10]], "scores": [.7]}},
               {"image_id": "empty", "gt": {"boxes": []},
                "pred": {"boxes": [[0, 0, 10, 10]], "scores": [.8]}}]
    summary, curve = fast.summarize(records, metrics, fast.policy(.35))
    assert summary["at_recall90"]["confidence"] == .7
    assert (summary["at_recall90"]["tp"], summary["at_recall90"]["fp"]) == (1, 1)
    assert summary["at_confidence_0_75"]["tp"] == 0
    assert summary["at_confidence_0_75"]["false_positives_on_empty_gt_images"] == 1
    assert curve["candidate_thresholds"] == 2
    assert "truncated" in summary["evaluation_policy"]["curve_scope"]


def test_native_flip_inverts_support_and_roi_pixels_together():
    from apple_seg.od_evaluate import _validate_prediction
    pattern = torch.tensor([[[[0., .25], [.75, 1.]]]])
    raw = {"boxes": torch.tensor([[0., 1., 4., 5.]]), "scores": torch.tensor([.9]),
           "labels": torch.tensor([0]), "roi_visible_masks": pattern,
           "roi_amodal_masks": 1. - pattern, "image_hw": (8, 10)}
    result = fast.native_prediction(raw, (8, 10), (16, 20), True, _validate_prediction)
    assert result["boxes"].tolist() == [[12., 2., 20., 10.]]
    torch.testing.assert_close(result["visible"], pattern[:, 0].flip(-1))
    torch.testing.assert_close(result["amodal"], (1. - pattern[:, 0]).flip(-1))
    # Validate never changes model outputs in place.
    assert raw["boxes"].tolist() == [[0., 1., 4., 5.]]
    inverse = fast.native_prediction(raw, (8, 10), (16, 20), False, _validate_prediction)
    assert inverse["boxes"].tolist() == [[0., 2., 8., 10.]]


@pytest.fixture
def full_records():
    records, images, truth = [], [], {}
    for index in range(814):
        name = str(index)
        count = 5 if index < 245 else 4 if index < 801 else 0
        boxes = [[0., 0., 1., 1.] for _ in range(count)]
        images.append({"image_id": name})
        truth[name] = [{"box": box} for box in boxes]
        records.append({"image_id": name, "gt": {"boxes": boxes}, "pred": {"boxes": [], "scores": []}})
    return records, {"images": images, "truth": truth}


def test_full_gt_inventory_and_negative_frames_required(full_records):
    records, contract = full_records
    fast.audit_records(records, contract)
    records[0]["gt"]["boxes"] = []
    with pytest.raises(ValueError, match="genuine original GT"):
        fast.audit_records(records, contract)


def test_partial_val_cannot_be_completed(full_records):
    records, contract = full_records
    with pytest.raises(ValueError, match="complete 814"):
        fast.audit_records(records[:-1], contract)
    altered = deepcopy(records)
    altered[0], altered[1] = altered[1], altered[0]
    with pytest.raises(ValueError, match="identity/order"):
        fast.audit_records(altered, contract)


def test_512_full_frame_cannot_be_misrepresented_as_local_accuracy():
    args = fast.build_parser().parse_args(["--config", "original.json", "--stage", "small", "--output", "out", "--short-side", "512"])
    spec = fast.deployment_spec(args)
    assert spec["local_512_accuracy_claimed"] is False
    assert "FULL-frame" in spec["spatial_scope_note"]
    assert "never GT-centred" in spec["image_scope"]


@pytest.mark.parametrize("cutoff", [-.01, .7501, float("nan"), float("inf")])
def test_cutoff_cannot_remove_fixed_point_evidence(cutoff):
    with pytest.raises(ValueError):
        fast.policy(cutoff)


def test_original_low_score_control_is_available():
    assert fast.optional_probability("none") is None
    assert not fast.policy(None)["fixed_only"]
    assert "original model" in fast.policy(None)["curve_scope"]


def test_final_source_and_config_recheck(tmp_path, monkeypatch):
    config = tmp_path / "cfg.json"
    config.write_text('{"fixed": true}')
    metadata = {"config": str(config), "config_sha256": fast.sha(config)}
    source = {"frozen.py": "original"}
    monkeypatch.setattr(fast, "source_inventory", lambda: source.copy())
    assert all(fast.final_integrity(metadata, source).values())
    config.write_text('{"fixed": false}')
    with pytest.raises(RuntimeError, match="source or original config changed"):
        fast.final_integrity(metadata, source)
    metadata["config_sha256"] = fast.sha(config)
    with pytest.raises(RuntimeError, match="source or original config changed"):
        fast.final_integrity(metadata, {"frozen.py": "changed"})


def test_runtime_flags_cannot_be_implicitly_omitted():
    with pytest.raises(ValueError, match="flags are missing"):
        fast.runtime_numerics({"torch": "known"})
    flags = {"torch": "known", "gpu": "same GPU", "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": True}
    assert fast.runtime_numerics(flags) == flags


def test_original_only_is_explicit_and_cannot_use_paired_batch2():
    argv = ["--config", "original.json", "--stage", "small", "--output", "out", "--views", "original", "--short-side", "512"]
    args = fast.build_parser().parse_args(argv)
    assert fast.deployment_spec(args)["views"] == "original"
    args.paired_backbone_batch2 = True
    with pytest.raises(ValueError, match="incompatible with paired"):
        fast.deployment_spec(args)


def test_four_view_fusion_rejects_original_only_before_inference():
    args = fast.build_parser().parse_args(["--config", "original.json", "--stage", "fuse", "--output", "out", "--views", "original"])
    with pytest.raises(ValueError, match="requires --views both"):
        fast.deployment_spec(args)
