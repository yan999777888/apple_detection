"""CPU-only fixtures for full-VAL receipt checks and independent mask semantics."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("premask_summary_under_test", ROOT / "scripts/summarize_premask_validation.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)
METRICS = summary.load_metrics()


def score(value):
    return float(np.float32(value))


def fixture():
    names = ("heavy", "anomaly", "empty")
    truth = {
        "heavy": [{"image_id": "heavy", "gt_index": 0, "apple_id": "h", "box": [0., 0., 4., 4.],
                   "occlusion_stratum": "heavy_visible_lt_0.40", "visible_area_fraction": .25}],
        "anomaly": [{"image_id": "anomaly", "gt_index": 0, "apple_id": "a", "box": [8., 0., 12., 4.],
                     "occlusion_stratum": "annotation_visible_area_exceeds_amodal", "visible_area_fraction": 1.1}],
        "empty": [],
    }
    contract = {"images": [{"image_id": name} for name in names], "truth": truth,
                "strata": {"heavy_visible_lt_0.40": 1, "annotation_visible_area_exceeds_amodal": 1},
                "scope": {"GT_instances": 2, "empty_GT_images": 1}}
    records = [{"image_id": name, "gt": {"boxes": [g["box"] for g in truth[name]]},
                "pred": {"boxes": [g["box"] for g in truth[name]], "scores": [score(.8)] * len(truth[name])}}
               for name in names]
    return contract, records


def metadata(records, confidence=.75):
    return {"view_metrics": {"original": {
        "at_confidence_0_75": METRICS.evaluate_at_threshold(records, .75),
        "at_recall90": METRICS.evaluate_at_threshold(records, confidence)}}}


def test_original_gt_order_and_anomalies_are_preserved():
    contract, records = fixture()
    summary.audit_records(records, contract, METRICS)
    changed = deepcopy(records)
    changed[1]["gt"]["boxes"] = []
    with pytest.raises(ValueError, match="GT identity"):
        summary.audit_records(changed, contract, METRICS)
    with pytest.raises(ValueError, match="GT identity"):
        summary.audit_records(records[::-1], contract, METRICS)
    with pytest.raises(ValueError, match="inventory"):
        summary.audit_records(records[:-1], contract, METRICS)


def test_recorded_points_are_recomputed_without_calibration(monkeypatch):
    _, records = fixture()
    saved = metadata(records)
    monkeypatch.setattr(METRICS, "calibrate_threshold", lambda *_a, **_kw: pytest.fail("No threshold scan allowed"))
    result = summary.recorded_points(records, saved, METRICS)
    assert result["at_recall90"]["tp"] == 2
    saved["view_metrics"]["original"]["at_recall90"]["fp"] = 1
    with pytest.raises(ValueError, match="at_recall90.fp"):
        summary.recorded_points(records, saved, METRICS)


def test_null_operating_point_is_not_fabricated():
    _, records = fixture()
    saved = metadata(records)
    saved["view_metrics"]["original"]["at_recall90"] = None
    result = summary.recorded_points(records, saved, METRICS)
    assert result["at_recall90"] is None and result["null_P_at_R90"]


def test_invalid_recall90_cannot_be_reported_as_valid():
    _, records = fixture()
    records[0]["pred"] = {"boxes": [], "scores": []}
    with pytest.raises(ValueError, match="does not attain"):
        summary.recorded_points(records, metadata(records), METRICS)


def test_lost_recovered_are_per_gt_and_do_not_cancel_away():
    contract, old = fixture()
    new = deepcopy(old)
    old[1]["pred"] = {"boxes": [], "scores": []}
    new[0]["pred"] = {"boxes": [], "scores": []}
    new[2]["pred"] = {"boxes": [[20., 0., 24., 4.]], "scores": [score(.9)]}
    result = summary.compare_recall(old, new, contract, .75, .75, METRICS)
    assert (result["lost"], result["recovered"], result["delta"]["tp"], result["delta"]["fp"]) == (1, 1, 0, 1)
    assert result["groups"]["heavy_visible_lt_0.40"]["lost"] == 1
    assert result["groups"]["annotation_visible_area_exceeds_amodal"]["recovered"] == 1
    assert len(result["ground_truth_instances"]) == 2


def test_comparison_distinguishes_own_thresholds_from_common_new_threshold():
    contract, records = fixture()
    reference = {"records": deepcopy(records), "points": {"at_recall90": {"confidence": .7}}}
    candidate = {"records": records, "points": {"at_recall90": {"confidence": .8}}}
    result = summary.comparisons(reference, candidate, contract, METRICS)
    assert result["each_recorded_P_at_R90"]["reference_confidence"] == .7
    assert result["each_recorded_P_at_R90"]["candidate_confidence"] == .8
    assert result["both_at_candidate_P_at_R90_confidence"]["reference_confidence"] == .8


def save_roi(path, record, **replacements):
    count = len(record["pred"]["scores"])
    arrays = {"boxes": np.asarray(record["pred"]["boxes"], dtype=np.float32).reshape(-1, 4),
              "scores": np.asarray(record["pred"]["scores"], dtype=np.float32),
              "visible": np.ones((count, 2, 2), dtype=np.float32),
              "amodal": np.ones((count, 2, 2), dtype=np.float32)}
    arrays.update(replacements)
    np.savez(path, **arrays)


def test_archive_exact_binding_probability_validation_and_empty_shapes(tmp_path):
    _, records = fixture()
    for i in (0, 2):
        path = tmp_path / f"{i}.npz"
        save_roi(path, records[i])
        evidence = {}
        arrays = summary.load_roi_archive(path, records[i], evidence)
        assert arrays["visible"].shape == (len(records[i]["pred"]["scores"]), 2, 2)
        assert evidence[str(path)]["sha256"] == summary.sha(path)
    changed = deepcopy(records[0])
    changed["pred"]["scores"][0] = float(np.nextafter(np.float32(.8), np.float32(1.)))
    with pytest.raises(ValueError, match="exactly match"):
        summary.load_roi_archive(tmp_path / "0.npz", changed, {})
    save_roi(tmp_path / "bad.npz", records[0], visible=np.full((1, 2, 2), 1.01, dtype=np.float32))
    with pytest.raises(ValueError, match="probability"):
        summary.load_roi_archive(tmp_path / "bad.npz", records[0], {})


def test_input_receipt_detects_replaced_file(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"value":1}')
    evidence = {}
    assert summary.read_json(path, evidence) == {"value": 1}
    path.write_text('{"value":2}')
    with pytest.raises(ValueError, match="changed"):
        summary.read_json(path, evidence)


def test_missing_mask_archive_explicitly_disables_claim(tmp_path):
    contract, records = fixture()
    candidate = {"directory": str(tmp_path), "records": records}
    result = summary.evaluate_masks(candidate, contract, tmp_path, METRICS, {}, lambda *_a, **_kw: None)
    assert result["phase"] == "unavailable" and result["mask_metrics_claimed"] is False
    assert len(result["missing_archives"]) == 3 and "points" not in result


def test_real_cpu_mask_iou_separates_v_from_a_and_keeps_empty_fp(tmp_path, monkeypatch):
    import torch
    import apple_seg.data as data
    torch.set_num_threads(1)
    amodal = np.ones((1, 4, 4), dtype=np.uint8)
    visible = np.zeros_like(amodal)
    visible[:, :2, :2] = 1
    rows = [{"image_id": "apple", "gt_index": 0, "apple_id": "1", "box": [0., 0., 4., 4.],
             "visible_area_pixels": 4, "amodal_area_pixels": 16, "visible_area_fraction": .25,
             "visible_pixels_outside_amodal": 0, "occlusion_stratum": "heavy_visible_lt_0.40",
             "visible_mask_uint8_sha256": hashlib.sha256(visible[0].tobytes()).hexdigest(),
             "amodal_mask_uint8_sha256": hashlib.sha256(amodal[0].tobytes()).hexdigest()}]
    images = [{"image_id": "apple", "height": 4, "width": 4, "apple_ids_in_target_order": ["1"]},
              {"image_id": "empty", "height": 4, "width": 4, "apple_ids_in_target_order": []}]
    contract = {"images": images, "truth": {"apple": rows, "empty": []},
                "scope": {"GT_instances": 1, "empty_GT_images": 1}, "strata": {"heavy_visible_lt_0.40": 1},
                "provenance": {"binding": {"current_annotation_sha256": {}}}}
    records = [{"image_id": "apple", "gt": {"boxes": [[0., 0., 4., 4.]]},
                "pred": {"boxes": [[0., 0., 4., 4.]], "scores": [score(.8)]}},
               {"image_id": "empty", "gt": {"boxes": []},
                "pred": {"boxes": [[0., 0., 4., 4.]], "scores": [score(.7)]}}]
    class Native:
        annotation_sha256 = {}
        annotation_paths = {}
        def __init__(self, *_args, **_kwargs):
            pass
        def __getitem__(self, index):
            image = images[index]
            v = visible if index == 0 else np.empty((0, 4, 4), dtype=np.uint8)
            a = amodal if index == 0 else np.empty((0, 4, 4), dtype=np.uint8)
            return {"target": {"visible_masks": torch.from_numpy(v), "amodal_masks": torch.from_numpy(a),
                               "annotation_validity": {"visible": True, "amodal": True},
                               "labels": torch.zeros(len(v), dtype=torch.int64), "original_hw": [4, 4],
                               "instance_ids": image["apple_ids_in_target_order"],
                               "source": {"dataset": "PApple", "split": "val", "image_id": image["image_id"]}}}
    monkeypatch.setattr(data, "PAppleDataset", Native)
    directory = tmp_path / "roi_masks/original"
    directory.mkdir(parents=True)
    for i, record in enumerate(records):
        save_roi(directory / f"{i:06d}.npz", record)
    candidate = {"directory": str(tmp_path), "records": records, "points": {"at_recall90": {"confidence": .6}},
                 "metadata": {"roi_masks_saved": True}}
    progress = []
    result = summary.evaluate_masks(candidate, contract, tmp_path, METRICS, {}, lambda *a, **kw: progress.append((a, kw)))
    assert result["phase"] == "complete" and result["mask_metrics_claimed"] is True
    assert result["reference_mask_metrics_claimed"] is False
    fixed, low = result["points"]["at_confidence_0_75"], result["points"]["at_recall90"]
    assert (fixed["visible"]["tp"], fixed["visible"]["fn"], fixed["amodal"]["tp"]) == (0, 1, 1)
    assert fixed["amodal"]["fp"] == 0 and low["amodal"]["fp"] == 1
    assert fixed["occlusion"]["groups"]["heavy_visible_lt_0.40"]["amodal_mask_recall"] == 1.
    assert len(result["roi_archive_receipts"]) == 2 and progress[-1][1]["images"] == 2


def test_load_stage_rejects_incomplete_before_accessing_predictions(tmp_path):
    (tmp_path / "prediction_metadata.json").write_text(json.dumps({"phase": "inference", "model": "base"}))
    contract, _ = fixture()
    with pytest.raises(ValueError, match="Incomplete"):
        summary.load_stage(tmp_path, "base", "weight", contract, METRICS, {}, kind="candidate")


def test_required_inference_source_receipts_cannot_be_omitted_or_forged():
    with pytest.raises(ValueError, match="required frozen"):
        summary.verify_sources({"source_sha256": {}, "source_before_after_identical": True}, {})
    inventory = {name: summary.sha(ROOT / name) for name in summary.REQUIRED_SOURCES}
    metadata = {"source_sha256": inventory, "source_before_after_identical": True}
    summary.verify_sources(metadata, {})
    inventory[next(iter(inventory))] = "wrong-sha"
    with pytest.raises(ValueError, match="source differs"):
        summary.verify_sources(metadata, {})


def test_complete_npz_files_without_inference_mask_receipt_are_not_claimed(tmp_path):
    contract, records = fixture()
    directory = tmp_path / "roi_masks/original"
    directory.mkdir(parents=True)
    for index, record in enumerate(records):
        save_roi(directory / f"{index:06d}.npz", record)
    candidate = {"directory": str(tmp_path), "records": records, "metadata": {"roi_masks_saved": False}}
    result = summary.evaluate_masks(candidate, contract, tmp_path, METRICS, {}, lambda *_a, **_kw: None)
    assert result["phase"] == "unavailable" and result["missing_archives"] == []
    assert result["mask_metrics_claimed"] is False


def test_script_has_no_model_construction_or_gpu_execution():
    import ast
    tree = ast.parse((ROOT / "scripts/summarize_premask_validation.py").read_text(encoding="utf-8"))
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any("cuda" in call or "build_od_amodal_detector" in call or "load_checkpoint" in call for call in calls)
