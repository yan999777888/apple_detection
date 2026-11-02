"""Independent CPU replay of frozen fusion and fixed operating points; no GPU.

Writes only the explicitly authorized paired_audit.json after all checks pass.
Does not infer any images, train, mutate prior evidence or search thresholds.
"""
from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/premask_20261102/paired_validation/paired_audit.json"
NEW = ROOT / "runs/premask_nms080_k64_paired_20261102"
SPEC = importlib.util.spec_from_file_location("paired_audit_helpers", ROOT / "scripts/summarize_premask_validation.py")
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)
from apple_seg.evaluate import load_papple_contract

receipts = {}


def bind(path):
    path = Path(path).resolve()
    value = {"path": str(path), "sha256": helpers.sha(path), "bytes": path.stat().st_size}
    if str(path) in receipts:
        assert receipts[str(path)] == value, f"Input changed: {path}"
    receipts[str(path)] = value
    return value


def read(path):
    receipt = bind(path)
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    assert helpers.sha(path) == receipt["sha256"]
    return value


def check_curve(directory, points):
    curve = read(directory / "fusion_calibration_curve.json")["curve"]
    eligible = [row for row in curve if row["recall"] >= .9]
    best = max(eligible, key=lambda row: (row["precision"], row["recall"], row["f1"], row["confidence"]))
    assert best == points["at_recall90"]
    assert len(curve) == points["candidate_thresholds"]
    return {"existing_curve_points": len(curve), "chosen_point_matches_saved_curve": True,
            "threshold_search_rerun": False}


def check_points(records, stored, metrics):
    results = {}
    for label in ("at_confidence_0_75", "at_recall90"):
        results[label] = metrics.evaluate_at_threshold(records, stored[label]["confidence"], .5)
        helpers.verify_point(results[label], stored[label], label)
    return results


def main():
    if OUT.exists():
        raise FileExistsError("The independent audit artifact already exists; it will not be overwritten")
    provenance = ROOT.parent / "Apple_DualBranch_3080_20261022/assets/papple_val_provenance.json"
    bind(provenance)
    bind(__file__)
    bind(ROOT / "scripts/summarize_premask_validation.py")
    contract = load_papple_contract(provenance)
    metrics = helpers.load_metrics()
    summary, status = read(NEW / "fusion_summary.json"), read(NEW / "fusion_status.json")
    assert {k: v for k, v in status.items() if k != "elapsed_seconds"} == summary
    pipeline = read(ROOT / "reports/premask_20261102/paired_validation/jobs/pipeline_status.json")
    assert pipeline["phase"] == "complete" and len(pipeline["stages"]) == 3
    assert bind(pipeline["plan"])["sha256"] == pipeline["plan_sha256"]
    for stage in pipeline["stages"]:
        assert stage["phase"] == "complete" and stage["exit_code"] == 0
        assert bind(stage["arguments"][2])["sha256"] == stage["script_sha256"]
    assert summary["phase"] == "complete" and (summary["images"], summary["GT"], summary["empty"]) == (814, 3449, 13)
    assert summary["original_strong_models_only"] and not summary["training_performed"] and not summary["mask_fusion"]
    assert summary["provenance_sha256"] == contract["provenance_sha256"]
    for field in ("source_before_after_identical", "config_before_after_identical",
                  "candidate_policy_before_after_identical", "original_weights_before_after_identical"):
        assert summary[field] is True
    config = read(summary["config"])
    assert bind(summary["config"])["sha256"] == summary["config_sha256"]
    assert summary["fusion"] == config["fusion"]
    policy = summary["deployment"]["candidate_policy_file"]
    assert bind(policy["path"])["sha256"] == policy["sha256"]
    assert read(policy["path"]) == summary["deployment"]["candidate_policy"] == {
        "score_threshold": None, "box_nms_iou": .8, "topk": 64}
    for relative, digest in summary["source_sha256"].items():
        assert bind(ROOT / relative)["sha256"] == digest
    for binding in summary["inputs"].values():
        assert bind(binding["path"])["sha256"] == binding["sha256"]
    layers, model_receipts = [], {}
    for model in ("base", "small"):
        meta = read(NEW / model / "prediction_metadata.json")
        assert meta["phase"] == "complete" and (meta["images"], meta["GT"], meta["empty"]) == (814, 3449, 13)
        assert meta["checkpoint_sha256"] == helpers.ORIGINAL_CHECKPOINTS[model]
        assert meta["from_fresh_weight_inference"] and not meta["training_performed"]
        assert meta["provenance_sha256"] == contract["provenance_sha256"]
        assert meta["deployment"] == summary["deployment"] and meta["source_sha256"] == summary["source_sha256"]
        assert meta["config_sha256"] == summary["config_sha256"]
        assert meta["strict_restore"] == {"strict": True, "tensor_count": 434, "missing_keys": [],
                                           "unexpected_keys": [], "all_tensors_equal_to_checkpoint": True}
        dep = meta["deployment"]
        assert dep["paired_backbone_batch2"] and dep["views"] == "both" and dep["precision"] == "fp16"
        assert dep["roi_chunk_size"] == 64 and dep["roi_precision"] == "autocast" and dep["pre_nms_threshold"] is None
        assert dep["short_side"] == 1024 and dep["max_size"] == 1280 and dep["candidate_selection_uses_gt"] is False
        weight = summary["original_weight_bindings"][model]
        assert bind(weight["path"])["sha256"] == weight["sha256"] == helpers.ORIGINAL_CHECKPOINTS[model]
        for view in ("original", "hflip"):
            path = NEW / model / f"{view}_predictions.json"
            assert bind(path)["sha256"] == meta["predictions_sha256"][view] == summary["inputs"][model + "_" + view]["sha256"]
            layer = read(path)
            helpers.audit_records(layer, contract, metrics)
            layers.append(layer)
        model_receipts[model] = {"metadata": bind(NEW / model / "prediction_metadata.json"),
                                 "weight": weight, "completed_utc": meta["completed_utc"], "strict_restore": meta["strict_restore"]}
    raw = read(NEW / "fused_predictions.json")
    assert bind(NEW / "fused_predictions.json")["sha256"] == summary["predictions_sha256"]
    helpers.audit_records(raw, contract, metrics)
    tree = ast.parse((ROOT / "strong_runtime/tmp/flip_tta.py").read_text(encoding="utf-8-sig"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_rescale", "fuse_entries", "fuse_records")]
    namespace = {"np": np, "math": math, "metrics": metrics}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "frozen_numpy_fusion", "exec"), namespace)
    assert namespace["fuse_records"](layers, **config["fusion"]) == raw
    del layers
    points = check_points(raw, summary["metrics"], metrics)
    curve_check = check_curve(NEW, summary["metrics"])
    assert metrics.evaluate_at_threshold(raw, 0., .5)["recall"] == summary["metrics"]["max_attainable_recall"]
    references, comparisons = {}, {}
    for name, directory, expected_summary_sha in (
        ("original_fp32", ROOT / "runs/strong_baseline_20261030", "eef8ab1de232bb18ca61e1f0b25132a4df92b5b6e9c621464dd46df2143a1772"),
        ("old_fp16_uncut", ROOT / "runs/fast_fp16_all_paired_1024_20261031", "c4bfa38d81fd06fafb9f9561e9ac8eb98a502b456ff0878f03604cfe89257b5a"),
    ):
        old = read(directory / "fusion_summary.json")
        assert bind(directory / "fusion_summary.json")["sha256"] == expected_summary_sha
        old_points = old.get("metrics", old)
        old_raw = read(directory / "fused_predictions.json")
        helpers.audit_records(old_raw, contract, metrics)
        assert old["fusion"] == summary["fusion"]
        details = {"summary": bind(directory / "fusion_summary.json"), "raw": bind(directory / "fused_predictions.json"),
                   "points": check_points(old_raw, old_points, metrics)}
        if name == "old_fp16_uncut":
            assert bind(directory / "fused_predictions.json")["sha256"] == old["predictions_sha256"]
            assert old["deployment"]["pre_nms_threshold"] is None and "candidate_policy" not in old["deployment"]
            details["curve_check"] = check_curve(directory, old_points)
        else:
            details["curve_check"] = {"available": False, "note": "Original baseline has no saved curve here; pinned summary point independently recomputed, no new search"}
        references[name] = details
        comparisons[name] = {}
        for label, oc, nc in (
            ("fixed_0_75", .75, .75),
            ("each_recorded_P_at_R90", old_points["at_recall90"]["confidence"], points["at_recall90"]["confidence"]),
            ("both_at_candidate_P_at_R90_confidence", points["at_recall90"]["confidence"], points["at_recall90"]["confidence"]),
        ):
            result = helpers.compare_recall(old_raw, raw, contract, oc, nc, metrics)
            result["changed_ground_truth_instances"] = [g for g in result["ground_truth_instances"] if g["outcome"] in ("lost", "recovered")]
            comparisons[name][label] = result
    baseline = references["original_fp32"]["points"]
    accepted = points["at_recall90"]["precision"] >= baseline["at_recall90"]["precision"] and points["at_recall90"]["recall"] >= .9
    assert accepted is False
    for receipt in list(receipts.values()):
        assert bind(receipt["path"]) == receipt
    report = {
        "schema": "independent_original_strong_premask_paired_cpu_audit_v1", "phase": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "inference_completed_utc": summary["completed_utc"],
        "pipeline_completed_utc": pipeline["completed_utc"], "images": 814, "GT": 3449, "empty": 13,
        "GT_occlusion_strata": contract["strata"], "no_GT_repair_or_exclusion": True,
        "GPU_used": False, "training_performed": False, "image_inference_performed_by_this_audit": False,
        "fresh_weight_inference_source": "New original Base and Small FP16 paired batch2 original+hflip predictions; four sources verified",
        "candidate_deployment": summary["deployment"], "fusion_parameters": summary["fusion"],
        "model_receipts": model_receipts, "points": points, "existing_curve_check": curve_check,
        "independently_recomputed_fusion_records_exactly_equal": True, "full_fusion_record_equality_scope": "All boxes, scores, GT and image/instance order",
        "fusion_masks_verified": False, "method": [
            "Verify three pipeline stages exit0, script/plan receipts, six fusion input receipts, 38 source bindings and original weight file SHA",
            "Verify complete raw GT/image identity against frozen original provenance, including all 14 V-area>A-area anomalies",
            "Execute exact frozen NumPy fusion functions on the four fresh FP16 layers and compare full records by exact equality",
            "Recompute fixed .75 and stored P90 points with frozen score-descending stable one-to-one IoU>=.5 metrics",
            "Read saved curves to verify selected maximum P at R>=.90; no threshold calibration search rerun",
            "Recompute all per-GT lost/recovered and occlusion groups at fixed .75, respective recorded P90, and shared new P90 threshold",
        ], "references": references, "comparisons": comparisons,
        "acceptance": {"maintains_original_FP32_P_at_R90": accepted,
                       "fixed_0_75_recall_ge_90": points["at_confidence_0_75"]["recall"] >= .9,
                       "maintains_original_fixed_0_75_P_and_R": all(points["at_confidence_0_75"][k] >= baseline["at_confidence_0_75"][k] for k in ("precision", "recall")),
                       "promoted": False, "decision": "Reject promotion under the approved original strong baseline preservation gate; retain original baseline",
                       "reason": "New P@R90 precision is lower than original FP32 four-view precision; fixed .75 recall is strictly below .90"},
        "limitations": ["No fused-pixel-mask or D405 tracking validation; this audit covers fused boxes/scores",
                        "No new RGB/GT-mask raster audit in this step; archived full GT boxes and identity are bound to frozen provenance",
                        "P90 is selected on VAL and does not constitute an independent test result",
                        "Paired batch2 fresh outputs are distinct from batch1 single-view evidence; no new .35 truncation applied"],
        "source_and_input_receipts": list(receipts.values()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    assert json.loads(OUT.read_text(encoding="utf-8")) == report
    print(json.dumps({"phase": "complete", "path": str(OUT), "sha256": helpers.sha(OUT),
                      "receipt_count": len(receipts), "points": points, "acceptance": report["acceptance"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
