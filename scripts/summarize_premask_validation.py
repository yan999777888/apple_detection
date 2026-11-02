"""CPU-only full-VAL audit of original-weight pre-mask deployment outputs.

No model is constructed. Recorded P@R90 operating points are independently
recomputed, not searched again. Optional mask PR is new-candidate-only and
requires all 814 native-oriented ROI archives plus genuine original V/A GT.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "strong_runtime"
sys.path.insert(0, str(RUNTIME / "src"))
METRICS_PATH = RUNTIME / "reference/local_amodal/scripts/metrics.py"
METRICS_SHA = "1b0a7e385940fd0c6160cc08b75ddaf2e1f43f3ea00f5bd37d547fea6b08010d"
POINT_FIELDS = ("tp", "fp", "fn", "gt_count", "prediction_count", "precision", "recall", "f1")
ORIGINAL_CHECKPOINTS = {
    "base": "0fe257c5bfe115a77ac7deac830af479e4c2b0ad6ff69e609901b24c7980fa9b",
    "small": "e0c1f659c2509d7a01d617e12cb83c673768b440ea5b4b96cb4f763e4207e7e3",
}
REQUIRED_SOURCES = {"strong_runtime/reference/local_amodal/scripts/metrics.py",
                    *{f"strong_runtime/src/apple_seg/{name}.py" for name in
                      ("data", "evaluate", "od_evaluate", "od_model", "od_roi")}}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def bind(path, evidence):
    path = Path(path).resolve()
    row = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
    old = evidence.get(str(path))
    require(old is None or old == row, f"Input changed while reading: {path}")
    evidence[str(path)] = row
    return row


def read_json(path, evidence):
    before = bind(path, evidence)
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(sha(path) == before["sha256"], f"JSON changed while reading: {path}")
    return value


def load_metrics():
    require(sha(METRICS_PATH) == METRICS_SHA, "Frozen metrics source SHA differs")
    spec = importlib.util.spec_from_file_location("premask_frozen_metrics", METRICS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_records(records, contract, metrics):
    """Keep every image, empty frame, GT instance, anomaly and original order."""
    require(isinstance(records, list) and len(records) == len(contract["images"]), "Full image inventory differs")
    for record, image in zip(records, contract["images"], strict=True):
        expected = [row["box"] for row in contract["truth"][image["image_id"]]]
        require(record.get("image_id") == image["image_id"] and record.get("gt") == {"boxes": expected},
                "Original image/GT identity, order or boxes differ")
        metrics._normalize_record(record)
    require(sum(len(r["gt"]["boxes"]) for r in records) == contract["scope"]["GT_instances"]
            and sum(not r["gt"]["boxes"] for r in records) == contract["scope"]["empty_GT_images"],
            "Full GT/empty-frame inventory differs")


def verify_point(actual, recorded, name):
    require(isinstance(recorded, dict), f"Recorded operating point missing: {name}")
    require(recorded.get("confidence") == actual["confidence"] and recorded.get("iou_threshold") == .5,
            f"Recorded confidence/IoU differs: {name}")
    for key in POINT_FIELDS:
        require(key in recorded and actual[key] == recorded[key], f"Recorded {name}.{key} differs from raw predictions")


def recorded_points(records, metadata, metrics):
    """Verify stored chosen points without a new threshold optimization."""
    saved = metadata["view_metrics"]["original"]
    fixed = metrics.evaluate_at_threshold(records, .75, .5)
    if "at_confidence_0_75" in saved:
        verify_point(fixed, saved["at_confidence_0_75"], "fixed_0_75")
    require("at_recall90" in saved, "Metadata must explicitly record at_recall90 (including null)")
    chosen = saved["at_recall90"]
    at90 = None
    if chosen is not None:
        confidence = chosen.get("confidence")
        require(type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1,
                "Invalid recorded P@R90 confidence")
        at90 = metrics.evaluate_at_threshold(records, confidence, .5)
        verify_point(at90, chosen, "at_recall90")
        require(at90["recall"] >= .9, "Recorded P@R90 does not attain recall >= .90")
    return {"at_confidence_0_75": fixed, "at_recall90": at90,
            "P_at_R90_selection": "Recorded full-VAL threshold independently recomputed; no threshold search repeated",
            "null_P_at_R90": "No recorded attainable point; this summarizer does not independently rescan the curve" if at90 is None else None}


def verify_sources(metadata, evidence):
    inventory = metadata.get("source_sha256", {})
    require(isinstance(inventory, dict) and REQUIRED_SOURCES <= {key.replace("\\", "/") for key in inventory},
            "Inference receipt lacks required frozen source bindings")
    for relative, digest in inventory.items():
        source = REPO / relative.replace("\\", "/")
        require(bind(source, evidence)["sha256"] == digest, f"Inference source differs: {relative}")
    require(metadata.get("source_before_after_identical") is True, "Inference source integrity evidence missing")


def load_stage(directory, model, expected_weight, contract, metrics, evidence, *, kind):
    directory = Path(directory).resolve()
    metadata = read_json(directory / "prediction_metadata.json", evidence)
    require(metadata.get("phase") == "complete" and metadata.get("model") == model,
            f"Incomplete/wrong model stage: {directory}")
    require((metadata.get("images"), metadata.get("GT"), metadata.get("empty")) == (814, 3449, 13),
            "Stage must cover original full VAL 814/3449/13")
    require(metadata.get("from_fresh_weight_inference") is True
            and metadata.get("checkpoint_sha256") == expected_weight
            and metadata.get("provenance_sha256") == contract["provenance_sha256"]
            and metadata.get("training_performed", False) is False,
            "Stage original-weight/fresh-inference/GT identity differs")
    restore = metadata.get("strict_restore", {})
    require(restore.get("strict") is True and restore.get("all_tensors_equal_to_checkpoint") is True
            and restore.get("missing_keys") == [] and restore.get("unexpected_keys") == [],
            "Original weight strict restoration evidence missing")
    if kind == "fp32_reference":
        require(metadata.get("FP32") is True and metadata.get("candidate_mode") is False,
                "Reference must be the original FP32 strong model")
    elif kind == "fp16_reference":
        dep = metadata.get("deployment", {})
        require(dep.get("precision") == "fp16" and dep.get("views") == "original"
                and dep.get("paired_backbone_batch2") is False and dep.get("short_side") == 1024
                and dep.get("pre_nms_threshold") is None and "candidate_policy" not in dep,
                "FP16 reference must be prior unfiltered single-view 1024 inference")
    else:
        dep = metadata.get("deployment", {})
        require(dep.get("deployment_experiment") == "pre_mask_candidate_selection_v1"
                and dep.get("candidate_selection_uses_gt") is False
                and dep.get("pre_nms_threshold") is None
                and dep.get("evaluation_policy", {}).get("fixed_only") is False,
                "Candidate must be GT-free pre-mask inference with all returned scores")
        require(metadata.get("candidate_policy_before_after_identical") is True
                and metadata.get("original_weights_before_after_identical") is True,
                "Candidate policy/weight before-after integrity receipt missing")
        binding = dep.get("candidate_policy_file", {})
        require(binding.get("path") and binding.get("sha256"), "Candidate policy file binding missing")
        require(bind(binding["path"], evidence)["sha256"] == binding["sha256"], "Candidate policy file SHA differs")
        policy = read_json(binding["path"], evidence)
        keys = ("score_threshold", "box_nms_iou", "topk")
        require(isinstance(policy, dict) and not set(policy) - set(keys)
                and {key: policy.get(key) for key in keys} == dep.get("candidate_policy"), "Candidate policy payload differs")
    verify_sources(metadata, evidence)
    path = directory / "original_predictions.json"
    records = read_json(path, evidence)
    expected = metadata.get("predictions_sha256", {}).get("original")
    if expected is not None:
        require(evidence[str(path)]["sha256"] == expected, "Raw predictions differ from inference receipt")
    elif kind != "fp32_reference":
        raise ValueError("New/FP16 stage missing prediction SHA receipt")
    audit_records(records, contract, metrics)
    return {"directory": str(directory), "metadata": metadata, "records": records,
            "points": recorded_points(records, metadata, metrics),
            "raw_hash_binding": "inference metadata SHA verified" if expected else
            "legacy FP32 metadata has no prediction SHA; current raw SHA recorded and saved operating point independently verified"}


def recalled_gt(records, confidence, metrics):
    result = {}
    for row in records:
        match = metrics.match_image(row["gt"]["boxes"], row["pred"]["boxes"], row["pred"]["scores"], .5, confidence)
        result[row["image_id"]] = {int(g) for g in match["gt_indices"] if g >= 0}
    return result


def compare_recall(reference, candidate, contract, reference_confidence, candidate_confidence, metrics):
    old = recalled_gt(reference, reference_confidence, metrics)
    new = recalled_gt(candidate, candidate_confidence, metrics)
    groups = {name: {"GT": count, "reference_TP": 0, "candidate_TP": 0,
                     "lost": 0, "recovered": 0, "both_recalled": 0, "both_missed": 0}
              for name, count in contract["strata"].items()}
    instances = []
    for image in contract["images"]:
        for gt in contract["truth"][image["image_id"]]:
            a, b = gt["gt_index"] in old[image["image_id"]], gt["gt_index"] in new[image["image_id"]]
            outcome = "both_recalled" if a and b else "lost" if a else "recovered" if b else "both_missed"
            group = groups[gt["occlusion_stratum"]]
            group["reference_TP"] += int(a)
            group["candidate_TP"] += int(b)
            group[outcome] += 1
            instances.append({"image_id": gt["image_id"], "gt_index": gt["gt_index"], "apple_id": gt["apple_id"],
                              "occlusion_stratum": gt["occlusion_stratum"], "visible_area_fraction": gt["visible_area_fraction"],
                              "reference_recalled": a, "candidate_recalled": b, "outcome": outcome})
    for group in groups.values():
        group["reference_recall"] = group["reference_TP"] / group["GT"] if group["GT"] else None
        group["candidate_recall"] = group["candidate_TP"] / group["GT"] if group["GT"] else None
        require(group["candidate_TP"] - group["reference_TP"] == group["recovered"] - group["lost"], "Recall delta invariant failed")
    old_point = metrics.evaluate_at_threshold(reference, reference_confidence, .5)
    new_point = metrics.evaluate_at_threshold(candidate, candidate_confidence, .5)
    return {"reference_confidence": reference_confidence, "candidate_confidence": candidate_confidence,
            "reference": old_point, "candidate": new_point,
            "delta": {key: new_point[key] - old_point[key] for key in POINT_FIELDS},
            "lost": sum(g["lost"] for g in groups.values()), "recovered": sum(g["recovered"] for g in groups.values()),
            "groups": groups, "ground_truth_instances": instances,
            "interpretation": "GT-stratum recall, with all annotation anomalies retained; FP has no GT occlusion stratum"}


def comparisons(reference, candidate, contract, metrics):
    result = {"fixed_0_75": compare_recall(reference["records"], candidate["records"], contract, .75, .75, metrics)}
    old90, new90 = reference["points"]["at_recall90"], candidate["points"]["at_recall90"]
    result["each_recorded_P_at_R90"] = (compare_recall(reference["records"], candidate["records"], contract,
                                                     old90["confidence"], new90["confidence"], metrics)
                                          if old90 is not None and new90 is not None else None)
    result["both_at_candidate_P_at_R90_confidence"] = (compare_recall(reference["records"], candidate["records"], contract,
                                                                      new90["confidence"], new90["confidence"], metrics)
                                                           if new90 is not None else None)
    return result


def load_roi_archive(path, record, evidence):
    """Check exact coordinates/score values and probability validity before slicing."""
    receipt = bind(path, evidence)
    with np.load(path, allow_pickle=False) as archive:
        require(set(("boxes", "scores", "visible", "amodal")).issubset(archive.files), "ROI archive fields missing")
        arrays = {name: archive[name] for name in ("boxes", "scores", "visible", "amodal")}
    require(sha(path) == receipt["sha256"], "ROI archive changed during read")
    expected_boxes = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
    expected_scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
    require(arrays["boxes"].shape == expected_boxes.shape and arrays["scores"].shape == expected_scores.shape
            and np.array_equal(arrays["boxes"].astype(np.float64), expected_boxes)
            and np.array_equal(arrays["scores"].astype(np.float64), expected_scores),
            "ROI boxes/scores must exactly match raw JSON values/order")
    count = len(expected_scores)
    shape = None
    for branch in ("visible", "amodal"):
        value = arrays[branch]
        require(np.issubdtype(value.dtype, np.floating) and value.ndim == 3 and value.shape[0] == count
                and min(value.shape[-2:]) > 0 and (shape is None or value.shape == shape)
                and np.isfinite(value).all() and ((value >= 0) & (value <= 1)).all(),
                "ROI masks must be aligned finite V/A probability arrays [N,Hroi,Wroi]")
        shape = value.shape
    return arrays


def evaluate_masks(candidate, contract, papple_root, metrics, evidence, progress):
    from apple_seg import evaluate as base
    from apple_seg.data import PAppleDataset
    from apple_seg.od_evaluate import _roi_mask_overlaps
    import torch

    directory = Path(candidate["directory"]) / "roi_masks/original"
    files = [directory / f"{index:06d}.npz" for index in range(len(contract["images"]))]
    missing = [str(path) for path in files if not path.is_file()]
    receipts = [bind(path, evidence) for path in files if path.is_file()]
    archives_declared = candidate.get("metadata", {}).get("roi_masks_saved") is True
    if missing or not archives_declared:
        return {"phase": "unavailable", "mask_metrics_claimed": False, "missing_archives": missing,
                "available_archive_count": len(receipts), "roi_archive_receipts": receipts,
                "inference_receipt_declares_saved_ROI_masks": archives_declared,
                "reason": "All original-view ROI archives and the inference saved-mask receipt are required; no partial/full mask accuracy is fabricated",
                "reference_mask_metrics_claimed": False}
    native = PAppleDataset(papple_root, "val", 32, augment=False)
    require(native.annotation_sha256 == contract["provenance"]["binding"]["current_annotation_sha256"], "Actual V/A annotation SHA differs")
    for path in native.annotation_paths.values():
        bind(path, evidence)
    points = {"at_confidence_0_75": .75}
    at90 = candidate["points"]["at_recall90"]
    if at90 is not None:
        points["at_recall90"] = at90["confidence"]
    minimum = min(points.values())
    diagnostics = []
    processed = 0
    with torch.inference_mode():
        for index, (record, frozen, path) in enumerate(zip(candidate["records"], contract["images"], files, strict=True)):
            arrays = load_roi_archive(path, record, evidence)
            selected = np.flatnonzero(np.asarray(record["pred"]["scores"], dtype=np.float64) >= minimum)
            target = native[index]["target"]
            truth = base._audit_target(target, frozen, contract["truth"][frozen["image_id"]])
            scores = [record["pred"]["scores"][int(p)] for p in selected]
            prediction = {"boxes": torch.from_numpy(np.asarray(arrays["boxes"][selected], dtype=np.float32)),
                          "scores": scores, "visible": torch.from_numpy(np.asarray(arrays["visible"][selected], dtype=np.float32)),
                          "amodal": torch.from_numpy(np.asarray(arrays["amodal"][selected], dtype=np.float32))}
            matrices = (_roi_mask_overlaps(prediction, truth, (frozen["height"], frozen["width"]),
                                           gt_chunk_size=8, tile_pixels=65536) if len(selected) else
                        {branch + "_mask": np.empty((0, len(truth["boxes"]))) for branch in ("visible", "amodal")})
            diagnostic = {"image_id": frozen["image_id"], "scores": scores,
                          "box_gt_indices_by_prediction": base._greedy_assignment(
                              metrics.box_iou(prediction["boxes"].numpy(), truth["boxes"]), scores)}
            for branch, matrix in matrices.items():
                diagnostic[branch + "_gt_indices_by_prediction"] = base._greedy_assignment(matrix, scores)
            diagnostics.append(diagnostic)
            processed += len(selected)
            if (index + 1) % 10 == 0 or index + 1 == len(files):
                progress("mask_metrics", images=index + 1, selected_mask_predictions=processed)
            del arrays, prediction, truth, target, matrices
    results = {}
    for label, confidence in points.items():
        results[label] = {"confidence": confidence,
                          "visible": base._mask_pr(diagnostics, "visible_mask", confidence, contract["scope"]["GT_instances"], metrics),
                          "amodal": base._mask_pr(diagnostics, "amodal_mask", confidence, contract["scope"]["GT_instances"], metrics),
                          "occlusion": base._occlusion(diagnostics, contract, confidence)}
    return {"phase": "complete", "mask_metrics_claimed": True, "images": len(contract["images"]),
            "GT": contract["scope"]["GT_instances"], "empty": contract["scope"]["empty_GT_images"],
            "minimum_processed_confidence": minimum, "selected_mask_predictions": processed,
            "points": results, "roi_archive_receipts": receipts, "reference_mask_metrics_claimed": False,
            "archive_binding": "NPZ SHA captured during this CPU audit; exact box/score equality checked against inference-bound raw JSON",
            "interpretation": "New original single view only; independent V/A mask IoU>=.5 matching at recorded box thresholds; no mask AP or fused-mask claim"}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage-dir", type=Path, required=True)
    result.add_argument("--reference-fp32", type=Path)
    result.add_argument("--reference-fp16", type=Path)
    result.add_argument("--config", type=Path, default=REPO / "configs/strong_baseline_4060.json")
    result.add_argument("--output", type=Path)
    result.add_argument("--mask-metrics", action="store_true")
    result.add_argument("--cpu-threads", type=int, default=4)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    output = (args.output or args.stage_dir / "cpu_validation").resolve()
    require(output.drive.upper() == "F:", "Summary files must stay on F:")
    require(args.cpu_threads >= 1, "CPU thread count must be positive")
    if (output / "summary.json").exists() or (output / "status.json").exists():
        raise FileExistsError("Use a fresh output directory; prior summary/status evidence is never overwritten")
    evidence = {}
    started = time.perf_counter()
    status = {"schema": "premask_cpu_validation_status_v1", "phase": "initializing", "created_utc": now(),
              "stage_directory": str(args.stage_dir.resolve()), "GPU_used": False, "model_constructed": False}

    def progress(phase, **values):
        status.update(phase=phase, updated_utc=now(), elapsed_seconds=time.perf_counter() - started, **values)
        atomic_json(output / "status.json", status)

    progress("initializing")
    try:
        import torch
        from apple_seg.evaluate import load_papple_contract
        torch.set_num_threads(args.cpu_threads)
        config = read_json(args.config, evidence)
        root = (args.config.resolve().parent / config["path_base"]).resolve()
        paths = {key: (root / value).resolve() for key, value in config["paths"].items()}
        contract = load_papple_contract(paths["provenance"])
        bind(paths["provenance"], evidence)
        metrics = load_metrics()
        for path in (Path(__file__), METRICS_PATH, RUNTIME / "src/apple_seg/evaluate.py",
                     RUNTIME / "src/apple_seg/data.py", RUNTIME / "src/apple_seg/od_evaluate.py", RUNTIME / "src/apple_seg/od_roi.py"):
            bind(path, evidence)
        preliminary = read_json(args.stage_dir / "prediction_metadata.json", evidence)
        model = preliminary.get("model")
        require(model in ("base", "small"), "Only original Base or Small single-model stages supported")
        weight = config[model + "_checkpoint_sha256"]
        require(weight == ORIGINAL_CHECKPOINTS[model], "Config does not identify the pinned original strong model")
        require(bind(paths[model + "_checkpoint"], evidence)["sha256"] == weight, "Original model checkpoint SHA differs")
        directories = {"candidate": args.stage_dir,
                       "fp32_reference": args.reference_fp32 or REPO / "runs/strong_baseline_20261030" / model,
                       "fp16_reference": args.reference_fp16 or REPO / "runs/fast_fp16_all_single_1024_20261031" / model}
        stages = {}
        for name, directory in directories.items():
            progress("load_" + name)
            stages[name] = load_stage(directory, model, weight, contract, metrics, evidence, kind=name)
        progress("box_comparisons")
        candidate = stages["candidate"]
        comparison = {name: comparisons(stages[name], candidate, contract, metrics)
                      for name in ("fp32_reference", "fp16_reference")}
        masks = (evaluate_masks(candidate, contract, paths["papple_root"], metrics, evidence, progress) if args.mask_metrics else
                 {"phase": "not_requested", "mask_metrics_claimed": False, "reference_mask_metrics_claimed": False})
        progress("final_integrity")
        for row in evidence.values():
            require(Path(row["path"]).stat().st_size == row["bytes"] and sha(row["path"]) == row["sha256"],
                    f"Source changed during CPU audit: {row['path']}")
        result = {"schema": "original_strong_premask_full_val_cpu_summary_v1", "phase": "complete", "completed_utc": now(),
                  "model": model, "images": 814, "GT": 3449, "empty": 13, "GT_occlusion_strata": contract["strata"],
                  "no_GT_repair_or_exclusion": True, "GPU_used": False, "model_constructed": False,
                  "training_performed": False, "source_before_after_identical": True, "automatically_promoted": False,
                  "candidate_deployment": candidate["metadata"]["deployment"],
                  "stage_points": {key: value["points"] for key, value in stages.items()},
                  "stage_identity": {key: {"directory": value["directory"], "checkpoint_sha256": value["metadata"]["checkpoint_sha256"],
                                           "raw_hash_binding": value["raw_hash_binding"]} for key, value in stages.items()},
                  "comparisons": comparison, "mask_metrics": masks,
                  "input_and_source_evidence": list(evidence.values()), "elapsed_seconds": time.perf_counter() - started,
                  "limitations": ["VAL-derived operating points are not independent test accuracy",
                                  "Single original view results do not establish four-view fusion or local D405 tracking accuracy",
                                  "No old FP16 mask metrics are inferred from box predictions"]}
        atomic_json(output / "summary.json", result)
        progress("complete", summary_path=str(output / "summary.json"), summary_sha256=sha(output / "summary.json"),
                 mask_metrics_phase=masks["phase"])
        print(json.dumps({"phase": "complete", "summary": str(output / "summary.json"),
                          "points": candidate["points"], "mask_metrics_phase": masks["phase"]}, ensure_ascii=False), flush=True)
        return 0
    except BaseException as error:
        progress("failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
