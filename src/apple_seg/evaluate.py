"""Complete, identity-checked PApple VAL evaluation using the query policy.

The public entry point has no subset or confidence-override option. It binds
the original 814-image/3449-instance/13-empty-image provenance, checks actual
CPU annotation pixels and RGB bytes, and selects one BOX confidence on all VAL.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_SHA256 = "fcd96d08c75e1663c83efecb8eb321bbc10220dc74314c8227f1f8565a2ccb3a"
METRICS_PATH = ROOT / "reference/local_amodal/scripts/metrics.py"
POSTPROCESS_PATH = ROOT / "reference/local_amodal/temp/query_dual_mask_candidate_20261021/query_dual_postprocess.py"
REFERENCE_PINS = {
    METRICS_PATH: "1b0a7e385940fd0c6160cc08b75ddaf2e1f43f3ea00f5bd37d547fea6b08010d",
    POSTPROCESS_PATH: "d19cedb68be01e68957acff0026b3c50a6c6e4d7fe2bf176b317e37057cfbf9d",
}
OUTPUT_FILES = ("raw_predictions.json", "mask_pair_diagnostics.json", "box_threshold_curve.json",
                "occlusion_diagnostics.json", "evaluation_metadata.json", "summary.json")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _atomic_json(path, value):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    return repr(value)


def _contract_from_provenance(provenance, *, official=False):
    """Structural validation, also usable by explicit small CPU test fixtures."""
    scope = provenance.get("scope", {})
    _require(provenance.get("phase") == "complete" and provenance.get("passed") is True
             and scope.get("split") == "val", "Completed VAL provenance required")
    images, instances = provenance.get("image_metadata", []), provenance.get("ground_truth_instances", [])
    _require(len(images) == scope.get("images") and len(instances) == scope.get("GT_instances")
             and len({row["image_id"] for row in images}) == len(images), "Full provenance inventory differs")
    if official:
        _require((len(images), len(instances), scope.get("empty_GT_images")) == (814, 3449, 13),
                 "Full PApple VAL must retain 814 images, 3449 GT and 13 empty images")
    truth = {row["image_id"]: [] for row in images}
    for row in instances:
        _require(row["image_id"] in truth, "GT references an unknown image")
        truth[row["image_id"]].append(row)
    for index, image in enumerate(images):
        rows = truth[image["image_id"]]
        _require(image["image_index"] == index and image["gt_count"] == len(rows)
                 and [row["gt_index"] for row in rows] == list(range(len(rows)))
                 and [row["apple_id"] for row in rows] == image["apple_ids_in_target_order"],
                 "Original image/GT identity and order differ")
        _require(not official or [image["height"], image["width"]] == [1300, 1300],
                 "Original full-VAL dimensions differ")
    _require(sum(not row["gt_count"] for row in images) == scope.get("empty_GT_images"),
             "Empty-GT inventory differs")
    strata = Counter(row["occlusion_stratum"] for row in instances)
    annotation = provenance["annotation_diagnostics_GT_only"]
    _require(dict(strata) == annotation["strata_counts"], "Original occlusion inventory differs")
    if official:
        _require(annotation.get("instances_visible_area_exceeds_amodal") == 14
                 and strata["annotation_visible_area_exceeds_amodal"] == 14
                 and annotation.get("no_clipping_repair_or_GT_exclusion") is True,
                 "All 14 original area anomalies must remain included")
    return {"provenance": provenance, "images": images, "truth": truth,
            "official_full_val": official, "scope": scope, "strata": dict(strata)}


def load_papple_contract(provenance_path):
    """Accept only the exact original full-VAL provenance bytes."""
    path = Path(provenance_path).expanduser().absolute()
    _require(_sha(path) == PROVENANCE_SHA256, "PApple provenance SHA256 differs")
    provenance = json.loads(path.read_text(encoding="utf-8-sig"))
    contract = _contract_from_provenance(provenance, official=True)
    contract.update(provenance_path=path, provenance_sha256=PROVENANCE_SHA256)
    return contract


def _nonzero_box(mask):
    # Row/column reductions avoid allocating one coordinate pair per positive
    # original-resolution pixel for large or nearly full-frame masks.
    xs, ys = np.flatnonzero(np.any(mask, axis=0)), np.flatnonzero(np.any(mask, axis=1))
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _audit_target(target, frozen, rows):
    """Recompute every GT byte hash, area and box from actual original masks."""
    source = target.get("source", {})
    _require(source.get("dataset") == "PApple" and source.get("split") == "val"
             and source.get("image_id") == frozen["image_id"], "Target source/image identity differs")
    _require(not source.get("horizontal_flip", False), "VAL cannot use a flipped image, including empty-GT frames")
    if "crop_xyxy" in source:
        _require(source["crop_xyxy"] == [0, 0, frozen["width"], frozen["height"]],
                 "VAL source must identify the full original frame")
    validity = target.get("annotation_validity", {})
    _require(isinstance(validity, dict) and set(validity) == {"visible", "amodal"}
             and validity["visible"] is True and validity["amodal"] is True,
             "Complete real visible/amodal VAL annotations required")
    count, hw = len(rows), [frozen["height"], frozen["width"]]
    labels = target.get("labels")
    _require(isinstance(labels, torch.Tensor) and labels.device.type == "cpu" and labels.dtype == torch.int64
             and tuple(labels.shape) == (count,) and bool((labels == 0).all()), "Original labels must be CPU int64[N] apple=0")
    _require(target.get("instance_ids") == frozen["apple_ids_in_target_order"]
             and list(target.get("original_hw", [])) == hw, "Original GT IDs/order/dimensions differ")
    arrays = {}
    for key in ("visible_masks", "amodal_masks"):
        mask = target.get(key)
        _require(isinstance(mask, torch.Tensor) and mask.device.type == "cpu"
                 and mask.dtype in (torch.uint8, torch.bool) and not mask.requires_grad
                 and tuple(mask.shape) == (count, *hw), "Actual original CPU binary V/A masks required")
        arrays[key] = mask.detach().numpy()
    boxes, visible_bounds = [], []
    for g, row in enumerate(rows):
        for key, prefix in (("visible_masks", "visible"), ("amodal_masks", "amodal")):
            original = arrays[key][g]
            _require(not original.size or int(original.max()) <= 1, "GT masks must be binary 0/1")
            # One original instance at a time; bool is canonicalized to its
            # original uint8 0/1 representation for the frozen byte hash.
            canonical = np.ascontiguousarray(original, dtype=np.uint8)
            _require(hashlib.sha256(memoryview(canonical)).hexdigest() == row[prefix + "_mask_uint8_sha256"]
                     and int(np.count_nonzero(original)) == row[prefix + "_area_pixels"],
                     "Original GT mask SHA256/area differs")
        a_mask, v_mask = arrays["amodal_masks"][g], arrays["visible_masks"][g]
        box = _nonzero_box(a_mask)
        _require(box is not None and box == row["box"], "Original A-derived max+1 GT box differs")
        outside = int(np.count_nonzero(v_mask & (a_mask == 0)))
        ratio = row["visible_area_pixels"] / row["amodal_area_pixels"]
        _require(outside == row["visible_pixels_outside_amodal"] and ratio == row["visible_area_fraction"],
                 "Original GT occlusion values differ")
        boxes.append(box)
        visible_bounds.append(_nonzero_box(v_mask))
    if "boxes" in target:
        supplied = target["boxes"]
        supplied = supplied.detach().cpu().tolist() if isinstance(supplied, torch.Tensor) else supplied
        _require(supplied == boxes, "Supplied GT boxes differ from original A support")
    return {**arrays, "boxes": boxes, "visible_bounds": visible_bounds}


def _rgb_path(target):
    source = target["source"]
    path = next((source[key] for key in ("rgb_path", "RGB_path", "image_path") if key in source), None)
    _require(path is not None, "Target source must identify the actual consumed rgb_path")
    return Path(path).expanduser().absolute()


def _audit_rgb(path, frozen):
    _require(path.stat().st_size == frozen["RGB_bytes"] and _sha(path) == frozen["RGB_sha256"],
             "Actual original RGB bytes/SHA256 differ")
    return frozen["RGB_sha256"]


def _mask_overlaps(outputs, prediction, truth, hw, post):
    scores, queries = prediction["scores"].tolist(), prediction["query_indices"].tolist()
    matrices = {}
    for branch, output_key, mask_key, bound_key in (
        ("visible_mask", "visible_masks_queries_logits", "visible_masks", "visible_bounds"),
        ("amodal_mask", "amodal_masks_queries_logits", "amodal_masks", None),
    ):
        masks = truth[mask_key]
        bounds = truth[bound_key] if bound_key else truth["boxes"]
        areas = np.count_nonzero(masks, axis=(1, 2))
        overlaps = np.zeros((len(scores), len(masks)), dtype=np.float64)
        for start in range(0, len(queries), post.MASK_CHUNK_SIZE):
            chunk = post._binary_chunk(outputs[output_key], queries[start:start + post.MASK_CHUNK_SIZE], hw, torch, F.interpolate)
            for offset, tensor in enumerate(chunk):
                predicted = tensor.numpy()
                area, pred_box = int(np.count_nonzero(predicted)), _nonzero_box(predicted)
                if pred_box is None:
                    continue
                for g, gt_box in enumerate(bounds):
                    if gt_box is None:
                        continue
                    x0, y0 = int(max(pred_box[0], gt_box[0])), int(max(pred_box[1], gt_box[1]))
                    x1, y1 = int(min(pred_box[2], gt_box[2])), int(min(pred_box[3], gt_box[3]))
                    intersection = int(np.count_nonzero(predicted[y0:y1, x0:x1] & masks[g, y0:y1, x0:x1])) if x1 > x0 and y1 > y0 else 0
                    union = area + int(areas[g]) - intersection
                    overlaps[start + offset, g] = intersection / union if union else 0.0
            del chunk
        matrices[branch] = overlaps
    return matrices


def _greedy_assignment(overlaps, scores, threshold=0.5):
    used = np.zeros(overlaps.shape[1], dtype=bool)
    assigned = np.full(len(scores), -1, dtype=np.int64)
    for p in np.argsort(-np.asarray(scores), kind="stable"):
        if not len(used):
            break
        available = np.where(~used, overlaps[p], -1.0)
        g = int(np.argmax(available))
        if available[g] >= threshold:
            used[g] = True
            assigned[p] = g
    return assigned.tolist()


def _mask_pr(diagnostics, branch, confidence, gt_count, metrics):
    tp = sum(g >= 0 and score >= confidence for row in diagnostics
             for g, score in zip(row[branch + "_gt_indices_by_prediction"], row["scores"], strict=True))
    count = sum(score >= confidence for row in diagnostics for score in row["scores"])
    result = metrics._counts(tp, count - tp, gt_count, confidence, .5)
    result.update(mask_iou_threshold=.5, is_mask_ap=False,
                  threshold_policy="Same full-VAL selected BOX confidence; branch independently matched at mask IoU>=0.5")
    return result


def _envelope(curve):
    high_p, high_r = [row for row in curve if row["precision"] >= .98], [row for row in curve if row["recall"] >= .95]
    return {"max_recall_at_precision_ge_0_98": max(high_p, key=lambda row: (row["recall"], row["precision"], row["confidence"])) if high_p else None,
            "max_precision_at_recall_ge_0_95": max(high_r, key=lambda row: (row["precision"], row["recall"], row["confidence"])) if high_r else None,
            "both_at_one_threshold": any(row["precision"] >= .98 and row["recall"] >= .95 for row in curve),
            "interpretation": "Distinct attainable endpoints with complete P/R/confidence tuples; never combine thresholds."}


def _occlusion(diagnostics, contract, confidence):
    groups = {name: {"GT": count, "box_TP": 0, "visible_mask_TP": 0, "amodal_mask_TP": 0}
              for name, count in contract["strata"].items()}
    per_gt = []
    for diagnostic in diagnostics:
        matched = {branch: {g for g, score in zip(diagnostic[branch + "_gt_indices_by_prediction"], diagnostic["scores"], strict=True)
                            if g >= 0 and score >= confidence} for branch in ("box", "visible_mask", "amodal_mask")}
        for row in contract["truth"][diagnostic["image_id"]]:
            flags = {branch + "_recalled": row["gt_index"] in matched[branch] for branch in matched}
            for branch in matched:
                groups[row["occlusion_stratum"]][branch + "_TP"] += flags[branch + "_recalled"]
            per_gt.append({**row, **flags})
    for group in groups.values():
        for branch in ("box", "visible_mask", "amodal_mask"):
            group[branch + "_recall"] = group[branch + "_TP"] / group["GT"] if group["GT"] else 0.0
    return {"confidence": confidence, "iou_threshold": .5, "groups": groups, "ground_truth_instances": per_gt,
            "no_GT_repair_or_exclusion": True, "interpretation": "GT-stratum recall; FP has no GT occlusion group."}


def _source_snapshot(model, dataset, contract):
    paths = [Path(__file__).resolve(), *REFERENCE_PINS, contract["provenance_path"]]
    for value in (model, dataset):
        try:
            path = Path(inspect.getfile(type(value))).absolute()
            if path.is_file():
                paths.append(path)
        except (TypeError, OSError):
            pass
    annotation_paths = getattr(dataset, "annotation_paths", None)
    if annotation_paths is not None:
        _require(isinstance(annotation_paths, dict) and set(annotation_paths) == {"visible", "amodal"},
                 "annotation_paths must identify visible and amodal annotation JSON")
        for key, path in annotation_paths.items():
            _require(_sha(path) == contract["provenance"]["binding"]["current_annotation_sha256"][key],
                     "Actual annotation JSON SHA256 differs")
            paths.append(Path(path).absolute())
    return {str(path): _sha(path) for path in dict.fromkeys(paths)}


def _evaluate_with_contract(model, dataset, device, output_dir, contract):
    """Evaluation engine; public production calls always use the pinned contract.

    Explicit tiny unit contracts remain marked ineligible for model selection.
    No CLI or production option exposes this fixture path.
    """
    _require(len(dataset) == len(contract["images"]), "Full VAL dataset length differs; subsets cannot select a model")
    _require(getattr(dataset, "annotation_sha256", None) == contract["provenance"]["binding"]["current_annotation_sha256"],
             "Dataset annotation identity differs from frozen provenance")
    configuration = getattr(dataset, "configuration", {})
    _require(not bool(getattr(dataset, "augment", False)), "VAL dataset augmentation must be disabled")
    _require(not any(configuration.get(key, False) for key in ("augment", "padding", "external_padding", "crop")),
             "VAL must use an unaugmented, unpadded whole frame")
    for path, expected in REFERENCE_PINS.items():
        _require(_sha(path) == expected, "Versioned reference metric/postprocess source changed")
    metrics = _module(METRICS_PATH, "_apple_eval_reference_metrics")
    post = _module(POSTPROCESS_PATH, "_apple_eval_reference_postprocess")
    before = _source_snapshot(model, dataset, contract)
    output_dir = Path(output_dir).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    _require(not any((output_dir / name).exists() for name in OUTPUT_FILES), "Use a fresh evaluation directory")
    metadata_path = output_dir / "evaluation_metadata.json"
    metadata = {"schema": "apple_seg_complete_papple_val_evaluation_v1", "phase": "running",
                "created_utc": datetime.now(timezone.utc).isoformat(), "dataset_split": "val",
                "official_full_val": contract["official_full_val"], "model_selection_eligible": contract["official_full_val"],
                "device": str(device), "model_initialization_provenance": _jsonable(getattr(model, "checkpoint_provenance", None)),
                "images": len(contract["images"]), "GT_instances": sum(len(rows) for rows in contract["truth"].values()),
                "empty_GT_images": sum(not rows for rows in contract["truth"].values()),
                "provenance": {"path": str(contract["provenance_path"]), "sha256": contract["provenance_sha256"]},
                "preprocess": _jsonable(configuration), "postprocess_policy": post.POSTPROCESS_POLICY,
                "source_sha256_before": before, "no_GT_repair_or_exclusion": True,
                "subset_filter": False, "test_data_read": False, "iou_threshold": .5,
                "mask_PR": "Independently computed exact visible/amodal mask IoU>=.5 matching; same BOX threshold; not AP",
                "annotation_source_files_hashed": getattr(dataset, "annotation_paths", None) is not None}
    _atomic_json(metadata_path, metadata)
    records, diagnostics, rgb_records, gt_identity = [], [], [], []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for index, frozen in enumerate(contract["images"]):
                item = dataset[index]
                _require(isinstance(item, dict) and "target" in item, "Dataset item must contain the original target")
                target, rows = item["target"], contract["truth"][frozen["image_id"]]
                truth = _audit_target(target, frozen, rows)
                rgb = _rgb_path(target)
                rgb_before = _audit_rgb(rgb, frozen)
                source_rgb_sha = target["source"].get("rgb_sha256", target["source"].get("RGB_sha256"))
                _require(source_rgb_sha is None or source_rgb_sha == rgb_before, "Dataset-consumed RGB fingerprint differs")
                pixels, pixel_mask = item.get("pixel_values"), item.get("pixel_mask")
                _require(isinstance(pixels, torch.Tensor) and pixels.is_floating_point(), "Floating pixel_values required")
                pixels = pixels.unsqueeze(0) if pixels.ndim == 3 else pixels
                _require(pixels.ndim == 4 and tuple(pixels.shape[:2]) == (1, 3)
                         and bool(torch.isfinite(pixels).all()), "Single whole-frame RGB tensor required")
                if pixel_mask is not None:
                    pixel_mask = pixel_mask.unsqueeze(0) if pixel_mask.ndim == 2 else pixel_mask
                    _require(tuple(pixel_mask.shape) == (1, *pixels.shape[-2:]) and bool((pixel_mask == 1).all()),
                             "Only full-frame all-one pixel_mask is supported")
                    pixel_mask = pixel_mask.to(device=device)
                outputs = model(pixels.to(device=device), pixel_mask=pixel_mask)
                _require(isinstance(outputs, dict) and not any(key in outputs for key in ("auxiliary_logits", "auxiliary_outputs", "amodal_boxes")),
                         "Require final-only query V/A outputs without a direct box head")
                hw = (frozen["height"], frozen["width"])
                prediction = post.postprocess_single_frame(outputs, hw, return_masks=False)
                scores, boxes = prediction["scores"].tolist(), prediction["boxes"].tolist()
                record = {"image_id": frozen["image_id"], "gt": {"boxes": truth["boxes"]}, "pred": {"boxes": boxes, "scores": scores}}
                match = metrics.match_image(truth["boxes"], boxes, scores, .5, 0.0)
                box_assignment = [-1] * len(scores)
                for p, g in zip(match["pred_indices"], match["gt_indices"], strict=True):
                    box_assignment[int(p)] = int(g)
                matrices = _mask_overlaps(outputs, prediction, truth, hw, post)
                diagnostic = {"image_id": frozen["image_id"], "gt_count": len(rows),
                              "apple_ids_in_target_order": target["instance_ids"], "scores": scores,
                              "query_indices": prediction["query_indices"].tolist(), "labels": prediction["labels"].tolist(),
                              "box_gt_indices_by_prediction": box_assignment, "postprocess": prediction["metadata"]}
                for branch, matrix in matrices.items():
                    assigned = _greedy_assignment(matrix, scores)
                    diagnostic[branch + "_gt_indices_by_prediction"] = assigned
                    diagnostic[branch + "_tp_flags"] = [g >= 0 for g in assigned]
                # Reaudit after the model call: neither model nor surrounding
                # hooks may mutate GT, IDs, RGB bytes or frozen source files.
                _audit_target(target, frozen, rows)
                rgb_after = _audit_rgb(rgb, frozen)
                rgb_records.append({"image_id": frozen["image_id"], "path": str(rgb), "before": rgb_before, "after": rgb_after})
                gt_identity.append({"image_id": frozen["image_id"], "instance_ids": list(target["instance_ids"]),
                                    "original_hw": list(target["original_hw"]), "GT": rows, "source": _jsonable(target["source"])})
                records.append(record)
                diagnostics.append(diagnostic)
                del outputs, matrices, truth, item, target, prediction
        calibrated = metrics.calibrate_threshold(records, .5, return_curve=True)
        confidence = calibrated["confidence"]
        replay = metrics.evaluate_at_threshold(records, confidence, .5)
        _require(replay == calibrated["summary"], "Selected BOX threshold replay differs")
        _require(replay["images"] == metadata["images"] and replay["gt_count"] == metadata["GT_instances"]
                 and replay["empty_gt_images"] == metadata["empty_GT_images"], "Consumed full scope differs")
        box_summary = {key: value for key, value in replay.items()
                       if key not in ("amodal_mask_iou_on_box_tp", "visible_mask_iou_on_box_tp")}
        for branch in ("visible", "amodal"):
            box_summary[branch + "_instance_mask_pr_iou50"] = _mask_pr(diagnostics, branch + "_mask", confidence, replay["gt_count"], metrics)
        # Recheck all consumed RGBs at the end as well: a late frame must not
        # mutate an earlier image after that image's local before/after audit.
        for saved, frozen in zip(rgb_records, contract["images"], strict=True):
            saved["after"] = _audit_rgb(Path(saved["path"]), frozen)
        after = {path: _sha(path) for path in before}
        _require(before == after, "Consumed source/provenance/annotation files changed during evaluation")
        metadata.update(phase="complete", consumed_images=len(records), consumed_GT=replay["gt_count"],
                        source_sha256_after=after, source_before_after_identical=True,
                        RGB_sha256_before_after=rgb_records, all_RGB_before_after_verified=True,
                        all_original_GT_numeric_fields_and_mask_SHA256_verified=True,
                        ground_truth_identity=gt_identity, annotation_visible_area_exceeds_amodal_GT_retained=14 if contract["official_full_val"] else None)
        summary = {"schema": "apple_seg_papple_val_summary_v1", "phase": "complete",
                   "confidence": confidence, "score": min(replay["precision"], replay["recall"]) if contract["official_full_val"] else None,
                   "model_selection_eligible": contract["official_full_val"], "summary": box_summary,
                   "inference": {"dataset_split": "val", "preprocess": _jsonable(configuration), "postprocess_policy": post.POSTPROCESS_POLICY},
                   "calibration": {"selection_rule": calibrated["selection_rule"], "candidate_thresholds": calibrated["candidate_thresholds"],
                                   "scope": "All original VAL frames; complete distinct raw-score curve with whole score ties; no TEST tuning"},
                   "BOX_threshold_envelope": _envelope(calibrated["curve"]),
                   "files": {name.removesuffix(".json"): str(output_dir / name) for name in OUTPUT_FILES}}
        _atomic_json(output_dir / "raw_predictions.json", records)
        _atomic_json(output_dir / "mask_pair_diagnostics.json", diagnostics)
        _atomic_json(output_dir / "box_threshold_curve.json", calibrated["curve"])
        _atomic_json(output_dir / "occlusion_diagnostics.json", _occlusion(diagnostics, contract, confidence))
        _atomic_json(metadata_path, metadata)
        _atomic_json(output_dir / "summary.json", summary)  # Completion marker written last.
        return summary
    except Exception as error:
        metadata.update(phase="failed", failure_type=type(error).__name__, failure=str(error), consumed_images=len(records))
        _atomic_json(metadata_path, metadata)
        raise
    finally:
        model.train(was_training)


def evaluate_papple(model, dataset, device, output_dir, provenance_path):
    """Run actual full PApple VAL; no loader/model construction or partial score.

    ``dataset[i]`` supplies pixel_values, pixel_mask and a genuine original
    CPU target. Its ``annotation_sha256`` must bind both original annotations;
    optional ``annotation_paths`` are also hashed before/after. Target source
    identifies the consumed RGB via rgb_path. The model is already on device.
    This method restores its previous training/eval mode, including on failure.
    """
    contract = load_papple_contract(provenance_path)
    return _evaluate_with_contract(model, dataset, device, output_dir, contract)
