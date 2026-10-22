"""Full, provenance-bound PApple VAL evaluation for the shared-query candidate.

Import is standard-library-only. evaluate_full_papple is the sole runtime entry:
it accepts the new whole-frame adapter and never constructs a loader/model or
reads TEST. One image's GT and <=8 CPU boolean predicted masks per branch are
live at a time; only boxes, scores, IDs and scalar diagnostics survive a frame.
No runtime, quality result or 8 GB fit is implied by the existence of this file.
"""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import time
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_PATH = ROOT / "reports" / "papple_val_provenance_20261021.json"
PROVENANCE_SHA256 = "fcd96d08c75e1663c83efecb8eb321bbc10220dc74314c8227f1f8565a2ccb3a"
SOURCE_SHA256 = {
    "scripts/evaluate.py": "a42cd063ffe6aedfd28d2807b800fc368b0782b3073556cba2408d8159236492",
    "scripts/metrics.py": "1b0a7e385940fd0c6160cc08b75ddaf2e1f43f3ea00f5bd37d547fea6b08010d",
    "scripts/atomic_io.py": "3b5ca14efc337ce3f511f60f1829c14c24e5ab6fc9772714e051bf265e645380",
    "scripts/dataset.py": "4d8fe0f1ac98e3b54dd4f029675ad240b621766140b259a537e99290ace9a0a0",
    "temp/query_dual_mask_candidate_20261021/query_papple_data.py": "1fbb142f4710454be90833bdacc3ff4841c9339ca1d5332897f743da36be96df",
    "temp/query_dual_mask_candidate_20261021/query_dual_postprocess.py": "d19cedb68be01e68957acff0026b3c50a6c6e4d7fe2bf176b317e37057cfbf9d",
    "temp/query_dual_mask_candidate_20261021/query_dual_mask.py": "cef59347445af673eceff65cd5d0ed7caa5769683c5e78eb7f4610deb2e48878",
    "temp/query_dual_mask_candidate_20261021/query_dual_criterion.py": "6ceec79c357b0e30864dfd47e0113b574ce7b65eecd24c40ae193bdf9e5b74ee",
}
MASK_HELPERS = ("_nonzero_box", "_overlaps_one_mask", "_greedy_mask_flags",
                "_mask_pr", "_pair_summary")
IMAGE_COUNT, GT_COUNT, EMPTY_IMAGE_COUNT = 814, 3449, 13
ORIGINAL_HW = (1300, 1300)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_module(relative, *, names=None, numpy=None):
    """Compile the verified bytes, avoiding stale pyc or import-path aliases."""
    path = ROOT / relative
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256[relative]:
        raise ValueError(f"Pinned evaluation dependency changed: {path}")
    module = ModuleType("_query_evaluation_" + path.stem)
    module.__file__ = str(path)
    if names is None:
        code = compile(source, str(path), "exec")
    else:
        tree = ast.parse(source, filename=str(path))
        nodes = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in nodes} != set(names):
            raise ValueError("Pinned mask helper set is incomplete")
        module.__dict__["np"] = numpy
        code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")
    exec(code, module.__dict__)
    return module


def _atomic_json(path, value, atomic):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = Path(stream.name)
    # A failed bounded replace preserves the completed temporary and old target.
    atomic.replace_with_retry(temporary, path)


def _source_snapshot(provenance):
    expected = {str(ROOT / name): digest for name, digest in SOURCE_SHA256.items()}
    expected[str(PROVENANCE_PATH)] = PROVENANCE_SHA256
    base = ROOT / "data" / "papple" / "data" / "gt_json" / "val"
    for branch, filename in (("visible", "via_region_data_instance.json"),
                             ("amodal", "via_region_data_amodal.json")):
        expected[str(base / filename)] = provenance["binding"]["current_annotation_sha256"][branch]
    result = {path: _sha256(path) for path in expected}
    if result != expected:
        changed = [path for path in expected if result[path] != expected[path]]
        raise ValueError(f"Consumed source/VAL annotation identity mismatch: {changed}")
    result[str(Path(__file__).resolve())] = _sha256(__file__)
    return result


def _validate_scope(dataset, provenance):
    config = getattr(dataset, "configuration", {})
    if (config.get("split") != "val" or config.get("input_size") not in (512, 768)
            or config.get("batch_size") != 1 or config.get("augment") is not False
            or config.get("external_padding") is not False):
        raise ValueError("Require the new unaugmented, unpadded 512/768 whole-frame VAL adapter")
    base = getattr(dataset, "_base", None)
    if (base is None or base.split != "val" or base.augment
            or Path(base.root).resolve() != (ROOT / "data" / "papple" / "data").resolve()):
        raise ValueError("Adapter must bind the genuine complete local VAL source")
    if dataset.annotation_sha256 != provenance["binding"]["current_annotation_sha256"]:
        raise ValueError("Adapter annotation snapshot does not match frozen VAL provenance")
    images = provenance["image_metadata"]
    instances = provenance["ground_truth_instances"]
    scope = provenance["scope"]
    if (provenance.get("phase") != "complete" or provenance.get("passed") is not True
            or scope.get("split") != "val" or scope.get("images") != IMAGE_COUNT
            or scope.get("GT_instances") != GT_COUNT or scope.get("empty_GT_images") != EMPTY_IMAGE_COUNT
            or len(dataset) != IMAGE_COUNT or len(base.samples) != IMAGE_COUNT
            or len(images) != IMAGE_COUNT or len(instances) != GT_COUNT):
        raise ValueError("Full VAL scope must retain 814 images, 3449 GT, and 13 empty images")
    by_image = {entry["image_id"]: [] for entry in images}
    for gt in instances:
        by_image[gt["image_id"]].append(gt)
    for index, frozen in enumerate(images):
        sample, info = base.samples[index], dataset.image_metadata(index)
        rows = by_image[frozen["image_id"]]
        if (frozen["image_index"] != index or sample["filename"] != frozen["image_id"]
                or info["filename"] != frozen["image_id"]
                or list(info["apple_ids"]) != frozen["apple_ids_in_target_order"]
                or list(sample["apple_ids"]) != frozen["apple_ids_in_target_order"]
                or tuple(info["original_size"]) != ORIGINAL_HW
                or tuple(info["input_size"]) != (config["input_size"],) * 2
                or Path(sample["path"]).resolve() != Path(frozen["RGB_path"]).resolve()
                or frozen["gt_count"] != len(rows)
                or [row["gt_index"] for row in rows] != list(range(len(rows)))
                or [row["apple_id"] for row in rows] != frozen["apple_ids_in_target_order"]):
            raise ValueError(f"Original VAL image/ID/GT order changed at index {index}")
    if sum(not entry["gt_count"] for entry in images) != EMPTY_IMAGE_COUNT:
        raise ValueError("Empty VAL image scope changed")
    return images, by_image


def _original_arrays(item, frozen, truth_rows, index, np):
    """Audit every original GT pixel/number against the frozen true-mask report."""
    if (list(item["apple_ids"]) != frozen["apple_ids_in_target_order"]
            or item["metadata"]["filename"] != frozen["image_id"]):
        raise ValueError("Item identity/order differs from the audited adapter snapshot")
    truth = {}
    for key in ("boxes", "labels", "masks", "visible_masks", "image_id", "area", "iscrowd"):
        tensor = item["original_target"][key]
        if tensor.device.type != "cpu":
            raise ValueError("Untouched original GT must remain on CPU")
        truth[key] = tensor.detach().numpy()
    n = len(truth_rows)
    expected_boxes = np.asarray([row["box"] for row in truth_rows], dtype=np.float32).reshape(-1, 4)
    if (not np.array_equal(truth["boxes"], expected_boxes)
            or truth["labels"].dtype != np.int64 or truth["labels"].shape != (n,)
            or not np.all(truth["labels"] == 1)
            or not np.array_equal(truth["image_id"], [index])
            or not np.array_equal(truth["area"], [row["amodal_area_pixels"] for row in truth_rows])
            or truth["iscrowd"].shape != (n,) or np.any(truth["iscrowd"])):
        raise ValueError(f"Original GT numeric fields changed: {frozen['image_id']}")
    for branch, digest_key, area_key in (
            ("masks", "amodal_mask_uint8_sha256", "amodal_area_pixels"),
            ("visible_masks", "visible_mask_uint8_sha256", "visible_area_pixels")):
        masks = truth[branch]
        if masks.dtype != np.uint8 or masks.shape != (n, *ORIGINAL_HW):
            raise ValueError("GT masks must be untouched uint8[N,1300,1300]")
        for gt_index, row in enumerate(truth_rows):
            mask = masks[gt_index]
            if (not mask.flags.c_contiguous or hashlib.sha256(memoryview(mask)).hexdigest() != row[digest_key]
                    or int(np.count_nonzero(mask)) != row[area_key]):
                raise ValueError(f"Original GT mask bytes/area changed: {frozen['image_id']}/{gt_index}/{branch}")
    for gt_index, row in enumerate(truth_rows):
        outside = int(np.count_nonzero(truth["visible_masks"][gt_index] & (truth["masks"][gt_index] == 0)))
        ratio = row["visible_area_pixels"] / row["amodal_area_pixels"]
        if outside != row["visible_pixels_outside_amodal"] or ratio != row["visible_area_fraction"]:
            raise ValueError("Frozen occlusion numeric fields differ from untouched GT")
    return truth


def _mask_assignment(overlaps, scores, helpers, np):
    """Same stable greedy math as _greedy_mask_flags, additionally retain GT IDs."""
    used = np.zeros(overlaps.shape[1], dtype=bool)
    assigned = np.full(len(scores), -1, dtype=np.int64)
    for prediction in np.argsort(-np.asarray(scores), kind="stable"):
        if not len(used):
            break
        available = np.where(~used, overlaps[prediction], -1.0)
        target = int(np.argmax(available))
        if available[target] >= 0.5:
            assigned[prediction] = target
            used[target] = True
    if not np.array_equal(assigned >= 0, helpers._greedy_mask_flags(overlaps, scores)):
        raise AssertionError("Mask assignments disagree with the pinned independent-mask helper")
    return assigned


def _frame_diagnostics(outputs, item, frozen, rows, index, post, metrics, helpers, torch, np):
    truth = _original_arrays(item, frozen, rows, index, np)
    prediction = post.postprocess_single_frame(outputs, ORIGINAL_HW, return_masks=False)
    boxes, scores = prediction["boxes"].numpy(), prediction["scores"].numpy()
    if prediction["labels"].tolist() != [1] * len(scores):
        raise ValueError("Query outputs must explicitly map to original apple label 1")
    # Keep precisely the existing calibrate_threshold record schema, GT order intact.
    raw = {"image_id": frozen["image_id"], "gt": {"boxes": truth["boxes"].tolist()},
           "pred": {"boxes": boxes.tolist(), "scores": scores.tolist()}}
    box_match = metrics.match_image(truth["boxes"], boxes, scores, 0.5, 0.0)
    matrices, assignments = {}, {}
    query_indices = prediction["query_indices"].tolist()
    for branch, output_key in (("visible_masks", "visible_masks_queries_logits"),
                               ("masks", "amodal_masks_queries_logits")):
        gt_masks = truth[branch].astype(bool, copy=False)
        areas = np.count_nonzero(gt_masks, axis=(1, 2))
        bounds = [helpers._nonzero_box(mask) for mask in gt_masks]
        overlaps = np.zeros((len(scores), len(rows)), dtype=np.float64)
        for start in range(0, len(scores), post.MASK_CHUNK_SIZE):
            chunk = post._binary_chunk(outputs[output_key], query_indices[start:start + post.MASK_CHUNK_SIZE],
                                       ORIGINAL_HW, torch, torch.nn.functional.interpolate)
            for offset in range(len(chunk)):
                overlaps[start + offset] = helpers._overlaps_one_mask(chunk[offset].numpy(), gt_masks, areas, bounds)
            del chunk
        matrices[branch] = overlaps
        assignments[branch] = _mask_assignment(overlaps, scores, helpers, np)
        del gt_masks, areas, bounds
    pairs = [{"prediction_index": int(p), "gt_index": int(g), "score": float(scores[p]),
              "visible_iou": float(matrices["visible_masks"][p, g]),
              "amodal_iou": float(matrices["masks"][p, g])}
             for p, g in zip(box_match["pred_indices"], box_match["gt_indices"]) if g >= 0]
    box_assignments = [-1] * len(scores)
    for p, g in zip(box_match["pred_indices"], box_match["gt_indices"]):
        box_assignments[int(p)] = int(g)
    diagnostics = {"image_id": frozen["image_id"], "gt_count": len(rows),
                   "apple_ids_in_target_order": frozen["apple_ids_in_target_order"],
                   "scores": scores.tolist(), "labels": prediction["labels"].tolist(),
                   "query_indices": query_indices, "box_tp_mask_pairs": pairs,
                   "box_gt_indices_by_prediction": box_assignments,
                   "visible_mask_gt_indices_by_prediction": assignments["visible_masks"].tolist(),
                   "amodal_mask_gt_indices_by_prediction": assignments["masks"].tolist(),
                   "visible_mask_tp_flags": (assignments["visible_masks"] >= 0).tolist(),
                   "amodal_mask_tp_flags": (assignments["masks"] >= 0).tolist(),
                   "postprocess": prediction["metadata"]}
    return raw, diagnostics


def _occlusion_diagnostics(diagnostics, by_image, confidence, provenance):
    names = provenance["annotation_diagnostics_GT_only"]["strata_counts"]
    groups = {name: {"gt_count": 0, "box_tp": 0, "visible_mask_tp": 0, "amodal_mask_tp": 0}
              for name in names}
    per_gt, seen = [], Counter()
    for record in diagnostics:
        rows = by_image[record["image_id"]]
        matched = {}
        for kind, field in (("box", "box_gt_indices_by_prediction"),
                            ("visible_mask", "visible_mask_gt_indices_by_prediction"),
                            ("amodal_mask", "amodal_mask_gt_indices_by_prediction")):
            matched[kind] = {g for score, g in zip(record["scores"], record[field])
                             if score >= confidence and g >= 0}
        for gt_index, row in enumerate(rows):
            group = groups[row["occlusion_stratum"]]
            group["gt_count"] += 1
            seen[row["occlusion_stratum"]] += 1
            flags = {kind + "_recalled": gt_index in matched[kind] for kind in matched}
            for kind in matched:
                group[kind + "_tp"] += int(flags[kind + "_recalled"])
            per_gt.append({**row, **flags})  # Exact frozen IDs, ratios, anomaly group and mask hashes.
    if dict(seen) != names or len(per_gt) != GT_COUNT:
        raise AssertionError("Occlusion groups do not cover the complete untouched VAL GT")
    for group in groups.values():
        for kind in ("box", "visible_mask", "amodal_mask"):
            group[kind + "_recall"] = group[kind + "_tp"] / group["gt_count"] if group["gt_count"] else 0.0
    return {"schema": "query_dual_full_val_occlusion_recall_v1", "confidence": confidence,
            "iou_threshold": 0.5, "groups": groups, "ground_truth_instances": per_gt,
            "grouping_source": {"path": str(PROVENANCE_PATH), "sha256": PROVENANCE_SHA256},
            "annotation_diagnostics_GT_only": provenance["annotation_diagnostics_GT_only"],
            "interpretation": "GT-stratum recall only. False positives have no GT occlusion group; group precision is undefined.",
            "mask_matching": "Each mask branch independently matched at IoU>=0.5, using the global BOX confidence.",
            "no_GT_repair_or_exclusion": True, "ap_computed": False}


def evaluate_full_papple(model, dataset, device, output_dir) -> dict:
    """Evaluate all 814 VAL frames; calibrate one BOX confidence on full VAL.

    The model must already be loaded on device. dataset is the new adapter,
    input_size=768 or 512. Returns summary/confidence/score/files and occlusion
    diagnostics for future training selection; score is min(box precision,recall).
    No max_images, TEST, threshold override, loader or model construction exists.
    Runtime restores model.training even after failure. Source and every RGB
    byte stream are checked before/after consumption; all GT pixels and original
    metadata must match the frozen provenance before any success is written.
    """
    if _sha256(PROVENANCE_PATH) != PROVENANCE_SHA256:
        raise ValueError("Frozen full VAL provenance changed")
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8-sig"))
    source_before = _source_snapshot(provenance)
    images, by_image = _validate_scope(dataset, provenance)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists():
        raise FileExistsError("Use a fresh evaluation directory; an existing completed summary is preserved")

    # Lazy numerical imports occur only at the explicitly called runtime boundary.
    import numpy as np
    import torch

    metrics = _bound_module("scripts/metrics.py")
    helpers = _bound_module("scripts/evaluate.py", names=MASK_HELPERS, numpy=np)
    atomic = _bound_module("scripts/atomic_io.py")
    post = _bound_module("temp/query_dual_mask_candidate_20261021/query_dual_postprocess.py")
    device = torch.device(device)
    is_cuda = device.type == "cuda"
    config = dataset.configuration
    metadata = {"schema": "query_dual_full_val_evaluation_v1", "phase": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(), "dataset_split": "val",
                "images": IMAGE_COUNT, "GT_instances": GT_COUNT, "empty_GT_images": EMPTY_IMAGE_COUNT,
                "GT_coordinates_hw": list(ORIGINAL_HW), "iou_threshold": 0.5,
                "architecture": "shared-query Mask2Former Swin-Tiny with independent visible/amodal readouts",
                "candidate_metadata_caller_supplied": getattr(model, "candidate_metadata", None),
                "architecture_and_preprocessing_differ_from_old_R50": True,
                "comparison_basis": "Same original GT, original exclusive XYXY coordinates and IoU>=0.5; architecture/preprocessing differ.",
                "preprocess": {"input_hw": [config["input_size"]] * 2,
                               "resize": "Pillow RGB bilinear whole frame 1300 square", "augment": False,
                               "padding": False, "crop": False, "rgb_scale": "uint8/255",
                               "normalization_mean": [0.485, 0.456, 0.406],
                               "normalization_std": [0.229, 0.224, 0.225]},
                "postprocess_policy": post.POSTPROCESS_POLICY,
                "device": str(device), "amp": is_cuda, "amp_dtype": "float16" if is_cuda else None,
                "runtime_versions": {name: version(name) for name in ("torch", "transformers", "numpy", "Pillow")},
                "threshold_policy": "Full VAL only: global BOX max(min(P,R)); tie max(F1); tie higher confidence. Same threshold for independent V/A matching.",
                "split_independence_verified": False, "test_data_read": False, "subset_filter": False,
                "provenance": {"path": str(PROVENANCE_PATH), "sha256": PROVENANCE_SHA256,
                               "binding_limitations": provenance["binding"]["limitation"]},
                "helper_reuse": {"box_and_calibration": "scripts/metrics.py",
                                 "mask_helpers_exact_AST_functions": list(MASK_HELPERS),
                                 "mask_helpers_source": "scripts/evaluate.py",
                                 "mask_assignment": "identical stable greedy math, checked against pinned flags per image/branch"},
                "memory_policy": "One frame GT; <=8 CPU bool predicted masks per branch; scalar-only cross-frame records; no PNG or dense-mask files.",
                "source_sha256_before": source_before, "RGB_sha256_before_after": []}
    _atomic_json(output_dir / "evaluation_metadata.json", metadata, atomic)
    old_training = bool(model.training)
    raw_records, diagnostic_records, latencies = [], [], []
    started = time.perf_counter()
    try:
        model.eval()
        with torch.inference_mode():
            for index, frozen in enumerate(images):
                image_path = Path(frozen["RGB_path"])
                before = _sha256(image_path)
                if before != frozen["RGB_sha256"] or image_path.stat().st_size != frozen["RGB_bytes"]:
                    raise ValueError(f"Frozen VAL RGB bytes changed: {image_path}")
                item = dataset[index]
                size = config["input_size"]
                if (tuple(item["pixel_values"].shape) != (3, size, size)
                        or tuple(item["pixel_mask"].shape) != (size, size)
                        or not bool(item["pixel_mask"].all())
                        or not bool(torch.isfinite(item["pixel_values"]).all())):
                    raise ValueError("Adapter input must be finite normalized complete unpadded frame")
                pixels = item["pixel_values"].unsqueeze(0).to(device)
                pixel_mask = item["pixel_mask"].unsqueeze(0).to(device)
                if is_cuda:
                    torch.cuda.synchronize(device)
                begin = time.perf_counter()
                context = torch.autocast(device_type="cuda", dtype=torch.float16) if is_cuda else nullcontext()
                with context:
                    outputs = model(pixel_values=pixels, pixel_mask=pixel_mask)
                if is_cuda:
                    torch.cuda.synchronize(device)
                latencies.append(time.perf_counter() - begin)
                raw, diagnostics = _frame_diagnostics(outputs, item, frozen, by_image[frozen["image_id"]],
                                                       index, post, metrics, helpers, torch, np)
                after = _sha256(image_path)
                if after != before or image_path.stat().st_size != frozen["RGB_bytes"]:
                    raise ValueError(f"VAL RGB bytes changed during consumption: {image_path}")
                metadata["RGB_sha256_before_after"].append({"image_id": frozen["image_id"],
                                                          "path": str(image_path), "before": before, "after": after})
                raw_records.append(raw)
                diagnostic_records.append(diagnostics)
                del item, pixels, pixel_mask, outputs
                if (index + 1) % 25 == 0 or index + 1 == IMAGE_COUNT:
                    print(f"QUERY FULL VAL {index + 1}/{IMAGE_COUNT} elapsed={time.perf_counter() - started:.1f}s", flush=True)

        # Calibration consumes the exact box-record schema and the complete VAL.
        calibration = metrics.calibrate_threshold(raw_records, iou_threshold=0.5, return_curve=True)
        confidence, summary = calibration["confidence"], calibration["summary"]
        if (summary["images"] != IMAGE_COUNT or summary["gt_count"] != GT_COUNT
                or summary["empty_gt_images"] != EMPTY_IMAGE_COUNT):
            raise AssertionError("Metric numeric scope does not cover the full VAL")
        mask_tp = {"visible": 0, "amodal": 0}
        paired = {"visible": [], "amodal": []}
        kept = 0
        for record in diagnostic_records:
            keep = np.asarray(record["scores"]) >= confidence
            kept += int(keep.sum())
            for branch in mask_tp:
                mask_tp[branch] += int((np.asarray(record[branch + "_mask_tp_flags"], dtype=bool) & keep).sum())
                paired[branch].extend(pair[branch + "_iou"] for pair in record["box_tp_mask_pairs"]
                                      if pair["score"] >= confidence)
        if kept != summary["prediction_count"] or any(len(values) != summary["tp"] for values in paired.values()):
            raise AssertionError("Global box threshold/pair counts disagree")
        for branch in mask_tp:
            summary[branch + "_mask_iou_on_box_tp"] = helpers._pair_summary(paired[branch], summary["tp"])
            summary[branch + "_instance_mask_pr_iou50"] = helpers._mask_pr(mask_tp[branch], kept - mask_tp[branch], GT_COUNT, confidence)
        occlusion = _occlusion_diagnostics(diagnostic_records, by_image, confidence, provenance)
        source_after = _source_snapshot(provenance)
        if source_after != source_before:
            raise ValueError("Consumed source/annotation bytes changed during full evaluation")
        metadata.update(phase="complete", source_sha256_after=source_after,
                        finished_utc=datetime.now(timezone.utc).isoformat(),
                        consumed_images=len(raw_records), consumed_GT=sum(len(row["gt"]["boxes"]) for row in raw_records),
                        all_original_GT_numeric_fields_and_mask_SHA256_verified=True,
                        all_RGB_before_after_verified=True, no_GT_repair_or_exclusion=True)
        files = {"raw_predictions": "raw_predictions.json", "mask_pair_diagnostics": "mask_pair_diagnostics.json",
                 "summary": "summary.json", "occlusion_diagnostics": "occlusion_diagnostics.json",
                 "metadata": "evaluation_metadata.json", "threshold_curve": "threshold_curve.json"}
        result = {"summary": summary, "confidence": confidence,
                  "score": min(summary["precision"], summary["recall"]), "files": files,
                  "threshold_selection": {key: value for key, value in calibration.items() if key not in ("summary", "curve")},
                  "occlusion_diagnostics": {key: value for key, value in occlusion.items() if key != "ground_truth_instances"},
                  "inference": {"architecture": metadata["architecture"], "preprocess": metadata["preprocess"],
                                "postprocess_policy": post.POSTPROCESS_POLICY, "dataset_split": "val"},
                  "prediction_cap_diagnostic": {"cap": 100, "images_reaching_cap": sum(
                      record["postprocess"]["capped_candidate_count"] == 100 for record in diagnostic_records),
                      "empty_amodal_candidates_dropped": sum(len(record["postprocess"]["invalid_candidates"]) for record in diagnostic_records)},
                  "timing": {"total_seconds": time.perf_counter() - started,
                             "model_seconds_per_image_mean": float(np.mean(latencies)),
                             "model_seconds_per_image_median": float(np.median(latencies)),
                             "model_seconds_per_image_p95": float(np.percentile(latencies, 95)),
                             "warmup_excluded": False,
                             "model_timing_excludes": "image loading, H2D copy, provenance hashes, postprocessing and metrics"},
                  "ap_computed": False}
        _atomic_json(output_dir / files["raw_predictions"], raw_records, atomic)
        _atomic_json(output_dir / files["mask_pair_diagnostics"], diagnostic_records, atomic)
        _atomic_json(output_dir / files["occlusion_diagnostics"], occlusion, atomic)
        _atomic_json(output_dir / files["threshold_curve"], calibration["curve"], atomic)
        _atomic_json(output_dir / files["metadata"], metadata, atomic)
        _atomic_json(output_dir / files["summary"], result, atomic)
        return result
    except BaseException as exc:
        metadata.update(phase="failed", error=f"{type(exc).__name__}: {exc}",
                        consumed_images=len(raw_records), consumed_GT=sum(len(row["gt"]["boxes"]) for row in raw_records))
        _atomic_json(output_dir / "evaluation_metadata.json", metadata, atomic)
        raise
    finally:
        model.train(old_training)
