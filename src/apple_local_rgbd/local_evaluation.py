"""Visible-instance evaluation on untouched PApple development rasters.

The custom AP50 here is 101-point interpolation of one-class mask PR, with
confidence-first greedy instance matching. It is not pycocotools COCO mAP.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
import torch

from .local_training import (MANIFEST_SCHEMA, OFFICIAL_SHA256, TrainingContractError,
    build_visible_model, configure_f_runtime, decode_packed_masks, load_local_checkpoint,
    require_f_path, sha256_file)
from .artifact_io import write_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def native_mask_areas_boxes(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact pixel area and half-open nonzero bounds; no resized IoU proxies."""
    if masks.ndim != 3 or masks.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
        raise TrainingContractError("Masks must be binary bool/uint8 N,H,W")
    areas = np.count_nonzero(masks, axis=(1, 2)).astype(np.int64)
    boxes = np.zeros((len(masks), 4), dtype=np.int32)
    for i, mask in enumerate(masks):
        if not areas[i]:
            continue
        yy, xx = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        boxes[i] = (xx[0], yy[0], xx[-1] + 1, yy[-1] + 1)
    return areas, boxes


def native_mask_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape[1:] != gt.shape[1:]:
        raise TrainingContractError("Native prediction/GT mask coordinates differ")
    pa, pb = native_mask_areas_boxes(pred)
    ga, gb = native_mask_areas_boxes(gt)
    out = np.zeros((len(pred), len(gt)), dtype=np.float64)
    for i in range(len(pred)):
        if not pa[i]:
            continue
        for j in range(len(gt)):
            x1, y1 = max(pb[i, 0], gb[j, 0]), max(pb[i, 1], gb[j, 1])
            x2, y2 = min(pb[i, 2], gb[j, 2]), min(pb[i, 3], gb[j, 3])
            if x2 <= x1 or y2 <= y1:
                continue
            intersection = int(np.count_nonzero(np.bitwise_and(pred[i, y1:y2, x1:x2], gt[j, y1:y2, x1:x2])))
            union = int(pa[i] + ga[j] - intersection)
            out[i, j] = intersection / union if union else 0.0
    return out


def box_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred, gt = np.asarray(pred, dtype=np.float64).reshape(-1, 4), np.asarray(gt, dtype=np.float64).reshape(-1, 4)
    area_p = np.prod(np.maximum(pred[:, 2:] - pred[:, :2], 0), axis=1)
    area_g = np.prod(np.maximum(gt[:, 2:] - gt[:, :2], 0), axis=1)
    intersection = np.prod(np.maximum(np.minimum(pred[:, None, 2:], gt[None, :, 2:]) -
                                     np.maximum(pred[:, None, :2], gt[None, :, :2]), 0), axis=2)
    union = area_p[:, None] + area_g[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def greedy_match(scores: np.ndarray, iou: np.ndarray, threshold: float = .5) -> tuple[np.ndarray, np.ndarray]:
    """High-score first; each detection gets best unused GT at inclusive IoU gate.

    Equal-score prediction order is retained; metric threshold groups include
    every equal-score member and cannot select a fractional tie prefix.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if iou.ndim != 2 or len(scores) != iou.shape[0] or not np.isfinite(scores).all() or not np.isfinite(iou).all():
        raise TrainingContractError("Invalid score/IoU matrix")
    matched = np.full(len(scores), -1, dtype=np.int64)
    matched_iou = np.zeros(len(scores), dtype=np.float64)
    used = np.zeros(iou.shape[1], dtype=bool)
    for pi in np.argsort(-scores, kind="stable"):
        possible = np.flatnonzero(~used & (iou[pi] >= threshold))
        if len(possible):
            gi = int(possible[np.argmax(iou[pi, possible])])
            used[gi] = True
            matched[pi], matched_iou[pi] = gi, float(iou[pi, gi])
    return matched, matched_iou


def counts(tp: int, fp: int, n_gt: int, threshold: float | None) -> dict[str, Any]:
    return {"threshold": threshold, "TP": tp, "FP": fp, "FN": n_gt - tp,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / n_gt if n_gt else 0.0}


def confidence_metrics(scores: list[float], true_positive: list[bool], n_gt: int,
                       fixed_threshold: float = .25) -> dict[str, Any]:
    s, tp = np.asarray(scores, dtype=np.float64), np.asarray(true_positive, dtype=bool)
    if len(s) != len(tp) or not np.isfinite(s).all() or int(tp.sum()) > n_gt:
        raise TrainingContractError("Invalid PR input")
    fixed = s >= fixed_threshold
    fixed_counts = counts(int(tp[fixed].sum()), int(fixed.sum() - tp[fixed].sum()), n_gt, fixed_threshold)
    order = np.argsort(-s, kind="stable")
    s, tp = s[order], tp[order]
    if not len(s):
        return {"fixed": fixed_counts, "P_at_R90": None, "maximum_recall": 0.0, "AP50_101point": 0.0,
                "curve": [], "AP_definition": "101 recall samples of interpolated precision; one-class IoU=0.5; not COCO mAP"}
    cumulative_tp = np.cumsum(tp)
    # A deployment threshold must retain the whole tied score group.
    endpoints = np.flatnonzero(np.r_[s[:-1] != s[1:], True])
    curve = [counts(int(cumulative_tp[i]), int(i + 1 - cumulative_tp[i]), n_gt, float(s[i])) for i in endpoints]
    maximum_recall = curve[-1]["recall"]
    choices = [row for row in curve if row["recall"] >= .9]
    best = max(choices, key=lambda row: (row["precision"], row["threshold"])) if choices else None
    precision = np.asarray([row["precision"] for row in curve])
    recall = np.asarray([row["recall"] for row in curve])
    ap = float(np.mean([precision[recall >= target].max(initial=0) for target in np.linspace(0, 1, 101)])) if n_gt else 0.0
    return {"fixed": fixed_counts, "P_at_R90": best, "maximum_recall": maximum_recall,
            "AP50_101point": ap, "curve": curve,
            "AP_definition": "101 recall samples of interpolated precision; one-class IoU=0.5; not COCO mAP"}


def stratified_gt_metrics(gt: list[dict], predictions: list[dict], threshold: float,
                          match_field: str, iou_field: str) -> dict[str, Any]:
    matches = {row[match_field]: row for row in predictions if row[match_field] is not None and row["score"] >= threshold}
    result = {}
    for stratum in sorted({row["visibility_stratum"] for row in gt}):
        members = [row for row in gt if row["visibility_stratum"] == stratum]
        hit = [matches[row["key"]] for row in members if row["key"] in matches]
        result[stratum] = {"GT": len(members), "TP": len(hit), "FN": len(members) - len(hit),
            "recall": len(hit) / len(members),
            "matched_mean_iou": float(np.mean([row[iou_field] for row in hit])) if hit else None,
            "missed_GT_keys": [row["key"] for row in members if row["key"] not in matches]}
    return result


def load_val_manifest(path: Path, limit: int | None = None) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("split") != "val" or manifest.get("role") != "development":
        raise TrainingContractError("Only original VAL/development manifest allowed; sealed test forbidden")
    rows = manifest["samples"]
    actual = (len(rows), sum(row["instance_count"] for row in rows), sum(row["instance_count"] == 0 for row in rows))
    if actual != (814, 3449, 13) or any(row["split"] != "val" for row in rows):
        raise TrainingContractError(f"Full official VAL contract differs: {actual}")
    if limit is not None:
        if not 1 <= limit <= len(rows):
            raise TrainingContractError("Diagnostic limit must be 1..814")
        rows = rows[:limit]
    return manifest, rows


def make_predictor(*, weights: Path, kind: str, imgsz: int, conf: float, max_det: int,
                   output: Path, device: str = "cuda:0"):
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment import SegmentationPredictor
    if kind == "local":
        model = load_local_checkpoint(weights)
        apple = 0
    elif kind == "migrated_initial":
        model, _ = build_visible_model(weights)
        apple = 0
    elif kind == "coco":
        if sha256_file(weights) != OFFICIAL_SHA256:
            raise TrainingContractError("Official COCO initialization hash mismatch")
        model = YOLO(str(weights), task="segment").model
        apple = 47
    else:
        raise TrainingContractError(f"Unknown model kind: {kind}")
    if model.names[apple] != "apple":
        raise TrainingContractError("Checkpoint class mapping does not identify apple")
    predictor = SegmentationPredictor(overrides={"task": "segment", "imgsz": imgsz, "conf": conf,
        "iou": .7, "classes": [apple], "max_det": max_det, "device": device,
        "quantize": 16 if device != "cpu" else None, "retina_masks": True, "rect": False,
        "batch": 1, "verbose": False, "save": False, "show": False, "save_txt": False,
        "project": str(output), "name": "predictor", "exist_ok": True})
    # setup_model fuses only this newly loaded evaluation object, never a live
    # trainer or checkpoint file. Both initial and trained models use this path.
    predictor.setup_model(model=model.eval(), verbose=False)
    return predictor, apple


def predict_native(predictor, bgr: np.ndarray, *, synchronize: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    if synchronize:
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = predictor(source=bgr, stream=False)[0]
    boxes = result.boxes.data.detach().float().cpu().numpy()
    masks = result.masks.data.detach().cpu().numpy() if result.masks is not None else np.zeros((0, *bgr.shape[:2]), np.uint8)
    if synchronize:
        torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000
    if masks.shape != (len(boxes), *bgr.shape[:2]) or not np.isfinite(boxes).all():
        raise TrainingContractError("Nonfinite prediction or native mask coordinate mismatch")
    if masks.dtype != np.uint8:
        masks = masks.astype(np.uint8)
    return boxes, masks, ms


def read_bgr(row: dict) -> np.ndarray:
    for key in ("rgb_path", "mask_path"):
        require_f_path(row[key])
    if sha256_file(row["rgb_path"]) != row["rgb_sha256"]:
        raise TrainingContractError(f"RGB hash mismatch: {row['image_id']}")
    image = cv2.imdecode(np.fromfile(row["rgb_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or list(image.shape[:2]) != row["original_hw"]:
        raise TrainingContractError(f"Unreadable/mismatched RGB: {row['image_id']}")
    return image


def benchmark(predictor, all_rows: list[dict], repo: Path, output: Path, *, sizes: list[int],
              warmup: int = 5, rounds: int = 5) -> dict:
    sample_path = repo / "configs/premask_audit_samples_20261102.json"
    sample_ids = [row["image_id"] for row in json.loads(sample_path.read_text(encoding="utf-8-sig"))["samples"][:8]]
    lookup = {row["image_id"]: row for row in all_rows}
    rows = [lookup[key] for key in sample_ids]
    images = [read_bgr(row) for row in rows]
    saved = {key: getattr(predictor.args, key) for key in ("imgsz", "conf", "max_det")}
    cases = []
    try:
        predictor.args.conf, predictor.args.max_det = .25, 64
        for size in sizes:
            predictor.args.imgsz = size
            cold = predict_native(predictor, images[0])[2]
            for i in range(warmup):
                predict_native(predictor, images[i % len(images)])
            torch.cuda.reset_peak_memory_stats()
            frames = []
            for repeat in range(rounds):
                for row, image in zip(rows, images):
                    boxes, masks, latency = predict_native(predictor, image)
                    frames.append({"round": repeat + 1, "image_id": row["image_id"], "ms": latency,
                                   "instances": len(boxes), "native_hw": list(masks.shape[1:])})
            times = np.asarray([row["ms"] for row in frames])
            cases.append({"imgsz": size, "cold_ms": cold, "warmup": warmup, "rounds": rounds,
                "frames": frames, "mean_ms": float(times.mean()), "median_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)), "fps_from_mean": float(1000 / times.mean()),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()})
    finally:
        for key, value in saved.items():
            setattr(predictor.args, key, value)
    value = {"schema": "local_visible_speed_benchmark_v1", "conf": .25, "max_det": 64,
        "precision": "FP16", "batch": 1, "input": "whole field of view letterboxed; no target crop",
        "timing": "predecoded BGR -> native V masks and boxes delivered to CPU; includes preprocessing and GPU sync",
        "excludes": ["disk", "model loading", "drawing", "depth", "geometry", "association", "camera acquisition"],
        "sample_manifest_sha256": sha256_file(sample_path), "cases": cases,
        "D405_real_tracking_test": False, "end_to_end_20Hz_verified": False}
    write_json(output / "speed_benchmark.json", value)
    return value


def run_evaluation(*, repo: Path, manifest: Path, weights: Path, output: Path, kind: str = "local",
                   imgsz: int = 640, conf_floor: float = .001, fixed_threshold: float = .25,
                   max_det: int = 300, limit: int | None = None, benchmark_only: bool = False,
                   run_benchmark: bool = False, benchmark_sizes: list[int] | None = None,
                   save_mask_images: int = 8) -> dict:
    configure_f_runtime(repo)
    manifest, weights, output = map(require_f_path, (manifest, weights, output))
    if not 0 <= conf_floor < fixed_threshold <= 1 or imgsz % 32 or imgsz < 32 or max_det < 1:
        raise TrainingContractError("Invalid evaluation protocol")
    if output.exists() and any(output.iterdir()):
        raise TrainingContractError("Use a new output directory; existing evidence cannot be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise TrainingContractError("CUDA required for executed evaluation; CPU is reserved for contract tests")
    manifest_data, rows = load_val_manifest(manifest, limit)
    weight_sha = sha256_file(weights)
    status = {"phase": "initializing", "pid": os.getpid(), "created_utc": now(),
        "source_sha256": sha256_file(__file__), "manifest_sha256": sha256_file(manifest),
        "weights": str(weights), "weights_sha256": weight_sha, "kind": kind,
        "imgsz": imgsz, "confidence_floor_strictly_greater_than": conf_floor, "fixed_threshold_inclusive": fixed_threshold,
        "max_det": max_det, "mask_iou_threshold": .5, "mask_definition": "visible native raster",
        "bbox_definition": "secondary visible tight GT boxes; NOT strong model amodal bbox acceptance",
        "retina_masks": True, "scope": "full_814_VAL" if limit is None else f"diagnostic_first_{limit}_VAL",
        "val_is_development_not_independent_test": True, "test_used": False, "training_performed": False,
        "GPU": torch.cuda.get_device_name(), "torch": torch.__version__,
        "ultralytics": __import__("ultralytics").__version__, "frames_completed": 0}
    write_json(output / "status.json", status)
    try:
        predictor, apple = make_predictor(weights=weights, kind=kind, imgsz=imgsz, conf=conf_floor,
                                          max_det=max_det, output=output)
        status["apple_class"] = apple
        parameter_dtype = next(predictor.model.model.parameters()).dtype
        status["actual_precision"] = {"fp16": bool(predictor.model.fp16), "device": str(predictor.device),
                                       "parameter_dtype": str(parameter_dtype)}
        if not predictor.model.fp16 or predictor.device.type != "cuda" or parameter_dtype != torch.float16:
            raise TrainingContractError("FP16 CUDA requested but not active")
        if benchmark_only:
            speed = benchmark(predictor, manifest_data["samples"], repo, output, sizes=benchmark_sizes or [640, 512])
            status.update(phase="benchmark_complete", updated_utc=now())
            write_json(output / "status.json", status)
            return speed
        detections, truth, empty_images = [], [], []
        with torch.inference_mode():
            for index, row in enumerate(rows):
                image = read_bgr(row)
                if sha256_file(row["mask_path"]) != row["mask_sha256"]:
                    raise TrainingContractError(f"Native GT mask hash mismatch: {row['image_id']}")
                gt = decode_packed_masks(row["mask_path"])
                if gt["instance_ids"] != [str(x["instance_id"]) for x in row["instances"]]:
                    raise TrainingContractError("Manifest instance order does not match native mask pack")
                boxes, masks, elapsed = predict_native(predictor, image)
                if len(boxes) and not np.all(boxes[:, 5] == apple):
                    raise TrainingContractError("Predictor returned an unexpected class after apple filtering")
                miou = native_mask_iou(masks, gt["visible"])
                biou = box_iou(boxes[:, :4], gt["boxes_visible_xyxy"])
                mmatch, matched_iou = greedy_match(boxes[:, 4], miou)
                bmatch, bmatched_iou = greedy_match(boxes[:, 4], biou)
                keys = [f"{row['image_id']}::{key}" for key in gt["instance_ids"]]
                truths = [{"key": key, "image_id": row["image_id"], "instance_id": inst["instance_id"],
                           "visibility_stratum": inst["visibility_stratum"],
                           "visibility_ratio": inst["visibility_V_area_over_A_area"]}
                          for key, inst in zip(keys, row["instances"])]
                truth.extend(truths)
                current = []
                for j, box in enumerate(boxes):
                    item = {"image_id": row["image_id"], "prediction_index": j,
                        "score": float(box[4]), "box_xyxy": box[:4].tolist(),
                        "mask_GT": keys[mmatch[j]] if mmatch[j] >= 0 else None,
                        "mask_IoU": float(matched_iou[j]),
                        "mask_best_IoU": float(miou[j].max(initial=0)),
                        "bbox_GT": keys[bmatch[j]] if bmatch[j] >= 0 else None,
                        "bbox_IoU": float(bmatched_iou[j]), "empty_GT_image": not bool(keys)}
                    current.append(item)
                detections.extend(current)
                if not keys:
                    empty_images.append({"image_id": row["image_id"], "predictions": len(boxes),
                                         "FP_at_fixed": int((boxes[:, 4] >= fixed_threshold).sum())})
                frame = {"image_id": row["image_id"], "rgb_sha256": row["rgb_sha256"],
                    "GT_mask_sha256": row["mask_sha256"], "truth": truths, "predictions": current,
                    "native_prediction_mask_shape": list(masks.shape), "inference_delivery_ms": elapsed}
                with (output / "frames.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n")
                if index < save_mask_images:
                    saved = boxes[:, 4] >= fixed_threshold
                    destination = output / "diagnostic_native_masks"
                    destination.mkdir(exist_ok=True)
                    np.savez_compressed(destination / (Path(row["image_id"]).stem + ".npz"),
                        boxes=boxes[saved], visible_packed=np.packbits(masks[saved], axis=-1, bitorder="little"),
                        original_hw=np.asarray(image.shape[:2], dtype=np.int32), threshold=np.asarray(fixed_threshold))
                status.update(phase="evaluating", frames_completed=index + 1, updated_utc=now())
                if (index + 1) % 20 == 0 or index + 1 == len(rows):
                    write_json(output / "status.json", status)
                    print(json.dumps({"frames": index + 1, "total": len(rows)}, ensure_ascii=False), flush=True)
                del image, masks, gt, miou, biou
        scores = [row["score"] for row in detections]
        mask_metrics = confidence_metrics(scores, [row["mask_GT"] is not None for row in detections], len(truth), fixed_threshold)
        bbox_metrics = confidence_metrics(scores, [row["bbox_GT"] is not None for row in detections], len(truth), fixed_threshold)
        strata = {"fixed": stratified_gt_metrics(truth, detections, fixed_threshold, "mask_GT", "mask_IoU")}
        if mask_metrics["P_at_R90"]:
            strata["P_at_R90"] = stratified_gt_metrics(truth, detections, mask_metrics["P_at_R90"]["threshold"], "mask_GT", "mask_IoU")
        selected = [row for row in detections if row["score"] >= fixed_threshold and row["mask_GT"] is not None]
        summary = {**status, "phase": "complete", "updated_utc": now(), "images": len(rows),
            "GT_instances": len(truth), "empty_GT_images": len(empty_images), "predictions": len(detections),
            "mask": mask_metrics, "visible_bbox_secondary": bbox_metrics, "occlusion_mask": strata,
            "matched_mask_mean_IoU_at_fixed": float(np.mean([row["mask_IoU"] for row in selected])) if selected else None,
            "empty_image_FP_at_fixed": sum(row["FP_at_fixed"] for row in empty_images),
            "empty_images": empty_images, "matching": "confidence-descending stable greedy best-unused-GT IoU>=0.5; ties kept as whole threshold groups",
            "PR_domain": f"retained native candidates score>{conf_floor}, max_det={max_det}; no extrapolation below floor",
            "independent_test_claimed": False, "compared_with_global_amodal_bbox_baseline": False,
            "timing_from_validation_is_deployment_benchmark": False, "weights_unchanged": sha256_file(weights) == weight_sha}
        if not summary["weights_unchanged"]:
            raise TrainingContractError("Evaluation changed the checkpoint")
        write_json(output / "validation_summary.json", summary)
        matched_keys = {row["mask_GT"] for row in selected}
        write_json(output / "GT_matches_fixed.json", {"threshold": fixed_threshold,
            "GT": [{**row, "matched": row["key"] in matched_keys} for row in truth]})
        if run_benchmark:
            benchmark(predictor, manifest_data["samples"], repo, output, sizes=benchmark_sizes or [640, 512])
        status.update(phase="complete", updated_utc=now())
        write_json(output / "status.json", status)
        return {"mask": {k: v for k, v in mask_metrics.items() if k != "curve"},
                "images": len(rows), "GT": len(truth), "summary": str(output / "validation_summary.json")}
    except BaseException as error:
        status.update(phase="failed", updated_utc=now(), error=repr(error))
        write_json(output / "status.json", status)
        raise
