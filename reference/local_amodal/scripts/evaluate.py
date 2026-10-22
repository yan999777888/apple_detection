"""Streaming validation for the remote dual-mask experiment.

No torch/model import is performed at module import. evaluate_model first calls
the standard-library-only train_amodal.remote_guard, which checks the hostname
before importing torch and then checks the RTX 3080 device.
Only boxes/scores and scalar mask diagnostics persist across images; full masks
are released after each image. Official split independence is not established.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from metrics import calibrate_threshold, evaluate_at_threshold, match_image


RAW_CONFIDENCE = 0.001
VISUAL_CONFIDENCE = 0.25
VISUAL_TOP_K = 30
MAX_PREVIEWS = 6


def _atomic_json(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _nonzero_box(mask):
    rows = np.flatnonzero(np.any(mask, axis=1))
    cols = np.flatnonzero(np.any(mask, axis=0))
    if not len(rows) or not len(cols):
        return (0, 0, 0, 0)
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def _overlaps_one_mask(predicted, truth_masks, truth_areas, truth_boxes):
    """Exact mask IoU with intersections restricted to overlapping pixel bounds."""
    predicted = np.asarray(predicted, dtype=bool)
    area = int(np.count_nonzero(predicted))
    result = np.zeros(len(truth_masks), dtype=np.float64)
    if not area:
        return result
    px1, py1, px2, py2 = _nonzero_box(predicted)
    for index, (gx1, gy1, gx2, gy2) in enumerate(truth_boxes):
        x1, y1, x2, y2 = max(px1, gx1), max(py1, gy1), min(px2, gx2), min(py2, gy2)
        if x2 <= x1 or y2 <= y1:
            continue
        intersection = int(np.count_nonzero(
            predicted[y1:y2, x1:x2] & truth_masks[index, y1:y2, x1:x2]))
        union = area + int(truth_areas[index]) - intersection
        result[index] = intersection / union if union else 0.0
    return result


def _greedy_mask_flags(overlaps, scores, iou_threshold=0.5):
    """One-to-one mask matching, independently for each branch."""
    overlaps = np.asarray(overlaps, dtype=np.float64)
    if overlaps.ndim != 2 or overlaps.shape[0] != len(scores):
        raise ValueError("Mask IoU matrix must be prediction_count x GT_count")
    assigned = np.zeros(overlaps.shape[1], dtype=bool)
    flags = np.zeros(len(scores), dtype=bool)
    for prediction in np.argsort(-np.asarray(scores), kind="stable"):
        if not len(assigned):
            break
        available = np.where(~assigned, overlaps[prediction], -1.0)
        target = int(np.argmax(available))
        if available[target] >= iou_threshold:
            flags[prediction] = True
            assigned[target] = True
    return flags


def _mask_pr(tp, fp, gt_count, confidence):
    tp, fp, gt_count = int(tp), int(fp), int(gt_count)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / gt_count if gt_count else 0.0
    return {"confidence": float(confidence), "mask_iou_threshold": 0.5,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp, "fp": fp, "fn": gt_count - tp, "gt_count": gt_count,
            "threshold_policy": "Same validation-selected BOX confidence; masks matched independently.",
            "is_mask_ap": False}


def _pair_summary(values, total_box_tp):
    return {"pairs_evaluated": len(values), "box_true_positives": int(total_box_tp),
            "mean_iou": float(np.mean(values)) if values else None,
            "median_iou": float(np.median(values)) if values else None,
            "fraction_iou_ge_0_5": float(np.mean(np.asarray(values) >= 0.5)) if values else None,
            "interpretation": "Conditional mask IoU on matched box TP, not mask AP or overall mask recall."}


def _color(index):
    return ((index * 79 + 80) % 206 + 30,
            (index * 131 + 10) % 206 + 30,
            (index * 47 + 120) % 206 + 30)


def _overlay_mask(canvas, mask, color):
    alpha = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 92)
    canvas.paste(color, (0, 0, canvas.width, canvas.height), alpha)


def _save_preview(path, rgb, visible_canvas, amodal_canvas, boxes, scores, indices, filename):
    height, width = rgb.shape[:2]
    for canvas in (visible_canvas, amodal_canvas):
        drawing = ImageDraw.Draw(canvas)
        for index in indices:
            box = tuple(map(float, boxes[index]))
            color = _color(int(index))
            drawing.rectangle(box, outline=color, width=2)
            drawing.text((max(0, box[0]), max(0, box[1] - 12)),
                         f"{index}: {scores[index]:.3f}", fill=color,
                         stroke_width=1, stroke_fill=(0, 0, 0))
    header = 56
    preview = Image.new("RGB", (width * 3, height + header), (245, 245, 245))
    preview.paste(Image.fromarray(rgb), (0, header))
    preview.paste(visible_canvas, (width, header))
    preview.paste(amodal_canvas, (width * 2, header))
    drawing = ImageDraw.Draw(preview)
    drawing.text((8, 4), f"{filename} | Visualization only: confidence >= 0.25, top 30; NOT metric threshold.", fill="black")
    for index, label in enumerate(("RGB", "Visible predicted mask", "Amodal predicted mask (inferred)")):
        drawing.text((index * width + 8, 31), label, fill="black")
    preview.thumbnail((2100, 1000), Image.Resampling.LANCZOS)
    preview.save(path)


def evaluate_model(model, dataset, device, output_dir, max_images=None, *,
                   fixed_confidence=None, threshold_source=None):
    """Evaluate boxes and both masks, sharing one box confidence.

    Default val behavior calibrates on val. Test requires a previously frozen
    val confidence and its provenance; it can never calibrate or use a subset.
    The model outputs masks and visible_masks in original image coordinates.
    """
    split = getattr(dataset, "split", None)
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    if fixed_confidence is None:
        if split != "val" or threshold_source is not None:
            raise ValueError("Test requires a frozen validation confidence and source")
    else:
        fixed_confidence = float(fixed_confidence)
        if not np.isfinite(fixed_confidence) or not RAW_CONFIDENCE <= fixed_confidence <= 1.0:
            raise ValueError("Fixed confidence must be finite and within [RAW_CONFIDENCE, 1]")
        if (not isinstance(threshold_source, dict) or threshold_source.get("dataset_split") != "val"
                or threshold_source.get("confidence") != fixed_confidence):
            raise ValueError("Fixed confidence requires matching validation provenance")
        threshold_source = json.loads(json.dumps(threshold_source, allow_nan=False))
    if split == "test" and max_images is not None:
        raise ValueError("Heldout test must evaluate the complete split")
    from train_amodal import remote_guard
    torch = remote_guard()
    if getattr(dataset, "augment", False):
        raise ValueError("Validation dataset must have augment=False")
    if not str(device).startswith("cuda"):
        raise ValueError("This evaluator is restricted to the authorized remote CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("Remote CUDA device is unavailable")

    count = len(dataset) if max_images is None else min(len(dataset), int(max_images))
    if count <= 0:
        raise ValueError("Validation needs at least one image")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    old_training = bool(model.training)
    old_score_threshold = float(model.roi_heads.score_thresh)
    model.roi_heads.score_thresh = RAW_CONFIDENCE
    model.eval()
    box_records, diagnostic_records, latencies = [], [], []
    raw_prediction_counts, cap_hit_images = [], []
    started = time.perf_counter()
    infer_context = {"raw_confidence": RAW_CONFIDENCE,
                     "nms_threshold": float(model.roi_heads.nms_thresh),
                     "max_detections": int(model.roi_heads.detections_per_img),
                     "amp": True, "device": str(device), "mask_binarization_threshold": 0.5,
                     "images": count, "dataset_split": split, "official_split_independence_verified": False,
                     "scope": ("Official random PApple split, exploratory validation only." if split == "val"
                               else "Official heldout random PApple split; orchard/sequence independence is not established."),
                     "visualization": {"confidence": VISUAL_CONFIDENCE, "top_k": VISUAL_TOP_K,
                                       "images": min(count, MAX_PREVIEWS),
                                       "threshold_is_metric_threshold": False}}
    if fixed_confidence is not None:
        infer_context.update(fixed_confidence=fixed_confidence, threshold_source=threshold_source,
                             threshold_policy="Frozen validation BOX confidence; no calibration on this split.")
    _atomic_json(output_dir / "evaluation_metadata.json", infer_context)
    try:
        with torch.inference_mode():
            for image_index in range(count):
                rgb, truth = dataset.load_sample(image_index)
                filename = dataset.samples[image_index]["filename"]
                tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255.0).to(device)
                torch.cuda.synchronize(device)
                begin = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = model([tensor])[0]
                torch.cuda.synchronize(device)
                latencies.append(time.perf_counter() - begin)
                for field in ("boxes", "scores", "masks", "visible_masks"):
                    if field not in prediction:
                        raise ValueError(f"Missing prediction field {field}")
                all_boxes = prediction["boxes"].detach().float().cpu().numpy()
                all_scores = prediction["scores"].detach().float().cpu().numpy()
                if not np.isfinite(all_boxes).all() or not np.isfinite(all_scores).all():
                    raise ValueError(f"Nonfinite model boxes/scores on {filename}")
                selected = np.flatnonzero(all_scores >= RAW_CONFIDENCE)
                boxes, scores = all_boxes[selected], all_scores[selected]
                raw_prediction_counts.append(len(boxes))
                if len(all_scores) >= infer_context["max_detections"]:
                    cap_hit_images.append(filename)
                gt_boxes = truth["boxes"]
                record = {"image_id": filename, "gt": {"boxes": gt_boxes.tolist()},
                          "pred": {"boxes": boxes.tolist(), "scores": scores.tolist()}}
                box_records.append(record)
                gt_count, pred_count = len(gt_boxes), len(boxes)
                raw_match = match_image(gt_boxes, boxes, scores, 0.5, RAW_CONFIDENCE)
                branch_matrix = {}
                visualization_indices = np.argsort(-scores, kind="stable")
                visualization_indices = visualization_indices[scores[visualization_indices] >= VISUAL_CONFIDENCE][:VISUAL_TOP_K]
                preview_enabled = image_index < MAX_PREVIEWS
                visible_canvas = Image.fromarray(rgb) if preview_enabled else None
                amodal_canvas = Image.fromarray(rgb) if preview_enabled else None
                visualization_set = set(map(int, visualization_indices))
                for branch, canvas in (("visible_masks", visible_canvas), ("masks", amodal_canvas)):
                    gt_masks = truth[branch]
                    if gt_masks.shape != (gt_count, rgb.shape[0], rgb.shape[1]):
                        raise ValueError(f"GT {branch} shape mismatch")
                    gt_areas = np.count_nonzero(gt_masks, axis=(1, 2))
                    gt_bounds = [_nonzero_box(mask) for mask in gt_masks]
                    matrix = np.zeros((pred_count, gt_count), dtype=np.float64)
                    masks_tensor = prediction[branch]
                    if masks_tensor.shape[0] != len(all_scores):
                        raise ValueError(f"{branch} instance count differs from scores")
                    if not bool(torch.isfinite(masks_tensor).all().item()):
                        raise ValueError(f"Nonfinite {branch} probabilities on {filename}")
                    for p_index, source_index in enumerate(selected):
                        # Move only one full-image probability mask at a time.
                        mask = masks_tensor[int(source_index)].detach()
                        if mask.ndim == 3 and mask.shape[0] == 1:
                            mask = mask[0]
                        if tuple(mask.shape) != tuple(rgb.shape[:2]):
                            raise ValueError(f"Predicted {branch} not in original image coordinates")
                        binary = mask.ge(0.5).cpu().numpy()
                        matrix[p_index] = _overlaps_one_mask(binary, gt_masks, gt_areas, gt_bounds)
                        if preview_enabled and p_index in visualization_set:
                            _overlay_mask(canvas, binary, _color(p_index))
                        del binary, mask
                    branch_matrix[branch] = matrix
                pairs = []
                for p_index, g_index in zip(raw_match["pred_indices"], raw_match["gt_indices"]):
                    if g_index >= 0:
                        pairs.append({"prediction_index": int(p_index), "gt_index": int(g_index),
                                      "score": float(scores[p_index]),
                                      "visible_iou": float(branch_matrix["visible_masks"][p_index, g_index]),
                                      "amodal_iou": float(branch_matrix["masks"][p_index, g_index])})
                diagnostic_records.append({
                    "image_id": filename, "gt_count": gt_count, "scores": scores.tolist(),
                    "box_tp_mask_pairs": pairs,
                    "visible_mask_tp_flags": _greedy_mask_flags(branch_matrix["visible_masks"], scores).tolist(),
                    "amodal_mask_tp_flags": _greedy_mask_flags(branch_matrix["masks"], scores).tolist()})
                if preview_enabled:
                    _save_preview(preview_dir / f"preview_{image_index:03d}.png", rgb,
                                  visible_canvas, amodal_canvas, boxes, scores,
                                  visualization_indices, filename)
                del prediction, tensor, truth, branch_matrix, masks_tensor
                if (image_index + 1) % 25 == 0 or image_index + 1 == count:
                    print(f"{split.upper()} {image_index + 1}/{count} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        if fixed_confidence is None:
            calibration = calibrate_threshold(box_records)
            confidence = calibration["confidence"]
            summary = calibration["summary"]
        else:
            confidence = fixed_confidence
            summary = evaluate_at_threshold(box_records, confidence, iou_threshold=0.5)
            calibration = {"confidence": confidence, "iou_threshold": 0.5,
                           "selection_rule": "Frozen validation BOX confidence; no calibration on this split.",
                           "threshold_source": threshold_source}
        visible_values, amodal_values = [], []
        visible_tp = amodal_tp = predictions_kept = gt_total = 0
        for record in diagnostic_records:
            keep = np.asarray(record["scores"]) >= confidence
            predictions_kept += int(keep.sum())
            visible_tp += int((np.asarray(record["visible_mask_tp_flags"], dtype=bool) & keep).sum())
            amodal_tp += int((np.asarray(record["amodal_mask_tp_flags"], dtype=bool) & keep).sum())
            gt_total += record["gt_count"]
            for pair in record["box_tp_mask_pairs"]:
                if pair["score"] >= confidence:
                    visible_values.append(pair["visible_iou"])
                    amodal_values.append(pair["amodal_iou"])
        if len(visible_values) != summary["tp"] or len(amodal_values) != summary["tp"]:
            raise AssertionError("Selected box TP and scalar mask-pair counts disagree")
        summary["visible_mask_iou_on_box_tp"] = _pair_summary(visible_values, summary["tp"])
        summary["amodal_mask_iou_on_box_tp"] = _pair_summary(amodal_values, summary["tp"])
        summary["visible_instance_mask_pr_iou50"] = _mask_pr(visible_tp, predictions_kept-visible_tp, gt_total, confidence)
        summary["amodal_instance_mask_pr_iou50"] = _mask_pr(amodal_tp, predictions_kept-amodal_tp, gt_total, confidence)
        result = {"summary": summary, "confidence": confidence,
                  "score": min(summary["precision"], summary["recall"]),
                  "mask_diagnostics": {"visible": summary["visible_mask_iou_on_box_tp"],
                                       "amodal": summary["amodal_mask_iou_on_box_tp"]},
                  "threshold_selection": {key: value for key, value in calibration.items() if key != "summary"},
                  "inference": infer_context,
                  "prediction_cap_diagnostic": {
                      "raw_confidence": RAW_CONFIDENCE,
                      "max_detections": infer_context["max_detections"],
                      "images_reaching_cap": len(cap_hit_images),
                      "fraction_reaching_cap": len(cap_hit_images) / count,
                      "filenames_reaching_cap": cap_hit_images,
                      "raw_prediction_count_quantiles": {
                          str(q): float(np.percentile(raw_prediction_counts, q))
                          for q in (0, 50, 90, 95, 99, 100)},
                      "interpretation": "Cap saturation can hide lower-score detections; it does not itself prove missed GT. Compare a higher cap if saturation is material."},
                  "timing": {"total_seconds": time.perf_counter()-started,
                             "model_seconds_per_image_mean": float(np.mean(latencies)),
                             "model_seconds_per_image_median": float(np.median(latencies)),
                             "model_seconds_per_image_p95": float(np.percentile(latencies, 95)),
                             "warmup_excluded": False,
                             "model_timing_excludes": "image load, H2D copy, mask metrics and PNG saving"},
                  "files": {"raw_predictions": "raw_predictions.json",
                            "mask_pair_diagnostics": "mask_pair_diagnostics.json",
                            "metadata": "evaluation_metadata.json", "previews": "previews"}}
        _atomic_json(output_dir / "raw_predictions.json", box_records)
        _atomic_json(output_dir / "mask_pair_diagnostics.json", diagnostic_records)
        _atomic_json(output_dir / "summary.json", result)
        return result
    finally:
        model.roi_heads.score_thresh = old_score_threshold
        model.train(old_training)
