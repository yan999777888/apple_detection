"""CPU-only diagnosis of completed, original full PApple VAL archives.

This module reads predictions, never models or images. Each image's Q x GT
IoU matrix is computed once and reused for matching, error categories and the
optional filtered-pool graph bound. Historical evaluation source hashes are
checked as recorded snapshots; they need not equal today's training sources.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


PROVENANCE_SHA256 = "fcd96d08c75e1663c83efecb8eb321bbc10220dc74314c8227f1f8565a2ccb3a"
METRICS_SHA256 = "1b0a7e385940fd0c6160cc08b75ddaf2e1f43f3ea00f5bd37d547fea6b08010d"
FULL_SCOPE = (814, 3449, 13)
SELECTION_RULE = "validation max(min(P,R)); tie max(F1); tie higher confidence"
FN_LOW = "FN_low_score_only_qualifying_candidate"
FN_COMPETITION = "FN_selected_qualifying_candidate_competed_for_other_GT"
FN_MISSING = "FN_no_qualifying_archived_candidate"
FP_BACKGROUND = "background_maxIoU_lt0.1"
FP_LOCALIZATION = "localization_0.1_leIoU_lt0.5"
FP_DUPLICATE = "duplicate_IoU_ge0.5_to_already_matched_GT"
GT_FIELDS = ("image_id", "gt_index", "apple_id", "box", "amodal_area_pixels", "visible_area_pixels",
             "visible_area_fraction", "visible_pixels_outside_amodal", "occlusion_stratum",
             "amodal_mask_uint8_sha256", "visible_mask_uint8_sha256")
POOL_LIMITATIONS = [
    "All coverage and graph bounds refer only to the archived, already filtered model-returned pool.",
    "They do not describe all RPN or pre-NMS proposals; an absent archived box does not prove RPN failure.",
    "GT-dependent graph matching is an oracle statistic, not learned accuracy or achievable model P/R.",
    "Per-GT candidate coverage is not jointly attainable recall when candidates compete for GT.",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _counts(tp, fp, gt, confidence):
    tp, fp, gt = int(tp), int(fp), int(gt)
    precision, recall = tp / (tp + fp) if tp + fp else 0., tp / gt if gt else 0.
    return {"confidence": float(confidence), "iou_threshold": .5, "tp": tp, "fp": fp, "fn": gt - tp,
            "gt_count": gt, "prediction_count": tp + fp, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "min_precision_recall": min(precision, recall)}


def _boxes(value, *, gt=False):
    array = np.asarray(value, dtype=np.float64)
    if not array.size:
        return array.reshape(0, 4)
    _require(array.ndim == 2 and array.shape[1] == 4 and np.isfinite(array).all(), "Boxes must be finite N x 4")
    _require(not gt or bool((array[:, 2:] > array[:, :2]).all()), "GT boxes must have positive area")
    return array


def box_iou(predictions, truth):
    """Reference XYXY exclusive-extent IoU; invalid-area predictions have IoU0."""
    a, b = _boxes(predictions), _boxes(truth, gt=True)
    wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]))
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def greedy_assignment(overlaps, scores):
    """Assign all predictions once; any >=confidence prefix has the same matches.

    Equal scores retain prediction order; equal IoUs retain original GT order.
    The returned vector is in original prediction order with -1 for FP.
    """
    overlaps, scores = np.asarray(overlaps, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    _require(overlaps.ndim == 2 and scores.shape == (len(overlaps),) and np.isfinite(overlaps).all()
             and np.isfinite(scores).all() and bool(((scores >= 0) & (scores <= 1)).all())
             and bool(((overlaps >= 0) & (overlaps <= 1)).all()), "Invalid overlap matrix or probability scores")
    result, used = np.full(len(scores), -1, dtype=np.int64), np.zeros(overlaps.shape[1], dtype=bool)
    if not len(used):
        return result
    for q in np.argsort(-scores, kind="stable"):
        candidates = np.where(~used, overlaps[q], -1.)
        g = int(np.argmax(candidates))
        if candidates[g] >= .5:
            result[q], used[g] = g, True
    return result


def filtered_pool_matching_bound(overlaps):
    """Maximum cardinality on Q x GT edges >=.5; no Q x Q allocation."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    if not min(overlaps.shape):
        return 0
    return int(np.count_nonzero(maximum_bipartite_matching(csr_matrix(overlaps >= .5), perm_type="column") >= 0))


def visible_size_bin(area, *, equivalent_short_side=None):
    side = math.sqrt(area)
    if equivalent_short_side is not None:
        side *= equivalent_short_side / 1300
    return "lt16" if side < 16 else "16to32" if side < 32 else "32to64" if side < 64 else "ge64"


def occlusion_bin(visible, amodal):
    _require(amodal > 0 and visible >= 0, "Original mask areas must be valid")
    if visible > amodal:
        return "annotation_visible_area_exceeds_amodal"
    ratio = visible / amodal
    return ("heavy_visible_lt_0.40" if ratio < .4 else "moderate_visible_0.40_to_0.70" if ratio < .7 else
            "light_visible_0.70_to_0.90" if ratio < .9 else "mostly_visible_0.90_to_1.00")


def analyze_image(record, truth, confidence, *, pool_bound=False):
    """Small pure mathematical kernel, also used by boundary fixtures."""
    _require(isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
             and math.isfinite(confidence) and 0 <= confidence <= 1, "Invalid exact confidence")
    _require(record["gt"]["boxes"] == [x["box"] for x in truth], "Archived GT boxes/order differ from original provenance")
    boxes, gt_boxes = _boxes(record["pred"]["boxes"]), _boxes(record["gt"]["boxes"], gt=True)
    scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
    _require(scores.shape == (len(boxes),), "Prediction box/score lengths differ")
    overlaps = box_iou(boxes, gt_boxes)  # The only IoU construction per image.
    assignments = greedy_assignment(overlaps, scores)
    selected = scores >= confidence
    recalled = set(assignments[selected]) - {-1}
    outcomes, false_positives = [], []
    for g, gt in enumerate(truth):
        column = overlaps[:, g]
        qualifying = np.flatnonzero(column >= .5)
        selected_good = qualifying[selected[qualifying]]
        hit = g in recalled
        category = ("TP" if hit else FN_MISSING if not len(qualifying) else
                    FN_LOW if not len(selected_good) else FN_COMPETITION)
        best = int(np.argmax(column)) if len(column) else None
        outcomes.append({**gt, "recalled_at_selected_threshold": hit, "error_category": category,
                         "archived_IoU50_candidate_count": int(len(qualifying)),
                         "selected_IoU50_candidate_count": int(len(selected_good)),
                         "best_archived_box_iou": float(column[best]) if best is not None else None,
                         "highest_score_IoU50_candidate": float(scores[qualifying].max()) if len(qualifying) else None})
    for q in np.flatnonzero(selected & (assignments < 0)):
        nearest = int(np.argmax(overlaps[q])) if len(truth) else None
        maximum = float(overlaps[q, nearest]) if nearest is not None else 0.
        category = FP_DUPLICATE if maximum >= .5 else FP_LOCALIZATION if maximum >= .1 else FP_BACKGROUND
        false_positives.append({"image_id": record["image_id"], "prediction_index": int(q),
                                "box": boxes[q].tolist(), "score": float(scores[q]), "max_GT_box_iou": maximum,
                                "closest_gt_index": nearest, "category": category, "empty_GT_image": not truth})
    tp, count = len(recalled), int(selected.sum())
    per_image = {"image_id": record["image_id"], "GT": len(truth), "raw_predictions": len(scores),
                 "selected_predictions": count, "TP": tp, "FP": count - tp, "FN": len(truth) - tp}
    if pool_bound:
        raw_bound, selected_bound = filtered_pool_matching_bound(overlaps), filtered_pool_matching_bound(overlaps[selected])
        covered = sum(x["archived_IoU50_candidate_count"] > 0 for x in outcomes)
        _require(tp <= selected_bound <= raw_bound <= covered <= len(truth), "Filtered-pool bound invariant failed")
        per_image.update(filtered_raw_pool_maximum_matching_TP=raw_bound,
                         filtered_selected_pool_maximum_matching_TP=selected_bound)
    return outcomes, false_positives, per_image, assignments


def _selected_threshold(scores, flags, gt_count):
    """Recompute calibration from cached match flags, without recomputing IoU."""
    scores, flags = np.asarray(scores), np.asarray(flags, dtype=np.int64)
    if not len(scores):
        return _counts(0, 0, gt_count, 1.)
    order = np.argsort(-scores, kind="stable")
    scores, cumulative = scores[order], np.cumsum(flags[order])
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    # Whole equal-score groups are selected together, across images too.
    best = None
    for i in ends:
        point = _counts(cumulative[i], i + 1 - cumulative[i], gt_count, scores[i])
        if best is None or (point["min_precision_recall"], point["f1"], point["confidence"]) > (
                best["min_precision_recall"], best["f1"], best["confidence"]):
            best = point
    return best


def _point(summary):
    _require(isinstance(summary, dict) and isinstance(summary.get("summary"), dict), "A validation summary object is required")
    point = summary.get("summary", {})
    confidence = summary.get("confidence")
    _require(isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
             and math.isfinite(confidence) and 0 <= confidence <= 1 and point.get("confidence") == confidence,
             "Summary must provide identical exact unrounded confidence values")
    _require(tuple(point.get(k) for k in ("images", "gt_count", "empty_gt_images")) == FULL_SCOPE,
             "Diagnosis requires all original 814 images / 3449 GT / 13 empty images")
    _require(point.get("iou_threshold") == .5, "Only IoU>=.5 archived protocol is supported")
    return point, confidence


def _truth_contract(provenance):
    _require(provenance.get("phase") == "complete" and provenance.get("passed") is True
             and provenance.get("scope", {}).get("split") == "val", "Completed original VAL provenance required")
    images, rows = provenance["image_metadata"], provenance["ground_truth_instances"]
    truth = {x["image_id"]: [] for x in images}
    _require(len(images) == len(truth) == 814 and len(rows) == 3449, "Provenance is not full original VAL")
    for row in rows:
        _require(row["image_id"] in truth, "Unknown provenance image")
        _require(occlusion_bin(row["visible_area_pixels"], row["amodal_area_pixels"]) == row["occlusion_stratum"],
                 "Provenance occlusion stratum differs from original mask areas")
        truth[row["image_id"]].append(row)
    for i, image in enumerate(images):
        rows = truth[image["image_id"]]
        _require(image["image_index"] == i and image["gt_count"] == len(rows)
                 and (image["height"], image["width"]) == (1300, 1300)
                 and [x["gt_index"] for x in rows] == list(range(len(rows)))
                 and [x["apple_id"] for x in rows] == image["apple_ids_in_target_order"], "Original GT identity/order differs")
    _require(sum(not x for x in truth.values()) == 13, "Original 13 empty images must remain")
    _require(sum(x["occlusion_stratum"] == "annotation_visible_area_exceeds_amodal" for x in provenance["ground_truth_instances"]) == 14,
             "Original 14 annotation area anomalies must remain")
    _require(sum(x["occlusion_stratum"] == "heavy_visible_lt_0.40" for x in provenance["ground_truth_instances"]) == 1514,
             "Original heavy-occlusion group must contain 1514 GT")
    _require(sum(visible_size_bin(x["visible_area_pixels"], equivalent_short_side=800) == "16to32"
                 for x in provenance["ground_truth_instances"]) == 389, "Fixed original 800-equivalent 16-32 group must contain 389 GT")
    return images, truth


def _validate_completion(summary, metadata, provenance):
    point, confidence = _point(summary)
    _require(summary.get("schema") == "apple_seg_papple_val_summary_v1"
             and metadata.get("schema") == "apple_seg_complete_papple_val_evaluation_v1"
             and summary.get("phase") == metadata.get("phase") == "complete"
             and summary.get("evaluation_family") == metadata.get("evaluation_family") == "OD_ROI"
             and summary.get("model_selection_eligible") is True, "Complete eligible formal OD evaluation required")
    _require(isinstance(summary.get("calibration"), dict)
             and summary["calibration"].get("selection_rule") == SELECTION_RULE
             and metadata.get("iou_threshold") == .5, "Archived selection/matching protocol differs")
    for key in ("official_full_val", "model_selection_eligible", "no_GT_repair_or_exclusion", "annotation_source_files_hashed",
                "source_before_after_identical", "all_RGB_before_after_verified", "all_normalized_inputs_before_after_verified",
                "all_inputs_recomputed_from_original_RGB", "all_original_GT_numeric_fields_and_mask_SHA256_verified"):
        _require(metadata.get(key) is True, f"Missing complete evaluation audit: {key}")
    _require(metadata.get("dataset_split") == "val" and metadata.get("subset_filter") is False
             and metadata.get("test_data_read") is False, "Subset/TEST or non-VAL archive rejected")
    _require(tuple(metadata.get(k) for k in ("images", "GT_instances", "empty_GT_images")) == FULL_SCOPE
             and metadata.get("consumed_images") == 814 and metadata.get("consumed_GT") == 3449
             and metadata.get("annotation_visible_area_exceeds_amodal_GT_retained") == 14, "Incomplete consumed original VAL scope")
    before, after = metadata.get("source_sha256_before"), metadata.get("source_sha256_after")
    _require(isinstance(before, dict) and before and before == after and all(_digest(v) for v in before.values()),
             "Recorded evaluation source hashes differ or are invalid")
    _require(metadata.get("provenance", {}).get("sha256") == PROVENANCE_SHA256
             and PROVENANCE_SHA256 in before.values() and METRICS_SHA256 in before.values(), "Recorded provenance/metrics pins differ")
    _require(all(v in before.values() for v in provenance["binding"]["current_annotation_sha256"].values()),
             "Recorded original annotation SHA256 pins missing")
    return point, confidence


def _same_point(actual, saved):
    for key in ("confidence", "iou_threshold", "tp", "fp", "fn", "gt_count", "prediction_count", "precision", "recall", "f1", "min_precision_recall"):
        _require(actual[key] == saved.get(key), f"Archive replay differs from summary: {key}")


def _parent_protocol(document):
    """Distinguish formally eligible OD from the older R50 archive format."""
    _require(document.get("phase", "complete") == "complete"
             and document.get("inference", {}).get("dataset_split") == "val", "Parent must be a completed original VAL summary")
    if "schema" in document or "evaluation_family" in document:
        _require(document.get("schema") == "apple_seg_papple_val_summary_v1"
                 and document.get("evaluation_family") == "OD_ROI" and document.get("model_selection_eligible") is True,
                 "OD parent summary must be formally eligible; incompatible schema rejected")
        return "formal_OD"
    _require(document.get("model_selection_eligible", True) is True
             and isinstance(document.get("threshold_selection"), dict)
             and document["threshold_selection"].get("selection_rule") == SELECTION_RULE
             and document["threshold_selection"].get("confidence") == document.get("confidence"),
             "Legacy parent must provide the exact reference selection protocol/confidence")
    return "legacy_R50"


def _validate_identity_audit(metadata, images, truth):
    ids = [x["image_id"] for x in images]
    audits = [metadata.get(k, []) for k in ("RGB_sha256_before_after", "normalized_input_sha256_before_after", "ground_truth_identity")]
    _require(all([x.get("image_id") for x in rows] == ids for rows in audits), "Incomplete or reordered source/input/GT audit")
    for image, rgb, inp, identity in zip(images, *audits, strict=True):
        _require(rgb.get("before") == rgb.get("after") == image["RGB_sha256"], "Archived original RGB SHA256 differs")
        _require(inp.get("before") == inp.get("after") == inp.get("sha256") and _digest(inp.get("sha256"))
                 and inp.get("recomputed_from_original_RGB") is True, "Archived normalized RGB input audit differs")
        rows = truth[image["image_id"]]
        _require(identity.get("original_hw") == [image["height"], image["width"]]
                 and identity.get("instance_ids") == image["apple_ids_in_target_order"]
                 and len(identity.get("GT", [])) == len(rows), "Archived GT identity inventory differs")
        _require(all(all(g.get(key) == frozen[key] for key in GT_FIELDS) for g, frozen in zip(identity["GT"], rows, strict=True)),
                 "Archived GT numeric/mask identity differs from original provenance")


def _assignments(value, count, gt_count, name):
    _require(isinstance(value, list) and len(value) == count
             and all(type(g) is int and -1 <= g < gt_count for g in value), f"Invalid {name} assignments")
    positive = [g for g in value if g >= 0]
    _require(len(positive) == len(set(positive)), f"Repeated GT in {name} assignments")
    return np.asarray(value, dtype=np.int64)


def _validate_mask_metric_scope(metadata, records, diagnostics, confidence):
    """Verify that every selected detection was evaluated against all GT.

    Historical online archives may lack the explicit flag. Deferred archives
    must distinguish intentionally uncomputed low-score entries from misses.
    """
    deferred = metadata.get("defer_mask_metrics", False)
    _require(type(deferred) is bool, "Mask evaluation strategy flag must be boolean")
    _require(isinstance(records, list) and isinstance(diagnostics, list)
             and len(records) == len(diagnostics), "Mask scope inventory differs")
    computed_count = selected_count = returned_count = 0
    for record, diagnostic in zip(records, diagnostics, strict=True):
        _require(record.get("image_id") == diagnostic.get("image_id"), "Mask scope image order differs")
        scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
        selected = scores >= confidence
        expected = selected if deferred else np.ones(len(scores), dtype=bool)
        flags = diagnostic.get("mask_metrics_computed_by_prediction")
        if flags is None:
            _require(not deferred, "Deferred archive must identify every computed mask prediction")
        else:
            _require(isinstance(flags, list) and len(flags) == len(scores)
                     and all(type(value) is bool for value in flags)
                     and np.array_equal(np.asarray(flags, dtype=bool), expected),
                     "Computed mask flags differ from the declared exact threshold scope")
        if deferred:
            for branch in ("visible", "amodal"):
                assignments = diagnostic.get(branch + "_mask_gt_indices_by_prediction")
                _require(isinstance(assignments, list) and len(assignments) == len(scores)
                         and all(assignments[i] == -1 for i in np.flatnonzero(~selected)),
                         "Unevaluated low-score masks cannot claim GT assignments")
        computed_count += int(expected.sum())
        selected_count += int(selected.sum())
        returned_count += len(scores)
    declared = metadata.get("mask_metrics_scope")
    _require(declared is not None or not deferred, "Deferred archive lacks its declared mask scope")
    if declared is not None:
        expected_scope = {"mode": "deferred_selected" if deferred else "online_all",
                          "raw_predictions": returned_count, "selected_predictions": selected_count,
                          "computed_predictions": computed_count, "confidence": confidence,
                          "unevaluated_assignment_sentinel": -1,
                          "computed_flag": "mask_metrics_computed_by_prediction"}
        _require(isinstance(declared, dict) and all(declared.get(key) == value for key, value in expected_scope.items()),
                 "Declared mask scope counts, threshold or computation convention differ")
    return {"defer_mask_metrics": deferred, "computed_predictions": computed_count,
            "model_returned_predictions": returned_count, "selected_predictions": selected_count,
            "all_selected_predictions_computed": True,
            "all_original_GT_participated": True,
            "unselected_mask_assignment_minus1_is_uncomputed": deferred}


def _collect(records, truth, confidence, *, diagnostics=None, pool_bound=False):
    _require(isinstance(records, list) and [r.get("image_id") for r in records] == list(truth), "Archive is incomplete, reordered or has duplicate images")
    if diagnostics is not None:
        _require([r.get("image_id") for r in diagnostics] == list(truth), "Mask diagnostics inventory differs")
    outcomes, fps, per_image, all_scores, all_flags = [], [], [], [], []
    mask_tp = Counter()
    for index, record in enumerate(records):
        rows = truth[record["image_id"]]
        result, image_fp, item, assignments = analyze_image(record, rows, confidence, pool_bound=pool_bound)
        scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
        selected = scores >= confidence
        all_scores.extend(scores.tolist())
        all_flags.extend((assignments >= 0).tolist())
        if diagnostics is not None:
            diagnostic = diagnostics[index]
            _require(diagnostic.get("scores") == record["pred"]["scores"] and diagnostic.get("gt_count") == len(rows)
                     and diagnostic.get("apple_ids_in_target_order") == [x["apple_id"] for x in rows], "Mask diagnostics score/GT identity differs")
            box_assignment = _assignments(diagnostic.get("box_gt_indices_by_prediction"), len(scores), len(rows), "BOX")
            _require(np.array_equal(box_assignment, assignments), "Stored BOX assignment differs from exact stable replay")
            for branch in ("visible", "amodal"):
                vector = _assignments(diagnostic.get(branch + "_mask_gt_indices_by_prediction"), len(scores), len(rows), branch)
                hits = set(vector[selected]) - {-1}
                mask_tp[branch] += len(hits)
                for g, row in enumerate(result):
                    row[branch + "_mask_recalled_at_same_threshold"] = g in hits
        outcomes.extend(result)
        fps.extend(image_fp)
        per_image.append(item)
    tp, fp, gt = sum(x["TP"] for x in per_image), len(fps), sum(len(x) for x in truth.values())
    replay = _counts(tp, fp, gt, confidence)
    replay.update(images=len(records), empty_gt_images=sum(not x for x in truth.values()),
                  false_positives_on_empty_gt_images=sum(x["empty_GT_image"] for x in fps))
    return {"box_replay": replay, "ground_truth_instances": outcomes, "false_positive_predictions": fps,
            "per_image": per_image, "calibration_replay": _selected_threshold(all_scores, all_flags, gt),
            "same_threshold_mask_PR": {branch: _counts(mask_tp[branch], tp + fp - mask_tp[branch], gt, confidence)
                                       for branch in ("visible", "amodal")} if diagnostics is not None else None}


def _group_report(rows, key, parent=None):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    result = {}
    for name, values in groups.items():
        tp = sum(x["recalled_at_selected_threshold"] for x in values)
        item = {"GT": len(values), "TP": tp, "FN": len(values) - tp, "recall": tp / len(values),
                "GT_with_archived_IoU50_candidate": sum(x["archived_IoU50_candidate_count"] > 0 for x in values),
                "FN_low_score_only": sum(x["error_category"] == FN_LOW for x in values)}
        if parent is not None:
            old_hits = [parent[(x["image_id"], x["gt_index"])]["recalled_at_selected_threshold"] for x in values]
            gained = sum(x["recalled_at_selected_threshold"] and not old for x, old in zip(values, old_hits, strict=True))
            lost = sum(old and not x["recalled_at_selected_threshold"] for x, old in zip(values, old_hits, strict=True))
            item.update(parent_TP=sum(old_hits), parent_recall=sum(old_hits) / len(values), gained_TP=gained,
                        lost_TP=lost, net_TP_change=gained - lost, recall_change_percentage_points=100 * (tp - sum(old_hits)) / len(values))
        result[name] = item
    return result


def analyze_od_archive(validation_dir, *, provenance_path, metrics_path, parent_raw=None, parent_summary=None,
                       parent_raw_sha256=None, pool_bound=False):
    """Read a complete archive and return an independent report; never run VAL.

    Parent comparison accepts the legacy R50 summary or another completed OD
    summary, always reading the exact confidence from that summary. Its original
    GT inventory and calibrated BOX replay must agree with the same provenance.
    """
    validation_dir = Path(validation_dir).expanduser().resolve()
    paths = {name: validation_dir / (name + ".json") for name in ("summary", "evaluation_metadata", "raw_predictions", "mask_pair_diagnostics")}
    paths.update(provenance=Path(provenance_path).expanduser().resolve(), reference_metrics=Path(metrics_path).expanduser().resolve())
    _require((parent_raw is None) == (parent_summary is None), "Supply both --parent-raw and --parent-summary")
    _require(parent_raw is not None or parent_raw_sha256 is None, "Parent SHA requires parent raw archive")
    if parent_raw is not None:
        paths.update(parent_raw=Path(parent_raw).expanduser().resolve(), parent_summary=Path(parent_summary).expanduser().resolve())
    before = {name: {"path": str(path), "sha256": _sha(path)} for name, path in paths.items()}
    _require(before["provenance"]["sha256"] == PROVENANCE_SHA256, "Original VAL provenance SHA256 differs")
    _require(before["reference_metrics"]["sha256"] == METRICS_SHA256, "Reference matching metrics SHA256 differs")
    provenance, summary, metadata = _read(paths["provenance"]), _read(paths["summary"]), _read(paths["evaluation_metadata"])
    images, truth = _truth_contract(provenance)
    point, confidence = _validate_completion(summary, metadata, provenance)
    _validate_identity_audit(metadata, images, truth)
    raw_records, mask_diagnostics = _read(paths["raw_predictions"]), _read(paths["mask_pair_diagnostics"])
    mask_scope = _validate_mask_metric_scope(metadata, raw_records, mask_diagnostics, confidence)
    current = _collect(raw_records, truth, confidence, diagnostics=mask_diagnostics, pool_bound=pool_bound)
    _same_point(current["box_replay"], point)
    _same_point(current["calibration_replay"], point)
    _require(current["box_replay"]["false_positives_on_empty_gt_images"] == point.get("false_positives_on_empty_gt_images"), "Empty-image FP replay differs")
    for branch in ("visible", "amodal"):
        _same_point(current["same_threshold_mask_PR"][branch], point.get(branch + "_instance_mask_pr_iou50", {}))
    parent_map, comparison = None, None
    if parent_raw is not None:
        _require(parent_raw_sha256 is None or before["parent_raw"]["sha256"] == parent_raw_sha256, "Parent raw SHA256 differs")
        parent_document = _read(paths["parent_summary"])
        old_point, old_confidence = _point(parent_document)
        parent_format = _parent_protocol(parent_document)
        if parent_format == "formal_OD":
            paths["parent_evaluation_metadata"] = paths["parent_summary"].parent / "evaluation_metadata.json"
            before["parent_evaluation_metadata"] = {"path": str(paths["parent_evaluation_metadata"]), "sha256": _sha(paths["parent_evaluation_metadata"])}
            parent_metadata = _read(paths["parent_evaluation_metadata"])
            _validate_completion(parent_document, parent_metadata, provenance)
            _validate_identity_audit(parent_metadata, images, truth)
        old = _collect(_read(paths["parent_raw"]), truth, old_confidence)
        _same_point(old["box_replay"], old_point)
        _same_point(old["calibration_replay"], old_point)
        parent_map = {(x["image_id"], x["gt_index"]): x for x in old["ground_truth_instances"]}
        comparison = {"archive_format": parent_format, "confidence_exact": old_confidence, "box_replay": old["box_replay"],
                      "GT_transitions": _group_report(current["ground_truth_instances"], lambda x: "all", parent_map)["all"],
                      "precision_change_percentage_points": 100 * (point["precision"] - old_point["precision"]),
                      "recall_change_percentage_points": 100 * (point["recall"] - old_point["recall"]),
                      "filtering_policy_from_parent_summary": parent_document.get("inference"),
                      "interpretation": "Each model uses its own exact full-VAL selected threshold; original GT identities and group membership are identical. This comparison does not isolate architectural causes."}
    outcomes = current["ground_truth_instances"]
    gt_counts, fp_counts = Counter(x["error_category"] for x in outcomes), Counter(x["category"] for x in current["false_positive_predictions"])
    pool = {"raw_predictions": sum(x["raw_predictions"] for x in current["per_image"]),
            "GT_with_any_IoU50_candidate": sum(x["archived_IoU50_candidate_count"] > 0 for x in outcomes),
            "filtering_policy_from_metadata": metadata.get("postprocess_policy"), "limitations": POOL_LIMITATIONS,
            "maximum_matching_computed": bool(pool_bound)}
    if pool_bound:
        bound = sum(x["filtered_raw_pool_maximum_matching_TP"] for x in current["per_image"])
        pool.update(filtered_raw_pool_maximum_matching_TP=bound, filtered_raw_pool_maximum_matching_recall_bound=bound / 3449,
                    filtered_selected_pool_maximum_matching_TP=sum(x["filtered_selected_pool_maximum_matching_TP"] for x in current["per_image"]))
    after = {name: {"path": str(path), "sha256": _sha(path)} for name, path in paths.items()}
    _require(before == after, "Consumed archive/provenance/reference/comparison inputs changed during analysis")
    return {"schema": "apple_od_archived_error_diagnosis_v1", "phase": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
            "validation_directory": str(validation_dir), "scope": {"images": 814, "GT": 3449, "empty_images": 13,
            "annotation_area_anomalies_retained": 14, "GT_modified": False}, "confidence_exact": confidence,
            "input_sha256_before": before, "input_sha256_after": after, "archived_inputs_unchanged": True,
            "mask_metric_computation_scope": mask_scope,
            "model_initialization_provenance_from_metadata": metadata.get("model_initialization_provenance"),
            "mask_metric_replay_method": "Replay archived exact independent V/A mask assignment arrays at the selected BOX confidence; no pixel masks are recomputed by this CPU archive tool.",
            "evaluation_source_sha256_recorded": metadata["source_sha256_before"],
            "source_hash_policy": "Recorded historical source snapshots must match before/after; current archive/provenance/reference inputs are rehashed. Historical source absolute paths are not reopened or required to equal current training files.",
            **current, "GT_outcome_categories": dict(gt_counts), "FN_categories": {k: gt_counts[k] for k in (FN_LOW, FN_COMPETITION, FN_MISSING)},
            "FP_categories": {k: fp_counts[k] for k in (FP_BACKGROUND, FP_LOCALIZATION, FP_DUPLICATE)},
            "FP_category_definitions": {FP_BACKGROUND: "max IoU<.1, including all predictions on empty GT images; geometric category, not semantic label",
                FP_LOCALIZATION: ".1<=max IoU<.5", FP_DUPLICATE: "max IoU>=.5 for a selected unmatched prediction; the qualifying GT was already consumed"},
            "archived_filtered_pool": pool, "parent_comparison": comparison,
            "occlusion_groups": _group_report(outcomes, lambda x: x["occlusion_stratum"], parent_map),
            "original_visible_size_groups": _group_report(outcomes, lambda x: visible_size_bin(x["visible_area_pixels"]), parent_map),
            "fixed_800_equivalent_visible_size_groups": {"definition": "sqrt(original visible mask area)*800/1300; half-open [16,32) group contains the same original389 GT. Independent of actual model resize/input preprocessing.",
                "groups": _group_report(outcomes, lambda x: visible_size_bin(x["visible_area_pixels"], equivalent_short_side=800), parent_map)},
            "execution": {"CPU_only": True, "model_imported_or_executed": False, "GPU_jobs": False,
                          "pairwise_IoU_computations_per_image_per_archive": 1, "prediction_pair_IoU_or_QxQ_assignment": False}}


def write_report(report, output_path):
    """Create one independent JSON; reject overwrite and paths inside VAL."""
    output = Path(output_path).expanduser().resolve()
    _require(not output.is_relative_to(Path(report["validation_directory"]).resolve()), "Write diagnosis outside the original validation directory")
    if "parent_raw" in report["input_sha256_before"]:
        _require(not output.is_relative_to(Path(report["input_sha256_before"]["parent_raw"]["path"]).resolve().parent),
                 "Write diagnosis outside the parent raw archive directory")
    _require(all(output != Path(x["path"]).resolve() for x in report["input_sha256_before"].values()), "Cannot replace an input artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return output
