"""Fixed CPU-friendly contract for native/shared-ONNX output comparisons.

No threshold changes, filtering, extra NMS, model inference or filesystem writes.
Standard Results are materialized without modifying the caller's arrays/tensors.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import math

CONFIDENCE = 0.25
BOUNDARY_EPS = 1e-4
BOX_ATOL = 0.1
SCORE_ATOL = 1e-4
MATCH_IOU_MIN = 0.99
MASK_IOU_MIN = 0.999
RAW_SHAPES = {
    "stage_detections": (1, 300, 38),
    "stage_prototypes": (1, 32, 160, 160),
    "defect_detections": (1, 300, 6),
}
NAMES = {
    "stage": {0: "immature apple", 1: "semi-mature apple", 2: "mature apple"},
    "defect": {0: "class_0", 1: "class_1"},
}
NATIVE_SEGMENT_PREDICT_SHA256 = "2f80c259782f330b12b3f4b5eb11345416b83e99d6af86da3bbe594089bde01c"
PROTOCOL = {
    "schema": "dual_head_native_onnx_output_contract_v1",
    "input": [1, 3, 640, 640], "dtype": "float32", "output_shapes": {k: list(v) for k, v in RAW_SHAPES.items()},
    "confidence": CONFIDENCE, "confidence_operator": ">", "boundary_epsilon": BOUNDARY_EPS,
    "matching": "Class-constrained Hungarian minimum 1-IoU; canonical box/score order, mask-SHA tie-break for identical boxes/scores, then original index",
    "matching_box_iou_min": MATCH_IOU_MIN, "box_coordinate_atol_px": BOX_ATOL,
    "score_atol": SCORE_ATOL, "binary_mask_iou_min": MASK_IOU_MIN,
    "count_or_class_count_changes_allowed": False, "threshold_boundary_exceptions": False,
    "native_masks_required": True, "no_extra_nms_primary": True,
    "native_empty_mask_semantics": "Without ReID: all empty masks are filtered with their boxes; mixed empty masks are filtered; no input detections has masks=None, while mask-filtered-empty has an empty mask tensor.",
    "native_segment_predict_sha256": NATIVE_SEGMENT_PREDICT_SHA256,
}


def _array(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def validate_raw_outputs(outputs):
    """Check three fixed FP32 outputs; return a JSON-safe report, never repair.

    Accepts a mapping with RAW_SHAPES keys or a tuple/list in that exact order.
    No rowwise native/ONNX comparison is done: top-k ordering may differ.
    """
    import numpy as np
    issues, records = [], {}
    if isinstance(outputs, dict):
        if set(outputs) != set(RAW_SHAPES):
            issues.append("raw_output_names_mismatch")
        values = outputs
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        values = dict(zip(RAW_SHAPES, outputs))
    else:
        return {"passed": False, "issues": ["raw_output_container_mismatch"], "outputs": {}, "protocol": PROTOCOL}
    for name, expected_shape in RAW_SHAPES.items():
        if name not in values:
            continue
        try:
            a = _array(values[name])
            numeric = np.issubdtype(a.dtype, np.number)
            finite = bool(np.isfinite(a).all()) if numeric else False
            record = {"shape": list(a.shape), "dtype": str(a.dtype), "finite": finite}
            records[name] = record
            if a.shape != expected_shape:
                issues.append(name + ":shape_mismatch")
            if a.dtype != np.float32:
                issues.append(name + ":requires_float32")
            if not finite:
                issues.append(name + ":nonfinite_or_nonnumeric")
            if name.endswith("detections") and a.shape == expected_shape and numeric and finite:
                boxes, scores, classes = a[0, :, :4], a[0, :, 4], a[0, :, 5]
                nc = 3 if name.startswith("stage") else 2
                if not bool(((scores >= 0) & (scores <= 1)).all()):
                    issues.append(name + ":invalid_scores")
                if not bool(((classes == np.floor(classes)) & (classes >= 0) & (classes < nc)).all()):
                    issues.append(name + ":invalid_classes")
                boundary = np.flatnonzero(np.abs(scores.astype(np.float64) - CONFIDENCE) <= BOUNDARY_EPS)
                record.update(score_min=float(scores.min()), score_max=float(scores.max()),
                    above_confidence_count=int((scores > CONFIDENCE).sum()),
                    at_confidence_count=int((scores == CONFIDENCE).sum()),
                    threshold_boundary_rows=[{"index": int(i), "score": float(scores[i]), "class": int(classes[i]) if float(classes[i]).is_integer() else float(classes[i])} for i in boundary])
        except (ValueError, TypeError, RuntimeError) as exc:
            issues.append(name + ":materialization_failed")
            records[name] = {"error": str(exc)}
    return {"passed": not issues, "issues": sorted(set(issues)), "outputs": records, "protocol": PROTOCOL}


def _materialize(result, task):
    import numpy as np
    issues = []
    if task not in NAMES:
        raise ValueError("task must be stage or defect")
    try:
        names = {int(k): str(v) for k, v in result.names.items()}
        if names != NAMES[task]:
            issues.append("class_name_metadata_mismatch")
        shape = tuple(int(x) for x in result.orig_shape)
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("Expected positive native image H,W")
        boxes = _array(result.boxes.data)
        if boxes.ndim != 2 or boxes.shape[1] != 6:
            raise ValueError("Expected standard untracked Results boxes Nx6")
        if not np.issubdtype(boxes.dtype, np.number) or not bool(np.isfinite(boxes).all()):
            raise ValueError("Result boxes must be finite numeric values")
        if boxes.dtype != np.float32:
            issues.append("result_boxes_requires_float32")
        boxes = boxes.astype(np.float64, copy=False)
        if not bool(((boxes[:, 4] > CONFIDENCE) & (boxes[:, 4] <= 1)).all()):
            issues.append("result_confidence_outside_fixed_native_threshold")
        if not bool(((boxes[:, 5] == np.floor(boxes[:, 5])) & (boxes[:, 5] >= 0) & (boxes[:, 5] < len(NAMES[task]))).all()):
            issues.append("result_class_outside_source_authority")
        if not bool(((boxes[:, 2] >= boxes[:, 0]) & (boxes[:, 3] >= boxes[:, 1])).all()):
            issues.append("result_box_has_negative_extent")
        masks = None if result.masks is None else _array(result.masks.data)
        state = "none" if masks is None else "empty_tensor" if masks.shape[0] == 0 else "nonempty_tensor"
        if task == "defect" and masks is not None:
            issues.append("detection_head_unexpected_masks")
        if task == "stage":
            if masks is None:
                if len(boxes):
                    issues.append("stage_boxes_missing_masks")
            elif masks.ndim != 3 or masks.shape != (len(boxes), *shape):
                issues.append("stage_mask_native_shape_or_count_mismatch")
            elif not bool(np.isfinite(masks).all()) or not bool(((masks == 0) | (masks == 1)).all()):
                issues.append("stage_masks_not_finite_binary")
            elif len(boxes) and not bool(masks.reshape(len(boxes), -1).any(axis=1).all()):
                issues.append("stage_contains_empty_mask_after_native_filter")
        valid = not issues
        return {"valid": valid, "issues": issues, "boxes": boxes, "masks": masks,
                "orig_shape": shape, "mask_state": state, "names": names}
    except (AttributeError, ValueError, TypeError, RuntimeError, IndexError) as exc:
        return {"valid": False, "issues": ["invalid_results_contract"], "error": str(exc)}


def _ious(a, b):
    import numpy as np
    left = np.maximum(a[:, None, :2], b[None, :, :2])
    right = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    intersection = np.maximum(right - left, 0).prod(axis=2)
    areas_a = np.maximum(a[:, 2:4] - a[:, :2], 0).prod(axis=1)
    areas_b = np.maximum(b[:, 2:4] - b[:, :2], 0).prod(axis=1)
    union = areas_a[:, None] + areas_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _ordered(boxes, masks, cls):
    import numpy as np
    indices = [i for i, row in enumerate(boxes) if int(row[5]) == cls]
    base = {i: tuple(float(v) for v in boxes[i, :5]) for i in indices}
    occurrences = Counter(base.values())
    def key(i):
        tie = ""
        if occurrences[base[i]] > 1 and masks is not None:
            tie = hashlib.sha256(np.ascontiguousarray(masks[i], dtype=bool).tobytes()).hexdigest()
        return (*base[i], tie, i)
    return sorted(indices, key=key)


def compare_results(native_result, onnx_result, task):
    """Compare one native/ONNX Results pair without modifying/filtering either.

    Count changes always fail, even within 1e-4 of the fixed confidence boundary.
    Hungarian association uses only class/box IoU; masks follow matched indices.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    a = _materialize(native_result, task)
    b = _materialize(onnx_result, task)
    report = {"task": task, "protocol": PROTOCOL,
              "native_contract": {k: v for k, v in a.items() if k not in ("boxes", "masks")},
              "onnx_contract": {k: v for k, v in b.items() if k not in ("boxes", "masks")}}
    if not a["valid"] or not b["valid"]:
        return {**report, "passed": False, "issues": ["invalid_native_or_onnx_results"], "matches": []}
    boxes_a, boxes_b, masks_a, masks_b = a["boxes"], b["boxes"], a["masks"], b["masks"]
    issues, matches = [], []
    counts_a = Counter(int(x) for x in boxes_a[:, 5])
    counts_b = Counter(int(x) for x in boxes_b[:, 5])
    if a["orig_shape"] != b["orig_shape"]:
        issues.append("native_image_shape_changed")
    if a["mask_state"] != b["mask_state"]:
        issues.append("mask_representation_changed")
    if len(boxes_a) != len(boxes_b):
        issues.append("detection_count_changed")
    if counts_a != counts_b:
        issues.append("class_counts_changed")
    eligible_a, eligible_b = set(), set()
    for cls in sorted(set(counts_a) | set(counts_b)):
        ai, bi = _ordered(boxes_a, masks_a, cls), _ordered(boxes_b, masks_b, cls)
        if not ai or not bi:
            continue
        ious = _ious(boxes_a[ai, :4], boxes_b[bi, :4])
        # Ineligible associations are retained as failures for diagnostics, never
        # silently counted as a matched object.
        costs = np.where(ious >= MATCH_IOU_MIN, 1 - ious, 1e6)
        rows, cols = linear_sum_assignment(costs)
        for r, c in zip(rows, cols):
            i, j = ai[int(r)], bi[int(c)]
            overlap = float(ious[r, c])
            coordinate_error = float(np.abs(boxes_a[i, :4] - boxes_b[j, :4]).max())
            score_error = abs(float(boxes_a[i, 4] - boxes_b[j, 4]))
            failures = []
            if overlap < MATCH_IOU_MIN:
                failures.append("matching_iou")
            else:
                eligible_a.add(i); eligible_b.add(j)
            if coordinate_error > BOX_ATOL:
                failures.append("box_coordinate")
            if score_error > SCORE_ATOL:
                failures.append("score")
            mask_iou = xor = None
            if task == "stage" and masks_a is not None and masks_b is not None:
                if masks_a[i].shape != masks_b[j].shape:
                    failures.append("mask_native_shape")
                else:
                    ma, mb = masks_a[i].astype(bool, copy=False), masks_b[j].astype(bool, copy=False)
                    union = int(np.count_nonzero(ma | mb))
                    intersection = int(np.count_nonzero(ma & mb))
                    mask_iou = intersection / union if union else 1.0
                    xor = union - intersection
                    if mask_iou < MASK_IOU_MIN:
                        failures.append("mask_iou")
            matches.append({"native_index": i, "onnx_index": j, "class_id": cls,
                "box_iou": overlap, "box_max_abs_error_px": coordinate_error,
                "score_abs_error": score_error, "mask_iou": mask_iou,
                "mask_xor_pixels": xor, "failures": failures})
            if failures:
                issues.append("matched_output_tolerance_failed")
    unmatched_a = sorted(set(range(len(boxes_a))) - eligible_a)
    unmatched_b = sorted(set(range(len(boxes_b))) - eligible_b)
    if unmatched_a or unmatched_b:
        issues.append("unmatched_detections")
    def boundary(boxes):
        return [{"index": i, "class_id": int(row[5]), "score": float(row[4])}
                for i, row in enumerate(boxes) if abs(float(row[4]) - CONFIDENCE) <= BOUNDARY_EPS]
    return {**report, "passed": not issues, "issues": sorted(set(issues)),
        "native_count": len(boxes_a), "onnx_count": len(boxes_b),
        "native_class_counts": {str(k): v for k, v in sorted(counts_a.items())},
        "onnx_class_counts": {str(k): v for k, v in sorted(counts_b.items())},
        "native_threshold_boundary": boundary(boxes_a), "onnx_threshold_boundary": boundary(boxes_b),
        "unmatched_native_indices": unmatched_a, "unmatched_onnx_indices": unmatched_b,
        "native_boxes_xyxy_score_class": boxes_a.tolist(), "onnx_boxes_xyxy_score_class": boxes_b.tolist(),
        "matches": matches}
