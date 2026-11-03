"""CPU-only PApple apparent-diameter diagnostic, never a 3D-center benchmark.

Compares unfiltered mean, the author's strict mean +/- 2*std trimmed mean,
and a predeclared median-depth variant. Human V/A are an oracle-input control;
the teacher branch reuses historical Base original-view FP16 T15 ROI outputs.
Diameter GT is accessed only after estimates exist and is never a fit prior.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
METHODS = ("unfiltered_mean", "author_trimmed_mean", "predeclared_median")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "strong_runtime/src"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path,
                   default=REPO / "reports/local_rgbd_20261103/data_sample_v1/sample_manifest.json")
    p.add_argument("--teacher-dir", type=Path,
                   default=REPO / "runs/stage2_t15_paired_20261102/base")
    p.add_argument("--output", type=Path,
                   default=REPO / "reports/local_rgbd_20261103/diameter_v2")
    p.add_argument("--author-source", type=Path,
                   default=REPO / "reports/local_rgbd_20261103/diameter_v2/upstream_utils_diameter.py")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--self-check-only", action="store_true")
    return p


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(path, evidence):
    path = Path(path).resolve()
    row = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
    require(str(path) not in evidence or evidence[str(path)] == row,
            f"Input changed while reading: {path}")
    evidence[str(path)] = row
    return row


def read_json(path, evidence):
    bind(path, evidence)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def estimate_diameter(depth, visible, amodal, focal_px):
    """No annotations/labels/GT diameter accepted by this estimator."""
    import numpy as np
    require(depth.shape == visible.shape == amodal.shape, "Depth/mask shape mismatch")
    require(np.isfinite(focal_px) and focal_px > 0, "Invalid focal length")
    v, a = np.asarray(visible, bool), np.asarray(amodal, bool)
    finite_positive = np.isfinite(depth) & (depth > 0)
    selected = np.asarray(depth[v & finite_positive], np.float64)
    v_count, a_count = int(v.sum()), int(a.sum())
    info = {"visible_area_pixels": v_count, "amodal_area_pixels": a_count,
            "visible_outside_amodal_pixels": int((v & ~a).sum()),
            "visible_fraction": v_count / a_count if a_count else None,
            "valid_depth_count_in_visible": int(selected.size),
            "invalid_depth_fraction_in_visible": 1 - selected.size / v_count if v_count else None,
            "invalid_depth_fraction_whole_frame": 1 - float(finite_positive.mean()),
            "focal_px": float(focal_px), "depth_unit": "metre", "diameter_unit": "mm"}
    # Preserve the author's exact positive-depth selection (including +inf),
    # strict '<', NumPy population std, and zero return for an empty trimmed set.
    # The two diagnostic variants independently exclude non-finite depth.
    author_input = depth[(depth > 0) & v]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if author_input.size:
            author_mean = np.mean(author_input)
            author_std = np.std(author_input)
            trimmed = author_input[np.abs(author_input - author_mean) < 2 * author_std]
        else:
            author_mean, author_std = np.nan, np.nan
            trimmed = author_input
    author_value = (float(np.sqrt(4 / (np.pi * np.square(focal_px))) * 1000
                          * trimmed.mean() * np.sqrt(a_count)) if trimmed.size else 0.)
    author_valid = bool(trimmed.size and a_count and math.isfinite(author_value) and author_value > 0)
    author_detail = {"initial_positive_depth_count": int(author_input.size),
                     "initial_nonfinite_count": int((~np.isfinite(author_input)).sum()),
                     "initial_mean_m": float(author_mean) if np.isfinite(author_mean) else None,
                     "initial_std_m_ddof0": float(author_std) if np.isfinite(author_std) else None,
                     "strict_rule": "abs(z - mean(z)) < 2 * std(z, ddof=0)",
                     "retained_count": int(trimmed.size),
                     "retained_mean_m": float(trimmed.mean()) if trimmed.size else None,
                     "zero_std_behavior": "strict rule retains no points and original author function returns 0",
                     "empty_trimmed_return_mm": 0., "valid_measurement": author_valid}
    if not v_count or not a_count or not selected.size:
        return {**info, "status": "invalid_empty_mask_or_no_depth",
                "author_rule": author_detail,
                "method_validity": {"unfiltered_mean": False, "author_trimmed_mean": author_valid,
                                    "predeclared_median": False},
                "estimates_mm": {"unfiltered_mean": None, "author_trimmed_mean": author_value,
                                 "predeclared_median": None}}
    equivalent_diameter_px = 2 * math.sqrt(a_count / math.pi)
    depth_mean, depth_median = float(selected.mean()), float(np.median(selected))
    return {**info, "status": "estimated", "equivalent_amodal_diameter_px": equivalent_diameter_px,
            "depth_mean_m": depth_mean, "depth_median_m": depth_median,
            "valid_depth_quantiles_m": np.quantile(selected, [0, .1, .5, .9, 1]).tolist(),
            "author_rule": author_detail,
            "method_validity": {"unfiltered_mean": True, "author_trimmed_mean": author_valid,
                                "predeclared_median": True},
            "estimates_mm": {"unfiltered_mean": equivalent_diameter_px * depth_mean * 1000 / focal_px,
                             "author_trimmed_mean": author_value,
                             "predeclared_median": equivalent_diameter_px * depth_median * 1000 / focal_px}}


def self_check():
    import numpy as np
    # Known area and constant depth; zeros, negative and NaN must not enter mean.
    depth = np.array([[2., 2., 0.], [np.nan, -2., 5.]])
    visible = np.array([[1, 1, 1], [1, 1, 0]], bool)
    amodal = np.ones((2, 3), bool)
    result = estimate_diameter(depth, visible, amodal, 1000.)
    expected = 4 * math.sqrt(6 / math.pi)
    require(result["valid_depth_count_in_visible"] == 2, "Invalid-depth exclusion failed")
    require(abs(result["estimates_mm"]["unfiltered_mean"] - expected) < 1e-12,
            "Metric unit/area formula check failed")
    require(result["estimates_mm"]["unfiltered_mean"] == result["estimates_mm"]["predeclared_median"],
            "Constant-depth mean/median check failed")
    empty = estimate_diameter(np.zeros((2, 3)), visible, amodal, 1000.)
    require(not any(empty["method_validity"].values()), "No-depth case must be invalid")
    require(empty["estimates_mm"]["author_trimmed_mean"] == 0,
            "Author empty-depth sentinel must be preserved")
    require(result["estimates_mm"]["author_trimmed_mean"] == 0
            and not result["method_validity"]["author_trimmed_mean"],
            "Author strict zero-std behavior must not be silently repaired")
    boundary = estimate_diameter(np.array([[1., 1., 1., 1., 6.]]),
                                 np.ones((1, 5), bool), np.ones((1, 5), bool), 1000.)
    require(boundary["author_rule"]["retained_count"] == 4
            and boundary["author_rule"]["initial_std_m_ddof0"] == 2.,
            "Strict two-sigma boundary exclusion differs from original author rule")
    return {"passed": True, "checks": ["known_area_metric_formula", "invalid_depth_exclusion",
                                        "constant_depth_mean_median", "empty_depth_rejection",
                                        "author_zero_std_returns_zero", "author_strict_2sigma_boundary"]}


def nonzero_box(mask):
    import numpy as np
    ys, xs = np.nonzero(mask)
    require(xs.size > 0, "Empty GT mask")
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def save_masks(output, stem, visible, amodal, **arrays):
    import numpy as np
    path = output / "masks" / (stem + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, visible=np.asarray(visible, np.uint8),
                        amodal=np.asarray(amodal, np.uint8), **arrays)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
            "format": "native-image-resolution uint8 binary V/A; additional ROI arrays for teacher"}


def evaluate_after_estimation(estimate, gt_diameter_mm):
    require(math.isfinite(gt_diameter_mm) and gt_diameter_mm > 0, "Invalid evaluation label")
    if estimate.get("estimates_mm") is None:
        return None
    return {name: ({"signed_error_mm": value - gt_diameter_mm,
                   "absolute_error_mm": abs(value - gt_diameter_mm),
                   "absolute_percentage_error": abs(value - gt_diameter_mm) / gt_diameter_mm * 100}
                  if estimate["method_validity"][name] else None)
            for name, value in estimate["estimates_mm"].items()}


def statistics(rows):
    import numpy as np
    answer = []
    for source in ("human_GT_VA_oracle", "historical_Base_original_FP16_T15"):
        for split in ("train", "val"):
            subset = [r for r in rows if r["input_source"] == source and r["split"] == split]
            if not subset:
                continue
            for method in METHODS:
                valid = [r for r in subset if r.get("evaluation") is not None
                         and r["evaluation"].get(method) is not None]
                errors = np.asarray([r["evaluation"][method]["signed_error_mm"] for r in valid])
                by_apple = defaultdict(list)
                for r in valid:
                    by_apple[r["diameter_key"]].append(r["evaluation"][method]["absolute_error_mm"])
                groups = {}
                for group in ("heavy", "moderate", "light", "mostly_visible"):
                    group_rows = [r for r in valid if r["GT_visibility_group"] == group]
                    groups[group] = {"sample_count": len(group_rows),
                        "unique_apple_count": len({r["diameter_key"] for r in group_rows}),
                        "mae_mm": float(np.mean([r["evaluation"][method]["absolute_error_mm"]
                                                for r in group_rows])) if group_rows else None}
                answer.append({"input_source": source, "split": split, "method": method,
                    "eligible_diameter_samples": len(subset), "estimated_samples": len(valid),
                    "missing_or_invalid_samples": len(subset) - len(valid),
                    "eligible_unique_apples": len({r["diameter_key"] for r in subset}),
                    "estimated_unique_apples": len(by_apple),
                    "mae_mm": float(np.mean(np.abs(errors))) if len(errors) else None,
                    "rmse_mm": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
                    "bias_mm": float(np.mean(errors)) if len(errors) else None,
                    "max_absolute_error_mm": float(np.max(np.abs(errors))) if len(errors) else None,
                    "equal_apple_weight_mae_mm": float(np.mean([np.mean(v) for v in by_apple.values()])) if by_apple else None,
                    "occlusion_groups": groups,
                    "interpretation": "Conditional on successful detection and valid depth for teacher; oracle uses human masks; diagnostic sample only."})
    return answer


def main(argv=None):
    args = parser().parse_args(argv)
    require(args.cpu_threads > 0, "cpu-threads must be positive")
    checks = self_check()
    if args.self_check_only:
        print(json.dumps(checks))
        return 0
    import numpy as np
    import torch
    from PIL import Image
    from apple_seg.data import _via_images, _polygons, _rasterize
    from apple_seg.od_roi import paste_roi_mask
    torch.set_num_threads(args.cpu_threads)
    output = args.output.resolve()
    require(os.name != "nt" or output.drive.upper() == "F:", "All outputs must stay on F:")
    output.mkdir(parents=True, exist_ok=True)
    evidence = {}
    author_receipt = bind(args.author_source, evidence)
    require(author_receipt["sha256"] == "d83678a855a15e095bfd3d97bb04df17fd9d296658bed98524e4768d5dce0c84",
            "Audited upstream diameter function snapshot differs")
    manifest_path, teacher = args.manifest.resolve(), args.teacher_dir.resolve()
    manifest = read_json(manifest_path, evidence)
    require(manifest.get("center_ground_truth_available") is False,
            "This diagnostic expects diameter labels only; no center-error claim is implemented")
    samples = manifest["samples"]
    require(len(samples) == manifest["sample_count"] and samples, "Invalid sample manifest")
    require(len({(r["split"], r["image"]) for r in samples}) == len(samples), "Duplicate sample image")
    metadata = read_json(teacher / "prediction_metadata.json", evidence)
    prediction_path = teacher / "original_predictions.json"
    records = read_json(prediction_path, evidence)
    require(metadata.get("phase") == "complete" and metadata.get("model") == "base"
            and metadata.get("roi_masks_saved") is True, "Expected completed historical Base ROI outputs")
    require(sha(prediction_path) == metadata["predictions_sha256"]["original"], "Teacher JSON receipt mismatch")
    by_image = {r["image_id"]: (i, r) for i, r in enumerate(records)}
    require(len(by_image) == len(records), "Duplicate teacher image IDs")
    metrics_path = REPO / "strong_runtime/reference/local_amodal/scripts/metrics.py"
    bind(metrics_path, evidence)
    spec = importlib.util.spec_from_file_location("frozen_diameter_match_metrics", metrics_path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    for path in (Path(__file__), REPO / "strong_runtime/src/apple_seg/data.py",
                 REPO / "strong_runtime/src/apple_seg/od_roi.py"):
        bind(path, evidence)
    annotation_cache, rows, detection_images = {}, [], []
    for sample_index, sample in enumerate(samples):
        image_name, split = sample["image"], sample["split"]
        source_receipts = {}
        for field in ("rgb_path", "depth_path", "visible_annotation", "amodal_annotation", "focal_path"):
            source_receipts[field] = bind(sample[field], evidence)
        for field in ("rgb", "depth"):
            require(source_receipts[field + "_path"]["sha256"] == sample[field + "_sha256"],
                    f"Manifest {field} hash mismatch: {image_name}")
        with Image.open(sample["rgb_path"]) as rgb:
            width, height = rgb.size
        depth = np.load(sample["depth_path"], allow_pickle=False)
        require(depth.shape == (height, width), "Depth is not aligned to supplied RGB shape")
        vpath, apath = sample["visible_annotation"], sample["amodal_annotation"]
        for path in (vpath, apath):
            if path not in annotation_cache:
                annotation_cache[path] = _via_images(path)
        vp, ap = (_polygons(annotation_cache[path][image_name]) for path in (vpath, apath))
        ids = sorted(set(vp) | set(ap))
        require(set(vp) == set(ap), "Unpaired GT V/A instances must not be silently discarded")
        target_id = str(sample["target_apple_id"])
        require(target_id in ids, "Manifest target ID missing from annotations")
        vgt = _rasterize(vp[target_id], width, height).astype(bool)
        agt = _rasterize(ap[target_id], width, height).astype(bool)
        require(int(vgt.sum()) == sample["visible_pixels"] and int(agt.sum()) == sample["amodal_pixels"],
                "GT raster area differs from data audit")
        common = {"sample_index": sample_index, "image": image_name, "split": split,
                  "apple_id": target_id, "diameter_key": sample["diameter_key"],
                  "GT_visibility": sample["visibility"], "GT_visibility_group": sample["visibility_group"],
                  "sources": source_receipts}
        start = time.perf_counter()
        oracle = estimate_diameter(depth, vgt, agt, float(sample["focal_px"]))
        compute_ms = (time.perf_counter() - start) * 1000
        # Evaluation label is intentionally accessed only after the estimate.
        truth = float(sample["diameter_gt_mm"])
        oracle_masks = save_masks(output, f"{sample_index:02d}_oracle_{Path(image_name).stem}_{target_id}", vgt, agt)
        rows.append({**common, "input_source": "human_GT_VA_oracle", "estimate": oracle,
                     "diameter_gt_mm": truth, "evaluation": evaluate_after_estimation(oracle, truth),
                     "mask_artifact": oracle_masks, "estimation_cpu_ms": compute_ms,
                     "timing_scope": "single CPU formula call; excludes IO, rasterization, segmentation; not FPS benchmark"})
        if split != "val":
            continue
        require(image_name in by_image, "Validation image missing from historical teacher cache")
        val_index, record = by_image[image_name]
        gt_boxes = [nonzero_box(_rasterize(ap[i], width, height)) for i in ids]
        require(gt_boxes == record["gt"]["boxes"], "Frozen teacher GT bbox/order differs from original annotation")
        archive_path = teacher / "roi_masks/original" / f"{val_index:06d}.npz"
        receipt = bind(archive_path, evidence)
        with np.load(archive_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in ("boxes", "scores", "visible", "amodal")}
        require(np.array_equal(arrays["boxes"].astype(np.float64), np.asarray(record["pred"]["boxes"]))
                and np.array_equal(arrays["scores"].astype(np.float64), np.asarray(record["pred"]["scores"])),
                "Historical ROI coordinates/scores differ from JSON")
        for key in ("visible", "amodal"):
            value = arrays[key]
            require(value.shape == (len(arrays["scores"]), 56, 56) and np.isfinite(value).all()
                    and ((value >= 0) & (value <= 1)).all(), "Invalid historical ROI probabilities")
        match = metrics.match_image(gt_boxes, arrays["boxes"], arrays["scores"], .5, .75)
        assignment = {int(g): (int(p), float(iou)) for p, g, iou in
                      zip(match["pred_indices"], match["gt_indices"], match["matched_ious"]) if g >= 0}
        target_index = ids.index(target_id)
        detection_images.append({"image": image_name, "historical_val_index": val_index,
            "GT_instances": len(ids), "predictions_ge_075": len(match["pred_indices"]),
            "TP": len(assignment), "FP": len(match["pred_indices"]) - len(assignment),
            "FN": len(match["unmatched_gt_indices"]),
            "unmatched_GT_apple_ids": [ids[int(i)] for i in match["unmatched_gt_indices"]],
            "diameter_target_apple_id": target_id, "diameter_target_matched": target_index in assignment,
            "matches": [{"gt_index": g, "apple_id": ids[g], "prediction_index": p, "box_iou": iou}
                        for g, (p, iou) in assignment.items()]})
        predicted = {**common, "input_source": "historical_Base_original_FP16_T15",
                     "diameter_gt_mm": truth, "historical_val_index": val_index,
                     "prediction_json": evidence[str(prediction_path)], "ROI_archive": receipt,
                     "target_GT_index": target_index, "evaluation": None}
        if target_index not in assignment:
            rows.append({**predicted, "status": "target_not_detected_at_frozen_threshold",
                         "estimate": None, "mask_artifact": None})
            continue
        pred_index, iou = assignment[target_index]
        native_masks = []
        projection_start = time.perf_counter()
        for key in ("visible", "amodal"):
            crop, window = paste_roi_mask(torch.from_numpy(arrays[key][pred_index]),
                                          arrays["boxes"][pred_index], [height, width], mask_threshold=.5)
            x0, y0, x1, y1 = window
            native = np.zeros((height, width), bool)
            native[y0:y1, x0:x1] = crop.numpy()
            native_masks.append(native)
        projection_ms = (time.perf_counter() - projection_start) * 1000
        start = time.perf_counter()
        estimated = estimate_diameter(depth, *native_masks, float(sample["focal_px"]))
        compute_ms = (time.perf_counter() - start) * 1000
        mask_file = save_masks(output, f"{sample_index:02d}_teacher_{Path(image_name).stem}_{target_id}",
            *native_masks, roi_visible=arrays["visible"][pred_index], roi_amodal=arrays["amodal"][pred_index],
            box_xyxy=arrays["boxes"][pred_index], score=arrays["scores"][pred_index],
            window_xyxy=np.asarray(window, np.int64))
        rows.append({**predicted, "status": "box_matched", "prediction_index": pred_index,
            "score": float(arrays["scores"][pred_index]), "box_xyxy": arrays["boxes"][pred_index].tolist(),
            "match_box_iou": iou, "estimate": estimated,
            "evaluation": evaluate_after_estimation(estimated, truth), "mask_artifact": mask_file,
            "native_ROI_projection_cpu_ms": projection_ms, "estimation_cpu_ms": compute_ms,
            "timing_scope": "single CPU paste/formula call; excludes model, IO; not FPS benchmark"})
    counts = Counter(s["diameter_key"] for s in samples)
    detection_total = {key: sum(r[key] for r in detection_images)
                       for key in ("GT_instances", "predictions_ge_075", "TP", "FP", "FN")}
    target_count = len(detection_images)
    target_matched = sum(r["diameter_target_matched"] for r in detection_images)
    detection_total.update({"images": len(detection_images), "diameter_eligible_targets": target_count,
                            "diameter_targets_matched": target_matched,
                            "diameter_targets_missed": target_count - target_matched,
                            "fixed_confidence": .75, "box_iou_threshold": .5})
    summary = {"schema": "papple_diameter_diagnostic_v2", "phase": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "cpu_only": True,
        "training_performed": False, "new_teacher_inference_performed": False,
        "teacher_scope": "Historical 2026-11-02 Base original view, FP16 T15, paired-inference output branch; not four-view fused masks",
        "formula": "D_mm=2*sqrt(A_area_pixels/pi)*depth_statistic_in_metres*1000/focal_px; author branch uses original depth>0 then strict 2sigma rule, diagnostic mean/median use finite positive depth",
        "formula_source": "https://github.com/GRAP-UdL-AT/Amodal_Fruit_Sizing/blob/main/utils/utils_diameter.py",
        "methods_predeclared": list(METHODS),
        "v1_correction": "v1 author_mean was actually an unfiltered mean. v1 numbers remain unchanged; v2 renames it and adds the exact author two-sigma rule, without parameter tuning.",
        "author_source": author_receipt,
        "author_rule_scope": "Reproduces original diameter-estimator depth rule and zero sentinel only; GT rasterization and frozen bbox matching remain our declared evaluation contract, not the author's entire paper evaluation pipeline.",
        "invalid_author_output_policy": "Original zero sentinel is recorded verbatim, flagged invalid, and excluded from conditional MAE with the failure count retained; not silently replaced by an unfiltered mean.",
        "parameter_tuning_performed": False, "diameter_GT_usage": "evaluation only, after estimation",
        "center_error_available": False, "center_error_reason": "No 3D center GT; missing principal point/crop offsets; no camera-frame center calculated",
        "sample_images": len(samples), "unique_apple_IDs": len(counts),
        "repeated_apple_views": {k: v for k, v in counts.items() if v > 1},
        "split_counts": dict(Counter(s["split"] for s in samples)),
        "independent_test_claimed": False, "data_source": manifest.get("depth_provenance"),
        "oracle_interpretation": "Human V/A input-quality reference, not a model prediction nor a guaranteed numerical upper bound",
        "teacher_detection_diagnostic": detection_total, "teacher_per_image": detection_images,
        "statistics": statistics(rows), "rows": rows, "self_check": checks,
        "ROI_projection": "apple_seg.od_roi.paste_roi_mask, align_corners=False, image pixel centres, threshold0.5",
        "input_provenance": evidence,
        "teacher_checkpoint_recorded_sha256": metadata.get("checkpoint_sha256"),
        "teacher_metadata": evidence[str(teacher / "prediction_metadata.json")],
        "limitations": ["Eight selected images are a plumbing/measurement diagnostic, not final test evidence.",
                        "Same physical apples may repeat across images; report unique IDs and apple-balanced MAE.",
                        "Teacher diameter errors are conditional on box matching and valid depth; misses remain in counts.",
                        "Ground-truth amodal contours are human annotations, not measured hidden 3D shape.",
                        "Depth is photogrammetric SfM/MVS, not D405 measurements.",
                        "Unfiltered mean, exact author trimmed mean and median are all reported without picking the better one.",
                        "No RANSAC/LM sphere-center error is inferred from these diameter estimates."]}
    for record in evidence.values():
        require(sha(record["path"]) == record["sha256"], "Input changed during diagnostic")
    summary["inputs_unchanged_after_run"] = True
    (output / "diameter_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    table = ["| 输入 | split | 深度统计 | 有效/应评样本 | 独立果数 | MAE mm | RMSE mm |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in summary["statistics"]:
        def number(key):
            return "—" if row[key] is None else f"{row[key]:.4f}"
        table.append(f"| {row['input_source']} | {row['split']} | {row['method']} | "
                     f"{row['estimated_samples']}/{row['eligible_diameter_samples']} | {row['estimated_unique_apples']} | "
                     f"{number('mae_mm')} | {number('rmse_mm')} |")
    command = f'"{sys.executable}" "{Path(__file__).resolve()}" --manifest "{manifest_path}" --teacher-dir "{teacher}" --author-source "{args.author_source.resolve()}" --output "{output}" --cpu-threads {args.cpu_threads}'
    md = "# PApple 果径 CPU 诊断\n\n" + "\n".join(table)
    md += (f"\n\n共 {len(samples)} 张诊断图、{len(counts)} 个独立果实；重复果实：{dict((k,v) for k,v in counts.items() if v>1)}。"
           f"\n\n旧教师四张原图的检测统计：TP={detection_total['TP']}，FP={detection_total['FP']}，FN={detection_total['FN']}。"
           f"有果径真值的目标 {target_count} 个，匹配 {target_matched} 个，漏检 {target_count-target_matched} 个。"
           "\n\n教师统计仅为旧 Base 单原图分支 FP16 T15 缓存复算，未重新推理，非四视图融合掩膜。"
           "匹配后的果径 MAE 是条件指标，不能把未匹配目标视作定位成功。Human GT V/A 是人工掩膜输入参考，不是模型结果。"
           "\n\n公式：D_mm = 2 × sqrt(A面积/π) × V内有效正有限深度统计值(m) × 1000 / f。"
           "未筛选均值、作者严格两倍标准差筛选后均值和中位数三种统计全部报告；真实果径只用于计算后的误差评价。"
           "\n\nv1中的 author_mean 名称不准确，实际为未筛选均值；v1数值保持不变。v2将其改名unfiltered_mean，另补author_trimmed_mean，未调参。"
           "作者规则为先取depth>0且在V内的值，再保留abs(z-mean)<2*std（ddof=0）的值。std=0时原函数会返回0；这里保留0并标记失败，不伪装成有效尺寸。"
           "原函数不显式排除正无穷；原样分支保留该行为，另外两种诊断分支使用有限正深度。本批深度没有NaN/Inf，因此不影响实际样本。"
           "\n\n数据只有真实果径，没有三维果心真值；缺少可靠主点和原图裁剪偏移，未计算或报告果心毫米误差。"
           "深度来自摄影测量 SfM/MVS，不能冒充 D405 实机效果。"
           "\n\n每个样本的输入路径/hash、GT ID、遮挡分组、无效深度比例、估计值、误差和 V/A 数组均保存在 JSON 与 masks/。"
           "CPU 单次耗时仅作管线诊断，不是完整定位 FPS。"
           f"\n\n复算命令：\n\n```text\n{command}\n```\n")
    (output / "diameter_report.md").write_text(md, encoding="utf-8")
    (output / "reproduce_command.txt").write_text(command + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "statistics": summary["statistics"],
                      "teacher_detection": detection_total}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
