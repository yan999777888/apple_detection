"""Compare completed query VAL JSON with fixed PApple03/e6/power0 JSON.

No model, checkpoint, Torch, image decode, dataset scan or threshold selection.
Each arm uses its own already-selected single global full-VAL confidence.
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
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUERY_RUN = ROOT / "runs/query_dual_papple_train01_20261021"
OLD = ROOT / "temp/context_dual_papple_quality_candidate_20261021/papple03_e006_full01/power_0"
PROVENANCE = ROOT / "reports/papple_val_provenance_20261021.json"
METRICS = ROOT / "scripts/metrics.py"
ATOMIC = ROOT / "scripts/atomic_io.py"
EVALUATOR = HERE / "query_dual_evaluate.py"
PINS = {
    PROVENANCE: "fcd96d08c75e1663c83efecb8eb321bbc10220dc74314c8227f1f8565a2ccb3a",
    METRICS: "1b0a7e385940fd0c6160cc08b75ddaf2e1f43f3ea00f5bd37d547fea6b08010d",
    ATOMIC: "3b5ca14efc337ce3f511f60f1829c14c24e5ab6fc9772714e051bf265e645380",
    EVALUATOR: "52ddf3972d650eeb9ab0ac666d17cbc4949ecbd3e4711f6ef78b8da3e548d629",
}
COUNT_KEYS = ("tp", "fp", "fn", "gt_count", "prediction_count", "images", "empty_gt_images",
              "false_positives_on_empty_gt_images")
RATE_KEYS = ("confidence", "iou_threshold", "precision", "recall", "f1", "min_precision_recall")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def truth_index(provenance):
    if (provenance.get("phase") != "complete" or provenance.get("passed") is not True
            or provenance["scope"]["images"] != 814 or provenance["scope"]["GT_instances"] != 3449
            or provenance["scope"]["empty_GT_images"] != 13):
        raise ValueError("Pinned complete PApple814/3449/13 provenance required")
    images = provenance["image_metadata"]
    truth = defaultdict(list)
    for row in provenance["ground_truth_instances"]:
        truth[row["image_id"]].append(row)
    if len(images) != 814 or len({row["image_id"] for row in images}) != 814:
        raise ValueError("Provenance image inventory differs")
    for index, image in enumerate(images):
        rows = truth[image["image_id"]]
        if (image["image_index"] != index or len(rows) != image["gt_count"]
                or [row["gt_index"] for row in rows] != list(range(len(rows)))
                or [row["apple_id"] for row in rows] != image["apple_ids_in_target_order"]):
            raise ValueError("Original GT identity and order must remain unchanged")
    counts = Counter(row["occlusion_stratum"] for rows in truth.values() for row in rows)
    if sum(counts.values()) != 3449 or dict(counts) != provenance["annotation_diagnostics_GT_only"]["strata_counts"]:
        raise ValueError("Original GT strata inventory differs")
    return images, dict(truth), counts


def load_arm(folder, query, images, truth):
    result = {name: read(folder / filename) for name, filename in (
        ("summary", "summary.json"), ("metadata", "evaluation_metadata.json"),
        ("raw", "raw_predictions.json"), ("diagnostics", "mask_pair_diagnostics.json"))}
    summary, meta, records, diagnostics = (result[key] for key in ("summary", "metadata", "raw", "diagnostics"))
    numeric = summary["summary"]; confidence = summary["confidence"]
    if (not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1
            or numeric["confidence"] != confidence or numeric["iou_threshold"] != .5
            or numeric["images"] != 814 or numeric["gt_count"] != 3449 or numeric["empty_gt_images"] != 13):
        raise ValueError("Each arm requires its actual single global full-VAL threshold and814/3449/13 scope")
    if query:
        if (meta.get("schema") != "query_dual_full_val_evaluation_v1" or meta.get("phase") != "complete"
                or meta.get("dataset_split") != "val" or meta.get("images") != 814 or meta.get("GT_instances") != 3449
                or meta.get("empty_GT_images") != 13 or meta.get("consumed_images") != 814 or meta.get("consumed_GT") != 3449
                or meta.get("no_GT_repair_or_exclusion") is not True or meta.get("subset_filter") is not False
                or meta.get("test_data_read") is not False or meta.get("all_RGB_before_after_verified") is not True
                or meta.get("all_original_GT_numeric_fields_and_mask_SHA256_verified") is not True
                or meta.get("source_sha256_before") != meta.get("source_sha256_after")
                or meta["source_sha256_before"].get(str(EVALUATOR)) != PINS[EVALUATOR]
                or meta["source_sha256_before"].get(str(PROVENANCE)) != PINS[PROVENANCE]
                or meta.get("postprocess_policy", {}).get("raw_score_filter", {}).get("threshold") != .001
                or summary.get("inference", {}).get("dataset_split") != "val"):
            raise ValueError("New query evaluation must actually complete full unfiltered true-GT VAL")
    elif (meta != summary.get("inference") or meta.get("images") != 814
          or meta.get("dataset_split") != "val" or meta.get("raw_confidence") != .001):
        raise ValueError("Fixed old power0 full-VAL inference metadata differs")
    names = [image["image_id"] for image in images]
    if ([row["image_id"] for row in records] != names or [row["image_id"] for row in diagnostics] != names):
        raise ValueError("All original814 images must appear once in original order")
    for record, diagnostic, image in zip(records, diagnostics, images, strict=True):
        rows = truth[record["image_id"]]
        if (record["gt"]["boxes"] != [row["box"] for row in rows]
                or diagnostic["gt_count"] != len(rows) or diagnostic["scores"] != record["pred"]["scores"]
                or query and diagnostic.get("apple_ids_in_target_order") != image["apple_ids_in_target_order"]):
            raise ValueError("Original3449 GT boxes/IDs/order or raw-to-diagnostic alignment differs")
        if any(not .001 <= float(score) <= 1 for score in record["pred"]["scores"]):
            raise ValueError("Saved raw scores must respect the declared.001 cutoff")
    return result


def gt_id(row):
    return {key: row[key] for key in ("image_id", "gt_index", "apple_id", "box",
                                     "visible_area_fraction", "occlusion_stratum")}


def diagnose_arm(arm, truth, metrics):
    confidence = arm["summary"]["confidence"]
    replay = metrics.evaluate_at_threshold(arm["raw"], confidence, .5)
    original = arm["summary"]["summary"]
    if any(replay[key] != original[key] for key in COUNT_KEYS + RATE_KEYS):
        raise ValueError("Exact saved-threshold TP/FP/FN and precision/recall replay failed")
    detected, misses, false_positives = {}, [], []
    coverage = Counter(); fn_categories = Counter(); fp_categories = Counter(); mask_tp = Counter()
    for record, diagnostic in zip(arm["raw"], arm["diagnostics"], strict=True):
        name = record["image_id"]; rows = truth[name]
        boxes, scores = record["pred"]["boxes"], record["pred"]["scores"]
        overlaps = metrics.box_iou(boxes, record["gt"]["boxes"])
        match = metrics.match_image(record["gt"]["boxes"], boxes, scores, .5, confidence)
        assigned = {int(p): int(g) for p, g in zip(match["pred_indices"], match["gt_indices"])}
        for g, row in enumerate(rows):
            candidate = [p for p in range(len(scores)) if overlaps[p, g] >= .5]
            coverage[row["occlusion_stratum"]] += bool(candidate)
        for p, g, iou in zip(match["pred_indices"], match["gt_indices"], match["matched_ious"]):
            p, g = int(p), int(g)
            if g >= 0:
                detected[(name, g)] = {"prediction_index": p, "score": scores[p], "box_iou": float(iou)}
                continue
            best = int(overlaps[p].argmax()) if rows else None
            best_iou = float(overlaps[p, best]) if best is not None else 0.
            category = ("duplicate_GT_IoU_ge_0.50" if best_iou >= .5 else
                        "localization_IoU_0.10_to_below_0.50" if best_iou >= .1 else
                        "low_overlap_unknown_background_or_missing_annotation")
            fp_categories[category] += 1
            false_positives.append({"image_id": name, "prediction_index": p, "box": boxes[p], "score": scores[p],
                "category": category, "maximum_GT_IoU": best_iou, "empty_GT_image": not bool(rows),
                "nearest_GT": gt_id(rows[best]) if best is not None else None})
        for g in match["unmatched_gt_indices"]:
            g = int(g)
            candidates = [p for p in range(len(scores)) if overlaps[p, g] >= .5]
            selected = [p for p in candidates if scores[p] >= confidence]
            category = ("no_raw_IoU50_candidate" if not candidates else
                        "all_IoU50_candidates_below_global_confidence" if not selected else "matching_competition")
            fn_categories[category] += 1
            best = max(candidates, key=lambda p: scores[p]) if candidates else None
            misses.append({**gt_id(rows[g]), "category": category,
                "raw_IoU50_candidate_count": len(candidates), "selected_IoU50_candidate_count": len(selected),
                "highest_score_IoU50_candidate": None if best is None else {"prediction_index": best,
                    "score": scores[best], "IoU": float(overlaps[best, g]), "matched_GT_index": assigned.get(best)},
                "maximum_raw_IoU": float(overlaps[:, g].max()) if len(scores) else 0.})
        for branch in ("visible", "amodal"):
            flags = diagnostic[branch + "_mask_tp_flags"]
            if len(flags) != len(scores) or any(type(flag) is not bool for flag in flags):
                raise ValueError("Independent mask flags must align with every saved raw candidate")
            mask_tp[branch] += sum(flag and score >= confidence for flag, score in zip(flags, scores))
    mask_reports = {}
    for branch in ("visible", "amodal"):
        numeric = original[branch + "_instance_mask_pr_iou50"]
        tp = mask_tp[branch]; count = replay["prediction_count"]
        if (numeric["confidence"] != confidence or numeric["gt_count"] != 3449 or numeric.get("is_mask_ap") is not False
                or numeric["tp"] != tp or numeric["fp"] != count - tp or numeric["fn"] != 3449 - tp
                or numeric["precision"] != (tp / count if count else 0.) or numeric["recall"] != tp / 3449):
            raise ValueError("Saved independent mask P/R disagrees with saved full-VAL flags at the actual global threshold")
        mask_reports[branch] = numeric
    if len(detected) != replay["tp"] or len(misses) != replay["fn"] or len(false_positives) != replay["fp"]:
        raise ValueError("Per-GT/FP diagnosis must partition the exact saved metric counts")
    return {"global_confidence": confidence, "box_metrics": {key: replay[key] for key in COUNT_KEYS + RATE_KEYS},
        "raw_candidate_coverage": {"covered_GT": sum(coverage.values()), "total_GT": 3449,
            "fraction": sum(coverage.values()) / 3449, "covered_GT_by_stratum": dict(coverage),
            "is_recall": False, "definition": "GT with any saved raw.001 box at IoU>=0.5; many GT may share one candidate. Saved candidates already reflect each arm's cap/NMS/empty-A policy."},
        "FN_categories": dict(fn_categories), "missed_GT": misses, "FP_categories": dict(fp_categories),
        "top20_FP_for_review": sorted(false_positives, key=lambda row: -row["score"])[:20],
        "independent_mask_PR_IoU50": mask_reports,
        "mask_PR_evidence": "Existing evaluator's independent per-mask flags replayed at saved BOX confidence; mask pixels are not reloaded or recomputed; this is not mask AP."}, detected


def compare(args):
    if "torch" in sys.modules:
        raise RuntimeError("Run a fresh model-free comparison process")
    sys.dont_write_bytecode = True
    baseline = args.baseline_only
    folder, output = (None if baseline else args.validation_directory.resolve()), args.output.resolve()
    match = None if baseline else re.fullmatch(r"epoch_([0-9]{3})_attempt_[0-9]+", folder.name)
    if not baseline and (folder.parent != QUERY_RUN / "validation" or not match or not 1 <= int(match[1]) <= 6):
        raise ValueError("Only the current fixed query run's completed epoch validation directory is accepted")
    if output.parent != ROOT / "reports" or output.suffix != ".json" or output.exists():
        raise ValueError("Use a fresh JSON directly beneath ROOT/reports")
    files = [*PINS, Path(__file__).resolve(), *(base / name for base in ((OLD,) if baseline else (OLD, folder)) for name in
             ("summary.json", "evaluation_metadata.json", "raw_predictions.json", "mask_pair_diagnostics.json"))]
    before = {str(path): sha(path) for path in files}
    if any(before[str(path)] != digest for path, digest in PINS.items()):
        raise ValueError("Pinned metrics/provenance/evaluator/atomic source changed")
    metrics = module(METRICS, "query_comparison_pinned_metrics")
    provenance = read(PROVENANCE)
    images, truth, strata = truth_index(provenance)
    old = load_arm(OLD, False, images, truth)
    old_result, old_tp = diagnose_arm(old, truth, metrics)
    if baseline:
        groups = {}
        for group, count in strata.items():
            tp = sum((row["image_id"], row["gt_index"]) in old_tp for rows in truth.values()
                     for row in rows if row["occlusion_stratum"] == group)
            groups[group] = {"GT": count, "TP": tp, "FN": count - tp, "recall": tp / count}
        result = {"schema": "fixed_papple03e6_fullval_baseline_diagnosis_v1", "phase": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(), "baseline_only": True,
            "old_fixed_directory": str(OLD), "scope": {"images": 814, "GT_instances": 3449,
                "empty_GT_images": 13, "GT_filter_or_repair": False},
            "old": old_result, "GT_strata": groups, "heavy_occlusion_recall": groups["heavy_visible_lt_0.40"],
            "interpretation": "Saved full-VAL global threshold replay only. FN categories describe retained raw.001 boxes, not architectural cause. Raw candidate coverage is not recall. IoU<0.1 FP remains unknown background or possible missing annotation; mask P/R is not AP."}
        return save_result(output, result, before)
    new = load_arm(folder, True, images, truth)
    new_result, new_tp = diagnose_arm(new, truth, metrics)
    groups = {}
    for group, count in strata.items():
        keys = {(row["image_id"], row["gt_index"]) for rows in truth.values() for row in rows if row["occlusion_stratum"] == group}
        gained, lost = keys & (new_tp.keys() - old_tp.keys()), keys & (old_tp.keys() - new_tp.keys())
        def listed(values):
            return [{**gt_id(truth[name][index]), "old_detection": old_tp.get((name, index)),
                     "new_detection": new_tp.get((name, index))} for name, index in sorted(values)]
        a, b = len(keys & old_tp.keys()), len(keys & new_tp.keys())
        groups[group] = {"GT": count, "old_TP": a, "new_TP": b, "old_recall": a / count,
                         "new_recall": b / count, "recall_delta": (b - a) / count,
                         "gained_GT": listed(gained), "lost_GT": listed(lost),
                         "gained_count": len(gained), "lost_count": len(lost)}
    heavy = groups["heavy_visible_lt_0.40"]
    result = {"schema": "query_vs_fixed_papple03e6_fullval_comparison_v1", "phase": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "query_epoch": int(match[1]),
        "query_validation_directory": str(folder), "old_fixed_directory": str(OLD),
        "scope": {"images": 814, "GT_instances": 3449, "empty_GT_images": 13,
                  "original_GT_boxes_ID_order_equal": True, "GT_filter_or_repair": False},
        "threshold_policy": "Each arm uses its own saved single global full-VAL BOX confidence; no new tuning, per-image thresholds or shared-threshold substitution.",
        "old": old_result, "query": new_result,
        "box_metric_delta_query_minus_old": {key: new_result["box_metrics"][key] - old_result["box_metrics"][key]
            for key in ("tp", "fp", "fn", "precision", "recall", "f1", "min_precision_recall")},
        "GT_strata_gain_loss": groups,
        "heavy_occlusion_recall": {key: heavy[key] for key in ("GT", "old_TP", "new_TP", "old_recall", "new_recall", "recall_delta")},
        "interpretation": {"FN_causes": "Descriptive partition within saved raw.001 predictions, not proof of architectural cause; absent candidates may reflect raw cutoff, top100, old NMS or query empty-A drop.",
            "FP_low_overlap": "IoU<0.1 is unknown background or possible missing annotation; no automatic true-background claim.",
            "FP_duplicate": "FP with IoU>=0.5 to an annotated GT already occupied by one-to-one matching.",
            "limitations": "Exploratory validation comparison at separately selected thresholds; not TEST/generalization evidence, significance test or mask AP."},
        "baseline_only": False}
    return save_result(output, result, before)


def save_result(output, result, before):
    atomic = module(ATOMIC, "query_comparison_atomic")
    after = {path: sha(path) for path in before}
    if after != before or "torch" in sys.modules:
        raise RuntimeError("Consumed source/JSON bytes changed or forbidden Torch import occurred")
    result.update(source_sha256_before=before, source_sha256_after=after, source_before_after_identical=True,
        torch_imported=False, model_executed=False, GPU_used=False, RGB_decoded_or_scanned=False, checkpoint_loaded=False)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    if output.exists():
        raise FileExistsError("Comparison output appeared; never overwrite it")
    atomic.replace_with_retry(temporary, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-directory", type=Path)
    parser.add_argument("--baseline-only", action="store_true", help="Replay only the fixed old full814/3449 arm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.baseline_only == (args.validation_directory is not None):
        parser.error("Choose --baseline-only or one explicit --validation-directory")
    result = compare(args)
    print(json.dumps({"phase": result["phase"], "output": str(args.output.resolve()),
        "baseline_only": result["baseline_only"], "old_FN_categories": result["old"]["FN_categories"],
        "old_FP_categories": result["old"]["FP_categories"],
        "box_delta": result.get("box_metric_delta_query_minus_old")}))


if __name__ == "__main__":
    main()
