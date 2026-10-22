"""Offline comparison of completed residual full-VAL JSON against query epoch 6.

No model/checkpoint/Torch/image access. A replay mode exercises the full fixed
query-e6 evidence path without pretending a residual evaluation has completed.
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
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD_HERE = ROOT / "temp/query_dual_mask_candidate_20261021"
RESIDUAL_HERE = ROOT / "temp/query_residual_candidate_20261022"
QUERY_E6 = ROOT / "runs/query_dual_papple_train01_20261021/validation/epoch_006_attempt_1792605002725864400"
RESIDUAL_VALIDATION = ROOT / "runs/residual_dual_train01_20261022/validation"
PROVENANCE = ROOT / "reports/papple_val_provenance_20261021.json"
METRICS = ROOT / "scripts/metrics.py"
ATOMIC = ROOT / "scripts/atomic_io.py"
OLD_COMPARISON = OLD_HERE / "compare_query_fullval.py"
QUERY_EVALUATOR = OLD_HERE / "query_dual_evaluate.py"
RESIDUAL_EVALUATOR = RESIDUAL_HERE / "residual_dual_evaluate.py"
PINS = {
    PROVENANCE: "fcd96d08c75e1663c83efecb8eb321bbc10220dc74314c8227f1f8565a2ccb3a",
    METRICS: "1b0a7e385940fd0c6160cc08b75ddaf2e1f43f3ea00f5bd37d547fea6b08010d",
    ATOMIC: "3b5ca14efc337ce3f511f60f1829c14c24e5ab6fc9772714e051bf265e645380",
    OLD_COMPARISON: "360c6c600499a326e0e3e49285f9d19f3d009c2735e060979d937d8ac33e1dde",
    QUERY_EVALUATOR: "52ddf3972d650eeb9ab0ac666d17cbc4949ecbd3e4711f6ef78b8da3e548d629",
    RESIDUAL_EVALUATOR: "2b150dd1fec1d5a0ecb0d7c6a7f33ba1df0ca0e66edf53980709ce936c65d407",
    RESIDUAL_HERE / "residual_dual_postprocess.py": "20cf95359a4e7db1f9cd444c602a3fb7a55bbb1a67550e393eee06cbdbade3d7",
    RESIDUAL_HERE / "residual_dual_model.py": "bfacdcb32850d6c13a7c932eaf27860c947af7fb56cc6328d52a569f65b8c1d0",
}
JSON_FILES = {"summary": "summary.json", "metadata": "evaluation_metadata.json",
              "raw": "raw_predictions.json", "diagnostics": "mask_pair_diagnostics.json",
              "occlusion": "occlusion_diagnostics.json"}
HEAVY = "heavy_visible_lt_0.40"
BRANCHES = ("box", "visible_mask", "amodal_mask")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def assert_truth_rows(actual, expected):
    """Compare every frozen field, including BOTH original uint8 mask hashes."""
    require(len(actual) == len(expected), "Full original per-GT inventory required")
    for saved, frozen in zip(actual, expected, strict=True):
        require(all(key in saved and saved[key] == value for key, value in frozen.items()),
                "Original per-GT IDs/order/numbers/visible-or-amodal mask SHA256 differ")


def load_arm(folder, kind, images, truth, provenance, old):
    arm = {key: read(folder / filename) for key, filename in JSON_FILES.items()}
    meta, summary = arm["metadata"], arm["summary"]
    evaluator = RESIDUAL_EVALUATOR if kind == "residual" else QUERY_EVALUATOR
    schema = kind + "_dual_full_val_evaluation_v1"
    require(meta.get("schema") == schema and meta.get("phase") == "complete",
            "Require an actually completed evaluation of the declared architecture")
    for key, expected in {"dataset_split": "val", "images": 814, "GT_instances": 3449,
                          "empty_GT_images": 13, "consumed_images": 814, "consumed_GT": 3449,
                          "GT_coordinates_hw": [1300, 1300], "iou_threshold": .5,
                          "no_GT_repair_or_exclusion": True, "subset_filter": False,
                          "test_data_read": False, "all_RGB_before_after_verified": True,
                          "all_original_GT_numeric_fields_and_mask_SHA256_verified": True}.items():
        require(meta.get(key) == expected, "Incomplete full-VAL evidence: " + key)
    source = meta.get("source_sha256_before", {})
    require(source == meta.get("source_sha256_after"), "Evaluation source-before/after differs")
    for path in (evaluator, METRICS, ATOMIC, PROVENANCE):
        require(source.get(str(path)) == PINS[path], "Evaluation source/provenance pin differs: " + path.name)
    for filename, digest in provenance["binding"]["current_annotation_sha256"].items():
        name = "via_region_data_instance.json" if filename == "visible" else "via_region_data_amodal.json"
        path = ROOT / "data/papple/data/gt_json/val" / name
        require(source.get(str(path)) == digest, "Original polygon annotation hash differs")
    rgb = meta.get("RGB_sha256_before_after", [])
    require(len(rgb) == 814, "All 814 original RGB identity records required")
    for saved, frozen in zip(rgb, images, strict=True):
        require(saved.get("image_id") == frozen["image_id"] and saved.get("path") == frozen["RGB_path"]
                and saved.get("before") == saved.get("after") == frozen["RGB_sha256"],
                "Per-image original RGB identity differs")
    require(meta.get("provenance", {}).get("sha256") == PINS[PROVENANCE], "Unbound provenance")
    policy = meta.get("postprocess_policy", {})
    require(policy.get("raw_score_filter") == {"operator": ">", "threshold": .001, "comparison_dtype": "float32"},
            "Raw class-score policy changed")
    expected_size = 1024 if kind == "residual" else 768
    prep = meta.get("preprocess", {})
    require(prep.get("input_hw") == [expected_size, expected_size]
            and all(prep.get(key) is False for key in ("augment", "padding", "crop")), "Preprocess identity changed")
    if kind == "residual":
        require(policy.get("schema") == "residual_amodal_box_direct_postprocess_v1"
                and policy.get("box_source") == "final amodal_boxes normalized cxcywh; independent of thresholded mask support"
                and policy.get("max_candidates_before_invalid_box_drop") == 100
                and policy.get("empty_amodal", "").startswith("retain valid box")
                and meta.get("auxiliary_supervision_layers") == 1
                and meta.get("annotation_visible_area_exceeds_amodal_GT_retained") == 14,
                "Residual direct-box/empty-mask/auxiliary policy changed")
        for path in (RESIDUAL_HERE / "residual_dual_postprocess.py", RESIDUAL_HERE / "residual_dual_model.py"):
            require(source.get(str(path)) == PINS[path], "Residual source pin differs")
    else:
        require(policy.get("schema") == "query_dual_postprocess_v1"
                and policy.get("max_candidates_before_empty_amodal_drop") == 100, "Query mask-derived-box policy changed")
    confidence = summary.get("confidence")
    require(type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1
            and summary["summary"]["confidence"] == confidence
            and summary.get("inference", {}).get("dataset_split") == "val"
            and summary.get("score") == min(summary["summary"]["precision"], summary["summary"]["recall"]),
            "Invalid saved global BOX confidence/score")
    require(all(summary["inference"].get(key) == meta.get(key)
                for key in ("architecture", "preprocess", "postprocess_policy")), "Summary/metadata inference differs")
    names = [row["image_id"] for row in images]
    require([row["image_id"] for row in arm["raw"]] == names
            and [row["image_id"] for row in arm["diagnostics"]] == names, "814 images/order differ")
    for record, diagnostic, frozen in zip(arm["raw"], arm["diagnostics"], images, strict=True):
        rows = truth[record["image_id"]]
        require(record["gt"]["boxes"] == [row["box"] for row in rows]
                and diagnostic["gt_count"] == len(rows)
                and diagnostic.get("apple_ids_in_target_order") == frozen["apple_ids_in_target_order"]
                and diagnostic["scores"] == record["pred"]["scores"], "Per-image original GT/raw alignment differs")
        scores = record["pred"]["scores"]
        require(len(scores) == len(record["pred"]["boxes"]) <= 100 and scores == sorted(scores, reverse=True)
                and all(type(score) in (int, float) and math.isfinite(score) and .001 < score <= 1 for score in scores)
                and diagnostic.get("labels") == [1] * len(scores), "Candidate scores/order/labels/cap differ")
        for branch in BRANCHES:
            assignment = diagnostic[branch + "_gt_indices_by_prediction"]
            require(len(assignment) == len(scores) and all(type(g) is int and -1 <= g < len(rows) for g in assignment)
                    and len([g for g in assignment if g >= 0]) == len({g for g in assignment if g >= 0}),
                    "Each branch requires independent one-to-one GT assignment")
            if branch != "box":
                require(diagnostic[branch + "_tp_flags"] == [g >= 0 for g in assignment], "Mask flags/assignments differ")
    occ = arm["occlusion"]
    assert_truth_rows(occ["ground_truth_instances"], provenance["ground_truth_instances"])
    require(occ.get("schema") == kind + "_dual_full_val_occlusion_recall_v1"
            and occ.get("confidence") == confidence and occ.get("iou_threshold") == .5
            and occ.get("no_GT_repair_or_exclusion") is True
            and occ.get("grouping_source", {}).get("sha256") == PINS[PROVENANCE]
            and occ.get("annotation_diagnostics_GT_only") == provenance["annotation_diagnostics_GT_only"],
            "Occlusion GT binding differs")
    arm["identity"] = {"kind": kind, "directory": str(folder), "architecture": meta["architecture"],
                       "preprocess": prep, "postprocess_policy": policy,
                       "GT_and_original_VA_mask_hashes_equal": True, "RGB_identity_equal": True}
    return arm


def endpoint_envelope(curve):
    """Keep each endpoint's P/R/confidence together; never combine thresholds."""
    high_p = [row for row in curve if row["precision"] >= .98]
    high_r = [row for row in curve if row["recall"] >= .95]
    return {"max_recall_at_precision_ge_0_98": {
                "attainable": bool(high_p), "point": max(high_p, key=lambda row: (row["recall"], row["precision"], row["confidence"])) if high_p else None},
            "max_precision_at_recall_ge_0_95": {
                "attainable": bool(high_r), "point": max(high_r, key=lambda row: (row["precision"], row["recall"], row["confidence"])) if high_r else None},
            "both_precision_ge_0_98_and_recall_ge_0_95_at_one_threshold": any(row["precision"] >= .98 and row["recall"] >= .95 for row in curve),
            "candidate_thresholds": len(curve),
            "scope": "Exploratory full-VAL envelope over distinct retained raw scores; whole score ties. Each point is one attainable threshold. Not deployment threshold selection, TEST evidence, or AP."}


def groups_at(arm, truth, confidence):
    groups = {row["occlusion_stratum"]: {"GT": 0, **{branch + "_TP": 0 for branch in BRANCHES}}
              for rows in truth.values() for row in rows}
    detected = {branch: {} for branch in BRANCHES}
    for record, diagnostic in zip(arm["raw"], arm["diagnostics"], strict=True):
        name = record["image_id"]
        for row in truth[name]:
            groups[row["occlusion_stratum"]]["GT"] += 1
        for branch in BRANCHES:
            for p, g in enumerate(diagnostic[branch + "_gt_indices_by_prediction"]):
                if g >= 0 and record["pred"]["scores"][p] >= confidence:
                    groups[truth[name][g]["occlusion_stratum"]][branch + "_TP"] += 1
                    detected[branch][(name, g)] = {"prediction_index": p, "score": record["pred"]["scores"][p]}
    for group in groups.values():
        for branch in BRANCHES:
            group[branch + "_recall"] = group[branch + "_TP"] / group["GT"]
    return groups, detected


def analyze(arm, truth, confidence, metrics, old, check_saved=False):
    # Reuse the pinned old comparator's exact greedy-box and independent-mask
    # replay/FN/FP partitions. Only override its expected summary for a common
    # threshold, explicitly retaining the original saved confidence separately.
    view = arm
    if not check_saved:
        view = {**arm, "summary": {"confidence": confidence,
                "summary": metrics.evaluate_at_threshold(arm["raw"], confidence, .5)}}
        count = view["summary"]["summary"]["prediction_count"]
        for branch in ("visible", "amodal"):
            tp = sum(flag and score >= confidence for row in arm["diagnostics"]
                     for flag, score in zip(row[branch + "_mask_tp_flags"], row["scores"], strict=True))
            numeric = metrics._counts(tp, count - tp, 3449, confidence, .5)
            numeric.update(is_mask_ap=False)
            view["summary"]["summary"][branch + "_instance_mask_pr_iou50"] = numeric
    result, box_detected = old.diagnose_arm(view, truth, metrics)
    groups, detected = groups_at(arm, truth, confidence)
    require(set(detected["box"]) == set(box_detected)
            and all(all(detected["box"][key][field] == box_detected[key][field]
                        for field in ("prediction_index", "score")) for key in box_detected),
            "Saved per-prediction box assignment disagrees with actual greedy replay")
    if check_saved:
        occ = arm["occlusion"]
        for group, values in groups.items():
            for branch in BRANCHES:
                require(occ["groups"][group][branch + "_tp"] == values[branch + "_TP"]
                        and occ["groups"][group][branch + "_recall"] == values[branch + "_recall"], "Saved group recall differs")
        for row in occ["ground_truth_instances"]:
            for branch in BRANCHES:
                require(row[branch + "_recalled"] == ((row["image_id"], row["gt_index"]) in detected[branch]),
                        "Saved per-GT recall differs")
    result.update(GT_strata_recall=groups, heavy_occlusion_recall=groups[HEAVY],
                  threshold_shared_by_BOX_V_A=True, identity=arm["identity"])
    return result, detected


def gain_loss(baseline, candidate, truth):
    result = {}
    for branch in BRANCHES:
        old_keys, new_keys = set(baseline[branch]), set(candidate[branch])
        def listed(keys):
            return [{**truth[name][index], "baseline_detection": baseline[branch].get((name, index)),
                     "candidate_detection": candidate[branch].get((name, index))} for name, index in sorted(keys)]
        result[branch] = {"gained_count": len(new_keys - old_keys), "lost_count": len(old_keys - new_keys),
                          "gained_GT": listed(new_keys - old_keys), "lost_GT": listed(old_keys - new_keys)}
    return result


def compare(args):
    require("torch" not in sys.modules, "Run a fresh model-free comparison process")
    sys.dont_write_bytecode = True
    replay = args.replay_query_e6
    folder = QUERY_E6 if replay else args.validation_directory.resolve()
    if not replay:
        match = re.fullmatch(r"epoch_([0-9]{3})_attempt_[0-9]+", folder.name)
        require(folder.parent == RESIDUAL_VALIDATION and match is not None and 1 <= int(match[1]) <= 12,
                "Only this fixed residual run's explicit completed full-VAL epoch is accepted")
    output = args.output.resolve()
    require(output.parent in (HERE, ROOT / "reports") and output.suffix == ".json" and not output.exists(),
            "Use a fresh JSON in the comparison directory or ROOT/reports")
    files = [*PINS, Path(__file__).resolve(), *{base / filename for base in (QUERY_E6, folder) for filename in JSON_FILES.values()}]
    before = {str(path): sha(path) for path in files}
    require(all(before[str(path)] == digest for path, digest in PINS.items()), "Pinned comparison dependency changed")
    old = module(OLD_COMPARISON, "_residual_comparison_old_helpers")
    metrics = module(METRICS, "_residual_comparison_metrics")
    provenance = read(PROVENANCE)
    images, truth, strata = old.truth_index(provenance)
    baseline = load_arm(QUERY_E6, "query", images, truth, provenance, old)
    candidate = baseline if replay else load_arm(folder, "residual", images, truth, provenance, old)
    arms = {"query_e6_baseline": baseline, "query_e6_replay" if replay else "residual": candidate}
    native, common, native_detected, common_detected, envelopes = {}, {}, {}, {}, {}
    common_confidence = baseline["summary"]["confidence"]
    for name, arm in arms.items():
        native[name], native_detected[name] = analyze(arm, truth, arm["summary"]["confidence"], metrics, old, check_saved=True)
        common[name], common_detected[name] = analyze(arm, truth, common_confidence, metrics, old)
        calibrated = metrics.calibrate_threshold(arm["raw"], .5, return_curve=True)
        require(calibrated["confidence"] == arm["summary"]["confidence"], "Saved threshold is not the complete raw-record optimum")
        require(all(calibrated["summary"][key] == arm["summary"]["summary"][key] for key in old.COUNT_KEYS + old.RATE_KEYS),
                "Full threshold-curve replay differs from saved summary")
        envelopes[name] = endpoint_envelope(calibrated["curve"])
    candidate_name = "query_e6_replay" if replay else "residual"
    gains = {"saved_per_arm_global_confidence": gain_loss(native_detected["query_e6_baseline"], native_detected[candidate_name], truth),
             "common_query_e6_global_confidence": gain_loss(common_detected["query_e6_baseline"], common_detected[candidate_name], truth)}
    result = {"schema": "residual_vs_query_e6_fullval_comparison_v1", "phase": "complete",
              "created_utc": datetime.now(timezone.utc).isoformat(), "replay_only": replay,
              "residual_result_compared": not replay, "candidate": candidate_name,
              "scope": {"images": 814, "GT_instances": 3449, "empty_GT_images": 13,
                        "strata_counts": dict(strata), "annotation_anomalies_retained": 14,
                        "original_GT_order_numbers_and_VA_mask_hashes_equal": True, "GT_filter_or_repair": False},
              "at_saved_per_arm_global_BOX_confidence": native,
              "at_common_query_e6_global_BOX_confidence": {"confidence": common_confidence, "arms": common},
              "per_GT_gained_lost": gains, "full_VAL_BOX_threshold_envelope": envelopes,
              "interpretation": {"thresholds": "Saved per-arm BOX operating points and a separate common fixed query-e6 confidence are both reported. Every BOX/V/A triplet uses one identical confidence. Common-threshold comparison is descriptive; architectures have different score calibration.",
                  "geometry": "Query-e6 boxes are derived from binary A masks; residual boxes are the saved direct geometry-head predictions. Neither is reconstructed from masks here.",
                  "mask_metrics": "Independent visible/amodal IoU50 matches replayed from saved per-prediction assignments, never conditional box-TP mask IoU or mask AP. No mask pixels are reloaded.",
                  "occlusion": "GT recall only, including all 14 area anomalies. FP has no assignable GT occlusion group.",
                  "causality": "FP/FN partitions describe saved raw candidates and policies; they do not identify architectural causes. Low-overlap FP may be background or missing annotation.",
                  "endpoints": "P>=.98/maxR and R>=.95/maxP retain their own complete P/R/threshold tuple. They must not be combined as a single achieved operating point.",
                  "limits": "Exploratory full VAL only, no TEST/generalization/significance claim; joint geometry/aux/input/optimizer recipe comparison cannot isolate a causal component."}}
    after = {path: sha(path) for path in before}
    require(after == before and "torch" not in sys.modules, "Consumed source/JSON changed or Torch imported")
    result.update(source_sha256_before=before, source_sha256_after=after, source_before_after_identical=True,
                  torch_imported=False, model_executed=False, GPU_used=False, checkpoint_loaded=False,
                  RGB_decoded_or_scanned=False, original_data_or_frozen_sources_modified=False)
    atomic = module(ATOMIC, "_residual_comparison_atomic")
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    require(not output.exists(), "Output appeared; never overwrite")
    atomic.replace_with_retry(temporary, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-directory", type=Path)
    parser.add_argument("--replay-query-e6", action="store_true", help="Actual old JSON replay; not a residual success claim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replay_query_e6 == (args.validation_directory is not None):
        parser.error("Choose --replay-query-e6 or one --validation-directory")
    result = compare(args)
    print(json.dumps({"phase": result["phase"], "replay_only": result["replay_only"],
                      "residual_result_compared": result["residual_result_compared"], "output": str(args.output.resolve()),
                      "threshold_envelope": result["full_VAL_BOX_threshold_envelope"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
