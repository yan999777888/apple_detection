"""CPU counterfactual: filter original FP32 post-NMS caches, then fuse.

This is NOT checkpoint inference, FP16 validation, or a speed benchmark. It
isolates deletion of low-score cached members; the original unfiltered fusion
is read from its already-validated cache and is never recomputed here.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS_SHA = "1b0a7e385940fd0c6160cc08b75ddaf2e1f43f3ea00f5bd37d547fea6b08010d"
FUSION_SHA = "db2ad2578b0b1aa900e0f44c048349caf6f5110dd716d90a4e6aa235f734c131"
PROVENANCE_SHA = "fcd96d08c75e1663c83efecb8eb321bbc10220dc74314c8227f1f8565a2ccb3a"
VIEWS = ["base/original", "base/hflip", "small/original", "small/hflip"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_functions():
    metrics_path = ROOT / "strong_runtime/reference/local_amodal/scripts/metrics.py"
    fusion_path = ROOT / "strong_runtime/tmp/flip_tta.py"
    assert sha(metrics_path) == METRICS_SHA
    assert sha(fusion_path) == FUSION_SHA
    spec = importlib.util.spec_from_file_location("cached_cutoff_frozen_metrics", metrics_path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    # Compile the two exact, hash-bound pure NumPy function bodies, without
    # executing flip_tta's model/dataset imports. No algorithm is reimplemented.
    tree = ast.parse(fusion_path.read_text(encoding="utf-8-sig"), filename=str(fusion_path))
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("_rescale", "fuse_entries")]
    assert [n.name for n in definitions] == ["_rescale", "fuse_entries"]
    namespace = {"np": np, "math": math, "metrics": metrics}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(fusion_path), "exec"), namespace)
    return metrics, namespace["fuse_entries"]


def scope(records, ids, truth, metrics):
    assert len(records) == 814 and [r["image_id"] for r in records] == ids
    for record in records:
        gt, _, _ = metrics._normalize_record(record)
        expected = np.asarray(truth[record["image_id"]], dtype=np.float64).reshape(-1, 4)
        assert np.array_equal(gt, expected), record["image_id"]
    assert sum(len(r["gt"]["boxes"]) for r in records) == 3449
    assert sum(not r["gt"]["boxes"] for r in records) == 13


def add_identity(digest, image_id, indices, record):
    for index in indices:
        identity = [image_id, int(index), record["pred"]["boxes"][index], record["pred"]["scores"][index]]
        digest.update(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")


def compare_images(original, candidate, original_conf, candidate_conf, metrics):
    changed_prediction, changed_count, changed_metrics = [], [], []
    details = []
    for old, new in zip(original, candidate):
        assert old["image_id"] == new["image_id"] and old["gt"] == new["gt"]
        pairs = []
        for record, confidence in [(old, original_conf), (new, candidate_conf)]:
            pred = record["pred"]
            match = metrics.match_image(record["gt"]["boxes"], pred["boxes"], pred["scores"], .5, confidence)
            indices = match["pred_indices"].tolist()
            tp = int(match["tp"].sum())
            pairs.append({"tp": tp, "fp": len(indices) - tp,
                          "fn": len(record["gt"]["boxes"]) - tp,
                          "prediction_count": len(indices),
                          "matched_GT": sorted(int(x) for x in match["gt_indices"] if x >= 0),
                          "boxes": [pred["boxes"][i] for i in indices],
                          "scores": [pred["scores"][i] for i in indices]})
        a, b = pairs
        image_id = old["image_id"]
        if a["boxes"] != b["boxes"] or a["scores"] != b["scores"]:
            changed_prediction.append(image_id)
        if a["prediction_count"] != b["prediction_count"]:
            changed_count.append(image_id)
        if any(a[k] != b[k] for k in ("tp", "fp", "fn")):
            changed_metrics.append(image_id)
            details.append({"image_id": image_id,
                "original": {k: a[k] for k in ("tp", "fp", "fn", "prediction_count")},
                "filtered": {k: b[k] for k in ("tp", "fp", "fn", "prediction_count")},
                "newly_recalled_GT_indices": sorted(set(b["matched_GT"]) - set(a["matched_GT"])),
                "newly_missed_GT_indices": sorted(set(a["matched_GT"]) - set(b["matched_GT"]))})
    return {"original_confidence": original_conf, "filtered_confidence": candidate_conf,
            "selected_boxes_or_scores_changed_image_ids": changed_prediction,
            "selected_count_changed_image_ids": changed_count,
            "TP_FP_FN_changed_image_ids": changed_metrics,
            "TP_FP_FN_changed_details": details}


def audit(baseline, output, provenance_path):
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    if output.exists():
        raise FileExistsError("Use a fresh audit output directory")
    metrics, fuse_entries = load_functions()
    gate_path = ROOT / "reports/strong_c0c1_20261030/baseline_gate.json"
    gate = read(gate_path)
    assert gate["passed"] is True and (gate["images"], gate["GT"], gate["empty"]) == (814, 3449, 13)
    assert Path(gate["baseline_directory"]).resolve() == baseline.resolve()
    assert gate["base_checkpoint_sha256"] == "0fe257c5bfe115a77ac7deac830af479e4c2b0ad6ff69e609901b24c7980fa9b"
    assert gate["small_checkpoint_sha256"] == "e0c1f659c2509d7a01d617e12cb83c673768b440ea5b4b96cb4f763e4207e7e3"
    input_hashes = {}
    for relative, expected in gate["artifact_sha256"].items():
        actual = sha(baseline / relative)
        assert actual == expected, relative
        input_hashes[relative] = actual
    for relative, expected in gate["source_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
    for model in ("base", "small"):
        meta = read(baseline / model / "prediction_metadata.json")
        assert meta["phase"] == "complete" and meta["FP32"] is True
        assert meta["from_fresh_weight_inference"] and meta["source_before_after_identical"]
        assert (meta["images"], meta["GT"], meta["empty"]) == (814, 3449, 13)
        assert meta["checkpoint_sha256"] == gate[model + "_checkpoint_sha256"]
        assert meta["provenance_sha256"] == PROVENANCE_SHA
    assert sha(provenance_path) == PROVENANCE_SHA
    provenance = read(provenance_path)
    ids = [x["image_id"] for x in provenance["image_metadata"]]
    assert len(ids) == len(set(ids)) == 814 and all(ids)
    truth = {i: [] for i in ids}
    for instance in provenance["ground_truth_instances"]:
        assert instance["gt_index"] == len(truth[instance["image_id"]])
        truth[instance["image_id"]].append(instance["box"])
    original_summary = read(baseline / "fusion_summary.json")
    assert original_summary["phase"] == "complete"
    assert original_summary["fusion"] == {
        "iou_threshold": .45, "score_mode": "mean", "box_mode": "score_weighted",
        "coverage_floor": 1., "view_weights": [.65, .65, .35, .35],
        "consensus_groups": [0, 0, 1, 1], "consensus_bonus": 1.10,
        "isolation_penalty": .90}
    assert original_summary["at_confidence_0_75"] == gate["fusion"]
    assert (original_summary["at_confidence_0_75"]["tp"], original_summary["at_confidence_0_75"]["fp"], original_summary["at_confidence_0_75"]["fn"]) == (3105, 183, 344)
    output.mkdir(parents=True)
    cutoff = np.float32(.35)
    layers, view_stats = [], {}
    for view in VIEWS:
        records = read(baseline / (view + "_predictions.json"))
        scope(records, ids, truth, metrics)
        filtered = []
        changed_ids = []
        before_total = after_total = high_total = 0
        before_digest, after_digest = hashlib.sha256(), hashlib.sha256()
        counts = []
        for record in records:
            pred = record["pred"]
            scores = np.asarray(pred["scores"], dtype=np.float32)
            keep = np.flatnonzero(scores >= cutoff).tolist()
            high = np.flatnonzero(scores >= np.float32(.75)).tolist()
            assert [i for i in keep if scores[i] >= np.float32(.75)] == high
            add_identity(before_digest, record["image_id"], high, record)
            add_identity(after_digest, record["image_id"], [i for i in keep if scores[i] >= np.float32(.75)], record)
            filtered.append({"image_id": record["image_id"], "gt": record["gt"],
                "pred": {"boxes": [pred["boxes"][i] for i in keep],
                         "scores": [pred["scores"][i] for i in keep]}})
            before_total += len(scores)
            after_total += len(keep)
            high_total += len(high)
            if len(scores) != len(keep):
                changed_ids.append(record["image_id"])
            counts.append({"image_id": record["image_id"], "before": len(scores),
                           "after": len(keep), "at_least_075": len(high)})
        assert before_digest.hexdigest() == after_digest.hexdigest()
        view_stats[view] = {"before_candidates": before_total, "after_candidates": after_total,
            "removed_candidates": before_total - after_total, "high_075_candidates": high_total,
            "high_075_identity_preserved": True, "high_075_identity_sha256": before_digest.hexdigest(),
            "changed_image_ids": changed_ids, "counts_per_image": counts}
        filename = view.replace("/", "_") + "_filtered.json"
        write(output / filename, filtered)
        view_stats[view]["filtered_cache"] = filename
        view_stats[view]["filtered_cache_sha256"] = sha(output / filename)
        layers.append(filtered)
        del records
        gc.collect()
        print(json.dumps({"phase": "filtered_view", "view": view, "before": before_total, "after": after_total}, ensure_ascii=False), flush=True)
    filter_finished = time.perf_counter()
    fused = []
    for index, image_id in enumerate(ids):
        entries = [layer[index] for layer in layers]
        boxes, scores = fuse_entries(entries, **original_summary["fusion"])
        fused.append({"image_id": image_id, "gt": entries[0]["gt"],
                      "pred": {"boxes": boxes.tolist(), "scores": scores.tolist()}})
    scope(fused, ids, truth, metrics)
    write(output / "filtered_fused_predictions.json", fused)
    fusion_finished = time.perf_counter()
    calibrated = metrics.calibrate_threshold(fused, .5, True)
    r90 = max((x for x in calibrated["curve"] if x["recall"] >= .90),
              key=lambda x: (x["precision"], x["recall"], x["f1"], x["confidence"]))
    fixed = metrics.evaluate_at_threshold(fused, .75, .5)
    write(output / "filtered_complete_curve.json", calibrated)
    write(output / "filtered_fusion_summary.json", {"images": 814, "GT": 3449, "empty": 13,
        "at_confidence_0_75": fixed, "at_recall90": r90,
        "balanced": calibrated["summary"], "fusion": original_summary["fusion"]})
    original = read(baseline / "fused_predictions.json")
    scope(original, ids, truth, metrics)
    raw_changed = [a["image_id"] for a, b in zip(original, fused) if a["pred"] != b["pred"]]
    comparisons = {
        "fixed_confidence_075": compare_images(original, fused, .75, .75, metrics),
        "own_R90_workpoint": compare_images(original, fused,
            original_summary["at_recall90"]["confidence"], r90["confidence"], metrics)}
    comparison_finished = time.perf_counter()
    report = {"schema": "cached_FP32_postNMS_cutoff035_counterfactual_v1", "phase": "complete",
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"images": 814, "GT": 3449, "empty": 13,
            "kind": "OLD FP32 PREDICTION CACHE REPLAY, NOT WEIGHT INFERENCE",
            "GT_boxes_match_original_provenance": True, "weights_read": False,
            "FP16_precision_evaluated": False, "speed_benchmark": False,
            "GPU_used": False, "original_unfiltered_fusion_rerun": False},
        "cutoff": {"requested": .35, "float32_actual": float(cutoff), "comparison": ">=",
                   "stage": "original post-NMS single-view cache, before cross-view fusion",
                   "original_kept_box_score_values_and_order_unchanged": True},
        "original_summary": {k: original_summary[k] for k in ["at_confidence_0_75", "at_recall90", "balanced", "fusion"]},
        "filtered_summary": {"at_confidence_0_75": fixed, "at_recall90": r90,
                             "balanced": calibrated["summary"], "full_curve_thresholds": calibrated["candidate_thresholds"]},
        "views": view_stats, "raw_fusion_changed_image_ids": raw_changed,
        "fused_candidates_before": sum(len(r["pred"]["scores"]) for r in original),
        "fused_candidates_after": sum(len(r["pred"]["scores"]) for r in fused),
        "comparisons": comparisons,
        "timing_seconds": {"total_cpu_diagnostic": comparison_finished - timer,
            "binding_and_filtering": filter_finished - timer,
            "filtered_fusion": fusion_finished - filter_finished,
            "calibration_and_comparison": comparison_finished - fusion_finished},
        "binding": {"baseline_directory": str(baseline), "baseline_gate_sha256": sha(gate_path),
            "original_artifact_sha256": input_hashes, "provenance_sha256": PROVENANCE_SHA,
            "metrics_sha256": METRICS_SHA, "fusion_source_sha256": FUSION_SHA,
            "fuser_loading": "Exact AST function definitions _rescale/fuse_entries from hash-bound frozen source; model imports not executed",
            "script_sha256": sha(Path(__file__))},
        "limitations": "Deletion in post-NMS FP32 caches isolates fusion-membership effects only. It does not measure actual pre-NMS AMP numerical effects, changed NMS outcomes, checkpoint inference or runtime speed. No model/GT/weight/source changes; no test-set result.",
        "torch_imported": "torch" in sys.modules}
    assert report["torch_imported"] is False
    write(output / "audit_summary.json", report)
    print(json.dumps({"output": str(output), "finished_utc": report["finished_utc"],
        "elapsed_seconds": report["timing_seconds"]["total_cpu_diagnostic"],
        "fixed_075": fixed, "R90": r90,
        "changed_metric_images_fixed": len(comparisons["fixed_confidence_075"]["TP_FP_FN_changed_image_ids"])}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "runs/strong_baseline_20261030")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/fast_deploy_20261031/cached_cutoff035_audit")
    parser.add_argument("--provenance", type=Path,
        default=ROOT.parent / "Apple_DualBranch_3080_20261022/assets/papple_val_provenance.json")
    args = parser.parse_args()
    audit(args.baseline_dir.resolve(), args.output_dir.resolve(), args.provenance.resolve())


if __name__ == "__main__":
    main()
