"""Read-only CPU replay of the 32-image candidate audit, never GPU inference.

Only the requested fresh output directory is written. Model/GT/inference code
and previous evidence are never edited. Missing 814-frame initial logits are
reported as missing; final-score caches cannot establish early-score safety.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "strong_runtime/src"),
               str(REPO / "strong_runtime/tmp")]

import numpy as np
import torch
import flip_tta
from apple_seg.evaluate import load_papple_contract
from deployment.candidate_policy import select_candidates, policy_description


INITIAL_THRESHOLDS = (None, .03, .05, .10, .12, .15)
FINAL_THRESHOLDS = (.738, .35)
GROUPS = {"fixed_nms080_k64": (.8, 64), "user_combination_nms075_k48": (.75, 48)}
PERCENTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
METRICS = flip_tta.metrics
INPUTS = {}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(path):
    path = Path(path).resolve()
    value = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
    previous = INPUTS.setdefault(str(path), value)
    if value != previous:
        raise RuntimeError("Input changed since earlier read: " + str(path))
    return value


def read_json(path):
    bind(path)
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    bind(path)
    return value


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(len(values)),
            "quantiles": {f"p{q:02d}": float(np.percentile(values, q)) if len(values) else None
                          for q in PERCENTILES},
            "mean": float(np.mean(values)) if len(values) else None,
            "below_initial_threshold": {str(t): int((values < np.float32(t)).sum())
                                        for t in INITIAL_THRESHOLDS if t is not None}}


def roi_summary(values):
    values = np.asarray(values, dtype=np.int64)
    return {"images": len(values), "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "min": int(values.min()), "max": int(values.max()), "total": int(values.sum()),
            "quantile_method": "numpy linear percentile, so count quantiles may be fractional"}


def point_matches(prediction, outputs, threshold):
    match = METRICS.match_image(prediction["gt"]["boxes"], prediction["pred"]["boxes"],
                                prediction["pred"]["scores"], .5, threshold)
    return [(int(p), int(g), int(outputs[p])) for p, g in
            zip(match["pred_indices"], match["gt_indices"], strict=True) if g >= 0]


def gt_identity(row):
    return {key: row[key] for key in
            ("image_id", "gt_index", "apple_id", "occlusion_stratum", "visible_area_fraction")}


def score_label(value):
    return "none" if value is None else f"{value:.2f}"


def check_full_val_cache_schema():
    """Inspect available relevant caches; do not infer unrecorded initial scores."""
    report = []
    roots = ("strong_baseline_20261030", "fast_fp16_all_single_1024_20261031",
             "fast_fp16_all_paired_1024_20261031", "premask_nms080_k64_single_20261102",
             "premask_nms080_k64_paired_20261102")
    for run in roots:
        for name in ("base", "small"):
            directory = REPO / "runs" / run / name
            if not directory.exists():
                continue
            for view in ("original", "hflip"):
                path = directory / (view + "_predictions.json")
                if not path.exists():
                    continue
                records = read_json(path)
                keys = sorted({key for row in records for key in row["pred"]})
                archives = []
                mask_dir = directory / "roi_masks" / view
                files = sorted(mask_dir.glob("*.npz")) if mask_dir.exists() else []
                for archive in (files[:1] + files[-1:] if len(files) > 1 else files):
                    bind(archive)
                    with np.load(archive, allow_pickle=False) as data:
                        archives.append({"path": str(archive), "keys": sorted(data.files)})
                report.append({"run": run, "model": name, "view": view, "images": len(records),
                               "prediction_keys": keys, "mask_archive_count": len(files),
                               "sampled_mask_archive_schemas": archives,
                               "initial_scores_in_prediction_records": "initial_scores" in keys})
                del records
    candidate_archives = Counter()
    for location in (REPO / "runs", REPO / "reports"):
        for path in location.rglob("candidates_*.npz"):
            candidate_archives[str(path.parent.relative_to(REPO))] += 1
    return {"inspected_complete_val_schemas": report,
            "candidate_npz_inventory": dict(candidate_archives),
            "conclusion": "Relevant full-VAL predictions store final boxes/scores, and sampled ROI archives store boxes/scores/V/A. No full-814 initial-score dataset was found. Only 32-image Base+Small audit archives support this replay.",
            "cannot_establish_814_image_initial_threshold_safety": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "reports/premask_stage2_20261102/cpu_audit")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != "F:":
        raise ValueError("Output must remain on F:")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    for path in (Path(__file__), REPO / "deployment/candidate_policy.py",
                 REPO / "strong_runtime/reference/local_amodal/scripts/metrics.py"):
        bind(path)
    report_root = REPO / "reports/premask_20261102/audit_v1"
    audit_status = read_json(report_root / "status.json")
    assert audit_status["phase"] == "complete" and audit_status["mode"] == "audit"
    samples = audit_status["samples"]
    assert len(samples) == 32 and len({s["image_id"] for s in samples}) == 32
    manifest = read_json(audit_status["sample_manifest"])
    assert manifest["samples"] == samples and sha(audit_status["sample_manifest"]) == audit_status["sample_sha256"]
    config = read_json(REPO / "configs/strong_baseline_4060.json")
    provenance = (REPO / config["paths"]["provenance"]).resolve()
    bind(provenance)
    contract = load_papple_contract(provenance)
    report = {"schema": "premask_stage2_cpu_initial_score_audit_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "phase": "running", "GPU_used": False, "model_constructed": False, "training_performed": False,
              "scope": "Existing stratified 32-image original-view audit only; not full 814 VAL or an independent test.",
              "source_provenance_sha256": contract["provenance_sha256"],
              "GT_used_in_selection": False, "GT_used_for_offline_matching_only": True,
              "sample_manifest": audit_status["sample_manifest"], "images": len(samples),
              "quantile_method": "numpy linear percentile", "models": {},
              "interpretation": {
                  "all_candidates": "Pre-final-NMS mask-dependent scores; many proposals may describe the same GT. Counts are not apple counts.",
                  "true_positive_distribution": "Original final model outputs, score-filtered at requested value then independently one-to-one IoU>=.5 matched; mapped to their decoded candidate indices.",
                  "deleted_original_TP_candidate": "A previously matched output proposal is excluded by replayed early selection. Another retained proposal may recover that GT, so this is not measured final Recall loss.",
                  "no_frozen_qualified_alternative": "Among saved candidate boxes/scores, no retained proposal has IoU>=.5 to the lost GT and saved final score>=the point. Mask NMS, chunk numerical changes and new forward are not simulated.",
                  "full_val_P90": "Threshold copied from original FP16 single-view full-VAL metadata; no new threshold search on this subset.",
                  "strategy_groups": "Compare initial thresholds within fixed NMS.8/K64. NMS.75/K48 is a separate two-parameter user combination, not an attribution to score threshold alone."}}

    for model in ("base", "small"):
        directory = report_root / model / "none"
        predictions = read_json(directory / "predictions.json")
        original_audit = read_json(directory / "candidate_audit.json")
        subset_summary = read_json(directory / "summary.json")
        full_meta_path = REPO / "runs/fast_fp16_all_single_1024_20261031" / model / "prediction_metadata.json"
        full_meta = read_json(full_meta_path)
        assert full_meta["phase"] == "complete" and full_meta["images"] == 814 and full_meta["GT"] == 3449
        points = {"fixed_075": .75, "original_full_VAL_recorded_P90": full_meta["view_metrics"]["original"]["at_recall90"]["confidence"]}
        entries = []
        cohorts = {str(t): {"all_mask_candidates": [], "post_nms_TP": [], "heavy_post_nms_TP": []}
                   for t in FINAL_THRESHOLDS}
        baseline_tp = {p: 0 for p in points}
        baseline_heavy = {p: 0 for p in points}
        for sample, prediction in zip(samples, predictions, strict=True):
            idx, image_id = sample["image_index"], sample["image_id"]
            assert prediction["image_id"] == image_id == contract["images"][idx]["image_id"]
            rows = contract["truth"][image_id]
            assert prediction["gt"]["boxes"] == [row["box"] for row in rows]
            path = directory / f"candidates_{idx:04d}.npz"
            bind(path)
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: np.array(data[key], copy=True) for key in data.files}
            boxes, initial, final = (arrays[key] for key in ("decoded_boxes", "initial_scores", "final_scores"))
            outputs = arrays["output_indices"]
            assert len(initial) == len(final) == len(boxes) == 1000
            assert np.array_equal(arrays["mask_indices"], np.arange(len(initial)))
            assert np.isfinite(boxes).all() and np.isfinite(initial).all() and np.isfinite(final).all()
            native_boxes = boxes * np.float32(1300 / 1024)
            assert np.array_equal(native_boxes[outputs], np.asarray(prediction["pred"]["boxes"], dtype=np.float32).reshape(-1, 4))
            assert np.array_equal(final[outputs], np.asarray(prediction["pred"]["scores"], dtype=np.float32))
            overlaps = METRICS.box_iou(native_boxes, prediction["gt"]["boxes"])
            matches = {p: point_matches(prediction, outputs, threshold) for p, threshold in points.items()}
            for point, matched in matches.items():
                baseline_tp[point] += len(matched)
                baseline_heavy[point] += sum(rows[g]["occlusion_stratum"] == "heavy_visible_lt_0.40" for _, g, _ in matched)
            for threshold in FINAL_THRESHOLDS:
                cohort = cohorts[str(threshold)]
                cohort["all_mask_candidates"].extend(initial[final >= np.float32(threshold)].tolist())
                for _, g, j in point_matches(prediction, outputs, threshold):
                    cohort["post_nms_TP"].append(float(initial[j]))
                    if rows[g]["occlusion_stratum"] == "heavy_visible_lt_0.40":
                        cohort["heavy_post_nms_TP"].append(float(initial[j]))
            entries.append(dict(image_id=image_id, image_index=idx, rows=rows, boxes=boxes, initial=initial,
                                final=final, outputs=outputs, overlaps=overlaps, matches=matches))
        result = {"checkpoint_sha256": full_meta["checkpoint_sha256"], "images": len(entries),
                  "GT": sum(len(e["rows"]) for e in entries), "empty_images": sum(not e["rows"] for e in entries),
                  "GT_strata": dict(Counter(row["occlusion_stratum"] for e in entries for row in e["rows"])),
                  "full_val_P90_metadata": str(full_meta_path), "evaluation_points": points,
                  "subset_diagnostic_P90_not_used_for_selection": subset_summary["summary"]["at_recall90"],
                  "baseline_TP": baseline_tp, "baseline_heavy_TP": baseline_heavy,
                  "initial_score_distributions": {t: {k: describe(v) for k, v in group.items()} for t, group in cohorts.items()},
                  "policy_groups": {}}

        for group, (nms, k) in GROUPS.items():
            replay = {}
            none_lost = {point: set() for point in points}
            for threshold in INITIAL_THRESHOLDS:
                spec = dict(score_threshold=threshold, box_nms_iou=nms, topk=k)
                counts = []
                per_image = []
                dropped = {point: [] for point in points}
                removed_high = {str(t): 0 for t in FINAL_THRESHOLDS}
                for entry in entries:
                    retained = select_candidates(torch.from_numpy(entry["boxes"]), torch.from_numpy(entry["initial"]), **spec).numpy()
                    mask = np.zeros(len(entry["initial"]), dtype=bool)
                    mask[retained] = True
                    counts.append(len(retained))
                    per_image.append({"image_id": entry["image_id"], "mask_ROIs": len(retained)})
                    for t in FINAL_THRESHOLDS:
                        removed_high[str(t)] += int(((entry["final"] >= np.float32(t)) & ~mask).sum())
                    for point, final_threshold in points.items():
                        for pred_index, gi, proposal in entry["matches"][point]:
                            if mask[proposal]:
                                continue
                            row = entry["rows"][gi]
                            qualified = retained[(entry["overlaps"][retained, gi] >= .5)
                                                 & (entry["final"][retained] >= np.float32(final_threshold))]
                            compatible_any_score = retained[entry["overlaps"][retained, gi] >= .5]
                            identity = (entry["image_id"], gi)
                            if threshold is None:
                                none_lost[point].add(identity)
                            dropped[point].append({**gt_identity(row), "original_prediction_index": pred_index,
                                "decoded_candidate_index": proposal, "initial_score": float(entry["initial"][proposal]),
                                "saved_final_score": float(entry["final"][proposal]),
                                "incremental_loss_vs_same_nms_k_no_initial_cutoff": identity not in none_lost[point],
                                "retained_GT_compatible_candidates_any_final_score": len(compatible_any_score),
                                "retained_GT_compatible_candidates_passing_saved_final_threshold": len(qualified),
                                "no_frozen_qualified_alternative": len(qualified) == 0})
                comparisons = {}
                for point, lost in dropped.items():
                    heavy = lambda row: row["occlusion_stratum"] == "heavy_visible_lt_0.40"
                    comparisons[point] = {
                        "baseline_TP": baseline_tp[point], "baseline_heavy_TP": baseline_heavy[point],
                        "old_TP_proposal_deleted": len(lost), "heavy_old_TP_proposal_deleted": sum(heavy(r) for r in lost),
                        "incremental_old_TP_proposal_deleted_vs_no_initial_cutoff": sum(r["incremental_loss_vs_same_nms_k_no_initial_cutoff"] for r in lost),
                        "heavy_incremental_old_TP_proposal_deleted_vs_no_initial_cutoff": sum(heavy(r) and r["incremental_loss_vs_same_nms_k_no_initial_cutoff"] for r in lost),
                        "deleted_TP_without_frozen_qualified_alternative": sum(r["no_frozen_qualified_alternative"] for r in lost),
                        "heavy_deleted_TP_without_frozen_qualified_alternative": sum(heavy(r) and r["no_frozen_qualified_alternative"] for r in lost),
                        "deleted_instances": lost,
                    }
                replay[score_label(threshold)] = {"selection": policy_description(**spec), "ROI_counts": roi_summary(counts),
                                                 "per_image_ROI_counts": per_image, "removed_final_high_candidates": removed_high,
                                                 "points": comparisons}
                if group == "fixed_nms080_k64" and threshold is None:
                    previous = original_audit["policy_totals"]["nms080_k64"]
                    assert previous["lost_original_TP_proposals"] == comparisons["fixed_075"]["old_TP_proposal_deleted"]
                    assert previous["mean_mask_candidates"] == roi_summary(counts)["mean"]
            result["policy_groups"][group] = replay
        report["models"][model] = result
        print("CPU_STAGE2_MODEL " + json.dumps({"model": model, "GT": result["GT"], "points": points,
            "distributions": result["initial_score_distributions"],
            "policies": {g: {t: {"ROI": v["ROI_counts"], "deleted075": v["points"]["fixed_075"]["old_TP_proposal_deleted"],
                "heavy_deleted075": v["points"]["fixed_075"]["heavy_old_TP_proposal_deleted"],
                "incremental_deleted075": v["points"]["fixed_075"]["incremental_old_TP_proposal_deleted_vs_no_initial_cutoff"],
                "deleted_P90": v["points"]["original_full_VAL_recorded_P90"]["old_TP_proposal_deleted"],
                "heavy_deleted_P90": v["points"]["original_full_VAL_recorded_P90"]["heavy_old_TP_proposal_deleted"]}
                for t,v in rows.items()} for g,rows in result["policy_groups"].items()}}, ensure_ascii=False), flush=True)

    report["full_VAL_initial_score_availability"] = check_full_val_cache_schema()
    for receipt in list(INPUTS.values()):
        assert sha(receipt["path"]) == receipt["sha256"] and Path(receipt["path"]).stat().st_size == receipt["bytes"]
    assert not torch.cuda.is_initialized()
    report.update(phase="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                  inputs_before_after_identical=True, CUDA_initialized=False,
                  input_receipts=list(INPUTS.values()))
    path = output / "summary.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("CPU_STAGE2_COMPLETE " + str(path) + " SHA256=" + sha(path), flush=True)


if __name__ == "__main__":
    main()
