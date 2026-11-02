"""CPU-only independent native V/A mask PR for the 41-image Stage 2 screen.

Requires a completed screen with inference-bound ROI archive SHA receipts. This
is a regression diagnostic, not full-VAL acceptance, mask AP, fused-mask quality
or D405 tracking evidence. The original GT and confidence thresholds are frozen.
No model is constructed and CUDA is hidden before importing the tensor runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "strong_runtime/src")]
from summarize_premask_validation import (
    ORIGINAL_CHECKPOINTS, METRICS_PATH, atomic_json, audit_records, bind, load_metrics,
    load_roi_archive, now, read_json, require, sha,
)

P90_CONFIDENCE = {"base": .7394223809242249, "small": .743447482585907}
EXPECTED_SCOPE = (41, 227, 4)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--screen-dir", type=Path, required=True)
    result.add_argument("--cases", nargs="+", default=["F1", "T15", "G1", "L1", "H1"])
    result.add_argument("--models", nargs="+", choices=("base", "small"), default=["base", "small"])
    result.add_argument("--config", type=Path, default=REPO / "configs/strong_baseline_4060.json")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--cpu-threads", type=int, default=2)
    return result


def normalize_cases(cases):
    require(bool(cases) and len(set(cases)) == len(cases), "Cases must be nonempty and unique")
    require(all(isinstance(name, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,30}", name) for name in cases),
            "Invalid case directory name")
    # Baseline is always inferred by this same screen; never silently borrow a cache.
    return ["F1", *[name for name in cases if name != "F1"]]


def subset_contract(contract, samples):
    require(isinstance(samples, list) and len(samples) == EXPECTED_SCOPE[0], "Exactly 41 diagnostic samples required")
    images, truth = [], {}
    for sample in samples:
        index, name = sample.get("image_index"), sample.get("image_id")
        require(type(index) is int and 0 <= index < len(contract["images"]), "Manifest index outside original contract")
        frozen = contract["images"][index]
        require(name == frozen["image_id"] and name not in truth, "Manifest image identity/order or uniqueness differs")
        images.append(frozen)
        truth[name] = contract["truth"][name]
    observed = (len(images), sum(len(rows) for rows in truth.values()), sum(not rows for rows in truth.values()))
    require(observed == EXPECTED_SCOPE, f"Diagnostic scope differs: {observed}, expected {EXPECTED_SCOPE}")
    return {**contract, "images": images, "truth": truth, "official_full_val": False,
            "scope": {"images": observed[0], "GT_instances": observed[1], "empty_GT_images": observed[2]},
            "strata": dict(Counter(row["occlusion_stratum"] for rows in truth.values() for row in rows))}


def verify_archive_manifest(directory, entries, records, samples, evidence):
    """Check inference-captured SHA and strict per-image archive identity."""
    directory = Path(directory).resolve()
    require(len(entries) == len(records) == len(samples), "ROI manifest count differs")
    paths = []
    for index, (entry, record, sample) in enumerate(zip(entries, records, samples, strict=True)):
        expected = directory / "roi_masks" / f"{index:06d}.npz"
        actual = Path(entry.get("path", "")).resolve()
        require(actual == expected.resolve() and actual.parent == (directory / "roi_masks").resolve()
                and not actual.is_symlink(), "ROI archive path leaves the declared case or changes output order")
        require(entry.get("image_id") == record["image_id"] == sample["image_id"]
                and entry.get("manifest_index") == sample["image_index"], "ROI archive image/GT-index identity differs")
        receipt = bind(actual, evidence)
        require(receipt["sha256"] == entry.get("sha256") and receipt["bytes"] == entry.get("bytes"),
                "ROI archive differs from inference SHA/size receipt")
        require(entry.get("dtype") == "float32" and entry.get("shape") == [len(record["pred"]["scores"]), 56, 56],
                "ROI manifest must declare original float32 [N,56,56] masks")
        paths.append(actual)
    return paths


def point_assignments(diagnostics, branch, confidence):
    result = {}
    for row in diagnostics:
        name = row["image_id"]
        require(name not in result, "Duplicate mask diagnostic image")
        assignments = row[branch + "_mask_gt_indices_by_prediction"]
        require(len(assignments) == len(row["scores"]), "Mask score/assignment count differs")
        selected = [int(gt) for gt, score in zip(assignments, row["scores"], strict=True) if gt >= 0 and score >= confidence]
        require(len(selected) == len(set(selected)), "Mask matcher recalled the same GT more than once")
        result[name] = set(selected)
    return result


def compare_mask_recall(reference, candidate, contract, branch, confidence):
    old = point_assignments(reference, branch, confidence)
    new = point_assignments(candidate, branch, confidence)
    image_ids = [row["image_id"] for row in contract["images"]]
    require(list(old) == image_ids and list(new) == image_ids, "Mask diagnostic image order differs from GT contract")
    groups = {name: {"GT": count, "reference_TP": 0, "candidate_TP": 0,
                     "lost": 0, "recovered": 0, "both_recalled": 0, "both_missed": 0}
              for name, count in contract["strata"].items()}
    instances = []
    for image_id in image_ids:
        for gt in contract["truth"][image_id]:
            before, after = gt["gt_index"] in old[image_id], gt["gt_index"] in new[image_id]
            outcome = "both_recalled" if before and after else "lost" if before else "recovered" if after else "both_missed"
            group = groups[gt["occlusion_stratum"]]
            group["reference_TP"] += int(before)
            group["candidate_TP"] += int(after)
            group[outcome] += 1
            instances.append({"image_id": image_id, "gt_index": gt["gt_index"], "apple_id": gt["apple_id"],
                              "occlusion_stratum": gt["occlusion_stratum"], "visible_area_fraction": gt["visible_area_fraction"],
                              "reference_recalled": before, "candidate_recalled": after, "outcome": outcome})
    for group in groups.values():
        group["reference_recall"] = group["reference_TP"] / group["GT"] if group["GT"] else None
        group["candidate_recall"] = group["candidate_TP"] / group["GT"] if group["GT"] else None
        require(group["candidate_TP"] - group["reference_TP"] == group["recovered"] - group["lost"],
                "Mask recall difference identity failed")
    return {"branch": branch, "confidence": confidence, "mask_iou_threshold": .5,
            "lost": sum(g["lost"] for g in groups.values()), "recovered": sum(g["recovered"] for g in groups.values()),
            "groups": groups, "ground_truth_instances": instances,
            "interpretation": "Each branch independently matched by native mask IoU; box matches do not determine mask TP"}


def complete_stage(screen, model, case, contract, samples, metrics, evidence):
    directory = screen / model / case
    summary = read_json(directory / "summary.json", evidence)
    require(summary.get("phase") == "complete" and summary.get("model") == model and summary.get("case") == case,
            f"Incomplete/wrong case: {model}/{case}")
    require(summary.get("numeric_parameters_unchanged") is True, "Inference parameter numerical integrity receipt missing")
    records = read_json(directory / "predictions.json", evidence)
    require(sha(directory / "predictions.json") == summary.get("predictions_sha256"), "Prediction SHA differs from inference receipt")
    audit_records(records, contract, metrics)
    archive_file = directory / "roi_archive_manifest.json"
    entries = read_json(archive_file, evidence)
    require(sha(archive_file) == summary.get("roi_archive_manifest_sha256"), "ROI manifest SHA differs from inference receipt")
    paths = verify_archive_manifest(directory, entries, records, samples, evidence)
    points = {"fixed_075": .75, "original_full_VAL_P90_confidence": P90_CONFIDENCE[model]}
    box_points = {}
    for label, confidence in points.items():
        observed = metrics.evaluate_at_threshold(records, confidence, .5)
        require(observed == summary.get("points", {}).get(label), "Saved box metric/operating point differs from fresh raw recomputation")
        box_points[label] = observed
    return {"model": model, "case": case, "directory": str(directory), "summary": summary,
            "records": records, "archive_paths": paths, "diagnostics": [], "points": points,
            "box_points": box_points, "roi_archive_receipts": entries}


def main(argv=None):
    args = parser().parse_args(argv)
    cases = normalize_cases(args.cases)
    require(args.cpu_threads >= 1 and len(set(args.models)) == len(args.models), "Invalid CPU count/model inventory")
    screen, output, config_path = args.screen_dir.resolve(), args.output.resolve(), args.config.resolve()
    require(all(path.drive.upper() == "F:" for path in (screen, output, config_path)), "All project files must remain on F:")
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    evidence = {}
    started = time.perf_counter()
    status = {"schema": "stage2_mask_cpu_audit_v1", "phase": "initializing", "created_utc": now(),
              "screen_directory": str(screen), "models": args.models, "cases": cases,
              "GPU_used": False, "model_constructed": False, "training_performed": False,
              "scope": "41-image regression diagnostic, not 814-image acceptance or independent test", "completed": {}}

    def progress(phase, **fields):
        status.update(phase=phase, updated_utc=now(), elapsed_seconds=time.perf_counter()-started, **fields)
        atomic_json(output / "status.json", status)
        print(json.dumps({"phase": phase, **fields}, ensure_ascii=False), flush=True)

    progress("initializing")
    try:
        import numpy as np
        import torch
        from apple_seg import evaluate as base
        from apple_seg.data import PAppleDataset
        from apple_seg.od_evaluate import _roi_mask_overlaps
        torch.set_num_threads(args.cpu_threads)
        require(not torch.cuda.is_initialized(), "CPU audit must not initialize CUDA")
        metrics = load_metrics()
        pipeline = read_json(screen / "pipeline_status.json", evidence)
        require(pipeline.get("phase") == "complete" and pipeline.get("from_fresh_weight_inference") is True
                and pipeline.get("training_performed") is False and pipeline.get("GT_used_for_inference") is False,
                "Only completed fresh-weight screen inference is eligible; wait for GPU screen completion")
        require(pipeline.get("source_before_after_identical") is True and pipeline.get("inputs_before_after_identical") is True,
                "Screen source/input final integrity receipts missing")
        cfg = read_json(config_path, evidence)
        config_receipts = [row for row in pipeline.get("input_receipts", [])
                           if Path(row["path"]).resolve() == config_path]
        require(len(config_receipts) == 1 and config_receipts[0]["sha256"] == sha(config_path),
                "Config differs from the screen's input SHA receipt")
        root = (config_path.parent / cfg["path_base"]).resolve()
        paths = {key: (root / value).resolve() for key, value in cfg["paths"].items()}
        require(all(path.drive.upper() == "F:" for path in paths.values()), "Configured paths must remain on F:")
        contract = base.load_papple_contract(paths["provenance"])
        bind(paths["provenance"], evidence)
        require(pipeline.get("provenance_sha256") == contract["provenance_sha256"], "Inference GT provenance differs")
        samples = pipeline.get("samples")
        subset = subset_contract(contract, samples)
        # Bind the predeclared 41-image manifest through the inference input receipt.
        manifests = [r for r in pipeline.get("input_receipts", [])
                     if Path(r["path"]).name == "premask_stage2_review_samples_20261102.json"]
        require(len(manifests) == 1, "Inference is not bound to the predeclared 41-image manifest")
        manifest = read_json(manifests[0]["path"], evidence)
        require(bind(manifests[0]["path"], evidence)["sha256"] == manifests[0]["sha256"]
                and manifest.get("samples") == samples, "Manifest hash/sample inventory differs")
        source_files = [Path(__file__), REPO / "scripts/summarize_premask_validation.py", METRICS_PATH,
                        *(REPO / "strong_runtime/src/apple_seg" / name for name in ("evaluate.py", "data.py", "od_evaluate.py", "od_roi.py"))]
        for path in source_files:
            bind(path, evidence)
        for relative, digest in pipeline.get("source_sha256", {}).items():
            require(bind(REPO / relative, evidence)["sha256"] == digest, "Recorded inference source changed: " + relative)
        native = PAppleDataset(paths["papple_root"], "val", 32, augment=False)
        require(native.annotation_sha256 == contract["provenance"]["binding"]["current_annotation_sha256"],
                "Actual original V/A annotation SHA differs")
        for path in native.annotation_paths.values():
            bind(path, evidence)
        results = {}
        for model in args.models:
            model_meta = pipeline.get("models", {}).get(model, {})
            restore = model_meta.get("strict_restore", {})
            require(model_meta.get("checkpoint_sha256") == cfg[model + "_checkpoint_sha256"] == ORIGINAL_CHECKPOINTS[model]
                    and model_meta.get("original_weights_before_after_identical") is True
                    and restore.get("strict") is True and restore.get("tensor_count") == 434
                    and restore.get("all_tensors_equal_to_checkpoint") is True,
                    "Original strict-434 weight identity/integrity differs")
            # Infer-time receipt is pinned and on-disk bytes are independently checked.
            require(bind(paths[model + "_checkpoint"], evidence)["sha256"] == ORIGINAL_CHECKPOINTS[model],
                    "Original checkpoint file SHA differs")
            stages = {case: complete_stage(screen, model, case, subset, samples, metrics, evidence) for case in cases}
            minimum = min(.75, P90_CONFIDENCE[model])
            with torch.inference_mode():
                for index, sample in enumerate(samples):
                    full_index, image_id = sample["image_index"], sample["image_id"]
                    frozen = contract["images"][full_index]
                    target = native[full_index]["target"]
                    truth = base._audit_target(target, frozen, contract["truth"][image_id])
                    rgb_path = Path(target["source"]["rgb_path"])
                    bind(rgb_path, evidence)
                    base._audit_rgb(rgb_path, frozen)
                    for case, stage in stages.items():
                        record = stage["records"][index]
                        arrays = load_roi_archive(stage["archive_paths"][index], record, evidence)
                        require(all(arrays[k].dtype == np.float32 for k in ("boxes", "scores", "visible", "amodal")),
                                "Stage 2 ROI arrays must preserve float32 values")
                        require(list(arrays["visible"].shape) == list(arrays["amodal"].shape)
                                == stage["roi_archive_receipts"][index]["shape"],
                                "Actual ROI mask shape differs from inference archive manifest")
                        selected = np.flatnonzero(np.asarray(record["pred"]["scores"], dtype=np.float64) >= minimum)
                        scores = [record["pred"]["scores"][int(p)] for p in selected]
                        prediction = {"boxes": torch.from_numpy(arrays["boxes"][selected]), "scores": scores,
                                      "visible": torch.from_numpy(arrays["visible"][selected]),
                                      "amodal": torch.from_numpy(arrays["amodal"][selected])}
                        overlaps = (_roi_mask_overlaps(prediction, truth, (frozen["height"], frozen["width"]),
                                                       gt_chunk_size=8, tile_pixels=65536) if len(selected) else
                                    {b + "_mask": np.empty((0, len(truth["boxes"]))) for b in ("visible", "amodal")})
                        diagnostic = {"image_id": image_id, "gt_count": len(truth["boxes"]), "scores": scores,
                                      "original_prediction_indices": selected.tolist(),
                                      "box_gt_indices_by_prediction": base._greedy_assignment(
                                          metrics.box_iou(prediction["boxes"].numpy(), truth["boxes"]), scores)}
                        for branch in ("visible", "amodal"):
                            matrix = overlaps[branch + "_mask"]
                            diagnostic[branch + "_mask_gt_indices_by_prediction"] = base._greedy_assignment(matrix, scores)
                            diagnostic[branch + "_mask_iou_matrix"] = matrix.tolist()
                        stage["diagnostics"].append(diagnostic)
                        del arrays, prediction, overlaps
                    del target, truth
                    progress("mask_metrics", model=model, images=index+1, total_images=len(samples), cases=cases)
            model_results = {}
            for case, stage in stages.items():
                stage_results = {}
                for label, confidence in stage["points"].items():
                    point = {"confidence": confidence, "box": stage["box_points"][label]}
                    for branch in ("visible", "amodal"):
                        measured = base._mask_pr(stage["diagnostics"], branch + "_mask", confidence, 227, metrics)
                        measured["threshold_policy"] = "Fixed .75 or frozen original FP16 full-VAL BOX P90 confidence; independently matched mask IoU>=.5"
                        point[branch] = measured
                    point["occlusion"] = base._occlusion(stage["diagnostics"], subset, confidence)
                    if case != "F1":
                        point["vs_same_run_F1"] = {branch: compare_mask_recall(stages["F1"]["diagnostics"], stage["diagnostics"],
                            subset, branch, confidence) for branch in ("visible", "amodal")}
                    else:
                        point["vs_same_run_F1"] = None
                    stage_results[label] = point
                result = {"phase": "complete", "model": model, "case": case, "points": stage_results,
                          "mask_metrics_claimed": True, "images": 41, "GT": 227, "empty": 4,
                          "minimum_processed_confidence": minimum, "source_directory": stage["directory"],
                          "reference": "same-run F1" if case != "F1" else "baseline; no reference comparison",
                          "original_inference_spec": stage["summary"].get("spec"), "automatically_promoted": False,
                          "archive_binding": "Every ROI archive SHA/size verified against inference-written manifest; native boxes/scores exactly match inference-bound JSON"}
                destination = output / model / case
                atomic_json(destination / "diagnostics.json", stage["diagnostics"])
                atomic_json(destination / "summary.json", result)
                model_results[case] = result
                status["completed"][model + "_" + case] = {"summary": str(destination / "summary.json"), "sha256": sha(destination / "summary.json")}
                progress("case_complete", model=model, case=case)
            results[model] = model_results
        require(not torch.cuda.is_initialized(), "CUDA unexpectedly initialized during CPU audit")
        for receipt in evidence.values():
            require(Path(receipt["path"]).stat().st_size == receipt["bytes"] and sha(receipt["path"]) == receipt["sha256"],
                    "Input/source changed during CPU audit: " + receipt["path"])
        summary = {"schema": "stage2_independent_mask_audit_v1", "phase": "complete", "completed_utc": now(),
                   "screen_directory": str(screen), "scope": status["scope"], "images": 41, "GT": 227, "empty": 4,
                   "GT_strata": subset["strata"], "models": results, "GPU_used": False, "CUDA_initialized": False,
                   "model_constructed": False, "training_performed": False, "automatically_promoted": False,
                   "source_and_inputs_before_after_identical": True, "input_receipts": list(evidence.values()),
                   "elapsed_seconds": time.perf_counter()-started,
                   "limitations": ["41 regression images are not unbiased 814-image validation or independent test",
                       "Independent visible/amodal mask IoU>=.5 P/R at frozen box confidence; no mask AP",
                       "No pixel-level four-view fusion or D405 local tracking is evaluated",
                       "Mask PR alone does not measure subpixel contour or 3D fruit-centre error",
                       "No candidate is promoted automatically; any heavy loss remains explicitly listed"]}
        atomic_json(output / "summary.json", summary)
        progress("complete", summary=str(output / "summary.json"), summary_sha256=sha(output / "summary.json"))
        return summary
    except BaseException as error:
        progress("failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
