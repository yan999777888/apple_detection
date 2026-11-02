"""Explicit full VAL of ONE selected Stage2 configuration, never a search.

Run only after reviewing the diagnostic screen and choosing a configuration.
Original Base/Small paired inference is sequential; at most one child exists.
The complete uncut four-view point is checked before optional CPU-only .35
prefix derivation. A P@R90 below .94 stops this candidate, not the protected
.9443430656934306 baseline. No model is trained or automatically promoted.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import evaluate_stage2 as evaluator
import run_fast_validation_pipeline as previous
import summarize_premask_validation as audit

fast = evaluator.fast
CONFIG_PATH = REPO / "configs/strong_baseline_4060.json"
CONFIG_SHA = "2c90ef5e075aaa53ae033013e49668a916d83e08b99db7964dc10677a61f8e7a"
GLOBAL_LOCK = REPO / "reports/premask_stage2_20261102/full_validation_active.lock"
REFERENCES = {
    "protected_original_FP32": REPO / "runs/strong_baseline_20261030",
    "previous_uncut_FP16": REPO / "runs/fast_fp16_all_paired_1024_20261031",
    "previous_nms080_k64_FP16": REPO / "runs/premask_nms080_k64_paired_20261102",
}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage2-config", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True, help="New runs directory on F:")
    result.add_argument("--report", type=Path, required=True, help="New reports directory on F:")
    return result


def stage_arguments(stage, selected, output):
    if stage not in ("base", "small", "fuse"):
        raise ValueError("Only original Base, Small and frozen fusion are supported")
    return [sys.executable, "-B", str(REPO / "scripts/evaluate_stage2.py"),
            "--config", str(CONFIG_PATH), "--stage2-config", str(selected), "--stage", stage,
            "--output", str(output), "--precision", "fp16", "--pre-nms-threshold", "none",
            "--roi-chunk-size", "64", "--roi-precision", "autocast", "--no-skip-dead-amodal",
            "--short-side", "1024", "--max-size", "1280", "--views", "both",
            "--paired-backbone-batch2", "--save-roi-masks", "--progress-every", "25"]


def source_inventory():
    sources = {**previous.source_inventory(), **fast.source_inventory()}
    names = (*evaluator.EXTRA_SOURCES, "scripts/run_stage2_full_validation.py",
             "scripts/run_fast_validation_pipeline.py", "scripts/summarize_premask_validation.py",
             "scripts/runtime_f_drive.ps1")
    sources.update({str(Path(name)): fast.sha(REPO / name) for name in names})
    return sources


def evaluator_sources():
    sources = fast.source_inventory()
    sources.update({str(Path(name)): fast.sha(REPO / name) for name in evaluator.EXTRA_SOURCES})
    return sources


def require(value, message):
    if not value:
        raise ValueError(message)


def bound_json(path, evidence, expected=None):
    value = audit.read_json(path, evidence)
    if expected is not None:
        require(evidence[str(Path(path).resolve())]["sha256"] == expected, "Evidence SHA differs: " + str(path))
    return value


def verify_pins(sources, evidence):
    require(source_inventory() == sources, "Pinned pipeline/evaluator/frozen sources changed")
    for row in evidence.values():
        require(Path(row["path"]).stat().st_size == row["bytes"] and fast.sha(row["path"]) == row["sha256"],
                "Pinned input changed: " + row["path"])


def independent_points(records, saved, metrics):
    return audit.recorded_points(records, {"view_metrics": {"original": saved}}, metrics)


def hard_abort(points):
    chosen = points.get("at_recall90")
    return chosen is None or chosen["recall"] < .90 or chosen["precision"] < .94


def acceptance(points, reference):
    chosen, protected = points.get("at_recall90"), reference["at_recall90"]
    fixed, old_fixed = points["at_confidence_0_75"], reference["at_confidence_0_75"]
    return {"hard_abort_floor": .94, "hard_abort": hard_abort(points),
            "floor_is_promotion_threshold": False,
            "maintains_original_P_at_R90": bool(chosen and chosen["recall"] >= .90
                                                   and chosen["precision"] >= protected["precision"]),
            "maintains_original_P_and_R_at_P90": bool(chosen and chosen["precision"] >= protected["precision"]
                                                        and chosen["recall"] >= protected["recall"]),
            "maintains_original_fixed_075_P_and_R": (fixed["precision"] >= old_fixed["precision"]
                                                     and fixed["recall"] >= old_fixed["recall"]),
            "original_baseline_preserved": True, "automatically_promoted": False}


def claim_directories(output, report):
    output, report = Path(output).resolve(), Path(report).resolve()
    require(output.drive.upper() == report.drive.upper() == "F:", "Every output/report must stay on F:")
    require(output != report and output not in report.parents and report not in output.parents,
            "Output and report must be separate, non-nested directories")
    if output.exists() or report.exists():
        raise FileExistsError("Both output and report must be completely fresh directories")
    GLOBAL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GLOBAL_LOCK.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "created_utc": fast.now(), "output": str(output), "report": str(report)}, stream)
    try:
        output.mkdir(parents=True, exist_ok=False)
        report.mkdir(parents=True, exist_ok=False)
        with (report / "pipeline.lock").open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created_utc": fast.now()}, stream)
    except BaseException:
        GLOBAL_LOCK.unlink(missing_ok=True)
        raise
    return output, report


def load_references(cfg, contract, metrics, sources, evidence):
    """Old cached results are re-scored only at their existing working points."""
    result = {}
    for tag, directory in REFERENCES.items():
        meta = bound_json(directory / "fusion_summary.json", evidence)
        previous.require_full_scope(meta)
        previous.require_sources(meta["source_sha256"], sources)
        require(meta.get("from_fresh_weight_inference") is True and meta.get("fusion") == cfg["fusion"],
                "Historical original-weight fusion lineage/configuration differs")
        for model in ("base", "small"):
            model_meta = bound_json(directory / model / "prediction_metadata.json", evidence)
            previous.require_full_scope(model_meta)
            previous.require_sources(model_meta["source_sha256"], sources)
            require(model_meta.get("model") == model
                    and model_meta.get("checkpoint_sha256") == audit.ORIGINAL_CHECKPOINTS[model]
                    and model_meta.get("provenance_sha256") == contract["provenance_sha256"]
                    and model_meta.get("from_fresh_weight_inference") is True
                    and model_meta.get("training_performed", False) is False
                    and model_meta.get("source_before_after_identical") is True,
                    "Historical model/GT identity differs")
            restore = model_meta.get("strict_restore", {})
            require(restore.get("strict") is True and restore.get("all_tensors_equal_to_checkpoint") is True
                    and restore.get("missing_keys") == [] and restore.get("unexpected_keys") == [],
                    "Historical original strict-restoration receipt missing")
        expected = meta.get("predictions_sha256")
        records = bound_json(directory / "fused_predictions.json", evidence, expected)
        fast.audit_records(records, contract)
        audit.audit_records(records, contract, metrics)
        points = independent_points(records, meta.get("metrics", meta), metrics)
        if tag == "protected_original_FP32":
            gate_path = REPO / "reports/strong_c0c1_20261030/baseline_gate.json"
            gate = bound_json(gate_path, evidence)
            previous.require_sources(gate["source_sha256"], sources)
            require(gate.get("passed") is True and tuple(gate.get(k) for k in ("images", "GT", "empty")) == (814, 3449, 13)
                    and tuple(points["at_recall90"][k] for k in ("tp", "fp", "fn")) == (3105, 183, 344)
                    and gate["at_recall90"] == meta["at_recall90"]
                    and meta.get("candidate_mode") is False,
                    "Protected original 94.4343 baseline/gate differs")
        result[tag] = {"directory": str(directory), "points": points, "records": records,
                       "interpretation": "Historical cached predictions independently re-scored; NOT new GPU inference",
                       "raw_hash_binding": "Metadata SHA verified" if expected else
                       "Legacy original FP32 receipt has no raw SHA; current bytes pinned and stored working points independently verified"}
    return result


def audit_stage(meta, stage, selected, cfg, contract, expected_sources):
    previous.require_full_scope(meta)
    require(meta.get("model") == stage and meta.get("source_sha256") == expected_sources
            and meta.get("config_sha256") == CONFIG_SHA
            and meta.get("provenance_sha256") == contract["provenance_sha256"]
            and meta.get("from_fresh_weight_inference") is True
            and meta.get("original_strong_models_only") is True and meta.get("training_performed") is False
            and meta.get("source_before_after_identical") is True
            and meta.get("config_before_after_identical") is True
            and meta.get("stage2_configuration_before_after_identical") is True
            and meta.get("original_weights_before_after_identical") is True,
            "Stage original-weight/full-scope/configuration/source lineage differs")
    dep = meta["deployment"]
    require(dep.get("deployment_experiment") == "stage2_original_strong_full_val_v1"
            and dep.get("stage2_configuration") == selected["configuration"]
            and dep.get("stage2_config_file") == selected["binding"]
            and dep.get("pre_nms_threshold") is None
            and dep.get("candidate_selection_uses_gt") is False
            and dep.get("precision") == "fp16" and dep.get("roi_precision") == "autocast"
            and dep.get("roi_chunk_size") == 64 and dep.get("short_side") == 1024 and dep.get("max_size") == 1280
            and dep.get("views") == "both" and dep.get("paired_backbone_batch2") is True
            and dep.get("skip_dead_amodal") is False
            and dep.get("evaluation_policy", {}).get("fixed_only") is False,
            "Stage differs from the explicitly selected complete-PR configuration")
    if stage == "fuse":
        require(meta.get("fusion") == cfg["fusion"] and meta.get("mask_fusion") is False,
                "Frozen box fusion differs or unjustified fused-mask claim")
    else:
        restore = meta.get("strict_restore", {})
        require(meta.get("checkpoint_sha256") == audit.ORIGINAL_CHECKPOINTS[stage]
                and meta.get("annotation_sha256") == contract["provenance"]["binding"]["current_annotation_sha256"]
                and meta.get("roi_masks_saved") is True
                and meta.get("constructor_config", {}).get("backbone_name") == "convnext_" + stage
                and restore.get("strict") is True and restore.get("tensor_count") == 434
                and restore.get("all_tensors_equal_to_checkpoint") is True
                and restore.get("missing_keys") == [] and restore.get("unexpected_keys") == [],
                "Original strict-434 model, annotations or saved-mask evidence differs")
    return dep


def summarize(output, report, selected, cfg, contract, references, receipts, sources, evidence, progress):
    import numpy as np
    import flip_tta
    metrics = audit.load_metrics()
    expected_sources = evaluator_sources()
    receipt_by_stage = {row["id"]: row for row in receipts}
    require(len(receipts) == 3 and set(receipt_by_stage) == {"base", "small", "fuse"},
            "Exactly three completed stage receipts required")
    for stage, receipt in receipt_by_stage.items():
        expected_path = output / ("fusion_summary.json" if stage == "fuse" else stage + "/prediction_metadata.json")
        require(receipt.get("phase") == "complete" and receipt.get("exit_code") == 0
                and receipt.get("arguments") == stage_arguments(stage, selected["binding"]["path"], output)
                and Path(receipt.get("metadata", "")).resolve() == expected_path.resolve(),
                "Completed stage receipt differs from the single-candidate command")
    layers, metadata = [], {}
    archives = []
    for model in ("base", "small"):
        receipt = receipt_by_stage[model]
        meta = bound_json(output / model / "prediction_metadata.json", evidence, receipt["metadata_sha256"])
        audit_stage(meta, model, selected, cfg, contract, expected_sources)
        metadata[model] = meta
        for view in ("original", "hflip"):
            path = output / model / (view + "_predictions.json")
            rows = bound_json(path, evidence, meta["predictions_sha256"][view])
            fast.audit_records(rows, contract)
            audit.audit_records(rows, contract, metrics)
            independent_points(rows, meta["view_metrics"][view], metrics)
            layers.append(rows)
            directory = output / model / "roi_masks" / view
            expected_files = {f"{index:06d}.npz" for index in range(814)}
            require({p.name for p in directory.glob("*.npz")} == expected_files,
                    "Complete original/hflip ROI archive inventory differs")
            for index, row in enumerate(rows):
                archive = directory / f"{index:06d}.npz"
                values = audit.load_roi_archive(archive, row, evidence)
                archives.append({"model": model, "view": view, "image_id": row["image_id"],
                                 **evidence[str(archive.resolve())]})
                del values
                if (index + 1) % 100 == 0:
                    progress("audit_ROI_archives", model=model, view=view, images=index + 1)
    fusion = bound_json(output / "fusion_summary.json", evidence, receipt_by_stage["fuse"]["metadata_sha256"])
    audit_stage(fusion, "fuse", selected, cfg, contract, expected_sources)
    for name, meta in metadata.items():
        require(meta["deployment"] == fusion["deployment"]
                and fast.runtime_numerics(meta) == fusion["model_runtime_numerics"],
                "Four-view runtime/configuration differs between original models")
        for label, path in [(name + "_metadata", output / name / "prediction_metadata.json"),
                            *[(name + "_" + view, output / name / (view + "_predictions.json"))
                              for view in ("original", "hflip")]]:
            supplied = fusion["inputs"][label]
            require(Path(supplied["path"]).resolve() == path.resolve()
                    and supplied["sha256"] == evidence[str(path.resolve())]["sha256"],
                    "Fusion input receipt differs: " + label)
    uncut = bound_json(output / "fused_predictions.json", evidence, fusion["predictions_sha256"])
    fast.audit_records(uncut, contract)
    audit.audit_records(uncut, contract, metrics)
    uncut_points = independent_points(uncut, fusion["metrics"], metrics)
    # Persist receipt of the complete existing curve, not another uncut sweep.
    curve_path = output / "fusion_calibration_curve.json"
    audit.bind(curve_path, evidence)
    summary = {"schema": "original_strong_stage2_full_validation_v1", "phase": "summarizing",
               "images": 814, "GT": 3449, "empty": 13, "GT_occlusion_strata": contract["strata"],
               "no_GT_repair_or_exclusion": True, "training_performed": False,
               "stage2": selected, "from_fresh_original_weight_inference": True,
               "uncut_four_view": uncut_points, "derived_cutoff035": None,
               "original_baseline_preserved": True, "automatically_promoted": False,
               "mask_metrics_claimed": False, "mask_fusion": False, "local_tracking_accuracy_claimed": False,
               "stage_receipts": receipts, "source_sha256": sources,
               "references": {tag: {key: value for key, value in ref.items() if key != "records"}
                              for tag, ref in references.items()},
               "acceptance": {"uncut": acceptance(uncut_points, references["protected_original_FP32"]["points"])},
               "derived_artifacts": {}, "limitations": [
                   "All operating points selected on VAL; no independent test-set accuracy claim",
                   "ROI archives are hash-bound and aligned to predictions; saving masks does not establish mask accuracy",
                   "Complete evaluator timings are not warmed deployment FPS",
                   "Four-view .35 filtering can change fusion scores/boxes; equality is not assumed"]}

    def store(name, value):
        path = report / name
        if path.exists():
            raise FileExistsError("Derived artifact already exists: " + str(path))
        fast.write(path, value)
        summary["derived_artifacts"][name] = {"path": str(path), "sha256": fast.sha(path), "bytes": path.stat().st_size}

    store("roi_archive_receipts.json", archives)

    def compare_candidate(tag, rows, points):
        candidate = {"records": rows, "points": points}
        comparison = {name: audit.comparisons(reference, candidate, contract, metrics)
                      for name, reference in references.items()}
        store(tag + "_GT_comparisons.json", comparison)
        return {name: {point: None if value is None else {
            "reference_confidence": value["reference_confidence"], "candidate_confidence": value["candidate_confidence"],
            "reference": value["reference"], "candidate": value["candidate"], "delta": value["delta"],
            "lost": value["lost"], "recovered": value["recovered"], "groups": value["groups"]}
            for point, value in comparisons.items()} for name, comparisons in comparison.items()}

    progress("uncut_GT_comparisons")
    summary["comparisons"] = {"uncut": compare_candidate("uncut", uncut, uncut_points)}
    if hard_abort(uncut_points):
        summary.update(phase="stopped_hard_floor",
                       stop_reason="Complete uncut four-view P@R90 is absent or below 94.0%; candidate stopped; no rescue sweep",
                       derived_cutoff035_status="skipped_due_to_uncut_hard_abort")
    else:
        progress("derive_cutoff035_CPU_only")
        filtered = [previous.filter_records(rows, .35) for rows in layers]
        counts = [{"layer": name, "before": sum(len(r["pred"]["scores"]) for r in rows),
                   "after": sum(len(r["pred"]["scores"]) for r in cut)}
                  for name, rows, cut in zip(("base_original", "base_hflip", "small_original", "small_hflip"), layers, filtered, strict=True)]
        for name, rows in zip(("base_original", "base_hflip", "small_original", "small_hflip"), filtered, strict=True):
            store(name + "_cutoff035_predictions.json", rows)
        derived = flip_tta.fuse_records(filtered, **cfg["fusion"])
        fast.audit_records(derived, contract)
        audit.audit_records(derived, contract, metrics)
        points, curve = fast.summarize(derived, metrics, fast.policy(.35))
        # Verify chosen points against the newly produced complete truncated curve.
        points = {**points, **independent_points(derived, points, metrics)}
        store("cutoff035_fused_predictions.json", derived)
        store("cutoff035_fusion_calibration_curve.json", curve)
        summary["derived_cutoff035"] = {"points": points, "requested_cutoff": .35,
            "effective_float32_cutoff": float(np.float32(.35)), "per_layer_candidate_counts": counts,
            "from_new_complete_predictions": True, "new_GPU_inference_performed": False,
            "derivation": "Exact high-score-prefix filtering of this run's four fresh layers, then frozen original fusion and complete truncated PR calibration",
            "no_loss_or_speed_claim": True}
        summary["acceptance"]["cutoff035"] = acceptance(points, references["protected_original_FP32"]["points"])
        progress("cutoff035_GT_comparisons")
        summary["comparisons"]["cutoff035"] = compare_candidate("cutoff035", derived, points)
        summary["phase"] = "complete"
        summary["derived_cutoff035_status"] = "complete_CPU_derivation"
    progress("final_integrity")
    verify_pins(sources, evidence)
    summary.update(completed_utc=fast.now(), input_and_source_evidence=list(evidence.values()),
                   source_and_input_before_after_identical=True)
    fast.write(report / "validation_summary.json", summary)
    return summary


def run(argv=None):
    args = parser().parse_args(argv)
    require(Path(sys.executable).resolve().drive.upper() == "F:", "Use the existing F-drive Python environment")
    for name, relative in {"TEMP": "tmp", "TMP": "tmp", "HF_HOME": "cache/huggingface",
                           "TORCH_HOME": "cache/torch", "PIP_CACHE_DIR": "cache/pip",
                           "CUDA_CACHE_PATH": "cache/cuda", "XDG_CACHE_HOME": "cache",
                           "PYTHONPYCACHEPREFIX": "cache/pycache"}.items():
        path = REPO / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    selected_path = args.stage2_config.resolve()
    configuration, binding = evaluator.load_stage2_config(selected_path)
    selected = {"configuration": configuration, "binding": binding}
    require(fast.sha(CONFIG_PATH) == CONFIG_SHA, "Frozen original baseline config differs")
    output, report = claim_directories(args.output, args.report)
    status = {"schema": "stage2_single_candidate_full_validation_pipeline_v1", "phase": "initializing",
              "pid": os.getpid(), "started_utc": fast.now(), "stage2": selected,
              "output": str(output), "report": str(report), "stages": [], "child_pid": None,
              "training_performed": False, "candidate_count": 1, "automatic_promotion": False,
              "events": []}
    child, entry, release_lock = None, None, True
    evidence = {}
    sources = {}

    def save():
        status["updated_utc"] = fast.now()
        fast.write(report / "pipeline_status.json", status)
        fast.write(report / "events.json", status["events"])

    def event(kind, **values):
        status["events"].append({"time_utc": fast.now(), "kind": kind, **values})
        save()

    def progress(phase, **values):
        status.update(phase=phase, cpu_summary_progress=values)
        save()

    try:
        save()
        sources = source_inventory()
        cfg = bound_json(CONFIG_PATH, evidence, CONFIG_SHA)
        bound_json(selected_path, evidence, binding["sha256"])
        pathbase = (CONFIG_PATH.parent / cfg["path_base"]).resolve()
        paths = {name: (pathbase / value).resolve() for name, value in cfg["paths"].items()}
        for name in ("base", "small"):
            require(cfg[name + "_checkpoint_sha256"] == audit.ORIGINAL_CHECKPOINTS[name]
                    and audit.bind(paths[name + "_checkpoint"], evidence)["sha256"] == audit.ORIGINAL_CHECKPOINTS[name],
                    "Only the exact original Base/Small checkpoint is authorized")
        from apple_seg.evaluate import load_papple_contract
        import torch
        torch.set_num_threads(cfg["cpu_threads"])
        metrics = audit.load_metrics()
        audit.bind(paths["provenance"], evidence)
        contract = load_papple_contract(paths["provenance"])
        require((len(contract["images"]), contract["scope"]["GT_instances"], contract["scope"]["empty_GT_images"]) == (814, 3449, 13),
                "Original full GT scope differs")
        references = load_references(cfg, contract, metrics, sources, evidence)
        status.update(source_sha256=sources, input_receipts_before_inference=list(evidence.values()),
                      baseline_points=references["protected_original_FP32"]["points"])
        event("inputs_and_protected_baseline_verified")
        for stage in ("base", "small", "fuse"):
            verify_pins(sources, evidence)
            meta_path = output / ("fusion_summary.json" if stage == "fuse" else stage + "/prediction_metadata.json")
            progress_path = output / "fusion_status.json" if stage == "fuse" else meta_path
            arguments = stage_arguments(stage, selected_path, output)
            entry = {"id": stage, "phase": "running", "started_utc": fast.now(), "arguments": arguments,
                     "metadata": str(meta_path), "log": str(report / (stage + ".log"))}
            status["stages"].append(entry)
            status.update(phase="running", current_stage=stage)
            with Path(entry["log"]).open("x", encoding="utf-8") as log:
                child = subprocess.Popen(arguments, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                status["child_pid"] = child.pid
                release_lock = False
                event("stage_started", stage=stage, child_pid=child.pid)
                while child.poll() is None:
                    if progress_path.exists():
                        try:
                            current = json.loads(progress_path.read_text(encoding="utf-8"))
                            entry["progress"] = {key: current.get(key) for key in ("phase", "images", "GT", "updated_utc", "error")}
                        except (OSError, json.JSONDecodeError):
                            pass
                    save()
                    time.sleep(10)
                code = child.returncode
                child = None
                release_lock = True
                status["child_pid"] = None
            entry["exit_code"] = code
            if code:
                entry["phase"] = "failed"
                raise RuntimeError(f"Stage {stage} failed (exit {code}); inspect {entry['log']}")
            meta = bound_json(meta_path, evidence)
            audit_stage(meta, stage, selected, cfg, contract, evaluator_sources())
            entry.update(phase="complete", completed_utc=fast.now(), metadata_sha256=evidence[str(meta_path.resolve())]["sha256"],
                         metrics=meta.get("view_metrics", meta.get("metrics")))
            event("stage_complete", stage=stage, metrics=entry["metrics"])
        progress("summarizing")
        summary = summarize(output, report, selected, cfg, contract, references, status["stages"], sources, evidence, progress)
        status.update(phase=summary["phase"], current_stage=None, child_pid=None, completed_utc=fast.now(),
                      summary=str(report / "validation_summary.json"), summary_sha256=fast.sha(report / "validation_summary.json"),
                      acceptance=summary["acceptance"], stop_reason=summary.get("stop_reason"))
        event("pipeline_" + summary["phase"], acceptance=summary["acceptance"])
        return summary
    except BaseException as error:
        if entry is not None and entry["phase"] == "running":
            entry["phase"] = "failed"
        status.update(phase="failed", error=str(error), traceback=traceback.format_exc(),
                      cleanup={"phase": "stopping_owned_child" if child is not None else "no_child"})
        try:
            event("pipeline_failed", error=str(error))
        except BaseException as persistence_error:
            status.setdefault("persistence_errors", []).append(str(persistence_error))
        if child is not None:
            cleanup = previous.stop_owned_child(child)
            release_lock = cleanup["exit_confirmed"]
            status.update(cleanup=cleanup, child_pid=None if release_lock else child.pid, lock_retained=not release_lock)
            if not release_lock:
                for lock in (GLOBAL_LOCK, report / "pipeline.lock"):
                    try:
                        lock.write_text(json.dumps({"pipeline_pid": os.getpid(), "child_pid": child.pid,
                            "report": str(report), "updated_utc": fast.now(),
                            "reason": "Owned child exit unconfirmed; inspect before removing this lock"}), encoding="utf-8")
                    except BaseException as persistence_error:
                        status.setdefault("persistence_errors", []).append(str(persistence_error))
        try:
            save()
        except BaseException as persistence_error:
            print("FAILED status persistence: " + str(persistence_error), file=sys.stderr, flush=True)
        raise
    finally:
        if release_lock:
            (report / "pipeline.lock").unlink(missing_ok=True)
            GLOBAL_LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    run()
