"""Fresh original-weight Stage 2 diagnostics, never full-VAL acceptance.

One original Base/Small model is resident at a time. All controls are compared
against the same-run R0/F1. No training, threshold search, or GT enters inference.
The primary timer excludes disk decoding, model loading, source/GT auditing and
archive writes. CUDA work is synchronized at frame boundaries; separate profile
frames add synchronized component/pool timing and are excluded from FPS.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "strong_runtime/src")]

from benchmark_premask import ORIGINAL_SHA, latency_summary
from evaluate_fast_deployment import now, sha, source_inventory, write
from summarize_premask_validation import compare_recall, load_metrics

P90_CONFIDENCE = {"base": .7394223809242249, "small": .743447482585907}
CASE_CHANGES = {
    "R0": {"pre_nms_threshold": None},
    "F1": {},
    "T10": {"score_threshold": .10},
    "T15": {"score_threshold": .15},
    "G1": {"preprocess_device": "cuda"},
    "L1": {"channels_last": True},
    "B1": {"cudnn_benchmark": True},
    "H1": {"roi_align_precision": "fp16"},
}


def case_spec(name):
    if name not in CASE_CHANGES:
        raise ValueError("Unknown Stage 2 case: " + name)
    return {"score_threshold": None, "box_nms_iou": .8, "topk": 64,
            "pre_nms_threshold": .738, "preprocess_device": "cpu", "channels_last": False,
            "cudnn_benchmark": False, "roi_align_precision": "fp32", **CASE_CHANGES[name]}


def full_validation_config(spec):
    """Reviewable config only; no candidate promotion, no final-score cutoff."""
    return {"candidate_policy": {k: spec[k] for k in ("score_threshold", "box_nms_iou", "topk")},
            "preprocess_device": spec["preprocess_device"], "channels_last": spec["channels_last"],
            "cudnn_benchmark": spec["cudnn_benchmark"], "cudnn_deterministic": False,
            "roi_align_precision": spec["roi_align_precision"]}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/strong_baseline_4060.json")
    parser.add_argument("--samples", type=Path, default=REPO / "configs/premask_stage2_review_samples_20261102.json")
    parser.add_argument("--benchmark-samples", type=Path, default=REPO / "configs/premask_audit_samples_20261102.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=tuple(ORIGINAL_SHA), default=list(ORIGINAL_SHA))
    parser.add_argument("--cases", nargs="+", choices=tuple(CASE_CHANGES), default=list(CASE_CHANGES))
    parser.add_argument("--limit", type=int, default=41)
    parser.add_argument("--benchmark-count", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--diagnostic-count", type=int, default=3)
    return parser


def validate_args(args):
    if min(args.limit, args.benchmark_count, args.rounds) < 1 or args.warmup < 0:
        raise ValueError("Positive counts/rounds and nonnegative warmup required")
    if not 0 <= args.diagnostic_count <= args.benchmark_count:
        raise ValueError("Diagnostic count must be within benchmark count")
    if len(args.models) != len(set(args.models)) or len(args.cases) != len(set(args.cases)):
        raise ValueError("Duplicate models/cases are not allowed")
    if not args.cases or args.cases[0] != "R0":
        raise ValueError("R0 must be first and freshly inferred in this run")
    if len(args.cases) > 1 and args.cases[1] != "F1":
        raise ValueError("F1 must be second before further independent controls")
    return args


def frozen_subset(contract, samples):
    rows = []
    for sample in samples:
        index, name = sample["image_index"], sample["image_id"]
        if type(index) is not int or not 0 <= index < len(contract["images"]):
            raise ValueError("Sample index is outside frozen contract")
        image = contract["images"][index]
        if image["image_id"] != name or Path(name).name != name:
            raise ValueError("Sample filename/index does not match frozen contract")
        rows.append(image)
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate diagnostic sample")
    truth = {r["image_id"]: contract["truth"][r["image_id"]] for r in rows}
    total = sum(map(len, truth.values()))
    return {**contract, "images": rows, "truth": truth,
            "strata": dict(Counter(r["occlusion_stratum"] for values in truth.values() for r in values)),
            "scope": {"images": len(rows), "GT_instances": total,
                      "empty_GT_images": sum(not values for values in truth.values())},
            "official_full_val": False}


def endpoints(arrays, confidence):
    import numpy as np
    scores = np.asarray(arrays["scores"])
    selected = scores >= confidence
    return {key: np.asarray(arrays[key])[selected] for key in ("boxes", "scores", "visible", "amodal")}


def compare_arrays(reference, candidate, confidence):
    """Ordered native-coordinate endpoint; maxabs only when shapes match."""
    import numpy as np
    old, new = endpoints(reference, confidence), endpoints(candidate, confidence)
    fields = {}
    for key in old:
        a, b = old[key], new[key]
        aligned = a.shape == b.shape
        fields[key] = {"reference_shape": list(a.shape), "candidate_shape": list(b.shape),
                       "same_dtype": str(a.dtype) == str(b.dtype),
                       "exact": aligned and a.dtype == b.dtype and bool(np.array_equal(a, b)),
                       "max_abs_same_output_order": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                           if aligned and a.size else 0. if aligned else None}
    return {"confidence": confidence, "reference_count": len(old["scores"]),
            "candidate_count": len(new["scores"]), "exact": all(r["exact"] for r in fields.values()),
            "fields": fields,
            "alignment": "Original model output order after score filtering; mismatched counts have no maxabs"}


def compare_archive_sets(reference, candidate, thresholds):
    import numpy as np
    if [x["image_id"] for x in reference] != [x["image_id"] for x in candidate]:
        raise ValueError("Archive inventory differs between cases")
    points = {key: {"confidence": confidence, "exact": True, "images": [], "different_images": 0}
              for key, confidence in thresholds.items()}
    for old, new in zip(reference, candidate, strict=True):
        arrays = []
        for entry in (old, new):
            if sha(entry["path"]) != entry["sha256"]:
                raise RuntimeError("ROI archive changed before endpoint comparison")
            with np.load(entry["path"], allow_pickle=False) as archive:
                arrays.append({key: archive[key] for key in ("boxes", "scores", "visible", "amodal")})
        for key, confidence in thresholds.items():
            row = {"image_id": old["image_id"], **compare_arrays(*arrays, confidence)}
            points[key]["images"].append(row)
            points[key]["exact"] &= row["exact"]
            points[key]["different_images"] += int(not row["exact"])
    return points


def enforce_f1_gate(comparison):
    if not comparison or not all(row["exact"] for row in comparison.values()):
        raise RuntimeError("F1 exactness gate failed: halt before all further Stage 2 cases")


def sources():
    files = ["scripts/run_stage2_screen.py", "scripts/benchmark_premask.py",
             "scripts/summarize_premask_validation.py", "scripts/runtime_f_drive.ps1",
             "deployment/stage2.py", "deployment/pre_mask.py", "deployment/candidate_policy.py"]
    return {**source_inventory(), **{name: sha(REPO / name) for name in files}}


def bind(path, receipts):
    path = Path(path).resolve()
    if path.drive.upper() != "F:":
        raise ValueError("All task inputs and outputs must stay on F:")
    entry = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
    if str(path) in receipts and receipts[str(path)] != entry:
        raise RuntimeError("Input changed during execution: " + str(path))
    receipts[str(path)] = entry
    return entry


def tensor_fingerprint(model):
    """Numerical bytes/dtypes independent of strides; outside every timer."""
    import torch
    digest = hashlib.sha256()
    dtypes = Counter()
    state = model.state_dict()
    for key, tensor in state.items():
        dtypes[str(tensor.dtype)] += 1
        digest.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode())
        array = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(array))
        del array
    return {"tensor_count": len(state), "dtype_counts": dict(dtypes), "numeric_sha256": digest.hexdigest()}


@contextmanager
def diagnostic_module_timers(model, enabled):
    """Only used on three separate diagnostic frames, never primary FPS."""
    import torch
    values, handles, starts = {}, [], {}
    if enabled:
        for name in ("backbone", "rpn"):
            def before(module, args, label=name):
                torch.cuda.synchronize()
                starts[label] = time.perf_counter()
            def after(module, args, result, label=name):
                torch.cuda.synchronize()
                values[label + "_ms"] = (time.perf_counter() - starts[label]) * 1000
            handles.extend([getattr(model, name).register_forward_pre_hook(before),
                            getattr(model, name).register_forward_hook(after)])
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()


def backend_flags(torch):
    return {"cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def run(args):
    args = validate_args(args)
    out = args.output.resolve()
    if out.drive.upper() != "F:":
        raise ValueError("Output must stay on F:")
    out.mkdir(parents=True, exist_ok=False)
    receipts, events = {}, []
    status = {"schema": "original_strong_stage2_diagnostic_v1", "phase": "initializing", "created_utc": now(),
              "scope": "Prespecified 41-image regression diagnostic (or explicit smoke subset); NOT 814-image acceptance",
              "training_performed": False, "GT_used_for_inference": False,
              "from_fresh_weight_inference": True, "model_residency": "one original model at a time",
              "model_order": args.models, "case_order": args.cases, "cases": {}, "models": {},
              "precision": "FP16 autocast with original FP32 parameter dtypes; ROIAlign as each case specifies",
              "short_side": 1024, "max_size": 1280, "roi_chunk_size": 64,
              "evaluation_points": {name: {"fixed_075": .75, "original_full_VAL_P90_confidence": P90_CONFIDENCE[name]}
                                    for name in args.models},
              "threshold_search": False, "P90_note": "Frozen original full-VAL confidence, not a recalibrated subset P@R90",
              "smoke_only": args.limit < 41 or args.benchmark_count < 8 or args.rounds < 5 or args.warmup < 5,
              "primary_timer": "Predecoded PIL RGB -> preprocessing/H2D -> GPU forward -> CPU native prediction validation -> >=.75 native V/A binary crops",
              "excluded_from_timer": ["disk decoding", "model loading", "GT audit", "archive writes", "source/hash checks", "diagnostic synchronized timings"],
              "benchmark_count": args.benchmark_count, "warmup_per_case": args.warmup,
              "rounds": args.rounds, "diagnostic_count": args.diagnostic_count}
    write(out / "pipeline_status.json", status)

    def update(message=None, event=None, **fields):
        status.update(updated_utc=now(), **fields)
        write(out / "pipeline_status.json", status)
        if event:
            events.append({"time_utc": now(), "event": event, **fields})
            write(out / "events.json", events)
        if message:
            line = now() + " " + message
            with (out / "screen.log").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
            print(line, flush=True)

    model, torch = None, None
    initial_backend = None
    try:
        config_path, sample_path, bench_path = args.config.resolve(), args.samples.resolve(), args.benchmark_samples.resolve()
        for path in (config_path, sample_path, bench_path):
            bind(path, receipts)
        cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
        manifest = json.loads(sample_path.read_text(encoding="utf-8-sig"))
        benchmark_manifest = json.loads(bench_path.read_text(encoding="utf-8-sig"))
        if args.limit > len(manifest["samples"]) or args.benchmark_count > len(benchmark_manifest["samples"]):
            raise ValueError("Requested count exceeds the fixed diagnostic/benchmark manifest")
        samples, benchmark_samples = manifest["samples"][:args.limit], benchmark_manifest["samples"][:args.benchmark_count]
        old = manifest.get("original_32_manifest", {})
        if old.get("sha256") != sha(bench_path):
            raise ValueError("Benchmark manifest differs from original 32-image binding")
        root = (config_path.parent / cfg["path_base"]).resolve()
        paths = {key: (root / value).resolve() for key, value in cfg["paths"].items()}
        for name in args.models:
            if cfg[name + "_checkpoint_sha256"] != ORIGINAL_SHA[name]:
                raise ValueError("Only the original Base/Small weights are accepted")
            if bind(paths[name + "_checkpoint"], receipts)["sha256"] != ORIGINAL_SHA[name]:
                raise ValueError("Original checkpoint bytes differ: " + name)
        status.update(source_sha256=sources(), samples=samples, benchmark_samples=benchmark_samples,
                      case_specs={name: case_spec(name) for name in args.cases})
        import numpy as np
        import torch
        from PIL import Image
        from apple_seg.evaluate import load_papple_contract, _audit_target, _audit_rgb
        from apple_seg.data import PAppleDataset
        from apple_seg.od_training import load_checkpoint
        from apple_seg.od_model import build_od_amodal_detector
        from apple_seg.od_inference import _strict_restore
        from apple_seg.od_evaluate import _validate_prediction
        from apple_seg.od_roi import paste_roi_mask
        from deployment.stage2 import Stage2DeploymentAdapter, prepare_pixels
        metrics = load_metrics()
        torch.set_num_threads(cfg["cpu_threads"])
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU or weaker-model fallback")
        initial_backend = backend_flags(torch)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        status["runtime"] = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                             "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                             "cpu_threads": torch.get_num_threads(), "initial_backend": initial_backend}
        bind(paths["provenance"], receipts)
        contract = load_papple_contract(paths["provenance"])
        subset = frozen_subset(contract, samples)
        frozen_subset(contract, benchmark_samples)
        status.update(diagnostic_scope=subset["scope"], diagnostic_strata=subset["strata"],
                      provenance_sha256=contract["provenance_sha256"])
        native = PAppleDataset(paths["papple_root"], "val", 32, augment=False)
        if native.annotation_sha256 != contract["provenance"]["binding"]["current_annotation_sha256"]:
            raise RuntimeError("Original annotation bytes differ")
        for path in native.annotation_paths.values():
            bind(path, receipts)
        rgbs = {}
        for sample in samples + benchmark_samples:
            name = sample["image_id"]
            if name in rgbs:
                continue
            path = paths["papple_root"] / "images/val" / name
            bind(path, receipts)
            _audit_rgb(path, contract["images"][sample["image_index"]])
            with Image.open(path) as image:
                rgbs[name] = image.convert("RGB")
        audited_GT = set()
        for name in args.models:
            update("Loading original " + name, "model_loading", phase="loading", current_model=name)
            payload = load_checkpoint(paths[name + "_checkpoint"])
            constructor = deepcopy(payload["constructor_config"])
            model = build_od_amodal_detector(weights=None, **constructor)
            restored = _strict_restore(model, payload["model"])
            if restored["tensor_count"] != 434 or model.constructor_config != constructor:
                raise RuntimeError("Original strict-434 constructor/restoration contract differs")
            baseline_fingerprint = tensor_fingerprint(model)
            if any(value.is_floating_point() and value.dtype != torch.float32 for value in model.state_dict().values()):
                raise RuntimeError("Original floating checkpoint tensors must stay FP32")
            del payload
            gc.collect()
            model.requires_grad_(False).eval().to("cuda")
            original_chunk = model.roi_heads.roi_chunk_size
            status["models"][name] = {"checkpoint": str(paths[name + "_checkpoint"]), "checkpoint_sha256": ORIGINAL_SHA[name],
                                      "constructor_config": constructor, "strict_restore": restored,
                                      "original_tensor_fingerprint": baseline_fingerprint}
            references = {}
            for case_name in args.cases:
                spec = case_spec(case_name)
                key = name + "_" + case_name
                directory = out / name / case_name
                (directory / "roi_masks").mkdir(parents=True, exist_ok=False)
                entry = {"phase": "running", "model": name, "case": case_name, "spec": spec,
                         "scope": status["scope"], "created_utc": now(), "images_completed": 0}
                status["cases"][key] = entry
                torch.backends.cudnn.benchmark = spec["cudnn_benchmark"]
                memory_format = torch.channels_last if spec["channels_last"] else torch.contiguous_format
                model.to(memory_format=memory_format)
                entry["backend_flags"] = backend_flags(torch)
                update("Starting " + key, "case_started", phase="diagnostic_inference", current_case=key)
                records, archives, inference_frames = [], [], []
                candidate_policy = {k: spec[k] for k in ("score_threshold", "box_nms_iou", "topk")}
                adapter = Stage2DeploymentAdapter(model, candidate_policy=candidate_policy, capture_candidates=False,
                    pre_nms_threshold=spec["pre_nms_threshold"], roi_chunk_size=64, skip_dead_amodal=False,
                    roi_precision="autocast", roi_align_precision=spec["roi_align_precision"],
                    profile_stages=False, profile_pools=False)
                with adapter, torch.inference_mode():
                    def infer(sample, diagnostic=False):
                        rgb = rgbs[sample["image_id"]]
                        native_hw = (rgb.height, rgb.width)
                        adapter.profile_stages = diagnostic
                        adapter.profile_pools = diagnostic
                        with diagnostic_module_timers(model, diagnostic) as component_times:
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            pixels = prepare_pixels(rgb, 1024, 1280, "cuda",
                                preprocess_device=spec["preprocess_device"], channels_last=spec["channels_last"])
                            input_hw = tuple(pixels.shape[-2:])
                            pixel_mask = torch.ones((1, *input_hw), device="cuda", dtype=torch.bool)
                            if diagnostic:
                                torch.cuda.synchronize()
                            t1 = time.perf_counter()
                            with torch.autocast("cuda", dtype=torch.float16):
                                raw = adapter.predict_views(pixels, pixel_mask)
                            if diagnostic:
                                torch.cuda.synchronize()
                            t2 = time.perf_counter()
                            prediction = _validate_prediction(raw, input_hw, native_hw, mask_chunk_size=8)
                            if diagnostic:
                                torch.cuda.synchronize()
                            t3 = time.perf_counter()
                            selected = [i for i, score in enumerate(prediction["scores"]) if score >= .75]
                            for index in selected:
                                for branch in ("visible", "amodal"):
                                    pasted = paste_roi_mask(prediction[branch][index], prediction["boxes"][index],
                                                            native_hw, mask_threshold=.5)
                                    del pasted
                            torch.cuda.synchronize()
                            t4 = time.perf_counter()
                        stats = deepcopy(adapter.last_stats)
                        if adapter.traces or len(stats.get("pre_mask", [])) != 1:
                            raise RuntimeError("Unexpected captured tensors or view count")
                        row = {"image_id": sample["image_id"], "end_to_end_ms": (t4-t0)*1000,
                               "input_hw": list(input_hw), "native_output_hw": list(native_hw),
                               "selected_count": len(selected), "native_VA_binary_crop_count": 2*len(selected),
                               "output_count": len(prediction["scores"]),
                               "mask_candidates": stats["pre_mask"][0]["mask_candidates"],
                               "decoded_candidates": stats["pre_mask"][0]["decoded_candidates"],
                               "input_channels_last": pixels.is_contiguous(memory_format=torch.channels_last),
                               "adapter_stats": stats}
                        if diagnostic:
                            row["synchronized_ms"] = {"preprocess_H2D_and_pixel_mask": (t1-t0)*1000,
                                "model_call": (t2-t1)*1000, "native_prediction_validation": (t3-t2)*1000,
                                "fixed_filter_native_VA_paste": (t4-t3)*1000, **component_times}
                        return prediction, row

                    for seq, sample in enumerate(samples):
                        prediction, row = infer(sample)
                        # Real GT is loaded/audited only after this image's model output.
                        image_id, index = sample["image_id"], sample["image_index"]
                        if image_id not in audited_GT:
                            item = native[index]
                            _audit_target(item["target"], contract["images"][index], contract["truth"][image_id])
                            audited_GT.add(image_id)
                            del item
                        records.append({"image_id": image_id,
                            "gt": {"boxes": [r["box"] for r in contract["truth"][image_id]]},
                            "pred": {"boxes": prediction["boxes"].tolist(), "scores": prediction["scores"]}})
                        arrays = {key: prediction[key].numpy() for key in ("boxes", "visible", "amodal")}
                        arrays["scores"] = np.asarray(prediction["scores"], dtype=np.float32)
                        archive = directory / "roi_masks" / f"{seq:06d}.npz"
                        with archive.open("xb") as stream:
                            np.savez_compressed(stream, **arrays)
                        archives.append({"image_id": image_id, "manifest_index": index, "path": str(archive),
                                         "sha256": sha(archive), "bytes": archive.stat().st_size,
                                         "shape": list(arrays["visible"].shape), "dtype": "float32"})
                        inference_frames.append(row)
                        entry.update(images_completed=seq+1)
                        update(f"{key} inference {seq+1}/{len(samples)}", completed_current_case=seq+1)
                        del prediction, arrays
                    write(directory / "predictions.json", records)
                    write(directory / "roi_archive_manifest.json", archives)
                    write(directory / "inference_frames.json", {"not_benchmark": True, "frames": inference_frames})
                    thresholds = {"fixed_075": .75, "original_full_VAL_P90_confidence": P90_CONFIDENCE[name]}
                    entry["points"] = {point: metrics.evaluate_at_threshold(records, confidence, .5)
                                       for point, confidence in thresholds.items()}
                    entry["comparisons"] = {}
                    for reference_name, reference in references.items():
                        if reference_name not in ("R0", "F1"):
                            continue
                        comparison = {"endpoint": compare_archive_sets(reference["archives"], archives,
                            {"cutoff_0738": .738, **thresholds}),
                            "recall": {point: compare_recall(reference["records"], records, subset,
                                                            confidence, confidence, metrics)
                                       for point, confidence in thresholds.items()}}
                        write(directory / ("comparison_to_" + reference_name + ".json"), comparison)
                        entry["comparisons"][reference_name] = {
                            "endpoint_exact": {p: v["exact"] for p, v in comparison["endpoint"].items()},
                            "recall": {p: {"lost": v["lost"], "recovered": v["recovered"],
                                "heavy": v["groups"].get("heavy_visible_lt_0.40", {})}
                                for p, v in comparison["recall"].items()}}
                        if case_name == "F1" and reference_name == "R0":
                            entry["F1_gate"] = {p: r["exact"] for p, r in comparison["endpoint"].items()}
                            write(directory / "summary.json", entry)
                            enforce_f1_gate(comparison["endpoint"])
                    references[case_name] = {"records": records, "archives": archives}
                    update("Accuracy diagnostic complete " + key, "diagnostic_complete", phase="benchmark")
                    for i in range(args.warmup):
                        infer(benchmark_samples[i % len(benchmark_samples)])
                    torch.cuda.reset_peak_memory_stats()
                    frames = []
                    for round_index in range(args.rounds):
                        for sample in benchmark_samples:
                            prediction, row = infer(sample)
                            row["round"] = round_index + 1
                            frames.append(row)
                            del prediction
                            update(f"{key} benchmark {len(frames)}/{args.rounds*args.benchmark_count} {row['end_to_end_ms']:.2f} ms",
                                   benchmark_frames_completed=len(frames))
                    peak = {"allocated_bytes": torch.cuda.max_memory_allocated(), "reserved_bytes": torch.cuda.max_memory_reserved()}
                    profile_frames = []
                    for sample in benchmark_samples[:args.diagnostic_count]:
                        prediction, row = infer(sample, diagnostic=True)
                        profile_frames.append(row)
                        del prediction
                    timing = {"scope": status["primary_timer"], "excluded": status["excluded_from_timer"],
                              "summary": latency_summary([r["end_to_end_ms"] for r in frames]),
                              "frames": frames, "diagnostic_frames_excluded_from_FPS": profile_frames,
                              "peak_cuda": peak, "backend_flags": entry["backend_flags"],
                              "ROI_counts": {"mean": float(np.mean([r["mask_candidates"] for r in frames])),
                                             "p50": float(np.percentile([r["mask_candidates"] for r in frames],50)),
                                             "p95": float(np.percentile([r["mask_candidates"] for r in frames],95)),
                                             "max": max(r["mask_candidates"] for r in frames)}}
                    write(directory / "timing.json", timing)
                fingerprint = tensor_fingerprint(model)
                if fingerprint != baseline_fingerprint:
                    raise RuntimeError("Original parameter numerical bytes/dtypes changed during case")
                if model.roi_heads.roi_chunk_size != original_chunk or model.constructor_config != constructor:
                    raise RuntimeError("Adapter did not restore original constructor/chunk")
                if any(k in model.roi_heads.__dict__ for k in ("forward", "_strong_fast_adapter_active", "_roi_predictions", "_decoded_boxes", "_final_nms", "_mask_predictions")):
                    raise RuntimeError("Temporary model overrides were not restored")
                entry.update(phase="complete", completed_utc=now(), numeric_parameters_unchanged=True,
                             tensor_fingerprint_after=fingerprint, latency=timing["summary"], ROI_counts=timing["ROI_counts"],
                             peak_cuda=peak, predictions_sha256=sha(directory / "predictions.json"),
                             roi_archive_manifest_sha256=sha(directory / "roi_archive_manifest.json"))
                write(directory / "full_validation_candidate_config.json", full_validation_config(spec))
                entry["full_validation_candidate_config_sha256"] = sha(directory / "full_validation_candidate_config.json")
                entry["candidate_config_note"] = "For explicit review only, not promoted; complete validation restores all final-score candidates"
                write(directory / "summary.json", entry)
                update("Completed " + key, "case_complete", phase="case_complete", current_case=key)
                model.to(memory_format=torch.contiguous_format)
                del adapter
            if sha(paths[name + "_checkpoint"]) != ORIGINAL_SHA[name]:
                raise RuntimeError("Original checkpoint file changed")
            status["models"][name]["original_weights_before_after_identical"] = True
            del model
            model = None
            gc.collect()
            torch.cuda.empty_cache()
        if sources() != status["source_sha256"]:
            raise RuntimeError("Inference sources changed during execution")
        for receipt in list(receipts.values()):
            bind(receipt["path"], receipts)
        status.update(input_receipts=list(receipts.values()), source_before_after_identical=True,
                      inputs_before_after_identical=True, full_validation_accepted=False,
                      acceptance_note="Diagnostic screening only; all 814 images and four-view fusion still require separate verification")
        update("Stage 2 diagnostic completed", "complete", phase="complete", completed_utc=now())
        write(out / "summary.json", status)
        return status
    except BaseException as error:
        current = status["cases"].get(status.get("current_case"))
        if current is not None and current["phase"] != "complete":
            current.update(phase="failed", error=str(error))
            write(out / current["model"] / current["case"] / "summary.json", current)
        update(str(error), "failed", phase="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if torch is not None and initial_backend is not None:
            torch.backends.cudnn.benchmark = initial_backend["cudnn_benchmark"]
            torch.backends.cudnn.deterministic = initial_backend["cudnn_deterministic"]
            torch.backends.cudnn.allow_tf32 = initial_backend["cudnn_allow_tf32"]
            torch.backends.cuda.matmul.allow_tf32 = initial_backend["cuda_matmul_allow_tf32"]


if __name__ == "__main__":
    run(build_parser().parse_args())
