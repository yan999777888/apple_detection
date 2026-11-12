"""Bounded, read-only-model T15 Base+Small four-view deployment benchmark.

No training, GT evaluation, camera, depth, tracking or fused-mask claim. Results
are from fresh inference; both original models stay resident on one CUDA GPU.
Only the explicit new output directory is written. Run --check-only first.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["YOLO_AUTOINSTALL"] = "False"

ORIGINAL_SHA = {
    "base": "0fe257c5bfe115a77ac7deac830af479e4c2b0ad6ff69e609901b24c7980fa9b",
    "small": "e0c1f659c2509d7a01d617e12cb83c673768b440ea5b4b96cb4f763e4207e7e3",
}
T15 = {
    "candidate_policy": {"score_threshold": .15, "box_nms_iou": .8, "topk": 64},
    "preprocess_device": "cpu", "channels_last": False,
    "cudnn_benchmark": False, "cudnn_deterministic": False,
    "roi_align_precision": "fp32",
}
FUSION = {
    "iou_threshold": .45, "score_mode": "mean", "box_mode": "score_weighted",
    "coverage_floor": 1.0, "view_weights": [.65, .65, .35, .35],
    "consensus_groups": [0, 0, 1, 1], "consensus_bonus": 1.10,
    "isolation_penalty": .90,
}
MODES = {"uncut": None, "cutoff035": .35}
WARMUP, ROUNDS, SAMPLE_COUNT, DIAGNOSTICS = 5, 5, 8, 3


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def latency_summary(values):
    values = sorted(float(v) for v in values)
    require(bool(values) and all(math.isfinite(v) and v > 0 for v in values), "Invalid latency")
    position = .95 * (len(values) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    mean = statistics.mean(values)
    return {"n_source_frames": len(values), "mean_ms": mean,
            "median_ms": statistics.median(values),
            "p95_ms": values[lo] + (values[hi] - values[lo]) * (position - lo),
            "min_ms": values[0], "max_ms": values[-1], "source_frames_per_second": 1000 / mean}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the read-only source repo")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs/hashes without importing torch or writing output")
    return parser


def inventory(repo):
    files = [Path(__file__).resolve(), repo / "strong_runtime/tmp/flip_tta.py",
             repo / "strong_runtime/reference/local_amodal/scripts/metrics.py"]
    files += sorted((repo / "strong_runtime/src/apple_seg").glob("*.py"))
    files += sorted((repo / "deployment").glob("*.py"))
    return {str(path): sha(path) for path in files}


def prepare(args):
    repo, output = args.repo.resolve(), args.output.resolve()
    require(repo.is_dir(), "Source repo does not exist")
    require(not output.exists(), "Output already exists; refusing to overwrite")
    require(not output.is_relative_to(repo), "Write output outside the read-only source repo")
    cfg_path = repo / "configs/strong_baseline_4060.json"
    stage_path = repo / "configs/stage2_selected_20261102.json"
    samples_path = repo / "configs/premask_audit_samples_20261102.json"
    cfg, stage, manifest = read(cfg_path), read(stage_path), read(samples_path)
    require(stage == T15, "T15 selected config differs from the frozen benchmark recipe")
    require(cfg["fusion"] == FUSION, "Original four-view fusion recipe changed")
    require(cfg["short_side"] == 1024 and cfg["max_size"] == 1280, "Original global input size changed")
    root = (cfg_path.parent / cfg["path_base"]).resolve()
    paths = {key: (root / value).resolve() for key, value in cfg["paths"].items()}
    receipts = {str(path): sha(path) for path in (cfg_path, stage_path, samples_path)}
    for name, expected in ORIGINAL_SHA.items():
        checkpoint = paths[name + "_checkpoint"]
        require(cfg[name + "_checkpoint_sha256"] == expected, "Checkpoint configuration differs: " + name)
        require(sha(checkpoint) == expected, "Checkpoint bytes differ: " + name)
        receipts[str(checkpoint)] = expected
    rows = manifest["samples"][:SAMPLE_COUNT]
    require(len(rows) == SAMPLE_COUNT and len({r["image_id"] for r in rows}) == SAMPLE_COUNT, "Need eight distinct fixed samples")
    samples = []
    for row in rows:
        rgb = paths["papple_root"] / "images/val" / row["image_id"]
        receipts[str(rgb)] = sha(rgb)
        samples.append({"image_id": row["image_id"], "rgb_path": str(rgb), "rgb_sha256": receipts[str(rgb)]})
    return repo, output, cfg, paths, samples, receipts


def gpu_snapshot():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,utilization.gpu,power.draw,temperature.gpu", "--format=csv,noheader"], text=True, timeout=10).strip()
    except Exception as error:
        return {"unavailable": str(error)}


def fingerprint(model):
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(model.state_dict().items()):
        data = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(data.dtype).encode("ascii"))
        digest.update(str(data.shape).encode("ascii"))
        digest.update(memoryview(data).cast("B"))
        count += 1
    return {"tensor_count": count, "sha256": digest.hexdigest()}


def run(args):
    repo, output, cfg, paths, samples, receipts = prepare(args)
    source_before = inventory(repo)
    if args.check_only:
        print(json.dumps({"phase": "check_only_passed", "GPU_initialized": False,
                          "output_created": False, "samples": samples, "modes": MODES,
                          "warmup_per_mode": WARMUP, "rounds_per_mode": ROUNDS,
                          "source_frames_per_mode": SAMPLE_COUNT * ROUNDS,
                          "diagnostic_frames_per_mode": DIAGNOSTICS,
                          "source_files": len(source_before), "input_files": len(receipts)}, ensure_ascii=False, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    runtime.mkdir()
    for name in ("TEMP", "TMP", "TORCH_HOME", "HF_HOME", "MPLCONFIGDIR", "XDG_CACHE_HOME",
                 "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR", "NUMBA_CACHE_DIR"):
        directory = runtime / name.lower()
        directory.mkdir()
        os.environ[name] = str(directory)
    sys.path[:0] = [str(repo / "strong_runtime/src"), str(repo / "strong_runtime/tmp"), str(repo)]
    status = {
        "schema": "global_T15_combined_four_view_benchmark_v1", "phase": "initializing",
        "pid": os.getpid(), "created_utc": now(), "repo_read_only": str(repo),
        "source_sha256_before": source_before, "input_sha256_before": receipts,
        "samples": samples, "stage2": T15, "fusion": FUSION,
        "precision": "FP16 autocast; original FP32 parameters; FP32 ROIAlign",
        "roi_chunk_size": 64, "short_side": 1024, "max_size": 1280,
        "model_order": ["base", "small"], "views_per_model": ["original", "hflip"],
        "model_residency": "Both original models continuously resident on CUDA; models run serially; each backbone batch=2",
        "timing_scope": "Predecoded PIL RGB -> CPU preprocessing -> H2D/flip -> Base original+hflip -> Small original+hflip -> CPU validation/flipback -> original fusion -> >=0.75 CPU boxes and scores",
        "timer": "perf_counter wall clock with CUDA sync only at complete frame boundaries; diagnostic pass synchronizes stages separately",
        "excludes": ["disk decode", "model initialization", "weight/source hashes", "result file serialization", "drawing", "camera acquisition", "depth", "geometry", "tracking", "robot communication"],
        "warmup_per_mode": WARMUP, "rounds_per_mode": ROUNDS,
        "source_frames_per_mode": SAMPLE_COUNT * ROUNDS, "diagnostic_frames_per_mode": DIAGNOSTICS,
        "FPS_definition": "1000 / mean complete source-frame milliseconds; never multiply by number of views",
        "mask_fusion": False, "accuracy_evaluated": False, "training_performed": False,
        "GT_used_for_inference": False, "test_split_used": False,
        "sample_selection_note": "First eight fixed development samples from existing manifest; original selection may use GT offline; inference never accesses annotations",
        "camera_to_robot_end_to_end": False, "models": {}, "cases": {},
        "user_reported_4_FPS_reproduced": False,
        "candidate_scope": "New T15 combined benchmark; does not establish the user's current deployment configuration",
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    write(output / "status.json", status)
    torch, models, original_chunks = None, {}, {}
    try:
        import numpy as np
        import torch
        from PIL import Image
        import flip_tta
        from apple_seg.od_training import load_checkpoint
        from apple_seg.od_model import build_od_amodal_detector
        from apple_seg.od_inference import _strict_restore
        from apple_seg.od_evaluate import _validate_prediction
        from deployment.stage2 import Stage2DeploymentAdapter, prepare_pixels

        torch.set_num_threads(cfg["cpu_threads"])
        torch.set_num_interop_threads(1)
        require(torch.cuda.is_available(), "CUDA unavailable; no CPU/model fallback")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        status["environment"].update(torch=torch.__version__, cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
            gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            cpu_threads=torch.get_num_threads(), gpu_before=gpu_snapshot(),
            tf32_matmul=False, tf32_cudnn=True, cudnn_benchmark=False, cudnn_deterministic=False)
        rgbs = []
        for sample in samples:
            with Image.open(sample["rgb_path"]) as image:
                rgb = image.convert("RGB")
            require(rgb.size == (1300, 1300), "Unexpected fixed sample dimensions")
            rgbs.append(rgb)
        for name in ("base", "small"):
            status.update(phase="loading", current_model=name, updated_utc=now())
            write(output / "status.json", status)
            payload = load_checkpoint(paths[name + "_checkpoint"])
            constructor = deepcopy(payload["constructor_config"])
            model = build_od_amodal_detector(weights=None, **constructor)
            restored = _strict_restore(model, payload["model"])
            require(restored["tensor_count"] == 434 and model.constructor_config == constructor, "Strict restore/constructor mismatch")
            require(all(not t.is_floating_point() or t.dtype == torch.float32 for t in model.state_dict().values()), "Original parameter dtype changed")
            status["models"][name] = {"checkpoint": str(paths[name + "_checkpoint"]),
                "checkpoint_sha256": ORIGINAL_SHA[name], "constructor_config": constructor,
                "strict_restore": restored, "tensor_fingerprint_before": fingerprint(model),
                "use_amodal_refinement": bool(model.roi_heads.use_amodal_refinement)}
            require(model.roi_heads.use_amodal_refinement, "Do not bypass genuine amodal branches")
            del payload
            gc.collect()
            model.requires_grad_(False).eval().to("cuda")
            original_chunks[name] = model.roi_heads.roi_chunk_size
            models[name] = model
        model = None
        status["loaded_cuda_allocated_bytes"] = torch.cuda.memory_allocated()

        def sync():
            torch.cuda.synchronize()
            return time.perf_counter()

        for mode, cutoff in MODES.items():
            case = {"phase": "running", "branch_final_score_pre_NMS_cutoff": cutoff,
                    "fused_output_confidence": .75, "frames": [], "diagnostic_frames": [],
                    "first_round_predictions": [], "fresh_inference": True}
            status["cases"][mode] = case
            status.update(phase="benchmarking", current_mode=mode, updated_utc=now())
            write(output / "status.json", status)
            with ExitStack() as stack, torch.inference_mode():
                adapters = {name: stack.enter_context(Stage2DeploymentAdapter(models[name],
                    candidate_policy=T15["candidate_policy"], capture_candidates=False,
                    profile_stages=False, profile_pools=False, roi_align_precision="fp32",
                    pre_nms_threshold=cutoff, roi_chunk_size=64, skip_dead_amodal=False,
                    roi_precision="autocast")) for name in ("base", "small")}

                def infer(rgb, image_id, diagnostic=False):
                    mark = sync if diagnostic else time.perf_counter
                    start = sync()
                    pixels = prepare_pixels(rgb, 1024, 1280, "cuda", preprocess_device="cpu", channels_last=False)
                    input_hw, native_hw = tuple(pixels.shape[-2:]), (rgb.height, rgb.width)
                    pair = torch.cat((pixels, pixels.flip(-1).contiguous()), dim=0)
                    pixel_mask = torch.ones((2, *input_hw), dtype=torch.bool, device="cuda")
                    previous = mark()
                    diagnostic_ms = {"preprocess_H2D_flip_mask": (previous - start) * 1000}
                    layers, counts = [], []
                    for name in ("base", "small"):
                        with torch.autocast("cuda", dtype=torch.float16):
                            raw = adapters[name].predict_views(pair, pixel_mask)
                        after_forward = mark()
                        require(len(raw) == 2, "Paired inference must return original and hflip")
                        for view, item in enumerate(raw):
                            pred = _validate_prediction([item], input_hw, native_hw, mask_chunk_size=8)
                            boxes = pred["boxes"].clone()
                            if view:
                                old = boxes.clone()
                                boxes[:, 0], boxes[:, 2] = native_hw[1] - old[:, 2], native_hw[1] - old[:, 0]
                            box_list = boxes.tolist()
                            scores = [float(score) for score in pred["scores"]]
                            layers.append({"image_id": image_id, "pred": {"boxes": box_list, "scores": scores}})
                            counts.append({"model": name, "view": "hflip" if view else "original", "candidates": len(scores)})
                            del pred, boxes, item
                        del raw
                        after_decode = mark()
                        diagnostic_ms[name + "_forward"] = (after_forward - previous) * 1000
                        diagnostic_ms[name + "_CPU_validation_flipback"] = (after_decode - after_forward) * 1000
                        previous = after_decode
                    boxes, scores = flip_tta.fuse_entries(layers, **FUSION)
                    after_fusion = mark()
                    keep = np.flatnonzero(scores >= .75)
                    final_boxes, final_scores = boxes[keep].tolist(), scores[keep].tolist()
                    end = sync()
                    diagnostic_ms["fusion"] = (after_fusion - previous) * 1000
                    diagnostic_ms["final_filter_CPU_delivery"] = (end - after_fusion) * 1000
                    # Everything below is outside the inference timer.
                    require(all(math.isfinite(v) for box in final_boxes for v in box) and all(math.isfinite(v) for v in final_scores), "Nonfinite result")
                    row = {"image_id": image_id, "complete_source_frame_ms": (end - start) * 1000,
                           "input_hw": list(input_hw), "native_hw": list(native_hw),
                           "selected_count": len(final_scores), "view_candidate_counts": counts,
                           "mask_candidate_counts": {n: [r["mask_candidates"] for r in adapters[n].last_stats["pre_mask"]] for n in adapters}}
                    if diagnostic:
                        row["synchronized_stage_ms"] = diagnostic_ms
                    require(all(not a.traces for a in adapters.values()), "Unexpected retained candidate tensors")
                    return row, {"image_id": image_id, "boxes": final_boxes, "scores": final_scores}

                for i in range(WARMUP):
                    infer(rgbs[i % SAMPLE_COUNT], samples[i % SAMPLE_COUNT]["image_id"])
                # Stage-synchronized diagnostics are a separate pass, excluded below.
                for i in range(DIAGNOSTICS):
                    row, _ = infer(rgbs[i], samples[i]["image_id"], diagnostic=True)
                    case["diagnostic_frames"].append(row)
                torch.cuda.reset_peak_memory_stats()
                loop_start = time.perf_counter()
                for repeat in range(ROUNDS):
                    for rgb, sample in zip(rgbs, samples):
                        row, prediction = infer(rgb, sample["image_id"])
                        row["round"] = repeat + 1
                        case["frames"].append(row)
                        if repeat == 0:
                            case["first_round_predictions"].append(prediction)
                    status.update(current_round=repeat + 1, completed_current_mode_frames=len(case["frames"]), updated_utc=now())
                    write(output / "status.json", status)
                    print(f"GLOBAL_BENCH {mode} round {repeat + 1}/{ROUNDS}", flush=True)
                case["measured_loop_wall_ms_including_housekeeping"] = (time.perf_counter() - loop_start) * 1000
                case["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                case["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
            case["summary"] = latency_summary([r["complete_source_frame_ms"] for r in case["frames"]])
            case["diagnostic_stage_mean_ms"] = {key: statistics.mean(row["synchronized_stage_ms"][key] for row in case["diagnostic_frames"]) for key in case["diagnostic_frames"][0]["synchronized_stage_ms"]}
            case.update(phase="complete", completed_utc=now())
            write(output / (mode + ".json"), case)
            write(output / "status.json", status)
            print("GLOBAL_BENCH_RESULT " + mode + " " + json.dumps(case["summary"]), flush=True)

        for name, model in models.items():
            after = fingerprint(model)
            require(after == status["models"][name]["tensor_fingerprint_before"], "Model tensor values changed: " + name)
            require(model.roi_heads.roi_chunk_size == original_chunks[name], "Adapter did not restore ROI chunk")
            status["models"][name]["tensor_fingerprint_after"] = after
        source_after = inventory(repo)
        input_after = {path: sha(path) for path in receipts}
        require(source_after == source_before, "Source bytes changed during benchmark")
        require(input_after == receipts, "Input/config/checkpoint bytes changed during benchmark")
        status.update(phase="complete", completed_utc=now(), source_sha256_after=source_after,
                      input_sha256_after=input_after, source_inputs_models_unchanged=True,
                      adapters_restored=True, gpu_after=gpu_snapshot())
        write(output / "benchmark_results.json", status)
        write(output / "status.json", status)
        return 0
    except BaseException as error:
        status.update(phase="failed", failed_utc=now(), error=str(error), traceback=traceback.format_exc(),
                      complete_benchmark_claim=False)
        write(output / "status.json", status)
        raise
    finally:
        models.clear()
        gc.collect()
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
