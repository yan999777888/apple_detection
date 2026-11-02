"""Measure original-weight pre-mask policies; no GT or accuracy claims.

Default: first eight frozen audit images, three warmups, two timed rounds,
then three separate synchronized diagnostic images, per policy and model.
RGB decoding and model loading are excluded. Optional centre cropping is
inside the timer and is only a local-window speed simulation, not D405 data.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


REPO = Path(__file__).resolve().parents[1]
ORIGINAL_SHA = {
    "base": "0fe257c5bfe115a77ac7deac830af479e4c2b0ad6ff69e609901b24c7980fa9b",
    "small": "e0c1f659c2509d7a01d617e12cb83c673768b440ea5b4b96cb4f763e4207e7e3",
}
POLICIES = {
    "none": {},
    "k64": {"topk": 64},
    "nms080_k64": {"box_nms_iou": .8, "topk": 64},
    "score003_nms080_k64": {"score_threshold": .03, "box_nms_iou": .8, "topk": 64},
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    files = [Path(__file__), *(REPO / "deployment" / name for name in
             ("candidate_policy.py", "pre_mask.py", "strong_fast.py")),
             *sorted((REPO / "strong_runtime/src/apple_seg").glob("*.py"))]
    return {str(path.relative_to(REPO)): sha(path) for path in files}


def center_crop_box(width, height, edge):
    if edge < 1 or min(width, height) < edge:
        raise ValueError("Image must contain the complete requested centre crop")
    left, top = (width - edge) // 2, (height - edge) // 2
    return left, top, left + edge, top + edge


def latency_summary(values):
    values = sorted(float(value) for value in values)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Positive finite frame latencies are required")
    position = .95 * (len(values) - 1)
    low, high = math.floor(position), math.ceil(position)
    mean = statistics.mean(values)
    return {"n_source_frames": len(values), "mean_ms": mean,
            "median_ms": statistics.median(values),
            "p95_ms": values[low] + (values[high] - values[low]) * (position - low),
            "min_ms": values[0], "max_ms": values[-1], "source_frames_per_second": 1000 / mean}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/strong_baseline_4060.json")
    parser.add_argument("--samples", type=Path, default=REPO / "configs/premask_audit_samples_20261102.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=tuple(ORIGINAL_SHA), default=list(ORIGINAL_SHA))
    parser.add_argument("--policies", nargs="+", choices=tuple(POLICIES), default=list(POLICIES))
    parser.add_argument("--precision", choices=("fp16", "fp32", "bf16"), default="fp16")
    parser.add_argument("--roi-chunk", type=int, default=64)
    parser.add_argument("--short-side", type=int, default=1024)
    parser.add_argument("--max-size", type=int, default=None,
                        help="Default 1280 at short-side 1024, otherwise equal to short-side")
    parser.add_argument("--crop-edge", type=int, default=None)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--diagnostic-count", type=int, default=3)
    return parser


def validate_args(args):
    if args.max_size is None:
        args.max_size = 1280 if args.short_side == 1024 else args.short_side
    if min(args.count, args.rounds, args.roi_chunk, args.short_side, args.max_size) < 1:
        raise ValueError("Count, rounds, chunk and sizes must be positive")
    if args.max_size < args.short_side or args.warmup < 0 or not 0 <= args.diagnostic_count <= args.count:
        raise ValueError("Invalid resize, warmup or diagnostic count")
    if args.crop_edge is not None and args.crop_edge < 1:
        raise ValueError("crop-edge must be positive")
    if len(set(args.models)) != len(args.models) or len(set(args.policies)) != len(args.policies):
        raise ValueError("Models and policies may not be repeated")
    return args


def run(args):
    args = validate_args(args)
    cfgpath, sample_path, out = args.config.resolve(), args.samples.resolve(), args.output.resolve()
    if any(path.drive.upper() != "F:" for path in (cfgpath, sample_path, out)):
        raise ValueError("Config, manifest and output must remain on F:")
    cfg = json.loads(cfgpath.read_text(encoding="utf-8-sig"))
    manifest = json.loads(sample_path.read_text(encoding="utf-8-sig"))
    if len(manifest["samples"]) < args.count:
        raise ValueError("Requested count exceeds the fixed manifest")
    selected = manifest["samples"][:args.count]
    if len({row["image_id"] for row in selected}) != args.count:
        raise ValueError("Duplicate sample image identity")
    root = (cfgpath.parent / cfg["path_base"]).resolve()
    paths = {key: (root / value).resolve() for key, value in cfg["paths"].items()}
    for name in args.models:
        if cfg[name + "_checkpoint_sha256"] != ORIGINAL_SHA[name]:
            raise ValueError("Config is not bound to the original " + name + " weights")
    out.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    status = {"schema": "original_strong_premask_latency_v1", "phase": "initializing",
              "created_utc": now(), "config_sha256": sha(cfgpath), "sample_manifest_sha256": sha(sample_path),
              "source_sha256": sources, "precision": args.precision, "roi_chunk_size": args.roi_chunk,
              "pre_nms_threshold": None, "final_threshold": .75, "skip_dead_amodal": False,
              "capture_candidates": False, "short_side": args.short_side, "max_size": args.max_size,
              "crop_edge": args.crop_edge, "source_image_count": args.count, "rounds": args.rounds,
              "warmup_per_case": args.warmup, "diagnostic_images_per_case": args.diagnostic_count,
              "smoke_only": args.count < 8 or args.rounds < 2 or args.warmup < 3,
              "model_order": args.models, "policy_order": args.policies,
              "model_residency": "one original model at a time; loaded once for all its policies",
              "spatial_scope": "fixed centre crop speed simulation; not D405 accuracy" if args.crop_edge else "full image resized",
              "primary_timer": "CUDA synchronized only at frame start/end; predecoded RGB through native selected V/A binary crops",
              "diagnostic_timer": "Separate pass; synchronized stages include extra synchronization overhead",
              "excluded": ["disk RGB decode", "checkpoint loading and integrity checks", "GT", "plotting", "camera and robot I/O"],
              "sample_selection_uses_GT_offline": manifest.get("selection_uses_GT_offline_only", False),
              "GT_used_for_inference": False, "accuracy_claim": None, "models": {}, "cases": {}, "samples": []}
    torch = model = None
    write(out / "benchmark_status.json", status)
    try:
        sys.path[:0] = [str(REPO), str(REPO / "strong_runtime/src")]
        import torch
        from PIL import Image
        from apple_seg.od_data import legacy_rgb_tensor
        from apple_seg.od_training import load_checkpoint
        from apple_seg.od_model import build_od_amodal_detector
        from apple_seg.od_inference import _strict_restore
        from apple_seg.od_evaluate import _validate_prediction
        from apple_seg.od_roi import paste_roi_mask
        from deployment.pre_mask import PreMaskDeploymentAdapter
        from deployment.candidate_policy import policy_description
        torch.set_num_threads(int(cfg["cpu_threads"]))
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU or weaker-model fallback")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unsupported on the selected device")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
        status["runtime"] = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                             "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                             "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                             "tf32_cudnn": torch.backends.cudnn.allow_tf32, "cpu_threads": torch.get_num_threads()}
        rgbs = []
        for sample in selected:
            name = sample["image_id"]
            if Path(name).name != name:
                raise ValueError("Manifest image_id must be a filename")
            path = (paths["papple_root"] / "images/val" / name).resolve()
            if path.drive.upper() != "F:":
                raise ValueError("RGB must be on F:")
            with Image.open(path) as source:
                rgb = source.convert("RGB")
            if args.crop_edge:
                center_crop_box(rgb.width, rgb.height, args.crop_edge)
            rgbs.append(rgb)
            status["samples"].append({"image_id": name, "manifest_index": sample["image_index"],
                                      "RGB_path": str(path), "RGB_sha256": sha(path), "native_hw": [rgb.height, rgb.width]})
        for name in args.models:
            checkpoint = paths[name + "_checkpoint"]
            if checkpoint.drive.upper() != "F:" or sha(checkpoint) != ORIGINAL_SHA[name]:
                raise ValueError("Original checkpoint bytes differ: " + name)
            print("PREMASK_BENCH loading original " + name, flush=True)
            payload = load_checkpoint(checkpoint)
            model = build_od_amodal_detector(weights=None, **payload["constructor_config"])
            restored = _strict_restore(model, payload["model"])
            constructor = deepcopy(payload["constructor_config"])
            if model.constructor_config != constructor:
                raise ValueError("Constructor changed during strict restoration")
            del payload
            model.requires_grad_(False).eval().to("cuda")
            versions = {key: value._version for key, value in model.state_dict(keep_vars=True).items()}
            original_chunk = model.roi_heads.roi_chunk_size
            status["models"][name] = {"checkpoint": str(checkpoint), "checkpoint_sha256": ORIGINAL_SHA[name],
                                      "constructor_config": constructor, "strict_restore": restored}
            for policy_name in args.policies:
                case_name = name + "_" + policy_name
                policy = POLICIES[policy_name]
                case = {"phase": "running", "model": name, "policy": policy_description(**policy),
                        "frames": [], "diagnostic_frames": []}
                status["cases"][case_name] = case
                status.update(phase="measuring", current_case=case_name)
                write(out / "benchmark_status.json", status)
                with PreMaskDeploymentAdapter(model, candidate_policy=policy, capture_candidates=False,
                        profile_stages=False, pre_nms_threshold=None, roi_chunk_size=args.roi_chunk,
                        skip_dead_amodal=False, roi_precision="autocast") as adapter, torch.inference_mode():
                    def infer(index, diagnostic=False):
                        def mark():
                            if diagnostic:
                                torch.cuda.synchronize()
                            return time.perf_counter()
                        rgb = rgbs[index]
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        window = center_crop_box(rgb.width, rgb.height, args.crop_edge) if args.crop_edge else None
                        current = rgb.crop(window) if window else rgb
                        pixels = legacy_rgb_tensor(current, args.short_side, args.max_size)
                        native_hw, input_hw = (current.height, current.width), tuple(pixels.shape[-2:])
                        t1 = mark()
                        inputs = pixels[None].to("cuda")
                        mask = torch.ones((1, *input_hw), dtype=torch.bool, device="cuda")
                        t2 = mark()
                        with torch.autocast("cuda", enabled=args.precision != "fp32", dtype=dtype):
                            raw = adapter.predict_views(inputs, mask)
                        t3 = mark()
                        prediction = _validate_prediction(raw, input_hw, native_hw, mask_chunk_size=8)
                        t4 = mark()
                        keep = [i for i, score in enumerate(prediction["scores"]) if score >= .75]
                        pasted = [paste_roi_mask(prediction[branch][i], prediction["boxes"][i], native_hw,
                                                 mask_threshold=.5) for i in keep for branch in ("visible", "amodal")]
                        torch.cuda.synchronize()
                        t5 = time.perf_counter()
                        stats = deepcopy(adapter.last_stats)
                        if len(stats.get("pre_mask", [])) != 1 or adapter.traces:
                            raise RuntimeError("Unexpected view count or captured candidate tensors")
                        row = {"image_id": selected[index]["image_id"], "diagnostic": diagnostic,
                               "end_to_end_ms": (t5 - t0) * 1000, "input_hw": list(input_hw),
                               "native_output_hw": list(native_hw), "crop_box_in_original": list(window) if window else None,
                               "selected_count": len(keep), "native_VA_binary_crop_count": len(pasted),
                               "decoded_candidates": stats["pre_mask"][0]["decoded_candidates"],
                               "mask_candidates": stats["pre_mask"][0]["mask_candidates"],
                               "adapter_stats": stats, "endpoint_box_coordinates": "native crop XYXY" if window else "native image XYXY",
                               "endpoint_boxes": prediction["boxes"][keep].tolist(),
                               "endpoint_scores": [prediction["scores"][i] for i in keep]}
                        if diagnostic:
                            row["synchronized_ms"] = {"preprocess_and_crop": (t1 - t0) * 1000,
                                "H2D_and_pixel_mask": (t2 - t1) * 1000, "model_call": (t3 - t2) * 1000,
                                "native_prediction_validation": (t4 - t3) * 1000,
                                "final_filter_and_native_VA_paste": (t5 - t4) * 1000}
                            row["roi_synchronized_ms"] = stats["pre_mask"][0]["synchronized_ms"]
                        return row
                    for i in range(args.warmup):
                        infer(i % args.count)
                    torch.cuda.reset_peak_memory_stats()
                    for round_index in range(args.rounds):
                        for i in range(args.count):
                            row = infer(i)
                            row["round"] = round_index + 1
                            case["frames"].append(row)
                            status["completed_current_case"] = len(case["frames"])
                            write(out / "benchmark_status.json", status)
                            print(f"PREMASK_BENCH {case_name} {len(case['frames'])}/{args.count*args.rounds} {row['end_to_end_ms']:.2f}ms mask_N={row['mask_candidates']}", flush=True)
                    case["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    case["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
                    adapter.profile_stages = True
                    for i in range(args.diagnostic_count):
                        case["diagnostic_frames"].append(infer(i, diagnostic=True))
                case.update(phase="complete", summary=latency_summary([r["end_to_end_ms"] for r in case["frames"]]),
                    mask_candidates={"mean": statistics.mean(r["mask_candidates"] for r in case["frames"]),
                                     "min": min(r["mask_candidates"] for r in case["frames"]),
                                     "max": max(r["mask_candidates"] for r in case["frames"])})
                if case["diagnostic_frames"]:
                    for group in ("synchronized_ms", "roi_synchronized_ms"):
                        case[group + "_mean"] = {key: statistics.mean(r[group][key] for r in case["diagnostic_frames"])
                                                  for key in case["diagnostic_frames"][0][group]}
                write(out / (case_name + "_timing.json"), case)
                write(out / "benchmark_status.json", status)
            state = model.state_dict(keep_vars=True)
            if not all(state[key]._version == version for key, version in versions.items()):
                raise RuntimeError("Checkpoint tensor versions changed during inference")
            if model.roi_heads.roi_chunk_size != original_chunk or model.constructor_config != constructor:
                raise RuntimeError("Adapter did not restore original model configuration")
            if any(key in model.roi_heads.__dict__ for key in
                   ("forward", "_strong_fast_adapter_active", "_roi_predictions", "_decoded_boxes", "_final_nms", "_mask_predictions")):
                raise RuntimeError("Temporary ROI method overrides were not restored")
            if sha(checkpoint) != ORIGINAL_SHA[name]:
                raise RuntimeError("Checkpoint file changed during benchmark")
            status["models"][name].update(tensor_versions_unchanged=True, checkpoint_before_after_identical=True,
                                          temporary_adapter_restored=True)
            del state, adapter, model
            model = None
            gc.collect()
            torch.cuda.empty_cache()
        if source_hashes() != sources or sha(cfgpath) != status["config_sha256"] or sha(sample_path) != status["sample_manifest_sha256"]:
            raise RuntimeError("Source/config/manifest changed during benchmark")
        if any(sha(row["RGB_path"]) != row["RGB_sha256"] for row in status["samples"]):
            raise RuntimeError("Input RGB changed during benchmark")
        status.update(phase="complete", completed_utc=now(), current_case=None,
                      source_config_manifest_before_after_identical=True, RGB_before_after_identical=True)
        write(out / "benchmark_results.json", status)
        write(out / "benchmark_status.json", status)
        print("PREMASK_BENCH_COMPLETE " + json.dumps({k: v["summary"] for k, v in status["cases"].items()}), flush=True)
        return 0
    except BaseException as error:
        status.update(phase="failed", error=str(error), traceback=traceback.format_exc())
        if status.get("current_case") in status["cases"]:
            status["cases"][status["current_case"]].update(phase="failed", error=str(error))
        write(out / "benchmark_status.json", status)
        raise
    finally:
        model = None
        gc.collect()
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
