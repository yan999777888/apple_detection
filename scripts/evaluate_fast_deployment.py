"""Audit full-VAL strong-model deployment candidates without changing training.

Each model stage exports original or original+hflip boxes/scores after the configured model
score cutoff, stable top-k and original dual-mask NMS. Fusion uses the frozen
four-view implementation. A 512-pixel run means resized FULL frames, never a
GT-centred crop or an independently validated local tracker.
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
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "strong_runtime"
sys.path[:0] = [str(REPO), str(RUNTIME / "src"), str(RUNTIME / "tmp")]
IMAGES, GT, EMPTY = 814, 3449, 13


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_inventory():
    files = [Path(__file__), REPO / "deployment/strong_fast.py",
             RUNTIME / "tmp/flip_tta.py", RUNTIME / "reference/local_amodal/scripts/metrics.py",
             *sorted((RUNTIME / "src/apple_seg").glob("*.py"))]
    return {str(p.relative_to(REPO)): sha(p) for p in files}


def final_integrity(metadata, sources):
    result = {"source_before_after_identical": source_inventory() == sources,
              "config_before_after_identical": sha(metadata["config"]) == metadata["config_sha256"]}
    if not all(result.values()):
        raise RuntimeError("Deployment source or original config changed before completion")
    return result


def runtime_numerics(metadata):
    names = ("gpu", "torch", "cuda_matmul_allow_tf32", "cudnn_allow_tf32")
    if any(name not in metadata for name in names):
        raise ValueError("Runtime numerical flags are missing")
    return {name: metadata[name] for name in names}


def policy(pre_nms_threshold, fixed_only=False):
    """Keep fixed-point evidence distinct from a truncated deployment PR curve."""
    if pre_nms_threshold is not None and (not math.isfinite(pre_nms_threshold) or not 0 <= pre_nms_threshold <= .75):
        raise ValueError("Pre-NMS threshold must be None or in [0,.75]; .75 fixed point must remain measurable")
    fixed_only = fixed_only or (pre_nms_threshold is not None and pre_nms_threshold >= .75)
    return {
        "fixed_only": fixed_only,
        "fixed_confidence": .75,
        "iou_threshold": .5,
        "P_at_R90_target": .90,
        "curve_scope": ("not computed: fixed-point-only score cutoff" if fixed_only else
                        "original model-returned candidate PR curve" if pre_nms_threshold is None else
                        "deployment-truncated candidate PR curve; low-score candidates unavailable"),
        "safe_exactness": "A label requiring FP32 fixed-point output comparison; not asserted by this evaluator",
        "threshold_selection_split": "VAL; not an independent test-set claim",
        "gt_used_for_inference": False,
        "mask_metrics_claimed": False,
        "record_candidates": "All returned post-model-score-filter, stable-top-k, dual-mask-NMS detections; no evaluator score filtering",
    }


def summarize(records, metrics, evaluation_policy):
    result = {"at_confidence_0_75": metrics.evaluate_at_threshold(records, .75, .5),
              "evaluation_policy": evaluation_policy}
    if evaluation_policy["fixed_only"]:
        result.update(at_recall90=None, balanced=None, candidate_thresholds=None)
        return result, None
    calibrated = metrics.calibrate_threshold(records, .5, return_curve=True)
    eligible = [row for row in calibrated["curve"] if row["recall"] >= .90]
    result.update(at_recall90=max(eligible, key=lambda r: (r["precision"], r["recall"], r["f1"], r["confidence"])) if eligible else None,
                  balanced=calibrated["summary"], candidate_thresholds=calibrated["candidate_thresholds"],
                  max_attainable_recall=max(row["recall"] for row in calibrated["curve"]))
    return result, calibrated


def audit_records(records, contract):
    if len(records) != IMAGES:
        raise ValueError("Only complete 814-frame VAL records can produce a completed summary")
    for record, frozen in zip(records, contract["images"], strict=True):
        expected = [row["box"] for row in contract["truth"][frozen["image_id"]]]
        if record["image_id"] != frozen["image_id"] or record["gt"] != {"boxes": expected}:
            raise ValueError("Image identity/order or genuine original GT differs")
        if len(record["pred"]["boxes"]) != len(record["pred"]["scores"]):
            raise ValueError("Prediction box/score counts differ")
    if sum(len(row["gt"]["boxes"]) for row in records) != GT or sum(not row["gt"]["boxes"] for row in records) != EMPTY:
        raise ValueError("Full-VAL GT/empty-frame inventory differs")


def native_prediction(raw, input_hw, original_hw, flipped, validate):
    """Return exclusive-XYXY boxes and optional masks in the original frame."""
    prediction = validate([raw], input_hw, original_hw, mask_chunk_size=8)
    if flipped:
        old = prediction["boxes"].clone()
        prediction["boxes"][:, 0] = original_hw[1] - old[:, 2]
        prediction["boxes"][:, 2] = original_hw[1] - old[:, 0]
        for branch in ("visible", "amodal"):
            prediction[branch] = prediction[branch].flip(-1).contiguous()
    return prediction


def precision_name(value):
    aliases = {"fp32": "fp32", "float32": "fp32", "fp16": "fp16", "float16": "fp16", "bf16": "bf16", "bfloat16": "bf16"}
    try:
        return aliases[value]
    except KeyError as error:
        raise argparse.ArgumentTypeError("Use fp32, fp16 or bf16") from error


def optional_probability(value):
    if value.lower() in ("none", "original"):
        return None
    try:
        result = float(value)
        policy(result)
        return result
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Original strong_baseline_4060 config")
    p.add_argument("--stage", choices=("base", "small", "fuse"), required=True)
    p.add_argument("--views", choices=("original", "both"), default="both",
                   help="Both views required for four-view fusion; original-only avoids a redundant 512 full-frame flip")
    p.add_argument("--output", type=Path, required=True, help="Fresh experiment root on F:; model stages have separate subdirectories")
    p.add_argument("--precision", type=precision_name, default="fp16")
    p.add_argument("--pre-nms-threshold", type=optional_probability, default=.35, metavar="none|0.35|0.75")
    p.add_argument("--roi-chunk-size", type=int, default=32)
    p.add_argument("--roi-precision", choices=("fp32", "autocast"), default="autocast")
    p.add_argument("--skip-dead-amodal", action=argparse.BooleanOptionalAction, default=False,
                   help="Optional only when adapter proves A refinement disabled; original strong Base/Small keep refinement active")
    p.add_argument("--paired-backbone-batch2", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--short-side", type=int, default=1024)
    p.add_argument("--max-size", type=int, default=1280)
    p.add_argument("--fixed-only", action="store_true")
    p.add_argument("--save-roi-masks", action="store_true", help="Optional native-oriented ROI V/A probability archives; not mask metrics")
    p.add_argument("--progress-every", type=int, default=25)
    return p


def deployment_spec(args):
    if min(args.short_side, args.max_size, args.roi_chunk_size, args.progress_every) < 1 or args.max_size < args.short_side:
        raise ValueError("Sizes/chunks/progress must be positive and max-size >= short-side")
    if args.views == "original" and args.paired_backbone_batch2:
        raise ValueError("Original-only evaluation is incompatible with paired backbone batch2")
    if args.views != "both" and args.stage == "fuse":
        raise ValueError("Four-view fusion requires --views both")
    return {"precision": args.precision, "pre_nms_threshold": args.pre_nms_threshold,
            "views": args.views,
            "roi_chunk_size": args.roi_chunk_size, "roi_precision": args.roi_precision,
            "skip_dead_amodal": args.skip_dead_amodal, "paired_backbone_batch2": args.paired_backbone_batch2,
            "short_side": args.short_side, "max_size": args.max_size,
            "evaluation_policy": policy(args.pre_nms_threshold, args.fixed_only),
            "image_scope": "full original frame resized; never GT-centred crops",
            "local_512_accuracy_claimed": False,
            "spatial_scope_note": ("512 FULL-frame downscale experiment; not local tracking/cropping P@R90" if args.short_side == 512 else
                                   "full-frame detection; candidate eligibility unchanged by distance")}


def run_fusion(args, cfg, contract, sources, out, metadata):
    import flip_tta
    layers, metas, inputs = [], [], {}
    for name in ("base", "small"):
        directory = out / name
        meta_path = directory / "prediction_metadata.json"
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        if m["deployment"].get("views") != "both":
            raise ValueError("Original-only predictions cannot participate in four-view fusion")
        if m["phase"] != "complete" or m["source_sha256"] != sources or m["config_sha256"] != metadata["config_sha256"]:
            raise ValueError("Fusion requires complete, source-identical model stages and same original config")
        if m["deployment"] != metadata["deployment"] or m["checkpoint_sha256"] != cfg[name + "_checkpoint_sha256"]:
            raise ValueError("Fusion model deployment spec or original strong checkpoint differs")
        if (m.get("model") != name or m.get("from_fresh_weight_inference") is not True
                or m.get("original_strong_models_only") is not True or m.get("training_performed") is not False
                or m.get("config_before_after_identical") is not True or m.get("source_before_after_identical") is not True):
            raise ValueError("Fusion requires audited original-model fresh-inference lineage")
        if m["provenance_sha256"] != contract["provenance_sha256"]:
            raise ValueError("Fusion provenance differs")
        if metas and runtime_numerics(m) != runtime_numerics(metas[0]):
            raise ValueError("Base/Small runtime numerical flags differ")
        for view in ("original", "hflip"):
            path = directory / f"{view}_predictions.json"
            if sha(path) != m["predictions_sha256"][view]:
                raise ValueError("Raw model prediction bytes changed since validation")
            records = json.loads(path.read_text(encoding="utf-8"))
            audit_records(records, contract)
            layers.append(records)
            inputs[f"{name}_{view}"] = {"path": str(path), "sha256": sha(path)}
        inputs[name + "_metadata"] = {"path": str(meta_path), "sha256": sha(meta_path)}
        metas.append(m)
    fused = flip_tta.fuse_records(layers, **cfg["fusion"])
    audit_records(fused, contract)
    result, curve = summarize(fused, flip_tta.metrics, metadata["deployment"]["evaluation_policy"])
    write(out / "fused_predictions.json", fused)
    if curve is not None:
        write(out / "fusion_calibration_curve.json", curve)
    metadata.update(phase="complete", images=IMAGES, GT=GT, empty=EMPTY,
                    metrics=result, fusion=cfg["fusion"], mask_fusion=False, inputs=inputs,
                    from_fresh_weight_inference=all(m["from_fresh_weight_inference"] for m in metas),
                    model_runtime_numerics=runtime_numerics(metas[0]),
                    predictions_sha256=sha(out / "fused_predictions.json"),
                    completed_utc=now(), **final_integrity(metadata, sources))
    write(out / "fusion_summary.json", metadata)
    return metadata


def run_model(args, cfg, paths, contract, sources, out, metadata):
    import numpy as np
    import torch
    import flip_tta
    from apple_seg.evaluate import _audit_target, _audit_rgb
    from apple_seg.data import PAppleDataset
    from apple_seg.od_data import ODInputDataset
    from apple_seg.od_training import load_checkpoint
    from apple_seg.od_model import build_od_amodal_detector
    from apple_seg.od_inference import _strict_restore
    from apple_seg.od_evaluate import _validate_prediction
    from deployment.strong_fast import FastDeploymentAdapter

    if not torch.cuda.is_available():
        raise RuntimeError("Full deployment inference requires CUDA; this script never falls back to a different model")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable on the selected GPU")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.set_num_interop_threads(1)
    checkpoint = paths[args.stage + "_checkpoint"]
    if sha(checkpoint) != cfg[args.stage + "_checkpoint_sha256"]:
        raise ValueError("Only the pinned original Base/Small checkpoints are supported")
    metadata.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
    payload = load_checkpoint(checkpoint)
    model = build_od_amodal_detector(weights=None, **payload["constructor_config"])
    metadata.update(strict_restore=_strict_restore(model, payload["model"]),
                    constructor_config=payload["constructor_config"], weights_are_ema=payload.get("weights_are_ema", False))
    if model.constructor_config != payload["constructor_config"]:
        raise ValueError("Original architecture differs after strict restoration")
    del payload
    model.requires_grad_(False).eval().to("cuda")
    metadata["model_types"] = {"model": type(model).__qualname__, "roi_heads": type(model.roi_heads).__qualname__}
    native = PAppleDataset(paths["papple_root"], "val", 32, augment=False)
    if len(native) != IMAGES or native.annotation_sha256 != contract["provenance"]["binding"]["current_annotation_sha256"]:
        raise ValueError("Actual full validation annotations differ from the frozen contract")
    dataset = ODInputDataset(native, short_sides=(args.short_side,), max_size=args.max_size)
    view_names = ("original", "hflip") if args.views == "both" else ("original",)
    records = {view: [] for view in view_names}
    timing_rows = []
    adapter = FastDeploymentAdapter(model, pre_nms_threshold=args.pre_nms_threshold,
                                    roi_chunk_size=args.roi_chunk_size, skip_dead_amodal=args.skip_dead_amodal,
                                    roi_precision=args.roi_precision)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    metadata.update(phase="inference", gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                    cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, resident_gpu_models=[args.stage],
                    annotation_sha256=native.annotation_sha256,
                    timing_scope=f"Per-image input transfer + {len(view_names)} view(s) + CPU prediction validation; excludes dataset preprocessing/GT audit/disk/fusion. Not a warmed deployment FPS benchmark.")
    write(out / "prediction_metadata.json", metadata)
    torch.cuda.reset_peak_memory_stats()
    with adapter, torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=args.precision != "fp32"):
        for index, frozen in enumerate(contract["images"]):
            item = dataset[index]
            target = item["target"]
            truth = _audit_target(target, frozen, contract["truth"][frozen["image_id"]])
            rgb_sha = _audit_rgb(Path(target["source"]["rgb_path"]), frozen)
            pixels = item["pixel_values"]
            input_hw, hw = tuple(pixels.shape[-2:]), tuple(target["original_hw"])
            torch.cuda.synchronize()
            started = time.perf_counter()
            gpu_pixels = pixels[None].to("cuda")
            mask = torch.ones((1, *input_hw), device="cuda", dtype=torch.bool)
            if args.paired_backbone_batch2:
                pair = torch.cat((gpu_pixels, gpu_pixels.flip(-1).contiguous()), 0)
                outputs = adapter.predict_views(pair, mask.expand(2, -1, -1).contiguous())
                adapter_stats = [deepcopy(adapter.last_stats)]
                if len(outputs) != 2:
                    raise RuntimeError("Paired-backbone mode must return both views without changing detector semantics")
                del pair
            else:
                outputs, adapter_stats = [], []
                view_tensors = (gpu_pixels, gpu_pixels.flip(-1).contiguous()) if args.views == "both" else (gpu_pixels,)
                for view_tensor in view_tensors:
                    current = adapter.predict_views(view_tensor, mask)
                    if len(current) != 1:
                        raise RuntimeError("Single-view mode must return one detection dictionary")
                    outputs.append(current[0])
                    adapter_stats.append(deepcopy(adapter.last_stats))
                del view_tensor, current, view_tensors
            predictions = [native_prediction(raw, input_hw, hw, view == "hflip", _validate_prediction)
                           for view, raw in zip(view_names, outputs, strict=True)]
            torch.cuda.synchronize()
            timing_rows.append({"image_id": frozen["image_id"], "views_ms": (time.perf_counter() - started) * 1000,
                                "view_names": list(view_names),
                                "input_hw": list(input_hw), "candidates": [len(p["scores"]) for p in predictions],
                                "adapter_stats": adapter_stats})
            for view, prediction in zip(view_names, predictions, strict=True):
                records[view].append({"image_id": frozen["image_id"], "gt": {"boxes": truth["boxes"]},
                                      "pred": {"boxes": prediction["boxes"].tolist(), "scores": prediction["scores"]}})
                if args.save_roi_masks:
                    directory = out / "roi_masks" / view
                    directory.mkdir(parents=True, exist_ok=True)
                    with (directory / f"{index:06d}.npz").open("xb") as stream:
                        np.savez(stream, boxes=prediction["boxes"].numpy(), scores=np.asarray(prediction["scores"], dtype=np.float32),
                                 visible=prediction["visible"].numpy(), amodal=prediction["amodal"].numpy())
            _audit_target(target, frozen, contract["truth"][frozen["image_id"]])
            if _audit_rgb(Path(target["source"]["rgb_path"]), frozen) != rgb_sha:
                raise RuntimeError("RGB changed during inference")
            if (index + 1) % args.progress_every == 0 or index + 1 == IMAGES:
                metadata.update(images=index + 1, GT=sum(len(row["gt"]["boxes"]) for row in records["original"]), updated_utc=now())
                write(out / "prediction_metadata.json", metadata)
                for view in records:
                    write(out / f"{view}_partial_predictions.json", records[view])
                write(out / "validation_timing.json", {"scope": metadata["timing_scope"], "frames": timing_rows})
                print(f"FAST_VAL {args.stage} {index + 1}/{IMAGES} {args.precision} cutoff={args.pre_nms_threshold}", flush=True)
            del item, target, truth, pixels, gpu_pixels, mask, outputs, predictions, prediction
    if source_inventory() != sources:
        raise RuntimeError("Deployment sources changed during full-VAL inference")
    view_metrics, prediction_hashes = {}, {}
    for view, layer in records.items():
        audit_records(layer, contract)
        path = out / f"{view}_predictions.json"
        write(path, layer)
        prediction_hashes[view] = sha(path)
        view_metrics[view], curve = summarize(layer, flip_tta.metrics, metadata["deployment"]["evaluation_policy"])
        if curve is not None:
            write(out / f"{view}_calibration_curve.json", curve)
    metadata.update(phase="complete", images=IMAGES, GT=GT, empty=EMPTY, updated_utc=now(), completed_utc=now(),
                    predictions_sha256=prediction_hashes, view_metrics=view_metrics,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                    roi_masks_saved=args.save_roi_masks, roi_mask_archive_convention="Native image-oriented ROI probabilities, with native XYXY support; hflip inverted for BOTH boxes and masks",
                    **final_integrity(metadata, sources))
    write(out / "prediction_metadata.json", metadata)
    del adapter, model
    gc.collect()
    torch.cuda.empty_cache()
    return metadata


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg_path = args.config.resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    root = (cfg_path.parent / cfg["path_base"]).resolve()
    paths = {key: (root / value).resolve() for key, value in cfg["paths"].items()}
    out_root = args.output.resolve()
    if out_root.drive.upper() != "F:":
        raise ValueError("Deployment files must stay on the user-requested F: drive")
    spec = deployment_spec(args)
    out = out_root if args.stage == "fuse" else out_root / args.stage
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / ("fusion_status.json" if args.stage == "fuse" else "prediction_metadata.json")
    fusion_files = ("fused_predictions.json", "fusion_summary.json", "fusion_calibration_curve.json")
    if (status_path.exists() or (args.stage != "fuse" and any(out.iterdir()))
            or (args.stage == "fuse" and any((out / name).exists() for name in fusion_files))):
        raise FileExistsError("Use a fresh stage output; completed/partial evidence is never overwritten")
    metadata = {"schema": "strong_fast_full_val_v1", "phase": "initializing", "model": args.stage,
                "created_utc": now(), "config": str(cfg_path), "config_sha256": sha(cfg_path), "deployment": spec,
                "images": 0, "GT": 0, "from_fresh_weight_inference": args.stage != "fuse",
                "original_strong_models_only": True, "training_performed": False}
    write(status_path, metadata)
    started = time.perf_counter()
    try:
        sources = source_inventory()
        from apple_seg.evaluate import load_papple_contract
        contract = load_papple_contract(paths["provenance"])
        metadata.update(source_sha256=sources, provenance_sha256=contract["provenance_sha256"])
        if args.stage == "fuse":
            result = run_fusion(args, cfg, contract, sources, out, metadata)
        else:
            result = run_model(args, cfg, paths, contract, sources, out, metadata)
        result["elapsed_seconds"] = time.perf_counter() - started
        write(status_path, result)
        print("FAST_VAL_COMPLETE " + json.dumps({"stage": args.stage, "phase": result["phase"], "images": result["images"],
                                                    "GT": result["GT"], "output": str(out)}, ensure_ascii=False), flush=True)
        return 0
    except BaseException as error:
        metadata.update(phase="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                        elapsed_seconds=time.perf_counter() - started, updated_utc=now())
        write(status_path, metadata)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
