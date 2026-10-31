"""Predict one RGB image with original strong weights; no GT or training.

Fast numerical/crop settings are deployment candidates pending complete VAL
acceptance. Outputs are always marked candidate_pending_full_VAL. --help does
not import torch, touch a checkpoint, or initialize CUDA.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA = {
    "base": "0fe257c5bfe115a77ac7deac830af479e4c2b0ad6ff69e609901b24c7980fa9b",
    "small": "e0c1f659c2509d7a01d617e12cb83c673768b440ea5b4b96cb4f763e4207e7e3",
}
ORIGINAL_FUSION = {"iou_threshold": .45, "score_mode": "mean", "box_mode": "score_weighted",
    "coverage_floor": 1., "view_weights": [.65, .65, .35, .35],
    "consensus_groups": [0, 0, 1, 1], "consensus_bonus": 1.10, "isolation_penalty": .90}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def optional_cutoff(value):
    if value.lower() == "none":
        return None
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use none, .35, .70 or .75") from error
    if not math.isfinite(result) or result not in (.35, .70, .75):
        raise argparse.ArgumentTypeError("Use none, .35, .70 or .75")
    return result


def crop_window(width, height, center, size):
    """Integer exclusive window chosen only by supplied RGB-space centre."""
    if min(width, height, size) < 1:
        raise ValueError("Image/crop dimensions must be positive")
    if center is None:
        return [0, 0, width, height]
    x, y = center
    if not all(math.isfinite(v) for v in (x, y)) or not (0 <= x < width and 0 <= y < height):
        raise ValueError("--crop-center must be inside the full RGB image")
    crop_w, crop_h = min(size, width), min(size, height)
    left = max(0, min(width - crop_w, math.floor(x - crop_w / 2)))
    top = max(0, min(height - crop_h, math.floor(y - crop_h / 2)))
    return [left, top, left + crop_w, top + crop_h]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Single RGB image, never an annotation file")
    parser.add_argument("--output", type=Path, required=True, help="New output directory on F:; never overwritten")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/strong_baseline_4060.json")
    parser.add_argument("--model", choices=("base", "small", "four-view"), default="base")
    parser.add_argument("--precision", choices=("fp16", "fp32", "bf16"), default="fp16")
    parser.add_argument("--roi-precision", choices=("autocast", "fp32"), default="autocast")
    parser.add_argument("--roi-chunk-size", type=int, default=64)
    parser.add_argument("--pre-nms-threshold", type=optional_cutoff, default=None, metavar="none|.35|.70|.75",
                        help="Default none retains original candidates; fusion cutoffs can change results")
    parser.add_argument("--confidence", type=float, default=.75, help="Final display/export score cutoff, default .75")
    parser.add_argument("--short-side", type=int, default=1024)
    parser.add_argument("--max-size", type=int, default=1280)
    parser.add_argument("--crop-center", type=float, nargs=2, metavar=("X", "Y"),
                        help="User/tracker supplied FULL-image pixel centre; no GT/automatic centre lookup")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warm calls after one model load; timings are not FPS")
    return parser


def source_inventory():
    paths = [Path(__file__), ROOT / "deployment/strong_fast.py",
             *sorted((ROOT / "strong_runtime/src/apple_seg").glob("*.py")),
             ROOT / "strong_runtime/tmp/flip_tta.py"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def export_native_masks(prediction, keep, model_name, output, offset, crop_hw, full_hw, paste, np, Image):
    directory = output / "masks" / model_name
    directory.mkdir(parents=True)
    boxes_crop = prediction["boxes"][keep].numpy().copy()
    boxes_full = boxes_crop + np.asarray([*offset, *offset], dtype=np.float32)
    # XY offsets above are [left,top,left,top], not width/height scale factors.
    scores = np.asarray([prediction["scores"][i] for i in keep], dtype=np.float32)
    visible = prediction["visible"][keep].numpy().copy()
    amodal = prediction["amodal"][keep].numpy().copy()
    archive = output / f"{model_name}_native_roi_masks.npz"
    with archive.open("xb") as stream:
        np.savez_compressed(stream, boxes_crop_xyxy=boxes_crop, boxes_full_xyxy=boxes_full,
            scores=scores, prediction_indices=np.asarray(keep, dtype=np.int64),
            visible_probabilities=visible, amodal_probabilities=amodal,
            offset_full_xy=np.asarray(offset, dtype=np.int64),
            crop_hw=np.asarray(crop_hw, dtype=np.int64), full_hw=np.asarray(full_hw, dtype=np.int64))
    detections = []
    for j, index in enumerate(keep):
        row = {"prediction_index": index, "score": float(prediction["scores"][index]),
               "box_crop_xyxy": boxes_crop[j].tolist(), "box_full_xyxy": boxes_full[j].tolist()}
        for branch in ("visible", "amodal"):
            binary, window = paste(prediction[branch][index], prediction["boxes"][index],
                                    crop_hw, mask_threshold=.5)
            array = binary.cpu().numpy()
            foreground = int(array.sum())
            native_window = list(map(int, window))
            full_window = [native_window[0] + offset[0], native_window[1] + offset[1],
                           native_window[2] + offset[0], native_window[3] + offset[1]]
            record = {"path": None, "window_crop_xyxy": native_window,
                      "window_full_xyxy": full_window, "offset_full_xy": list(offset),
                      "foreground_pixels": foreground, "sha256": None,
                      "mask_threshold": .5}
            if array.size:
                path = directory / f"det_{j + 1:03d}_{branch}.png"
                Image.fromarray(array.astype(np.uint8) * 255).save(path)
                record.update(path=path.relative_to(output).as_posix(), sha256=sha(path))
            row[branch + "_mask"] = record
        detections.append(row)
    return detections, {"path": archive.name, "sha256": sha(archive),
        "dtype": "float32", "lossless": True, "scope": "Selected single-model original-view ROI probabilities; not full-frame floating masks",
        "mask_support": "Each ROI grid uses its corresponding crop/full decoded box"}


def box_preview(rgb, detections, output, ImageDraw, ImageFont):
    image = rgb.copy()
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    face = ImageFont.truetype(str(font_path), max(13, min(26, min(image.size) // 50))) if font_path.exists() else ImageFont.load_default()
    for j, detection in enumerate(detections, 1):
        box = detection["box_full_xyxy"]
        draw.rectangle(box, outline=(20, 135, 255), width=max(2, min(image.size) // 350))
        text = f"#{j} {detection['score']:.3f}"
        bounds = draw.textbbox((0, 0), text, font=face)
        x, y = max(0, box[0]), max(0, box[1] - (bounds[3] - bounds[1]) - 8)
        draw.rectangle((x, y, min(image.width, x + bounds[2] + 8), y + bounds[3] - bounds[1] + 8), fill=(12, 23, 40))
        draw.text((x + 4, y + 3 - bounds[1]), text, font=face, fill=(255, 255, 255))
    image.save(output / "prediction_boxes.png")


def contour_preview(rgb, detections, branch, output, np, Image):
    pixels = np.array(rgb, dtype=np.uint8, copy=True)
    color = np.asarray((0, 235, 235) if branch == "visible" else (255, 70, 185), dtype=float)
    for detection in detections:
        record = detection[branch + "_mask"]
        if record["path"] is None:
            continue
        with Image.open(output / record["path"]) as source:
            mask = np.asarray(source) != 0
        x1, y1, x2, y2 = record["window_full_xyxy"]
        region = pixels[y1:y2, x1:x2]
        region[mask] = np.rint(region[mask].astype(float) * .8 + color * .2).astype(np.uint8)
        eroded = mask.copy()
        for _ in range(2):
            padded = np.pad(eroded, 1, constant_values=False)
            eroded = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
                       & padded[1:-1, :-2] & padded[1:-1, 2:])
        region[mask & ~eroded] = color.astype(np.uint8)
    Image.fromarray(pixels, "RGB").save(output / f"{branch}_contours.png")


def run(args):
    if min(args.short_side, args.max_size, args.roi_chunk_size, args.crop_size) < 1 or args.max_size < args.short_side:
        raise ValueError("Positive sizes required; max-size must be >= short-side")
    if not 0 <= args.warmup_runs <= 20:
        raise ValueError("warmup-runs must be in [0,20]")
    if not math.isfinite(args.confidence) or not 0 <= args.confidence <= 1:
        raise ValueError("confidence must be a finite probability")
    if args.pre_nms_threshold is not None and args.pre_nms_threshold > args.confidence:
        raise ValueError("pre-NMS threshold must not exceed final confidence")
    output, image_path, config_path = args.output.resolve(), args.image.resolve(), args.config.resolve()
    if output.drive.lower() != "f:":
        raise ValueError("This local deployment package requires --output on F:")
    if output.exists():
        raise FileExistsError("Use a new output directory; previous evidence is never overwritten")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if args.model == "four-view" and config["fusion"] != ORIGINAL_FUSION:
        raise ValueError("Four-view mode requires the original fixed fusion configuration")
    root = (config_path.parent / config["path_base"]).resolve()
    names = ("base", "small") if args.model == "four-view" else (args.model,)
    checkpoint_paths = {name: (root / config["paths"][name + "_checkpoint"]).resolve() for name in names}
    for name in names:
        if config[name + "_checkpoint_sha256"] != ORIGINAL_SHA[name]:
            raise ValueError("Config does not bind the original strong " + name + " checkpoint")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "strong_runtime/src"))
    sys.path.insert(0, str(ROOT / "strong_runtime/tmp"))
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch
    import flip_tta
    from apple_seg.od_data import legacy_rgb_tensor, LEGACY_RGB_PREPROCESS
    from apple_seg.od_training import load_checkpoint
    from apple_seg.od_model import build_od_amodal_detector
    from apple_seg.od_inference import _strict_restore
    from apple_seg.od_evaluate import _validate_prediction
    from apple_seg.od_roi import paste_roi_mask
    from deployment.strong_fast import FastDeploymentAdapter
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; never silently substitute another model or precision")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU/runtime does not support BF16")
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = source_inventory()
    image_sha = sha(image_path)
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
    window = crop_window(rgb.width, rgb.height, args.crop_center, args.crop_size)
    view_rgb = rgb.crop(window)
    offset = window[:2]
    crop_hw, full_hw = (view_rgb.height, view_rgb.width), (rgb.height, rgb.width)
    pixels = legacy_rgb_tensor(view_rgb, args.short_side, args.max_size)
    input_hw = tuple(pixels.shape[-2:])
    output.mkdir(parents=True)
    metadata = {"schema": "single_RGB_original_strong_fast_candidate_v1", "phase": "initializing",
        "acceptance_status": "candidate_pending_full_VAL", "training_performed": False,
        "config": str(config_path), "config_sha256": sha(config_path),
        "GT_read": False, "original_weights_modified": False, "model": args.model,
        "image": str(image_path), "image_sha256": image_sha, "full_hw": list(full_hw),
        "crop": {"enabled": args.crop_center is not None, "center_source": "supplied_user_or_tracker" if args.crop_center is not None else None,
            "requested_center_full_xy": args.crop_center, "requested_size": args.crop_size if args.crop_center is not None else None,
            "window_full_xyxy": window, "offset_full_xy": offset, "native_hw": list(crop_hw),
            "policy": "Clamp user-centred window to RGB extent; no padding and no GT centre lookup"},
        "preprocess": {**LEGACY_RGB_PREPROCESS, "short_side": args.short_side, "max_size": args.max_size,
            "input_hw": list(input_hw), "normalized_input_sha256": hashlib.sha256(pixels.numpy().tobytes()).hexdigest()},
        "deployment": {"precision": args.precision, "roi_precision": args.roi_precision,
            "roi_chunk_size": args.roi_chunk_size, "pre_nms_threshold": args.pre_nms_threshold,
            "final_confidence": args.confidence, "mask_threshold": .5, "skip_dead_amodal": False,
            "paired_backbone_batch2": args.model == "four-view"},
        "runtime": {"torch": torch.__version__, "GPU": torch.cuda.get_device_name(0),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
        "weights": {}, "source_sha256": sources,
        "timing_scope": "Separate load, explicit warmup, and one invocation including H2D/forward/CPU validation. No FPS or steady-state benchmark is claimed.",
        "fusion_mask_supported": False, "created_utc": datetime.now(timezone.utc).isoformat()}
    write(output / "metadata.json", metadata)
    rgb.save(output / "input_rgb.png")
    view_rgb.save(output / "input_view.png")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    layers = []
    try:
        for name in names:
            started = time.perf_counter()
            checkpoint = checkpoint_paths[name]
            actual_sha = sha(checkpoint)
            if actual_sha != ORIGINAL_SHA[name]:
                raise ValueError("Original checkpoint bytes differ for " + name)
            payload = load_checkpoint(checkpoint)
            model = build_od_amodal_detector(weights=None, **payload["constructor_config"])
            restore = _strict_restore(model, payload["model"])
            if model.constructor_config != payload["constructor_config"]:
                raise ValueError("Restored constructor differs from original checkpoint")
            weight_meta = {"path": str(checkpoint), "sha256": actual_sha,
                "strict_restore": restore, "constructor_config": deepcopy(payload["constructor_config"]),
                "weights_are_ema": bool(payload.get("weights_are_ema", False)), "checkpoint_load_count": 1}
            del payload
            model.requires_grad_(False).eval().to("cuda")
            torch.cuda.synchronize()
            weight_meta["load_seconds_including_SHA_build_restore_H2D"] = time.perf_counter() - started
            adapter = FastDeploymentAdapter(model, pre_nms_threshold=args.pre_nms_threshold,
                roi_chunk_size=args.roi_chunk_size, roi_precision=args.roi_precision, skip_dead_amodal=False)
            def infer():
                gpu = pixels[None].to("cuda")
                mask = torch.ones((1, *input_hw), device="cuda", dtype=torch.bool)
                if args.model == "four-view":
                    pair = torch.cat((gpu, gpu.flip(-1).contiguous()), 0)
                    return adapter.predict_views(pair, mask.expand(2, -1, -1).contiguous())
                return adapter.predict_views(gpu, mask)
            with adapter, torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=args.precision != "fp32"):
                start = time.perf_counter()
                for _ in range(args.warmup_runs):
                    warmed = infer()
                    del warmed
                torch.cuda.synchronize()
                weight_meta["warmup_runs"] = args.warmup_runs
                weight_meta["warmup_seconds"] = time.perf_counter() - start
                start = time.perf_counter()
                outputs = infer()
                predictions = [_validate_prediction([raw], input_hw, crop_hw, mask_chunk_size=8) for raw in outputs]
                torch.cuda.synchronize()
                weight_meta["one_invocation_seconds_H2D_forward_CPU_validation"] = time.perf_counter() - start
                weight_meta["adapter_stats"] = deepcopy(adapter.last_stats)
                del outputs
            if args.model == "four-view":
                if len(predictions) != 2:
                    raise RuntimeError("Paired TTA did not return both original and flip")
                for i, prediction in enumerate(predictions):
                    boxes = prediction["boxes"].clone()
                    if i == 1:
                        old = boxes.clone()
                        boxes[:, 0], boxes[:, 2] = crop_hw[1] - old[:, 2], crop_hw[1] - old[:, 0]
                    boxes += torch.tensor([*offset, *offset], dtype=boxes.dtype)
                    layer = {"image_id": image_path.name,
                             "pred": {"boxes": boxes.tolist(), "scores": prediction["scores"]}}
                    layers.append(layer)
                    write(output / f"{name}_{'original' if i == 0 else 'hflip'}_boxes.json", layer)
            else:
                if len(predictions) != 1:
                    raise RuntimeError("Original-view mode returned unexpected views")
                prediction = predictions[0]
                keep = [i for i, score in enumerate(prediction["scores"]) if score >= args.confidence]
                detections, archive = export_native_masks(prediction, keep, name, output, offset,
                    crop_hw, full_hw, paste_roi_mask, np, Image)
                result = {"acceptance_status": "candidate_pending_full_VAL", "model": name,
                    "view": "original_only", "coordinate_system": "full original RGB, exclusive XYXY",
                    "raw_post_NMS_count": len(prediction["scores"]), "final_confidence": args.confidence,
                    "detection_count": len(detections), "detections": detections, "compact_roi_masks": archive,
                    "mask_scope": "Native original-view V/A predictions; no four-view mask fusion; occluded A pixels are predictions"}
            metadata["weights"][name] = weight_meta
            write(output / "metadata.json", metadata)
            del predictions, adapter, model
            gc.collect()
            torch.cuda.empty_cache()
        if args.model == "four-view":
            boxes, scores = flip_tta.fuse_entries(layers, **config["fusion"])
            keep = np.flatnonzero(scores >= args.confidence)
            detections = [{"fusion_index": int(i), "score": float(scores[i]),
                           "box_full_xyxy": boxes[i].tolist()} for i in keep]
            result = {"acceptance_status": "candidate_pending_full_VAL", "model": "base+small",
                "views": ["base_original", "base_hflip", "small_original", "small_hflip"],
                "coordinate_system": "full original RGB, exclusive XYXY", "fusion": config["fusion"],
                "raw_fused_count": len(scores), "final_confidence": args.confidence,
                "detection_count": len(detections), "detections": detections,
                "mask_scope": "No fused masks: this mode exports only boxes and scores",
                "fusion_mask_supported": False}
        box_preview(rgb, detections, output, ImageDraw, ImageFont)
        if args.model != "four-view":
            for branch in ("visible", "amodal"):
                contour_preview(rgb, detections, branch, output, np, Image)
        write(output / "predictions.json", result)
        if sha(image_path) != image_sha or source_inventory() != sources:
            raise RuntimeError("Input RGB or source files changed during this invocation")
        metadata.update(phase="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                        detection_count=len(detections), source_before_after_identical=True,
                        input_RGB_before_after_identical=True,
                        outputs={p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != "metadata.json"})
        write(output / "metadata.json", metadata)
        print(json.dumps({"status": "candidate_pending_full_VAL", "output": str(output),
                          "detections": len(detections), "model": args.model,
                          "timing_claim": "No FPS claim; see separate load/warmup/invocation timings"}, ensure_ascii=False))
        return result
    except BaseException as error:
        metadata.update(phase="failed", error=str(error), traceback=traceback.format_exc())
        write(output / "metadata.json", metadata)
        raise


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
