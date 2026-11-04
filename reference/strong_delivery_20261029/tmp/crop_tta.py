"""Overlapping crop inference for small or heavily occluded apples.

Crops are taken from the same deterministic 1024-pixel VAL view, resized back
to 1024 so apples occupy a larger fraction of the detector input, then mapped
to original-frame coordinates. The full view is retained as a separate layer.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import flip_tta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apple_seg.data import PAppleDataset  # noqa: E402
from apple_seg.od_data import ODInputDataset  # noqa: E402
from apple_seg.od_model import build_od_amodal_detector  # noqa: E402


def tile_windows(height, width, fraction=.625):
    """Two-by-two windows covering the complete view, with at least 20% overlap."""
    if not .5 <= fraction < 1:
        raise ValueError("fraction must be in [0.5,1) for complete 2x2 coverage")
    side_h = max(1, int(round(height * fraction)))
    side_w = max(1, int(round(width * fraction)))
    ys = sorted({0, height - side_h})
    xs = sorted({0, width - side_w})
    return [(x1, y1, x1 + side_w, y1 + side_h) for y1 in ys for x1 in xs]


def map_crop_boxes(boxes, window, input_hw, output_hw, original_hw):
    x1, y1, x2, y2 = window
    input_h, input_w = input_hw
    output_h, output_w = output_hw
    original_h, original_w = original_hw
    array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    array[:, [0, 2]] = array[:, [0, 2]] * (x2 - x1) / output_w + x1
    array[:, [1, 3]] = array[:, [1, 3]] * (y2 - y1) / output_h + y1
    array *= np.asarray([original_w / input_w, original_h / input_h] * 2)
    return array


def subset_indices(count, size):
    if size is None or size >= count:
        return list(range(count))
    return sorted({int(round(value)) for value in np.linspace(0, count - 1, size)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--short-side", type=int, default=1024)
    parser.add_argument("--max-size", type=int, default=1280)
    parser.add_argument("--crop-side", type=int, default=1024)
    parser.add_argument("--fraction", type=float, default=.625)
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_od_amodal_detector(weights=None, **payload["constructor_config"])
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    dataset = ODInputDataset(PAppleDataset(ROOT / "assets/papple", "val", 32),
                             short_sides=(args.short_side,), max_size=args.max_size)
    base = flip_tta.load_records(args.base_predictions)
    if len(base) != len(dataset):
        raise ValueError("Base predictions and dataset differ in length")
    indices = subset_indices(len(dataset), args.subset_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_subset = [base[index] for index in indices]
    (out_dir / "base_subset.json").write_text(json.dumps(full_subset, indent=1), encoding="utf-8")
    view_records = None
    started = time.time()
    with torch.inference_mode():
        for step, index in enumerate(indices):
            item = dataset[index]
            pixels = item["pixel_values"]
            input_h, input_w = pixels.shape[-2:]
            original_hw = item["target"]["original_hw"]
            windows = tile_windows(input_h, input_w, args.fraction)
            if view_records is None:
                view_records = [[] for _ in windows]
            for view_index, window in enumerate(windows):
                x1, y1, x2, y2 = window
                crop = pixels[:, y1:y2, x1:x2]
                resized = F.interpolate(crop[None], size=(args.crop_side, args.crop_side),
                                        mode="bilinear", align_corners=False, antialias=False)[0]
                output = model(resized[None].to(device))[0]
                output_h, output_w = output["image_hw"]
                boxes = map_crop_boxes(output["boxes"].cpu().numpy(), window,
                                       (input_h, input_w), (output_h, output_w), original_hw)
                view_records[view_index].append({
                    "image_id": base[index]["image_id"], "gt": base[index]["gt"],
                    "pred": {"boxes": boxes.tolist(), "scores": output["scores"].cpu().numpy().tolist()}})
            if (step + 1) % args.progress_every == 0 or step + 1 == len(indices):
                print(f"CROP {step + 1}/{len(indices)} windows={len(windows)} "
                      f"elapsed={(time.time() - started) / 60:.1f} min", flush=True)
    for view_index, records in enumerate(view_records):
        path = out_dir / f"view_{view_index}.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        print("wrote", path, flush=True)
    metadata = {"schema": "apple_crop_tta_predictions_v1", "images": len(indices),
                "subset_evenly_spaced": len(indices) != len(dataset), "all_image_indices": indices,
                "short_side": args.short_side, "max_size": args.max_size,
                "crop_side": args.crop_side, "fraction": args.fraction,
                "windows_first_image": tile_windows(args.short_side, args.short_side,
                                                    args.fraction),
                "checkpoint_sha256": flip_tta.sha256(args.checkpoint),
                "gt_used_for_inference": False}
    (out_dir / "prediction_metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    print("wrote", out_dir / "prediction_metadata.json")


if __name__ == "__main__":
    main()
