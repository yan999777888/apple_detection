"""Capture real pre-NMS candidates for box-voting research.

The model is unchanged; only the private final-NMS callable is replaced for
this diagnostic so every post-classifier candidate survives. No GT enters
inference.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import crop_tta
import flip_tta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apple_seg.data import PAppleDataset  # noqa: E402
from apple_seg.od_data import ODInputDataset  # noqa: E402
from apple_seg.od_model import build_od_amodal_detector  # noqa: E402


def keep_all(self, masks, boxes, scores, image_shape):
    return torch.arange(len(scores), device=scores.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--short-side", type=int, default=1024)
    parser.add_argument("--max-size", type=int, default=1280)
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--candidate-limit", type=int, default=600)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_od_amodal_detector(weights=None, **payload["constructor_config"])
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    model.roi_heads.detections_per_img = args.candidate_limit
    model.roi_heads._final_nms = keep_all.__get__(model.roi_heads, model.roi_heads.__class__)
    dataset = ODInputDataset(PAppleDataset(ROOT / "assets/papple", "val", 32),
                             short_sides=(args.short_side,), max_size=args.max_size)
    base = flip_tta.load_records(args.base_predictions)
    indices = crop_tta.subset_indices(len(dataset), args.subset_size)
    records = []
    started = time.time()
    with torch.inference_mode():
        for step, index in enumerate(indices):
            item = dataset[index]
            output = model(item["pixel_values"].unsqueeze(0).to(device))[0]
            original_h, original_w = item["target"]["original_hw"]
            input_h, input_w = output["image_hw"]
            scale = np.asarray([original_w / input_w, original_h / input_h] * 2)
            boxes = output["boxes"].cpu().numpy() * scale
            scores = output["scores"].cpu().numpy()
            records.append({"image_id": base[index]["image_id"], "gt": base[index]["gt"],
                            "pred": {"boxes": boxes.tolist(), "scores": scores.tolist()}})
            if (step + 1) % args.progress_every == 0 or step + 1 == len(indices):
                print(f"PRE_NMS {step + 1}/{len(indices)} candidates="
                      f"{sum(len(row['pred']['scores']) for row in records)} "
                      f"elapsed={(time.time() - started) / 60:.1f} min", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(records), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
