"""Find high-confidence round-mask training detections absent from GT.

These boxes are candidate missing annotations, not GT edits. Training may use
them only as negative-loss ignores. The script writes evidence and never
changes the annotation files.
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
from apple_seg.od_model import build_od_amodal_detector, _mask_boxes  # noqa: E402


def fill_ratio(mask):
    binary = mask[0].numpy() > .5
    rows, columns = np.where(binary)
    if not len(rows):
        return 0., 0.
    area = float(len(rows))
    box_area = float((rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1))
    return area, area / max(box_area, 1.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--short-side", type=int, default=1024)
    parser.add_argument("--max-size", type=int, default=1280)
    parser.add_argument("--score", type=float, default=.70)
    parser.add_argument("--max-gt-iou", type=float, default=.10)
    parser.add_argument("--min-fill", type=float, default=.55)
    parser.add_argument("--min-mask-area", type=float, default=40.)
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_od_amodal_detector(weights=None, **payload["constructor_config"])
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    dataset = ODInputDataset(PAppleDataset(ROOT / "assets/papple", "train", 32),
                             short_sides=(args.short_side,), max_size=args.max_size)
    indices = crop_tta.subset_indices(len(dataset), args.subset_size)
    records, images_with, total_candidates = {}, 0, 0
    started = time.time()
    with torch.inference_mode():
        for step, index in enumerate(indices):
            item = dataset[index]
            output = model(item["pixel_values"].unsqueeze(0).to(device))[0]
            original_h, original_w = item["target"]["original_hw"]
            input_h, input_w = output["image_hw"]
            boxes = output["boxes"].cpu().numpy() * np.asarray(
                [original_w / input_w, original_h / input_h] * 2)
            scores = output["scores"].cpu().numpy()
            gt = _mask_boxes(item["target"]["amodal_masks"]).numpy()
            overlaps = flip_tta.metrics.box_iou(boxes, gt) if len(gt) else np.zeros((len(boxes), 0))
            best = overlaps.max(axis=1) if overlaps.size else np.zeros(len(boxes))
            visible_masks = output["roi_visible_masks"].cpu()
            chosen = []
            for row in np.where((scores >= args.score) & (best < args.max_gt_iou))[0]:
                area, fill = fill_ratio(visible_masks[row])
                if area >= args.min_mask_area and fill >= args.min_fill:
                    chosen.append({"box": boxes[row].tolist(), "score": float(scores[row]),
                                   "mask_area": area, "fill": float(fill),
                                   "best_gt_iou": float(best[row])})
            if chosen:
                image_id = item["target"]["source"]["image_id"]
                records[image_id] = chosen
                images_with += 1
                total_candidates += len(chosen)
            if (step + 1) % args.progress_every == 0 or step + 1 == len(indices):
                print(f"IGNORE {step + 1}/{len(indices)} images_with={images_with} "
                      f"candidates={total_candidates} elapsed={(time.time() - started) / 60:.1f} min",
                      flush=True)
    report = {"schema": "apple_train_ignore_candidate_boxes_v1",
              "checkpoint": str(Path(args.checkpoint).resolve()),
              "checkpoint_sha256": flip_tta.sha256(args.checkpoint),
              "images_examined": len(indices), "subset": len(indices) != len(dataset),
              "criteria": {"score": args.score, "max_gt_iou": args.max_gt_iou,
                           "min_fill": args.min_fill, "min_mask_area": args.min_mask_area,
                           "short_side": args.short_side, "max_size": args.max_size},
              "images_with_candidates": images_with, "candidate_boxes": total_candidates,
              "records": records, "annotation_files_modified": False}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out} images_with={images_with} boxes={total_candidates}")


if __name__ == "__main__":
    main()
