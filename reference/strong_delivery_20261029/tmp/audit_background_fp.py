"""Contact sheet of the background false positives at the P@R90 operating point.

If a meaningful share of these are actually unlabelled apples, then the measured
precision is an annotation artefact rather than a detector failure, and the
number has to be reported with that caveat.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tmp"))
import flip_tta  # noqa: E402
from flip_tta import metrics  # noqa: E402


def font(size):
    for name in ("msyhbd.ttc", "msyh.ttc", "arialbd.ttf"):
        path = Path("C:/Windows/Fonts") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--max-iou", type=float, default=0.1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--panel", type=int, default=210)
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--index-list", help="Optional JSON path to also write the audit list")
    args = parser.parse_args()

    records = flip_tta.load_records(args.predictions)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    sources = {row["image_id"]: row["source"] for row in metadata["ground_truth_identity"]}

    hits = []
    for record in records:
        gt = np.asarray(record["gt"]["boxes"], dtype=np.float64).reshape(-1, 4)
        boxes = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
        keep = scores >= args.threshold
        boxes, scores = boxes[keep], scores[keep]
        if not len(boxes):
            continue
        matched = metrics.match_image(gt, boxes, scores, 0.5, args.threshold)
        matched_gt = dict(zip(matched["pred_indices"], matched["gt_indices"]))
        for position in range(len(boxes)):
            if matched_gt.get(position, -1) >= 0:
                continue
            best = float(metrics.box_iou(boxes[position:position + 1], gt)[0].max()) if len(gt) else 0.0
            if best >= args.max_iou:
                continue
            hits.append({"image_id": record["image_id"], "box": boxes[position],
                         "score": float(scores[position]), "gt": len(gt)})
    hits.sort(key=lambda row: -row["score"])
    if args.index_list:
        Path(args.index_list).write_text(json.dumps(
            [{"rank": rank + 1, "image_id": hit["image_id"], "score": round(hit["score"], 4),
              "box": [round(float(value), 1) for value in hit["box"]], "gt_in_image": hit["gt"]}
             for rank, hit in enumerate(hits)], ensure_ascii=False, indent=1), encoding="utf-8")
    total = len(hits)
    hits = hits[args.offset:args.offset + args.limit]
    print("background candidates", len(hits), "of", total, flush=True)

    panel, columns = args.panel, args.columns
    rows = (len(hits) + columns - 1) // columns
    caption = 26
    sheet = Image.new("RGB", (columns * panel, rows * (panel + caption)), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = font(15)
    for index, hit in enumerate(hits):
        source = sources[hit["image_id"]]
        image = Image.open(source["rgb_path"]).convert("RGB")
        crop_box = source.get("crop_xyxy")
        if crop_box and list(image.size) != [crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]]:
            image = image.crop(tuple(crop_box))
        x1, y1, x2, y2 = hit["box"]
        pad = max(x2 - x1, y2 - y1) * .45
        left, top = max(0, int(x1 - pad)), max(0, int(y1 - pad))
        right, bottom = min(image.width, int(x2 + pad)), min(image.height, int(y2 + pad))
        crop = image.crop((left, top, right, bottom))
        crop.thumbnail((panel - 8, panel - 8), Image.LANCZOS)
        column, row = index % columns, index // columns
        cell_x, cell_y = column * panel + (panel - crop.width) // 2, row * (panel + caption) + 4
        sheet.paste(crop, (cell_x, cell_y))
        scale = crop.width / max(right - left, 1)
        box_in_crop = [(x1 - left) * scale + cell_x, (y1 - top) * scale + cell_y,
                       (x2 - left) * scale + cell_x, (y2 - top) * scale + cell_y]
        draw.rectangle(box_in_crop, outline=(220, 30, 30), width=3)
        draw.text((column * panel + 6, row * (panel + caption) + panel + 4),
                  f"{index + 1}  s={hit['score']:.2f}  gt={hit['gt']}", font=label_font, fill=(20, 20, 20))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=92)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
