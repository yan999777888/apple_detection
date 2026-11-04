"""Render VAL error crops so the failure mode can be inspected directly.

Reads an already published raw_predictions.json (no model forward, no GPU) and
draws two montages in original image coordinates:

  * false negatives whose box already matches a GT at IoU>=0.5 but whose score
    stayed below the selected threshold, and
  * false positives that have exactly zero overlap with any GT.

It is a read-only diagnostic: nothing is written into the run directory.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
TILE = 128


def iou_matrix(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.)


def crop(image, box, context=1.6):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * context, (y2 - y1) * context
    left = max(0, int(round(cx - w / 2)))
    top = max(0, int(round(cy - h / 2)))
    right = min(image.width, int(round(cx + w / 2)))
    bottom = min(image.height, int(round(cy + h / 2)))
    if right - left < 8 or bottom - top < 8:
        return None
    return image.crop((left, top, right, bottom)).resize((TILE, TILE), Image.BICUBIC)


def collect(raw, images_root, confidence, limit, min_score):
    found = {"fn_correct_box": [], "fp_background": []}
    for entry in raw:
        gt = np.asarray(entry["gt"]["boxes"], float)
        boxes = np.asarray(entry["pred"]["boxes"], float)
        scores = np.asarray(entry["pred"]["scores"], float)
        if not len(boxes):
            continue
        matrix = iou_matrix(boxes, gt)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        selected = [i for i in order if scores[i] >= confidence]
        used, matched = set(), set()
        for i in selected:
            candidates = [(matrix[i, j], j) for j in range(matrix.shape[1]) if j not in used and matrix[i, j] >= .5]
            if candidates:
                used.add(max(candidates)[1])
                matched.add(i)
        for j in range(len(gt)):
            if j in used:
                continue
            column = matrix[:, j] if matrix.shape[0] else np.zeros(0)
            if not len(column) or column.max() < .5:
                continue
            k = int(column.argmax())
            if scores[k] >= min_score:
                found["fn_correct_box"].append(dict(image=entry["image_id"], box=boxes[k].tolist(),
                                                    score=float(scores[k]), value=float(column[k])))
        for i in selected:
            if i in matched:
                continue
            if matrix.shape[1] and matrix[i].max() > 0:
                continue
            if scores[i] < min_score:
                continue
            found["fp_background"].append(dict(image=entry["image_id"], box=boxes[i].tolist(),
                                               score=float(scores[i]), value=0.))
    for key in found:
        found[key].sort(key=lambda row: -row["score"])
        found[key] = found[key][:limit]
    return found


def filter_scores(rows, low, high):
    return [row for row in rows if low <= row["score"] < high]


def montage(rows, images_root, out_path, title):
    columns = 6
    scale = max(1, len(rows) // columns + (1 if len(rows) % columns else 0))
    sheet = Image.new("RGB", (columns * TILE, scale * (TILE + 16)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    cache = {}
    for index, row in enumerate(rows):
        name = row["image"]
        if name not in cache:
            with Image.open(Path(images_root) / name) as handle:
                cache[name] = handle.convert("RGB").copy()
        tile = crop(cache[name], row["box"])
        if tile is None:
            continue
        column, line = index % columns, index // columns
        sheet.paste(tile, (column * TILE, line * (TILE + 16)))
        draw.text((column * TILE + 2, line * (TILE + 16) + TILE + 2),
                  f"s={row['score']:.2f}", fill=(0, 0, 0))
    sheet.save(out_path)
    print(f"{title}: {len(rows)} crops -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--update", required=True)
    parser.add_argument("--images", default=str(ROOT / "assets/papple/images/val"))
    parser.add_argument("--out", default=str(ROOT / "reports/val_error_crops"))
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    directory = next((ROOT / args.run / "validation").glob(f"update_{args.update}_*"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    confidence = summary["confidence"]
    raw = json.loads((directory / "raw_predictions.json").read_text(encoding="utf-8"))
    found = collect(raw, args.images, confidence, 10 ** 9, 0.02)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    band = filter_scores(found["fn_correct_box"], 0.02, confidence)[:args.limit]
    montage(band, args.images, out / f"fn_below_threshold_{args.update}.png",
            "FN whose box is already IoU>=0.5, score below the selected threshold")
    montage(found["fp_background"][:args.limit], args.images, out / f"fp_background_{args.update}.png",
            "FP with zero GT overlap")
    stats = {key: dict(count=len(value),
                       score=[round(row["score"], 4) for row in value[:8]]) for key, value in found.items()}
    stats["fn_below_threshold"] = dict(
        count=len(found["fn_correct_box"]) - len(filter_scores(found["fn_correct_box"], confidence, 10 ** 9)),
        total=len(found["fn_correct_box"]))
    (out / f"crop_index_{args.update}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "fn_correct_box"}, indent=1))


if __name__ == "__main__":
    main()
