"""Render annotated examples from the published full-VAL horizontal-flip TTA."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import flip_tta

ROOT = Path(__file__).resolve().parents[1]
FONT_REGULAR = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD = Path("C:/Windows/Fonts/msyhbd.ttc")
GREEN, RED, ORANGE = (0, 190, 80), (225, 45, 65), (255, 145, 0)
DARK = (24, 28, 32)


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def text_size(draw, text, used_font):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=used_font)
    return right - left, bottom - top


def label(draw, x, y, text, color, used_font, canvas_width):
    width, height = text_size(draw, text, used_font)
    x = max(2, min(int(x), canvas_width - width - 12))
    y = max(2, int(y))
    draw.rounded_rectangle((x, y, x + width + 10, y + height + 8), radius=4,
                           fill=(16, 20, 24), outline=color, width=2)
    draw.text((x + 5, y + 2), text, font=used_font, fill=color)


def structure(record, metadata, threshold):
    gt = np.asarray(record["gt"]["boxes"], dtype=np.float64).reshape(-1, 4)
    boxes = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
    match = flip_tta.metrics.match_image(gt, boxes, scores, .5, threshold)
    selected = list(match["pred_indices"])
    assignment = dict(zip(selected, (int(value) for value in match["gt_indices"])))
    used = {value for value in assignment.values() if value >= 0}
    heavy = sum(1 for row in metadata["ground_truth_identity"]
                if row["image_id"] == record["image_id"]
                for gt_row in row["GT"] if gt_row["occlusion_stratum"].startswith("heavy"))
    return {"gt": gt, "boxes": boxes, "scores": scores, "selected": selected,
            "assignment": assignment, "used": used, "heavy": heavy,
            "tp": len(used), "fp": len(selected) - len(used), "fn": len(gt) - len(used)}


def render(record, source, metadata, threshold, title, out_path):
    stats = structure(record, metadata, threshold)
    image = Image.open(source["rgb_path"]).convert("RGB")
    if list(image.size) != [source["crop_xyxy"][2] - source["crop_xyxy"][0],
                            source["crop_xyxy"][3] - source["crop_xyxy"][1]]:
        image = image.crop(tuple(source["crop_xyxy"]))
    array = np.asarray(image)
    active_columns = np.flatnonzero(array.mean(axis=(0, 2)) > 1.5)
    active_rows = np.flatnonzero(array.mean(axis=(1, 2)) > 1.5)
    left, right = int(active_columns[0]), int(active_columns[-1]) + 1
    top, bottom = int(active_rows[0]), int(active_rows[-1]) + 1
    image = image.crop((left, top, right, bottom))
    stats["boxes"] = stats["boxes"] - np.asarray([left, top, left, top])
    stats["gt"] = stats["gt"] - np.asarray([left, top, left, top])
    width, height = image.size
    body = image.copy()
    draw = ImageDraw.Draw(body)
    box_font = font(max(20, round(min(width, height) / 50)), True)
    small_font = font(max(20, round(min(width, height) / 55)), True)

    def clipped(box):
        return (max(0., box[0]), max(0., box[1]), min(float(width), box[2]), min(float(height), box[3]))

    for row in stats["selected"]:
        gt_index = stats["assignment"][row]
        x1, y1, x2, y2 = clipped(stats["boxes"][row])
        if gt_index >= 0:
            draw.rectangle((x1, y1, x2, y2), outline=GREEN, width=5)
            label(draw, x1, y1 - 36, f"命中 {stats['scores'][row]:.2f}", GREEN, box_font, width)
        else:
            draw.rectangle((x1, y1, x2, y2), outline=RED, width=5)
            label(draw, x1, y1 - 36, f"误检 {stats['scores'][row]:.2f}", RED, box_font, width)
    for gt_index, box in enumerate(stats["gt"]):
        if gt_index in stats["used"]:
            continue
        x1, y1, x2, y2 = clipped(box)
        draw.rectangle((x1, y1, x2, y2), outline=ORANGE, width=5)
        label(draw, x1, y2 + 4, "漏检 GT", ORANGE, box_font, width)

    overlay = Image.new("RGBA", body.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    legend = "绿框 命中    红框 误检    橙框 漏检 GT"
    legend_width, legend_height = text_size(odraw, legend, small_font)
    odraw.rounded_rectangle((14, 14, 28 + legend_width, 26 + legend_height + 10),
                            radius=8, fill=(*DARK, 205))
    odraw.text((21, 19), legend, font=small_font, fill="white")
    body = Image.alpha_composite(body.convert("RGBA"), overlay).convert("RGB")

    header = max(96, round(height * .085))
    canvas = Image.new("RGB", (width, height + header), "white")
    canvas.paste(body, (0, header))
    cdraw = ImageDraw.Draw(canvas)
    cdraw.rectangle((0, 0, width, header), fill=DARK)
    title_font = font(max(30, round(header * .34)), True)
    detail_font = font(max(23, round(header * .235)))
    cdraw.text((24, 14), title, font=title_font, fill="white")
    detail = (f"GT {len(stats['gt'])}   命中 {stats['tp']}   误检 {stats['fp']}   "
              f"漏检 {stats['fn']}   重遮挡 {stats['heavy']}   阈值 {threshold:.4f}   {record['image_id']}")
    cdraw.text((24, 18 + int(header * .45)), detail, font=detail_font, fill=(210, 218, 226))
    canvas.save(out_path, quality=95)
    return stats


def contact_sheet(items, out_path, columns=2):
    thumb_width = 720
    with Image.open(items[0][0]) as first:
        thumb_height = int(round(thumb_width * first.height / first.width))
    margin, gap, caption = 18, 14, 54
    cell_h = thumb_height + caption
    rows = (len(items) + columns - 1) // columns
    width = margin * 2 + columns * thumb_width + (columns - 1) * gap
    height = margin * 2 + rows * cell_h + (rows - 1) * gap
    sheet = Image.new("RGB", (width, height), (238, 241, 244))
    draw = ImageDraw.Draw(sheet)
    caption_font = font(22, True)
    for index, (path, title, stats) in enumerate(items):
        row, column = divmod(index, columns)
        x = margin + column * (thumb_width + gap)
        y = margin + row * (cell_h + gap)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize((thumb_width, thumb_height), Image.LANCZOS)
            sheet.paste(image, (x, y))
        draw.rectangle((x, y, x + thumb_width, y + thumb_height), outline=(180, 185, 190), width=2)
        draw.text((x + 2, y + thumb_height + 8),
                  f"{title}  |  命中 {stats['tp']} · 误检 {stats['fp']} · 漏检 {stats['fn']}",
                  font=caption_font, fill=DARK)
    sheet.save(out_path, quality=94)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--threshold", type=float, default=.56856968998909)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ids", nargs="+", default=[
        "_MG_2690_21.png", "_MG_2776_14.png", "_MG_6354_12.png", "_MG_6510_01.png"])
    args = parser.parse_args()
    records = {row["image_id"]: row for row in flip_tta.load_records(args.predictions)}
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    sources = {row["image_id"]: row["source"] for row in metadata["ground_truth_identity"]}
    titles = {
        "_MG_2690_21.png": "典型成功案例",
        "_MG_2776_14.png": "重遮挡案例",
        "_MG_6354_12.png": "漏检案例",
        "_MG_6510_01.png": "误检案例（红框=未匹配 GT 的预测）",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items, report = [], {}
    for image_id in args.ids:
        if image_id not in records or image_id not in sources:
            raise ValueError(f"Unknown image: {image_id}")
        path = out_dir / f"{Path(image_id).stem}_annotated.jpg"
        stats = render(records[image_id], sources[image_id], metadata, args.threshold,
                       titles.get(image_id, "检测示例"), path)
        items.append((path, titles.get(image_id, "检测示例"), stats))
        report[image_id] = {key: stats[key] for key in ("tp", "fp", "fn", "heavy")}
        report[image_id]["gt"] = int(len(stats["gt"]))
        report[image_id]["output"] = str(path)
        print(image_id, report[image_id], flush=True)
    sheet = out_dir / "苹果检测示例_4图.jpg"
    contact_sheet(items, sheet)
    (out_dir / "examples_summary.json").write_text(json.dumps({
        "threshold": args.threshold, "predictions": args.predictions, "examples": report}, indent=1),
        encoding="utf-8")
    print("wrote", sheet)


if __name__ == "__main__":
    main()
