"""Visualise the detector's native visible and amodal (whole-fruit) masks."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import flip_tta
from apple_seg.od_roi import paste_roi_mask

ROOT = Path(__file__).resolve().parents[1]
FONT = Path("C:/Windows/Fonts/msyhbd.ttc")
DARK = (20, 25, 30)
VIS = (40, 220, 100)
AMO = (40, 170, 255)
HID = (255, 145, 0)


def font(size):
    return ImageFont.truetype(str(FONT), size=size)


def crop_content(image):
    array = np.asarray(image)
    cols = np.flatnonzero(array.mean(axis=(0, 2)) > 1.5)
    rows = np.flatnonzero(array.mean(axis=(1, 2)) > 1.5)
    box = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    return image.crop(box), box


def get_masks(record, cache_file, selected):
    count = len(record["pred"]["scores"])
    with np.load(cache_file, allow_pickle=False) as archive:
        visible = archive["visible"]
        amodal = archive["amodal"]
    if visible.shape != (count, 56, 56) or amodal.shape != (count, 56, 56):
        raise ValueError("Mask cache shape differs from raw detections")
    height, width = 1300, 1300
    visible_full = np.zeros((height, width), dtype=bool)
    amodal_full = np.zeros((height, width), dtype=bool)
    boxes = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
    for index in selected:
        box = torch.tensor(boxes[index], dtype=torch.float32)
        for source, target in ((visible[index], visible_full), (amodal[index], amodal_full)):
            binary, (left, top, right, bottom) = paste_roi_mask(
                torch.from_numpy(source), box, (height, width))
            target[top:bottom, left:right] |= binary.numpy()
    return visible_full, amodal_full


def overlay(image, visible, amodal, mode):
    if mode == "original":
        return image.copy(), "原图"
    rgba = Image.new("RGBA", image.size, (0, 0, 0, 0))
    layer = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    hidden = amodal & ~visible
    if mode == "visible":
        layer[visible] = (*VIS, 110)
        edge = visible
        edge_color = VIS
        title = "可见果皮 V"
    elif mode == "amodal":
        layer[amodal] = (*AMO, 58)
        edge = amodal
        edge_color = AMO
        title = "完整果实 A（预测含遮挡）"
    else:
        layer[visible] = (*VIS, 42)
        layer[hidden] = (*HID, 175)
        edge = amodal
        edge_color = AMO
        title = "遮挡补全 H = A 且非 V"
    padded = np.pad(edge, 1, constant_values=False)
    inner = (padded[:-2, 1:-1] & padded[2:, 1:-1]
             & padded[1:-1, :-2] & padded[1:-1, 2:])
    outline_array = edge & ~inner
    thick = outline_array.copy()
    for axis, offset in ((0, 1), (0, -1), (1, 1), (1, -1)):
        thick |= np.roll(outline_array, offset, axis=axis)
    layer[thick] = (*edge_color, 255)
    rgba = Image.fromarray(layer, "RGBA")
    return Image.alpha_composite(image.convert("RGBA"), rgba).convert("RGB"), title


def render(image, visible, amodal):
    image, content = crop_content(image)
    left, top, right, bottom = content
    panels = []
    for mode in ("original", "visible", "amodal", "hidden"):
        panel, title = overlay(image, visible[top:bottom, left:right],
                              amodal[top:bottom, left:right], mode)
        panels.append((panel, title))
    return panels


def mask_examples(record, cache_file, metadata, threshold, title):
    gt = np.asarray(record["gt"]["boxes"], dtype=np.float64).reshape(-1, 4)
    boxes = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
    match = flip_tta.metrics.match_image(gt, boxes, scores, .5, threshold)
    selected = [int(p) for p, g in zip(match["pred_indices"], match["gt_indices"])
                if g >= 0 and scores[p] >= threshold]
    visible, amodal = get_masks(record, cache_file, selected)
    source = next(row["source"] for row in metadata["ground_truth_identity"]
                  if row["image_id"] == record["image_id"])
    image = Image.open(source["rgb_path"]).convert("RGB")
    return dict(image=image, visible=visible, amodal=amodal, masks=len(selected),
                title=title, image_id=record["image_id"])


def build_sheet(examples, out_path):
    panel_width, caption, margin, gap = 460, 44, 18, 12
    rows = len(examples)
    width = margin * 2 + panel_width * 4 + gap * 3
    height = margin * 2 + rows * (panel_width + caption + 42) + gap * (rows - 1)
    sheet = Image.new("RGB", (width, height), (238, 241, 244))
    draw = ImageDraw.Draw(sheet)
    caption_font = font(23)
    row_font = font(26)
    legend = "绿 = 可见果皮 V     蓝 = 完整果实 A     橙 = 遮挡补全 H（A 中不可见部分）"
    draw.text((margin, 6), legend, font=font(21), fill=DARK)
    for row, example in enumerate(examples):
        y = margin + row * (panel_width + caption + 42 + gap) + 28
        draw.text((margin, y), f"{example['title']}   {example['image_id']}    "
                               f"预测果实 {example['masks']} 个", font=row_font, fill=DARK)
        for column, (panel, title) in enumerate(example["panels"]):
            x = margin + column * (panel_width + gap)
            panel = panel.resize((panel_width, panel_width), Image.LANCZOS)
            sheet.paste(panel, (x, y + 34))
            draw.text((x + 4, y + 34 + panel_width + 7), title, font=caption_font, fill=DARK)
    sheet.save(out_path, quality=94)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-predictions", required=True)
    parser.add_argument("--mask-cache", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--threshold", type=float, default=.6156370043754578)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ids", nargs="+", default=["_MG_2690_21.png", "_MG_2776_14.png"])
    args = parser.parse_args()
    records = flip_tta.load_records(args.raw_predictions)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    index_by_id = {record["image_id"]: index for index, record in enumerate(records)}
    titles = {"_MG_2690_21.png": "典型成功案例", "_MG_2776_14.png": "重遮挡案例"}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    examples, report = [], {}
    for image_id in args.ids:
        index = index_by_id[image_id]
        cache_file = Path(args.mask_cache) / f"roi_{index:06d}.npz"
        if not cache_file.is_file():
            raise FileNotFoundError(cache_file)
        example = mask_examples(records[index], cache_file, metadata, args.threshold,
                                titles.get(image_id, image_id))
        example["panels"] = render(example["image"], example["visible"], example["amodal"])
        width = example["panels"][0][0].width
        montage = Image.new("RGB", (width * 4 + 36, width + 56), "white")
        montage_draw = ImageDraw.Draw(montage)
        for column, (panel, title) in enumerate(example["panels"]):
            montage.paste(panel, (column * (width + 12), 56))
            montage_draw.text((column * (width + 12) + 8, 10), title, font=font(32), fill=DARK)
        single = out / f"{Path(image_id).stem}_amodal_masks.jpg"
        montage.save(single, quality=95)
        report[image_id] = {"predicted_fruit_masks": example["masks"], "output": str(single),
                            "title": example["title"]}
        examples.append(example)
        print(image_id, report[image_id], flush=True)
    sheet = out / "完整果实掩膜预测_示例.jpg"
    build_sheet(examples, sheet)
    (out / "masks_summary.json").write_text(json.dumps({
        "threshold": args.threshold, "mask_source": args.mask_cache,
        "examples": report, "sheet": str(sheet)}, indent=1), encoding="utf-8")
    print("wrote", sheet)


if __name__ == "__main__":
    main()
