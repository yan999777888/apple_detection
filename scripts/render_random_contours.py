"""CPU-only display of native visible/amodal instance mask PNG crops.

Run after real inference has written sample_manifest.json and
contour_predictions.json in --input-dir. No model, GT, ellipse fit, contour
smoothing, or bounding-box rendering is used by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CYAN = (0, 235, 235)
PINK = (255, 70, 185)
INK = (28, 39, 58)
MODEL_NAMES = {"base": "原 ConvNeXt-Base", "small": "原 ConvNeXt-Small"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chinese_font_path(requested=None):
    candidates = ([Path(requested)] if requested else []) + [
        Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("A local Chinese TTF/TTC is required; use --font")


def face(path, size):
    return ImageFont.truetype(str(path), max(10, int(size)))


def text_width(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def fit_font(draw, text, path, width, size):
    while size > 11 and text_width(draw, text, face(path, size)) > width:
        size -= 1
    return face(path, size)


def draw_text(draw, xy, text, font, fill=INK):
    bounds = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - bounds[0], xy[1] - bounds[1]), text, font=font, fill=fill)


def native_boundary(mask, width=2):
    """Inside band from repeated 4-neighbour erosion; no curve fitting/smoothing.

    Outer shape, holes, disconnected fragments, and single-pixel structures are
    preserved. This changes display-band thickness only, not the binary mask.
    """
    eroded = np.asarray(mask, dtype=bool).copy()
    for _ in range(width):
        padded = np.pad(eroded, 1, constant_values=False)
        eroded = (padded[1:-1, 1:-1] & padded[:-2, 1:-1]
                  & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    return np.asarray(mask, dtype=bool) & ~eroded


def mask_path(input_dir, value):
    path = Path(value)
    if path.is_absolute():
        raise ValueError("Mask paths must be relative to --input-dir")
    resolved = (input_dir / path).resolve()
    if not resolved.is_relative_to(input_dir):
        raise ValueError("Mask path escapes --input-dir")
    return resolved


def load_mask(input_dir, entry, width, height):
    if entry.get("path") is None:
        if entry["foreground_pixels"] != 0:
            raise ValueError("Null mask path must have zero foreground_pixels")
        # No PNG exists for a truly empty native support. Rendering skips this
        # entry rather than inventing pixels, a contour, or an ellipse.
        return {"path": None, "relative_path": None,
                "window_xyxy": entry.get("window_xyxy", [0, 0, 0, 0]),
                "mask": np.zeros((0, 0), dtype=bool),
                "foreground_pixels": 0, "sha256": None}
    path = mask_path(input_dir, entry["path"])
    raw_window = entry["window_xyxy"]
    if len(raw_window) != 4 or any(isinstance(x, bool) or int(x) != x for x in raw_window):
        raise ValueError("window_xyxy must have four integer coordinates")
    x1, y1, x2, y2 = [int(x) for x in raw_window]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Mask crop window is outside native RGB geometry: " + str(path))
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError("Mask crop must be lossless PNG")
        array = np.asarray(image)
    if array.ndim != 2 or array.shape != (y2 - y1, x2 - x1):
        raise ValueError("Mask crop pixel shape differs from window_xyxy: " + str(path))
    values = set(np.unique(array).tolist())
    if not values.issubset({0, 1, 255, False, True}) or (1 in values and 255 in values):
        raise ValueError("PNG must already be binary; no rethresholding or smoothing: " + str(path))
    mask = array != 0
    pixels = int(mask.sum())
    if pixels != entry["foreground_pixels"]:
        raise ValueError("Mask foreground_pixels evidence mismatch: " + str(path))
    digest = sha256(path)
    if entry.get("sha256") is not None and entry["sha256"] != digest:
        raise ValueError("Mask PNG SHA differs from inference evidence: " + str(path))
    return {"path": str(path), "relative_path": entry["path"],
            "window_xyxy": [x1, y1, x2, y2], "mask": mask,
            "foreground_pixels": pixels, "sha256": digest}


def overlay(original, detections, branches, combined=False):
    """Composite each native mask crop at its exact integer image support."""
    pixels = np.array(original, dtype=np.uint8, copy=True)
    # In combined preview, show A first then V. Both use their true supports.
    for branch in branches:
        color = np.asarray(CYAN if branch == "visible_mask" else PINK, dtype=float)
        alpha = .19 if branch == "visible_mask" else .12 if combined else .19
        for detection in detections:
            item = detection[branch]
            if not item["mask"].size:
                continue
            x1, y1, x2, y2 = item["window_xyxy"]
            region = pixels[y1:y2, x1:x2]
            mask = item["mask"]
            region[mask] = np.rint(region[mask].astype(float) * (1 - alpha)
                                    + color * alpha).astype(np.uint8)
    for branch in branches:
        color = CYAN if branch == "visible_mask" else PINK
        # The combined image has a wider inside A band so coincident boundaries
        # can show pink beside the cyan edge. No mask support is expanded.
        band_width = (1 if branch == "visible_mask" else 3) if combined else 2
        for detection in detections:
            item = detection[branch]
            if not item["mask"].size:
                continue
            x1, y1, x2, y2 = item["window_xyxy"]
            region = pixels[y1:y2, x1:x2]
            region[native_boundary(item["mask"], band_width)] = color
    return Image.fromarray(pixels, "RGB")


def triple_panel(original, visible, amodal, row, font_path):
    width, height = original.size
    gap, top, bottom = 14, 88, 135
    result = Image.new("RGB", (width * 3 + gap * 4, height + top + bottom), (244, 248, 253))
    draw = ImageDraw.Draw(result)
    titles = ["原图", "可见掩膜 V：青色", "完整掩膜 A：粉色（遮挡部分为预测）"]
    for i, (image, title) in enumerate(zip([original, visible, amodal], titles)):
        x = gap + i * (width + gap)
        draw_text(draw, (x + 7, 26), title,
                  fit_font(draw, title, font_path, width - 14, 30))
        result.paste(image, (x, top))
    y = top + height + 20
    title = f"{MODEL_NAMES[row['model']]} · 原图单视图 · {row['image_id']} · 过阈值实例 {row['detection_count']}"
    draw_text(draw, (gap + 7, y), title,
              fit_font(draw, title, font_path, result.width - gap * 2 - 14, 25))
    note = "confidence≥0.75；mask threshold=0.5。边界直接来自二值掩膜；未画框、未拟合、未平滑。"
    draw_text(draw, (gap + 7, y + 43), note,
              fit_font(draw, note, font_path, result.width - gap * 2 - 14, 22))
    return result


def make_contact(rows, model, output, font_path):
    cols, cell, gap = min(4, len(rows)), 570, 22
    top, cell_height = 150, 660
    result = Image.new("RGB", (cols * cell + (cols + 1) * gap,
                               top + math.ceil(len(rows) / cols) * cell_height + gap),
                       (240, 245, 252))
    assert result.width <= 2400
    draw = ImageDraw.Draw(result)
    title = MODEL_NAMES[model] + " · 原图单视图轮廓（全部随机样本）"
    draw_text(draw, (gap, 23), title,
              fit_font(draw, title, font_path, result.width - 2 * gap, 31))
    note = "青色：可见 V；粉色：完整 A（包含预测遮挡部分）。confidence≥0.75；mask≥0.5。"
    draw_text(draw, (gap, 72), note,
              fit_font(draw, note, font_path, result.width - 2 * gap, 22))
    note2 = "未画框、未拟合/平滑；这里不是四视图融合掩膜。"
    draw_text(draw, (gap, 108), note2,
              fit_font(draw, note2, font_path, result.width - 2 * gap, 20))
    for i, row in enumerate(rows):
        x = gap + i % cols * (cell + gap)
        y = top + i // cols * cell_height
        with Image.open(output / row["combined_image"]) as source:
            image = source.convert("RGB")
        # Nearest neighbour preview keeps pixelated edges; full native PNG is
        # linked separately in the HTML for accurate contour inspection.
        image.thumbnail((cell, 570), Image.Resampling.NEAREST)
        result.paste(image, (x + (cell - image.width) // 2, y))
        title = f"{i + 1:02d}  {row['image_id']}"
        draw_text(draw, (x + 5, y + 585), title,
                  fit_font(draw, title, font_path, cell - 10, 24))
        detail = f"过阈值实例 {row['detection_count']}；按原随机清单展示"
        draw_text(draw, (x + 5, y + 623), detail,
                  fit_font(draw, detail, font_path, cell - 10, 20))
    path = output / f"contact_{model}.jpg"
    result.save(path, quality=95, subsampling=0)
    return path.name


def write_gallery(manifest, model_rows, output):
    esc = lambda value: html.escape(str(value), quote=True)
    sections = []
    for model, rows in model_rows.items():
        cards = []
        for i, row in enumerate(rows):
            scores = "；".join(f"原预测#{d['prediction_index']}={d['score']:.3f}" for d in row["detections"])
            cards.append(f'''<article><h3>{i + 1:02d} · {esc(row['image_id'])}</h3>
<p>原图单视图，过阈值实例 {row['detection_count']}。{esc(scores or '没有实例超过该阈值。')}</p>
<a href="{esc(row['triple_image'])}" target="_blank"><img loading="lazy" src="{esc(row['triple_image'])}" alt="原图、可见青色和完整粉色掩膜"></a>
<p><a href="{esc(row['visible_image'])}" target="_blank">原尺寸可见轮廓 V</a> · <a href="{esc(row['amodal_image'])}" target="_blank">原尺寸完整轮廓 A</a> · <a href="{esc(row['combined_image'])}" target="_blank">两类轮廓叠加</a></p></article>''')
        sections.append(f'''<section id="{model}"><h2>{esc(MODEL_NAMES[model])} · 原图单视图</h2>
<a href="contact_{model}.jpg" target="_blank"><img src="contact_{model}.jpg" alt="{esc(MODEL_NAMES[model])} 所有随机图的轮廓总览"></a>
<p>点击总览或下方图片放大。缩略图只用于浏览，轮廓细节以链接的原尺寸 PNG 为准。</p>{''.join(cards)}</section>''')
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>原强模型的真实可见与完整掩膜轮廓</title><style>body{{margin:0;background:#edf2f9;color:#1b293d;font:16px/1.75 "Microsoft YaHei",sans-serif}}main{{max-width:1650px;margin:auto;padding:28px}}header,section{{background:#fff;border:1px solid #dce4ef;border-radius:14px;padding:25px;margin-bottom:24px}}h1{{font-size:29px;margin:0}}h2{{font-size:25px}}h3{{font-size:21px;margin:0}}article{{border-top:1px solid #dce4ef;padding:23px 0}}img{{max-width:100%;height:auto;display:block}}a{{color:#1268bc}}.v{{background:#c8ffff;padding:4px 10px}}.a{{background:#ffe0f4;padding:4px 10px}}@media(max-width:800px){{main{{padding:12px}}header,section{{padding:16px}}}}</style>
<main><header><h1>原强模型 · 真实可见／完整实例掩膜</h1>
<p>按原 sample_manifest 的顺序展示全部 {len(manifest['samples'])} 张；seed＝{esc(manifest['seed'])}。不按分割效果挑图，不读取 GT。confidence≥0.75；mask threshold＝0.5。</p>
<p><span class="v">青色：可见区域 V</span> <span class="a">粉色：完整区域 A</span>。半透明填充与边界均来自模型输出的原生二值掩膜，保留孔洞、碎片和锯齿；没有使用矩形框、椭圆拟合或轮廓平滑。</p>
<p><strong>Base、Small 分开显示各自原图单视图的 native masks。现有四视图方案只融合框和分数，没有实现或声称融合像素级掩膜。</strong>完整掩膜中被枝叶挡住的部分是模型预测，不能当作真实可见边界或深度测量。</p>
<p>这组图片用于检查轮廓外观，未进行 GT 掩膜评分，小样本不能代替完整 VAL、独立测试或实地采摘验收。</p>
<nav><a href="#base">查看 Base</a> · <a href="#small">查看 Small</a> · <a href="summary.json">结构化结果</a></nav></header>{''.join(sections)}</main></html>'''
    (output / "gallery.html").write_text(content, encoding="utf-8")


def render(input_dir, output_dir=None, requested_font=None):
    input_dir = input_dir.resolve()
    output = (output_dir or input_dir).resolve()
    manifest_path = input_dir / "sample_manifest.json"
    predictions_path = input_dir / "contour_predictions.json"
    manifest, predictions = read_json(manifest_path), read_json(predictions_path)
    if predictions["confidence"] != .75 or predictions["mask_threshold"] != .5:
        raise ValueError("This display protocol requires confidence=.75, mask_threshold=.5")
    if "confidence" in manifest and manifest["confidence"] != .75:
        raise ValueError("Manifest confidence differs from contour protocol")
    samples = manifest["samples"]
    ids = [s["image_id"] for s in samples]
    if not ids or len(ids) != len(set(ids)) or any(not i for i in ids):
        raise ValueError("Manifest sample IDs must be nonempty and unique")
    if len({s["index"] for s in samples}) != len(samples):
        raise ValueError("Manifest sample indices must be unique")
    for name in ["images", "summary.json", "gallery.html", "contact_base.jpg", "contact_small.jpg"]:
        if (output / name).exists():
            raise FileExistsError("Existing contour display output; use a fresh --output-dir")
    font_path = chinese_font_path(requested_font)
    models = {}
    # Validate identities and source dimensions before output is written.
    for model in MODEL_NAMES:
        records = predictions["models"][model]
        if len(records) != len(ids) or {r["image_id"] for r in records} != set(ids):
            raise ValueError("Each model must provide exactly the manifest samples")
        by_id = {r["image_id"]: r for r in records}
        models[model] = by_id
        for sample in samples:
            record = by_id[sample["image_id"]]
            if (record["width"], record["height"]) != (sample["width"], sample["height"]):
                raise ValueError("Prediction record dimensions differ from manifest")
            paths = []
            for value in [sample["rgb_path"], record["rgb_path"]]:
                path = Path(value)
                paths.append((path if path.is_absolute() else input_dir / path).resolve())
            if paths[0] != paths[1]:
                raise ValueError("RGB identity differs between manifest and predictions")
            with Image.open(paths[0]) as image:
                if image.size != (sample["width"], sample["height"]):
                    raise ValueError("Native RGB geometry differs from manifest")
            indices = [d["prediction_index"] for d in record["detections"]]
            if len(indices) != len(set(indices)):
                raise ValueError("Duplicate native prediction_index")
            for detection in record["detections"]:
                if not math.isfinite(detection["score"]) or not 0 <= detection["score"] <= 1:
                    raise ValueError("Invalid detection score")
    output.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    for model in MODEL_NAMES:
        image_dir = output / "images" / model
        image_dir.mkdir(parents=True)
        rows = []
        for position, sample in enumerate(samples, 1):
            record = models[model][sample["image_id"]]
            rgb_path = Path(sample["rgb_path"])
            rgb_path = (rgb_path if rgb_path.is_absolute() else input_dir / rgb_path).resolve()
            with Image.open(rgb_path) as source:
                original = source.convert("RGB")
            selected = sorted((d for d in record["detections"] if d["score"] >= .75),
                              key=lambda d: -d["score"])
            masks = []
            serializable = []
            for detection in selected:
                item = {"prediction_index": detection["prediction_index"],
                        "score": detection["score"]}
                for branch in ["visible_mask", "amodal_mask"]:
                    item[branch] = load_mask(input_dir, detection[branch], *original.size)
                masks.append(item)
                serializable.append({"prediction_index": item["prediction_index"],
                    "score": item["score"], **{b: {k: v for k, v in item[b].items() if k != "mask"}
                                             for b in ["visible_mask", "amodal_mask"]}})
            visible = overlay(original, masks, ["visible_mask"])
            amodal = overlay(original, masks, ["amodal_mask"])
            combined = overlay(original, masks, ["amodal_mask", "visible_mask"], combined=True)
            prefix = f"{position:02d}_{Path(sample['image_id']).stem}"
            paths = {kind + "_image": f"images/{model}/{prefix}_{kind}.png"
                     for kind in ["visible", "amodal", "combined", "triple"]}
            row = {"model": model, "manifest_index": sample["index"],
                   "image_id": sample["image_id"], "rgb_path": str(rgb_path),
                   "width": original.width, "height": original.height,
                   "detection_count": len(masks), "detections": serializable, **paths}
            visible.save(output / row["visible_image"])
            amodal.save(output / row["amodal_image"])
            combined.save(output / row["combined_image"])
            triple_panel(original, visible, amodal, row, font_path).save(output / row["triple_image"])
            row["output_sha256"] = {k: sha256(output / v) for k, v in paths.items()}
            rows.append(row)
        make_contact(rows, model, output, font_path)
        all_rows[model] = rows
    summary = {"schema": "random_native_VA_contour_gallery_v1", "seed": manifest["seed"],
               "confidence": .75, "mask_threshold": .5, "sample_count_per_model": len(samples),
               "image_order": ids, "models": all_rows,
               "scope": "Original Base and Small separately, original-image single-view native masks; no fused masks; no GT scoring",
               "GT_read": False, "boxes_drawn": False, "ellipse_fit": False,
               "contour_smoothing": False, "GPU_used": False, "model_checkpoint_loaded": False,
               "original_RGB_or_masks_modified": False,
               "boundary_method": "Direct inside binary boundary band from 4-neighbour erosion; holes, fragments and pixel steps preserved; no expansion of support",
               "occluded_amodal_region_is_prediction": True,
               "manifest_sha256": sha256(manifest_path),
               "contour_predictions_sha256": sha256(predictions_path),
               "font_path": str(font_path), "contact_sheets": ["contact_base.jpg", "contact_small.jpg"],
               "gallery": "gallery.html"}
    write_gallery(manifest, all_rows, output)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert "torch" not in sys.modules, "CPU display unexpectedly imported torch"
    print(json.dumps({"gallery": str(output / "gallery.html"), "summary": str(output / "summary.json"),
                      "samples_per_model": len(samples),
                      "instances": {k: sum(x["detection_count"] for x in rows) for k, rows in all_rows.items()}}, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--font")
    args = parser.parse_args()
    render(args.input_dir, args.output_dir, args.font)


if __name__ == "__main__":
    main()
