"""Render supplied random-sample predictions on CPU, without loading a model.

Inputs in --input-dir: sample_manifest.json and fused_predictions.json.
Outputs: images/predictions, images/comparisons, contact_sheet.jpg,
gallery.html, and summary.json. Original RGB files are opened read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


METRICS_SHA256 = "1b0a7e385940fd0c6160cc08b75ddaf2e1f43f3ea00f5bd37d547fea6b08010d"
BLUE = (20, 135, 255)
GREEN = (20, 195, 100)
RED = (245, 65, 75)
ORANGE = (255, 165, 35)
INK = (25, 38, 60)
WHITE = (255, 255, 255)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_metrics():
    path = (Path(__file__).resolve().parents[1]
            / "strong_runtime/reference/local_amodal/scripts/metrics.py")
    if file_sha256(path) != METRICS_SHA256:
        raise ValueError("Frozen NumPy metrics source SHA does not match")
    spec = importlib.util.spec_from_file_location("random_overlay_frozen_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def font_path(requested: str | None) -> Path:
    candidates = ([Path(requested)] if requested else []) + [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("A Chinese font is required; pass --font with a local TTF/TTC")


def font(path: Path, size: int):
    return ImageFont.truetype(str(path), max(10, int(size)))


def text_size(draw, value, face):
    left, top, right, bottom = draw.textbbox((0, 0), value, font=face)
    return right - left, bottom - top


def draw_text(draw, xy, value, face, fill=INK):
    # Anchor the visible ink, not the font's ascender padding.
    bounds = draw.textbbox((0, 0), value, font=face)
    draw.text((xy[0] - bounds[0], xy[1] - bounds[1]), value, font=face, fill=fill)


def fit_text(draw, value, face_path, max_width, size, min_size=12):
    while size > min_size:
        face = font(face_path, size)
        if text_size(draw, value, face)[0] <= max_width:
            return face
        size -= 1
    return font(face_path, min_size)


def intersection_area(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))


def place_label(box, width, height, label_width, label_height, blocked, used):
    """Avoid prediction rectangles using only prediction geometry, never GT.

    A global free-space grid is used if nearby labels do not fit. If a dense
    image has no free rectangle, choose the least-overlapping location and
    record that limitation rather than silently omit a selected prediction.
    """
    x1, y1, x2, y2 = box
    gap = 5
    candidates = [
        (x1, y1 - label_height - gap), (x1, y2 + gap),
        (x2 - label_width, y1 - label_height - gap),
        (x2 - label_width, y2 + gap),
        (x2 + gap, y1), (x1 - label_width - gap, y1),
    ]
    for y in range(3, max(4, height - label_height), label_height + 7):
        for x in range(3, max(4, width - label_width), max(30, label_width // 2)):
            candidates.append((x, y))
    unique = set()
    ranked = []
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    for x, y in candidates:
        x = int(max(1, min(width - label_width - 1, x)))
        y = int(max(1, min(height - label_height - 1, y)))
        if (x, y) in unique:
            continue
        unique.add((x, y))
        rect = (x, y, x + label_width, y + label_height)
        overlaps_boxes = sum(intersection_area(rect, b) for b in blocked)
        overlaps_labels = sum(intersection_area(rect, b) for b in used)
        distance = (x + label_width / 2 - cx) ** 2 + (y + label_height / 2 - cy) ** 2
        ranked.append(((overlaps_boxes, overlaps_labels, distance), rect))
    _, chosen = min(ranked, key=lambda row: row[0])
    collision = any(intersection_area(chosen, b) > 0 for b in blocked)
    return chosen, collision


def dashed_line(draw, begin, end, color, width, dash):
    distance = math.dist(begin, end)
    if distance == 0:
        return
    for start in range(0, int(math.ceil(distance)), dash * 2):
        stop = min(start + dash, distance)
        a = tuple(begin[k] + (end[k] - begin[k]) * start / distance for k in (0, 1))
        b = tuple(begin[k] + (end[k] - begin[k]) * stop / distance for k in (0, 1))
        draw.line([a, b], fill=color, width=width)


def dashed_box(draw, box, color, width):
    x1, y1, x2, y2 = box
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    for a, b in zip(points[:-1], points[1:]):
        dashed_line(draw, a, b, color, width, max(6, width * 3))


def overlay_predictions(original, selected, face_path, colors=None, suffixes=None,
                        missed_gt=None):
    """Pure display passes no GT; comparison display explicitly receives GT."""
    image = original.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(2, round(min(width, height) / 330))
    face = font(face_path, max(12, min(28, round(min(width, height) / 53))))
    boxes = [list(x["box"]) for x in selected]
    used = []
    fallback_count = 0
    colors = colors if colors is not None else [BLUE] * len(selected)
    suffixes = suffixes if suffixes is not None else [""] * len(selected)
    if missed_gt:
        for row in missed_gt:
            dashed_box(draw, row["box"], ORANGE, line_width)
    for row, color in zip(selected, colors):
        draw.rectangle(row["box"], outline=color, width=line_width)
    for row, color, suffix in zip(selected, colors, suffixes):
        label = f"#{row['display_index']} {row['score']:.3f}{suffix}"
        tw, th = text_size(draw, label, face)
        rect, overlaps = place_label(row["box"], width, height, tw + 10, th + 8,
                                      boxes, used)
        fallback_count += int(overlaps)
        used.append(rect)
        x1, y1, x2, y2 = row["box"]
        label_center = ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
        nearest = (max(x1, min(x2, label_center[0])), max(y1, min(y2, label_center[1])))
        if math.dist(nearest, label_center) > line_width + 10:
            draw.line([nearest, label_center], fill=color, width=max(1, line_width // 2))
        draw.rounded_rectangle(rect, radius=3, fill=(12, 23, 40), outline=color,
                               width=max(1, line_width // 2))
        draw_text(draw, (rect[0] + 5, rect[1] + 4), label, face, WHITE)
    return image, fallback_count


def comparison_panel(original, predictions, comparison, row, face_path):
    panel_width = min(1100, original.width)
    panel_height = max(1, round(original.height * panel_width / original.width))
    gutter, header, footer = 12, 70, 116
    canvas = Image.new("RGB", (panel_width * 3 + gutter * 4,
                                panel_height + header + footer), (245, 248, 253))
    draw = ImageDraw.Draw(canvas)
    labels = ["原图", "模型预测：统一蓝色，与 GT 无关", "GT 对照：按 IoU≥0.5 一对一匹配"]
    for column, (image, title) in enumerate(zip([original, predictions, comparison], labels)):
        x = gutter + column * (panel_width + gutter)
        face = fit_text(draw, title, face_path, panel_width - 12, 25)
        draw_text(draw, (x + 6, 22), title, face)
        canvas.paste(image.resize((panel_width, panel_height), Image.Resampling.LANCZOS),
                     (x, header))
    y = header + panel_height + 18
    info = f"{row['image_id']}    TP {row['tp']}  /  FP {row['fp']}  /  FN {row['fn']}"
    draw_text(draw, (gutter + 8, y), info, font(face_path, 23))
    legend_y = y + 35
    x = gutter + 8
    for color, text in [(BLUE, "蓝：全部过阈值预测"), (GREEN, "绿：预测 TP"),
                        (RED, "红：预测 FP"), (ORANGE, "橙色虚线：GT 漏检 FN")]:
        draw.rectangle((x, legend_y + 3, x + 20, legend_y + 19), fill=color)
        draw_text(draw, (x + 28, legend_y), text, font(face_path, 20))
        x += text_size(draw, text, font(face_path, 20))[0] + 66
    return canvas


def contact_sheet(rows, output, face_path, confidence, model_label):
    cols = min(4, len(rows))
    cell_width, pad, top = 570, 22, 130
    thumb_height, cell_height = 570, 670
    nrows = math.ceil(len(rows) / cols)
    canvas = Image.new("RGB", (cols * cell_width + (cols + 1) * pad,
                                top + nrows * cell_height + pad), (240, 245, 252))
    assert canvas.width <= 2400
    draw = ImageDraw.Draw(canvas)
    title = "随机样本 · " + model_label
    draw_text(draw, (pad, 24), title,
              fit_text(draw, title, face_path, canvas.width - pad * 2, 31))
    draw_text(draw, (pad, 72), f"蓝框仅显示模型输出；confidence≥{confidence:g}。计数按 GT/IoU≥0.5 判断。",
              fit_text(draw, f"蓝框仅显示模型输出；confidence≥{confidence:g}。计数按 GT/IoU≥0.5 判断。", face_path, canvas.width - pad * 2, 22))
    for i, row in enumerate(rows):
        x = pad + (i % cols) * (cell_width + pad)
        y = top + (i // cols) * cell_height
        with Image.open(output / row["prediction_image"]) as source:
            image = source.convert("RGB")
        image.thumbnail((cell_width, thumb_height), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (cell_width - image.width) // 2, y))
        title = f"{i + 1:02d}  {row['image_id']}"
        face = fit_text(draw, title, face_path, cell_width - 10, 24)
        draw_text(draw, (x + 5, y + thumb_height + 14), title, face)
        detail = f"预测 {row['prediction_count']}    TP {row['tp']} / FP {row['fp']} / FN {row['fn']}"
        draw_text(draw, (x + 5, y + thumb_height + 53), detail,
                  fit_text(draw, detail, face_path, cell_width - 10, 22))
    canvas.save(output / "contact_sheet.jpg", quality=95, subsampling=0)


def write_gallery(manifest, rows, summary, output):
    esc = lambda value: html.escape(str(value), quote=True)
    scope = manifest.get("dataset_scope", manifest.get("scope", "给定 sample_manifest.json 的样本范围"))
    if not isinstance(scope, str):
        scope = json.dumps(scope, ensure_ascii=False)
    cards = []
    for i, row in enumerate(rows):
        selected_list = "；".join(f"#{p['display_index']}={p['score']:.3f}" for p in row["selected_predictions"])
        cards.append(f'''<article><h2>{i + 1:02d} · {esc(row['image_id'])}</h2>
<p>预测 {row['prediction_count']} 个 · TP {row['tp']} · FP {row['fp']} · FN {row['fn']}</p>
<a href="{esc(row['prediction_image'])}" target="_blank"><img loading="lazy" src="{esc(row['prediction_image'])}" alt="{esc(row['image_id'])} 的纯模型预测"></a>
<p class="caption">原尺寸 {row['width']} × {row['height']}；点击图片放大。蓝框与编号/置信度只来自预测。</p>
<p class="scores">{esc(selected_list or '没有预测超过该阈值。')}</p>
<details><summary>查看原图 | 模型预测 | GT 对照</summary>
<a href="{esc(row['comparison_image'])}" target="_blank"><img loading="lazy" src="{esc(row['comparison_image'])}" alt="原图、纯预测与GT匹配对照"></a>
<p>第三列中绿/红均为模型框，橙色虚线为漏检 GT；按冻结的一对一 IoU≥0.5 规则判读。</p></details></article>''')
    metrics = summary["sample_metrics"]
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>随机图片识别效果</title>
<style>body{{margin:0;background:#edf2f9;color:#18263b;font:16px/1.7 "Microsoft YaHei",sans-serif}}main{{max-width:1520px;margin:auto;padding:28px}}header,article{{background:white;border:1px solid #dce4ef;border-radius:14px;padding:24px;margin-bottom:24px}}h1{{margin:0 0 12px;font-size:28px}}h2{{font-size:21px;margin:0}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}}article{{min-width:0;margin:0}}img{{display:block;max-width:100%;height:auto;border-radius:7px}}a{{color:#1364be}}summary{{cursor:pointer;font-weight:bold}}.caption,.scores{{font-size:14px;color:#516077}}.scores{{overflow-wrap:anywhere}}.legend span{{padding:4px 10px;border-radius:4px;margin-right:8px;display:inline-block}}.blue{{background:#d9ebff}}.green{{background:#d6f8e3}}.red{{background:#ffe1e2}}.orange{{background:#ffedce}}details{{margin-top:15px}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}main{{padding:14px}}}}</style>
<main><header><h1>{esc(manifest['model_label'])} · 随机图片识别效果</h1>
<p>范围：{esc(scope)}；本次 {len(rows)} 张；抽样清单 seed＝{esc(manifest['seed'])}；confidence≥{manifest['confidence']:g}；匹配 IoU≥0.5。</p>
<p>全部展示给定随机清单，不按检测效果挑图或隐藏失败样本。此图廊不重新抽样。随机小样本不能替代完整 VAL、独立测试或真实采摘验证。</p>
<p>样本合计 TP {metrics['tp']} / FP {metrics['fp']} / FN {metrics['fn']}。这些计数只描述本次样本，不能作为全 VAL 的精度结论。</p>
<p class="legend"><span class="blue">蓝：纯模型预测</span><span class="green">绿：预测 TP</span><span class="red">红：预测 FP</span><span class="orange">橙色虚线：GT 漏检 FN</span></p>
<p>TP/FP/FN 由现有 GT 的一对一匹配判定，未逐图人工确认标注是否完整。纯预测蓝框图不使用 GT 决定显示内容。</p>
<a href="contact_sheet.jpg" target="_blank"><img src="contact_sheet.jpg" alt="全部随机样本预测总览"></a>
<p><a href="summary.json">查看结构化结果 summary.json</a> · 点击总览或每张图可打开原尺寸文件。</p></header>
<section class="grid">{''.join(cards)}</section></main></html>'''
    (output / "gallery.html").write_text(content, encoding="utf-8")


def render(input_dir: Path, output_dir: Path | None = None, requested_font=None):
    input_dir = input_dir.resolve()
    output = (output_dir or input_dir).resolve()
    manifest_path = input_dir / "sample_manifest.json"
    prediction_path = input_dir / "fused_predictions.json"
    manifest = read_json(manifest_path)
    confidence = float(manifest["confidence"])
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid manifest confidence")
    if not manifest.get("model_label") or "seed" not in manifest:
        raise ValueError("Manifest needs model_label and seed")
    samples = manifest["samples"]
    if not samples or len({x["image_id"] for x in samples}) != len(samples):
        raise ValueError("Samples must be nonempty with unique image_id")
    if len({x["index"] for x in samples}) != len(samples):
        raise ValueError("Sample indices must be unique")
    records = read_json(prediction_path)
    if not isinstance(records, list) or any(not r.get("image_id") for r in records):
        raise ValueError("Predictions must be metrics records with nonempty image_id")
    if len({x["image_id"] for x in records}) != len(records):
        raise ValueError("Duplicate prediction record image_id")
    record_by_id = {r["image_id"]: r for r in records}
    if any(s["image_id"] not in record_by_id for s in samples):
        raise ValueError("A manifest sample is missing its prediction record")
    metrics, metrics_path = load_metrics()
    face_path = font_path(requested_font)
    selected_records = []
    # Validate all inputs and guard outputs before writing any display artifacts.
    for sample in samples:
        if not sample["image_id"] or Path(sample["image_id"]).name != sample["image_id"]:
            raise ValueError("image_id must be a plain filename")
        record = record_by_id[sample["image_id"]]
        metrics._normalize_record(record)
        rgb = Path(sample["rgb_path"])
        if not rgb.is_absolute():
            rgb = input_dir / rgb
        with Image.open(rgb) as original:
            if original.size != (int(sample["width"]), int(sample["height"])):
                raise ValueError("RGB geometry differs from manifest: " + sample["image_id"])
        selected_records.append(record)
    targets = [output / name for name in ["summary.json", "gallery.html", "contact_sheet.jpg"]]
    if any(p.exists() for p in targets) or (output / "images").exists():
        raise FileExistsError("Display output exists; use a new --output-dir instead of overwriting")
    (output / "images/predictions").mkdir(parents=True)
    (output / "images/comparisons").mkdir(parents=True)
    rows = []
    for position, (sample, record) in enumerate(zip(samples, selected_records), 1):
        rgb = Path(sample["rgb_path"])
        if not rgb.is_absolute():
            rgb = input_dir / rgb
        with Image.open(rgb) as original:
            original = original.convert("RGB")
        boxes = record["pred"]["boxes"]
        scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
        order = np.argsort(-scores, kind="stable")
        order = order[scores[order] >= confidence]
        # This list and all-blue display depend solely on prediction fields.
        selected = [{"display_index": i + 1, "prediction_index": int(pi),
                     "score": float(scores[pi]), "box": list(boxes[pi])}
                    for i, pi in enumerate(order)]
        pure, fallbacks = overlay_predictions(original, selected, face_path)
        matching = metrics.match_image(record["gt"]["boxes"], boxes, scores, .5, confidence)
        assert np.array_equal(order, matching["pred_indices"])
        flags = matching["tp"]
        missed = [{"gt_index": int(gi), "box": record["gt"]["boxes"][gi]}
                  for gi in matching["unmatched_gt_indices"]]
        compared, _ = overlay_predictions(original, selected, face_path,
            colors=[GREEN if flag else RED for flag in flags],
            suffixes=[" TP" if flag else " FP" for flag in flags], missed_gt=missed)
        name = f"{position:02d}_{Path(sample['image_id']).stem}.png"
        pred_rel = "images/predictions/" + name
        comp_rel = "images/comparisons/" + name
        pure.save(output / pred_rel)
        row = {"manifest_index": sample["index"], "image_id": sample["image_id"],
               "rgb_path": str(rgb.resolve()), "width": original.width, "height": original.height,
               "gt_count": len(record["gt"]["boxes"]), "prediction_count": len(selected),
               "tp": int(flags.sum()), "fp": int((~flags).sum()), "fn": len(missed),
               "prediction_image": pred_rel, "comparison_image": comp_rel,
               "prediction_label_overlap_fallbacks": fallbacks,
               "selected_predictions": selected,
               "matched_gt_indices_in_display_order": matching["gt_indices"].tolist(),
               "unmatched_gt_indices": matching["unmatched_gt_indices"].tolist()}
        panel = comparison_panel(original, pure, compared, row, face_path)
        panel.save(output / comp_rel)
        row["prediction_image_sha256"] = file_sha256(output / pred_rel)
        row["comparison_image_sha256"] = file_sha256(output / comp_rel)
        rows.append(row)
    subset = metrics.evaluate_at_threshold(selected_records, confidence, .5)
    assert (sum(r["tp"] for r in rows), sum(r["fp"] for r in rows), sum(r["fn"] for r in rows)) == (subset["tp"], subset["fp"], subset["fn"])
    contact_sheet(rows, output, face_path, confidence, manifest["model_label"])
    summary = {"schema": "random_strong_prediction_gallery_v1", "seed": manifest["seed"],
               "model_label": manifest["model_label"], "confidence": confidence,
               "matching_IoU": .5, "samples": rows, "sample_count": len(rows),
               "sample_metrics": subset,
               "scope": "Only all samples in the provided manifest; no result-based filtering; not full-VAL/test/deployment metrics",
               "prediction_display_uses_GT": False,
               "GT_comparison_is_separate": True, "original_RGB_modified": False,
               "GPU_used": False, "model_checkpoint_loaded": False,
               "metrics_path": str(metrics_path), "metrics_sha256": METRICS_SHA256,
               "font_path": str(face_path), "manifest_sha256": file_sha256(manifest_path),
               "prediction_records_sha256": file_sha256(prediction_path),
               "contact_sheet": "contact_sheet.jpg", "gallery": "gallery.html",
               "label_placement": "Labels avoid selected prediction boxes, without GT; least-overlap fallback counts are recorded when no free rectangle exists"}
    write_gallery(manifest, rows, summary, output)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert "torch" not in sys.modules, "CPU renderer unexpectedly imported torch"
    print(json.dumps({"summary": str(output / "summary.json"),
                      "gallery": str(output / "gallery.html"), "samples": len(rows),
                      "TP": subset["tp"], "FP": subset["fp"], "FN": subset["fn"]}, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--font", help="Local Chinese TTF/TTC; default Windows Microsoft YaHei")
    args = parser.parse_args()
    render(args.input_dir, args.output_dir, args.font)


if __name__ == "__main__":
    main()
