"""Create a deterministic, UNLABELED fruit-review queue from local train originals.

Only LOCAL_apple/LOCAL_shuju original training images are opened. All crops and
reports are written to a new output directory; no dataset files are changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

from PIL import Image


DEFAULT_ROOT = Path(r"F:\研究生毕业设计\模块一定位补偿以及预测")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def candidate_groups(manifest, root):
    groups = defaultdict(list)
    for row_index, row in enumerate(manifest):
        if (row.get("split") != "train"
                or row.get("source") not in {"LOCAL_apple", "LOCAL_shuju"}
                or row.get("augmentation") != "none"):
            continue
        if row.get("original_split") != "train":
            raise ValueError("Selected row is not an original training image")
        path = Path(row["original_file"]).resolve()
        folder = "apple" if row["source"] == "LOCAL_apple" else "shuju"
        allowed = (root / "apple" / folder / "images" / "train").resolve()
        if allowed not in path.parents:
            raise ValueError(f"Source path not in approved train-only root: {path}")
        if row["group"] is None:
            raise ValueError("Missing source group")
        if len(row["boxes"]) != row["instances"]:
            raise ValueError("Manifest instance count mismatch")
        valid = []
        for bbox_index, box in enumerate(row["boxes"]):
            if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
                raise ValueError("Invalid box")
            x1, y1, x2, y2 = map(float, box)
            if not (0 <= x1 < x2 <= row["width"] + 0.01 and 0 <= y1 < y2 <= row["height"] + 0.01):
                raise ValueError("Out-of-bounds source box")
            valid.append(bbox_index)
        if valid:
            groups[(row["source"], row["group"])].append((row_index, row, valid))
    return groups


def select_candidates(groups, seed, limit, per_group_cap):
    rng = random.Random(seed)
    sources = sorted({key[0] for key in groups})
    source_groups = {source: sorted(k for k in groups if k[0] == source) for source in sources}
    for keys in source_groups.values():
        rng.shuffle(keys)
    # Each first pass visits all groups before any second fruit is considered.
    # Source interleaving avoids making the second pass depend on source size.
    ordered_keys = []
    max_groups = max(map(len, source_groups.values()), default=0)
    for pos in range(max_groups):
        for source in sources:
            if pos < len(source_groups[source]):
                ordered_keys.append(source_groups[source][pos])
    chosen = []
    used_boxes = set()
    used_images_by_group = defaultdict(set)
    for pass_index in range(per_group_cap):
        for key in ordered_keys:
            if len(chosen) >= limit:
                return chosen
            candidates = sorted(groups[key], key=lambda item: item[1]["original_file"])
            remaining = [(idx, row, [b for b in boxes if (idx, b) not in used_boxes])
                         for idx, row, boxes in candidates]
            remaining = [item for item in remaining if item[2]]
            if not remaining:
                continue
            fresh_images = [item for item in remaining if item[0] not in used_images_by_group[key]]
            idx, row, boxes = rng.choice(fresh_images or remaining)
            bbox_index = rng.choice(boxes)
            chosen.append((idx, row, bbox_index, pass_index + 1))
            used_boxes.add((idx, bbox_index))
            used_images_by_group[key].add(idx)
    return chosen


def build_queue(root, output, seed=20261106, limit=60, max_per_group=2, preview_max_side=384):
    root, output = root.resolve(), output.resolve()
    if root == output or root in output.parents:
        raise ValueError("Output must be outside the source tree")
    if output.exists():
        raise FileExistsError("Output exists; use a new versioned directory")
    manifest_path = root / "apple_vision/reports/local_v2/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest_sha = sha256(manifest_path)
    groups = candidate_groups(manifest, root)
    selected = select_candidates(groups, seed, limit, max_per_group)
    if not selected:
        raise ValueError("No eligible original training boxes")
    output.mkdir(parents=True)
    crop_dir = output / "crops"
    crop_dir.mkdir()
    queue = []
    for number, (row_index, row, bbox_index, group_pass) in enumerate(selected, 1):
        path = Path(row["original_file"])
        image_sha = sha256(path)
        if row.get("sha256") and image_sha != row["sha256"]:
            raise ValueError(f"Source image hash mismatch: {path}")
        raw_box = list(map(float, row["boxes"][bbox_index]))
        x1, y1, x2, y2 = raw_box
        with Image.open(path) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError(f"Image dimensions differ from manifest: {path}")
            orientation = image.getexif().get(274, 1)
            if orientation != 1:
                raise ValueError(f"EXIF rotation needs annotation-aware review: {path}")
            # Show context without changing or relabeling the supplied box.
            pad_x, pad_y = 0.10 * (x2 - x1), 0.10 * (y2 - y1)
            bounds = [max(0, math.floor(x1 - pad_x)), max(0, math.floor(y1 - pad_y)),
                      min(image.width, math.ceil(x2 + pad_x)), min(image.height, math.ceil(y2 + pad_y))]
            preview = image.convert("RGB").crop(tuple(bounds))
            native_crop_size = list(preview.size)
            preview.thumbnail((preview_max_side, preview_max_side), Image.Resampling.LANCZOS)
            item_id = f"local_review_{number:03d}"
            preview_path = crop_dir / f"{item_id}.jpg"
            preview.save(preview_path, format="JPEG", quality=95, subsampling=0)
            preview_size = list(preview.size)
        # The original YOLO label is identified for provenance only. Selection
        # and box index refer precisely to the frozen prepared manifest.
        upstream_label = path.parent.parent.parent / "labels" / "train" / (path.stem + ".txt")
        queue.append({
            "item_id": item_id,
            "original_path": str(path), "image_sha256": image_sha,
            "original_dimensions_wh": [row["width"], row["height"]],
            "box_xyxy_original_pixels": raw_box,
            "crop_bounds_xyxy_original_pixels": bounds,
            "preview_path": str(preview_path), "preview_relative_path": f"crops/{item_id}.jpg",
            "preview_sha256": sha256(preview_path), "preview_dimensions_wh": preview_size,
            "native_crop_dimensions_wh": native_crop_size,
            "preview_resize_purpose": "Human review only. Original bbox is unchanged; preview is not new geometric truth.",
            "source": row["source"], "split": "train", "augmentation": "none",
            "source_group": row["group"], "group_selection_pass": group_pass,
            "annotation_source": {"manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
                                  "manifest_row_index_zero_based": row_index,
                                  "bbox_index_zero_based": bbox_index,
                                  "upstream_label_path": str(upstream_label),
                                  "upstream_label_sha256": sha256(upstream_label),
                                  "existing_annotation": "single-class apple bounding box only"},
            "maturity": None, "visible_defects": None, "grade": None, "fruitID": None,
            "cultivar": None, "human_scene_id": None, "annotator": None,
            "annotation_protocol_version": None, "annotation_status": "pending_human_review",
            "fruit_identity_status": "unknown; same fruit may occur in different views or groups",
            "source_group_limitation": "Stored EXIF time or filename-sequence grouping is a proxy, not a verified fruit/tree identity.",
            "preview_assessability": None, "human_notes": None,
        })
    queue_path = output / "annotation_queue_v1.jsonl"
    queue_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in queue), encoding="utf-8")
    summary = {
        "schema": "local_unlabeled_attribute_queue_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "requested_limit": limit, "max_per_group": max_per_group,
        "selection": "All stored groups before second fruit per group; source-interleaved; random image then random bbox; prefer different image on second pass. No appearance/color-based selection or labeling.",
        "items": len(queue), "source_counts": dict(Counter(r["source"] for r in queue)),
        "distinct_source_groups": len({(r["source"], r["source_group"]) for r in queue}),
        "distinct_original_images": len({r["image_sha256"] for r in queue}),
        "group_counts": dict(Counter(r["source_group"] for r in queue)),
        "eligible_group_counts": dict(Counter(k[0] for k in groups)),
        "raw_test_or_val_images_labels_opened": False, "source_modified": False, "GPU_used": False,
        "mixed_manifest_read": True, "test_and_val_rows_used_for_selection": False,
        "attribute_labels_generated": False,
        "queue_path": str(queue_path), "queue_sha256": sha256(queue_path),
        "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
        "script_path": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
        "limitations": [
            "60 image views are not 60 verified unique fruits; human fruit/scene IDs remain null.",
            "Small/occluded crops may be unassessable. Do not infer normal quality or maturity from missing visible evidence.",
            "Maturity requires a defined cultivar-specific labeling/measurement protocol; RGB appearance alone does not provide physical maturity truth.",
            "Only training originals are present, so this queue supplies no validation/test classification claims.",
            "This is a versioned annotation starting point, not a trained classifier or an automatically labeled dataset.",
        ],
    }
    (output / "queue_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    instructions = (
        "待人工标注队列：全部来自 LOCAL_apple/LOCAL_shuju 的原始训练图。\n"
        "成熟度、可见缺陷、等级及 fruitID 均为空；不要把空值当作健康、成熟或无缺陷。\n"
        "先约定品种与成熟度标注/实测协议，再填写 maturity；不能单凭红绿颜色冒充成熟度真值。\n"
        "visible_defects 仅描述当前可见表面，遮挡、模糊和分辨率不足填写不可判定原因。\n"
        "人工核对重复果实/场景，填写 fruitID 与 human_scene_id；source_group 只是时间或编号代理。\n"
        "预览保留10%上下文并缩小到最长边384像素，原坐标、原路径、SHA和框索引保存在JSONL。\n"
        "需要精细缺陷判定时查看对应原图；缩小预览不能代表全部果面。\n"
        "这些样本仅为训练标注起点；不应重新随机拆分裁片得到验证或测试成绩。\n"
    )
    (output / "ANNOTATION_INSTRUCTIONS.txt").write_text(instructions, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data/annotation_queue_v1")
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--max-per-group", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.limit <= 60 or not 1 <= args.max_per_group <= 2:
        parser.error("This review queue is limited to 60 items and at most 2 per stored group")
    summary = build_queue(args.root, args.output, args.seed, args.limit, args.max_per_group)
    print(json.dumps({k: summary[k] for k in ("queue_path", "items", "source_counts", "distinct_source_groups", "distinct_original_images", "queue_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
