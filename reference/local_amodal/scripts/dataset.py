"""PApple VIA modal/amodal data loader; audit mode requires only Pillow/numpy.

Dataset tensors are created lazily in __getitem__, which is for the remote GPU
computer. This module never imports torch merely to inspect annotations.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


class AnnotationError(ValueError):
    """Annotations cannot safely be converted without a human/data decision."""


def resolve_data_root(root):
    root = Path(root).expanduser().resolve()
    if (root / "gt_json").is_dir() and (root / "images").is_dir():
        return root
    if (root / "data" / "gt_json").is_dir():
        return root / "data"
    raise FileNotFoundError(f"Expected images/ and gt_json/ under {root} or {root / 'data'}")


def _json_images(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        raw = json.load(stream)
    if isinstance(raw, dict) and "_via_img_metadata" in raw:
        raw = raw["_via_img_metadata"]
    values = raw.values() if isinstance(raw, dict) else raw
    records = {}
    for value in values:
        if not isinstance(value, dict) or "filename" not in value:
            raise AnnotationError(f"Unexpected VIA image record in {path}")
        filename = str(value["filename"])
        if Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise AnnotationError(f"Expected a simple image filename, got {filename!r}")
        if filename in records:
            raise AnnotationError(f"Duplicate image record: {filename} in {path}")
        records[filename] = value
    return records


def _regions(record):
    regions = record.get("regions", {})
    if not isinstance(regions, (dict, list)):
        raise AnnotationError("VIA regions must be a list or dictionary")
    return regions.values() if isinstance(regions, dict) else regions


def _parse_instances(record, filename, branch, issues):
    instances = defaultdict(list)
    for number, region in enumerate(_regions(record)):
        attrs = region.get("region_attributes", {})
        apple_id = attrs.get("apple_ID")
        if apple_id is None or str(apple_id).strip() == "":
            issues.append({"file": filename, "branch": branch, "region": number,
                           "code": "missing_apple_id"})
            continue
        apple_id = str(apple_id).strip()
        shape = region.get("shape_attributes", {})
        xs, ys = shape.get("all_points_x", []), shape.get("all_points_y", [])
        try:
            xy = np.asarray(list(zip(xs, ys)), dtype=np.float64)
            valid = (len(xs) == len(ys) and xy.ndim == 2 and xy.shape[1] == 2
                     and len(xy) >= 3 and np.isfinite(xy).all())
            if valid:
                # Polygon shoelace area, before rasterization/clipping.
                area2 = abs(float(np.dot(xy[:, 0], np.roll(xy[:, 1], 1))
                                  - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))))
                valid = area2 > 0
        except (ValueError, TypeError):
            valid = False
        if not valid:
            issues.append({"file": filename, "branch": branch, "apple_ID": apple_id,
                           "region": number, "code": "invalid_polygon"})
            continue
        instances[apple_id].append(xy.astype(np.float32))
    return dict(instances)


def _load_split(root, split):
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    gt_dir = root / "gt_json" / split
    modal = _json_images(gt_dir / "via_region_data_instance.json")
    amodal = _json_images(gt_dir / "via_region_data_amodal.json")
    image_dir = root / "images" / split
    files = {p.name: p for p in image_dir.iterdir()
             if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}}
    issues, samples = [], []
    for filename in sorted(set(modal) | set(amodal) | set(files)):
        if filename not in modal or filename not in amodal:
            issues.append({"file": filename, "code": "missing_annotation_record",
                           "missing_modal": filename not in modal,
                           "missing_amodal": filename not in amodal})
        if filename not in files:
            issues.append({"file": filename, "code": "missing_image"})
            continue
        try:
            with Image.open(files[filename]) as image:
                width, height = image.size
        except (OSError, ValueError) as exc:
            issues.append({"file": filename, "code": "unreadable_image", "detail": str(exc)})
            continue
        visible = _parse_instances(modal.get(filename, {}), filename, "visible", issues)
        complete = _parse_instances(amodal.get(filename, {}), filename, "amodal", issues)
        if set(visible) != set(complete):
            issues.append({"file": filename, "code": "unpaired_instance_ids",
                           "visible_only": sorted(set(visible) - set(complete)),
                           "amodal_only": sorted(set(complete) - set(visible))})
        if "0" in visible or "0" in complete:
            # The author's loader deletes this ID without specifying its semantics.
            # Neither silently train it as apple nor silently turn it into background.
            issues.append({"file": filename, "code": "unresolved_apple_id_zero",
                           "detail": "Author removes ID 0; meaning must be checked before training."})
        samples.append({"filename": filename, "path": files[filename], "width": width,
                        "height": height, "visible": visible, "amodal": complete,
                        "apple_ids": sorted(set(visible) & set(complete))})
    return samples, issues


def _rasterize(polygons, width, height):
    canvas = Image.new("L", (width, height), 0)
    drawing = ImageDraw.Draw(canvas)
    for polygon in polygons:
        # Each disconnected visible component is drawn separately, never connected.
        drawing.polygon([tuple(map(float, point)) for point in polygon], fill=1)
    return np.asarray(canvas, dtype=np.uint8).copy()


def _numpy_target(sample, image_id):
    height, width = sample["height"], sample["width"]
    count = len(sample["apple_ids"])
    masks = np.zeros((count, height, width), dtype=np.uint8)
    visible = np.zeros_like(masks)
    boxes = np.zeros((count, 4), dtype=np.float32)
    for i, apple_id in enumerate(sample["apple_ids"]):
        masks[i] = _rasterize(sample["amodal"][apple_id], width, height)
        visible[i] = _rasterize(sample["visible"][apple_id], width, height)
        ys, xs = np.nonzero(masks[i])
        if not len(xs) or not np.any(visible[i]):
            raise AnnotationError(f"Empty rasterized mask: {sample['filename']}, ID {apple_id}")
        # torchvision boxes use continuous exclusive right/bottom coordinates.
        boxes[i] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    return {"boxes": boxes, "labels": np.ones(count, dtype=np.int64),
            "masks": masks, "visible_masks": visible,
            "image_id": np.asarray([image_id], dtype=np.int64),
            "area": masks.sum(axis=(1, 2), dtype=np.float64).astype(np.float32),
            "iscrowd": np.zeros(count, dtype=np.int64)}


class PAppleDataset:
    def __init__(self, root, split, max_images=None, augment=False):
        self.root = resolve_data_root(root)
        self.split = split
        self.augment = bool(augment)
        samples, issues = _load_split(self.root, split)
        if issues:
            counts = dict(Counter(item["code"] for item in issues))
            raise AnnotationError(f"Unsafe annotations for {split}: {counts}. Run dataset.py audit; "
                                  f"first issues: {issues[:3]}")
        if max_images is not None:
            if int(max_images) < 1:
                raise ValueError("max_images must be positive")
            samples = samples[:int(max_images)]
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def load_sample(self, index):
        """Return RGB uint8 HWC image and a numpy target, with no augmentation."""
        sample = self.samples[index]
        with Image.open(sample["path"]) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        return image, _numpy_target(sample, index)

    def __getitem__(self, index):
        # This path is only used on the explicitly authorized training computer.
        import torch
        image, target = self.load_sample(index)
        if self.augment and torch.rand(()).item() < 0.5:
            width = image.shape[1]
            image = np.ascontiguousarray(image[:, ::-1])
            target["masks"] = np.ascontiguousarray(target["masks"][:, :, ::-1])
            target["visible_masks"] = np.ascontiguousarray(target["visible_masks"][:, :, ::-1])
            old = target["boxes"].copy()
            target["boxes"][:, 0] = width - old[:, 2]
            target["boxes"][:, 2] = width - old[:, 0]
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)
        return tensor, {key: torch.from_numpy(value) for key, value in target.items()}


def _capture_year(filename):
    explicit = re.search(r"(?:^|[^0-9])(20[0-9]{2})(?:[^0-9]|$)", filename)
    if explicit:
        return explicit.group(1)
    parts = Path(filename).stem.split("_")
    if len(parts) > 2 and parts[2]:
        if parts[2][0] in "67":
            return "2020_author_filename_rule"
        if parts[2][0] in "23":
            return "2018_author_filename_rule"
    return "unknown_year"


def audit_dataset(root, check_masks=False):
    """Pure CPU audit. Reports risk; never silently changes official splits."""
    root = resolve_data_root(root)
    report = {"root": str(root), "splits": {}, "cross_split": {},
              "split_policy": "Official random split retained for exploratory reproduction.",
              "independence_verified": False,
              "id_zero_policy": "Unresolved ID 0 blocks training; no automatic deletion.",
              "notes": ["Author reports >75% overlapping original photos and random patch split.",
                        "Filename-prefix and ID overlaps are risks, not proof of a shared physical object.",
                        "No model import or model inference is performed by this audit."]}
    all_names, all_ids, all_prefixes = {}, {}, {}
    for split in ("train", "val", "test"):
        samples, issues = _load_split(root, split)
        all_names[split] = {sample["filename"] for sample in samples}
        all_ids[split] = {(_capture_year(sample["filename"]), apple_id)
                          for sample in samples for apple_id in sample["apple_ids"] if apple_id != "0"}
        all_prefixes[split] = {Path(sample["filename"]).stem.rsplit("_", 1)[0]
                               for sample in samples if "_" in Path(sample["filename"]).stem}
        raster_stats = {"images_checked": 0, "visible_pixels_outside_amodal": 0,
                        "instances_with_visible_outside_amodal": 0}
        if check_masks:
            for index, sample in enumerate(samples):
                try:
                    target = _numpy_target(sample, index)
                    outside = target["visible_masks"].astype(bool) & ~target["masks"].astype(bool)
                    raster_stats["images_checked"] += 1
                    raster_stats["visible_pixels_outside_amodal"] += int(outside.sum())
                    raster_stats["instances_with_visible_outside_amodal"] += int(
                        np.any(outside, axis=(1, 2)).sum())
                except AnnotationError as exc:
                    issues.append({"file": sample["filename"], "code": "raster_error", "detail": str(exc)})
        report["splits"][split] = {
            "images": len(samples), "paired_instances": sum(len(s["apple_ids"]) for s in samples),
            "empty_gt_images": sum(not s["visible"] and not s["amodal"] for s in samples),
            "visible_regions": sum(sum(len(p) for p in s["visible"].values()) for s in samples),
            "amodal_regions": sum(sum(len(p) for p in s["amodal"].values()) for s in samples),
            "dimensions": dict(Counter(f"{s['width']}x{s['height']}" for s in samples)),
            "issues_by_code": dict(Counter(item["code"] for item in issues)),
            "issues": issues, "safe_to_train": not issues,
            "mask_rasterization": raster_stats if check_masks else "not_requested"}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        names = sorted(all_names[left] & all_names[right])
        ids = sorted(all_ids[left] & all_ids[right])
        prefixes = sorted(all_prefixes[left] & all_prefixes[right])
        report["cross_split"][f"{left}__{right}"] = {
            "same_filename_count": len(names), "same_filenames": names,
            "year_and_apple_id_overlap_count": len(ids), "year_and_apple_ids": [list(v) for v in ids],
            "heuristic_filename_prefix_overlap_count": len(prefixes),
            "heuristic_filename_prefixes": prefixes,
            "prefix_rule": "stem before last underscore; not confirmed source-photo identity"}
    report["safe_to_train"] = all(v["safe_to_train"] for v in report["splits"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit PApple annotations without importing torch")
    parser.add_argument("command", choices=["audit"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-masks", action="store_true")
    args = parser.parse_args()
    result = audit_dataset(args.root, args.check_masks)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "safe_to_train": result["safe_to_train"],
                      "splits": {name: {key: stats[key] for key in
                                 ("images", "paired_instances", "empty_gt_images", "issues_by_code")}
                                 for name, stats in result["splits"].items()}}, ensure_ascii=False))
    return 0 if result["safe_to_train"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
