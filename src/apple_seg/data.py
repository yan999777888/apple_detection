"""Portable genuine PApple V/A and MinneApple visible-only CPU datasets.

Paths are supplied by callers. No model, checkpoint, old temp module, CUDA,
annotation repair, pseudo amodal target or target device transfer is used here.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw
import torch

MINNE_TILES = tuple((x, y, x + 512, y + 512)
                   for y in (0, 384, 768) for x in (0, 208))
NORMALIZATION_MEAN = (.485, .456, .406)
NORMALIZATION_STD = (.229, .224, .225)


class DataContractError(ValueError):
    pass


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _filename(value):
    name = str(value)
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        raise DataContractError(f"Expected a simple source filename: {name!r}")
    return name


def _input_size(value):
    if type(value) is not int or value < 1:
        raise ValueError("input_size must be a positive integer")
    return value


def _pixels(image, input_size):
    # Whole supplied view, no external padding. RGB normalization is identical
    # for PApple frames and genuine Minne visible fragments.
    resized = image.convert("RGB").resize((input_size, input_size), Image.Resampling.BILINEAR)
    value = np.asarray(resized, dtype=np.float32) / np.float32(255.)
    value = (value - np.asarray(NORMALIZATION_MEAN, np.float32)) / np.asarray(NORMALIZATION_STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1)))


def _item(image, visible, amodal, ids, source, input_size, augment):
    height, width = visible.shape[-2:]
    target = {"labels": torch.zeros(len(ids), dtype=torch.int64),
              "visible_masks": torch.from_numpy(np.ascontiguousarray(visible, dtype=np.uint8)),
              "amodal_masks": None if amodal is None else torch.from_numpy(np.ascontiguousarray(amodal, dtype=np.uint8)),
              "annotation_validity": {"visible": True, "amodal": amodal is not None},
              "source": {**source, "horizontal_flip": False,
                         "resize_hw": [input_size, input_size],
                         "original_to_input_scale_xy": [input_size / width, input_size / height]},
              "instance_ids": list(ids), "original_hw": [height, width]}
    result = {"pixel_values": _pixels(image, input_size),
              "pixel_mask": torch.ones((input_size, input_size), dtype=torch.bool), "target": target}
    return flip_item_horizontal(result) if augment and random.random() < .5 else result


def flip_item_horizontal(item):
    """One exact CPU transform; applying twice restores pixels, GT and flag."""
    target = {**item["target"], "source": {**item["target"]["source"]}}
    target["source"]["horizontal_flip"] = not target["source"].get("horizontal_flip", False)
    for key in ("visible_masks", "amodal_masks"):
        target[key] = None if target[key] is None else target[key].flip(-1).contiguous()
    return {"pixel_values": item["pixel_values"].flip(-1).contiguous(),
            "pixel_mask": item["pixel_mask"].flip(-1).contiguous(), "target": target}


def collate_single(items):
    if len(items) != 1:
        raise ValueError("This data contract requires batch_size=1")
    item = items[0]
    return {"pixel_values": item["pixel_values"].unsqueeze(0),
            "pixel_mask": item["pixel_mask"].unsqueeze(0), "targets": [item["target"]]}


def _via_images(path):
    raw = _read_json(path)
    if isinstance(raw, dict) and "_via_img_metadata" in raw:
        raw = raw["_via_img_metadata"]
    records = {}
    for value in (raw.values() if isinstance(raw, dict) else raw):
        if not isinstance(value, dict) or "filename" not in value:
            raise DataContractError(f"Invalid VIA image record: {path}")
        name = _filename(value["filename"])
        if name in records:
            raise DataContractError(f"Duplicate VIA filename: {name}")
        records[name] = value
    return records


def _polygons(record):
    result = defaultdict(list)
    regions = record.get("regions", {})
    for region in (regions.values() if isinstance(regions, dict) else regions):
        apple_id = region.get("region_attributes", {}).get("apple_ID")
        if apple_id is None or not str(apple_id).strip():
            raise DataContractError("Missing apple_ID: no GT region may be silently discarded")
        shape = region.get("shape_attributes", {})
        xs, ys = shape.get("all_points_x", []), shape.get("all_points_y", [])
        xy = np.asarray(list(zip(xs, ys)), dtype=np.float64)
        if len(xs) != len(ys) or xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
            raise DataContractError("Invalid original VIA polygon")
        area2 = abs(float(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))))
        if area2 <= 0:
            raise DataContractError("Zero-area original VIA polygon")
        result[str(apple_id).strip()].append(xy.astype(np.float32))
    return dict(result)


def _rasterize(polygons, width, height):
    canvas = Image.new("L", (width, height), 0)
    drawing = ImageDraw.Draw(canvas)
    for polygon in polygons:
        drawing.polygon([tuple(map(float, point)) for point in polygon], fill=1)
    return np.asarray(canvas, dtype=np.uint8).copy()


class PAppleDataset:
    def __init__(self, data_root, split="train", input_size=768, augment=False):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        root = Path(data_root).expanduser().resolve()
        if not (root / "gt_json").is_dir() and (root / "data/gt_json").is_dir():
            root = root / "data"
        self.root, self.split = root, split
        self.input_size, self.augment = _input_size(input_size), bool(augment)
        self.configuration = {"input_size": self.input_size, "input_hw": [self.input_size] * 2,
            "resize_hw": [self.input_size] * 2, "resize": "Pillow RGB bilinear whole supplied frame",
            "rgb_scale": "uint8/255", "normalization_mean": list(NORMALIZATION_MEAN),
            "normalization_std": list(NORMALIZATION_STD), "augment": self.augment,
            "padding": False, "external_padding": False, "crop": False}
        if split != "train" and self.augment:
            raise ValueError("Evaluation data cannot be augmented")
        self.annotation_paths = {"visible": root / "gt_json" / split / "via_region_data_instance.json",
                                 "amodal": root / "gt_json" / split / "via_region_data_amodal.json"}
        self.annotation_sha256 = {key: sha256_file(path) for key, path in self.annotation_paths.items()}
        visible, amodal = (_via_images(self.annotation_paths[key]) for key in ("visible", "amodal"))
        image_dir = root / "images" / split
        files = {path.name: path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in
                 (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")}
        names = sorted(set(visible) | set(amodal) | set(files))
        self.samples = []
        for name in names:
            if name not in files or name not in visible or name not in amodal:
                raise DataContractError(f"Missing original image/VIA branch record: {name}")
            v, a = _polygons(visible[name]), _polygons(amodal[name])
            # Union is retained in inventory. Unpaired branches are diagnosed
            # when read, never intersected away or replaced by fake zero GT.
            self.samples.append({"filename": name, "rgb_path": files[name], "visible": v, "amodal": a,
                                 "instance_ids": sorted(set(v) | set(a))})

    def __len__(self):
        return len(self.samples)

    def metadata(self):
        return {"dataset": "PApple", "split": self.split, "items": len(self),
                "original_images": len(self), "annotation_paths": {k: str(v) for k, v in self.annotation_paths.items()},
                "annotation_sha256": dict(self.annotation_sha256), "input_size": self.input_size,
                "augment": self.augment, "GT_ID_union_retained": True, "V_intersected_with_A": False,
                "GT_instances": sum(len(row["instance_ids"]) for row in self.samples),
                "empty_GT_images": sum(not row["instance_ids"] for row in self.samples)}

    def __getitem__(self, index):
        sample = self.samples[index]
        ids = sample["instance_ids"]
        if set(sample["visible"]) != set(sample["amodal"]):
            raise DataContractError(f"Unpaired per-instance V/A requires explicit validity support; union={ids}; no GT dropped or fabricated")
        with Image.open(sample["rgb_path"]) as source:
            image = source.convert("RGB")
        width, height = image.size
        visible = np.empty((len(ids), height, width), dtype=np.uint8)
        amodal = np.empty_like(visible)
        for row, instance_id in enumerate(ids):
            visible[row] = _rasterize(sample["visible"][instance_id], width, height)
            amodal[row] = _rasterize(sample["amodal"][instance_id], width, height)
            if not visible[row].any() or not amodal[row].any():
                raise DataContractError(f"Empty original polygon raster: {sample['filename']}/{instance_id}")
        source = {"dataset": "PApple", "split": self.split, "image_id": sample["filename"],
                  "original_image_id": sample["filename"], "rgb_path": str(sample["rgb_path"]),
                  "crop_xyxy": [0, 0, width, height], "annotation_sha256": dict(self.annotation_sha256)}
        return _item(image, visible, amodal, ids, source, self.input_size, self.augment)


class MinneAppleDataset:
    def __init__(self, raw_root, split_manifest, split="train", input_size=768, tiled=True, augment=False):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        if split != "train" and (tiled or augment):
            raise ValueError("Minne validation/test require tiled=False and augment=False; inference tiles are separate")
        self.root = Path(raw_root).expanduser().resolve()
        self.split_manifest = Path(split_manifest).expanduser().resolve()
        self.split_manifest_sha256 = sha256_file(self.split_manifest)
        self.split, self.input_size, self.tiled, self.augment = split, _input_size(input_size), bool(tiled), bool(augment)
        self.configuration = {"input_size": self.input_size, "input_hw": [self.input_size] * 2,
            "resize_hw": [self.input_size] * 2, "resize": "Pillow RGB bilinear whole supplied view",
            "rgb_scale": "uint8/255", "normalization_mean": list(NORMALIZATION_MEAN),
            "normalization_std": list(NORMALIZATION_STD), "augment": self.augment,
            "padding": False, "external_padding": False, "crop": self.tiled,
            "fixed_tiles_xyxy": [list(box) for box in MINNE_TILES] if self.tiled else []}
        records = _read_json(self.split_manifest)
        if not isinstance(records, list):
            raise DataContractError("Expected existing Minne filename manifest JSON list")
        official = "test" if split == "test" else "train"
        self.samples = [deepcopy(row) for row in records if row.get("official_split") == official and row.get("split") == split]
        names = [_filename(row["source_filename"]) for row in self.samples]
        if not self.samples or len(set(names)) != len(names):
            raise DataContractError("Empty/duplicate selected Minne filename split")
        self.annotation_paths = {"visible": self.split_manifest, "amodal": None}
        self.annotation_sha256 = {"visible": self.split_manifest_sha256, "amodal": None}
        self._capacity_cache = None
        for row in self.samples:
            row["rgb_path"], row["mask_path"] = self._paths(row["source_filename"])
            if not row["rgb_path"].is_file() or not row["mask_path"].is_file():
                raise FileNotFoundError(f"Missing genuine Minne image/mask: {row['source_filename']}")

    def _paths(self, name):
        if self.split == "test":
            return self.root / "test_data/detection/images" / name, self.root / "test_data/segmentation/masks" / name
        return self.root / "detection/train/images" / name, self.root / "detection/train/masks" / name

    def __len__(self):
        return len(self.samples) * (len(MINNE_TILES) if self.tiled else 1)

    def _idmap(self, sample):
        digest = sha256_file(sample["mask_path"])
        expected = sample.get("mask_sha256")
        if expected is not None and digest != expected:
            raise DataContractError(f"Original mask hash differs from filename manifest: {sample['source_filename']}")
        with Image.open(sample["mask_path"]) as source:
            value = np.array(source, copy=True)
        if value.ndim != 2 or not np.issubdtype(value.dtype, np.integer) or int(value.min()) < 0:
            raise DataContractError("Expected original integer ID map; only value0 is background")
        if value.shape != (sample["height"], sample["width"]):
            raise DataContractError("Minne mask dimensions differ from original manifest")
        return value, digest

    def query_capacity_audit(self):
        if self._capacity_cache is None:
            maximum, over = 0, []
            for index, sample in enumerate(self.samples):
                idmap, _ = self._idmap(sample)
                crops = MINNE_TILES if self.tiled else ((0, 0, idmap.shape[1], idmap.shape[0]),)
                for tile, (x1, y1, x2, y2) in enumerate(crops):
                    count = int(np.count_nonzero(np.unique(idmap[y1:y2, x1:x2])))
                    maximum = max(maximum, count)
                    if count > 100:
                        over.append({"original_image_index": index, "filename": sample["source_filename"], "tile": tile, "GT": count})
            self._capacity_cache = {"max_GT": maximum, "over_capacity_items": over, "model_queries": 100,
                                    "formal_training_compatible": not over, "GT_truncated": False}
        return deepcopy(self._capacity_cache)

    def metadata(self):
        return {"dataset": "MinneApple", "split": self.split, "items": len(self), "original_images": len(self.samples),
                "input_size": self.input_size, "augment": self.augment, "tiled": self.tiled,
                "tiles": [list(box) for box in MINNE_TILES] if self.tiled else [],
                "annotation_paths": {"visible": str(self.split_manifest), "amodal": None},
                "annotation_sha256": dict(self.annotation_sha256), "split_manifest_sha256": self.split_manifest_sha256,
                "query_capacity_audit": self.query_capacity_audit(), "amodal_annotation": "unknown_None",
                "onepixel_and_boundary_instances_retained": True}

    def _read(self, original_index, crop=None, tile_index=None):
        sample = self.samples[original_index]
        idmap, digest = self._idmap(sample)
        original_h, original_w = idmap.shape
        if crop is not None and (original_h, original_w) != (1280, 720):
            raise DataContractError("Fixed Minne grid requires original H1280/W720, never GT-selected/adaptive tiles")
        x1, y1, x2, y2 = crop or (0, 0, original_w, original_h)
        view = idmap[y1:y2, x1:x2]
        ids = [int(value) for value in np.unique(view) if value != 0]
        if crop is not None and len(ids) > 100:
            raise DataContractError(f"Tile has {len(ids)} GT >100 queries; never truncate: {sample['source_filename']} tile{tile_index}")
        visible = np.empty((len(ids), y2-y1, x2-x1), dtype=np.uint8)
        for index, instance_id in enumerate(ids):
            visible[index] = view == instance_id
        with Image.open(sample["rgb_path"]) as source:
            if source.size != (original_w, original_h):
                raise DataContractError("RGB/mask dimensions differ")
            image = source.convert("RGB").crop((x1, y1, x2, y2))
        name = sample["source_filename"]
        identity = name if tile_index is None else f"{name}::tile{tile_index}"
        source = {"dataset": "MinneApple", "split": self.split, "image_id": identity,
                  "original_image_id": name, "rgb_path": str(sample["rgb_path"]),
                  "crop_xyxy": [x1, y1, x2, y2], "original_image_hw": [original_h, original_w],
                  "tile_index": tile_index, "annotation_sha256": {"visible": digest, "amodal": None},
                  "split_manifest_sha256": self.split_manifest_sha256}
        return _item(image, visible, None, [str(value) for value in ids], source, self.input_size, self.augment)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.tiled:
            original, tile = divmod(index, len(MINNE_TILES))
            return self._read(original, MINNE_TILES[tile], tile)
        return self._read(index)

    def iter_inference_tiles(self, original_index):
        """Separate fixed-grid views; original whole-image evaluation GT stays intact."""
        if self.augment:
            raise ValueError("Inference tiles cannot use augmentation")
        for tile, crop in enumerate(MINNE_TILES):
            yield self._read(original_index, crop, tile)


def build_papple_dataset(data_root, split="train", *, input_size=768, augment=False):
    return PAppleDataset(data_root, split, input_size, augment)


def build_minneapple_dataset(raw_root, split_manifest, split="train", *, input_size=768, tiled=True, augment=False):
    return MinneAppleDataset(raw_root, split_manifest, split, input_size, tiled, augment)
