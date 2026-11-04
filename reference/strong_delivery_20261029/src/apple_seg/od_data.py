"""Aspect-preserving RGB views with untouched native-resolution annotations."""
from __future__ import annotations

import math
import hashlib
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .data import NORMALIZATION_MEAN, NORMALIZATION_STD, flip_item_horizontal


LEGACY_RGB_PREPROCESS = {
    "resize_backend": "torch.nn.functional.interpolate",
    "normalization_before_resize": True,
    "resize_align_corners": False,
    "resize_antialias": False,
    "resize_recompute_scale_factor": True,
    "input_operation_order": "uint8_RGB -> CHW_float32/255 -> ImageNet_normalize -> bilinear_aspect_resize",
}


def _resize_scale(height, width, short_side, max_size):
    for value in (height, width, short_side, max_size):
        if type(value) is not int or value < 1:
            raise ValueError("Image and resize dimensions must be positive integers")
    return min(short_side / min(height, width), max_size / max(height, width))


def resized_hw(height, width, short_side, max_size):
    scale = _resize_scale(height, width, short_side, max_size)
    result = math.floor(height * scale), math.floor(width * scale)
    if min(result) < 1:
        raise ValueError("Legacy torchvision resize would produce an empty dimension")
    return result


def legacy_rgb_tensor(rgb, short_side, max_size):
    """Match the eager legacy GeneralizedRCNNTransform before image padding.

    Convert uint8 RGB to float32 CHW/255, normalize BEFORE resizing, then use
    the exact scalar-scale bilinear path with recompute_scale_factor=True.
    In particular, do not use a Pillow uint8 resize or antialias filtering.
    """
    if not isinstance(rgb, Image.Image):
        raise TypeError("legacy_rgb_tensor expects a PIL image")
    rgb = rgb.convert("RGB")
    width, height = rgb.size
    scale = _resize_scale(height, width, short_side, max_size)
    resized_hw(height, width, short_side, max_size)
    array = np.asarray(rgb, dtype=np.uint8)
    image = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float().div_(255.)
    mean = torch.as_tensor(NORMALIZATION_MEAN, dtype=image.dtype)[:, None, None]
    std = torch.as_tensor(NORMALIZATION_STD, dtype=image.dtype)[:, None, None]
    image = (image - mean) / std
    return F.interpolate(image[None], scale_factor=scale, mode="bilinear", align_corners=False,
                         recompute_scale_factor=True, antialias=False)[0].contiguous()


class ODInputDataset:
    """Reload genuine source RGB/crop; keep original V/A masks and every ID.

    The wrapped native dataset may use a tiny RGB preprocessing size: its
    normalized pixels are discarded, while its original annotation contract is
    reused. Resize selection and flips use the checkpointed Python RNG.
    """
    def __init__(self, native_dataset, *, short_sides=(1024,), max_size=1536,
                 augment=False):
        if getattr(native_dataset, "augment", False):
            raise ValueError("Wrap an unaugmented native dataset")
        if not isinstance(short_sides, (tuple, list)) or not short_sides:
            raise ValueError("Supply at least one short-side size")
        for value in (*short_sides, max_size):
            if type(value) is not int or value < 1:
                raise ValueError("Resize dimensions must be positive integers")
        if getattr(native_dataset, "split", None) != "train" and (augment or len(short_sides) != 1):
            raise ValueError("Evaluation requires one deterministic whole-image resize")
        self.native_dataset = native_dataset
        self.short_sides, self.max_size = tuple(short_sides), max_size
        self.augment, self.split = bool(augment), native_dataset.split
        self.annotation_paths = native_dataset.annotation_paths
        self.annotation_sha256 = dict(native_dataset.annotation_sha256)
        self.configuration = {
            "short_sides": list(short_sides), "max_size": max_size,
            "resize": "Legacy torchvision float32 bilinear aspect resize; normalize before resize; no antialias",
            **LEGACY_RGB_PREPROCESS,
            "normalization_mean": list(NORMALIZATION_MEAN),
            "normalization_std": list(NORMALIZATION_STD), "rgb_scale": "uint8/255",
            "augment": self.augment, "external_padding": False, "padding": False,
            "model_padding": "bottom/right zeros to divisor32 after normalization",
            "crop": bool(getattr(native_dataset, "tiled", False)),
            "GT_resolution": "original supplied view; CPU; no resize, repair or truncation",
        }

    def __len__(self):
        return len(self.native_dataset)

    def metadata(self):
        result = dict(self.native_dataset.metadata())
        result.update(augment=self.augment, input_preprocessing=dict(self.configuration))
        result.pop("input_size", None)
        return result

    def __getitem__(self, index):
        target = self.native_dataset[index]["target"]
        if target["source"].get("horizontal_flip", False):
            raise ValueError("Native annotation view must be unflipped")
        height, width = target["original_hw"]
        source = dict(target["source"])
        path = Path(source["rgb_path"])
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            crop = source.get("crop_xyxy", (0, 0, *rgb.size))
            if len(crop) != 4 or any(type(v) is not int for v in crop):
                raise ValueError("Source crop must be an integer xyxy window")
            x1, y1, x2, y2 = crop
            if not (0 <= x1 < x2 <= rgb.width and 0 <= y1 < y2 <= rgb.height):
                raise ValueError("Source crop is outside original RGB")
            rgb = rgb.crop(tuple(crop))
            if rgb.size != (width, height):
                raise ValueError("Source RGB crop and native mask sizes differ")
            short_side = random.choice(self.short_sides)
            pixels = legacy_rgb_tensor(rgb, short_side, self.max_size)
            new_h, new_w = pixels.shape[-2:]
        source.update(resize_hw=[new_h, new_w], horizontal_flip=False,
                      original_to_input_scale_xy=[new_w / width, new_h / height],
                      resize_policy="torchvision_GeneralizedRCNNTransform_float32_normalize_then_resize",
                      **LEGACY_RGB_PREPROCESS)
        target = {**target, "source": source}
        item = dict(pixel_values=pixels,
                    pixel_mask=torch.ones((new_h, new_w), dtype=torch.bool), target=target)
        if self.augment and random.random() < .5:
            item = flip_item_horizontal(item)
        item["target"]["source"]["normalized_input_sha256"] = hashlib.sha256(
            memoryview(item["pixel_values"].numpy())).hexdigest()
        return item
