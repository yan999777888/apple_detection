"""Lazy PApple TRAIN/VAL adapter for the frozen dual-query candidate.

Importing this module uses only the standard library. The factory is a future
runtime boundary: it verifies the existing loader and genuine polygon bytes,
then imports Torch/Pillow/numpy and constructs PAppleDataset(augment=False).
No model, weights, cache, optimizer, evaluator, augmentation or TEST path exists.

Each item has unbatched pixel_values[3,S,S], pixel_mask[S,S] (all True), a
criterion target, the untouched original_target dictionary and original apple
ID order. collate_query_papple_batch accepts exactly one item. Masks stay binary
at original 1300x1300; only the complete RGB frame is resized and normalized.
The frozen criterion accepts TRAIN targets only; VAL targets are for evaluation.

The optional --verify-cpu entry point reads exactly two distinct TRAIN RGB
images (one annotated, one empty), and never constructs a model. It must only
be run when the parent authorizes the additional CPU/RAM load.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re


ANNOTATION_SHA256 = {
    "train": {
        "visible": "423afe0e0fca7251c9e9da5b9ebae8b60e6b0f7240b9bcf39bc7ff8a6c0d9cab",
        "amodal": "4b3a739628a974ca42ff84d1e0f431cc98adc1591377552b6262454f41c71751",
    },
    "val": {
        "visible": "6f6f3c2eb35afa2a431d507f18e6c1258764168778550167efc69226e7bed93e",
        "amodal": "aa659d827ea3efad82cb2a6cb0a3ef1126abc9c150e342baa133689b9f409845",
    },
}
DATASET_LOADER_SHA256 = "4d8fe0f1ac98e3b54dd4f029675ad240b621766140b259a537e99290ace9a0a0"
ORIGINAL_SIZE = (1300, 1300)  # height, width
SUPPORTED_INPUT_SIZES = (512, 768, 1024)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def audit_adapter_config(*, split, input_size, augment=False, batch_size=1,
                         external_padding=False):
    """Validate metadata without importing Torch or opening any dataset file."""
    if split not in ("train", "val"):
        raise ValueError("Only exact lowercase train/val splits are permitted; TEST is forbidden")
    if (isinstance(input_size, bool) or not isinstance(input_size, int)
            or input_size not in SUPPORTED_INPUT_SIZES or input_size % 32):
        raise ValueError("Explicit square input_size must be one of 512/768/1024, divisible by 32")
    if augment is not False:
        raise NotImplementedError("Augmentation has no audited paired-mask transform; request rejected")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size != 1:
        raise ValueError("Only batch_size=1 is supported")
    if external_padding is not False:
        raise NotImplementedError("External padding has no audited full-frame alignment; request rejected")
    return {"split": split, "input_size": input_size, "batch_size": 1,
            "augment": False, "external_padding": False}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths(root, split):
    # Resolve only the requested TRAIN/VAL root, never discover or inspect TEST.
    root = Path(root).expanduser().resolve()
    if not ((root / "gt_json").is_dir() and (root / "images").is_dir()):
        root = root / "data"
    if not ((root / "gt_json").is_dir() and (root / "images").is_dir()):
        raise FileNotFoundError("Expected genuine PApple images/ and gt_json/ directories")
    annotation_paths = {
        "visible": root / "gt_json" / split / "via_region_data_instance.json",
        "amodal": root / "gt_json" / split / "via_region_data_amodal.json",
    }
    return root, annotation_paths


def _verify_annotations(paths, split):
    actual = {branch: _sha256(path) for branch, path in paths.items()}
    if actual != ANNOTATION_SHA256[split]:
        raise ValueError(f"{split} annotation bytes do not match the fixed genuine paired-polygon SHA256")
    return actual


def build_query_papple_dataset(root, split, *, input_size, augment=False,
                               batch_size=1, external_padding=False):
    """Lazy runtime factory; leaves source/data/cache unchanged and loads no PT.

    The parsed sample snapshot is bound to hashes checked immediately before
    and after PAppleDataset initialization. All official samples, empty images,
    paired instances and original ordering are retained; there is no max_images
    or filtering option. Source geometry/count mismatches fail instead of being
    repaired, clipped to a subset, or replaced by box masks.
    """
    config = audit_adapter_config(split=split, input_size=input_size, augment=augment,
                                  batch_size=batch_size, external_padding=external_padding)
    root, paths = _source_paths(root, split)
    annotation_sha256 = _verify_annotations(paths, split)
    loader_path = Path(__file__).resolve().parents[2] / "scripts" / "dataset.py"
    if _sha256(loader_path) != DATASET_LOADER_SHA256:
        raise ValueError("Existing PAppleDataset source changed; audit the loader before using this adapter")

    # Numerical dependencies are imported only after explicit runtime entry.
    import numpy as np
    from PIL import Image
    import torch

    specification = importlib.util.spec_from_file_location("_query_papple_bound_original_dataset", loader_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load existing PAppleDataset from {loader_path}")
    loader = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loader)
    base = loader.PAppleDataset(root, split, augment=False)
    _verify_annotations(paths, split)
    if _sha256(loader_path) != DATASET_LOADER_SHA256:
        raise ValueError("Existing PAppleDataset source changed during initialization")
    for sample in base.samples:
        if (sample["height"], sample["width"]) != ORIGINAL_SIZE:
            raise ValueError(f"Expected original square 1300x1300 frame: {sample['filename']}")
        if not re.fullmatch(r"_MG_[0-9]+_[0-9]+\.png", sample["filename"]):
            raise ValueError(f"Unexpected genuine PApple filename: {sample['filename']}")
        if len(sample["apple_ids"]) > 100:
            raise ValueError("GT count exceeds 100 queries; never truncate original instances")

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]

    class QueryPAppleDataset:
        def __init__(self):
            self._base = base
            self.configuration = deepcopy(config)
            self.annotation_sha256 = deepcopy(annotation_sha256)

        def __len__(self):
            return len(self._base)

        def image_metadata(self, index):
            sample = self._base.samples[index]
            return {"filename": sample["filename"], "apple_ids": tuple(sample["apple_ids"]),
                    "original_size": (sample["height"], sample["width"]),
                    "input_size": (input_size, input_size)}

        def __getitem__(self, index):
            # Use the loader's unaugmented path; do not call its augmentation API.
            rgb, original = self._base.load_sample(index)
            sample = self._base.samples[index]
            count = len(sample["apple_ids"])
            if rgb.dtype != np.uint8 or tuple(rgb.shape) != (*ORIGINAL_SIZE, 3):
                raise ValueError("Decoded source must be unpadded RGB uint8[1300,1300,3]")
            for name in ("masks", "visible_masks"):
                mask = original[name]
                if mask.dtype != np.uint8 or tuple(mask.shape) != (count, *ORIGINAL_SIZE):
                    raise ValueError("Original real V/A masks must be uint8[N,1300,1300]")
                if np.any((mask != 0) & (mask != 1)):
                    raise ValueError("Source polygon rasterization must remain strictly binary")
                if count and not mask.reshape(count, -1).any(axis=1).all():
                    raise ValueError("An empty per-instance polygon raster is rejected, never filtered")
            if original["labels"].dtype != np.int64 or not np.all(original["labels"] == 1):
                raise ValueError("Expected unchanged source apple labels=1 before explicit query mapping")

            resized = Image.fromarray(rgb).resize((input_size, input_size), Image.Resampling.BILINEAR)
            resized_rgb = np.asarray(resized, dtype=np.uint8).copy()
            pixels = torch.from_numpy(resized_rgb.transpose(2, 0, 1).copy()).to(dtype=torch.float32)
            pixels.div_(255.0).sub_(mean).div_(std)
            original_target = {key: torch.from_numpy(value) for key, value in original.items()}
            metadata = {
                **self.image_metadata(index), "resize": "Pillow RGB bilinear whole frame",
                "coordinate_mapping": "normalized_xy_whole_frame; criterion align_corners=False",
                "mask_size": ORIGINAL_SIZE, "external_padding": False, "crop": False,
                "augment": False, "rgb_scale_before_normalization": "uint8/255 in [0,1]",
                "normalization_mean": IMAGENET_MEAN, "normalization_std": IMAGENET_STD,
            }
            source = {
                "dataset": "PApple", "split": split,
                "mask_origin": "paired_instance_polygons", "filename": sample["filename"],
                "annotation_sha256": deepcopy(annotation_sha256),
                "annotation_files": {key: str(path) for key, path in paths.items()},
                "dataset_loader_sha256": DATASET_LOADER_SHA256,
                "image_path": str(sample["path"]),
                "binding": "parsed snapshot; SHA256 checked before and after existing loader initialization",
            }
            target = {
                "labels": torch.zeros(count, dtype=torch.int64),
                "mask_valid": torch.ones(count, dtype=torch.bool),
                "visible_masks": original_target["visible_masks"],
                "amodal_masks": original_target["masks"],
                "source": source, "metadata": metadata,
            }
            # No V<=A repair/check: original V outside A remains exactly present.
            return {"pixel_values": pixels,
                    "pixel_mask": torch.ones((input_size, input_size), dtype=torch.bool),
                    "target": target, "original_target": original_target,
                    "apple_ids": tuple(sample["apple_ids"]), "metadata": metadata}

    return QueryPAppleDataset()


def collate_query_papple_batch(items):
    """DataLoader collate_fn for batch1; no padding or numerical imports here.

    wrapper(batch['pixel_values'], batch['pixel_mask']) consumes the image fields;
    criterion(outputs, batch['targets']) consumes TRAIN targets. original_targets
    preserve the existing evaluator's labels=1, masks, boxes and image IDs.
    """
    if not isinstance(items, (list, tuple)) or len(items) != 1:
        raise ValueError("Exactly one item is required; multi-image batches/padding are forbidden")
    item = items[0]
    return {"pixel_values": item["pixel_values"].unsqueeze(0),
            "pixel_mask": item["pixel_mask"].unsqueeze(0),
            "targets": [item["target"]], "original_targets": [item["original_target"]],
            "apple_ids": [item["apple_ids"]], "metadata": [item["metadata"]]}


def verify_cpu_adapter(root, *, input_size=512):
    """Deferred bounded CPU verification: one ordinary + one empty real TRAIN.

    Runtime only. Verifies independent numpy normalization, repeatability,
    original-mask/box/ID equality, original anomalies, no padding, and provenance.
    It never imports the model wrapper/criterion, reads PT, or uses CUDA.
    """
    import numpy as np
    from PIL import Image
    import torch

    torch.set_num_threads(1)
    adapter = build_query_papple_dataset(root, "train", input_size=input_size)
    ordinary = next((i for i in range(len(adapter)) if adapter.image_metadata(i)["apple_ids"]), None)
    empty = next((i for i in range(len(adapter)) if not adapter.image_metadata(i)["apple_ids"]), None)
    if ordinary is None or empty is None:
        raise AssertionError("Real TRAIN must contain both ordinary and empty examples")

    def digest_tensor(tensor):
        return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()

    results = []
    for kind, index in (("ordinary", ordinary), ("empty", empty)):
        item = adapter[index]
        target = item["target"]
        rgb, original = adapter._base.load_sample(index)
        count = len(item["apple_ids"])
        assert (count > 0) == (kind == "ordinary")
        assert item["apple_ids"] == tuple(adapter._base.samples[index]["apple_ids"])
        assert target["source"]["annotation_sha256"] == ANNOTATION_SHA256["train"]
        assert target["source"]["filename"] == adapter.image_metadata(index)["filename"]
        assert target["labels"].dtype == torch.int64 and tuple(target["labels"].shape) == (count,)
        assert bool((target["labels"] == 0).all())
        assert target["mask_valid"].dtype == torch.bool and bool(target["mask_valid"].all())
        for key, expected in original.items():
            np.testing.assert_array_equal(item["original_target"][key].numpy(), expected)
        np.testing.assert_array_equal(target["visible_masks"].numpy(), original["visible_masks"])
        np.testing.assert_array_equal(target["amodal_masks"].numpy(), original["masks"])
        for key in ("visible_masks", "amodal_masks"):
            assert tuple(target[key].shape) == (count, *ORIGINAL_SIZE)
            assert target[key].dtype == torch.uint8
            assert bool(((target[key] == 0) | (target[key] == 1)).all())
        original_outside = int(np.count_nonzero(original["visible_masks"] & (1-original["masks"])))
        adapter_outside = int(torch.count_nonzero(target["visible_masks"] & (1-target["amodal_masks"])))
        assert adapter_outside == original_outside
        reference = np.asarray(Image.fromarray(rgb).resize((input_size, input_size),
                               Image.Resampling.BILINEAR), dtype=np.float32) / np.float32(255.0)
        reference = (reference - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
        np.testing.assert_allclose(item["pixel_values"].numpy(), reference.transpose(2, 0, 1), rtol=0, atol=1e-6)
        assert item["pixel_values"].device.type == "cpu" and torch.isfinite(item["pixel_values"]).all()
        assert tuple(item["pixel_values"].shape) == (3, input_size, input_size)
        assert bool(item["pixel_mask"].all())
        batch = collate_query_papple_batch([item])
        assert tuple(batch["pixel_values"].shape) == (1, 3, input_size, input_size)
        assert tuple(batch["pixel_mask"].shape) == (1, input_size, input_size)
        assert batch["targets"][0] is target and batch["original_targets"][0] is item["original_target"]
        digests = {key: digest_tensor(value) for key, value in
                   (("pixels", item["pixel_values"]), ("visible", target["visible_masks"]),
                    ("amodal", target["amodal_masks"]))}
        result = {"kind": kind, "filename": target["source"]["filename"], "instances": count,
                  "apple_ids": item["apple_ids"], "original_size": ORIGINAL_SIZE,
                  "input_size": (input_size, input_size), "visible_pixels_outside_amodal": original_outside,
                  "tensor_sha256": digests}
        del item, target, batch, rgb, original, reference
        repeated = adapter[index]
        assert digest_tensor(repeated["pixel_values"]) == digests["pixels"]
        assert digest_tensor(repeated["target"]["visible_masks"]) == digests["visible"]
        assert digest_tensor(repeated["target"]["amodal_masks"]) == digests["amodal"]
        del repeated
        results.append(result)
    _, paths = _source_paths(root, "train")
    _verify_annotations(paths, "train")
    return {"schema": "query_papple_adapter_bounded_cpu_verification_v1",
            "cpu_verification_passed": True, "model_executed": False,
            "criterion_executed": False, "test_accessed": False,
            "source_sha256": _sha256(__file__), "loader_sha256": DATASET_LOADER_SHA256,
            "train_annotation_sha256": deepcopy(ANNOTATION_SHA256["train"]), "samples": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-cpu", action="store_true", help="explicitly execute the bounded real-TRAIN CPU check")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--input-size", type=int, choices=SUPPORTED_INPUT_SIZES, default=512)
    args = parser.parse_args()
    if not args.verify_cpu:
        parser.error("Only the explicit --verify-cpu entry point is available")
    print(json.dumps(verify_cpu_adapter(args.data_root, input_size=args.input_size), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
