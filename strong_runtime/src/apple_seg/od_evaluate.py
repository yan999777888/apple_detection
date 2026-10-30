"""Strict full PApple VAL evaluation for ROI-local OD detector predictions.

Original CPU annotations and identities use the pinned query evaluator's audit
contract. Boxes are the detector's A-regression boxes in native coordinates;
V/A masks are independently pasted at native pixel centers in bounded tiles.
No GT enters the model, and a completed summary is written only after audits.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from . import evaluate as base
from . import od_roi
from .od_data import ODInputDataset, NORMALIZATION_MEAN, NORMALIZATION_STD, LEGACY_RGB_PREPROCESS, resized_hw
from .od_roi import paste_roi_mask_tile


POSTPROCESS_POLICY = {
    "schema": "od_roi_native_postprocess_v1",
    "prediction_family": "two_stage_ROI_visible_hidden_amodal",
    "score": "supplied model apple probability; see model configuration",
    "evaluator_score_filter": "none; all model-returned detections retained",
    "order": "model output order; score matching uses descending score and stable input-order ties",
    "box_format": "predicted A-regression XYXY continuous pixel extents, rescaled input to original image",
    "mask_transform": "ROI probabilities -> original image pixel-center bilinear sampling -> >=0.5",
    "align_corners": False,
    "mask_padding": "zero probability outside sampled ROI support",
    "cross_branch_projection": "none; visible is not intersected with amodal",
    "instance_fusion": "none in evaluator; see model NMS configuration",
    "overlapping_amodal_instances": "as returned by the configured model NMS",
    "output_mask_dtype": "bool",
    "ground_truth_used": False,
    "accuracy_or_ap_claimed": False,
}


def _postprocess_policy(model, *, gt_chunk_size, tile_pixels, defer_mask_metrics):
    configuration = getattr(model, "constructor_config", None)
    if not isinstance(configuration, dict):
        configuration = getattr(getattr(model, "roi_heads", None), "configuration", None)
    configuration = configuration if isinstance(configuration, dict) else {}
    gate, refine, nms = (configuration.get(key) for key in
                         ("use_visible_gate", "use_amodal_refinement", "final_nms"))
    verifier = configuration.get("occlusion_verifier")
    policy = {**POSTPROCESS_POLICY, "GT_chunk_size": gt_chunk_size, "max_tile_pixels": tile_pixels,
              "model_postprocess_configuration": base._jsonable(configuration),
              "visible_gate_enabled": gate, "amodal_refinement_enabled": refine,
              "final_nms": nms, "defer_mask_metrics": defer_mask_metrics,
              "mask_metric_scope": ("Only score>=globally calibrated BOX confidence; other predictions explicitly unevaluated"
                                    if defer_mask_metrics else "All returned predictions before BOX calibration")}
    if gate is not None:
        policy["score"] = ("supplied predicted-visible gated classifier apple probability" if gate
                           else "supplied base box classifier apple probability; visible gate disabled")
    if verifier is True:
        policy["score"] = (policy["score"] + " plus the zero-initialized amodal evidence verifier;"
                           " all three terms come from predicted masks only")
    if refine is not None:
        policy["amodal_mask"] = ("trained independent amodal base plus conditioned refinement" if refine
                                 else "trained independent amodal base; conditioned refinement disabled")
    if nms in ("box", "dual_mask"):
        policy["instance_fusion"] = ("none in evaluator; model applies box NMS" if nms == "box"
                                      else "none in evaluator; model applies predicted A/V dual-mask NMS")
    return policy


class _DeferredMaskCache:
    """One lossless float32 ROI archive per image, owned by one output directory.

    Only one image is loaded at a time. Records bind archive bytes, array shapes,
    original output order and image identity. Failed runs retain their evidence;
    successful cleanup unlinks only verified files created by this object.
    """
    DIRECTORY = "deferred_roi_mask_cache"

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir).resolve()
        self.directory = self.output_dir / self.DIRECTORY
        base._require(not self.directory.exists() and not self.directory.is_symlink(),
                      "Use a fresh deferred mask cache directory; existing caches are never overwritten")
        self.directory.mkdir(exist_ok=False)
        self.entries = []
        self.metadata = {"directory": self.DIRECTORY, "format": "numpy_npz_uncompressed",
                         "dtype": "float32", "lossless": True, "entries": self.entries,
                         "read_verified_files": 0, "cleanup_complete": False}

    def _path(self, entry):
        expected = f"roi_{entry['image_index']:06d}.npz"
        base._require(entry["path"] == f"{self.DIRECTORY}/{expected}", "Unsafe deferred cache path")
        path = self.directory / expected
        base._require(not self.directory.is_symlink() and self.directory.resolve() == self.directory
                      and self.directory.parent == self.output_dir and not path.is_symlink()
                      and path.resolve().parent == self.directory,
                      "Deferred cache resolved path leaves its owned output directory")
        return path

    def store(self, image_index, image_id, prediction):
        base._require(image_index == len(self.entries), "Deferred cache image order differs")
        shape = [len(prediction["scores"]), *prediction["roi_hw"]]
        entry = {"image_index": image_index, "image_id": image_id,
                 "path": f"{self.DIRECTORY}/roi_{image_index:06d}.npz", "shape": shape, "dtype": "float32"}
        path = self._path(entry)
        arrays = {branch: prediction[branch].detach().cpu().contiguous().numpy()
                  for branch in ("visible", "amodal")}
        base._require(all(array.dtype == np.float32 and list(array.shape) == shape for array in arrays.values()),
                      "Deferred cache requires exact float32 ROI probabilities")
        # Exclusive creation prevents replacing an old archive, even if a path
        # appears after the fresh-directory check.
        with path.open("xb") as stream:
            np.savez(stream, **arrays)
        entry.update(bytes=path.stat().st_size, sha256=base._sha(path))
        self.entries.append(entry)
        return entry

    def _verify(self, entry):
        path = self._path(entry)
        base._require(path.is_file() and path.stat().st_size == entry["bytes"]
                      and base._sha(path) == entry["sha256"], "Deferred cache file size/SHA256 differs")
        return path

    def load(self, entry, selected, *, image_index, image_id, count, roi_hw):
        base._require(entry["image_index"] == image_index and entry["image_id"] == image_id
                      and entry["shape"] == [count, *roi_hw] and entry["dtype"] == "float32",
                      "Deferred cache prediction/image identity differs")
        base._require(selected == sorted(set(selected)) and all(0 <= index < count for index in selected),
                      "Deferred cache selected output indices differ")
        path = self._verify(entry)
        result = {}
        with np.load(path, allow_pickle=False) as archive:
            base._require(sorted(archive.files) == ["amodal", "visible"], "Deferred cache branches differ")
            for branch in ("visible", "amodal"):
                array = archive[branch]
                base._require(array.dtype == np.float32 and list(array.shape) == entry["shape"]
                              and bool(np.isfinite(array).all()) and bool(((array >= 0) & (array <= 1)).all()),
                              "Deferred cache must contain finite float32 ROI probabilities with exact shapes")
                result[branch] = torch.from_numpy(np.array(array[selected], copy=True))
        self.metadata["read_verified_files"] += 1
        return result

    def cleanup(self):
        # Verify the complete directory before unlinking anything. Unknown files
        # and symlinks are never deleted, and no recursive operation is used.
        paths = [self._verify(entry) for entry in self.entries]
        base._require(set(self.directory.iterdir()) == set(paths),
                      "Deferred cache contains files not created by this evaluation")
        for entry in self.entries:
            self._verify(entry).unlink()
        self.directory.rmdir()
        self.metadata["cleanup_complete"] = True


def _integer(value, name, *, allow_zero=False):
    base._require(isinstance(value, int) and not isinstance(value, bool)
                  and value >= (0 if allow_zero else 1), f"{name} must be a positive integer")


_PACKAGE_DIRECTORY = Path(__file__).resolve().parent


def _local_implementation_sources(value):
    """Sources of every local class that actually implements this object.

    ``inspect.getfile(type(value))`` names only the concrete class module. A
    subclass that inherits ``forward``/``_classifier_training``/``_cascade_boxes``
    executes the parent body, so a before/after VAL proof built from the concrete
    type alone would omit that executed module and a strict loader would then
    reject its own official validation. Walk the real MRO and add each class that
    is defined inside this package; library classes are not part of the pinned OD
    inventory and are never added here.
    """
    paths = []
    try:
        classes = type(value).__mro__
    except (AttributeError, TypeError):
        return paths
    for cls in classes:
        try:
            path = Path(inspect.getfile(cls)).absolute()
        except (TypeError, OSError):
            continue
        if path.parent == _PACKAGE_DIRECTORY and path.is_file():
            paths.append(path)
    return paths


def _source_snapshot(model, dataset, contract):
    paths = [Path(__file__).resolve(), Path(base.__file__).resolve(), Path(od_roi.__file__).resolve(),
             Path(inspect.getfile(ODInputDataset)).absolute(), base.METRICS_PATH, contract["provenance_path"]]
    head = getattr(model, "roi_heads", None)
    # Context heads execute three additional local modules. Hash their actual
    # definitions on both sides of VAL, independently of training source pins.
    values = [model, dataset, head, getattr(model, "visible_proposal_branch", None),
              getattr(model, "shallow_detail_fusion", None)]
    if head is not None:
        values.extend(getattr(head, name, None) for name in (
            "context_fusion", "context_cascade", "foreground_head", "foreground_fusion",
            "mask_geometry_head", "mask_geometry_refiner", "occlusion_verifier"))
    seen = set()
    annotation_paths = None
    while values:
        value = values.pop()
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        try:
            path = Path(inspect.getfile(type(value))).absolute()
            if path.is_file():
                paths.append(path)
        except (TypeError, OSError):
            pass
        paths.extend(_local_implementation_sources(value))
        for name in ("native_dataset", "native", "base", "dataset"):
            wrapped = getattr(value, name, None)
            if wrapped is not None and wrapped is not value:
                values.append(wrapped)
        candidate = getattr(value, "annotation_paths", None)
        if candidate is not None:
            base._require(isinstance(candidate, dict) and set(candidate) == {"visible", "amodal"},
                          "annotation_paths must identify original visible and amodal JSON")
            if annotation_paths is not None:
                base._require({k: str(Path(v).absolute()) for k, v in candidate.items()} ==
                              {k: str(Path(v).absolute()) for k, v in annotation_paths.items()},
                              "Wrapped dataset annotation paths differ")
            annotation_paths = candidate
    base._require(annotation_paths is not None or not contract["official_full_val"],
                  "Official full VAL requires original annotation JSON paths for before/after hashing")
    if annotation_paths is not None:
        for branch, path in annotation_paths.items():
            base._require(base._sha(path) == contract["provenance"]["binding"]["current_annotation_sha256"][branch],
                          "Actual annotation JSON SHA256 differs")
            paths.append(Path(path).absolute())
    return {str(path): base._sha(path) for path in dict.fromkeys(paths)}, annotation_paths is not None


def _input_sha256(pixels):
    value = pixels[0] if pixels.ndim == 4 else pixels
    base._require(value.ndim == 3 and value.dtype == torch.float32,
                  "OD normalized RGB must use canonical float32 CHW values")
    array = value.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def _audit_input(pixels, target, dataset, frozen, *, recompute_rgb):
    """Bind consumed normalized pixels, resize geometry, and original RGB."""
    input_hw, source = tuple(pixels.shape[-2:]), target["source"]
    expected_scale = [input_hw[1] / frozen["width"], input_hw[0] / frozen["height"]]
    base._require(source.get("resize_hw") == list(input_hw), "Source resize_hw differs from actual model input")
    base._require(source.get("original_to_input_scale_xy") == expected_scale,
                  "Source original_to_input_scale_xy differs from actual model input")
    if "scale_xy" in source:
        base._require(source["scale_xy"] == expected_scale, "Source scale_xy differs from actual model input")
    digest = _input_sha256(pixels)
    base._require(source.get("normalized_input_sha256") == digest,
                  "Normalized input tensor SHA256 differs from dataset source binding")
    configuration = getattr(dataset, "configuration", {})
    if "input_hw" in configuration:
        base._require(configuration["input_hw"] == list(input_hw), "Public dataset input_hw differs from actual input")
    recomputed = False
    if recompute_rgb and isinstance(dataset, ODInputDataset):
        short_sides, max_size = configuration.get("short_sides"), configuration.get("max_size")
        base._require(isinstance(short_sides, list) and len(short_sides) == 1
                      and tuple(short_sides) == dataset.short_sides and max_size == dataset.max_size,
                      "Public OD resize configuration differs from the actual dataset")
        base._require(resized_hw(frozen["height"], frozen["width"], short_sides[0], max_size) == input_hw,
                      "OD aspect-preserving resize dimensions differ")
        base._require(configuration.get("normalization_mean") == list(NORMALIZATION_MEAN)
                      and configuration.get("normalization_std") == list(NORMALIZATION_STD)
                      and configuration.get("rgb_scale") == "uint8/255",
                      "Public OD normalization configuration differs")
        base._require(all(configuration.get(key) == value and source.get(key) == value
                          for key, value in LEGACY_RGB_PREPROCESS.items()),
                      "Public/source OD legacy resize backend or operation order differs")
        with Image.open(base._rgb_path(target)) as image:
            rgb = image.convert("RGB")
            base._require(rgb.size == (frozen["width"], frozen["height"]), "Original RGB dimensions differ")
            # Independently repeat the legacy float pipeline from audited RGB;
            # do not call the dataset's preprocessing helper for this audit.
            array = np.asarray(rgb, dtype=np.uint8)
        expected = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float().div_(255.)
        mean = torch.as_tensor(NORMALIZATION_MEAN, dtype=expected.dtype)[:, None, None]
        std = torch.as_tensor(NORMALIZATION_STD, dtype=expected.dtype)[:, None, None]
        expected = (expected - mean) / std
        scale = min(short_sides[0] / min(frozen["height"], frozen["width"]),
                    max_size / max(frozen["height"], frozen["width"]))
        expected = F.interpolate(expected[None], scale_factor=scale, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=True, antialias=False)[0].contiguous()
        base._require(hashlib.sha256(memoryview(expected.numpy())).hexdigest() == digest,
                      "Consumed normalized input differs from recomputed original RGB preprocessing")
        recomputed = True
    return {"image_id": frozen["image_id"], "input_hw": list(input_hw),
            "original_to_input_scale_xy": expected_scale, "sha256": digest,
            "preprocessing_backend": configuration.get("resize_backend"),
            "normalization_before_resize": configuration.get("normalization_before_resize"),
            "resize_antialias": configuration.get("resize_antialias"),
            "recomputed_from_original_RGB": recomputed}


def _validate_prediction(outputs, input_hw, original_hw, *, mask_chunk_size):
    base._require(isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], dict),
                  "OD inference must return a list containing one detection dictionary")
    output = outputs[0]
    required = ("boxes", "scores", "roi_visible_masks", "roi_amodal_masks", "image_hw")
    base._require(all(key in output for key in required), "OD prediction is missing required ROI fields")
    base._require(tuple(output["image_hw"]) == tuple(input_hw), "Prediction image_hw differs from unpadded RGB")
    boxes, scores = output["boxes"], output["scores"]
    base._require(isinstance(boxes, torch.Tensor) and boxes.is_floating_point() and boxes.ndim == 2
                  and boxes.shape[1] == 4 and bool(torch.isfinite(boxes).all())
                  and bool((boxes[:, 2:] > boxes[:, :2]).all()),
                  "Predicted A boxes must be finite positive floating[N,4] XYXY")
    count = len(boxes)
    base._require(isinstance(scores, torch.Tensor) and scores.is_floating_point()
                  and tuple(scores.shape) == (count,) and bool(torch.isfinite(scores).all())
                  and bool(((scores >= 0) & (scores <= 1)).all()), "Prediction scores must be probabilities[N]")
    masks = {}
    roi_hw = None
    for name in ("visible", "amodal"):
        value = output["roi_" + name + "_masks"]
        base._require(isinstance(value, torch.Tensor) and value.is_floating_point() and value.ndim == 4
                      and tuple(value.shape[:2]) == (count, 1) and min(value.shape[-2:]) > 0,
                      "ROI mask probabilities must have [N,1,Hroi,Wroi] dimensions")
        base._require(roi_hw is None or roi_hw == tuple(value.shape[-2:]), "V/A ROI mask grids differ")
        roi_hw = tuple(value.shape[-2:])
        for start in range(0, count, mask_chunk_size):
            chunk = value[start:start + mask_chunk_size]
            base._require(bool(torch.isfinite(chunk).all()) and bool(((chunk >= 0) & (chunk <= 1)).all()),
                          "ROI masks must contain finite probabilities, not logits")
        masks[name] = value.detach().to(device="cpu", dtype=torch.float32)[:, 0]
    if "labels" in output:
        labels = output["labels"]
        base._require(isinstance(labels, torch.Tensor) and tuple(labels.shape) == (count,)
                      and bool((labels == 0).all()), "OD prediction labels must preserve apple=0")
    native_boxes = boxes.detach().to(device="cpu", dtype=torch.float32).clone()
    original_h, original_w = original_hw
    input_h, input_w = input_hw
    native_boxes *= torch.tensor([original_w / input_w, original_h / input_h] * 2)
    return {"boxes": native_boxes, "scores": scores.detach().cpu().tolist(), **masks,
            "roi_hw": roi_hw, "prediction_indices": list(range(count))}


def _tiles(window, tile_pixels):
    left, top, right, bottom = window
    if left >= right or top >= bottom:
        return
    for x1 in range(left, right, tile_pixels):
        x2 = min(right, x1 + tile_pixels)
        rows = max(1, tile_pixels // (x2 - x1))
        for y1 in range(top, bottom, rows):
            yield (x1, y1, x2, min(bottom, y1 + rows))


def _roi_mask_overlaps(prediction, truth, original_hw, *, gt_chunk_size=8, tile_pixels=65536):
    """Exact native IoUs; workspace is one predicted tile plus <=8 GT tiles."""
    _integer(gt_chunk_size, "gt_chunk_size")
    base._require(gt_chunk_size <= 8, "GT chunks must not exceed eight original instances")
    _integer(tile_pixels, "tile_pixels")
    height, width = original_hw
    matrices = {}
    count = len(prediction["scores"])
    for branch in ("visible", "amodal"):
        ground_truth = truth[branch + "_masks"]
        gt_areas = np.count_nonzero(ground_truth, axis=(1, 2))
        intersections = np.zeros((count, len(ground_truth)), dtype=np.int64)
        predicted_areas = np.zeros(count, dtype=np.int64)
        for index, box in enumerate(prediction["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            window = (max(0, min(width, math.floor(x1))), max(0, min(height, math.floor(y1))),
                      max(0, min(width, math.ceil(x2))), max(0, min(height, math.ceil(y2))))
            for tile in _tiles(window, tile_pixels):
                left, top, right, bottom = tile
                binary = paste_roi_mask_tile(prediction[branch][index], box, tile, threshold=.5).numpy()
                predicted_areas[index] += np.count_nonzero(binary)
                for start in range(0, len(ground_truth), gt_chunk_size):
                    stop = min(start + gt_chunk_size, len(ground_truth))
                    source = ground_truth[start:stop, top:bottom, left:right]
                    intersections[index, start:stop] += np.count_nonzero(source & binary[None], axis=(1, 2))
        union = predicted_areas[:, None] + gt_areas[None] - intersections
        matrices[branch + "_mask"] = np.divide(intersections, union, out=np.zeros_like(union, dtype=np.float64),
                                                 where=union > 0)
    return matrices


def _set_mask_diagnostics(diagnostic, matrices, selected):
    """Scatter selected-mask assignments into the unchanged full raw order."""
    count, scores = len(diagnostic["scores"]), [diagnostic["scores"][index] for index in selected]
    computed = [False] * count
    for index in selected:
        computed[index] = True
    diagnostic["mask_metrics_computed_by_prediction"] = computed
    diagnostic["mask_metrics_not_computed_indices"] = [index for index, value in enumerate(computed) if not value]
    for branch in ("visible_mask", "amodal_mask"):
        assignment = [-1] * count
        if branch in matrices:
            matches = base._greedy_assignment(matrices[branch], scores)
            for index, gt in zip(selected, matches, strict=True):
                assignment[index] = gt
        diagnostic[branch + "_gt_indices_by_prediction"] = assignment
        diagnostic[branch + "_tp_flags"] = [gt >= 0 for gt in assignment]


def _evaluate_deferred_masks(cache, dataset, contract, records, diagnostics, input_records, confidence,
                             *, gt_chunk_size, tile_pixels, progress_every):
    base._require(len(cache.entries) == len(contract["images"]), "Deferred cache full image scope differs")
    evaluated = 0
    with torch.inference_mode():
        for index, frozen in enumerate(contract["images"]):
            record, diagnostic = records[index], diagnostics[index]
            base._require(record["image_id"] == diagnostic["image_id"] == frozen["image_id"],
                          "Deferred image identity/order differs")
            selected = [p for p, score in enumerate(record["pred"]["scores"]) if score >= confidence]
            masks = cache.load(cache.entries[index], selected, image_index=index, image_id=frozen["image_id"],
                               count=len(diagnostic["scores"]), roi_hw=diagnostic["postprocess"]["ROI_mask_size_hw"])
            # GT is reloaded one original image at a time and verified again;
            # neither cached masks nor score selection use GT information.
            item = dataset[index]
            target = item["target"]
            rows = contract["truth"][frozen["image_id"]]
            truth = base._audit_target(target, frozen, rows)
            audited = _audit_input(item["pixel_values"], target, dataset, frozen, recompute_rgb=False)
            base._require(audited["sha256"] == input_records[index]["sha256"],
                          "Normalized input changed between inference and deferred mask evaluation")
            prediction = {**masks,
                          "boxes": torch.tensor(record["pred"]["boxes"], dtype=torch.float32).reshape(-1, 4)[selected],
                          "scores": [diagnostic["scores"][p] for p in selected]}
            matrices = (_roi_mask_overlaps(prediction, truth, (frozen["height"], frozen["width"]),
                                          gt_chunk_size=gt_chunk_size, tile_pixels=tile_pixels) if selected else {})
            _set_mask_diagnostics(diagnostic, matrices, selected)
            diagnostic["mask_metrics_confidence"] = confidence
            base._audit_target(target, frozen, rows)
            evaluated += len(selected)
            if progress_every and ((index + 1) % progress_every == 0 or index + 1 == len(contract["images"])):
                print(f"OD MASK {index + 1}/{len(contract['images'])}: selected_predictions={evaluated} "
                      f"confidence={confidence}", flush=True)
            del masks, item, target, truth, prediction, matrices
    return evaluated


def _evaluate_od_with_contract(model, dataset, device, output_dir, contract, *, gt_chunk_size=8,
                               tile_pixels=65536, progress_every=25, defer_mask_metrics=False):
    """Private fixture engine; partial contracts can never select a model."""
    _integer(gt_chunk_size, "gt_chunk_size")
    base._require(gt_chunk_size <= 8, "GT chunks must not exceed eight original instances")
    _integer(tile_pixels, "tile_pixels")
    _integer(progress_every, "progress_every", allow_zero=True)
    base._require(type(defer_mask_metrics) is bool, "defer_mask_metrics must be an explicit boolean")
    base._require(len(dataset) == len(contract["images"]), "Full VAL dataset length differs; subsets cannot select a model")
    base._require(getattr(dataset, "annotation_sha256", None) ==
                  contract["provenance"]["binding"]["current_annotation_sha256"], "Dataset annotation identity differs")
    configuration = getattr(dataset, "configuration", {})
    base._require(not bool(getattr(dataset, "augment", False)) and
                  not any(configuration.get(key, False) for key in ("augment", "padding", "external_padding", "crop")),
                  "VAL must use unaugmented, externally unpadded whole-frame RGB")
    base._require(isinstance(dataset, ODInputDataset) or not contract["official_full_val"],
                  "Official OD VAL requires ODInputDataset for original-RGB input reconstruction")
    base._require(base._sha(base.METRICS_PATH) == base.REFERENCE_PINS[base.METRICS_PATH],
                  "Versioned reference metric source changed")
    metrics = base._module(base.METRICS_PATH, "_apple_od_eval_reference_metrics")
    before, annotation_hashed = _source_snapshot(model, dataset, contract)
    output_dir = Path(output_dir).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    base._require(not any((output_dir / name).exists() for name in base.OUTPUT_FILES), "Use a fresh evaluation directory")
    cache = _DeferredMaskCache(output_dir) if defer_mask_metrics else None
    policy = _postprocess_policy(model, gt_chunk_size=gt_chunk_size, tile_pixels=tile_pixels,
                                 defer_mask_metrics=defer_mask_metrics)
    metadata_path = output_dir / "evaluation_metadata.json"
    metadata = {"schema": "apple_seg_complete_papple_val_evaluation_v1", "evaluation_family": "OD_ROI",
                "phase": "running", "created_utc": datetime.now(timezone.utc).isoformat(), "dataset_split": "val",
                "official_full_val": contract["official_full_val"], "model_selection_eligible": contract["official_full_val"],
                "device": str(device), "model_initialization_provenance": base._jsonable(getattr(model, "checkpoint_provenance", None)),
                "images": len(contract["images"]), "GT_instances": sum(len(rows) for rows in contract["truth"].values()),
                "empty_GT_images": sum(not rows for rows in contract["truth"].values()),
                "provenance": {"path": str(contract["provenance_path"]), "sha256": contract["provenance_sha256"]},
                "preprocess": base._jsonable(configuration), "postprocess_policy": policy,
                "source_sha256_before": before, "no_GT_repair_or_exclusion": True, "subset_filter": False,
                "test_data_read": False, "iou_threshold": .5,
                "mask_PR": "Independently computed exact visible/amodal mask IoU>=.5 matching; same BOX threshold; not AP",
                "annotation_source_files_hashed": annotation_hashed,
                "defer_mask_metrics": defer_mask_metrics,
                "deferred_mask_cache": cache.metadata if cache is not None else None}
    base._atomic_json(metadata_path, metadata)
    records, diagnostics, rgb_records, gt_identity, input_records = [], [], [], [], []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for index, frozen in enumerate(contract["images"]):
                item = dataset[index]
                base._require(isinstance(item, dict) and "target" in item, "Dataset item must contain original target")
                target, rows = item["target"], contract["truth"][frozen["image_id"]]
                truth = base._audit_target(target, frozen, rows)
                rgb = base._rgb_path(target)
                rgb_before = base._audit_rgb(rgb, frozen)
                source_digest = target["source"].get("rgb_sha256", target["source"].get("RGB_sha256"))
                base._require(source_digest is None or source_digest == rgb_before, "Dataset-consumed RGB fingerprint differs")
                pixels, pixel_mask = item.get("pixel_values"), item.get("pixel_mask")
                base._require(isinstance(pixels, torch.Tensor) and pixels.is_floating_point(), "Floating pixel_values required")
                pixels = pixels.unsqueeze(0) if pixels.ndim == 3 else pixels
                base._require(pixels.ndim == 4 and tuple(pixels.shape[:2]) == (1, 3)
                              and min(pixels.shape[-2:]) > 0 and bool(torch.isfinite(pixels).all()),
                              "Single whole-frame RGB tensor required")
                input_hw = tuple(pixels.shape[-2:])
                input_before = _audit_input(pixels, target, dataset, frozen, recompute_rgb=True)
                if pixel_mask is not None:
                    pixel_mask = pixel_mask.unsqueeze(0) if pixel_mask.ndim == 2 else pixel_mask
                    base._require(tuple(pixel_mask.shape) == (1, *input_hw) and bool((pixel_mask == 1).all()),
                                  "Only full-frame all-one pixel_mask is supported")
                    pixel_mask = pixel_mask.to(device=device)
                inference_pixels = pixels.to(device=device)
                outputs = model(inference_pixels, pixel_mask=pixel_mask)
                hw = (frozen["height"], frozen["width"])
                prediction = _validate_prediction(outputs, input_hw, hw, mask_chunk_size=gt_chunk_size)
                boxes, scores = prediction["boxes"].tolist(), prediction["scores"]
                records.append({"image_id": frozen["image_id"], "gt": {"boxes": truth["boxes"]},
                                "pred": {"boxes": boxes, "scores": scores}})
                match = metrics.match_image(truth["boxes"], boxes, scores, .5, 0.0)
                box_assignment = [-1] * len(scores)
                for p, g in zip(match["pred_indices"], match["gt_indices"], strict=True):
                    box_assignment[int(p)] = int(g)
                matrices = {}
                if cache is not None:
                    cache.store(index, frozen["image_id"], prediction)
                else:
                    matrices = _roi_mask_overlaps(prediction, truth, hw, gt_chunk_size=gt_chunk_size, tile_pixels=tile_pixels)
                diagnostic = {"image_id": frozen["image_id"], "gt_count": len(rows),
                              "apple_ids_in_target_order": list(target["instance_ids"]), "scores": scores,
                              "query_indices": prediction["prediction_indices"], "prediction_indices": prediction["prediction_indices"],
                              "query_indices_semantics": "compatibility alias for stable ROI output indices; no query model",
                              "labels": [1] * len(scores), "box_gt_indices_by_prediction": box_assignment,
                              "postprocess": {"policy": policy, "original_size_hw": list(hw), "input_size_hw": list(input_hw),
                                              "ROI_mask_size_hw": list(prediction["roi_hw"]), "kept_count": len(scores),
                                              "returned_masks": False, "invalid_candidates": []}}
                _set_mask_diagnostics(diagnostic, matrices, [] if cache is not None else list(range(len(scores))))
                base._audit_target(target, frozen, rows)
                input_after = _audit_input(inference_pixels, target, dataset, frozen, recompute_rgb=False)
                base._require(input_before["sha256"] == input_after["sha256"],
                              "Normalized model input changed during inference")
                input_records.append({**input_before, "before": input_before["sha256"], "after": input_after["sha256"]})
                rgb_after = base._audit_rgb(rgb, frozen)
                rgb_records.append({"image_id": frozen["image_id"], "path": str(rgb), "before": rgb_before, "after": rgb_after})
                gt_identity.append({"image_id": frozen["image_id"], "instance_ids": list(target["instance_ids"]),
                                    "original_hw": list(target["original_hw"]), "GT": rows, "source": base._jsonable(target["source"])})
                diagnostics.append(diagnostic)
                if progress_every and ((index + 1) % progress_every == 0 or index + 1 == len(contract["images"])):
                    print(f"OD VAL {index + 1}/{len(contract['images'])}: GT={sum(len(r['gt']['boxes']) for r in records)} "
                          f"predictions={sum(len(r['pred']['scores']) for r in records)}", flush=True)
                del outputs, prediction, matrices, truth, item, target, pixels, pixel_mask, inference_pixels
        calibrated = metrics.calibrate_threshold(records, .5, return_curve=True)
        confidence = calibrated["confidence"]
        replay = metrics.evaluate_at_threshold(records, confidence, .5)
        base._require(replay == calibrated["summary"], "Selected BOX threshold replay differs")
        base._require(replay["images"] == metadata["images"] and replay["gt_count"] == metadata["GT_instances"]
                      and replay["empty_gt_images"] == metadata["empty_GT_images"], "Consumed full scope differs")
        if cache is not None:
            computed_masks = _evaluate_deferred_masks(cache, dataset, contract, records, diagnostics, input_records,
                                                       confidence, gt_chunk_size=gt_chunk_size,
                                                       tile_pixels=tile_pixels, progress_every=progress_every)
        else:
            computed_masks = sum(len(row["scores"]) for row in diagnostics)
        base._require(all(row["mask_metrics_computed_by_prediction"] ==
                          ([score >= confidence for score in row["scores"]] if defer_mask_metrics
                           else [True] * len(row["scores"])) for row in diagnostics),
                      "Exact native mask computation scope differs from the selected BOX confidence")
        metadata["mask_metrics_scope"] = {
            "mode": "deferred_selected" if defer_mask_metrics else "online_all",
            "raw_predictions": sum(len(row["scores"]) for row in diagnostics),
            "selected_predictions": sum(score >= confidence for row in diagnostics for score in row["scores"]),
            "computed_predictions": computed_masks, "confidence": confidence,
            "unevaluated_assignment_sentinel": -1,
            "computed_flag": "mask_metrics_computed_by_prediction",
        }
        box_summary = {key: value for key, value in replay.items()
                       if key not in ("amodal_mask_iou_on_box_tp", "visible_mask_iou_on_box_tp")}
        for branch in ("visible", "amodal"):
            box_summary[branch + "_instance_mask_pr_iou50"] = base._mask_pr(
                diagnostics, branch + "_mask", confidence, replay["gt_count"], metrics)
        for saved, frozen in zip(rgb_records, contract["images"], strict=True):
            saved["after"] = base._audit_rgb(Path(saved["path"]), frozen)
        after = {path: base._sha(path) for path in before}
        base._require(before == after, "Consumed source/provenance/annotation files changed during evaluation")
        metadata.update(phase="complete", consumed_images=len(records), consumed_GT=replay["gt_count"],
                        source_sha256_after=after, source_before_after_identical=True,
                        RGB_sha256_before_after=rgb_records, all_RGB_before_after_verified=True,
                        normalized_input_sha256_before_after=input_records,
                        all_normalized_inputs_before_after_verified=True,
                        all_inputs_recomputed_from_original_RGB=all(row["recomputed_from_original_RGB"] for row in input_records),
                        all_original_GT_numeric_fields_and_mask_SHA256_verified=True, ground_truth_identity=gt_identity,
                        annotation_visible_area_exceeds_amodal_GT_retained=14 if contract["official_full_val"] else None)
        summary = {"schema": "apple_seg_papple_val_summary_v1", "evaluation_family": "OD_ROI", "phase": "complete",
                   "confidence": confidence, "score": min(replay["precision"], replay["recall"]) if contract["official_full_val"] else None,
                   "model_selection_eligible": contract["official_full_val"], "summary": box_summary,
                   "inference": {"dataset_split": "val", "preprocess": base._jsonable(configuration), "postprocess_policy": policy},
                   "calibration": {"selection_rule": calibrated["selection_rule"], "candidate_thresholds": calibrated["candidate_thresholds"],
                                   "scope": "All original VAL frames; complete distinct returned-score curve with whole score ties; no TEST tuning"},
                   "BOX_threshold_envelope": base._envelope(calibrated["curve"]),
                   "files": {name.removesuffix(".json"): str(output_dir / name) for name in base.OUTPUT_FILES}}
        base._atomic_json(output_dir / "raw_predictions.json", records)
        base._atomic_json(output_dir / "mask_pair_diagnostics.json", diagnostics)
        base._atomic_json(output_dir / "box_threshold_curve.json", calibrated["curve"])
        base._atomic_json(output_dir / "occlusion_diagnostics.json", base._occlusion(diagnostics, contract, confidence))
        if cache is not None:
            cache.cleanup()
        base._atomic_json(metadata_path, metadata)
        base._atomic_json(output_dir / "summary.json", summary)  # Completion marker must remain last.
        return summary
    except Exception as error:
        metadata.update(phase="failed", failure_type=type(error).__name__, failure=str(error), consumed_images=len(diagnostics))
        base._atomic_json(metadata_path, metadata)
        raise
    finally:
        model.train(was_training)


def evaluate_od(model, dataset, output_dir, *, device, provenance_path, gt_chunk_size=8,
                tile_pixels=65536, progress_every=25, defer_mask_metrics=False):
    """Evaluate all original 814 VAL images/3449 GT/13 empty images.

    The caller places the model on ``device``. Dataset items supply already
    normalized unpadded RGB and unchanged original CPU targets. Only RGB and
    pixel_mask enter model inference. There is no subset, confidence override,
    GT alteration, or test-data option. The previous model mode is restored.
    ``defer_mask_metrics=True`` losslessly caches float32 ROI probabilities,
    calibrates the unchanged complete BOX curve, then computes exact native mask
    matches only for the selected score prefix. Low-score masks are explicitly
    marked unevaluated; successful caches are removed before summary completion.
    """
    contract = base.load_papple_contract(provenance_path)
    return _evaluate_od_with_contract(model, dataset, device, output_dir, contract,
                                      gt_chunk_size=gt_chunk_size, tile_pixels=tile_pixels,
                                      progress_every=progress_every, defer_mask_metrics=defer_mask_metrics)
