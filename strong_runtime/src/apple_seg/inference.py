"""Load this project's trained checkpoints and predict with frozen calibration.

Only model tensors are restored. Formal optimizer/RNG state is retained in the
training checkpoint for resumption, and is never restored for inference.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re

import torch

from .model import DualMaskModel, _strict_load_model_state

TRAINING_SCHEMA = "apple_seg_mixed_update_boundary_v1"


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _audit_trained_payload(payload, *, require_calibration=True):
    if not isinstance(payload, Mapping) or payload.get("schema") != TRAINING_SCHEMA:
        raise ValueError("Expected a checkpoint from the current mixed-data trainer")
    if payload.get("checkpoint_kind") != "formal_update_boundary" or payload.get("update_boundary") is not True:
        raise ValueError("Inference requires a formal checkpoint at a complete update boundary")
    identity, updates = payload.get("identity"), payload.get("updates")
    if not isinstance(identity, Mapping) or identity.get("smoke") is not False:
        raise ValueError("Smoke checkpoints are not inference models")
    recipe = identity.get("recipe")
    if not isinstance(recipe, Mapping) or type(recipe.get("input_size")) is not int or recipe["input_size"] < 1:
        raise ValueError("Checkpoint must specify its actual training input size")
    refinement = recipe.get("occlusion_refinement", False)
    if type(refinement) is not bool:
        raise ValueError("Checkpoint occlusion_refinement must be a boolean")
    if not isinstance(updates, Mapping) or any(type(updates.get(key)) is not int or updates[key] < 0
                                            for key in ("attempted", "successful", "skipped")):
        raise ValueError("Checkpoint update counters are invalid")
    if updates["attempted"] != updates["successful"] + updates["skipped"]:
        raise ValueError("Checkpoint attempted/successful/skipped counters differ")
    if not isinstance(payload.get("model"), Mapping) or not payload["model"]:
        raise ValueError("Checkpoint model tensors are missing")

    # last.pt may remember a best score from an older model. Only a completed
    # validation of these exact weights can provide the inference confidence.
    validations = payload.get("validations", [])
    if not isinstance(validations, list):
        raise ValueError("Checkpoint validation history must be a list")
    matching = [point for point in validations if isinstance(point, Mapping)
                and type(point.get("successful_updates")) is int
                and point["successful_updates"] == updates["successful"]]
    calibration = matching[-1] if matching else None
    if calibration is not None:
        summary = calibration.get("summary", {})
        if not isinstance(summary, Mapping) or (summary.get("images"), summary.get("gt_count"),
                                                summary.get("empty_gt_images")) != (814, 3449, 13):
            raise ValueError("Inference calibration requires the complete PApple validation scope")
        for key in ("confidence", "score"):
            value = calibration.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid inference calibration " + key)
        for key in ("precision", "recall", "f1"):
            value = summary.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid calibration metric " + key)
        if not math.isclose(calibration["score"], min(summary["precision"], summary["recall"]), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Calibration model-selection score differs from its P/R")
    if require_calibration and calibration is None:
        raise ValueError("These weights have no completed full validation; use a validated checkpoint such as best.pt")
    return {"input_size": recipe["input_size"], "occlusion_refinement": refinement,
            "successful_updates": updates["successful"], "calibration": deepcopy(calibration)}


def load_trained_checkpoint(path, expected_sha256, *, require_calibration=True):
    """Hash a trusted local checkpoint, reconstruct its architecture on CPU.

    This accepts the trainer's own formal schema. The historical epoch6 parent
    continues to use model.from_checkpoint, whose original checks are intact.
    """
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None:
        raise ValueError("An explicit 64-digit checkpoint SHA256 is required")
    path = Path(path).expanduser().absolute()
    expected_sha256 = expected_sha256.lower()
    if file_sha(path) != expected_sha256:
        raise ValueError("Checkpoint SHA256 differs; refusing deserialization")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    metadata = _audit_trained_payload(payload, require_calibration=require_calibration)
    model = DualMaskModel(payload["model_config"])
    if metadata["occlusion_refinement"]:
        model.enable_occlusion_refinement()
    strict = _strict_load_model_state(model, payload["model"])
    model.eval()
    provenance = {"schema": "apple_seg_trained_inference_v1", "checkpoint_path": str(path),
                  "checkpoint_sha256": expected_sha256, **metadata, "strict_model_load": strict,
                  "state_scope": "model_only", "optimizer_restored": False,
                  "checkpoint_RNG_restored": False, "training_resume": False,
                  "parent_provenance": deepcopy(payload.get("parent_provenance")),
                  "training_source_sha256": deepcopy(payload.get("identity", {}).get("source_sha256", {})),
                  "inference_precision": "float32; same as full PApple validation"}
    del payload
    if file_sha(path) != expected_sha256:
        raise RuntimeError("Checkpoint bytes changed while loading")
    model.checkpoint_provenance = deepcopy(provenance)
    return model, provenance


def predict_image(model, image, provenance, *, device="cuda", mask_directory=None):
    """Predict one complete RGB image at its model's frozen validation threshold.

    Optional masks are written in bounded eight-query chunks. No full image
    batch of 100 visible/amodal masks is retained in system memory.
    """
    from PIL import Image
    from torch.nn.functional import interpolate
    from .data import _pixels
    from .evaluate import POSTPROCESS_PATH, REFERENCE_PINS, _module

    calibration = provenance.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("Prediction requires calibration bound to these checkpoint weights")
    confidence = calibration["confidence"]
    if model.training:
        raise ValueError("Prediction model must be in eval mode")
    dev = torch.device(device)
    if dev.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA inference are supported")
    for path, expected in REFERENCE_PINS.items():
        if file_sha(path) != expected:
            raise ValueError("Versioned validation postprocessing/metrics source changed")
    post = _module(POSTPROCESS_PATH, "_apple_inference_postprocess")
    image = Path(image).expanduser().absolute()
    with Image.open(image) as source:
        rgb = source.convert("RGB")
    width, height = rgb.size
    pixels = _pixels(rgb, provenance["input_size"]).unsqueeze(0).to(dev)
    model.to(dev)
    with torch.inference_mode(), torch.autocast(device_type=dev.type, enabled=False):
        outputs = model(pixels)
        predicted = post.postprocess_single_frame(outputs, (height, width), return_masks=False)
        selected = torch.nonzero(predicted["scores"] >= confidence).flatten().tolist()
        detections = [{"query_index": int(predicted["query_indices"][index]),
                       "label": "apple", "score": float(predicted["scores"][index]),
                       "amodal_box_xyxy": predicted["boxes"][index].tolist()} for index in selected]
        if mask_directory is not None:
            mask_directory = Path(mask_directory).expanduser().absolute()
            mask_directory.mkdir(parents=True, exist_ok=True)
            queries = [row["query_index"] for row in detections]
            for start in range(0, len(queries), post.MASK_CHUNK_SIZE):
                chosen = queries[start:start + post.MASK_CHUNK_SIZE]
                for branch in ("visible", "amodal"):
                    masks = post._binary_chunk(outputs[branch + "_masks_queries_logits"], chosen,
                                               (height, width), torch, interpolate)
                    for offset, query in enumerate(chosen):
                        path = mask_directory / f"query_{query:03d}_{branch}.png"
                        with path.open("xb") as stream:
                            Image.fromarray(masks[offset].numpy().astype("uint8") * 255).save(stream, format="PNG")
                        detections[start + offset][branch + "_mask"] = str(path)
                    del masks
    return {"schema": "apple_seg_single_image_prediction_v1", "image": str(image),
            "original_hw": [height, width], "input_size": provenance["input_size"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "successful_updates": provenance["successful_updates"], "confidence": confidence,
            "inference_precision": "float32", "confidence_operator": ">=",
            "detections": detections, "count": len(detections),
            "accuracy_measured": False, "postprocess": predicted["metadata"]}
