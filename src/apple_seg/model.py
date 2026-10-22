"""Shared Mask2Former visible/amodal readouts and strict model-only warm start.

No Hub loader or legacy project imports are used. ``from_checkpoint`` constructs
on CPU and restores only model tensors; callers own device selection afterward.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
from importlib.metadata import version
from pathlib import Path
import re

import torch
from torch import nn


TRANSFORMERS_VERSION = "4.57.1"
PARENT_SCHEMA = "query_dual_papple_only_full_epoch_training_v1"
PARENT_RUN_ID = "a49fea73-dbfa-4c42-936c-a3449d6199d2"
PARENT_COMPLETED_EPOCH = 6
OUTPUT_KEYS = (
    "class_queries_logits",
    "visible_masks_queries_logits",
    "amodal_masks_queries_logits",
)


def _audit_config(config: Mapping) -> dict:
    if not isinstance(config, Mapping) or config.get("model_type") != "mask2former":
        raise ValueError("model_config must be an outer Mask2Former configuration")
    for key, expected in (("num_queries", 100), ("hidden_dim", 256), ("mask_feature_size", 256)):
        if type(config.get(key)) is not int or config[key] != expected:
            raise ValueError(f"Unsupported model_config dimension: {key}")
    backbone = config.get("backbone_config")
    if not isinstance(backbone, Mapping) or backbone.get("model_type") not in (None, "swin"):
        raise ValueError("A local Swin backbone_config is required")
    # The actual parent payload lacks nested model_type: the original HF
    # constructor popped that key. Identify the saved backbone from its actual
    # architecture fields and construct an explicit SwinConfig below.
    for key, expected in (("embed_dim", 96), ("depths", [2, 2, 6, 2]),
                          ("num_heads", [3, 6, 12, 24])):
        if backbone.get(key) != expected:
            raise ValueError(f"Expected parent Swin-Tiny backbone field: {key}")
    if config.get("backbone") is not None or config.get("use_pretrained_backbone") or config.get("use_timm_backbone"):
        raise ValueError("External pretrained/timm backbone loading is unsupported")
    labels = {int(key): value for key, value in config.get("id2label", {}).items()}
    if set(labels) != set(range(80)) or labels.get(47) != "apple" or config.get("label2id", {}).get("apple") != 47:
        raise ValueError("Preserve the parent outer COCO configuration and apple row47")
    return deepcopy(dict(config))


def _validate_parent_payload(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping) or payload.get("schema") != PARENT_SCHEMA:
        raise ValueError("Expected a complete original query training checkpoint")
    run = payload.get("run_config")
    if not isinstance(run, Mapping) or run.get("run_id") != PARENT_RUN_ID or run.get("schema") != PARENT_SCHEMA:
        raise ValueError("Checkpoint parent run identity differs")
    if type(payload.get("completed_epoch")) is not int or payload["completed_epoch"] != PARENT_COMPLETED_EPOCH:
        raise ValueError("Checkpoint must be the original completed epoch6")
    if payload.get("complete_train_epoch") is not True or payload.get("complete_full_val") is not True:
        raise ValueError("A complete TRAIN epoch and complete full VAL are required")
    if type(payload.get("micro_batches_in_accumulation")) is not int or payload["micro_batches_in_accumulation"] != 0:
        raise ValueError("Warm start requires a complete accumulation boundary")
    if not isinstance(payload.get("model"), Mapping) or not payload["model"]:
        raise ValueError("Checkpoint model state is absent")
    return _audit_config(payload.get("model_config"))


def _strict_load_model_state(model: nn.Module, state: Mapping) -> dict:
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    if missing or unexpected:
        raise RuntimeError(f"Strict model keys differ: missing={missing}, unexpected={unexpected}")
    for key, destination in expected.items():
        source = state[key]
        if not isinstance(source, torch.Tensor) or source.device.type != "cpu":
            raise TypeError(f"CPU checkpoint tensor required: {key}")
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise ValueError(f"Checkpoint tensor shape/dtype differs: {key}")
        if source.is_floating_point() and not bool(torch.isfinite(source).all()):
            raise ValueError(f"Nonfinite model checkpoint tensor: {key}")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Strict model restoration failed")
    for key, actual in model.state_dict().items():
        if actual.dtype != state[key].dtype or not torch.equal(actual, state[key]):
            raise RuntimeError(f"Restored tensor changed: {key}")
    return {"strict": True, "missing_keys": [], "unexpected_keys": [],
            "tensor_count": len(expected), "all_tensors_equal_to_checkpoint": True}


class DualMaskModel(nn.Module):
    """One image/query backbone, official visible masks and independent A MLP.

    Inputs are normalized floating RGB[1,3,H,W]. The module accepts arbitrary
    positive spatial dimensions supported by the backbone; the training driver
    fixes its experiment input size. No GT or externally padded batches enter
    the model. Outputs contain only the final class/V/A predictions.
    """

    def __init__(self, model_config: Mapping):
        super().__init__()
        self.model_config = _audit_config(model_config)
        if version("transformers") != TRANSFORMERS_VERSION:
            raise RuntimeError(f"This numerical path requires transformers=={TRANSFORMERS_VERSION}")
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, SwinConfig

        values = deepcopy(self.model_config)
        values["backbone_config"] = SwinConfig.from_dict(values["backbone_config"])
        config = Mask2FormerConfig.from_dict(values)
        if (config.backbone_config is None or config.backbone is not None
                or config.use_pretrained_backbone or config.use_timm_backbone):
            raise ValueError("Only config-local backbone construction is supported")
        self.constructed_model_config = deepcopy(config.to_dict())
        with torch.device("cpu"):
            self.shared = Mask2FormerForUniversalSegmentation(config).model.float().cpu()
        visible_embedder = self.shared.transformer_module.decoder.mask_predictor.mask_embedder
        self.amodal_mask_embedder = deepcopy(visible_embedder)
        self.class_predictor = nn.Linear(256, 2, device="cpu", dtype=torch.float32)
        self.checkpoint_provenance: dict | None = None
        self._check_readout_independence()

    def _check_readout_independence(self) -> None:
        visible = self.shared.transformer_module.decoder.mask_predictor.mask_embedder
        a_params, v_params = list(self.amodal_mask_embedder.parameters()), list(visible.parameters())
        if len(a_params) != len(v_params) or any(
            a.shape != v.shape or a.data_ptr() == v.data_ptr()
            for a, v in zip(a_params, v_params, strict=True)
        ):
            raise RuntimeError("Visible and amodal embedding parameters must be independent")

    @classmethod
    def from_checkpoint(cls, path: str | Path, expected_sha256: str) -> tuple["DualMaskModel", dict]:
        return from_checkpoint(path, expected_sha256)

    def forward(self, pixel_values, pixel_mask=None, *, targets=None) -> dict[str, torch.Tensor]:
        if targets is not None:
            raise ValueError("GT belongs in the separate criterion, never model.forward")
        if (not isinstance(pixel_values, torch.Tensor) or not pixel_values.is_floating_point()
                or pixel_values.ndim != 4 or tuple(pixel_values.shape[:2]) != (1, 3)
                or min(pixel_values.shape[-2:]) < 1):
            raise ValueError("Expected normalized floating RGB[1,3,H,W]")
        if pixel_mask is not None:
            if not isinstance(pixel_mask, torch.Tensor) or tuple(pixel_mask.shape) != (1, *pixel_values.shape[-2:]):
                raise ValueError("pixel_mask shape mismatch")
            if not bool((pixel_mask == 1).all()):
                raise ValueError("Only full-frame all-one pixel_mask is supported")
        outputs = self.shared(pixel_values=pixel_values, pixel_mask=pixel_mask,
                              output_hidden_states=True, output_attentions=False, return_dict=True)
        intermediates, masks = outputs.transformer_decoder_intermediate_states, outputs.masks_queries_logits
        pixels = outputs.pixel_decoder_last_hidden_state
        if not isinstance(intermediates, (tuple, list)) or not intermediates:
            raise RuntimeError("Missing normalized decoder query stages")
        if not isinstance(masks, (tuple, list)) or len(masks) != len(intermediates):
            raise RuntimeError("Visible masks and normalized decoder stages must align")
        normalized_qbc = intermediates[-1]
        if tuple(normalized_qbc.shape) != (100, 1, 256):
            raise RuntimeError("Expected normalized decoder queries[100,1,256]")
        query_bqc = normalized_qbc.transpose(0, 1)
        if pixels.ndim != 4 or tuple(pixels.shape[:2]) != (1, 256):
            raise RuntimeError("Expected pixel features[1,256,h,w]")
        visible = masks[-1]
        if tuple(visible.shape) != (1, 100, *pixels.shape[-2:]):
            raise RuntimeError("Official visible mask shape mismatch")
        amodal = torch.einsum("bqc,bchw->bqhw", self.amodal_mask_embedder(query_bqc), pixels)
        logits = self.class_predictor(query_bqc)
        return {"class_queries_logits": logits, "visible_masks_queries_logits": visible,
                "amodal_masks_queries_logits": amodal}


def from_checkpoint(path: str | Path, expected_sha256: str) -> tuple[DualMaskModel, dict]:
    """Verify a trusted local parent file and load its model tensors on CPU.

    The complete legacy training payload contains Python/NumPy RNG objects, so
    deserialization needs ``weights_only=False``. Its caller-supplied full-file
    SHA256 is checked *before* deserialization and again after restoration.
    Optimizer, scaler and saved RNG values are never restored or returned.
    """
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None:
        raise ValueError("expected_sha256 must be an explicit64-digit SHA256")
    # Opening and hashing verifies the actual file. Avoid Windows realpath's
    # extra handle-resolution permissions for otherwise readable local files.
    path = Path(path).expanduser().absolute()
    expected_sha256 = expected_sha256.lower()

    def file_sha() -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    if file_sha() != expected_sha256:
        raise ValueError("Checkpoint SHA256 differs; refusing deserialization")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    config = _validate_parent_payload(payload)
    model = DualMaskModel(config)
    strict = _strict_load_model_state(model, payload["model"])
    model._check_readout_independence()
    provenance = {
        "schema": "apple_seg_verified_query_parent_model_only_v1",
        "checkpoint_path": str(path), "checkpoint_sha256": expected_sha256,
        "parent_schema": payload["schema"], "parent_run_id": payload["run_config"]["run_id"],
        "parent_completed_epoch": payload["completed_epoch"],
        "complete_train_epoch": True, "complete_full_val": True,
        "model_config": deepcopy(config),
        "constructed_model_config": deepcopy(model.constructed_model_config),
        "strict_model_load": strict,
        "backbone_config_construction": "Explicit SwinConfig from saved fields; the actual parent omits nested model_type. Raw model_config remains unchanged.",
        "parent_runtime_versions": deepcopy(payload.get("runtime_versions", {})),
        "parent_source_sha256": deepcopy(payload.get("source_sha256", {})),
        "parent_architecture_metadata_record": deepcopy(payload.get("architecture_metadata", {})),
        "parent_architecture_record_note": "Historical payload fields are preserved verbatim, not reasserted as current execution facts.",
        "state_scope": "model_only", "optimizer_restored": False, "scaler_restored": False,
        "checkpoint_RNG_restored": False, "training_resume": False,
        "construction_device": "cpu", "runtime_device": "cpu",
        "AV_readout_parameters_independent": True,
        "output_class_mapping": {"apple": 0, "no_object": 1},
        "backbone_forwards_per_image": 1, "auxiliary_outputs": False,
        "direct_box_head": False,
    }
    del payload
    if file_sha() != expected_sha256:
        raise RuntimeError("Checkpoint bytes changed during restoration")
    model.checkpoint_provenance = deepcopy(provenance)
    return model, provenance
