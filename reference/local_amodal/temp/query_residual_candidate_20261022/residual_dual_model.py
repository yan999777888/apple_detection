"""Residual geometry extension of the completed public-PApple dual-query run.

Import and local_residual_model_metadata use only the standard library. The
explicit loader constructs the verified old COCO model on CPU, strictly restores
only the completed six-epoch dual-mask model state, then attaches a zero-output
residual box head. This source alone establishes no runtime or quality result.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "temp" / "query_dual_mask_candidate_20261021"
PARENT_RUN = ROOT / "runs" / "query_dual_papple_train01_20261021"
PARENT_CHECKPOINT = PARENT_RUN / "checkpoints" / "epoch_006_attempt_1792605302005129400.pt"
PARENT_CHECKPOINT_SHA256 = "aac6f257bf43af3b945dd3469ba66214de99d7ae878e13e85ec7bef0e20b3227"
PARENT_SUPERVISOR = ROOT / "reports" / "query_dual_papple_train01_20261021_supervisor.json"
PARENT_RUN_ID = "a49fea73-dbfa-4c42-936c-a3449d6199d2"
PARENT_SCHEMA = "query_dual_papple_only_full_epoch_training_v1"
FROZEN_SOURCE_SHA256 = {
    "query_dual_mask.py": "cef59347445af673eceff65cd5d0ed7caa5769683c5e78eb7f4610deb2e48878",
    "query_model_runtime.py": "d3aae8748062bf76aca7639bf0c7e169c74cedde73269cedd30302e81df16d5e",
}
RESIDUAL_BOX_CONFIGURATION = {
    "query_features": 256, "anchor_features": 8, "input_features": 264,
    "hidden_features": [256, 256], "output_features": 4,
    "area_ratio_feature": False, "final_linear_zero_initialized": True,
    "anchor_mask_threshold": 0.5, "anchor_grid": "native_mask_grid_pixel_edges",
    "anchor_format": "normalized_cxcywh", "anchor_gradients": False,
    "empty_amodal_fallback": "visible_then_center_0.5_0.5_0.1_0.1",
    "empty_visible_fallback": "center_0.5_0.5_0.1_0.1",
    "center_delta_scale": 0.5, "size_log_scale": math.log(2.0),
    "minimum_size": 1e-6, "maximum_size": 1.0, "center_range": [0.0, 1.0],
    "auxiliary_stages": "penultimate_only", "shared_head_across_stages": True,
}


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _import_frozen(module_name):
    path = (BASE / (module_name + ".py")).resolve()
    expected = FROZEN_SOURCE_SHA256[path.name]
    if _sha256(path) != expected:
        raise ValueError("Frozen baseline source changed: " + str(path))
    cached = sys.modules.get(module_name)
    if cached is not None and Path(getattr(cached, "__file__", "")).resolve() != path:
        raise ImportError("A shadow module would replace the verified baseline: " + module_name)
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))
    module = importlib.import_module(module_name)
    if Path(module.__file__).resolve() != path or _sha256(path) != expected:
        raise ImportError("Imported module is not the pinned baseline: " + module_name)
    return module


def _verify_history(history):
    if not isinstance(history, list) or len(history) != 6:
        raise ValueError("Parent must have all six complete TRAIN/full-VAL epochs")
    for epoch, row in enumerate(history, start=1):
        summary = row.get("validation", {}).get("summary", {})
        order = row.get("train_order", [])
        if (row.get("epoch") != epoch or row.get("train_images") != 2304
                or len(order) != 2304 or len(set(order)) != 2304
                or summary.get("images") != 814 or summary.get("gt_count") != 3449
                or summary.get("empty_gt_images") != 13):
            raise ValueError("Parent history contains an incomplete/subset epoch")


def local_residual_model_metadata():
    """Verify fixed local sources and completed parent evidence without Torch.

    The PT payload is verified inside the explicit loader; this metadata proves
    its fixed byte identity and the committed complete-parent JSON evidence.
    """
    for name, expected in FROZEN_SOURCE_SHA256.items():
        if _sha256(BASE / name) != expected:
            raise ValueError("Frozen baseline source changed: " + name)
    config_path = PARENT_RUN / "run_config.json"
    index_path = PARENT_RUN / "checkpoint_index.json"
    summary_path = PARENT_RUN / "training_summary.json"
    config, index, summary = map(_read, (config_path, index_path, summary_path))
    supervisor = _read(PARENT_SUPERVISOR)
    supervisor_final = supervisor.get("final_evidence", {})
    if (supervisor.get("phase") != "complete" or supervisor.get("exit_code") != 0
            or supervisor_final.get("phase") != "verified"
            or supervisor_final.get("run_id") != PARENT_RUN_ID
            or supervisor_final.get("epochs_completed") != 6
            or supervisor_final.get("test_used") is not False):
        raise ValueError("Parent supervisor must confirm complete/exit0/final verified")
    if (config.get("schema") != PARENT_SCHEMA or config.get("run_id") != PARENT_RUN_ID
            or index.get("schema") != PARENT_SCHEMA or index.get("run_id") != PARENT_RUN_ID
            or summary.get("schema") != PARENT_SCHEMA or summary.get("run_id") != PARENT_RUN_ID
            or summary.get("phase") != "complete" or summary.get("epochs_completed") != 6
            or summary.get("test_used") is not False
            or summary.get("source_before_after_identical") is not True
            or summary.get("source_sha256_before") != summary.get("source_sha256_after")
            or summary.get("source_sha256_before") != config.get("source_sha256")
            or Path(summary.get("run_dir", "")).resolve() != PARENT_RUN.resolve()):
        raise ValueError("The fixed six-epoch parent run is not committed complete")
    entry = index.get("last", {})
    if (entry.get("epoch") != 6 or entry.get("run_id") != PARENT_RUN_ID
            or entry.get("sha256") != PARENT_CHECKPOINT_SHA256
            or Path(entry.get("path", "")).resolve() != PARENT_CHECKPOINT.resolve()
            or summary.get("last") != entry or supervisor_final.get("last") != entry):
        raise ValueError("Completed parent checkpoint commit identity mismatch")
    _verify_history(summary.get("history"))
    if _sha256(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise ValueError("Fixed own-run six-epoch checkpoint SHA256 mismatch")
    for path in (config_path, index_path, summary_path, PARENT_CHECKPOINT):
        if supervisor_final.get("output_sha256", {}).get(str(path)) != _sha256(path):
            raise ValueError("Parent evidence differs from supervisor verified output: " + str(path))
    runtime = _import_frozen("query_model_runtime")
    # The old runtime also imports its wrapper by name; bind it explicitly first.
    _import_frozen("query_dual_mask")
    evidence = runtime.local_model_metadata()
    evidence["source_sha256_expected"].update({
        str(BASE / name): digest for name, digest in FROZEN_SOURCE_SHA256.items()})
    evidence["source_sha256_expected"].update({
        str(path): _sha256(path) for path in (config_path, index_path, summary_path, PARENT_SUPERVISOR)})
    evidence["source_sha256_expected"][str(PARENT_CHECKPOINT)] = PARENT_CHECKPOINT_SHA256
    evidence["source_sha256_expected"].update({
        str(path): _sha256(path) for path in (Path(__file__).resolve(), HERE / "residual_dual_criterion.py")})
    evidence["parent_checkpoint"] = {
        "path": str(PARENT_CHECKPOINT), "sha256": PARENT_CHECKPOINT_SHA256,
        "run_id": PARENT_RUN_ID, "completed_epoch": 6, "parent_run_complete": True,
        "checkpoint_payload_verified": False, "optimizer_restored": False,
        "required_train_images_per_epoch": 2304,
        "required_val_images": 814, "required_val_gt_count": 3449,
        "state_scope": "model_only",
    }
    evidence["parent_supervisor"] = {
        "path": str(PARENT_SUPERVISOR), "sha256": _sha256(PARENT_SUPERVISOR),
        "phase": "complete", "exit_code": 0, "final_evidence_phase": "verified",
    }
    evidence["residual_box_configuration"] = deepcopy(RESIDUAL_BOX_CONFIGURATION)
    return evidence


def _validate_parent_payload(checkpoint, evidence):
    config = _read(PARENT_RUN / "run_config.json")
    validation = checkpoint.get("validation", {}).get("summary", {})
    updates = checkpoint.get("updates", {})
    if (checkpoint.get("schema") != PARENT_SCHEMA or checkpoint.get("run_config") != config
            or checkpoint.get("source_sha256") != config.get("source_sha256")
            or checkpoint.get("complete_train_epoch") is not True
            or checkpoint.get("complete_full_val") is not True
            or checkpoint.get("completed_epoch") != 6
            or checkpoint.get("micro_batches_in_accumulation") != 0
            or validation.get("images") != 814 or validation.get("gt_count") != 3449
            or validation.get("empty_gt_images") != 13
            or updates.get("attempted") != 3456
            or updates.get("successful", -1) + updates.get("skipped", -1) != 3456
            or len(checkpoint.get("update_history", [])) != 3456
            or checkpoint.get("model_config") != evidence["outer_config"]
            or not isinstance(checkpoint.get("model"), dict)):
        raise ValueError("Fixed own-run checkpoint is not a complete six-epoch model")
    _verify_history(checkpoint.get("history"))
    validation_files = checkpoint.get("validation_files_sha256", {})
    if not validation_files:
        raise ValueError("Parent full-VAL artifact hashes are required")
    for path, expected in validation_files.items():
        if _sha256(path) != expected:
            raise ValueError("Parent full-VAL evidence changed: " + path)


def load_residual_dual_model(device="cpu"):
    """Return (model, evidence); CPU strict parent restoration precedes wrapping.

    Numerical imports occur only here. This fixed trusted checkpoint is read
    with weights_only=False because it contains our trainer's metadata; only
    checkpoint['model'] is restored. Optimizer/scaler/RNG state is discarded.
    """
    evidence = local_residual_model_metadata()
    runtime = _import_frozen("query_model_runtime")
    base, official_evidence = runtime.load_verified_query_model(device="cpu")
    # The frozen trainer saved the old loader's post-construction outer_config.
    # HF from_dict mutates nested backbone_config fields during construction;
    # compare against that exact pinned-loader representation, not raw JSON.
    evidence["raw_outer_config_before_construction"] = deepcopy(evidence["outer_config"])
    expected_constructed_config = deepcopy(evidence["outer_config"])
    backbone_config = expected_constructed_config.get("backbone_config", {})
    if ("attn_implementation" in expected_constructed_config
            or "attn_implementation" in backbone_config
            or backbone_config.get("model_type") != "swin"):
        raise ValueError("Raw official config differs from the three audited constructor normalizations")
    expected_constructed_config["attn_implementation"] = None
    backbone_config["attn_implementation"] = None
    del backbone_config["model_type"]
    if official_evidence["outer_config"] != expected_constructed_config:
        raise ValueError("Pinned loader changed more than the three audited config normalizations")
    evidence["outer_config"] = deepcopy(official_evidence["outer_config"])
    evidence["parent_config_comparison"] = "exact equality with frozen verified loader post-construction outer_config"
    evidence["audited_config_normalizations"] = {
        "/attn_implementation": "absent -> null",
        "/backbone_config/attn_implementation": "absent -> null",
        "/backbone_config/model_type": "swin -> absent",
    }
    import torch
    from torch import nn

    if _sha256(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise ValueError("Own-run parent bytes changed before deserialization")
    checkpoint = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Own-run parent checkpoint must be a dictionary")
    _validate_parent_payload(checkpoint, evidence)
    loaded = base.load_state_dict(checkpoint["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Strict complete parent model restoration failed")
    del checkpoint
    if _sha256(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise ValueError("Own-run parent bytes changed during CPU restoration")
    parent_keys = set(base.state_dict())

    def mask_anchor(logits):
        # Geometry is a detached readout of binary predictions on the native
        # mask grid. Pixel edges [min,max+1] preserve one-pixel box widths.
        with torch.no_grad():
            binary = logits.detach().float().sigmoid() >= 0.5
            batch, queries, height, width = binary.shape
            occupied_x = binary.any(dim=-2)
            occupied_y = binary.any(dim=-1)
            x = torch.arange(width, device=logits.device, dtype=torch.float32)
            y = torch.arange(height, device=logits.device, dtype=torch.float32)
            x1 = torch.where(occupied_x, x, float(width)).amin(dim=-1) / width
            y1 = torch.where(occupied_y, y, float(height)).amin(dim=-1) / height
            x2 = torch.where(occupied_x, x + 1, 0.0).amax(dim=-1) / width
            y2 = torch.where(occupied_y, y + 1, 0.0).amax(dim=-1) / height
            anchor = torch.stack(((x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1), dim=-1)
            present = binary.flatten(-2).any(dim=-1)
            fallback = logits.new_tensor((0.5, 0.5, 0.1, 0.1), dtype=torch.float32)
            anchor = torch.where(present[..., None], anchor, fallback.expand(batch, queries, -1))
            return anchor, present

    class ResidualDualQueryCandidate(nn.Module):
        def __init__(self):
            super().__init__()
            # Preserve every original parameter key and use one backbone.
            self.shared = base.shared
            self.class_predictor = base.class_predictor
            self.amodal_mask_embedder = base.amodal_mask_embedder
            self.residual_box_head = nn.Sequential(
                nn.Linear(264, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4))
            nn.init.zeros_(self.residual_box_head[-1].weight)
            nn.init.zeros_(self.residual_box_head[-1].bias)
            self.candidate_metadata = {
                "schema": "query_dual_mask_residual_geometry_penultimate_aux_candidate_v1",
                "parent_checkpoint": deepcopy(evidence["parent_checkpoint"]),
                "source_architecture": "shared Mask2Former query backbone; visible attention; independent amodal MLP readout",
                "initial_state": "strict model-only restoration of fixed complete six-epoch public-PApple run",
                "box_head": deepcopy(RESIDUAL_BOX_CONFIGURATION),
                "box_output": "normalized cxcywh; xyxy conversion for loss remains unclipped",
                "head_initial_output": "zero residual around detached binary predicted-mask geometry",
                "auxiliary_outputs": "penultimate decoder only; same class/V/A/box heads",
                "backbone_forwards_per_image": 1,
                "ground_truth_used_by_model_forward": False,
                "quality_improvement_verified": False,
                "runtime_verification": "must be supplied by an independent runtime report",
            }

        def _read_stage(self, normalized_qbc, visible, pixels):
            if tuple(normalized_qbc.shape) != (100, 1, 256):
                raise RuntimeError("Expected normalized public decoder queries[100,1,256]")
            query_bqc = normalized_qbc.transpose(0, 1)
            if tuple(visible.shape) != (1, 100, *pixels.shape[-2:]):
                raise RuntimeError("Visible mask/query stage shape mismatch")
            amodal = torch.einsum("bqc,bchw->bqhw", self.amodal_mask_embedder(query_bqc), pixels)
            # Keep geometry in float32 under the caller's mixed precision.
            with torch.autocast(device_type=query_bqc.device.type, enabled=False):
                amodal_anchor, amodal_present = mask_anchor(amodal)
                visible_anchor, _ = mask_anchor(visible)
                anchor = torch.where(amodal_present[..., None], amodal_anchor, visible_anchor).detach()
                head_input = torch.cat((query_bqc.float(), anchor, visible_anchor.detach()), dim=-1)
                delta = self.residual_box_head(head_input)
                center = (anchor[..., :2] + 0.5*anchor[..., 2:]*delta[..., :2].tanh()).clamp(0.0, 1.0)
                size = (anchor[..., 2:]*torch.exp(math.log(2.0)*delta[..., 2:].tanh())).clamp(1e-6, 1.0)
                boxes = torch.cat((center, size), dim=-1)
            return {"class_queries_logits": self.class_predictor(query_bqc),
                    "visible_masks_queries_logits": visible,
                    "amodal_masks_queries_logits": amodal,
                    "amodal_boxes": boxes}

        def forward(self, pixel_values, pixel_mask=None, *, targets=None):
            if targets is not None:
                raise ValueError("Pass real TRAIN targets only to the separate criterion")
            if pixel_values.ndim != 4 or tuple(pixel_values.shape[:2]) != (1, 3):
                raise ValueError("Input must be normalized full-frame RGB[1,3,H,W]")
            if pixel_mask is not None:
                if tuple(pixel_mask.shape) != (1, *pixel_values.shape[-2:]):
                    raise ValueError("pixel_mask shape mismatch")
                if not bool((pixel_mask == 1).all()):
                    raise NotImplementedError("Single full image only; external padding is unsupported")
            shared = self.shared(pixel_values=pixel_values, pixel_mask=pixel_mask,
                                 output_hidden_states=True, output_attentions=False, return_dict=True)
            stages = shared.transformer_decoder_intermediate_states
            masks = shared.masks_queries_logits
            pixels = shared.pixel_decoder_last_hidden_state
            if (not isinstance(stages, (list, tuple)) or len(stages) < 2
                    or not isinstance(masks, (list, tuple)) or len(masks) != len(stages)):
                raise RuntimeError("Aligned final and penultimate decoder query/mask stages are required")
            if pixels.ndim != 4 or tuple(pixels.shape[:2]) != (1, 256):
                raise RuntimeError("Public pixel embeddings must be[1,256,h,w]")
            final = self._read_stage(stages[-1], masks[-1], pixels)
            final["auxiliary_outputs"] = [self._read_stage(stages[-2], masks[-2], pixels)]
            return final

    model = ResidualDualQueryCandidate().float().cpu()
    if (model.shared is not base.shared or model.class_predictor is not base.class_predictor
            or model.amodal_mask_embedder is not base.amodal_mask_embedder):
        raise RuntimeError("The restored backbone and mask/class heads must be reused by reference")
    if {key for key in model.state_dict() if not key.startswith("residual_box_head.")} != parent_keys:
        raise RuntimeError("Residual wrapper changed inherited state-dictionary keys")
    evidence["parent_checkpoint"].update(checkpoint_payload_verified=True,
        complete_train_epoch=True, complete_full_val=True, strict_model_load=True,
        strict_parent_model_load=True,
        missing_keys=list(loaded.missing_keys), unexpected_keys=list(loaded.unexpected_keys))
    evidence["official_initialization"] = official_evidence["initialization"]
    evidence["initialization"] = {
        "strict_parent_model_only": True, "strict_parent_model_load": True,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "optimizer_restored": False, "inherited_parameter_keys_preserved": True,
        "residual_final_linear_zero_initialized": True,
    }
    model.candidate_metadata["parent_checkpoint"] = deepcopy(evidence["parent_checkpoint"])
    for path, expected in evidence["source_sha256_expected"].items():
        if _sha256(path) != expected:
            raise ValueError("Consumed local model/parent evidence changed: " + path)
    evidence["construction_device"] = "cpu"
    evidence["runtime_device"] = str(device)
    return model.to(device=device), evidence
