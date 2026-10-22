"""Strict local CPU loader for the frozen, CPU-proven dual-query candidate.

No hub loader is used. Import is standard-library-only; numerical imports occur
only in load_verified_query_model. The caller owns resource and CUDA gates.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WEIGHTS = ROOT / "weights/query_dual_m2f_22c4a2f"
MODEL_ID = "facebook/mask2former-swin-tiny-coco-instance"
REVISION = "22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6"
WRAPPER_SHA = "cef59347445af673eceff65cd5d0ed7caa5769683c5e78eb7f4610deb2e48878"
MANIFEST_SHA = "59cd59085da26e856da0d7aec8f2d211a5667aae066215fc55de695f73c52d4f"
LOCAL_FILES = {
    "config.json": (79763, "86e17340334de554d19f0ed176ca97ada069b72a40dfc880db6d07043e9a56ec"),
    "preprocessor_config.json": (537, "e2ee60db2b38023f11dbbf473cdcb5da600c9ed7ba8c2c2b7de99be59b52ae3b"),
    "model.safetensors": (189998168, "3550e8b21381d082c4d12b06a4b7c5f0f4a530125eee4c5f264b1e5f43dfab80"),
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_model_metadata():
    """Verify actual local bytes and return metadata, without importing Torch."""
    sources = {str(HERE / "query_dual_mask.py"): WRAPPER_SHA,
               str(WEIGHTS / "verified_download.json"): MANIFEST_SHA}
    manifest = json.loads((WEIGHTS / "verified_download.json").read_text(encoding="utf-8"))
    if (manifest.get("model_id") != MODEL_ID or manifest.get("revision") != REVISION
            or manifest.get("wrapper_sha256") != WRAPPER_SHA):
        raise ValueError("Verified local official-model identity mismatch")
    for name, (size, digest) in LOCAL_FILES.items():
        path = WEIGHTS / name
        entry = manifest["files"][name]
        if (path.stat().st_size != size or entry["bytes"] != size or entry["sha256"] != digest
                or Path(entry["path"]).resolve() != path.resolve()):
            raise ValueError("Verified local-model file metadata mismatch: " + name)
        sources[str(path)] = digest
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError("Frozen wrapper or verified local official-model bytes changed")
    raw_config = json.loads((WEIGHTS / "config.json").read_text(encoding="utf-8"))
    from query_dual_mask import audit_outer_config
    specification = audit_outer_config(raw_config)
    return {"source_sha256_expected": sources, "outer_config": raw_config,
            "outer_config_specification": specification,
            "source_identity": {"model_id": MODEL_ID, "revision": REVISION,
                "local_weights_verified": True, "weights_sha256": LOCAL_FILES["model.safetensors"][1]}}


def load_verified_query_model(device="cpu"):
    """Same constructor, strict safetensors load and initialization as CPU v3.

    Construct the official complete 81-class object on CPU in float32, strict
    load every key, then reuse its model, copy classifier rows47/80 and create
    an independent equal amodal MLP. No from_pretrained/from_hub path exists.
    """
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1")
    evidence = local_model_metadata()
    if version("transformers") != "4.57.1":
        raise RuntimeError("Only the audited Transformers4.57.1 is supported")
    import torch
    from safetensors.torch import load_file
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
    from query_dual_mask import wrap_audited_pretrained
    config = Mask2FormerConfig.from_dict(evidence["outer_config"])
    if (config.backbone_config is None or config.backbone is not None
            or config.use_pretrained_backbone or config.use_timm_backbone):
        raise ValueError("Only local config construction without a pretrained/timm backbone is supported")
    pretrained = Mask2FormerForUniversalSegmentation(config).to(device="cpu", dtype=torch.float32)
    state = load_file(str(WEIGHTS / "model.safetensors"), device="cpu")
    key_evidence = {"strict": True,
        "missing_keys": sorted(set(pretrained.state_dict()) - set(state)),
        "unexpected_keys": sorted(set(state) - set(pretrained.state_dict()))}
    loaded = pretrained.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Strict official safetensors loading failed")
    del state
    candidate = wrap_audited_pretrained(pretrained, source_identity=evidence["source_identity"]).float().cpu()
    visible = candidate.shared.transformer_module.decoder.mask_predictor.mask_embedder
    for destination, source in ((0, 47), (1, 80)):
        if not (torch.equal(candidate.class_predictor.weight[destination], pretrained.class_predictor.weight[source])
                and torch.equal(candidate.class_predictor.bias[destination], pretrained.class_predictor.bias[source])):
            raise RuntimeError("Audited classifier-row initialization changed")
    for amodal, original in zip(candidate.amodal_mask_embedder.parameters(), visible.parameters(), strict=True):
        if not torch.equal(amodal, original) or amodal.data_ptr() == original.data_ptr():
            raise RuntimeError("A/V MLP initialization must be equal and independent")
    if candidate.shared is not pretrained.model:
        raise RuntimeError("The official image/query model must be shared by reference")
    evidence["strict_load"] = key_evidence
    evidence["initialization"] = {"copied_classifier_rows": [47, 80],
        "AV_MLP_equal_and_independent": True, "shared_model_by_reference": True}
    if any(sha(path) != digest for path, digest in evidence["source_sha256_expected"].items()):
        raise RuntimeError("Consumed wrapper/official bytes changed during CPU construction")
    candidate = candidate.to(device=device)
    evidence["construction_device"] = "cpu"
    evidence["runtime_device"] = str(device)
    return candidate, evidence
