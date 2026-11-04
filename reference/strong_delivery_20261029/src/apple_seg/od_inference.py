"""Strict, self-contained inference for formal OD checkpoint sidecars.

The model is reconstructed from its complete constructor configuration rather
than a parent checkpoint. A threshold belongs only to the exact saved update
that completed the original 814-image validation. Smoke checkpoints are never
accepted. Prediction retains compact ROI masks; full-resolution exports use
one binary crop at a time and never an N-mask floating image batch.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw
import torch

from . import od_training
from .data import NORMALIZATION_MEAN, NORMALIZATION_STD
from .evaluate import PROVENANCE_SHA256, _jsonable
from .od_data import LEGACY_RGB_PREPROCESS, legacy_rgb_tensor
from .od_evaluate import POSTPROCESS_POLICY, _validate_prediction
from .od_model import build_od_amodal_detector
from .od_roi import paste_roi_mask


CONSTRUCTOR_FIELDS = frozenset((
    "mask_roi_size", "mask_hidden_channels", "roi_chunk_size", "mask_projection_chunk_size",
    "score_thresh", "detections_per_img", "pre_mask_nms_topk", "checkpoint_roi", "occlusion_weight",
    "focal_gamma", "use_visible_gate", "use_amodal_refinement", "final_nms", "rpn_pre_nms_top_n_train",
    "rpn_pre_nms_top_n_test", "rpn_post_nms_top_n_train", "rpn_post_nms_top_n_test",
    "box_batch_size_per_image", "box_positive_fraction", "anchor_sizes", "amodal_base",
    "hidden_tower_mode", "zero_initialize_hidden_predictor", "unknown_amodal_classifier_policy",
))
HIDDEN_TOWER_IDENTITY_FIELDS = frozenset(("hidden_tower_mode", "zero_initialize_hidden_predictor"))
CONTEXT_CONSTRUCTOR_FIELDS = frozenset((
    "context_cascade", "context_expansion", "cascade_iou_thresholds",
))
CONTEXT_EVALUATION_SOURCE_FILES = ("od_context.py", "od_foreground.py", "od_context_model.py")
VISIBLE_PROPOSAL_CONSTRUCTOR_FIELDS = frozenset(("visible_proposals", "visible_proposal_limit"))
SHALLOW_DETAIL_CONSTRUCTOR_FIELDS = frozenset(("shallow_detail",))
MASK_GEOMETRY_CONSTRUCTOR_FIELDS = frozenset(("mask_box_refinement",))
MASK_GEOMETRY_EVALUATION_SOURCE_FILES = ("od_mask_geometry.py", "od_mask_geometry_model.py")
VERIFIER_CONSTRUCTOR_FIELDS = frozenset(("occlusion_verifier",))
HIGHRES_CONSTRUCTOR_FIELDS = frozenset(("classification_highres",))
# A non-default backbone is recorded explicitly; resnet50 keeps the historical
# key set so every existing checkpoint still passes this audit unchanged.
BACKBONE_CONSTRUCTOR_FIELDS = frozenset(("backbone_name",))
DROPPATH_CONSTRUCTOR_FIELDS = frozenset(("backbone_stochastic_depth",))
BACKBONE_NAMES = ("resnet50", "convnext_small", "convnext_base", "convnext_large",
                  "swin_s", "swin_b")
VERIFIER_EVALUATION_SOURCE_FILES = ("od_verifier.py",)
SELECTION_RULE = "validation max(min(P,R)); tie max(F1); tie higher confidence"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value, name):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             name + " must be a lowercase SHA256")
    return value


def _number(value, name, *, minimum=0., maximum=None):
    _require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
             and value >= minimum and (maximum is None or value <= maximum), "Invalid " + name)
    return value


def _integer(value, name, *, minimum=1):
    _require(type(value) is int and value >= minimum, "Invalid integer " + name)
    return value


def _audit_context_fields(configuration):
    _require(configuration["context_cascade"] is True,
             "Saved context_cascade must be the explicit boolean True")
    expansion = _number(configuration["context_expansion"], "context_expansion", maximum=4.)
    _require(expansion > 1., "Invalid context_expansion: must be in (1, 4]")
    thresholds = configuration["cascade_iou_thresholds"]
    _require(isinstance(thresholds, list) and len(thresholds) == 3,
             "Saved cascade_iou_thresholds must be a list of three values")
    for value in thresholds:
        _number(value, "cascade_iou_thresholds", minimum=.5, maximum=.9)
    _require(thresholds[0] < thresholds[1] < thresholds[2],
             "Saved cascade_iou_thresholds must be strictly increasing")


def _audit_visible_proposal_fields(configuration):
    _require(configuration["visible_proposals"] is True,
             "Saved visible_proposals must be the explicit boolean True")
    limit = _integer(configuration["visible_proposal_limit"], "visible_proposal_limit")
    _require(limit <= 1000, "Invalid visible_proposal_limit: maximum is 1000")


def _audit_shallow_detail_fields(configuration):
    _require(configuration["shallow_detail"] is True,
             "Saved shallow_detail must be the explicit boolean True")


def _audit_mask_geometry_fields(configuration):
    _require(configuration["mask_box_refinement"] is True,
             "Saved mask_box_refinement must be the explicit boolean True")
    _require(configuration.get("amodal_base") is True,
             "Saved mask_box_refinement requires amodal_base=True")


def _audit_verifier_fields(configuration):
    _require(configuration["occlusion_verifier"] is True,
             "Saved occlusion_verifier must be the explicit boolean True")


def _audit_constructor(configuration):
    _require(isinstance(configuration, dict) and set(configuration) in
             (CONSTRUCTOR_FIELDS, CONSTRUCTOR_FIELDS | CONTEXT_CONSTRUCTOR_FIELDS,
              CONSTRUCTOR_FIELDS | CONTEXT_CONSTRUCTOR_FIELDS | VISIBLE_PROPOSAL_CONSTRUCTOR_FIELDS,
              CONSTRUCTOR_FIELDS | CONTEXT_CONSTRUCTOR_FIELDS | VISIBLE_PROPOSAL_CONSTRUCTOR_FIELDS
              | SHALLOW_DETAIL_CONSTRUCTOR_FIELDS,
              CONSTRUCTOR_FIELDS | CONTEXT_CONSTRUCTOR_FIELDS | VISIBLE_PROPOSAL_CONSTRUCTOR_FIELDS
              | SHALLOW_DETAIL_CONSTRUCTOR_FIELDS | MASK_GEOMETRY_CONSTRUCTOR_FIELDS,
              CONSTRUCTOR_FIELDS | CONTEXT_CONSTRUCTOR_FIELDS | VISIBLE_PROPOSAL_CONSTRUCTOR_FIELDS
              | SHALLOW_DETAIL_CONSTRUCTOR_FIELDS | MASK_GEOMETRY_CONSTRUCTOR_FIELDS
              | VERIFIER_CONSTRUCTOR_FIELDS,
              CONSTRUCTOR_FIELDS | CONTEXT_CONSTRUCTOR_FIELDS | VISIBLE_PROPOSAL_CONSTRUCTOR_FIELDS
              | SHALLOW_DETAIL_CONSTRUCTOR_FIELDS | HIGHRES_CONSTRUCTOR_FIELDS,
              CONSTRUCTOR_FIELDS | VERIFIER_CONSTRUCTOR_FIELDS,
              CONSTRUCTOR_FIELDS | VERIFIER_CONSTRUCTOR_FIELDS | BACKBONE_CONSTRUCTOR_FIELDS,
              CONSTRUCTOR_FIELDS | VERIFIER_CONSTRUCTOR_FIELDS | BACKBONE_CONSTRUCTOR_FIELDS
              | DROPPATH_CONSTRUCTOR_FIELDS),
             "Checkpoint must contain the complete exact OD constructor_config")
    result = deepcopy(configuration)
    if CONTEXT_CONSTRUCTOR_FIELDS.issubset(result):
        _audit_context_fields(result)
    if VISIBLE_PROPOSAL_CONSTRUCTOR_FIELDS.issubset(result):
        _audit_visible_proposal_fields(result)
    if SHALLOW_DETAIL_CONSTRUCTOR_FIELDS.issubset(result):
        _audit_shallow_detail_fields(result)
    if MASK_GEOMETRY_CONSTRUCTOR_FIELDS.issubset(result):
        _audit_mask_geometry_fields(result)
    if VERIFIER_CONSTRUCTOR_FIELDS.issubset(result):
        _audit_verifier_fields(result)
    if BACKBONE_CONSTRUCTOR_FIELDS.issubset(result):
        _require(result["backbone_name"] in BACKBONE_NAMES,
                 "Saved backbone_name must be resnet50, convnext_small or convnext_base")
    if DROPPATH_CONSTRUCTOR_FIELDS.issubset(result):
        _number(result["backbone_stochastic_depth"], "backbone_stochastic_depth", maximum=.5)
        _require(result.get("backbone_name", "resnet50") != "resnet50",
                 "Saved stochastic depth requires a replaced backbone")
    if HIGHRES_CONSTRUCTOR_FIELDS.issubset(result):
        _require(result["classification_highres"] is True,
                 "Saved classification_highres must be the explicit boolean True")
    booleans = ("checkpoint_roi", "use_visible_gate", "use_amodal_refinement", "amodal_base",
                "zero_initialize_hidden_predictor")
    integers = ("mask_roi_size", "mask_hidden_channels", "roi_chunk_size", "mask_projection_chunk_size",
                "detections_per_img", "pre_mask_nms_topk", "rpn_pre_nms_top_n_train", "rpn_pre_nms_top_n_test",
                "rpn_post_nms_top_n_train", "rpn_post_nms_top_n_test", "box_batch_size_per_image")
    for key in booleans:
        _require(type(result[key]) is bool, "Constructor boolean required: " + key)
    for key in integers:
        _integer(result[key], key)
    _require(result["mask_roi_size"] == 28, "The saved OD mask ROI size must be 28")
    _number(result["score_thresh"], "score_thresh", maximum=1.)
    _number(result["box_positive_fraction"], "box_positive_fraction", maximum=1.)
    for key in ("occlusion_weight", "focal_gamma"):
        _number(result[key], key)
    _require(result["final_nms"] in ("dual_mask", "box"), "Unknown saved OD NMS policy")
    _require(type(result["hidden_tower_mode"]) is str
             and result["hidden_tower_mode"] in ("shared_visible", "independent"),
             "Unknown saved OD hidden_tower_mode")
    _require(type(result["unknown_amodal_classifier_policy"]) is str
             and result["unknown_amodal_classifier_policy"] in
             ("legacy_box_iou", "ignore_overlapping_visible"),
             "Unknown saved OD unknown_amodal_classifier_policy")
    _require(isinstance(result["anchor_sizes"], list) and len(result["anchor_sizes"]) == 5,
             "Saved OD anchors must identify P2-P6")
    for value in result["anchor_sizes"]:
        _integer(value, "anchor size")
    _require(result["use_amodal_refinement"] or result["amodal_base"],
             "No-cascade weights require their independent amodal base")
    return result


def _verify_sources(expected):
    _require(isinstance(expected, dict) and set(expected) == set(od_training.SOURCE_FILES),
             "Checkpoint lacks the complete pinned OD training source inventory")
    for name, digest in expected.items():
        _digest(digest, "source " + name)
        _require(_sha(Path(__file__).with_name(name)) == digest,
                 "Consumed OD source differs from checkpoint: " + name)


def _audit_counts(summary, confidence, score):
    _require(isinstance(summary, dict) and
             (summary.get("images"), summary.get("gt_count"), summary.get("empty_gt_images")) == (814, 3449, 13),
             "Calibration requires all 814 images, 3449 GT and 13 empty images")
    for name in ("tp", "fp", "fn", "prediction_count"):
        _integer(summary.get(name), name, minimum=0)
    tp, fp = summary["tp"], summary["fp"]
    _require(tp + summary["fn"] == 3449 and summary["prediction_count"] == tp + fp,
             "Calibration counts are inconsistent")
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / 3449
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    for name, expected in (("precision", precision), ("recall", recall), ("f1", f1),
                            ("confidence", confidence), ("iou_threshold", .5)):
        value = _number(summary.get(name), "calibration " + name, maximum=1.)
        _require(math.isclose(value, expected, rel_tol=0, abs_tol=1e-12), "Calibration " + name + " differs")
    _require(math.isclose(score, min(precision, recall), rel_tol=0, abs_tol=1e-12),
             "Calibration score differs from its P/R")


def _validation_for_weights(payload, checkpoint_path, validation_root, *, required):
    points = payload.get("validations")
    _require(isinstance(points, list), "Checkpoint validation history is absent")
    matches = [point for point in points if isinstance(point, dict) and point.get("update") == payload["updates"]]
    if not matches:
        _require(not required, "These OD weights have no matching completed full validation; another update's confidence is forbidden")
        return None, None
    _require(len(matches) == 1, "Ambiguous validation records for the saved OD update")
    point = deepcopy(matches[0])
    _require(point.get("stage") == payload["stage"], "Validation stage differs from saved weights")
    confidence = _number(point.get("confidence"), "validation confidence", maximum=1.)
    score = _number(point.get("score"), "validation score", maximum=1.)
    _audit_counts(point.get("summary"), confidence, score)
    directory = point.get("directory")
    _require(isinstance(directory, str) and directory, "Validation source directory is missing")
    root = Path(validation_root).expanduser().resolve() if validation_root is not None else checkpoint_path.parent.resolve()
    relative = Path(directory)
    destination = (root / relative).resolve()
    _require(not relative.is_absolute() and destination.is_relative_to(root), "Validation directory escapes its run root")
    paths = {"summary": destination / "summary.json", "metadata": destination / "evaluation_metadata.json"}
    _require(all(path.is_file() for path in paths.values()), "Matching complete validation artifacts are required")
    hashes = {name: _sha(path) for name, path in paths.items()}
    summary = json.loads(paths["summary"].read_text(encoding="utf-8-sig"))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8-sig"))
    _require(summary.get("schema") == "apple_seg_papple_val_summary_v1" and
             summary.get("evaluation_family") == "OD_ROI" and summary.get("phase") == "complete"
             and summary.get("model_selection_eligible") is True,
             "Matching validation summary must be complete and model-selection eligible")
    _require(summary.get("summary") == point["summary"] and summary.get("confidence") == confidence
             and summary.get("score") == score, "Validation artifacts differ from the saved update's calibration")
    _require(summary.get("calibration", {}).get("selection_rule") == SELECTION_RULE,
             "Validation threshold selection rule differs")
    _require(metadata.get("schema") == "apple_seg_complete_papple_val_evaluation_v1"
             and metadata.get("evaluation_family") == "OD_ROI" and metadata.get("phase") == "complete"
             and metadata.get("official_full_val") is True and metadata.get("model_selection_eligible") is True
             and metadata.get("dataset_split") == "val"
             and (metadata.get("images"), metadata.get("GT_instances"), metadata.get("empty_GT_images")) == (814, 3449, 13)
             and (metadata.get("consumed_images"), metadata.get("consumed_GT")) == (814, 3449),
             "Matching validation metadata must prove complete original VAL scope")
    for flag in ("source_before_after_identical", "all_RGB_before_after_verified",
                 "all_original_GT_numeric_fields_and_mask_SHA256_verified", "no_GT_repair_or_exclusion",
                 "annotation_source_files_hashed", "all_normalized_inputs_before_after_verified",
                 "all_inputs_recomputed_from_original_RGB"):
        _require(metadata.get(flag) is True, "Validation audit is incomplete: " + flag)
    _require(metadata.get("subset_filter") is False and metadata.get("test_data_read") is False,
             "Validation cannot use a subset or TEST")
    sources = metadata.get("source_sha256_before")
    _require(isinstance(sources, dict) and sources and sources == metadata.get("source_sha256_after"),
             "Validation source hash audit differs")
    required_sources = ("od_model.py", "od_roi.py", "od_data.py", "od_evaluate.py", "data.py", "evaluate.py")
    if payload.get("constructor_config", {}).get("context_cascade") is True:
        required_sources += CONTEXT_EVALUATION_SOURCE_FILES
    if payload.get("constructor_config", {}).get("visible_proposals") is True:
        required_sources += ("od_proposals.py",)
    if payload.get("constructor_config", {}).get("shallow_detail") is True:
        required_sources += ("od_detail.py",)
    if payload.get("constructor_config", {}).get("mask_box_refinement") is True:
        required_sources += MASK_GEOMETRY_EVALUATION_SOURCE_FILES
    if payload.get("constructor_config", {}).get("occlusion_verifier") is True:
        required_sources += VERIFIER_EVALUATION_SOURCE_FILES
    for name in required_sources:
        consumed = [value for path, value in sources.items() if Path(path).name == name]
        expected = payload["identity"]["source_sha256"].get(name)
        _require(expected is not None and consumed and all(value == expected for value in consumed),
                 "Validation consumed different OD source: " + name)
    _require(metadata.get("annotation_visible_area_exceeds_amodal_GT_retained") == 14,
             "Validation must retain all 14 original visible-area anomalies")
    _require(metadata.get("provenance", {}).get("sha256") == PROVENANCE_SHA256
             == payload["identity"].get("papple_provenance_sha256"), "Original VAL provenance differs")
    recipe = payload["identity"]["recipe"]
    preprocessing = metadata.get("preprocess", {})
    _require(preprocessing.get("short_sides") == [recipe["validation_short_side"]]
             and preprocessing.get("max_size") == recipe["validation_max_size"]
             and preprocessing.get("normalization_mean") == list(NORMALIZATION_MEAN)
             and preprocessing.get("normalization_std") == list(NORMALIZATION_STD)
             and preprocessing.get("rgb_scale") == "uint8/255"
             and all(preprocessing.get(key) is False for key in ("augment", "padding", "external_padding", "crop")),
             "Validation preprocessing differs from the checkpoint inference recipe")
    _require(all(preprocessing.get(key) == value for key, value in LEGACY_RGB_PREPROCESS.items()),
             "Validation legacy resize backend or normalization order differs")
    _require(summary.get("inference", {}).get("preprocess") == preprocessing,
             "Validation summary and metadata preprocessing differ")
    for name, path in paths.items():
        _require(_sha(path) == hashes[name], "Validation artifact changed while loading")
    source = {name: {"path": str(paths[name]), "sha256": hashes[name]} for name in paths}
    return point, source


def _strict_restore(model, state):
    _require(isinstance(state, Mapping), "OD checkpoint tensor state is absent")
    expected = model.state_dict()
    _require(set(state) == set(expected), "Strict OD model tensor keys differ")
    for name, target in expected.items():
        value = state[name]
        _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                 and value.shape == target.shape and value.dtype == target.dtype,
                 "Strict OD tensor shape/dtype differs: " + name)
        _require(not value.is_floating_point() or bool(torch.isfinite(value).all()), "Nonfinite OD tensor: " + name)
    model.load_state_dict(state, strict=True)
    _require(all(torch.equal(value, state[name]) for name, value in model.state_dict().items()),
             "Restored OD tensors are not exactly equal")
    return dict(strict=True, tensor_count=len(expected), missing_keys=[], unexpected_keys=[],
                all_tensors_equal_to_checkpoint=True)


def load_od_checkpoint(path, expected_sha256=None, *, validation_root=None, require_validation=True):
    """Rebuild a formal OD checkpoint on CPU without its parent or optimizer.

    The adjacent ``<checkpoint suffix>.sha256`` is mandatory. An optional
    explicit digest must agree with it. By default the exact saved update must
    have complete eligible validation artifacts under its original run root;
    ``validation_root`` supports moving the checkpoint separately. Disabling
    validation is for model inspection only: prediction still refuses it.
    """
    path = Path(path).expanduser().absolute()
    _require(type(require_validation) is bool, "require_validation must be an explicit boolean")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    _require(sidecar.is_file(), "OD checkpoint SHA256 sidecar is required")
    digest = _digest(sidecar.read_text(encoding="ascii").strip(), "checkpoint sidecar")
    if expected_sha256 is not None:
        _require(isinstance(expected_sha256, str), "Explicit checkpoint SHA256 must be a string")
        _require(_digest(expected_sha256.lower(), "explicit checkpoint hash") == digest, "Explicit and sidecar checkpoint hashes differ")
    _require(_sha(path) == digest, "OD checkpoint hash differs")
    payload = od_training.load_checkpoint(path)
    _require(payload.get("smoke") is False and payload.get("identity", {}).get("smoke") is False
             and payload.get("phase") != "smoke_complete", "Smoke checkpoints are not eligible for OD inference")
    updates = _integer(payload.get("updates"), "saved update", minimum=0)
    _require(payload.get("stage") in ("joint", "classifier"), "Unknown saved OD training stage")
    version = payload.get("code_version")
    _require(isinstance(version, dict) and version.get("consumed_files_match_commit") is True
             and version.get("uncommitted_consumed_files") == []
             and isinstance(version.get("training_commit"), str)
             and re.fullmatch(r"[0-9a-f]{40}", version["training_commit"]) is not None,
             "Formal OD training code_version is required")
    identity = payload["identity"]
    _verify_sources(identity.get("source_sha256"))
    recipe = identity.get("recipe")
    _require(isinstance(recipe, dict), "Saved validation preprocessing recipe is missing")
    _integer(recipe.get("validation_short_side"), "validation_short_side")
    _integer(recipe.get("validation_max_size"), "validation_max_size")
    configuration = _audit_constructor(payload.get("constructor_config"))
    experiment_model = identity.get("model")
    _require(isinstance(experiment_model, dict) and set(experiment_model).issubset(configuration),
             "Checkpoint experiment model configuration is absent or incompatible")
    _require(HIDDEN_TOWER_IDENTITY_FIELDS.issubset(experiment_model),
             "Checkpoint experiment must explicitly identify hidden_tower_mode and zero_initialize_hidden_predictor")
    _require(type(experiment_model["hidden_tower_mode"]) is str
             and type(experiment_model["zero_initialize_hidden_predictor"]) is bool,
             "Checkpoint experiment hidden tower fields require an explicit string and boolean")
    _require(type(experiment_model.get("unknown_amodal_classifier_policy")) is str,
             "Checkpoint experiment must explicitly identify unknown_amodal_classifier_policy")
    if CONTEXT_CONSTRUCTOR_FIELDS.issubset(configuration):
        _require(CONTEXT_CONSTRUCTOR_FIELDS.issubset(experiment_model),
                 "Checkpoint experiment must explicitly identify all context constructor fields")
        _audit_context_fields(experiment_model)
    if VISIBLE_PROPOSAL_CONSTRUCTOR_FIELDS.issubset(configuration):
        _require(VISIBLE_PROPOSAL_CONSTRUCTOR_FIELDS.issubset(experiment_model),
                 "Checkpoint experiment must explicitly identify all visible proposal constructor fields")
        _audit_visible_proposal_fields(experiment_model)
    if SHALLOW_DETAIL_CONSTRUCTOR_FIELDS.issubset(configuration):
        _require(SHALLOW_DETAIL_CONSTRUCTOR_FIELDS.issubset(experiment_model),
                 "Checkpoint experiment must explicitly identify shallow_detail")
        _audit_shallow_detail_fields(experiment_model)
    if MASK_GEOMETRY_CONSTRUCTOR_FIELDS.issubset(configuration):
        _require(MASK_GEOMETRY_CONSTRUCTOR_FIELDS.issubset(experiment_model),
                 "Checkpoint experiment must explicitly identify mask_box_refinement")
        _audit_mask_geometry_fields(experiment_model)
    _require(all(configuration[name] == value for name, value in experiment_model.items()
                 if not (name == "amodal_base" and value is None)),
             "Saved constructor flags differ from the validated experiment")
    calibration, validation_source = _validation_for_weights(payload, path, validation_root, required=require_validation)
    model = build_od_amodal_detector(weights=None, **configuration)
    _require(model.constructor_config == configuration, "Reconstructed OD constructor configuration differs")
    strict = _strict_restore(model, payload.get("model"))
    model.set_training_stage(payload["stage"])
    model.requires_grad_(False).eval()
    provenance = dict(schema="apple_od_trained_inference_v1", checkpoint_path=str(path), checkpoint_sha256=digest,
                      checkpoint_sidecar=str(sidecar), updates=updates, stage=payload["stage"],
                      code_version=deepcopy(version), constructor_config=configuration,
                      validation_short_side=recipe["validation_short_side"], validation_max_size=recipe["validation_max_size"],
                      calibration=calibration, validation_source=validation_source,
                      validation_verified=calibration is not None, strict_model_load=strict,
                      training_source_sha256=deepcopy(identity["source_sha256"]),
                      parent_provenance=deepcopy(payload.get("parent_provenance")), state_scope="model_only",
                      optimizer_restored=False, checkpoint_RNG_restored=False, training_resume=False,
                      inference_precision="float32; same as complete full OD VAL")
    del payload
    _require(_sha(path) == digest and sidecar.read_text(encoding="ascii").strip() == digest,
             "OD checkpoint or sidecar changed while loading")
    model.checkpoint_provenance = deepcopy(provenance)
    return model, provenance


def _image_pixels(image, provenance):
    source_file = None
    if isinstance(image, (str, Path)):
        source_file = Path(image).expanduser().absolute()
        file_hash = _sha(source_file)
        with Image.open(source_file) as opened:
            rgb = opened.convert("RGB")
    else:
        _require(isinstance(image, Image.Image), "image must be a local path or PIL image")
        rgb, file_hash = image.convert("RGB"), None
    width, height = rgb.size
    pixels = legacy_rgb_tensor(rgb, provenance["validation_short_side"], provenance["validation_max_size"])
    resized = tuple(pixels.shape[-2:])
    rgb_hash = hashlib.sha256(memoryview(np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)))).hexdigest()
    normalized_hash = hashlib.sha256(memoryview(pixels.numpy())).hexdigest()
    return rgb, pixels, dict(image=str(source_file) if source_file else None, source_file_sha256=file_hash,
                            RGB_sha256=rgb_hash, RGB_hash_encoding="contiguous uint8 HWC canonical RGB",
                            normalized_input_sha256=normalized_hash, original_hw=[height, width], input_hw=list(resized),
                            original_to_input_scale_xy=[resized[1] / width, resized[0] / height])


def _verify_prediction_binding(model, provenance):
    _require(isinstance(provenance, dict) and provenance.get("schema") == "apple_od_trained_inference_v1"
             and provenance.get("validation_verified") is True and isinstance(provenance.get("calibration"), dict),
             "Prediction requires matching completed eligible full validation")
    attached = getattr(model, "checkpoint_provenance", None)
    _require(isinstance(attached, dict) and attached.get("checkpoint_sha256") == provenance.get("checkpoint_sha256")
             and attached.get("updates") == provenance.get("updates")
             and attached.get("calibration") == provenance.get("calibration"), "Model and inference calibration differ")
    _require(provenance["calibration"].get("update") == provenance["updates"], "A different update's confidence is forbidden")
    _verify_sources(provenance.get("training_source_sha256"))
    for source in provenance["validation_source"].values():
        _require(_sha(source["path"]) == source["sha256"], "Bound validation artifact changed after loading")


def predict_od_image(model, image, provenance, *, device="cpu", mask_directory=None, visualization_path=None):
    """Return native A boxes/scores plus compact CPU V/A probability tensors.

    The returned ``metadata`` and ``detections`` are JSON-compatible. The other
    tensors are compact ROI results, saved separately by ``save_od_prediction``.
    V/A exports are independent binary crops with explicit native XYXY windows.
    """
    _verify_prediction_binding(model, provenance)
    _require(not model.training, "OD prediction model must be in eval mode")
    dev = torch.device(device)
    _require(dev.type in ("cpu", "cuda"), "Only CPU and CUDA OD inference are supported")
    if dev.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA inference is unavailable")
    mask_directory = Path(mask_directory).expanduser().absolute() if mask_directory is not None else None
    visualization_path = Path(visualization_path).expanduser().absolute() if visualization_path is not None else None
    if mask_directory is not None:
        _require(not mask_directory.exists() or not any(mask_directory.iterdir()), "Use a fresh mask export directory")
        mask_directory.mkdir(parents=True, exist_ok=True)
    if visualization_path is not None:
        _require(not visualization_path.exists(), "Visualization output already exists")
        visualization_path.parent.mkdir(parents=True, exist_ok=True)
    rgb, pixels, image_metadata = _image_pixels(image, provenance)
    inference_source_before = _sha(__file__)
    height, width = image_metadata["original_hw"]
    model.to(dev)
    with torch.inference_mode(), torch.autocast(device_type=dev.type, enabled=False):
        inference_pixels = pixels[None].to(dev)
        outputs = model(inference_pixels, pixel_mask=torch.ones((1, *pixels.shape[-2:]), dtype=torch.bool, device=dev))
        prediction = _validate_prediction(outputs, tuple(pixels.shape[-2:]), (height, width), mask_chunk_size=8)
        actual_input = inference_pixels[0].detach().to(device="cpu").contiguous().numpy()
        _require(hashlib.sha256(memoryview(actual_input)).hexdigest() == image_metadata["normalized_input_sha256"],
                 "Model mutated normalized RGB during inference")
    confidence = provenance["calibration"]["confidence"]
    selected = [index for index, score in enumerate(prediction["scores"]) if score >= confidence]
    boxes = prediction["boxes"][selected]
    scores = torch.tensor([prediction["scores"][index] for index in selected], dtype=torch.float32)
    masks = {branch: prediction[branch][selected, None].contiguous() for branch in ("visible", "amodal")}
    detections = [dict(prediction_index=index, label="apple", score=prediction["scores"][index],
                       amodal_box_xyxy=prediction["boxes"][index].tolist()) for index in selected]
    canvas = np.asarray(rgb).copy() if visualization_path is not None else None
    if mask_directory is not None or canvas is not None:
        for index, detection in enumerate(detections):
            for branch, color, alpha in (("amodal", (40, 130, 255), .22), ("visible", (50, 230, 90), .42)):
                crop, window = paste_roi_mask(masks[branch][index, 0], boxes[index], (height, width))
                left, top, right, bottom = window
                if mask_directory is not None:
                    path = mask_directory / f"instance_{index:04d}_{branch}.png"
                    if crop.numel():
                        with path.open("xb") as stream:
                            Image.fromarray(crop.numpy().astype(np.uint8) * 255).save(stream, format="PNG")
                    detection[branch + "_mask"] = dict(path=str(path) if crop.numel() else None,
                                                        window_xyxy=list(window), foreground_pixels=int(crop.sum()))
                if canvas is not None and crop.numel():
                    region, active = canvas[top:bottom, left:right], crop.numpy()
                    region[active] = (region[active] * (1 - alpha) + np.asarray(color) * alpha).astype(np.uint8)
    if canvas is not None:
        rendered = Image.fromarray(canvas)
        drawing = ImageDraw.Draw(rendered)
        for detection in detections:
            x1, y1, x2, y2 = detection["amodal_box_xyxy"]
            drawing.rectangle((x1, y1, x2, y2), outline=(255, 190, 30), width=2)
            drawing.text((max(0., x1), max(0., y1)), f"apple {detection['score']:.3f}", fill=(255, 255, 255))
        with visualization_path.open("xb") as stream:
            rendered.save(stream, format="PNG")
    if image_metadata["image"] is not None:
        _require(_sha(image_metadata["image"]) == image_metadata["source_file_sha256"], "Input RGB file changed during inference")
    _verify_prediction_binding(model, provenance)
    _require(_sha(__file__) == inference_source_before, "OD inference source changed during prediction")
    metadata = dict(schema="apple_od_single_image_prediction_v1", **image_metadata,
                    checkpoint_path=provenance["checkpoint_path"], checkpoint_sha256=provenance["checkpoint_sha256"],
                    updates=provenance["updates"], code_version=deepcopy(provenance["code_version"]),
                    validation_source=deepcopy(provenance["validation_source"]), confidence=confidence,
                    confidence_operator=">=", count=len(selected), inference_precision="float32",
                    inference_source_sha256=inference_source_before, preprocessing=dict(
                        short_side=provenance["validation_short_side"], max_size=provenance["validation_max_size"],
                        normalization_mean=list(NORMALIZATION_MEAN), normalization_std=list(NORMALIZATION_STD),
                        resize="Legacy torchvision float32 bilinear aspect resize; same as full VAL", rgb_scale="uint8/255",
                        **LEGACY_RGB_PREPROCESS),
                    postprocess={**POSTPROCESS_POLICY, "constructor_config": deepcopy(provenance["constructor_config"])},
                    visualization=str(visualization_path) if visualization_path else None, accuracy_measured=False)
    return dict(boxes=boxes, scores=scores, roi_visible_masks=masks["visible"], roi_amodal_masks=masks["amodal"],
                prediction_indices=torch.tensor(selected, dtype=torch.int64), metadata=metadata, detections=detections)


def save_od_prediction(prediction, output_directory):
    """Save compact tensors with their own hash and a JSON prediction record."""
    output = Path(output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    compact_path, json_path = output / "compact_roi_masks.pt", output / "prediction.json"
    _require(not compact_path.exists() and not compact_path.with_suffix(".pt.sha256").exists() and not json_path.exists(),
             "Prediction output files already exist")
    compact = {key: prediction[key] for key in ("boxes", "scores", "roi_visible_masks", "roi_amodal_masks", "prediction_indices")}
    compact.update(schema="apple_od_compact_prediction_v1", metadata=prediction["metadata"])
    digest = od_training.save_checkpoint(compact_path, compact)
    record = {**prediction["metadata"], "detections": prediction["detections"],
              "compact_masks": {"path": str(compact_path), "sha256": digest,
                                "box_coordinates": "native original image XYXY",
                                "mask_coordinates": "compact ROI probabilities; paste with the native A box"}}
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(_jsonable(record), stream, ensure_ascii=False, indent=2, allow_nan=False)
    return record
