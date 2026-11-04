"""Reproducible mixed-source training of the high-resolution OD detector."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

import numpy as np
import psutil
import torch
import torchvision

from .data import PAppleDataset, MinneAppleDataset
from .evaluate import REFERENCE_PINS
from .io_compat import install_windows_replace_retry
from .od_data import ODInputDataset
from .od_model import build_od_amodal_detector
from .training import (CyclingOrder, atomic_json, sha, target_fingerprint,
                       _rng_state, _restore_rng)

SCHEMA = "apple_od_amodal_training_v1"
SOURCE_ORDER = ("PApple", "MinneApple", "PApple", "MinneApple")
SOURCE_FILES = ("od_training.py", "od_data.py", "od_model.py", "od_roi.py",
                "od_evaluate.py", "data.py", "evaluate.py", "training.py", "io_compat.py", "od_inference.py",
                "od_context.py", "od_foreground.py", "od_context_model.py", "od_proposals.py", "od_detail.py",
                "od_mask_geometry.py", "od_mask_geometry_model.py", "od_verifier.py")
_INITIALIZER_BASE_FILES = frozenset(("od_training.py", "od_data.py", "od_model.py", "od_roi.py",
                                    "od_evaluate.py", "data.py", "evaluate.py", "training.py", "io_compat.py"))
CLASSIFIER_POLICY_FIELD = "unknown_amodal_classifier_policy"
JOINT_POLICY_FIELD = "unknown_amodal_joint_policy"
JOINT_POLICIES = ("legacy_detection", "visible_mask_only")
CLASSIFIER_SAMPLING_FIELD = "classifier_negative_sampling"
CLASSIFIER_SAMPLING_POLICIES = ("random", "gate_ohem_half", "gate_ohem_half_heavy")
CLASSIFICATION_LOSS_FIELD = "classification_loss"
CLASSIFICATION_LOSS_TYPES = ("cross_entropy", "varifocal")
OCCLUSION_WEIGHT_OVERRIDE_FIELD = "occlusion_weight_override"
BOX_GIOU_FIELD = "box_giou_weight"
VFL_MARGIN_FIELD = "vfl_iou_margin"
VFL_QUALITY_GAMMA_FIELD = "vfl_iou_gamma"
VFL_NEGATIVE_FIELD = "vfl_negative_weight"
FPN_LR_FIELD = "fpn_lr"
EMA_DECAY_FIELD = "ema_decay"
DATA_SOURCE_POLICY_FIELD = "data_source_policy"
DATA_SOURCE_POLICIES = ("balanced", "papple_only")


def _training_source_order(recipe):
    if not isinstance(recipe, dict):
        raise ValueError("data_source_policy requires an explicit recipe mapping")
    policy = recipe.get(DATA_SOURCE_POLICY_FIELD, "balanced")
    if type(policy) is not str or policy not in DATA_SOURCE_POLICIES:
        raise ValueError("data_source_policy must be balanced or papple_only")
    return SOURCE_ORDER if policy == "balanced" else ("PApple", "PApple")


def _classifier_sampling_policy(recipe):
    policy = recipe.get(CLASSIFIER_SAMPLING_FIELD, "random")
    if type(policy) is not str or policy not in CLASSIFIER_SAMPLING_POLICIES:
        raise ValueError("classifier_negative_sampling must be random or gate_ohem_half")
    return policy


def _classification_loss_type(recipe):
    loss_type = recipe.get(CLASSIFICATION_LOSS_FIELD, "cross_entropy")
    if type(loss_type) is not str or loss_type not in CLASSIFICATION_LOSS_TYPES:
        raise ValueError("classification_loss must be cross_entropy or varifocal")
    for key, default in (("vfl_alpha", .75), ("vfl_gamma", 2.0)):
        value = recipe.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        recipe[key] = float(value)
    if recipe["vfl_alpha"] > 1:
        raise ValueError("vfl_alpha must lie in [0,1]")
    if OCCLUSION_WEIGHT_OVERRIDE_FIELD in recipe:
        weight = recipe[OCCLUSION_WEIGHT_OVERRIDE_FIELD]
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight < 0):
            raise ValueError("occlusion_weight_override must be finite and nonnegative")
        recipe[OCCLUSION_WEIGHT_OVERRIDE_FIELD] = float(weight)
    recipe[CLASSIFICATION_LOSS_FIELD] = loss_type
    return loss_type


def _box_giou_weight(recipe):
    """Normalize the optional overlap-aware localization weight.

    Zero is the historical Smooth L1 objective and is normalized into every
    recipe/identity so a resumed run can never silently change it.
    """
    value = recipe.get(BOX_GIOU_FIELD, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("box_giou_weight must be finite and nonnegative")
    recipe[BOX_GIOU_FIELD] = float(value)
    return recipe[BOX_GIOU_FIELD]


def _vfl_sharpening(recipe):
    """Optional precision-oriented VFL target; absent means the legacy objective."""
    margin = recipe.get(VFL_MARGIN_FIELD, 0.0)
    gamma = recipe.get(VFL_QUALITY_GAMMA_FIELD, 1.0)
    negative = recipe.get(VFL_NEGATIVE_FIELD, 1.0)
    for name, value in ((VFL_MARGIN_FIELD, margin), (VFL_QUALITY_GAMMA_FIELD, gamma),
                        (VFL_NEGATIVE_FIELD, negative)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if not 0 <= margin < 1:
        raise ValueError("vfl_iou_margin must lie in [0,1)")
    if gamma <= 0:
        raise ValueError("vfl_iou_gamma must be positive")
    if negative <= 0:
        raise ValueError("vfl_negative_weight must be positive")
    recipe[VFL_MARGIN_FIELD] = float(margin)
    recipe[VFL_QUALITY_GAMMA_FIELD] = float(gamma)
    recipe[VFL_NEGATIVE_FIELD] = float(negative)
    return recipe[VFL_MARGIN_FIELD], recipe[VFL_QUALITY_GAMMA_FIELD], recipe[VFL_NEGATIVE_FIELD]


def _fpn_lr(recipe):
    """Optional separate learning rate for a freshly initialized FPN.

    A replaced backbone leaves the FPN randomly initialized while the body is
    ImageNet-pretrained; giving both the same rate under-trains the FPN and
    over-trains the body. Absent means the historical single backbone group.
    """
    if FPN_LR_FIELD not in recipe:
        return None
    value = recipe[FPN_LR_FIELD]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("fpn_lr must be finite and positive")
    recipe[FPN_LR_FIELD] = float(value)
    return recipe[FPN_LR_FIELD]


def _ema_decay(recipe):
    """Exponential moving average decay; zero disables the shadow copy."""
    value = recipe.get(EMA_DECAY_FIELD, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("ema_decay must lie in [0,1)")
    recipe[EMA_DECAY_FIELD] = float(value)
    return recipe[EMA_DECAY_FIELD]


def _initialization_spec(raw, paths):
    specification = raw.get("initialization", {"kind": "r50"})
    if (not isinstance(specification, dict)
            or specification.get("kind") not in ("r50", "validated_od", "backbone_swap")):
        raise ValueError("initialization.kind must be r50, validated_od or backbone_swap")
    if specification["kind"] == "r50":
        if set(specification) != {"kind"} or "parent_validation_root" in paths:
            raise ValueError("R50 initialization cannot specify OD lineage or a parent validation root")
    elif specification["kind"] == "backbone_swap":
        # A backbone swap keeps every ROI tensor but replaces the feature
        # extractor, so it can only warm-start from one named parent and can
        # never inherit that parent's optimizer, samplers or update count.
        if set(specification) != {"kind"} or "parent_validation_root" in paths:
            raise ValueError("backbone_swap cannot claim a parent validation root")
    else:
        if (set(specification) != {"kind", "source_commit", "required_stage"}
                or not isinstance(specification["source_commit"], str)
                or re.fullmatch(r"[0-9a-f]{40}", specification["source_commit"]) is None
                or specification["required_stage"] != "joint"):
            raise ValueError("validated_od requires an exact source_commit and required_stage=joint")
    return deepcopy(specification)


def load_od_config(path, *, output=None, smoke=False):
    path = Path(path).absolute()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if raw.get("schema") != SCHEMA:
        raise ValueError("Expected dedicated OD training schema")
    root = (path.parent / raw["path_base"]).resolve()
    paths = {}
    for name, value in raw["paths"].items():
        destination = (root / value).resolve()
        if Path(value).is_absolute() or not destination.is_relative_to(root):
            raise ValueError("Training path escapes project: " + name)
        paths[name] = destination
    required = {"papple_root", "minne_root", "minne_manifest", "papple_provenance", "parent_checkpoint", "output"}
    if not required.issubset(paths):
        raise ValueError("Missing training paths")
    initialization = _initialization_spec(raw, paths)
    recipe = raw["training"]
    _training_source_order(recipe)
    recipe[DATA_SOURCE_POLICY_FIELD] = recipe.get(DATA_SOURCE_POLICY_FIELD, "balanced")
    visible_proposals = raw["model"].get("visible_proposals", False)
    if type(visible_proposals) is not bool:
        raise ValueError("visible_proposals must be an explicit bool")
    shallow_detail = raw["model"].get("shallow_detail", False)
    if type(shallow_detail) is not bool:
        raise ValueError("shallow_detail must be an explicit bool")
    if shallow_detail and not visible_proposals:
        raise ValueError("shallow_detail training requires visible_proposals=True")
    mask_geometry = raw['model'].get('mask_box_refinement', False)
    if type(mask_geometry) is not bool:
        raise ValueError('mask_box_refinement must be an explicit bool')
    if mask_geometry and (not shallow_detail or raw['model'].get('amodal_base') is not True):
        raise ValueError('mask_box_refinement requires shallow_detail=True and amodal_base=True')
    verifier = raw['model'].get('occlusion_verifier', False)
    if type(verifier) is not bool:
        raise ValueError('occlusion_verifier must be an explicit bool')
    highres = raw['model'].get('classification_highres', False)
    if type(highres) is not bool:
        raise ValueError('classification_highres must be an explicit bool')
    # The verifier works on the plain head too; only the A-geometry arm needs
    # mask_box_refinement.
    ranking = raw["training"].get("verifier_ranking", True)
    if type(ranking) is not bool:
        raise ValueError('verifier_ranking must be an explicit bool')
    if verifier and ranking and raw["loss_weights"]["PApple"].get("loss_verifier_rank", 0) <= 0:
        raise ValueError('verifier_ranking requires a positive PApple loss_verifier_rank weight')
    if not verifier and ranking is not True:
        raise ValueError('verifier_ranking is only meaningful with occlusion_verifier=True')
    recipe.setdefault("verifier_ranking", ranking)
    if visible_proposals:
        if (raw["model"].get("context_cascade") is not True
                or recipe[DATA_SOURCE_POLICY_FIELD] != "papple_only"):
            raise ValueError("visible_proposals training requires context_cascade=True and papple_only")
    # Include the algorithm in checkpoint/resume identity; inference is unchanged.
    recipe[CLASSIFIER_SAMPLING_FIELD] = _classifier_sampling_policy(recipe)
    _classification_loss_type(recipe)
    _box_giou_weight(recipe)
    _fpn_lr(recipe)
    _ema_decay(recipe)
    _vfl_sharpening(recipe)
    if raw["model"].get("occlusion_verifier", False) and recipe[CLASSIFIER_SAMPLING_FIELD] != "gate_ohem_half_heavy":
        raise ValueError('occlusion_verifier training requires classifier_negative_sampling=gate_ohem_half_heavy')
    joint_policy = recipe.get(JOINT_POLICY_FIELD, "legacy_detection")
    if type(joint_policy) is not str or joint_policy not in JOINT_POLICIES:
        raise ValueError("unknown_amodal_joint_policy must be legacy_detection or visible_mask_only")
    # Normalize the default into the saved recipe/identity, without changing
    # the configuration bytes or any historical initializer loss weights.
    recipe[JOINT_POLICY_FIELD] = joint_policy
    if type(recipe.get("defer_mask_metrics", False)) is not bool:
        raise ValueError("defer_mask_metrics must be an explicit boolean")
    if type(recipe.get("validate_initial", True)) is not bool:
        raise ValueError("validate_initial must be an explicit boolean")
    if not recipe.get("validate_initial", True) and initialization["kind"] not in ("validated_od", "backbone_swap"):
        raise ValueError("Skipping initial validation requires a fully audited validated_od or backbone_swap initializer")
    positive_ints = ("max_size", "validation_short_side", "validation_max_size",
                    "validate_every", "checkpoint_every", "warmup_updates",
                    "seed", "cpu_threads")
    for key in positive_ints:
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ValueError("Positive integer required: " + key)
    classifier_epochs = recipe["classifier_papple_epochs"]
    if type(classifier_epochs) is not int or classifier_epochs < 0:
        raise ValueError("Classifier epochs must be a nonnegative integer")
    joint_epochs = recipe["joint_papple_epochs"]
    if type(joint_epochs) is not int or joint_epochs < 0:
        raise ValueError("Joint epochs must be a nonnegative integer")
    if initialization["kind"] == "validated_od":
        if joint_epochs == 0 and classifier_epochs <= 0:
            raise ValueError("validated_od joint epochs0 requires classifier epochs>0")
        if raw["model"].get(CLASSIFIER_POLICY_FIELD) not in ("legacy_box_iou", "ignore_overlapping_visible"):
            raise ValueError("validated_od target must explicitly use a known classifier policy")
    elif joint_epochs == 0:
        raise ValueError("Joint epochs0 requires a validated_od initializer and positive classifier epochs")
    if not raw["model"].get("use_visible_gate", True) and classifier_epochs:
        raise ValueError("Gate-off ablation must skip the frozen classifier stage")
    floor = raw.get("parent_selection_floor")
    if isinstance(floor, bool) or not isinstance(floor, (float, int)) or not math.isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError("Explicit parent selection floor in [0,1] required")
    if not recipe["short_sides"] or any(type(s) is not int or s < 1 for s in recipe["short_sides"]):
        raise ValueError("Positive training short sides required")
    if any(type(u) is not int or u < 1 for u in recipe["pilot_validation_updates"]):
        raise ValueError("Positive pilot validation updates required")
    if type(recipe["augment"]) is not bool or recipe["amp_dtype"] not in ("bfloat16", "float32"):
        raise ValueError("Explicit augment and supported AMP dtype required")
    for key in ("backbone_lr", "pretrained_head_lr", "new_head_lr", "classifier_lr", "weight_decay", "gradient_clip_norm"):
        if isinstance(recipe[key], bool) or not math.isfinite(recipe[key]) or recipe[key] <= 0:
            raise ValueError("Positive finite training value required: " + key)
    for source, weights in raw["loss_weights"].items():
        if source not in SOURCE_ORDER or any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in weights.values()):
            raise ValueError("Invalid per-source loss weights")
    if set(raw["loss_weights"]) != {"PApple", "MinneApple"}:
        raise ValueError("Both source loss policies must be explicit")
    if any("amodal" in key or "hidden" in key or key == "loss_box_reg"
           for key in raw["loss_weights"]["MinneApple"]):
        raise ValueError("Visible-only source cannot supervise final A boxes or A/H masks")
    digest = raw.get("parent_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Pin exact parent checkpoint SHA256")
    return dict(path=path, root=root, raw=raw, paths=paths, recipe=recipe, initialization=initialization,
                output=Path(output).absolute() if output else paths["output"], smoke=bool(smoke))


def source_snapshot():
    return {name: sha(Path(__file__).with_name(name)) for name in SOURCE_FILES}


def attach_ignore_boxes(target, boxes, input_hw):
    """Attach TRAIN-only audit boxes in detector input coordinates.

    The source JSON is in original image pixels. Horizontal augmentation and
    resize are applied exactly as they are to pixels; annotation masks and
    source files are not changed. These boxes never become positives.
    """
    if boxes is None:
        return target
    box = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if not bool(torch.isfinite(box).all()) or bool((box[:, 2:] <= box[:, :2]).any()):
        raise ValueError("Ignore boxes must be finite positive XYXY")
    original_h, original_w = (int(value) for value in target["original_hw"])
    input_h, input_w = (int(value) for value in input_hw)
    box = box * torch.tensor([input_w / original_w, input_h / original_h] * 2)
    if target["source"].get("horizontal_flip", False):
        box[:, [0, 2]] = input_w - box[:, [2, 0]]
    box[:, [0, 2]] = box[:, [0, 2]].clamp(0, input_w)
    box[:, [1, 3]] = box[:, [1, 3]].clamp(0, input_h)
    valid = (box[:, 2:] > box[:, :2]).all(dim=1)
    target["ignore_boxes"] = box[valid].contiguous()
    return target


def code_version(root, config_path, *, smoke=False):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    consumed = ["src/apple_seg/" + name for name in SOURCE_FILES]
    consumed += ["scripts/train_od.py", config_path.relative_to(root).as_posix()]
    mismatches = []
    for name in consumed:
        blob = subprocess.run(["git", "show", commit + ":" + name], cwd=root, capture_output=True)
        if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != sha(root / name):
            mismatches.append(name)
    if mismatches and not smoke:
        raise ValueError("Commit consumed OD code/config before formal training: " + ", ".join(mismatches))
    return dict(training_commit=commit, consumed_files_match_commit=not mismatches,
                uncommitted_consumed_files=mismatches)


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    digest = sha(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    temporary.write_text(digest, encoding="ascii")
    os.replace(temporary, sidecar)
    return digest


def load_checkpoint(path):
    path = Path(path)
    if sha(path) != path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip():
        raise ValueError("OD checkpoint hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("Unexpected checkpoint schema")
    return payload


def _historical_source_inventory(commit, root):
    """Read only a literal source inventory from the committed trainer AST."""
    blob = subprocess.run(["git", "show", commit + ":src/apple_seg/od_training.py"],
                          cwd=root, capture_output=True)
    if blob.returncode:
        raise ValueError("OD initializer historical trainer source is unavailable")
    try:
        syntax = ast.parse(blob.stdout.decode("utf-8-sig"))
        declarations = []
        for statement in syntax.body:
            if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "SOURCE_FILES" for target in statement.targets):
                declarations.append(statement.value)
            elif (isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
                  and statement.target.id == "SOURCE_FILES"):
                declarations.append(statement.value)
        stores = [node for node in ast.walk(syntax) if isinstance(node, ast.Name)
                  and node.id == "SOURCE_FILES" and isinstance(node.ctx, ast.Store)]
        if len(declarations) != 1 or len(stores) != 1:
            raise ValueError("Expected one literal SOURCE_FILES declaration")
        inventory = ast.literal_eval(declarations[0])
        if (not isinstance(inventory, (tuple, list)) or not all(type(name) is str for name in inventory)
                or len(set(inventory)) != len(inventory)
                or set(inventory) not in (_INITIALIZER_BASE_FILES,
                                          _INITIALIZER_BASE_FILES | {"od_inference.py"},
                                          _INITIALIZER_BASE_FILES | {"od_inference.py", "od_context.py", "od_foreground.py", "od_context_model.py"},
                                          _INITIALIZER_BASE_FILES | {"od_inference.py", "od_context.py", "od_foreground.py", "od_context_model.py", "od_proposals.py"},
                                          _INITIALIZER_BASE_FILES | {"od_inference.py", "od_context.py", "od_foreground.py", "od_context_model.py", "od_proposals.py", "od_detail.py"},
                                          _INITIALIZER_BASE_FILES | {"od_inference.py", "od_context.py", "od_foreground.py", "od_context_model.py", "od_proposals.py", "od_detail.py", "od_mask_geometry.py", "od_mask_geometry_model.py"},
                                          frozenset(SOURCE_FILES))):
            raise ValueError("Expected a complete historical nine/ten/thirteen/fourteen/fifteen/seventeen/eighteen source inventory")
    except (SyntaxError, UnicodeError, TypeError, ValueError) as error:
        raise ValueError("Invalid historical OD SOURCE_FILES literal inventory") from error
    return tuple(inventory)


def _verify_initializer_history(payload, specification, root):
    """Verify source files against their own Git commit, never today's files."""
    commit = specification["source_commit"]
    version = payload.get("code_version", {})
    if (not isinstance(version, dict) or version.get("training_commit") != commit or version.get("consumed_files_match_commit") is not True
            or version.get("uncommitted_consumed_files") != []):
        raise ValueError("OD initializer source_commit/code_version differs or was not frozen")
    resolved = subprocess.run(["git", "rev-parse", "--verify", commit + "^{commit}"], cwd=root,
                              capture_output=True, text=True)
    if resolved.returncode or resolved.stdout.strip() != commit:
        raise ValueError("OD initializer source_commit is absent from local Git history")
    inventory = _historical_source_inventory(commit, root)
    sources = payload.get("identity", {}).get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != set(inventory):
        raise ValueError("OD initializer pins differ from its historical source inventory")
    for name, digest in sources.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Invalid OD initializer source SHA256: " + name)
        blob = subprocess.run(["git", "show", commit + ":src/apple_seg/" + name], cwd=root, capture_output=True)
        if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != digest:
            raise ValueError("OD initializer source does not match its Git history blob: " + name)
    return deepcopy(sources)


def _load_backbone_swap_initializer(cfg):
    """Warm-start a replaced backbone experiment from one R50 parent.

    Every non-backbone tensor must transfer by name, shape and dtype: that is
    the whole point of the arm, and a silent miss would make the run
    unattributable. The feature extractor and its FPN are the only new tensors,
    so they are counted and reported instead of being inherited.
    """
    path, expected = cfg["paths"]["parent_checkpoint"], cfg["raw"]["parent_sha256"]
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != expected:
        raise ValueError("OD initializer explicit/sidecar SHA256 differs")
    if sha(path) != expected:
        raise ValueError("OD initializer checkpoint SHA256 differs from the pinned value")
    payload = load_checkpoint(path)
    if payload.get("smoke") is not False or payload.get("stage") != "joint":
        raise ValueError("OD backbone_swap initializer must be a formal joint checkpoint")
    if type(payload.get("updates")) is not int or payload["updates"] < 0:
        raise ValueError("OD backbone_swap initializer update count is invalid")
    target = dict(cfg["raw"]["model"])
    # The amodal base tower is a topology fact of the parent, not a recipe
    # choice: inherit it or the copied tensor set would silently lose the A base.
    parent_constructor = payload.get("constructor_config")
    if not isinstance(parent_constructor, dict) or parent_constructor.get("amodal_base") is not True:
        raise ValueError("OD backbone_swap initializer must carry a genuine amodal base")
    target["amodal_base"] = True
    if target.get("backbone_name") not in ("convnext_small", "convnext_base",
                                            "convnext_large", "swin_s", "swin_b"):
        raise ValueError("backbone_swap requires an explicit supported replacement backbone_name")
    model = build_od_amodal_detector(weights=None, **target)
    old, new = payload["model"], model.state_dict()
    if not isinstance(old, dict) or not old:
        raise ValueError("OD backbone_swap initializer tensor state is absent")
    shared = {name for name in old
              if name in new and old[name].shape == new[name].shape and old[name].dtype == new[name].dtype}
    outside = {name for name in old if not name.startswith("backbone.")}
    missing = sorted(outside - shared)
    if missing:
        raise ValueError("backbone_swap failed to inherit every non-backbone tensor: " + ", ".join(missing[:5]))
    replaced = sorted(set(new) - shared)
    merged = dict(new)
    for name in shared:
        merged[name] = old[name]
    model.load_state_dict(merged, strict=True)
    lineage = dict(schema="apple_od_backbone_swap_v1", kind="backbone_swap",
                   checkpoint_path=str(path), checkpoint_sha256=expected,
                   source_updates=payload["updates"], source_stage=payload["stage"],
                   source_code_version=deepcopy(payload.get("code_version", {})),
                   replaced_backbone=target["backbone_name"],
                   inherited_tensor_count=len(shared),
                   inherited_non_backbone=len(shared & outside),
                   parent_non_backbone=len(outside),
                   new_tensor_count=len(replaced),
                   state_scope="model_only", optimizer_restored=False, sampler_restored=False,
                   checkpoint_RNG_restored=False, source_updates_history_inherited=False,
                   new_run_initial_update=0)
    model.checkpoint_provenance = deepcopy(lineage)
    return model, lineage


def _load_validated_od_initializer(cfg):
    """Copy a validated joint model into a new independent training experiment.

    This deliberately does not invoke the public inference loader's current
    source check: a new trainer has different source bytes. Instead, its literal
    historical source inventory is verified against that exact source commit,
    the public strict constructor/state/complete-validation audits are reused.
    No source optimizer, RNG, samplers, updates or calibration become run state.
    """
    from . import od_inference
    path, expected = cfg["paths"]["parent_checkpoint"], cfg["raw"]["parent_sha256"]
    specification = cfg["initialization"]
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != expected:
        raise ValueError("OD initializer explicit/sidecar SHA256 differs")
    payload = load_checkpoint(path)
    identity = payload.get("identity", {})
    if (not isinstance(identity, dict) or payload.get("smoke") is not False or identity.get("smoke") is not False
            or payload.get("phase") not in ("training", "complete", "interrupted")
            or payload.get("stage") != specification["required_stage"]
            or type(payload.get("updates")) is not int or payload["updates"] < 0):
        raise ValueError("OD initializer must be a formal validated joint checkpoint")
    source_hashes = _verify_initializer_history(payload, specification, cfg["root"])
    if (identity.get("papple_provenance_sha256") != sha(cfg["paths"]["papple_provenance"])
            or identity.get("minne_manifest_sha256") != sha(cfg["paths"]["minne_manifest"])):
        raise ValueError("OD initializer original VAL provenance or Minne split differs")
    historical_order = _training_source_order(identity.get("recipe"))
    if identity.get("source_order") != list(historical_order) or identity.get("loss_weights") != cfg["raw"]["loss_weights"]:
        raise ValueError("OD initializer source order must match its historical recipe and explicit loss weights")
    source_parent_sha = identity.get("parent_sha256")
    if (not isinstance(source_parent_sha, str) or re.fullmatch(r"[0-9a-f]{64}", source_parent_sha) is None
            or not isinstance(payload.get("parent_provenance"), dict) or not payload["parent_provenance"]):
        raise ValueError("OD initializer original parent SHA256/provenance is missing")
    original_constructor = deepcopy(payload.get("constructor_config"))
    if not isinstance(original_constructor, dict):
        raise ValueError("OD initializer constructor is missing")
    normalized_constructor = deepcopy(original_constructor)
    normalized_constructor.setdefault(CLASSIFIER_POLICY_FIELD, "legacy_box_iou")
    normalized_constructor = od_inference._audit_constructor(normalized_constructor)
    def same_constructor_value(name, value):
        if name == "amodal_base" and value is None:
            return True  # The builder's documented automatic base selection.
        expected_value = normalized_constructor[name]
        return (value == expected_value and
                (type(value) is bool) == (type(expected_value) is bool))

    experiment_model = identity.get("model")
    if (not isinstance(experiment_model, dict) or not set(experiment_model).issubset(normalized_constructor)
            or not od_inference.HIDDEN_TOWER_IDENTITY_FIELDS.issubset(experiment_model)
            or (CLASSIFIER_POLICY_FIELD in original_constructor and CLASSIFIER_POLICY_FIELD not in experiment_model)
            or not all(same_constructor_value(name, value) for name, value in experiment_model.items())):
        raise ValueError("OD initializer experiment identity and constructor differ")
    target_constructor = deepcopy(normalized_constructor)
    target_options = cfg["raw"]["model"]
    if target_options.get(CLASSIFIER_POLICY_FIELD) not in ("legacy_box_iou", "ignore_overlapping_visible"):
        raise ValueError("OD initializer target must explicitly use a known classifier policy")
    for name, value in target_options.items():
        if name == CLASSIFIER_POLICY_FIELD:
            target_constructor[name] = value
        elif name not in normalized_constructor or not same_constructor_value(name, value):
            raise ValueError("OD initializer forbids another constructor change: " + name)
    target_constructor = od_inference._audit_constructor(target_constructor)
    point, validation_source = od_inference._validation_for_weights(
        payload, path, cfg["paths"].get("parent_validation_root"), required=True)
    model = build_od_amodal_detector(weights=None, **target_constructor)
    if model.constructor_config != target_constructor:
        raise ValueError("OD initializer reconstructed target constructor differs")
    strict = od_inference._strict_restore(model, payload.get("model"))
    lineage = dict(schema="apple_od_validated_joint_initializer_v1", kind="validated_od",
                   checkpoint_path=str(path), checkpoint_sha256=expected, source_commit=specification["source_commit"],
                   source_updates=payload["updates"], source_stage=payload["stage"],
                   source_code_version=deepcopy(payload["code_version"]), source_training_sha256=source_hashes,
                   source_parent_sha256=source_parent_sha, source_parent_provenance=deepcopy(payload["parent_provenance"]),
                   source_identity=deepcopy(identity), original_source_constructor=original_constructor,
                   normalized_source_constructor=normalized_constructor, target_constructor=target_constructor,
                   constructor_change=({CLASSIFIER_POLICY_FIELD: dict(source=normalized_constructor[CLASSIFIER_POLICY_FIELD],
                                                                      target=target_constructor[CLASSIFIER_POLICY_FIELD])}
                                       if normalized_constructor[CLASSIFIER_POLICY_FIELD] != target_constructor[CLASSIFIER_POLICY_FIELD] else {}),
                   source_validation=point, source_validation_artifacts=validation_source,
                   strict_model_load=strict, state_scope="model_only", training_resume=False,
                   optimizer_restored=False, sampler_restored=False, checkpoint_RNG_restored=False,
                   source_updates_history_or_threshold_inherited=False, new_run_initial_update=0,
                   new_run_initial_stage="joint" if cfg["raw"]["training"]["joint_papple_epochs"] else "classifier",
                   initial_validation_skipped_on_verified_parent=not cfg["raw"]["training"].get("validate_initial", True))
    model.checkpoint_provenance = deepcopy(lineage)
    if sha(path) != expected or sidecar.read_text(encoding="ascii").strip() != expected:
        raise ValueError("OD initializer checkpoint or sidecar changed while loading")
    return model, lineage


@contextmanager
def validation_rng_isolation(generator, *, cuda=False):
    """Keep validation and cache replay from changing training augmentation/RNG.

    Restoration occurs before validated checkpoint publication so live and
    resumed training use exactly the same next sample and random operations.
    """
    state = _rng_state(torch, np, generator, cuda)
    try:
        yield
    finally:
        _restore_rng(state, torch, np, generator, cuda)


def make_optimizer(model, recipe, stage):
    sampling = _classifier_sampling_policy(recipe)
    loss_type = _classification_loss_type(recipe)
    model.set_training_stage(stage)
    model.roi_heads.classifier_negative_sampling = sampling
    model.roi_heads.classification_loss_type = loss_type
    model.roi_heads.vfl_alpha = recipe["vfl_alpha"]
    model.roi_heads.vfl_gamma = recipe["vfl_gamma"]
    margin, quality_gamma, negative = _vfl_sharpening(recipe)
    model.roi_heads.vfl_quality_margin = margin
    model.roi_heads.vfl_quality_gamma = quality_gamma
    model.roi_heads.vfl_negative_weight = negative
    model.roi_heads.box_giou_weight = _box_giou_weight(recipe)
    if OCCLUSION_WEIGHT_OVERRIDE_FIELD in recipe:
        model.roi_heads.occlusion_weight = recipe[OCCLUSION_WEIGHT_OVERRIDE_FIELD]
    model.roi_heads.verifier_ranking = bool(recipe.get("verifier_ranking", True))
    fpn_lr = _fpn_lr(recipe)
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if stage == "classifier":
            group, lr = "classifier", recipe["classifier_lr"]
        elif fpn_lr is not None and name.startswith("backbone.fpn."):
            group, lr = "fpn", fpn_lr
        elif name.startswith("backbone."):
            group, lr = "backbone", recipe["backbone_lr"]
        elif name.startswith(("roi_heads.gated_classifier.", "roi_heads.od_mask_head.hidden_predictor.",
                              "roi_heads.od_mask_head.hidden_tower.", "roi_heads.od_mask_head.hidden_upsample.",
                              "roi_heads.od_mask_head.amodal_tower.", "roi_heads.od_mask_head.amodal_predictor.",
                              "roi_heads.context_fusion.", "roi_heads.context_cascade.",
                              "roi_heads.foreground_head.", "roi_heads.foreground_fusion.",
                              "visible_proposal_branch.", "shallow_detail_fusion.",
                              "roi_heads.mask_geometry_head.", "roi_heads.mask_geometry_refiner.",
                              "roi_heads.occlusion_verifier.")):
            group, lr = "new_head", recipe["new_head_lr"]
        else:
            group, lr = "pretrained_head", recipe["pretrained_head_lr"]
        groups.setdefault(group, dict(params=[], lr=lr, base_lr=lr, name=group))["params"].append(parameter)
    if not groups:
        raise ValueError("Training stage has no trainable parameters")
    return torch.optim.AdamW(list(groups.values()), weight_decay=recipe["weight_decay"])


def _validate_loss_dict(losses, source, policies):
    """Validate every raw model loss before any supervision selection."""
    if not losses:
        raise ValueError("Detector returned no losses")
    policy = policies[source]
    if set(losses) - set(policy):
        raise ValueError("Every returned loss requires an explicit source weight")
    if source == "MinneApple" and any("amodal" in key or "hidden" in key or key == "loss_box_reg"
                                    or key.startswith('loss_mask_geometry_') for key in losses):
        raise ValueError("Visible-only source unexpectedly produced final A-box or A/H supervision")
    if any(value.ndim or not bool(torch.isfinite(value)) for value in losses.values()):
        raise ValueError("Detector losses must be finite scalars")
    return policy


def weighted_loss(losses, source, policies):
    policy = _validate_loss_dict(losses, source, policies)
    return sum(value * policy[key] for key, value in losses.items())


def supervised_loss(losses, source, policies, *, stage,
                    unknown_amodal_joint_policy="legacy_detection"):
    """Return (objective, optimized raw losses, omitted raw loss names).

    Only the explicit joint/MinneApple visible-mask policy removes detection
    loss edges. Omitted losses are never multiplied by zero. Validation still
    inspects the full original dictionary, including every omitted value.
    Other source/stage combinations preserve legacy weighting and order.
    """
    policy = _validate_loss_dict(losses, source, policies)
    if stage not in ("joint", "classifier"):
        raise ValueError("Supervision stage must be joint or classifier")
    if (type(unknown_amodal_joint_policy) is not str
            or unknown_amodal_joint_policy not in JOINT_POLICIES):
        raise ValueError("unknown_amodal_joint_policy must be legacy_detection or visible_mask_only")
    optimized = losses
    if (stage == "joint" and source == "MinneApple"
            and unknown_amodal_joint_policy == "visible_mask_only"):
        required = {"loss_visible_bce", "loss_visible_dice"}
        if not required.issubset(losses):
            raise ValueError("Visible-mask-only joint supervision requires both visible BCE and Dice losses")
        optimized = {name: value for name, value in losses.items() if name in required}
    omitted = tuple(name for name in losses if name not in optimized)
    objective = sum(value * policy[name] for name, value in optimized.items())
    return objective, optimized, omitted


class EmaWeights:
    """Shadow parameters for validation only.

    The published checkpoint always keeps the live weights; EMA is swapped in
    around a single evaluation and restored before any checkpoint is written,
    so a shocked EMA can never leak into a saved artifact.
    """

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = {name: tensor.detach().clone()
                       for name, tensor in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for name, tensor in model.state_dict().items():
            shadow = self.shadow[name]
            if shadow.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(tensor.detach(), alpha=1 - self.decay)
            else:
                shadow.copy_(tensor)

    @torch.no_grad()
    def swap_in(self, model):
        backup = {name: tensor.detach().clone()
                  for name, tensor in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return backup


def run_od_training(config_path, *, device="cuda", output=None, smoke=False, resume=False):
    install_windows_replace_retry()
    cfg = load_od_config(config_path, output=output, smoke=smoke)
    recipe, paths, out, root = cfg["recipe"], cfg["paths"], cfg["output"], cfg["root"]
    ignore_path = paths.get("ignore_boxes")
    ignore_map, ignore_sha = {}, None
    if ignore_path is not None:
        ignore_payload = json.loads(Path(ignore_path).read_text(encoding="utf-8"))
        if (ignore_payload.get("schema") != "apple_train_ignore_candidate_boxes_v1"
                or not isinstance(ignore_payload.get("records"), dict)):
            raise ValueError("Ignore box audit file has an unexpected schema")
        # The generated file stores one record with a list of candidates per image.
        ignore_map = {name: [candidate["box"] for candidate in rows]
                      for name, rows in ignore_payload["records"].items()}
        for name, boxes in ignore_map.items():
            for box in boxes:
                if (not isinstance(box, list) or len(box) != 4
                        or any(isinstance(value, bool) or not isinstance(value, (int, float))
                               or not math.isfinite(value) for value in box)):
                    raise ValueError("Ignore box must be a finite XYXY list")
        ignore_sha = sha(ignore_path)
    source_order = _training_source_order(recipe)
    accumulation = len(source_order)
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError("Use a fresh OD experiment output or --resume")
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "trainer.lock"
    if lock.exists():
        previous = json.loads(lock.read_text(encoding="utf-8"))
        try:
            process = psutil.Process(previous["pid"])
            alive = process.is_running() and abs(process.create_time() - previous["created"]) < .01
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            raise RuntimeError("A verified live trainer already owns this output")
        lock.unlink()
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), created=psutil.Process().create_time()), stream)
    updates = 0
    stop = dict(requested=False)
    previous_signal = signal.signal(signal.SIGINT, lambda *_: stop.update(requested=True))
    try:
        torch.set_num_threads(recipe["cpu_threads"])
        random.seed(recipe["seed"])
        np.random.seed(recipe["seed"])
        torch.manual_seed(recipe["seed"])
        dev = torch.device(device)
        if dev.type not in ("cpu", "cuda") or (dev.type == "cuda" and not torch.cuda.is_available()):
            raise RuntimeError("Requested training device is unavailable")
        if dev.type == "cuda" and recipe["amp_dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Configured BF16 is unsupported on this device")
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(recipe["seed"])
            torch.cuda.reset_peak_memory_stats()
        sources = source_snapshot()
        config_sha = sha(cfg["path"])
        version = code_version(root, cfg["path"], smoke=smoke)
        for path, digest in REFERENCE_PINS.items():
            if sha(path) != digest:
                raise ValueError("Frozen metric reference changed")
        datasets = {
            "PApple": ODInputDataset(PAppleDataset(paths["papple_root"], "train", 32),
                                     short_sides=recipe["short_sides"], max_size=recipe["max_size"], augment=recipe["augment"]),
            "MinneApple": ODInputDataset(MinneAppleDataset(paths["minne_root"], paths["minne_manifest"],
                                            split="train", input_size=32, tiled=True),
                                         short_sides=recipe["short_sides"], max_size=recipe["max_size"], augment=recipe["augment"]),
        }
        # Minne's ordinary metadata() performs a pixel-level capacity audit.
        # Keep its manifest/sampler lineage without opening inactive frames.
        metadata = {name: (dataset.metadata() if name in source_order else
                    dict(dataset=name, split="train", items=len(dataset), active_training_source=False,
                         metadata_scope="inactive_annotation_inventory_only_no_pixel_capacity_audit",
                         annotation_paths={key: str(path) if path is not None else None
                                           for key, path in dataset.annotation_paths.items()},
                         annotation_sha256=deepcopy(dataset.annotation_sha256),
                         input_preprocessing=deepcopy(getattr(dataset, "configuration", {}))))
                    for name, dataset in datasets.items()}
        papple_per_update = source_order.count("PApple")
        if len(datasets["PApple"]) % papple_per_update:
            raise ValueError("Epoch accounting requires an even original PApple inventory")
        epoch_updates = len(datasets["PApple"]) // papple_per_update
        joint_updates = recipe["joint_papple_epochs"] * epoch_updates
        maximum = 2 if smoke else joint_updates + recipe["classifier_papple_epochs"] * epoch_updates
        identity = dict(recipe=recipe, loss_weights=cfg["raw"]["loss_weights"], model=cfg["raw"]["model"],
                        parent_sha256=cfg["raw"]["parent_sha256"], source_sha256=sources, config_sha256=config_sha,
                        datasets=metadata, papple_provenance_sha256=sha(paths["papple_provenance"]),
                        minne_manifest_sha256=sha(paths["minne_manifest"]),
                        ignore_boxes_sha256=ignore_sha, smoke=smoke,
                        source_order=list(source_order), batch_size=1, accumulation=accumulation, workers=0,
                        joint_updates=joint_updates, maximum_updates=maximum,
                        native_GT_CPU=True, unknown_amodal_supervision=False,
                        runtime=dict(python=sys.version.split()[0], torch=torch.__version__,
                            torchvision=torchvision.__version__, cuda_build=torch.version.cuda,
                            device=str(dev), gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else None))
        if cfg["initialization"]["kind"] == "validated_od":
            identity["initialization"] = deepcopy(cfg["initialization"])
            identity["initial_validation_skipped_on_verified_parent"] = not recipe.get("validate_initial", True)
        samplers = {name: CyclingOrder(len(dataset), recipe["seed"] + i)
                    for i, (name, dataset) in enumerate(datasets.items())}
        generator = torch.Generator(device="cpu").manual_seed(recipe["seed"])
        history, validations, best, joint_best, transitions = [], [], None, None, []
        if resume:
            saved = load_checkpoint(out / "last.pt")
            if saved["identity"] != identity or saved["smoke"] or smoke:
                raise ValueError("Resume requires the identical formal experiment")
            model = build_od_amodal_detector(weights=None, **saved["constructor_config"])
            model.load_state_dict(saved["model"], strict=True)
            model.checkpoint_provenance = saved["parent_provenance"]
            stage = saved["stage"]
            model.to(dev)
            optimizer = make_optimizer(model, recipe, stage)
            optimizer.load_state_dict(saved["optimizer"])
            for name, sampler in samplers.items():
                sampler.load_state_dict(saved["samplers"][name])
            _restore_rng(saved["rng"], torch, np, generator, dev.type == "cuda")
            updates, history, validations, best = saved["updates"], saved["history"], saved["validations"], saved["best"]
            joint_best, transitions = saved["joint_best"], saved["transitions"]
            version = {**saved["code_version"], "resumed_on_commit": version["training_commit"]}
            del saved
        elif cfg["initialization"]["kind"] == "validated_od":
            model, _ = _load_validated_od_initializer(cfg)
            model.to(dev)
            stage = "joint" if joint_updates else "classifier"
            optimizer = make_optimizer(model, recipe, stage)
        elif cfg["initialization"]["kind"] == "backbone_swap":
            model, _ = _load_backbone_swap_initializer(cfg)
            model.to(dev)
            stage = "joint" if joint_updates else "classifier"
            optimizer = make_optimizer(model, recipe, stage)
        else:
            model = build_od_amodal_detector(weights=None, checkpoint_path=paths["parent_checkpoint"],
                        expected_sha256=cfg["raw"]["parent_sha256"], **cfg["raw"]["model"]).to(dev)
            stage = "joint"
            optimizer = make_optimizer(model, recipe, stage)
        decay = float(recipe.get("ema_decay", 0.) or 0.)
        ema = None if not decay else EmaWeights(model, decay)
        model.train()

        def boundary(phase):
            if source_snapshot() != sources:
                raise RuntimeError("Consumed OD sources changed during training")
            if sha(cfg["path"]) != config_sha:
                raise RuntimeError("OD config changed during training")
            for dataset in datasets.values():
                for key, path in dataset.annotation_paths.items():
                    if path is not None and sha(path) != dataset.annotation_sha256[key]:
                        raise RuntimeError("Training annotation changed")
            if ignore_sha is not None and sha(ignore_path) != ignore_sha:
                raise RuntimeError("Ignore box audit file changed")
            payload = dict(schema=SCHEMA, identity=identity, smoke=smoke, phase=phase, stage=stage,
                           updates=updates, model=model.state_dict(), optimizer=optimizer.state_dict(),
                           constructor_config=model.constructor_config, model_config=model.model_config,
                           parent_provenance=model.checkpoint_provenance, code_version=version,
                           samplers={name: sampler.state_dict() for name, sampler in samplers.items()},
                           rng=_rng_state(torch, np, generator, dev.type == "cuda"),
                           history=history, validations=validations, best=best, joint_best=joint_best,
                           transitions=transitions)
            digest = save_checkpoint(out / "last.pt", payload)
            return payload, digest

        def validate():
            nonlocal best, joint_best
            with validation_rng_isolation(generator, cuda=dev.type == "cuda"):
                from .od_evaluate import evaluate_od
                validation_dir = out / "validation" / f"update_{updates:06d}_{time.time_ns()}"
                dataset = ODInputDataset(PAppleDataset(paths["papple_root"], "val", 32),
                                         short_sides=(recipe["validation_short_side"],), max_size=recipe["validation_max_size"])
                atomic_json(out / "training_status.json", dict(phase="validating", pid=os.getpid(), updates=updates,
                    stage=stage, best=best, maximum_updates=maximum, code_version=version,
                    validation_directory=validation_dir.relative_to(out).as_posix(), expected_images=814,
                    expected_GT=3449, expected_empty_images=13))
                print(f"OD_VALIDATION starting update={updates} images=814", flush=True)
                backup = None
                if ema is not None:
                    backup = ema.swap_in(model)
                try:
                    evaluated = evaluate_od(model, dataset, validation_dir, device=dev,
                                            provenance_path=paths["papple_provenance"],
                                            defer_mask_metrics=recipe.get("defer_mask_metrics", False))
                finally:
                    if backup is not None:
                        model.load_state_dict(backup, strict=True)
                        del backup
            summary = evaluated["summary"]
            if (evaluated["phase"] != "complete" or not evaluated["model_selection_eligible"]
                    or (summary["images"], summary["gt_count"], summary["empty_gt_images"]) != (814, 3449, 13)):
                raise ValueError("Only complete original VAL can select weights")
            point = dict(update=updates, stage=stage, confidence=evaluated["confidence"], score=evaluated["score"],
                         summary=summary, directory=validation_dir.relative_to(out).as_posix())
            point["beats_parent"] = point["score"] > cfg["raw"]["parent_selection_floor"]
            validations.append(point)
            improved = best is None or point["score"] > best["score"]
            joint_improved = stage == "joint" and (joint_best is None or point["score"] > joint_best["score"])
            if improved:
                best = point
            if joint_improved:
                joint_best = point
            payload, _ = boundary("training")
            # When EMA drives selection, the published artifact must be the
            # weights that were actually scored: otherwise the reported number
            # belongs to a tensor set that no longer exists on disk and cannot
            # be reproduced by any downstream TTA or fusion pass.
            artifact = payload
            if ema is not None:
                artifact = dict(payload)
                artifact["model"] = ema.shadow
                artifact["weights_are_ema"] = True
                artifact["ema_decay"] = float(recipe["ema_decay"])
            save_checkpoint(out / f"validated_{updates:06d}.pt", artifact)
            if improved:
                save_checkpoint(out / "candidate_best.pt", artifact)
                if point["beats_parent"]:
                    save_checkpoint(out / "best.pt", artifact)
            if joint_improved:
                save_checkpoint(out / "joint_best.pt", artifact)
            model.train()
            print(f"OD_VALIDATION complete update={updates} precision={summary['precision']:.6f} recall={summary['recall']:.6f} improved={improved}", flush=True)

        atomic_json(out / "resolved_experiment.json", dict(identity=identity, code_version=version,
                    parent_provenance=model.checkpoint_provenance, constructor_config=model.constructor_config,
                    config_sha256=config_sha, parent_selection_floor=cfg["raw"].get("parent_selection_floor")))
        atomic_json(out / "training_status.json", dict(phase="initializing", pid=os.getpid(), updates=updates,
                    stage=stage, code_version=version, maximum_updates=maximum))
        if not resume and not smoke:
            if recipe.get("validate_initial", True):
                validate()
            else:
                # _load_validated_od_initializer has already proved the source
                # update's full VAL and copied every tensor exactly. Its point
                # remains external lineage; this run has no calibration yet.
                boundary("training")
        while updates < maximum and not stop["requested"] and not (out / "STOP").exists():
            if stage == "joint" and updates == joint_updates:
                if joint_best is None:
                    raise RuntimeError("Classifier stage requires a validated joint checkpoint")
                selected = load_checkpoint(out / "joint_best.pt")
                model.load_state_dict(selected["model"], strict=True)
                transitions.append(dict(at_update=updates, from_stage="joint", to_stage="classifier",
                    selected_update=joint_best["update"], selected_score=joint_best["score"],
                    selected_sha256=sha(out / "joint_best.pt"), optimizer_reset=True))
                del selected
                stage = "classifier"
                optimizer = make_optimizer(model, recipe, stage)
                model.train()
            started = time.monotonic()
            stage_update = updates if stage == "joint" else updates - joint_updates
            stage_maximum = joint_updates if stage == "joint" else maximum - joint_updates
            warmup = recipe["warmup_updates"]
            if stage_update < warmup:
                factor = (stage_update + 1) / warmup
            else:
                fraction = min(1., (stage_update - warmup) / max(1, stage_maximum - warmup))
                factor = .1 + .9 * .5 * (1 + math.cos(math.pi * fraction))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            rows, loss_totals = [], {}
            total = 0.
            for source in source_order:
                index = samplers[source].next()
                item = datasets[source][index]
                if source == "PApple" and ignore_map:
                    image_id = item["target"]["source"]["image_id"]
                    attach_ignore_boxes(item["target"], ignore_map.get(image_id),
                                        item["pixel_values"].shape[-2:])
                fingerprint = target_fingerprint(item["target"])
                pixels = item["pixel_values"].unsqueeze(0).to(dev)
                pixel_mask = item["pixel_mask"].unsqueeze(0).to(dev)
                enabled = dev.type == "cuda" and recipe["amp_dtype"] == "bfloat16"
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=enabled):
                    losses = model(pixels, pixel_mask, targets=[item["target"]])
                    loss, optimized_losses, omitted_names = supervised_loss(
                        losses, source, cfg["raw"]["loss_weights"], stage=stage,
                        unknown_amodal_joint_policy=recipe.get(JOINT_POLICY_FIELD, "legacy_detection"))
                (loss / accumulation).backward()
                if target_fingerprint(item["target"]) != fingerprint:
                    raise RuntimeError("Detector mutated native GT")
                if source == "MinneApple":
                    # Shared/V parameters may learn from V; unknown H/A have
                    # no supervision. Per-sample branch isolation is unit tested.
                    assert item["target"]["amodal_masks"] is None
                total += float(loss.detach()) / accumulation
                for name, value in optimized_losses.items():
                    loss_totals[name] = loss_totals.get(name, 0.) + float(value.detach()) / accumulation
                rows.append(dict(dataset=source, index=index, input_hw=list(pixels.shape[-2:]),
                    GT_count=len(item["target"]["instance_ids"]), source=dict(item["target"]["source"]),
                    rgb_sha256=sha(item["target"]["source"]["rgb_path"]),
                    optimized_loss_names=list(optimized_losses),
                    raw_omitted_loss_names=list(omitted_names),
                    raw_omitted_losses={name: float(losses[name].detach()) for name in omitted_names}))
                del losses, optimized_losses, loss, pixels, pixel_mask, item
            norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                    recipe["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            updates += 1
            row = dict(update=updates, stage=stage, loss=total, losses=loss_totals,
                loss_aggregation_scope="optimized_loss_components_unweighted_mean_over_accumulation",
                grad_norm=float(norm), seconds=time.monotonic() - started,
                samples=rows, learning_rates={group["name"]: group["lr"] for group in optimizer.param_groups},
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if dev.type == "cuda" else None)
            history.append(row)
            print(f"OD successful={updates}/{maximum} stage={stage} loss={total:.6f} seconds={row['seconds']:.2f}", flush=True)
            if not smoke and (updates in recipe["pilot_validation_updates"] or updates % recipe["validate_every"] == 0
                               or updates in (joint_updates, maximum)):
                validate()
            elif updates % recipe["checkpoint_every"] == 0:
                boundary("training")
            atomic_json(out / "training_status.json", dict(phase="training", pid=os.getpid(), updates=updates,
                stage=stage, last_update=row, best=best, joint_best=joint_best, maximum_updates=maximum,
                code_version=version, goal_achieved=bool(best and best["summary"].get("goal_95_met"))))
        phase = ("smoke_complete" if smoke else "complete") if updates == maximum else "interrupted"
        _, digest = boundary(phase)
        restored = load_checkpoint(out / "last.pt")
        for name, value in model.state_dict().items():
            if not torch.equal(value.detach().cpu(), restored["model"][name]):
                raise RuntimeError("Saved OD tensor roundtrip differs: " + name)
        del restored
        result = dict(schema=SCHEMA, phase=phase, pid=os.getpid(), updates=updates, stage=stage, best=best,
            joint_best=joint_best, validations=validations, transitions=transitions, identity=identity,
            checkpoint_sha256=digest, own_model_roundtrip_verified=True, code_version=version,
            goal_achieved=bool(best and best["summary"].get("goal_95_met")))
        atomic_json(out / "training_summary.json", result)
        atomic_json(out / "training_status.json", result)
        return result
    except BaseException as error:
        atomic_json(out / "training_status.json", dict(phase="failed", pid=os.getpid(), updates=updates,
                    failure_type=type(error).__name__, failure=str(error)))
        raise
    finally:
        signal.signal(signal.SIGINT, previous_signal)
        lock.unlink(missing_ok=True)
