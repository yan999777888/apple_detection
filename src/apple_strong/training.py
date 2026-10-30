"""Strong Base-only C0/C1 warm-start training with explicit four-view gates.

This module never loads a query student or Small detector. Legacy architecture
code is imported only from the immutable local strong_runtime snapshot.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import os
import random
import re
import signal
import time

SCHEMA = "apple_strong_c0c1_update_boundary_v1"
BASE_SHA = "0fe257c5bfe115a77ac7deac830af479e4c2b0ad6ff69e609901b24c7980fa9b"
SMALL_SHA = "e0c1f659c2509d7a01d617e12cb83c673768b440ea5b4b96cb4f763e4207e7e3"
IGNORE_SHA = "79d6e40e6ed74b179a159d491cb4940a7d2cdf5a979307cc6a6fc671e2d7d958"
BASELINE_ARTIFACTS = frozenset((
    "base/original_predictions.json", "base/hflip_predictions.json",
    "small/original_predictions.json", "small/hflip_predictions.json",
    "base/prediction_metadata.json", "small/prediction_metadata.json",
    "fused_predictions.json", "base/mask_metrics_at_recall90.json",
    "base/mask_pair_diagnostics.json", "fusion_summary.json",
))
DEFAULTS = dict(max_updates=1152, accumulate=4, checkpoint_every=50, gate_updates=288,
                seed=20261030, cpu_threads=4, short_sides=[768, 896, 1024], max_size=1536,
                head_lr_scale=.1, warmup_updates=50, ema_decay=.999,
                consistency_weight=.05, gradient_clip_norm=1., amp_dtype="bfloat16",
                smoke_indices=[63, 475, 0, 2060])


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_config(path):
    path = Path(path).absolute()
    raw = read(path)
    if raw.get("schema") != "apple_strong_c0c1_v1" or raw.get("arm") not in ("C0", "C1"):
        raise ValueError("An explicit strong C0/C1 configuration is required")
    base = (path.parent / raw["path_base"]).absolute()
    paths = {key: (base / value).absolute() for key, value in raw["paths"].items()}
    required = {"parent_checkpoint", "parent_recipe_config", "papple_root", "ignore_boxes", "baseline_gate", "output"}
    if not required <= paths.keys():
        raise ValueError("Missing strong-model path fields")
    if raw.get("parent_sha256") != BASE_SHA or raw.get("ignore_sha256") != IGNORE_SHA:
        raise ValueError("Strong Base and original TRAIN ignore identity must remain pinned")
    if base.drive.lower() != "f:" or any(p.drive.lower() != "f:" for p in paths.values()):
        raise ValueError("This authorized local experiment stores files and assets on F drive")
    recipe = {**DEFAULTS, **raw.get("training", {})}
    for key in ("max_updates", "accumulate", "checkpoint_every", "gate_updates", "cpu_threads", "warmup_updates", "max_size"):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ValueError("Positive integer required: " + key)
    if recipe["accumulate"] != 4 or recipe["gate_updates"] >= recipe["max_updates"]:
        raise ValueError("Use four original TRAIN frames/update and a real intermediate gate")
    if recipe["amp_dtype"] != "bfloat16":
        raise ValueError("Keep the inherited BF16 numerical policy")
    for key in ("head_lr_scale", "consistency_weight", "gradient_clip_norm"):
        if isinstance(recipe[key], bool) or not isinstance(recipe[key], (int, float)) or not math.isfinite(recipe[key]) or recipe[key] <= 0:
            raise ValueError("Positive finite coefficient required: " + key)
    if not 0 < recipe["ema_decay"] < 1:
        raise ValueError("EMA must remain enabled and persistent")
    if not recipe["short_sides"] or any(type(v) is not int or v < 1 for v in recipe["short_sides"]):
        raise ValueError("Positive short-side inventory required")
    if len(recipe["smoke_indices"]) != recipe["accumulate"] or any(type(v) is not int or v < 0 for v in recipe["smoke_indices"]):
        raise ValueError("Stress requires four explicit TRAIN indices")
    return dict(path=path, raw=raw, base=base, paths=paths, recipe=recipe, arm=raw["arm"])


def verify_baseline(path, runtime):
    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate key in baseline gate: " + key)
            result[key] = item
        return result
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)
    if value.get("schema") != "strong_full_four_view_gate_v1" or value.get("passed") is not True:
        raise ValueError("Full strong four-view reproduction must pass first")
    if value.get("base_checkpoint_sha256") != BASE_SHA or value.get("small_checkpoint_sha256") != SMALL_SHA:
        raise ValueError("Baseline gate does not bind the delivered strong models")
    scope = value.get("scope", value)
    images = scope.get("images", value.get("images"))
    gt = scope.get("gt_count", scope.get("GT", scope.get("GT_instances", value.get("gt_count"))))
    empty=scope.get("empty",scope.get("empty_gt_images",value.get("empty",value.get("empty_gt_images"))))
    if (images, gt, empty) != (814, 3449, 13):
        raise ValueError("Baseline gate must bind complete original 814-image/3449-GT VAL")
    sources = value.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Baseline gate lacks actual runtime source bindings")
    for name, digest in sources.items():
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = Path(runtime).parent / name
        try:
            candidate.resolve().relative_to(Path(runtime).parent.resolve())
        except ValueError:
            raise ValueError("Baseline source binding escaped repository: " + name) from None
        if not candidate.is_file() or sha(candidate) != digest:
            raise ValueError("Baseline source binding differs: " + name)
    fusion = value["fusion"]
    if (fusion.get("images"),fusion.get("gt_count"),fusion.get("empty_gt_images")) != (814,3449,13):
        raise ValueError("Fusion report scope differs from original full VAL")
    if (fusion["tp"], fusion["fp"], fusion["fn"]) != (3105, 183, 344):
        raise ValueError("Strong 94.43/R90.03 count gate has not been reproduced")
    # A passed scalar report is insufficient. Bind the actual fresh four-view
    # predictions, metadata, fused output and mask evidence before training.
    repository = Path(runtime).parent.resolve(strict=True)
    directory_value = value.get("baseline_directory")
    if not isinstance(directory_value, str) or not directory_value:
        raise ValueError("Baseline gate lacks the actual baseline directory")
    directory = Path(directory_value)
    if not directory.is_absolute() or directory.drive.lower() != "f:":
        raise ValueError("Baseline directory must be an absolute F-drive path")
    try:
        directory = directory.resolve(strict=True)
        directory.relative_to(repository)
    except (OSError, ValueError):
        raise ValueError("Baseline directory is missing or escaped repository") from None
    if directory.drive.lower() != "f:" or not directory.is_dir():
        raise ValueError("Baseline directory must be a real F-drive directory")
    artifacts = value.get("artifact_sha256")
    if not isinstance(artifacts, dict) or len(artifacts) != len(BASELINE_ARTIFACTS):
        raise ValueError("Baseline gate requires all ten artifact SHA256 bindings")
    normalized, resolved = set(), set()
    for name, digest in artifacts.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid baseline artifact path")
        relative = Path(name)
        if relative.anchor or relative.drive or relative.root or ".." in relative.parts:
            raise ValueError("Baseline artifact path must be relative and cannot escape")
        canonical = relative.as_posix().casefold()
        if canonical in normalized:
            raise ValueError("Duplicate normalized baseline artifact path")
        normalized.add(canonical)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Invalid baseline artifact SHA256: " + name)
        try:
            artifact = (directory / relative).resolve(strict=True)
            artifact.relative_to(directory)
        except (OSError, ValueError):
            raise ValueError("Baseline artifact is missing or escaped its directory: " + name) from None
        normalized_target = os.path.normcase(str(artifact))
        if normalized_target in resolved:
            raise ValueError("Duplicate resolved baseline artifact path")
        resolved.add(normalized_target)
        if not artifact.is_file() or sha(artifact) != digest:
            raise ValueError("Baseline artifact SHA256 differs: " + name)
    if normalized != BASELINE_ARTIFACTS:
        raise ValueError("Baseline artifact inventory differs from the full required output set")
    return value


def load_ignore(path, expected, train_names):
    if sha(path) != expected:
        raise ValueError("Original TRAIN ignore map hash differs")
    value = read(path)
    if value.get("schema") != "apple_train_ignore_candidate_boxes_v1" or not isinstance(value.get("records"), dict):
        raise ValueError("TRAIN ignore map format differs")
    if not set(value["records"]) <= set(train_names):
        raise ValueError("Ignore map contains images outside TRAIN")
    result = {}
    for name, records in value["records"].items():
        result[name] = []
        for record in records:
            box = record["box"]
            if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Invalid ignore box")
            result[name].append(list(box))
    if (len(result), sum(map(len, result.values()))) != (383, 444):
        raise ValueError("Original ignore inventory changed")
    return result


class Cursor:
    def __init__(self, count, seed):
        self.rng = random.Random(seed)
        self.order = list(range(count)); self.rng.shuffle(self.order)
        self.cursor = 0
    def next(self):
        if self.cursor == len(self.order):
            self.rng.shuffle(self.order); self.cursor = 0
        item = self.order[self.cursor]; self.cursor += 1
        return item
    def state_dict(self):
        return dict(order=list(self.order), cursor=self.cursor, rng=self.rng.getstate())
    def load_state_dict(self, value):
        if sorted(value["order"]) != list(range(len(self.order))) or not 0 <= value["cursor"] <= len(self.order):
            raise ValueError("Sampler inventory/cursor differs")
        self.order = list(value["order"]); self.cursor = value["cursor"]; self.rng.setstate(value["rng"])


def lr_factor(successful, maximum, warmup):
    step = successful + 1
    if step <= warmup:
        return step / warmup
    return .05 + .95 * .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, maximum - warmup))))


def authorize_continuation(arm, last_candidate, previous, supplied_path):
    path=Path(supplied_path) if supplied_path is not None else (Path(previous["path"]) if previous else None)
    if path is None:raise ValueError("A bound full four-view continuation gate is required")
    gate=read(path)
    candidate=previous["trigger_candidate_sha256"] if previous else last_candidate
    if gate.get("passed") is not True or gate.get("candidate_sha256")!=candidate or gate.get("arm")!=arm:
        raise ValueError("Continuation gate is not bound to this candidate")
    if previous and sha(path)!=previous["gate_sha256"]:raise ValueError("Previously authorized continuation gate changed")
    return dict(path=str(path.absolute()),gate_sha256=sha(path),trigger_candidate_sha256=candidate)


def update_boundary(recipe, target_updates, continuation, requested=None):
    if requested is not None and (type(requested) is not int or requested<1):
        raise ValueError("Stop boundary must be a positive integer")
    permitted=target_updates if continuation else recipe["gate_updates"]
    return min(requested or permitted,permitted,target_updates)


def freeze_backbone(model):
    model.backbone.requires_grad_(False)
    model.backbone.eval()
    return tuple(name for name, p in model.named_parameters() if not p.requires_grad)


def enter_train(model):
    model.train()
    model.backbone.eval()
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("Frozen strong backbone/FPN became trainable")


class CpuEma:
    """Persistent complete CPU shadow; frozen tensors never require GPU copies."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
        self.active = {n for n, p in model.named_parameters() if p.requires_grad}
        self.active.update(n for n, _ in model.named_buffers() if not n.startswith("backbone."))
    def update(self, model):
        for name, value in model.state_dict().items():
            if name not in self.active:
                continue
            current = value.detach().cpu()
            if current.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(current, alpha=1-self.decay)
            else:
                self.shadow[name].copy_(current)
    def state_dict(self):
        return dict(decay=self.decay, active=sorted(self.active), shadow=self.shadow)
    def load_state_dict(self, value):
        import torch
        if value["decay"] != self.decay or set(value["active"]) != self.active or set(value["shadow"]) != set(self.shadow):
            raise ValueError("EMA contract differs")
        for n, tensor in value["shadow"].items():
            if tensor.dtype != self.shadow[n].dtype or tensor.shape != self.shadow[n].shape:
                raise ValueError("EMA tensor differs")
            self.shadow[n].copy_(tensor)
    def verify_frozen(self, model):
        import torch
        for n, tensor in model.state_dict().items():
            if n.startswith("backbone.") and not torch.equal(tensor.detach().cpu(), self.shadow[n]):
                raise RuntimeError("Frozen backbone/FPN state changed: " + n)


@contextmanager
def capture_gt_predictions(model):
    """Capture the same train-forward graph; no second GT-conditioned head."""
    captures = {}
    head = model.roi_heads
    originals = {}
    for name in ("select_training_samples", "_roi_predictions"):
        originals[name] = (name in head.__dict__, head.__dict__.get(name), getattr(head, name))
    def select(proposals, targets):
        result = originals["select_training_samples"][2](proposals, targets)
        captures.update(proposals=result[0][0], matched=result[1][0], labels=result[2][0], targets=targets)
        return result
    def roi(*args, **kwargs):
        result = originals["_roi_predictions"][2](*args, **kwargs)
        captures["mask_predictions"] = result[0]
        return result
    def predictor(_module, _inputs, output):
        captures["box_regression"] = output[1]
    head.select_training_samples = select
    head._roi_predictions = roi
    handle = head.box_predictor.register_forward_hook(predictor)
    try:
        yield captures
    finally:
        handle.remove()
        for name, (had, value, _method) in originals.items():
            if had: setattr(head, name, value)
            else: delattr(head, name)


def select_gt_capture(model, captured, image_hw, detach=False):
    import torch
    from torchvision.ops import box_iou
    proposals, labels, matched = (captured[key] for key in ("proposals", "labels", "matched"))
    prepared = captured["targets"][0]
    decoded = model.roi_heads._decoded_boxes_all(captured["box_regression"], proposals, image_hw)
    chosen, identities = [], []
    for gt_index, instance_id in enumerate(prepared["instance_ids"]):
        candidates = torch.where((labels > 0) & (matched == gt_index))[0]
        candidates = candidates[(decoded[candidates, 2:] > decoded[candidates, :2]).all(-1)]
        if not len(candidates):
            continue
        overlap = box_iou(proposals[candidates].detach(), prepared["boxes"][gt_index:gt_index+1]).flatten()
        best = candidates[torch.argmax(overlap)]
        chosen.append(int(best)); identities.append(instance_id)
    index = torch.tensor(chosen, dtype=torch.int64, device=proposals.device)
    predictions = captured["mask_predictions"]
    result = dict(instance_ids=identities, boxes=decoded[index], roi_boxes=proposals[index],
                  visible_logits=predictions["visible_logits"][index], amodal_logits=predictions["amodal_logits"][index])
    if detach:
        result = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
    return result


def to_cpu(value):
    import torch
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, list): return [to_cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(to_cpu(v) for v in value)
    return value


def equal_state(left, right):
    import torch
    import numpy as np
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray): return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict): return isinstance(right, dict) and left.keys()==right.keys() and all(equal_state(left[k],right[k]) for k in left)
    if isinstance(left, (list,tuple)): return type(left) is type(right) and len(left)==len(right) and all(equal_state(a,b) for a,b in zip(left,right))
    return left==right


def save_torch_atomic(path, payload):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}.{time.time_ns()}")
    with temporary.open("xb") as stream:
        torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)
    digest=sha(path)
    sidecar=path.with_suffix(path.suffix+".sha256")
    tmp_sha=sidecar.with_name(sidecar.name+f".tmp.{os.getpid()}")
    tmp_sha.write_text(digest,encoding="ascii");os.replace(tmp_sha,sidecar)
    return digest


def run(config_path, *, device="cuda", smoke=False, resume=False, stop_after_updates=None,
        continue_after_gate=None, max_smoke_updates=2, smoke_indices=None):
    cfg=load_config(config_path);recipe=cfg["recipe"];paths=cfg["paths"]
    root=Path(__file__).absolute().parents[2];runtime=root/"strong_runtime"
    import torch, numpy as np, psutil
    from apple_seg.od_model import build_od_amodal_detector, prepare_detection_target
    from apple_seg.od_training import (load_checkpoint, make_optimizer, supervised_loss, attach_ignore_boxes)
    from apple_seg.od_inference import _strict_restore
    from apple_seg.data import PAppleDataset, flip_item_horizontal
    from apple_seg.od_data import ODInputDataset
    from apple_seg.training import target_fingerprint
    from .consistency import gt_anchored_consistency
    import apple_seg.od_model as model_source
    if Path(model_source.__file__).absolute() != runtime/"src/apple_seg/od_model.py":
        raise ValueError("Only the pinned strong_runtime model is allowed")
    baseline=verify_baseline(paths["baseline_gate"],runtime)
    dev=torch.device(device)
    if dev.type!="cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The authorized strong experiment requires CUDA BF16")
    torch.set_num_threads(recipe["cpu_threads"])
    random.seed(recipe["seed"]);np.random.seed(recipe["seed"]);torch.manual_seed(recipe["seed"]);torch.cuda.manual_seed_all(recipe["seed"])
    native=PAppleDataset(paths["papple_root"],"train",32,augment=False)
    if len(native)!=2304:raise ValueError("Full TRAIN inventory required")
    train_names=[row["filename"] for row in native.samples]
    ignore=load_ignore(paths["ignore_boxes"],cfg["raw"]["ignore_sha256"],train_names)
    dataset=ODInputDataset(native,short_sides=recipe["short_sides"],max_size=recipe["max_size"],augment=False)
    parent_recipe=read(paths["parent_recipe_config"])
    inherited=deepcopy(parent_recipe["training"])
    inherited.update(pretrained_head_lr=inherited["pretrained_head_lr"]*recipe["head_lr_scale"],
                     new_head_lr=inherited["new_head_lr"]*recipe["head_lr_scale"])
    if inherited.get("classification_loss")!="varifocal" or inherited.get("classifier_negative_sampling")!="gate_ohem_half_heavy":
        raise ValueError("The original strong scoring/negative policy must be inherited")
    if inherited.get("verifier_ranking") is not False:raise ValueError("Do not introduce a new ranking loss")
    source_paths=[Path(__file__),root/"src/apple_strong/consistency.py",root/"scripts/train_strong.py"]
    source_paths+=sorted((runtime/"src/apple_seg").glob("*.py"))
    sources={str(p.relative_to(root)):sha(p) for p in source_paths}
    identity=dict(arm=cfg["arm"],recipe=recipe,parent_sha256=BASE_SHA,ignore_sha256=IGNORE_SHA,
                  recipe_file_sha256=sha(paths["parent_recipe_config"]),baseline_gate_sha256=sha(paths["baseline_gate"]),
                  annotation_sha256=native.annotation_sha256,train_images=len(native),
                  source_sha256=sources,device_type=dev.type,amp_dtype="bfloat16",torch=torch.__version__,smoke=smoke)
    out=paths["output"]
    if out.exists() and any(p.name!="runtime.log" for p in out.iterdir()) and not resume:
        raise FileExistsError("Fresh output or exact resume required")
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/"training_status.json",dict(phase="initializing",identity=identity))
    if sha(paths["parent_checkpoint"])!=BASE_SHA:raise ValueError("Strong parent bytes differ")
    payload=load_checkpoint(paths["parent_checkpoint"])
    model=build_od_amodal_detector(weights=None,**payload["constructor_config"])
    restored=_strict_restore(model,payload["model"])
    constructor=deepcopy(payload["constructor_config"]);del payload
    if model.constructor_config != constructor:
        raise ValueError("Strong constructor reconstruction differs")
    model.to(dev)
    # Legacy make_optimizer applies every inherited loss/sampling switch. It
    # enables joint mode first; remove the frozen body/FPN from all groups next.
    optimizer=make_optimizer(model,inherited,"joint")
    freeze_backbone(model)
    for group in optimizer.param_groups:
        group["params"]=[p for p in group["params"] if p.requires_grad]
    optimizer.param_groups[:]=[g for g in optimizer.param_groups if g["params"]]
    if any(g["name"] in ("backbone","fpn") for g in optimizer.param_groups):raise RuntimeError("Frozen parameters reached optimizer")
    scaler=torch.amp.GradScaler("cuda",enabled=False)
    ema=CpuEma(model,recipe["ema_decay"])
    sampler=Cursor(len(dataset),recipe["seed"])
    updates=dict(attempted=0,successful=0,skipped=0);history=[];last_candidate=None;continuation=None
    def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    def restore_rng(state):
        random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch"]);torch.cuda.set_rng_state_all(state["cuda"])
    if resume:
        p=out/"last.pt"
        if sha(p)!=p.with_suffix(".pt.sha256").read_text().strip():raise ValueError("Own checkpoint hash differs")
        saved=torch.load(p,map_location="cpu",weights_only=False,mmap=False)
        if saved["schema"]!=SCHEMA or saved["identity"]!=identity or saved["smoke"] or not saved["update_boundary"]:raise ValueError("Own resume identity differs; smoke cannot warm-start")
        model.load_state_dict(saved["model"],strict=True);optimizer.load_state_dict(saved["optimizer"]);scaler.load_state_dict(saved["scaler"]);ema.load_state_dict(saved["ema"])
        sampler.load_state_dict(saved["sampler"]);updates=saved["updates"];history=saved["history"];last_candidate=saved["candidate_sha256"]
        continuation=saved.get("continuation_authorization")
        restore_rng(saved["rng"]);del saved
        if updates["successful"]>=recipe["gate_updates"]:
            continuation=authorize_continuation(cfg["arm"],last_candidate,continuation,continue_after_gate)
    stop=dict(requested=False);previous_signal=signal.signal(signal.SIGINT,lambda *_:stop.update(requested=True))
    torch.cuda.reset_peak_memory_stats()
    lock=out/"trainer.lock"
    if lock.exists():
        old=read(lock)
        try:alive=psutil.Process(old["pid"]).is_running() and abs(psutil.Process(old["pid"]).create_time()-old["created"])<.01
        except psutil.NoSuchProcess:alive=False
        if alive:raise RuntimeError("A live trainer owns this run")
    atomic_json(lock,dict(pid=os.getpid(),created=psutil.Process().create_time()))
    target_updates=max_smoke_updates if smoke else recipe["max_updates"]
    boundary=update_boundary(recipe,target_updates,continuation,stop_after_updates)
    if smoke:boundary=target_updates
    if boundary<updates["successful"]:raise ValueError("Requested boundary precedes restored updates")
    def save_boundary(phase):
        nonlocal last_candidate
        ema.verify_frozen(model)
        candidate=dict(schema="apple_od_amodal_training_v1",origin_schema=SCHEMA,
                       artifact_role="new_strong_candidate_for_revalidation",arm=cfg["arm"],model=ema.shadow,constructor_config=constructor,
                       weights_are_ema=True,ema_decay=ema.decay,updates=deepcopy(updates),parent_sha256=BASE_SHA,
                       source_sha256=sources,identity=identity,model_restore=restored,smoke=smoke,
                       inference_precision="float32",new_warmstart_lineage=True)
        last_candidate=save_torch_atomic(out/"candidate.pt",candidate)
        state=dict(schema=SCHEMA,identity=identity,smoke=smoke,update_boundary=True,phase=phase,
                   model=to_cpu(model.state_dict()),constructor_config=constructor,optimizer=to_cpu(optimizer.state_dict()),
                   scaler=scaler.state_dict(),ema=ema.state_dict(),rng=rng(),sampler=sampler.state_dict(),
                   updates=deepcopy(updates),history=history,candidate_sha256=last_candidate,
                   continuation_authorization=continuation)
        digest=save_torch_atomic(out/"last.pt",state)
        roundtrip=False
        if smoke:
            # Applying optimizer state can retain CPU scalar tensors. Do not
            # leave last.pt memory-mapped on Windows when exercising restore.
            loaded=torch.load(out/"last.pt",map_location="cpu",weights_only=False,mmap=False)
            if not equal_state(state,loaded):raise RuntimeError("Strong stress checkpoint roundtrip differs")
            # Apply the real saved state to the current model/optimizer rather
            # than only comparing serialized dictionaries. No second large
            # model is constructed and no smoke state may seed formal runs.
            model.load_state_dict(loaded["model"],strict=True)
            optimizer.load_state_dict(loaded["optimizer"])
            scaler.load_state_dict(loaded["scaler"])
            ema.load_state_dict(loaded["ema"])
            sampler.load_state_dict(loaded["sampler"])
            restore_rng(loaded["rng"])
            applied=dict(model=to_cpu(model.state_dict()),optimizer=to_cpu(optimizer.state_dict()),
                         scaler=scaler.state_dict(),ema=ema.state_dict(),rng=rng(),sampler=sampler.state_dict())
            if any(not equal_state(applied[key],loaded[key]) for key in applied):
                raise RuntimeError("Applying the real saved training state changed values")
            del applied
            del loaded;roundtrip=True
            exported=torch.load(out/"candidate.pt",map_location="cpu",weights_only=False,mmap=True)
            if not equal_state(candidate,exported):raise RuntimeError("EMA candidate serialization differs")
            del exported
        return dict(last_sha256=digest,candidate_sha256=last_candidate,own_roundtrip_verified=roundtrip,
                    own_state_reapplied_verified=roundtrip,EMA_candidate_roundtrip_verified=roundtrip)
    try:
        enter_train(model)
        while updates["successful"]<boundary and not stop["requested"] and not (out/"STOP").exists():
            begun=time.perf_counter();optimizer.zero_grad(set_to_none=True);micro=[]
            before_parameter={}
            if smoke:
                for branch,module in {"rpn":model.rpn.head,"classifier":model.roi_heads.box_predictor.cls_score,
                    "box_regression":model.roi_heads.box_predictor.bbox_pred,
                    "visible":model.roi_heads.od_mask_head.visible_predictor,
                    "hidden":model.roi_heads.od_mask_head.hidden_predictor,
                    "amodal":model.roi_heads.od_mask_head.amodal_predictor}.items():
                    parameter=next(p for p in module.parameters() if p.requires_grad)
                    before_parameter[branch]=(parameter,parameter.detach().clone())
            for micro_index in range(recipe["accumulate"]):
                index=(smoke_indices or recipe["smoke_indices"])[micro_index] if smoke else sampler.next()
                item=dataset[index]
                if item["target"]["source"]["split"]!="train":raise ValueError("Only TRAIN is allowed")
                # Two supervised orientations in both arms; no two live graphs.
                reference=None
                canonical=prepare_detection_target(item["target"],tuple(item["pixel_values"].shape[-2:]),dev)
                gt_boxes={name:canonical["boxes"][g].detach() for g,name in enumerate(canonical["instance_ids"])}
                pair_evidence=[]
                for view in ("original","hflip"):
                    view_item=item if view=="original" else flip_item_horizontal(item)
                    target=view_item["target"]
                    target={**target,"source":dict(target["source"])}
                    target["source"]["normalized_input_sha256"]=hashlib.sha256(memoryview(view_item["pixel_values"].contiguous().numpy())).hexdigest()
                    attach_ignore_boxes(target,ignore.get(target["source"]["image_id"]),tuple(view_item["pixel_values"].shape[-2:]))
                    before=target_fingerprint(target)
                    pixels=view_item["pixel_values"][None].to(dev)
                    mask=view_item["pixel_mask"][None].to(dev)
                    with capture_gt_predictions(model) as captured:
                        with torch.autocast("cuda",dtype=torch.bfloat16):
                            losses=model(pixels,pixel_mask=mask,targets=[target])
                        supervised,optimized,omitted=supervised_loss(losses,"PApple",parent_recipe["loss_weights"],stage="joint")
                        extracted=select_gt_capture(model,captured,tuple(pixels.shape[-2:]),detach=view=="original")
                    consistency={}
                    if view=="hflip" and cfg["arm"]=="C1":
                        with torch.autocast("cuda",enabled=False):
                            consistency=gt_anchored_consistency(extracted,reference,gt_boxes,image_hw=tuple(pixels.shape[-2:]),moving_flipped=True)
                    extra=sum(consistency.values()) if consistency else supervised.new_zeros(())
                    consistency_gradients={}
                    if smoke and consistency:
                        branches=("boxes","visible_logits","amodal_logits")
                        edges=torch.autograd.grad(extra,tuple(extracted[k] for k in branches),retain_graph=True,allow_unused=True)
                        consistency_gradients={k:float(g.abs().sum()) if g is not None else 0. for k,g in zip(branches,edges)}
                    total=(supervised+recipe["consistency_weight"]*extra)/(2*recipe["accumulate"])
                    if not bool(torch.isfinite(total)):raise RuntimeError("Nonfinite strong paired objective")
                    total.backward()
                    if target_fingerprint(target)!=before:raise RuntimeError("Original GT changed during training")
                    if view=="original":reference=extracted
                    pair_evidence.append(dict(view=view,GT_count=len(target["labels"]),GT_unchanged=True,
                        sampled_GT_ids=list(extracted["instance_ids"]),losses={k:float(v.detach()) for k,v in losses.items()},
                        common_consistency_GT_ids=sorted(set(extracted["instance_ids"]) & set(reference["instance_ids"])) if reference is not None else [],
                        consistency={k:float(v.detach()) for k,v in consistency.items()},
                        consistency_output_gradient_abs_sum=consistency_gradients,objective=float(total.detach())))
                    del losses,supervised,optimized,extra,total,captured,extracted,pixels,mask
                micro.append(dict(dataset_index=index,source=deepcopy(item["target"]["source"]),views=pair_evidence))
                del item,reference,canonical,gt_boxes
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):raise RuntimeError("Nonfinite/missing strong gradients")
            if any(p.grad is not None for p in model.backbone.parameters()):raise RuntimeError("Frozen backbone acquired gradient")
            norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],recipe["gradient_clip_norm"],error_if_nonfinite=True))
            gradient_evidence={name:float(p.grad.abs().sum()) if p.grad is not None else 0. for name,(p,_old) in before_parameter.items()}
            factor=lr_factor(updates["successful"],recipe["max_updates"],recipe["warmup_updates"])
            for group in optimizer.param_groups:group["lr"]=group["base_lr"]*factor
            optimizer.step();updates["attempted"]+=1;updates["successful"]+=1
            changed={name:float((p.detach()-old).abs().max()) for name,(p,old) in before_parameter.items()}
            if smoke and (not changed or not all(v>0 for v in changed.values()) or not all(v>0 for v in gradient_evidence.values())):raise RuntimeError("Stress did not update active RPN/class/box/V/H/A branches")
            ema.update(model);optimizer.zero_grad(set_to_none=True)
            row=dict(updates=deepcopy(updates),finite_gradients=True,frozen_backbone_gradients_absent=True,
                     unclipped_gradient_norm=norm,parameter_changes=changed,branch_gradient_abs_sum=gradient_evidence,micro_batches=micro,
                     seconds_for_update=time.perf_counter()-begun,
                     cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                     available_system_memory_bytes=psutil.virtual_memory().available,
                     process_memory=psutil.Process().memory_info()._asdict(),learning_rates={g["name"]:g["lr"] for g in optimizer.param_groups})
            history.append(row)
            if updates["successful"]%recipe["checkpoint_every"]==0:save_boundary("training")
            atomic_json(out/"training_status.json",dict(phase="training",arm=cfg["arm"],smoke=smoke,updates=updates,last_update=row,identity=identity))
            print(f"STRONG {cfg['arm']} successful={updates['successful']}/{recipe['max_updates']} seconds={row['seconds_for_update']:.2f}",flush=True)
        phase=("smoke_complete" if smoke else "complete") if updates["successful"]==target_updates else ("awaiting_evaluation" if updates["successful"]==boundary else "interrupted")
        checkpoint=save_boundary(phase)
        if any(sha(root/path)!=digest for path,digest in sources.items()):raise RuntimeError("Consumed strong source changed")
        summary=dict(schema=SCHEMA,phase=phase,arm=cfg["arm"],smoke=smoke,identity=identity,plan=recipe,updates=updates,
                     checkpoint=checkpoint,history=history,source_before_after_identical=True,full_validation_executed=False,
                     weak_model_used=False,backbone_FPN_frozen=True,small_model_loaded=False,
                     candidate_weight_kind="EMA",formal_resume_supported=not smoke,
                     continuation_authorization=continuation)
        atomic_json(out/"training_summary.json",summary);atomic_json(out/"training_status.json",summary)
        return summary
    except BaseException as error:
        atomic_json(out/"training_status.json",dict(phase="failed",arm=cfg["arm"],updates=updates,error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        signal.signal(signal.SIGINT,previous_signal)
        if lock.exists():lock.unlink()
