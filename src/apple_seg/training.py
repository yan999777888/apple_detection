"""PApple-only or mixed training at complete four-image update boundaries."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

SCHEMA = "apple_seg_mixed_update_boundary_v1"
PAPPLE_SCHEMA = "apple_seg_papple_s0_update_boundary_v1"
SOURCE_ORDER = ("PApple", "MinneApple", "PApple", "MinneApple")
SOURCE_ORDERS = {"mixed": SOURCE_ORDER, "papple_only": ("PApple",) * 4}
DEFAULTS = dict(input_size=768, max_updates=600, accumulate=4, validate_every=200,
                backbone_lr=1e-6, head_lr=5e-6, warmup_updates=50, workers=0,
                seed=20261022, weight_decay=.05, gradient_clip_norm=.1,
                checkpoint_every=50, cpu_threads=4, data_mode="mixed", shuffle=True,
                model_selection="balanced_min_pr", smoke_sample_indices=None)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_config(path, *, input_size=None, max_updates=None, smoke=False, output=None):
    path = Path(path).expanduser().absolute()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    base_value = raw.get("path_base", raw.get("root"))
    if not isinstance(base_value, str) or not base_value:
        raise ValueError("Config must explicitly set path_base (relative to its own directory) or root")
    base = (path.parent / base_value).absolute()
    def locate(value):
        return (base / Path(value).expanduser()).absolute()
    recipe = {**DEFAULTS, **raw.get("training", {})}
    if recipe["data_mode"] not in SOURCE_ORDERS:
        raise ValueError("data_mode must be mixed or papple_only")
    required = ("papple_root", "parent_checkpoint", "papple_provenance")
    if recipe["data_mode"] == "mixed":
        required += ("minne_root", "minne_manifest")
    paths = {key: locate(raw["paths"][key]) for key in required}
    if type(recipe["shuffle"]) is not bool:
        raise ValueError("shuffle must be an explicit boolean")
    if type(recipe["seed"]) is not int or not 0 <= recipe["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if recipe["model_selection"] not in ("balanced_min_pr", "precision_at_recall90"):
        raise ValueError("Unknown model_selection rule")
    fixed = recipe["smoke_sample_indices"]
    if fixed is not None:
        if not smoke or recipe["data_mode"] != "papple_only":
            raise ValueError("smoke_sample_indices are only allowed in PApple-only smoke; never formal training")
        if not isinstance(fixed, list) or len(fixed) != 4 or any(type(i) is not int or i < 0 for i in fixed):
            raise ValueError("smoke_sample_indices must contain four nonnegative TRAIN indices")
    if input_size is not None:
        recipe["input_size"] = input_size
    if max_updates is not None:
        recipe["max_updates"] = max_updates
    elif smoke:
        recipe["max_updates"] = 1
    for key in ("input_size", "max_updates", "validate_every", "warmup_updates", "checkpoint_every", "cpu_threads"):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ValueError(f"Positive integer required: {key}")
    if recipe["accumulate"] != 4 or recipe["workers"] != 0:
        raise ValueError("This recipe requires batch1, accumulation4, workers0 and pin_memory=False")
    for key in ("backbone_lr", "head_lr", "weight_decay", "gradient_clip_norm"):
        if not isinstance(recipe[key], (int, float)) or not math.isfinite(recipe[key]) or recipe[key] <= 0:
            raise ValueError(f"Positive finite recipe value required: {key}")
    recipe["max_attempts"] = recipe.get("max_attempts", 10 * recipe["max_updates"] + 100)
    if type(recipe["max_attempts"]) is not int or recipe["max_attempts"] < recipe["max_updates"]:
        raise ValueError("max_attempts cannot be smaller than successful-update target")
    return {"path": path, "base": base, "raw": raw, "paths": paths, "recipe": recipe,
            "source_order": SOURCE_ORDERS[recipe["data_mode"]],
            "checkpoint_schema": PAPPLE_SCHEMA if recipe["data_mode"] == "papple_only" else SCHEMA,
            "output": locate(output or raw.get("paths", {}).get("output") or raw.get("output", "runs/mixed_pilot01")), "smoke": bool(smoke)}


class CyclingOrder:
    """One source's shuffled order, cursor and RNG; cycles are not training epochs."""
    def __init__(self, length, seed, *, shuffle=True):
        if length < 1:
            raise ValueError("Training sources cannot be empty")
        self.rng = random.Random(seed)
        self.shuffle = shuffle
        self.order = list(range(length))
        if self.shuffle:
            self.rng.shuffle(self.order)
        self.cursor, self.cycles = 0, 0

    def next(self):
        if self.cursor == len(self.order):
            if self.shuffle:
                self.rng.shuffle(self.order)
            self.cursor = 0
            self.cycles += 1
        value = self.order[self.cursor]
        self.cursor += 1
        return value

    def state_dict(self):
        return {"order": list(self.order), "cursor": self.cursor, "cycles": self.cycles,
                "rng": self.rng.getstate(), "shuffle": self.shuffle}

    def load_state_dict(self, state):
        if state.get("shuffle", True) != self.shuffle:
            raise ValueError("Sampler shuffle policy differs")
        if sorted(state["order"]) != list(range(len(self.order))) or not 0 <= state["cursor"] <= len(self.order):
            raise ValueError("Sampler inventory or cursor differs")
        self.order, self.cursor, self.cycles = list(state["order"]), state["cursor"], state["cycles"]
        self.rng.setstate(state["rng"])


def lr_factor(successful_updates, maximum, warmup):
    ordinal = successful_updates + 1
    if ordinal <= warmup:
        return ordinal / warmup
    progress = min(1., (ordinal - warmup) / max(1, maximum - warmup))
    return .05 + .95 * .5 * (1 + math.cos(math.pi * progress))


def target_fingerprint(target):
    evidence = {"instance_ids": list(target["instance_ids"]),
                "annotation_validity": dict(target["annotation_validity"])}
    for key in ("labels", "visible_masks", "amodal_masks", "boxes"):
        value = target.get(key)
        if value is None:
            evidence[key] = None
        else:
            if value.device.type != "cpu" or value.requires_grad:
                raise ValueError("Original GT tensors must stay on CPU and fixed")
            array = value.detach().contiguous().numpy()
            evidence[key] = {"shape": list(array.shape), "dtype": str(array.dtype),
                             "sha256": hashlib.sha256(memoryview(array)).hexdigest()}
    return evidence


def _rng_state(torch, np, generator, cuda):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "criterion": generator.get_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if cuda else None}


def _restore_rng(state, torch, np, generator, cuda):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    generator.set_state(state["criterion"])
    if cuda:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _same(left, right, torch, np):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same(left[k], right[k], torch, np) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_same(a, b, torch, np) for a, b in zip(left, right))
    return left == right


def _datasets(paths, size, data_mode="mixed"):
    from .data import PAppleDataset
    result = {"PApple": PAppleDataset(paths["papple_root"], split="train", input_size=size)}
    if data_mode == "mixed":
        from .data import MinneAppleDataset
        result["MinneApple"] = MinneAppleDataset(paths["minne_root"], paths["minne_manifest"],
                                               split="train", input_size=size, tiled=True)
    elif data_mode != "papple_only":
        raise ValueError("Unknown training data mode")
    return result


def _papple_capacity(dataset):
    """Inspect every original instance ID without decoding or dropping masks."""
    counts = [len(sample["instance_ids"]) for sample in dataset.samples]
    if len(counts) != len(dataset):
        raise ValueError("PApple inventory and loader length differ")
    over = [index for index, count in enumerate(counts) if count > 100]
    return {"images": len(counts), "GT_instances": sum(counts), "empty_images": counts.count(0),
            "max_instances": max(counts, default=0), "query_capacity": 100,
            "over_capacity_indices": over, "formal_training_compatible": not over,
            "GT_truncated_or_skipped": False}


def select_validation_point(evaluated, directory, rule):
    """Select BOX workpoint and replay branch mask counts at that same threshold."""
    if rule == "balanced_min_pr":
        metric = evaluated["summary"]
        return {"eligible": True, "rule": rule, "confidence": evaluated["confidence"],
                "score": evaluated["score"], "summary": metric,
                "ranking": [evaluated["score"], metric["f1"], evaluated["confidence"]]}
    if rule != "precision_at_recall90":
        raise ValueError("Unknown checkpoint selection rule")
    curve = json.loads((Path(directory) / "box_threshold_curve.json").read_text(encoding="utf-8"))
    eligible = [row for row in curve if row["recall"] >= .90]
    if not eligible:
        return {"eligible": False, "rule": rule, "confidence": None, "score": None,
                "summary": None, "ranking": None, "reason": "Full original VAL never reaches Recall >= 0.90",
                "max_recall": max((row["recall"] for row in curve), default=0.)}
    chosen = max(eligible, key=lambda row: (row["precision"], row["recall"], row["f1"], row["confidence"]))
    metric = {**chosen, "images": evaluated["summary"]["images"],
              "empty_gt_images": evaluated["summary"]["empty_gt_images"]}
    from . import evaluate as evaluation
    if sha(evaluation.METRICS_PATH) != evaluation.REFERENCE_PINS[evaluation.METRICS_PATH]:
        raise ValueError("Frozen metric reference changed before alternate-threshold mask replay")
    metrics = evaluation._module(evaluation.METRICS_PATH, "_apple_s0_selection_reference_metrics")
    diagnostics_path = Path(directory) / "mask_pair_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    for branch in ("visible", "amodal"):
        metric[branch + "_instance_mask_pr_iou50"] = evaluation._mask_pr(
            diagnostics, branch + "_mask", chosen["confidence"], chosen["gt_count"], metrics)
    return {"eligible": True, "rule": rule, "confidence": chosen["confidence"],
            "score": chosen["precision"], "summary": metric,
            "ranking": [chosen["precision"], chosen["recall"], chosen["f1"], chosen["confidence"]],
            "mask_metric_scope": "V/A independently matched at mask IoU>=0.5; counts replayed at this selected BOX confidence from complete diagnostics",
            "mask_diagnostics_sha256": sha(diagnostics_path),
            "selection_scope": "Complete exact distinct-score development VAL curve; no TEST tuning"}


def criterion_in_fp32(criterion, outputs, targets, *, generator=None):
    """Matching/point losses stay FP32 even when model forward uses CUDA FP16.

    Casting retains autograd to the original readouts. Disabling autocast here
    also protects matrix products in pairwise BCE/Dice costs from FP16 sums.
    """
    import torch
    device_type = outputs["class_queries_logits"].device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return criterion({key: value.float() for key, value in outputs.items()}, targets, generator=generator)


def run_training(config_path, *, device="cuda", smoke=False, max_updates=None,
                 input_size=None, output=None, resume=None):
    """Run real source samples; Ctrl+C requests a stop after the current update."""
    cfg = load_config(config_path, input_size=input_size, max_updates=max_updates, smoke=smoke, output=output)
    import numpy as np
    import torch

    dev = torch.device(device)
    if dev.type not in ("cpu", "cuda"):
        raise ValueError("Only cpu/cuda devices are supported")
    if dev.type == "cpu" and torch.cuda.is_initialized():
        raise RuntimeError("CPU run requires CUDA to remain uninitialized")
    recipe, paths, out = cfg["recipe"], cfg["paths"], cfg["output"]
    source_order, checkpoint_schema = cfg["source_order"], cfg["checkpoint_schema"]
    if out.exists() and any(p.name != "runtime.log" for p in out.iterdir()) and resume is None:
        raise FileExistsError("Use a fresh output directory or --resume")
    out.mkdir(parents=True, exist_ok=True)
    if dev.type == "cuda" and not torch.cuda.is_available():
        message = "CUDA device is unavailable; no dataset or model was constructed"
        atomic_json(out / "training_status.json", {"phase": "failed", "failure_type": "CUDAUnavailable",
                    "failure": message, "updates": {"attempted": 0, "successful": 0, "skipped": 0}})
        raise RuntimeError(message)
    torch.set_num_threads(recipe["cpu_threads"])
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(recipe["seed"])
    source_files = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("losses.py"),
                    Path(__file__).with_name("data.py"), Path(__file__).with_name("evaluate.py")]
    sources = {p.name: sha(p) for p in source_files}
    datasets = _datasets(paths, recipe["input_size"], recipe["data_mode"])
    datasets_metadata = {name: ds.metadata() if hasattr(ds, "metadata") else {} for name, ds in datasets.items()}
    datasets_metadata["PApple"]["query_capacity_audit"] = _papple_capacity(datasets["PApple"])
    if recipe["smoke_sample_indices"] is not None and max(recipe["smoke_sample_indices"]) >= len(datasets["PApple"]):
        raise ValueError("A fixed smoke TRAIN index is outside the PApple inventory")
    for name, metadata in datasets_metadata.items():
        capacity = metadata.get("query_capacity_audit", {})
        if not smoke and (capacity.get("formal_training_compatible") is False or capacity.get("blocked") is True):
            message = f"Dataset {name} exceeds100-query capacity; no GT truncation or silent skip is allowed"
            atomic_json(out / "training_status.json", {"phase": "failed", "failure_type": "QueryCapacity",
                        "failure": message, "query_capacity_audit": capacity})
            raise ValueError(message)
    identity = {"recipe": recipe, "source_order": list(source_order), "smoke": smoke,
                "data_mode": recipe["data_mode"],
                "parent_sha256": cfg["raw"]["parent_sha256"], "source_sha256": sources,
                "papple_provenance_sha256": sha(paths["papple_provenance"]),
                "datasets": {name: {"items": len(ds), "annotation_sha256": getattr(ds, "annotation_sha256", None)} for name, ds in datasets.items()}}
    if recipe["data_mode"] == "mixed":
        identity["minne_manifest_sha256"] = sha(paths["minne_manifest"])
    else:
        identity["experiment_arm"] = cfg["raw"].get("experiment_arm", "S0_GT_only")
        identity["runtime"] = {"device_type": dev.type, "model_amp_dtype": "float16" if dev.type == "cuda" else "float32",
                               "criterion_dtype": "float32", "grad_scaler_enabled": dev.type == "cuda",
                               "torch_version": torch.__version__, "cuda_build": torch.version.cuda}
    atomic_json(out / "training_status.json", {"phase": "initializing", "identity": identity})
    from .model import from_checkpoint
    from .losses import build_annotation_aware_criterion
    model, parent = from_checkpoint(paths["parent_checkpoint"], cfg["raw"]["parent_sha256"])
    model.to(dev).train()
    criterion = build_annotation_aware_criterion().to(dev)
    encoder = {id(p) for p in model.shared.pixel_level_module.encoder.parameters()}
    groups = [[p for p in model.parameters() if (id(p) in encoder) == yes] for yes in (True, False)]
    optimizer = torch.optim.AdamW([
        {"params": groups[0], "lr": recipe["backbone_lr"], "base_lr": recipe["backbone_lr"], "name": "backbone"},
        {"params": groups[1], "lr": recipe["head_lr"], "base_lr": recipe["head_lr"], "name": "other"}],
        weight_decay=recipe["weight_decay"], foreach=False)
    scaler = torch.amp.GradScaler(dev.type, init_scale=128., enabled=dev.type == "cuda")
    generator = torch.Generator(device=dev).manual_seed(recipe["seed"])
    samplers = {name: CyclingOrder(len(ds), recipe["seed"] + n, shuffle=recipe["shuffle"])
                for n, (name, ds) in enumerate(datasets.items())}
    updates = {"attempted": 0, "successful": 0, "skipped": 0}
    history, validations, best = [], [], None
    if resume is not None:
        checkpoint_path = out / "last.pt" if resume is True else Path(resume).expanduser().absolute()
        expected = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256").read_text().strip()
        if sha(checkpoint_path) != expected:
            raise ValueError("Own resume checkpoint SHA256 differs")
        # Own optimizer state must not retain a file mapping of last.pt: on
        # Windows that mapping can prevent atomic replacement at the next save.
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=False)
        if saved["schema"] != checkpoint_schema or saved["identity"] != identity or not saved["update_boundary"] or saved["checkpoint_kind"] != "formal_update_boundary":
            raise ValueError("Resume requires this exact formal data plan/runtime at a complete update boundary; smoke is not a warm start")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        for name, sampler in samplers.items():
            sampler.load_state_dict(saved["samplers"][name])
        updates, history, validations, best = saved["updates"], saved["history"], saved["validations"], saved["best"]
        _restore_rng(saved["rng"], torch, np, generator, dev.type == "cuda")
        del saved
    step_calls = {"count": 0}
    hook = optimizer.register_step_post_hook(lambda *_: step_calls.__setitem__("count", step_calls["count"] + 1))
    stop = {"requested": False}
    old_handler = signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("requested", True))
    last_entry = None

    def save_boundary(phase, is_best=False):
        payload = {"schema": checkpoint_schema, "checkpoint_kind": "smoke_partial_not_formal_resume" if smoke else "formal_update_boundary",
                   "formal_training_resumable": not smoke, "update_boundary": True, "phase": phase,
                   "identity": identity, "model_config": model.model_config, "constructed_model_config": model.constructed_model_config,
                   "parent_provenance": parent, "datasets_metadata": datasets_metadata,
                   "criterion_config": criterion.configuration, "criterion_math_dtype": "float32",
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scaler": scaler.state_dict(), "rng": _rng_state(torch, np, generator, dev.type == "cuda"),
                   "samplers": {name: s.state_dict() for name, s in samplers.items()}, "updates": dict(updates),
                   "history": history, "validations": validations, "best": best}
        tmp = out / f"last.pt.tmp.{os.getpid()}.{time.time_ns()}"
        with tmp.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, out / "last.pt")
        digest = sha(out / "last.pt")
        (out / "last.pt.sha256").write_text(digest, encoding="ascii")
        if is_best:
            tmp_best = out / f"best.pt.tmp.{time.time_ns()}"
            try:
                os.link(out / "last.pt", tmp_best)
            except OSError:
                import shutil
                shutil.copyfile(out / "last.pt", tmp_best)
            os.replace(tmp_best, out / "best.pt")
            (out / "best.pt.sha256").write_text(digest, encoding="ascii")
        if smoke:
            restored = torch.load(out / "last.pt", map_location="cpu", weights_only=False, mmap=True)
            if not _same(payload, restored, torch, np):
                raise RuntimeError("Own smoke checkpoint model/optimizer/scaler/RNG/sampler roundtrip differs")
            del restored
        return {"path": "last.pt", "sha256": digest, "successful_updates": updates["successful"],
                "own_roundtrip_verified": smoke}

    try:
        while updates["successful"] < recipe["max_updates"]:
            if stop["requested"] or (out / "STOP").exists():
                break
            if updates["attempted"] >= recipe["max_attempts"]:
                raise RuntimeError("Attempt limit reached before requested successful updates")
            optimizer.zero_grad(set_to_none=True)
            update_started = time.perf_counter()
            micro = []
            for micro_index, source_name in enumerate(source_order):
                index = (recipe["smoke_sample_indices"][micro_index]
                         if recipe["smoke_sample_indices"] is not None else samplers[source_name].next())
                item = datasets[source_name][index]
                target = item["target"]
                if target["source"]["dataset"] != source_name or target["source"]["split"] != "train":
                    raise ValueError("Sampler returned the wrong source/split")
                before = target_fingerprint(target)
                pixels = item["pixel_values"].unsqueeze(0).to(dev)
                mask = item.get("pixel_mask")
                mask = mask.unsqueeze(0).to(dev) if mask is not None else None
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    outputs = model(pixels, pixel_mask=mask)
                losses = criterion_in_fp32(criterion, outputs, [target], generator=generator)
                total = sum(losses.values())
                if not bool(torch.isfinite(total)):
                    raise RuntimeError("Nonfinite real training loss")
                unknown = not target["annotation_validity"]["amodal"]
                absent = None
                if unknown:
                    if target.get("amodal_masks") is not None or any("amodal" in key for key in losses):
                        raise ValueError("Unknown MinneApple A must have neither fake GT nor A loss")
                    if smoke:
                        grads = torch.autograd.grad(total, tuple(model.amodal_mask_embedder.parameters()), allow_unused=True, retain_graph=True)
                        absent = all(grad is None for grad in grads)
                        if not absent:
                            raise RuntimeError("Unknown real A annotation acquired a gradient edge")
                scaler.scale(total / len(source_order)).backward()
                after = target_fingerprint(target)
                if before != after:
                    raise RuntimeError("Original real GT changed during backward")
                micro.append({"source": deepcopy(target["source"]), "dataset_index": index,
                              "GT_count": len(target["labels"]), "GT_fingerprint": before, "GT_unchanged": True,
                              "losses": {key: float(value.detach()) for key, value in losses.items()},
                              "unknown_A_gradient_absent_verified": absent})
                del item, target, outputs, losses, total, pixels, mask
            factor = lr_factor(updates["successful"], recipe["max_updates"], recipe["warmup_updates"])
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            scaler.unscale_(optimizer)
            finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            if not finite and dev.type == "cpu":
                raise RuntimeError("CPU update has nonfinite gradients; refusing optimizer step")
            gradient_evidence, before_parameters = {}, {}
            if smoke:
                readouts = {"amodal": model.amodal_mask_embedder,
                            "visible": model.shared.transformer_module.decoder.mask_predictor.mask_embedder,
                            "classifier": model.class_predictor,
                            "shared_encoder": model.shared.pixel_level_module.encoder}
                for name, module in readouts.items():
                    gradients = [p.grad for p in module.parameters() if p.grad is not None]
                    gradient_evidence[name] = sum(float(g.abs().sum()) for g in gradients) if finite else None
                    parameter = next(p for p in module.parameters() if p.grad is not None)
                    before_parameters[name] = (parameter, parameter.detach().clone())
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip_norm"], error_if_nonfinite=True)) if finite else None
            previous = step_calls["count"]
            scaler.step(optimizer)
            scaler.update()
            executed = step_calls["count"] - previous
            if executed not in (0, 1) or executed and not finite:
                raise RuntimeError("GradScaler execution and finite-gradient evidence disagree")
            updates["attempted"] += 1
            updates["successful"] += executed
            updates["skipped"] += 1 - executed
            parameter_changes = {name: float((p.detach() - old).abs().max()) for name, (p, old) in before_parameters.items()}
            if smoke and executed and (not all(value > 0 for value in parameter_changes.values())
                                       or not all(value > 0 for value in gradient_evidence.values())):
                raise RuntimeError("Real smoke must update A/V/class/shared parameters with nonzero finite gradients")
            optimizer.zero_grad(set_to_none=True)
            row = {"updates": dict(updates), "optimizer_step_executed": bool(executed), "lr_factor": factor,
                   "learning_rates": {g["name"]: g["lr"] for g in optimizer.param_groups},
                   "scaler_scale": scaler.get_scale(), "unclipped_gradient_norm": norm,
                   "finite_gradients": finite, "smoke_readout_gradient_abs_sum": gradient_evidence,
                   "smoke_parameter_max_abs_change": parameter_changes, "micro_batches": micro,
                   "seconds_for_update": time.perf_counter() - update_started,
                   "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
                   "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(dev) if dev.type == "cuda" else None}
            import psutil
            row["process_memory"] = psutil.Process().memory_info()._asdict()
            row["available_system_memory_bytes"] = psutil.virtual_memory().available
            history.append(row)
            is_best = False
            if executed and not smoke and (updates["successful"] % recipe["validate_every"] == 0
                                          or updates["successful"] == recipe["max_updates"]):
                from .data import PAppleDataset
                from .evaluate import evaluate_papple
                val = PAppleDataset(paths["papple_root"], split="val", input_size=recipe["input_size"])
                val_dir = out / "validation" / f"update_{updates['successful']:06d}_{time.time_ns()}"
                evaluated = evaluate_papple(model, val, dev, val_dir, paths["papple_provenance"])
                metric = evaluated["summary"]
                if evaluated["phase"] != "complete" or not evaluated["model_selection_eligible"] or (metric["images"], metric["gt_count"], metric["empty_gt_images"]) != (814, 3449, 13):
                    raise RuntimeError("Only complete original full PApple VAL can select best")
                selection = select_validation_point(evaluated, val_dir, recipe["model_selection"])
                point = {"successful_updates": updates["successful"], "directory": str(val_dir.relative_to(out)),
                         **selection, "balanced_evaluation": {"confidence": evaluated["confidence"],
                         "score": evaluated["score"], "summary": metric,
                         "mask_metric_scope": "V/A metrics apply only at this original balanced BOX confidence"}}
                atomic_json(val_dir / "checkpoint_selection.json", point)
                validations.append(point)
                is_best = point["eligible"] and (best is None or tuple(point["ranking"]) > tuple(best["ranking"]))
                if is_best:
                    best = point
                del val
            if is_best or updates["successful"] % recipe["checkpoint_every"] == 0:
                last_entry = save_boundary("running", is_best)
            atomic_json(out / "training_status.json", {"phase": "training", "smoke": smoke, "plan": recipe,
                        "updates": dict(updates), **updates, "last_update": row, "best": best})
            print(f"{recipe['data_mode'].upper()} successful={updates['successful']}/{recipe['max_updates']} attempted={updates['attempted']} skipped={updates['skipped']} lr_factor={factor:.6f}", flush=True)
        completed = updates["successful"] == recipe["max_updates"]
        phase = ("smoke_complete" if smoke else "complete") if completed else "interrupted"
        last_entry = save_boundary(phase)
        after = {p.name: sha(p) for p in source_files}
        if sources != after:
            raise RuntimeError("Consumed training/model/data/loss/evaluation source changed")
        if dev.type == "cpu" and torch.cuda.is_initialized():
            raise RuntimeError("CPU execution initialized CUDA")
        summary = {"schema": checkpoint_schema, "phase": phase, "smoke": smoke, "plan": recipe, "updates": updates,
                   "identity": identity, "config_sha256": sha(cfg["path"]),
                   "data_mode": recipe["data_mode"], "source_order": list(source_order),
                   "requested_successful_updates_completed": completed, "default600_training_complete": completed and not smoke and recipe["max_updates"] == 600,
                   "formal_training_resumable": not smoke, "checkpoint": last_entry, "best": best,
                   "criterion_math_dtype": "float32", "datasets_metadata": datasets_metadata,
                   "best_checkpoint": {"path": "best.pt", "sha256": sha(out / "best.pt"),
                                       "successful_updates": best["successful_updates"]} if best is not None else None,
                   "validation_history": validations, "update_history": history,
                   "source_sha256_before": sources, "source_sha256_after": after, "source_before_after_identical": True,
                   "cuda_initialized": torch.cuda.is_initialized(), "full_validation_executed": bool(validations),
                   "accuracy_claimed": bool(validations), "output_directory": str(out)}
        atomic_json(out / "training_summary.json", summary)
        atomic_json(out / "training_status.json", summary)
        return summary
    except BaseException as error:
        atomic_json(out / "training_status.json", {"phase": "failed", "updates": updates,
                    "failure_type": type(error).__name__, "failure": str(error),
                    "last_committed_checkpoint": last_entry})
        raise
    finally:
        hook.remove()
        signal.signal(signal.SIGINT, old_handler)
