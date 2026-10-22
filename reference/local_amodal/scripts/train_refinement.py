"""Paired-resolution refinement from a completed local baseline's best.pt.

Warm start copies model weights only, with a fresh optimizer and schedule.
Per-image flips and data order are matched across resolution arms. Exact resume
is separate and restores only this refinement run's last.pt. A zero-update
validation is always retained as a candidate, so training cannot silently replace
a better initialization. The official test split is never opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone

from atomic_io import replace_with_retry
from refinement_config import (MatchedFlipDataset, sha256_file,
                               validate_hyperparameters, validate_initial_checkpoint)

ALLOWED_HOST = "YAN"
OLD_PROJECT_LOCK = Path(__file__).resolve().parents[2] / "cache" / "training.lock"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def remote_guard():
    # Compatibility name used by the evaluator. Local execution was explicitly
    # authorized later by the user; enforce the inspected host, GPU and F: paths.
    from runtime import authorized_guard
    return authorized_guard()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, path)


def atomic_checkpoint(torch, path, checkpoint):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        torch.save(checkpoint, stream)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, path)


def _live_lock(path, psutil):
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        pid = int(payload.get("pid", payload.get("process_id", 0)))
        if pid < 1:
            raise ValueError("missing positive PID")
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Cannot safely interpret existing lock {path}: {exc}") from exc
    if not psutil.pid_exists(pid):
        return False
    try:
        process = psutil.Process(pid)
        saved_start = payload.get("process_create_time")
        if saved_start is not None and abs(process.create_time() - float(saved_start)) > 0.5:
            return False
        return process.is_running()
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True


class TrainingLock:
    def __init__(self, path, output, psutil):
        self.path, self.output, self.psutil = Path(path), str(output), psutil
        self.acquired = False
        self.old_handle = None

    def __enter__(self):
        # The existing MinneApple supervisor uses an OS byte lock, NOT a JSON
        # PID file. Acquire that exact byte and hold it for this entire run so
        # an old-project job cannot start between inspection and training.
        import msvcrt
        if OLD_PROJECT_LOCK.parent.is_dir():
            self.old_handle = OLD_PROJECT_LOCK.open("a+b")
            try:
                self.old_handle.seek(0, 2)
                if self.old_handle.tell() == 0:
                    self.old_handle.write(b"0")
                    self.old_handle.flush()
                self.old_handle.seek(0)
                msvcrt.locking(self.old_handle.fileno(), msvcrt.LK_NBLCK, 1)
            except BaseException:
                self.old_handle.close()
                self.old_handle = None
                raise RuntimeError(f"Previous project has an active training/install lock: {OLD_PROJECT_LOCK}")
        try:
            return self._acquire_own()
        except BaseException:
            self._release_old()
            raise

    def _acquire_own(self):
        if self.path.exists():
            if _live_lock(self.path, self.psutil):
                raise RuntimeError(f"Another amodal process owns {self.path}")
            backup = self.path.with_name(self.path.name + f".stale.{time.time_ns()}")
            os.replace(self.path, backup)
        payload = {"pid": os.getpid(), "host": platform.node(), "output": self.output,
                   "created_utc": utc_now(),
                   "process_create_time": self.psutil.Process().create_time()}
        # Exclusive creation closes the race between two new jobs.
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream)
        self.acquired = True
        return self

    def _release_old(self):
        if self.old_handle is not None:
            import msvcrt
            try:
                self.old_handle.seek(0)
                msvcrt.locking(self.old_handle.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self.old_handle.close()
                self.old_handle = None

    def __exit__(self, exc_type, exc, tb):
        if self.acquired:
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                if saved.get("pid") == os.getpid():
                    self.path.unlink()
            except (OSError, ValueError):
                pass
        self._release_old()


def dataset_signature(datasets):
    """Hash train/val image bytes and annotation bytes; never touches test."""
    digest = hashlib.sha256()
    for dataset in datasets:
        digest.update(dataset.split.encode("utf-8"))
        for name in ("via_region_data_instance.json", "via_region_data_amodal.json"):
            path = dataset.root / "gt_json" / dataset.split / name
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
        for sample in dataset.samples:
            digest.update(sample["filename"].encode("utf-8"))
            with sample["path"].open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _with_target_cache(train_base, val_base, cache):
    """Optional lossless storage layer; augmentation/RNG stay in their owner.

    Validate every compressed target at startup/resume, before building a model
    or restoring any optimizer. Per-sample validation remains active afterward.
    No arrays are retained by this pass and no source files are modified.
    """
    if cache is None:
        return train_base, val_base, None
    from target_cache import CachedTargetDataset
    cache = Path(cache).resolve()
    manifest_path = cache / 'manifest.json'
    module_path = Path(__file__).parent / 'target_cache.py'
    manifest_hash = sha256_file(manifest_path)
    module_hash = sha256_file(module_path)
    datasets = [CachedTargetDataset(base, cache) for base in (train_base, val_base)]
    verified = {}
    for dataset in datasets:
        if dataset.manifest_sha256 != manifest_hash:
            raise ValueError('Target-cache manifest changed while opening both splits')
        count = 0
        for sample in dataset.samples:
            entry = dataset.entries[sample['filename']]
            path = (cache / entry['file']).resolve()
            if (not path.is_relative_to(cache) or path.suffix != '.npz'
                    or sha256_file(path) != entry['sha256']):
                raise ValueError(f'Target-cache entry changed before training: {dataset.split}/{sample["filename"]}')
            count += 1
        verified[dataset.split] = count
    if sha256_file(manifest_path) != manifest_hash or sha256_file(module_path) != module_hash:
        raise ValueError('Target-cache manifest or reader changed during startup verification')
    provenance = {'root': str(cache), 'manifest': str(manifest_path),
                  'manifest_sha256': manifest_hash, 'reader_sha256': module_hash,
                  'schema': datasets[0].manifest['schema'],
                  'verified_images_by_split': verified,
                  'policy': 'bit_exact_targets_original_RGB_outer_matched_flip_v1'}
    return datasets[0], datasets[1], provenance


def _collate(batch):
    return tuple(zip(*batch))


def _rng_state(torch, np, loader_generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
            "loader_generator": loader_generator.get_state()}


def _restore_rng(torch, np, generator, state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["loader_generator"])


def _lr_factor(update, total_updates, warmup=250):
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if update < 0 or total_updates < 1:
        raise ValueError("update must be nonnegative and total_updates must be positive")
    if warmup > 0 and update < warmup:
        return (update + 1) / warmup
    progress = (update - warmup) / max(1, total_updates - warmup)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _write_history(path, history):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    keys = sorted({key for row in history for key in row})
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    replace_with_retry(temporary, path)


def _status(output, torch, psutil, **fields):
    fields.update(updated_utc=utc_now(), pid=os.getpid(), host=platform.node(),
                  process_rss_gib=psutil.Process().memory_info().rss / 2**30,
                  host_available_gib=psutil.virtual_memory().available / 2**30,
                  gpu_allocated_gib=torch.cuda.memory_allocated() / 2**30,
                  gpu_reserved_gib=torch.cuda.memory_reserved() / 2**30,
                  gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
    atomic_json(output / "status.json", fields)


def _write_quality_breakdown(validation_dir, dataset, confidence):
    """Postprocess saved validation output on CPU, sequentially after inference."""
    from quality_breakdown import build_report
    validation_dir = Path(validation_dir)
    records = json.loads((validation_dir / "raw_predictions.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((validation_dir / "mask_pair_diagnostics.json").read_text(encoding="utf-8"))
    report = build_report(records, dataset.samples, confidence, diagnostics, split=dataset.split)
    report["source"] = {"data": str(dataset.root), "evaluation_dir": str(validation_dir)}
    path = validation_dir / "quality_breakdown.json"
    atomic_json(path, report)
    print(json.dumps({"event": "occlusion_quality_complete", "path": str(path),
                      "recall_by_stratum": {name: {key: group[key] for key in ("gt_count", "tp", "fn", "recall")}
                                            for name, group in report["by_occlusion"].items()}}), flush=True)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-cache", type=Path,
                        help="Optional complete cache under project cache/targets; same version for both resolution arms")
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-updates", type=int, default=100,
                        help="Nonnegative optimizer-update warmup; 0 disables warmup")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    # Parsing, paths and the exclusive training lock precede any torch/CUDA use.
    from runtime import configure_paths
    import psutil
    root = configure_paths()
    validate_hyperparameters(args)
    output = args.output.expanduser().resolve()
    data = args.data.expanduser().resolve()
    initial_path = args.initial_checkpoint.expanduser().resolve()
    target_cache = args.target_cache.expanduser().resolve() if args.target_cache else None
    if target_cache is not None and (target_cache.drive.upper() != 'F:'
            or not target_cache.is_relative_to(root / 'cache/targets')
            or target_cache == root / 'cache/targets'):
        raise ValueError('Use a named target-cache version under this project cache/targets/')
    if not all(path.drive.upper() == 'F:' and path.is_relative_to(root)
               for path in (output, data, initial_path)):
        raise ValueError('All paths must remain inside this F: project')
    if not output.is_relative_to(root / 'runs') or output == root / 'runs':
        raise ValueError('Use a named output directory under this project runs/')
    if (not initial_path.is_relative_to(root / 'runs') or initial_path.name != 'best.pt'
            or not initial_path.is_file()):
        raise ValueError('Initial checkpoint must be an existing local runs/.../best.pt')
    if output == initial_path.parent:
        raise ValueError('Refinement cannot overwrite its source run')
    if args.resume:
        resume = args.resume.expanduser().resolve()
        if resume != output / "last.pt" or not resume.is_file():
            raise ValueError("Resume must point to this output directory's existing last.pt")
    elif output.exists():
        raise FileExistsError(f"Use a new output directory; existing results are protected: {output}")

    with TrainingLock(Path(__file__).resolve().parent / "training.lock", output, psutil):
        # Never consume a checkpoint that an active parent run can still replace.
        parent_status = json.loads((initial_path.parent / 'status.json').read_text(encoding='utf-8'))
        if parent_status.get('phase') != 'complete':
            raise ValueError('Wait for the source training run to complete')
        torch = remote_guard()
        import numpy as np
        from torch.utils.data import DataLoader
        from amodal_model import build_model, checkpoint_model_config, MODEL_SCHEMA
        from dataset import PAppleDataset
        from evaluate import evaluate_model
        if not args.resume:
            output.mkdir(parents=True, exist_ok=False)
        try:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_num_threads(4)
            train_base = PAppleDataset(data, "train", augment=False)
            val_base = PAppleDataset(data, "val", augment=False)
            train_base, val_data, cache_provenance = _with_target_cache(train_base, val_base, target_cache)
            # Resolution changes only the model transform. Both arms retain the
            # same sample order, epoch/filename flip decision and loader RNG.
            train_data = MatchedFlipDataset(train_base, args.seed)
            if not len(train_data) or not len(val_data):
                raise ValueError("Both train and val must contain images")
            _status(output, torch, psutil, phase="hashing_train_val", epoch=0,
                    train_images=len(train_data), val_images=len(val_data))
            signature = dataset_signature((train_data, val_data))
            initial_hash = sha256_file(initial_path)
            # Only an own, local completed-run checkpoint is accepted here.
            initial = torch.load(initial_path, map_location='cpu', weights_only=False)
            if sha256_file(initial_path) != initial_hash:
                raise RuntimeError('Initial checkpoint changed while being loaded')
            model_config = validate_initial_checkpoint(initial, signature, data, MODEL_SCHEMA)
            source_model_config = dict(model_config)
            model_config.update(min_size=args.min_size, max_size=args.max_size)
            source = {'path': str(initial_path), 'sha256': initial_hash,
                      'epoch': int(initial['epoch']), 'model_config': source_model_config,
                      'parent_run_config': initial['run_config'],
                      'optimizer_reused': False}
            code_files = ('train_refinement.py', 'refinement_config.py', 'amodal_model.py',
                          'dataset.py', 'evaluate.py', 'metrics.py', 'quality_breakdown.py',
                          'runtime.py', 'atomic_io.py')
            if target_cache is not None:
                code_files += ('target_cache.py',)
            code_snapshot = {name: sha256_file(Path(__file__).parent / name) for name in code_files}
            run_config = {"data": str(data), "epochs": args.epochs, "batch_size": args.batch_size,
                          "accumulate": args.accumulate, "min_size": args.min_size,
                          "max_size": args.max_size, "seed": args.seed,
                          "dataset_signature": signature, "workers": 0, "pin_memory": False,
                          "target_cache": cache_provenance,
                          "head_lr": args.head_lr, "backbone_lr": args.backbone_lr, "warmup_updates": args.warmup_updates,
                          "weight_decay": 0.0001, "amp": True, "amp_initial_scale": 128.0, "schema": MODEL_SCHEMA,
                          "baseline_validation": True,
                          "mode": "paired_resolution_weight_only_refinement_v1",
                          "source": source, "model_config": model_config,
                          "code_sha256": code_snapshot,
                          "augmentation": "sha256_seed_epoch_filename_horizontal_flip_v1",
                          "scope": "official_random_exploratory",
                          "train_images": len(train_data), "val_images": len(val_data),
                          "train_instances": sum(len(s["apple_ids"]) for s in train_data.samples),
                          "val_instances": sum(len(s["apple_ids"]) for s in val_data.samples),
                          "test_evaluation": "not_run"}
            previous = None
            if args.resume:
                # This is a checkpoint created by this runner in the same run;
                # optimizer and NumPy RNG structures require weights_only=False.
                previous = torch.load(resume, map_location="cpu", weights_only=False)
                if previous["run_config"] != run_config:
                    raise ValueError("Resume configuration/data differs from checkpoint; use original settings")
                if previous["architecture_metadata"]["schema"] != MODEL_SCHEMA:
                    raise ValueError("Checkpoint schema does not match this implementation")
                if previous['model_config'] != model_config:
                    raise ValueError('Resume model configuration differs from this refinement arm')
                model = build_model(**previous["model_config"])
                model.load_state_dict(previous["state_dict"], strict=True)
                model.architecture_metadata = previous["architecture_metadata"]
            else:
                atomic_json(output / "run_config.json", run_config)
                model = build_model(**model_config)
                model.load_state_dict(initial['state_dict'], strict=True)
                model.architecture_metadata = dict(initial['architecture_metadata'])
            del initial
            device = torch.device("cuda:0")
            model.to(device)
            groups = [
                {"params": [p for name, p in model.named_parameters()
                            if p.requires_grad and name.startswith("backbone.")],
                 "lr": args.backbone_lr, "base_lr": args.backbone_lr, "name": "backbone"},
                {"params": [p for name, p in model.named_parameters()
                            if p.requires_grad and not name.startswith("backbone.")],
                 "lr": args.head_lr, "base_lr": args.head_lr, "name": "heads"},
            ]
            optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
            scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
            generator = torch.Generator().manual_seed(args.seed)
            loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                generator=generator, num_workers=0, pin_memory=False,
                                collate_fn=_collate, drop_last=False)
            history, start_epoch, update, best_score = [], 0, 0, -1.0
            best_epoch = 0
            if previous:
                optimizer.load_state_dict(previous["optimizer"])
                scaler.load_state_dict(previous["scaler"])
                history = previous["history"]
                start_epoch = int(previous["epoch"])
                update = int(previous["update"])
                best_score = float(previous["best_score"])
                best_epoch = int(previous['best_epoch'])
                _restore_rng(torch, np, generator, previous["rng"])
                del previous
            atomic_json(output / "architecture.json", model.architecture_metadata)
            if history:
                _write_history(output / "results.csv", history)
            updates_per_epoch = math.ceil(len(loader) / args.accumulate)
            total_updates = updates_per_epoch * args.epochs
            if args.warmup_updates >= total_updates:
                raise ValueError('Warmup must finish before the final optimizer update')
            if not args.resume:
                _status(output, torch, psutil, phase="baseline_validation", epoch=0,
                        epochs=args.epochs, update=0)
                baseline_rng = _rng_state(torch, np, generator)
                baseline_dir = output / "validation" / "baseline"
                try:
                    baseline = evaluate_model(model, val_data, device, baseline_dir, max_images=None)
                    _status(output, torch, psutil, phase="baseline_quality_breakdown", epoch=0,
                            epochs=args.epochs, update=0)
                    _write_quality_breakdown(baseline_dir, val_data, baseline["confidence"])
                    baseline.setdefault("files", {})["quality_breakdown"] = "quality_breakdown.json"
                    atomic_json(output / "baseline_validation.json", baseline)
                    print(json.dumps({"event": "baseline_validation_complete", "confidence": baseline["confidence"],
                                      "precision": baseline["summary"]["precision"],
                                      "recall": baseline["summary"]["recall"],
                                      "score": baseline["score"]}), flush=True)
                finally:
                    # Diagnostic evaluation must not change subsequent training
                    # shuffle/augmentation RNG relative to an otherwise identical run.
                    _restore_rng(torch, np, generator, baseline_rng)
                best_score = float(baseline['score'])
                if not math.isfinite(best_score):
                    raise FloatingPointError('Nonfinite zero-update validation score')
                # Keep the initialization if all subsequent epochs regress.
                atomic_checkpoint(torch, output / 'best.pt', {
                    'state_dict': model.state_dict(), 'model_config': checkpoint_model_config(model),
                    'architecture_metadata': model.architecture_metadata, 'run_config': run_config,
                    'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                    'epoch': 0, 'update': 0, 'seed': args.seed,
                    'rng': _rng_state(torch, np, generator), 'best_score': best_score,
                    'best_epoch': 0, 'history': [], 'validation': baseline,
                    'saved_utc': utc_now(), 'resume_semantics': 'next_complete_epoch',
                })
                atomic_json(output / 'best_validation.json', baseline)
                del baseline_rng, baseline
            print(json.dumps({"event": "training_start", "host": platform.node(),
                              "gpu": torch.cuda.get_device_name(0), "start_epoch": start_epoch,
                              "train_images": len(train_data), "val_images": len(val_data),
                              "output": str(output)}, ensure_ascii=False), flush=True)

            for epoch in range(start_epoch, args.epochs):
                train_data.epoch = epoch + 1
                epoch_started = time.monotonic()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                running, samples, skipped_updates = {}, 0, 0
                for index, (images, targets) in enumerate(loader):
                    images = [image.to(device) for image in images]
                    targets = [{key: value.to(device) for key, value in target.items()}
                               for target in targets]
                    window_start = (index // args.accumulate) * args.accumulate
                    window_size = min(args.accumulate, len(loader) - window_start)
                    if index % args.accumulate == 0:
                        factor = _lr_factor(update, total_updates, warmup=args.warmup_updates)
                        for group in optimizer.param_groups:
                            group["lr"] = group["base_lr"] * factor
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        losses = model(images, targets)
                        total_loss = sum(losses.values())
                    current = {key: float(value.detach()) for key, value in losses.items()}
                    if not all(math.isfinite(value) for value in current.values()):
                        raise FloatingPointError(f"Nonfinite loss at epoch {epoch+1}, batch {index+1}: {current}")
                    scaler.scale(total_loss / window_size).backward()
                    samples += len(images)
                    for key, value in current.items():
                        running[key] = running.get(key, 0.0) + value * len(images)
                    if (index + 1) % args.accumulate == 0 or index + 1 == len(loader):
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        old_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        if not math.isfinite(float(grad_norm)) or scaler.get_scale() < old_scale:
                            skipped_updates += 1
                            print(json.dumps({"event": "amp_overflow_skipped", "epoch": epoch+1,
                                              "batch": index+1, "scale": scaler.get_scale()}), flush=True)
                        optimizer.zero_grad(set_to_none=True)
                        update += 1
                    if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(loader):
                        _status(output, torch, psutil, phase="training", epoch=epoch+1,
                                epochs=args.epochs, batch=index+1, batches=len(loader),
                                update=update, losses=current, skipped_updates=skipped_updates,
                                elapsed_epoch_seconds=time.monotonic()-epoch_started,
                                lrs={group["name"]: group["lr"] for group in optimizer.param_groups})
                        print(json.dumps({"event": "progress", "epoch": epoch+1, "batch": index+1,
                                          "batches": len(loader), "losses": current}), flush=True)
                    del images, targets, losses, total_loss
                if skipped_updates == updates_per_epoch:
                    raise FloatingPointError("AMP skipped every optimizer step in this epoch")
                _status(output, torch, psutil, phase="validation", epoch=epoch+1,
                        epochs=args.epochs, update=update)
                validation_dir = output / "validation" / f"epoch_{epoch+1:03d}"
                validation = evaluate_model(model, val_data, device, validation_dir, max_images=None)
                _status(output, torch, psutil, phase="quality_breakdown", epoch=epoch+1,
                        epochs=args.epochs, update=update)
                _write_quality_breakdown(validation_dir, val_data, validation["confidence"])
                validation.setdefault("files", {})["quality_breakdown"] = "quality_breakdown.json"
                score = float(validation["score"])
                if not math.isfinite(score):
                    raise FloatingPointError("Validation score is nonfinite")
                is_best = score > best_score + 1e-12
                if is_best:
                    best_epoch = epoch + 1
                best_score = max(best_score, score)
                row = {"epoch": epoch+1, "seconds": time.monotonic()-epoch_started,
                       "balanced_min": score, "confidence": validation.get("confidence"),
                       "skipped_updates": skipped_updates, "optimizer_updates": update}
                row.update({key: value / samples for key, value in running.items()})
                summary = validation.get("summary", {})
                for key in ("precision", "recall", "f1", "tp", "fp", "fn"):
                    if key in summary and isinstance(summary[key], (int, float)):
                        row["val_"+key] = summary[key]
                history.append(row)
                checkpoint = {
                    "state_dict": model.state_dict(), "model_config": checkpoint_model_config(model),
                    "architecture_metadata": model.architecture_metadata, "run_config": run_config,
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch+1, "update": update, "seed": args.seed,
                    "rng": _rng_state(torch, np, generator), "best_score": best_score,
                    "best_epoch": best_epoch,
                    "history": history, "validation": validation,
                    "saved_utc": utc_now(), "resume_semantics": "next_complete_epoch",
                }
                if is_best:
                    atomic_checkpoint(torch, output / "best.pt", checkpoint)
                    atomic_json(output / "best_validation.json", validation)
                atomic_checkpoint(torch, output / "last.pt", checkpoint)
                _write_history(output / "results.csv", history)
                _status(output, torch, psutil, phase="epoch_complete", epoch=epoch+1,
                        epochs=args.epochs, score=score, best_score=best_score, update=update)
                print(json.dumps({"event": "epoch_complete", **row, "best_score": best_score}), flush=True)
                del checkpoint
            _status(output, torch, psutil, phase="complete", epoch=args.epochs,
                    epochs=args.epochs, best_score=best_score, best_epoch=best_epoch, test_evaluation="not_run")
            return 0
        except BaseException as exc:
            _status(output, torch, psutil, phase="failed", error=repr(exc), traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    raise SystemExit(main())
