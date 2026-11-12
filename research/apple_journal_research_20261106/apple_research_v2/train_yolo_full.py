"""Audited, bounded FP32 YOLO26 baseline on existing F-drive data and packages.
No CUDA work occurs at import or with --preflight-only. No downloads or installs.
The final fixed epoch is primary; best development checkpoint is secondary.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import sys
import traceback

EXPECTED_VERSION = "8.4.115"
RESEARCH_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

class IntegrityError(Exception):
    """Fail the run, never enter upstream RuntimeError/OOM recovery."""


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f_path(path, *, exists=True):
    result = Path(path).resolve()
    if result.drive.upper() != "F:":
        raise IntegrityError(f"Every runtime/input path must be on F: {result}")
    if exists and not result.exists():
        raise IntegrityError(f"Missing local path: {result}")
    return result


def clean(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path, value):
    path = f_path(path, exists=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def configure_environment(root, name):
    root = f_path(root)
    runtime = root / "runtime" / name
    routes = {
        "YOLO_CONFIG_DIR": runtime / "ultralytics",
        "MPLCONFIGDIR": runtime / "matplotlib",
        "TORCH_HOME": runtime / "torch",
        "HF_HOME": runtime / "huggingface",
        "XDG_CACHE_HOME": runtime / "cache",
        "CUDA_CACHE_PATH": runtime / "cuda",
        "TORCH_EXTENSIONS_DIR": runtime / "torch_extensions",
        "PIP_CACHE_DIR": runtime / "pip",
        "TEMP": runtime / "tmp", "TMP": runtime / "tmp",
    }
    for name_, path in routes.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name_] = str(path)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                      PYTHONUTF8="1", PYTHONIOENCODING="utf-8", YOLO_OFFLINE="1",
                      YOLO_AUTOINSTALL="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      WANDB_MODE="disabled", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    sys.dont_write_bytecode = True
    os.chdir(root)
    return {k: os.environ[k] for k in routes} | {
        k: os.environ[k] for k in ("YOLO_OFFLINE", "YOLO_AUTOINSTALL", "CUBLAS_WORKSPACE_CONFIG")}


def block_network():
    def refuse(*args, **kwargs):
        raise IntegrityError("Network access is disabled during audited local training")
    socket.create_connection = refuse
    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse


def split_images(value, base):
    entries = value if isinstance(value, list) else [value]
    paths = []
    for entry in entries:
        path = Path(str(entry))
        path = f_path(path if path.is_absolute() else base / path)
        if path.is_dir():
            paths.extend(sorted(p.resolve() for p in path.rglob("*")
                                if p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line:
                    image = Path(line)
                    paths.append(f_path(image if image.is_absolute() else path.parent / image))
        else:
            raise IntegrityError(f"Split must be a local directory or image-list TXT: {path}")
    if not paths or len(set(paths)) != len(paths):
        raise IntegrityError("Empty split or duplicate input paths")
    return paths


def image_label_path(image):
    parts = list(image.parts)
    indices = [i for i, part in enumerate(parts) if part.lower() == "images"]
    if not indices:
        raise IntegrityError(f"Image path needs an images directory for YOLO label mapping: {image}")
    parts[indices[-1]] = "labels"
    return f_path(Path(*parts).with_suffix(".txt"))


def audit_inputs(args):
    import yaml
    from PIL import Image
    weights, data_path, manifest = map(f_path, (args.weights, args.data, args.manifest))
    if sha(weights) != args.expected_weights_sha256.lower():
        raise IntegrityError("Pretrained weight SHA256 mismatch")
    manifest_bytes = manifest.read_bytes()
    json.loads(manifest_bytes.decode("utf-8-sig"))
    if args.expected_manifest_sha256 and sha(manifest) != args.expected_manifest_sha256.lower():
        raise IntegrityError("Source manifest SHA256 mismatch")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or "download" in data:
        raise IntegrityError("Dataset YAML must be a mapping without a download command")
    if not data.get("train") or not data.get("val"):
        raise IntegrityError("Explicit train and val splits are required; no test fallback")
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or sorted(names) != list(range(len(names))) or not names:
        raise IntegrityError("Class names must have contiguous integer IDs")
    base = Path(data.get("path", data_path.parent))
    base = f_path(base if base.is_absolute() else data_path.parent / base)
    records, image_hashes = {}, {}
    for split in ("train", "val"):
        rows = []
        for image in split_images(data[split], base):
            image = f_path(image)
            label = image_label_path(image)
            with Image.open(image) as im:
                width, height = im.size
                im.verify()
            if min(width, height) <= 0:
                raise IntegrityError(f"Invalid image size: {image}")
            class_counts = Counter()
            seen_rows = set()
            for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                numbers = [float(v) for v in fields]
                if not all(math.isfinite(v) for v in numbers):
                    raise IntegrityError(f"Nonfinite label: {label}:{line_number}")
                cls, coords = numbers[0], numbers[1:]
                if not cls.is_integer() or int(cls) not in names or any(v < 0 or v > 1 for v in coords):
                    raise IntegrityError(f"Invalid class/coordinate: {label}:{line_number}")
                if args.task == "detect":
                    if len(coords) != 4 or coords[2] <= 0 or coords[3] <= 0:
                        raise IntegrityError(f"Expected positive YOLO box: {label}:{line_number}")
                    x, y, w, h = coords
                    if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
                        raise IntegrityError(f"Box extends outside image: {label}:{line_number}")
                elif len(coords) < 6 or len(coords) % 2:
                    raise IntegrityError(f"Expected YOLO polygon: {label}:{line_number}")
                row = tuple(numbers)
                if row in seen_rows:
                    raise IntegrityError(f"Duplicate label: {label}:{line_number}")
                seen_rows.add(row)
                class_counts[int(cls)] += 1
            rows.append(dict(image=str(image), label=str(label), image_sha256=sha(image),
                             label_sha256=sha(label), width=width, height=height,
                             instances=sum(class_counts.values()), class_counts=dict(class_counts)))
        records[split] = rows
        image_hashes[split] = {r["image_sha256"] for r in rows}
    if image_hashes["train"] & image_hashes["val"]:
        raise IntegrityError("Exact image duplicates cross train/val")
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    resolved = {"path": base, "train": data["train"], "val": data["val"],
                "names": names, "nc": len(names), "channels": 3, "yaml_file": str(data_path)}
    # Explicit lists avoid stale C-drive paths and preserve exactly the audited images.
    resolved["train"] = [r["image"] for r in records["train"]]
    resolved["val"] = [r["image"] for r in records["val"]]
    # YOLO dataset accepts image-list files, not lists of individual image paths.
    return resolved, {"schema": "audited_yolo_inputs_v1", "created_utc": utc(),
        "weights": str(weights), "weights_sha256": sha(weights),
        "data_yaml": str(data_path), "data_yaml_sha256": sha(data_path),
        "source_manifest": str(manifest), "source_manifest_sha256": sha(manifest),
        "source_manifest_role": "provenance; actual image and label bytes independently audited",
        "actual_inputs_sha256": digest, "records": records, "test_read": False,
        "independent_test_claimed": False, "split_role": "development; fruit independence unverified"}


def recipe(args):
    return dict(task=args.task, model=str(f_path(args.weights)), data=str(f_path(args.data)),
        project=str(RESEARCH_ROOT / "runs"), name=args.run_name, exist_ok=False,
        epochs=args.epochs, imgsz=640, batch=4, nbs=16, workers=0, device=0,
        optimizer="AdamW", lr0=.001, lrf=.01, cos_lr=True, momentum=.9, weight_decay=.0005,
        warmup_epochs=3., warmup_momentum=.8, warmup_bias_lr=0., seed=args.seed,
        deterministic=True, amp=False, patience=0, fraction=1., cache=False,
        compile=False, rect=False, multi_scale=0., save=True, save_period=5,
        val=True, plots=False, close_mosaic=5, hsv_h=0., hsv_s=0., hsv_v=.1,
        bgr=0., mosaic=.25, mixup=0., copy_paste=0., cutmix=0., scale=.25,
        fliplr=.5, flipud=0., degrees=0., shear=0., perspective=0., translate=.1,
        overlap_mask=False, mask_ratio=args.mask_ratio, cls_pw=0., cls_remap=True,
        resume=False, pretrained=True, single_cls=False, freeze=None, profile=False,
        channels_last=False, conf=.001, iou=.7, max_det=300, quantize=32,
        augment=False, verbose=True)


def trainer_class(torch, args, resolved, audit, run_dir):
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from ultralytics.utils.torch_utils import unwrap_model
    from ultralytics.utils import callbacks
    parent = SegmentationTrainer if args.task == "segment" else DetectionTrainer

    class AuditedTrainer(parent):
        def __init__(self, *a, **kw):
            self.pipeline_builds = 0
            self.actual_updates = 0
            self.participation = Counter()
            self.anchor_seen = {}
            self.epoch_reports = []
            self.forward_calls = 0
            super().__init__(*a, **kw)
            self.best = self.wdir / "best_dev.pt"

        def get_dataset(self):
            return deepcopy(resolved)

        def _build_train_pipeline(self):
            self.pipeline_builds += 1
            if self.pipeline_builds != 1 or self.batch_size != 4:
                raise IntegrityError("OOM/automatic batch rebuild forbidden; stop and report failed run")
            super()._build_train_pipeline()
            if len(self.train_loader.dataset) != len(audit["records"]["train"]):
                raise IntegrityError("Training loader silently removed/added images")
            if len(self.test_loader.dataset) != len(audit["records"]["val"]):
                raise IntegrityError("Validation loader silently removed/added images")

        def _finite(self, named, context):
            values = [(n, t) for n, t in named if torch.is_tensor(t) and t.is_floating_point()]
            if not values:
                return
            # A single host synchronization per device group, rather than per parameter.
            groups = {}
            for n, t in values:
                groups.setdefault(str(t.device), []).append((n, t))
            for group in groups.values():
                flags = torch.stack([torch.isfinite(t.detach()).all() for _, t in group])
                if not bool(flags.all().item()):
                    bad = [group[i][0] for i in (~flags).nonzero().flatten().cpu().tolist()]
                    raise IntegrityError(f"Nonfinite {context}: {bad[:8]}")

        def _forward_guard(self, module, inputs, output):
            if module.training:
                loss = output[0] if isinstance(output, tuple) else output
                values = list(loss.items()) if isinstance(loss, dict) else [("loss", loss)]
                self._finite(values, "forward loss before backward")
                self.forward_calls += 1

        def _setup_train(self):
            super()._setup_train()
            if self.amp or self.scaler.is_enabled() or self.world_size > 1:
                raise IntegrityError("Only audited single-device FP32 training is supported")
            model = unwrap_model(self.model)
            self._finite(model.state_dict().items(), "initial model")
            self._finite(self.ema.ema.state_dict().items(), "initial EMA")
            self.loss_guard = model.register_forward_hook(self._forward_guard)
            self.named_parameters = dict(model.named_parameters())
            write_json(run_dir / "runtime_contract.json", {
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0), "seed": args.seed,
                "trainer_seed_actual": args.seed, "pythonhashseed": os.getenv("PYTHONHASHSEED"),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "amp": self.amp, "all_trainable_parameters_fp32": all(
                    p.dtype == torch.float32 for p in model.parameters() if p.requires_grad),
                "batches_per_epoch": len(self.train_loader), "validation_batch": self.test_loader.batch_size,
                "train_images": len(self.train_loader.dataset), "val_images": len(self.test_loader.dataset),
                "model_end2end_head": bool(getattr(model, "end2end", False)),
                "independent_test_claimed": False})

        def preprocess_batch(self, batch):
            result = super().preprocess_batch(batch)
            counts = self.anchor_seen.setdefault(int(self.epoch), Counter())
            counts.update(str(Path(p).resolve()) for p in batch["im_file"])
            return result

        def optimizer_step(self):
            if self.amp or self.scaler.is_enabled():
                raise IntegrityError("Unexpected AMP change")
            participating = [(n, p) for n, p in self.named_parameters.items() if p.grad is not None]
            if not participating:
                raise IntegrityError("Optimizer update has no participating gradient")
            self._finite(((n, p.grad) for n, p in participating), "gradient before optimizer")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10., error_if_nonfinite=True)
            before = self.actual_updates
            fired = []
            handle = self.optimizer.register_step_post_hook(lambda *_: fired.append(True))
            try:
                self.optimizer.step()
            finally:
                handle.remove()
            if len(fired) != 1:
                raise IntegrityError("Actual optimizer step was not observed exactly once")
            self.actual_updates = before + 1
            self.participation.update(n for n, _ in participating)
            self.optimizer.zero_grad(set_to_none=True)
            self.ema.update(self.model)
            self._finite(unwrap_model(self.model).state_dict().items(), "updated model")
            self._finite(self.ema.ema.state_dict().items(), "updated EMA")

        def _audit_optimizer(self):
            mismatch = []
            for name, parameter in self.named_parameters.items():
                actual = self.optimizer.state.get(parameter, {}).get("step", 0)
                actual = int(actual.item()) if torch.is_tensor(actual) else int(actual)
                if actual != self.participation[name]:
                    mismatch.append((name, actual, self.participation[name]))
            if mismatch:
                raise IntegrityError(f"AdamW per-parameter step mismatch: {mismatch[:8]}")
            if self.ema.updates != self.actual_updates:
                raise IntegrityError("EMA updates differ from observed optimizer steps")

        def validate(self):
            metrics, fitness = super().validate()
            if self.validator.seen != len(audit["records"]["val"]):
                raise IntegrityError("Full validation image coverage failed")
            if not math.isfinite(float(fitness)) or any(not math.isfinite(float(v)) for v in metrics.values()):
                raise IntegrityError("Nonfinite validation metrics")
            return metrics, fitness

        def _handle_nan_recovery(self, epoch):
            self._finite([("loss", self.loss)], "epoch loss")
            if not math.isfinite(float(self.fitness)):
                raise IntegrityError("Nonfinite fitness; no automatic recovery")
            return False

        def save_model(self):
            expected = Counter(r["image"] for r in audit["records"]["train"])
            if self.anchor_seen.get(int(self.epoch), Counter()) != expected:
                raise IntegrityError("Incomplete/duplicate epoch anchor-image exposure")
            self._audit_optimizer()
            self._finite(unwrap_model(self.model).state_dict().items(), "live pre-serialization model")
            self._finite(self.ema.ema.state_dict().items(), "pre-serialization EMA")
            for parameter, state in self.optimizer.state.items():
                self._finite(state.items(), "optimizer state")
            ema = deepcopy(unwrap_model(self.ema.ema)).float().cpu()
            if hasattr(ema, "criterion"):
                ema.criterion = None
            self._finite(ema.state_dict().items(), "FP32 serialized EMA")
            per_class = clean(self.validator.metrics.summary(decimals=8))
            support = Counter()
            for image in audit["records"]["val"]:
                support.update({int(k): int(v) for k, v in image["class_counts"].items()})
            row = {"epoch_completed": self.epoch + 1, "optimizer_updates": self.actual_updates,
                   "anchor_images": sum(self.anchor_seen[self.epoch].values()),
                   "validation_images": self.validator.seen, "metrics": clean(self.metrics),
                   "fitness": float(self.fitness), "utc": utc(),
                   "per_class": per_class, "validation_gt_support": dict(support),
                   "precision_recall_operating_point": "Ultralytics max mean F1 on this development split; NOT fixed confidence 0.25",
                   "validation_candidate_confidence_floor": .001,
                   "actual_inputs_sha256": audit["actual_inputs_sha256"], "test_used": False}
            ckpt = {"epoch": self.epoch, "model": None, "ema": ema,
                    "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
                    "updates": self.ema.updates, "best_fitness": self.best_fitness,
                    "train_args": vars(self.args), "train_metrics": row["metrics"] | {"fitness": row["fitness"]},
                    "date": utc(), "version": EXPECTED_VERSION, "license": "AGPL-3.0",
                    "audit": row, "parameter_participation": dict(self.participation),
                    "serialization_precision": "FP32; no nan_to_num or sanitization",
                    "exact_resume_supported": False,
                    "resume_limitation": "EMA-only model snapshot; raw live model and RNG state are not saved. Do not claim exact resume."}
            output = io.BytesIO()
            torch.save(ckpt, output)
            data_bytes = output.getvalue()
            self.wdir.mkdir(parents=True, exist_ok=True)
            targets = [self.last]
            if self.best_fitness == self.fitness:
                targets.append(self.best)
            if (self.epoch + 1) % 5 == 0:
                targets.append(self.wdir / f"epoch_{self.epoch+1:03d}.pt")
            for path in targets:
                temporary = path.with_suffix(".pt.tmp")
                temporary.write_bytes(data_bytes)
                os.replace(temporary, path)
            self.epoch_reports.append(row)
            write_json(run_dir / "epoch_audit.json", self.epoch_reports)
            write_json(run_dir / "status.json", {"phase": "training", **row,
                       "total_epochs": args.epochs, "last": str(self.last), "best_dev": str(self.best)})
            return True

        def final_eval(self):
            # The final epoch already received full validation. Do not strip optimizer,
            # convert checkpoints to FP16, or replace the primary result with best-dev.
            if len(self.epoch_reports) != args.epochs or self.epoch + 1 != args.epochs:
                raise IntegrityError("Fixed epoch budget did not complete")
            self._audit_optimizer()
            write_json(run_dir / "final_selection.json", {
                "primary": {"path": str(self.last), "sha256": sha(self.last),
                            "selection": "fixed final epoch", "epoch": args.epochs,
                            "validation": self.epoch_reports[-1]},
                "secondary": {"path": str(self.best), "sha256": sha(self.best),
                              "selection": "best development fitness; not independent test"},
                "test_used": False, "independent_test_claimed": False,
                "checkpoint_precision": "FP32", "optimizer_updates": self.actual_updates,
                "exact_resume_supported": False,
                "resume_limitation": "No automatic restart. This runner deliberately refuses existing runs; checkpoints are not exact training-resume snapshots."})

    return AuditedTrainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("segment", "detect"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--mask-ratio", type=int, choices=(2, 4), default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name) or not 20 <= args.epochs <= 50:
        parser.error("Unique safe run-name and fixed 20-50 epoch budget required")
    env_record = configure_environment(RESEARCH_ROOT, args.run_name)
    block_network()
    resolved, audit = audit_inputs(args)
    contract = {"args": vars(args), "recipe": recipe(args), "environment": env_record,
                "source_sha256": sha(__file__), "input_audit": audit,
                "development_only": True, "primary_selection": "fixed final epoch",
                "preflight_only": args.preflight_only}
    preflight_path = RESEARCH_ROOT / "preflight" / (args.run_name + ".json")
    write_json(preflight_path, contract)
    if args.preflight_only:
        print(json.dumps({"phase": "preflight_passed", "report": str(preflight_path),
              "train_images": len(audit["records"]["train"]), "val_images": len(audit["records"]["val"]),
              "actual_inputs_sha256": audit["actual_inputs_sha256"], "gpu_used": False}))
        return 0
    # PYTHONHASHSEED must be set before this interpreter starts; fail instead of pretending
    # changing it now seeds existing Python hash state.
    if os.getenv("RANK", "-1") != "-1" or os.getenv("LOCAL_RANK", "-1") != "-1":
        raise IntegrityError("Distributed rank environments are not supported by this single-GPU protocol")
    if os.getenv("PYTHONHASHSEED") != str(args.seed):
        raise IntegrityError(f"Set PYTHONHASHSEED={args.seed} before launching this process")
    run_dir = RESEARCH_ROOT / "runs" / args.run_name
    if run_dir.exists():
        raise IntegrityError(f"Run directory already exists; no overwrite/resume: {run_dir}")
    import numpy as np
    import torch
    import ultralytics
    from ultralytics.utils import SETTINGS
    from ultralytics.utils import downloads
    import ultralytics.data.dataset as dataset_module
    if ultralytics.__version__ != EXPECTED_VERSION:
        raise IntegrityError(f"Requires ultralytics {EXPECTED_VERSION}, found {ultralytics.__version__}")
    SETTINGS.update({"datasets_dir": str(RESEARCH_ROOT / "apple_research_v2" / "datasets"),
                     "weights_dir": str(f_path(args.weights).parent), "runs_dir": str(RESEARCH_ROOT / "runs"),
                     "sync": False, "clearml": False, "comet": False, "dvc": False,
                     "mlflow": False, "neptune": False, "raytune": False, "tensorboard": False,
                     "wandb": False})
    def existing_asset(file, *a, **kw):
        return str(f_path(file))
    def no_download(*a, **kw):
        raise IntegrityError("Automatic asset/data download forbidden")
    def no_cache_read(*a, **kw):
        raise FileNotFoundError("Rebuild labels from independently audited text files")
    downloads.attempt_download_asset = existing_asset
    downloads.safe_download = no_download
    dataset_module.load_dataset_cache_file = no_cache_read
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise IntegrityError("A CUDA GPU is required for this full training recipe")
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Freeze audited image lists inside this run, so later YAML edits cannot change coverage.
    run_dir.mkdir(parents=True)
    try:
        for split in ("train", "val"):
            list_path = run_dir / (split + "_audited_images.txt")
            list_path.write_text("\n".join(resolved[split]) + "\n", encoding="utf-8")
            resolved[split] = str(list_path)
        package_root = Path(ultralytics.__file__).resolve().parent
        contract["upstream_source_sha256"] = {str(package_root / name): sha(package_root / name)
            for name in ("engine/trainer.py", "engine/validator.py", "nn/tasks.py",
                         "utils/loss.py", "utils/metrics.py", "cfg/default.yaml", "data/dataset.py",
                         "models/yolo/detect/train.py", "models/yolo/segment/train.py")}
        contract["exact_resume_supported"] = False
        write_json(run_dir / "experiment_contract.json", contract)
        # Trainer refuses an occupied run by default; only this newly-created isolated
        # directory contains the manifest files above. Existing runs were rejected earlier.
        options = recipe(args)
        options["exist_ok"] = True
        Trainer = trainer_class(torch, args, resolved, audit, run_dir)
        trainer = Trainer(overrides=options)
        write_json(run_dir / "status.json", {"phase": "starting", "pid": os.getpid(), "utc": utc()})
        trainer.train()
        if len(trainer.epoch_reports) != args.epochs:
            raise IntegrityError("Training returned before completing the fixed budget")
        _, end_audit = audit_inputs(args)
        for field in ("actual_inputs_sha256", "weights_sha256", "data_yaml_sha256", "source_manifest_sha256"):
            if end_audit[field] != audit[field]:
                raise IntegrityError(f"Inputs changed during training: {field}")
        if sha(__file__) != contract["source_sha256"]:
            raise IntegrityError("Runner source changed during training")
        for path, digest in contract["upstream_source_sha256"].items():
            if sha(path) != digest:
                raise IntegrityError(f"Upstream package source changed during training: {path}")
        write_json(run_dir / "input_recheck.json", {"phase": "unchanged", "utc": utc(),
            "actual_inputs_sha256": end_audit["actual_inputs_sha256"],
            "image_and_label_bytes_rehashed": True, "runner_and_upstream_sources_rehashed": True})
        write_json(run_dir / "status.json", {"phase": "complete", "utc": utc(), "pid": os.getpid(),
            "epochs_completed": args.epochs, "optimizer_updates": trainer.actual_updates,
            "primary": str(trainer.last), "secondary": str(trainer.best), "test_used": False,
            "independent_test_claimed": False, "actual_inputs_sha256": audit["actual_inputs_sha256"]})
    except BaseException as error:
        write_json(run_dir / "status.json", {"phase": "failed", "utc": utc(), "pid": os.getpid(),
                   "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
                   "no_automatic_resume": True, "test_used": False})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
