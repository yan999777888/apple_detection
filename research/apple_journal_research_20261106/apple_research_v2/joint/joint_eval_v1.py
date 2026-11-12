"""Reference-preserving official validation and native Results for a joint model.

The model API is names[stage|defect], head_for(task), predict_source(x, task),
predict_both(x), and mode (joint|separate). No model copies or criterion calls.
Use the EMA model for validation. No simultaneous training on the same object.
"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
import random
from pathlib import Path
import socket
import sys
import threading
from types import SimpleNamespace
import weakref

ROOT = Path(__file__).resolve().parents[2]
TASK_TYPES = {"stage": "segment", "defect": "detect"}
DATA_FILES = {"stage": ROOT / "apple_research_v2/datasets/mendeley_stage_seg_v1/data.yaml", "defect": ROOT / "apple_research_v2/datasets/ssda_date_v2/data.yaml"}
EXPECTED_IMAGES = {"stage": 72, "defect": 113}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def configure_runtime():
    if ROOT.drive.upper() != "F:":
        raise RuntimeError("Joint adapters must reside in the F research root")
    runtime = ROOT / "runtime/joint_eval_v1"
    routes = {"YOLO_CONFIG_DIR": "ultralytics", "MPLCONFIGDIR": "matplotlib", "TORCH_HOME": "torch", "HF_HOME": "hf", "XDG_CACHE_HOME": "cache", "CUDA_CACHE_PATH": "cuda", "TORCH_EXTENSIONS_DIR": "torch_extensions", "PIP_CACHE_DIR": "pip", "TEMP": "tmp", "TMP": "tmp", "TMPDIR": "tmp"}
    for key, subdir in routes.items():
        present = os.environ.get(key)
        dest = Path(present) if present and Path(present).drive.upper() == "F:" else runtime / subdir
        dest.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(dest)
    os.environ.update(YOLO_OFFLINE="1", YOLO_AUTOINSTALL="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "2")
    sys.dont_write_bytecode = True


configure_runtime()
import numpy as np
import torch
from torch import nn
import ultralytics
from ultralytics.utils import USER_CONFIG_DIR, YAML
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.segment.predict import SegmentationPredictor

if ultralytics.__version__ != "8.4.115" or Path(USER_CONFIG_DIR).drive.upper() != "F:":
    raise RuntimeError("Only audited local Ultralytics 8.4.115 with F settings is supported")

_LOCKS = weakref.WeakKeyDictionary()
_LOCKS_LOCK = threading.Lock()
_MISSING = object()
HEAD_ATTRS = ("shape", "anchors", "strides", "stride", "max_det", "agnostic_nms", "export", "format", "dynamic", "_end2end", "xyxy")


def _lock(model):
    with _LOCKS_LOCK:
        return _LOCKS.setdefault(model, threading.RLock())


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def f_path(path, *, within_root=False):
    p = Path(path).resolve()
    if p.drive.upper() != "F:" or (within_root and not p.is_relative_to(ROOT)):
        raise ValueError(f"F-only research path required: {p}")
    return p


def save_json(path, value):
    p = f_path(path, within_root=True)
    p.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@contextmanager
def offline_network():
    def refuse(*args, **kwargs):
        raise RuntimeError("Network is prohibited in local joint evaluation")
    old = (socket.create_connection, socket.socket.connect, socket.socket.connect_ex)
    socket.create_connection = socket.socket.connect = socket.socket.connect_ex = refuse
    try:
        yield
    finally:
        socket.create_connection, socket.socket.connect, socket.socket.connect_ex = old


def _device(model):
    tensors = list(model.parameters()) + list(model.buffers())
    devices = {t.device for t in tensors}
    if len(devices) != 1:
        raise ValueError("All shared-model parameters/buffers must already be on one device")
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in tensors):
        raise ValueError("Adapters require existing FP32 weights; no in-place precision conversion")
    return next(iter(devices))


def _tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _criteria_signature(model):
    criteria = getattr(model, "criteria", {})
    return {"dictionary_id": id(criteria), "criterion_epochs": dict(getattr(model, "criterion_epochs", {})), "objects": {k: {"id": id(v), **{a: getattr(v, a, None) for a in ("o2m", "o2o", "total", "o2m_copy", "final_o2m")}} for k, v in criteria.items()}}


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError("Unexpected non-tensor model state")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite model state: {name}")
        h.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@contextmanager
def preserve_eval_state(model, *, verify_values=True):
    """Restore exact mixed training/gradient flags and head caches, even on failure.

    State bytes are compared, never repaired. A mutated value aborts evaluation.
    This context serializes adapters; caller must also avoid concurrent training.
    """
    with _lock(model):
        _device(model)
        modules = list(model.modules())
        parameters = list(model.parameters())
        flags = [(m, m.training) for m in modules]
        grads = [(p, p.requires_grad, p.grad, None if p.grad is None else p.grad._version) for p in parameters]
        structure = [(n, id(p), p.data_ptr(), p.dtype, p.device) for n, p in model.named_parameters()]
        buffers = [(n, id(b), b.data_ptr(), b.dtype, b.device) for n, b in model.named_buffers()]
        head_attrs = [(head, name, head.__dict__.get(name, _MISSING)) for head in (model.head_for("stage"), model.head_for("defect")) for name in HEAD_ATTRS]
        criterion = model.__dict__.get("criterion", _MISSING)
        criteria_before = _criteria_signature(model)
        cache_hashes = [(head, name, _tensor_digest(value)) for head, name, value in head_attrs if isinstance(value, torch.Tensor)] if verify_values else []
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        cuda_rng = torch.cuda.get_rng_state(_device(model)) if _device(model).type == "cuda" else None
        before = state_digest(model) if verify_values else None
        report = {"verify_values": verify_values, "state_sha256_before": before, "deepcopy_used": False}
        model.eval()
        try:
            with torch.inference_mode():
                yield report
        finally:
            for head, name, value in head_attrs:
                if value is _MISSING:
                    head.__dict__.pop(name, None)
                else:
                    setattr(head, name, value)
            for p, requires_grad, grad, version in grads:
                p.requires_grad_(requires_grad)
            for module, training in flags:
                module.training = training
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, _device(model))
            current = [(n, id(p), p.data_ptr(), p.dtype, p.device) for n, p in model.named_parameters()]
            current_buffers = [(n, id(b), b.data_ptr(), b.dtype, b.device) for n, b in model.named_buffers()]
            if structure != current or buffers != current_buffers or [id(m) for m in modules] != [id(m) for m in model.modules()]:
                raise RuntimeError("Evaluation changed model structure/storage; no successful metrics may be used")
            if any(p.grad is not grad or (grad is not None and grad._version != version) for p, _, grad, version in grads):
                raise RuntimeError("Evaluation changed training gradients")
            if model.__dict__.get("criterion", _MISSING) is not criterion:
                raise RuntimeError("Evaluation initialized or changed the training criterion")
            if _criteria_signature(model) != criteria_before:
                raise RuntimeError("Evaluation altered a source criterion or loss schedule")
            if any(_tensor_digest(getattr(head, name)) != digest for head, name, digest in cache_hashes):
                raise RuntimeError("Evaluation altered preexisting head cache tensor values")
            after = state_digest(model) if verify_values else None
            if before != after:
                raise RuntimeError("Evaluation changed model/EMA values; results rejected rather than repaired")
            report.update(state_sha256_after=after, parameter_references_preserved=True, training_flags_restored=True, requires_grad_restored=True, gradients_preserved=True, head_runtime_attributes_restored=True, criterion_untouched=True, source_criteria_and_schedules_untouched=True, random_states_restored=True)


class TaskView(nn.Module):
    """A reference-only task view; fuse/to/float never modify the shared network."""
    def __init__(self, shared_model, source_task):
        super().__init__()
        if source_task not in TASK_TYPES:
            raise ValueError("Task must be stage or defect")
        self.shared_model = shared_model
        self.source_task = source_task
        self.task = TASK_TYPES[source_task]
        self.names = {int(k): str(v) for k, v in shared_model.names[source_task].items()}
        if sorted(self.names) != list(range(len(self.names))) or len(self.names) != shared_model.head_for(source_task).nc:
            raise ValueError("Head class metadata mismatch")
        self.yaml = {"channels": 3, "nc": len(self.names), "task": self.task}
        self.args = {"task": self.task, "imgsz": 640, "quantize": 32}
        self.forward_calls = 0
        self.noop_fuse_calls = 0
        self.inference_precision = "FP32"
        self.observed_conv_dtypes = set()

    @property
    def stride(self):
        return self.shared_model.head_for(self.source_task).stride

    @property
    def end2end(self):
        return self.shared_model.head_for(self.source_task).end2end

    def set_head_attr(self, **kwargs):
        if set(kwargs) - {"max_det", "agnostic_nms"}:
            raise ValueError("Unsupported temporary head metadata")
        for key, value in kwargs.items():
            setattr(self.shared_model.head_for(self.source_task), key, value)

    def fuse(self, *args, **kwargs):
        self.noop_fuse_calls += 1
        return self

    def to(self, device=None, *args, **kwargs):
        if args or kwargs:
            raise ValueError("TaskView forbids conversion; place the shared model explicitly before evaluation")
        if device is not None and torch.device(device) != _device(self.shared_model):
            raise ValueError("TaskView must use the model's existing device")
        return self

    def float(self):
        _device(self.shared_model)
        return self

    def half(self):
        raise ValueError("TaskView never mutates FP32 shared weights to FP16")

    def forward(self, images, augment=False, embed=None, **kwargs):
        if augment or embed is not None or kwargs:
            raise ValueError("Only ordinary non-augmented FP32 inference is supported")
        if self.shared_model.training or self.shared_model.head_for(self.source_task).training:
            raise RuntimeError("TaskView is eval-only")
        self.forward_calls += 1
        # BaseValidator disables its outer autocast for standalone validation.
        # Enter the requested context INSIDE forward and verify a real conv.
        trunk = self.shared_model.trunks["shared" if self.shared_model.mode == "joint" else self.source_task]
        conv = next(m for m in trunk.modules() if isinstance(m, nn.Conv2d))
        hook = conv.register_forward_hook(lambda module, inputs, output: self.observed_conv_dtypes.add(str(output.dtype)))
        try:
            with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=self.inference_precision == "AMP_FP16"):
                return self.shared_model.predict_source(images, self.source_task)
        finally:
            hook.remove()


def source_spec(task, data=None):
    if task not in TASK_TYPES:
        raise ValueError("Unknown task")
    path = f_path(data or DATA_FILES[task])
    if path != DATA_FILES[task].resolve() or not path.is_file():
        raise ValueError("Use the fixed source-specific development YAML")
    info = YAML.load(path)
    if info.get("download"):
        raise ValueError("Dataset download instructions are prohibited")
    names = {int(k): str(v) for k, v in info["names"].items()}
    base = f_path(info.get("path", path.parent))
    val = f_path(base / info["val"])
    if val.is_dir():
        images = sorted(str(p.resolve()) for p in val.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        index_hash = None
    elif val.suffix.lower() == ".txt":
        images = []
        for line in val.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                p = Path(line.strip())
                images.append(str(f_path(p if p.is_absolute() else val.parent / p)))
        index_hash = sha_file(val)
    else:
        raise ValueError("Unsupported fixed development index")
    if len(images) != EXPECTED_IMAGES[task] or len(set(images)) != len(images) or any(not Path(p).is_file() for p in images):
        raise ValueError("Development source count/identity does not match the frozen source")
    return {"source_task": task, "task": TASK_TYPES[task], "data": str(path), "data_sha256": sha_file(path), "names": names, "images": images, "expected_images": len(images), "index_sha256": index_hash}


def _per_class(validator):
    metrics = validator.metrics
    rows = []
    for class_id, name in sorted(validator.names.items()):
        row = {"class_id": class_id, "name": name, "instances": int(metrics.nt_per_class[class_id]), "images": int(metrics.nt_per_image[class_id])}
        for kind, metric in (("box", metrics.box), ("mask", getattr(metrics, "seg", None))):
            if metric is None:
                continue
            indices = list(metric.ap_class_index)
            if class_id not in indices:
                row[kind] = None
            else:
                idx = indices.index(class_id)
                row[kind] = {"precision": float(metric.p[idx]), "recall": float(metric.r[idx]), "f1": float(metric.f1[idx]), "AP50": float(metric.ap50[idx]), "AP50_95": float(metric.ap[idx])}
        rows.append(row)
    return rows


def evaluate_task(model, task, data=None, output_dir=None, device=None, batch=1, imgsz=640, *, dataloader=None, scope="full_development", expected_images=None, inference_precision="FP32"):
    """Run the actual official validator for one labeled source and one head.

    Returns a receipt whose metrics are source-specific. No fitness/combined
    score is returned. scope='cpu_contract_only' permits a supplied small real
    loader for interface tests; such results are not research evaluation.
    """
    spec = source_spec(task, data)
    if {int(k): str(v) for k, v in model.names[task].items()} != spec["names"]:
        raise ValueError("Model names differ from source label authority")
    actual_device = _device(model)
    if inference_precision not in ("FP32", "AMP_FP16") or (inference_precision == "AMP_FP16" and actual_device.type != "cuda"):
        raise ValueError("Supported validation precision: FP32 or CUDA AMP_FP16")
    if device is not None and torch.device(device) != actual_device:
        raise ValueError("Move the model explicitly; validation cannot move EMA/training weights")
    if output_dir is None or imgsz != 640 or batch < 1:
        raise ValueError("Explicit output and fixed 640 evaluation required")
    output = f_path(output_dir, within_root=True)
    if output.exists():
        raise FileExistsError("Each epoch/source evaluation must have a new output directory")
    if scope == "full_development":
        if expected_images not in (None, spec["expected_images"]):
            raise ValueError("Cannot change full development coverage")
        expected_images = spec["expected_images"]
    elif scope == "cpu_contract_only":
        if actual_device.type != "cpu" or dataloader is None or not isinstance(expected_images, int) or expected_images < 1:
            raise ValueError("CPU contract mode requires an explicit real subset loader/count")
    else:
        raise ValueError("Unknown evaluation scope")
    output.mkdir(parents=True, exist_ok=False)
    seen_paths = []
    base = SegmentationValidator if task == "stage" else DetectionValidator
    class AuditedValidator(base):
        def preprocess(self, batch_data):
            seen_paths.extend(str(f_path(p)) for p in batch_data["im_file"])
            return super().preprocess(batch_data)
    args = dict(task=TASK_TYPES[task], mode="val", data=str(spec["data"]), device=str(actual_device), batch=batch, imgsz=640, workers=0, rect=True, quantize=32, conf=0.001, iou=0.7, max_det=300, single_cls=False, agnostic_nms=False, augment=False, end2end=None, plots=False, save=False, save_json=False, save_txt=False, verbose=False, overlap_mask=False, mask_ratio=4, compile=False, channels_last=False, split="val", project=str(output.parent), name=output.name, exist_ok=False)
    view = TaskView(model, task)
    view.inference_precision = inference_precision
    receipt = {"schema": "joint_official_source_validation_v1", "source_task": task, "source": {k: v for k, v in spec.items() if k != "images"}, "scope": scope, "model_mode": model.mode, "args": args, "selection_rule": "Fixed epoch 40 primary; no task-combined score or checkpoint selection in this adapter", "precision_recall_convention": "Ultralytics maximum-mean-F1 development operating point; not fixed confidence 0.25", "adapter_sha256": sha_file(__file__)}
    try:
        with offline_network(), preserve_eval_state(model, verify_values=True) as guard:
            validator = AuditedValidator(dataloader=dataloader, save_dir=output, args=args)
            with torch.autocast(device_type=actual_device.type, dtype=torch.float16, enabled=inference_precision == "AMP_FP16"):
                metrics = validator(model=view)
            if view.forward_calls < 1:
                raise AssertionError("Official validator did not call the shared task view")
            expected_dtype = "torch.float16" if inference_precision == "AMP_FP16" else "torch.float32"
            if view.observed_conv_dtypes != {expected_dtype}:
                raise AssertionError(f"Actual convolution precision mismatch: {view.observed_conv_dtypes}")
            if validator.seen != expected_images or len(seen_paths) != expected_images or len(set(seen_paths)) != expected_images:
                raise AssertionError("Incomplete or duplicate source evaluation")
            if not set(seen_paths).issubset(set(spec["images"])):
                raise AssertionError("Validator consumed an image from a different source/split")
            if scope == "full_development" and Counter(seen_paths) != Counter(spec["images"]):
                raise AssertionError("Full development images were not covered exactly once")
            receipt.update(metrics={k: float(v) for k, v in metrics.items() if k != "fitness"}, per_class=_per_class(validator), seen=int(validator.seen), evaluated_image_paths=seen_paths, task_view_forward_calls=view.forward_calls, fusion_performed=False, noop_fuse_calls=view.noop_fuse_calls, contains_unfused_batchnorm=any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()), validator_class=base.__module__ + "." + base.__name__, validator_source_sha256=sha_file(sys.modules[base.__module__].__file__), model_reference_preserved=view.shared_model is model, inference_precision="FP32", state_guard=guard)
        receipt["inference_precision"] = inference_precision
        receipt["observed_conv_dtypes"] = sorted(view.observed_conv_dtypes)
        receipt["phase"] = "complete"
        save_json(output / "validation_receipt.json", receipt)
        return receipt
    except BaseException as exc:
        save_json(output / "FAILED.json", {**receipt, "phase": "failed", "exception": repr(exc), "partial_metrics_are_not_success": True})
        raise


class JointInferenceAdapter:
    """One common preprocess, one shared predict_both, native per-head Results.

    BGR uint8 HWC images are the only accepted public image input. Outputs are
    source-task dictionaries, not associated fruit grades. This adapter does
    not claim acquisition-to-actuator latency or fusion quality.
    """
    def __init__(self, model, *, imgsz=640, conf=0.25, iou=0.7, max_det=300, audit_values=False):
        if imgsz != 640 or not 0 <= conf <= 1 or not 0 <= iou <= 1 or max_det < 1:
            raise ValueError("Invalid fixed-640 inference parameters")
        self.model = model
        self.device = _device(model)
        self.audit_values = audit_values
        self.views = {task: TaskView(model, task) for task in TASK_TYPES}
        self.predictors = {}
        for task, cls in (("stage", SegmentationPredictor), ("defect", DetectionPredictor)):
            predictor = cls(overrides=dict(task=TASK_TYPES[task], mode="predict", imgsz=640, conf=conf, iou=iou, max_det=max_det, rect=False, quantize=32, retina_masks=True, save=False, save_txt=False, save_crop=False, show=False, verbose=False, augment=False, agnostic_nms=False, classes=None, compile=False, channels_last=False, project=str(ROOT / "runtime/joint_eval_v1/predict"), name=task, exist_ok=True))
            predictor.model = SimpleNamespace(names=self.views[task].names, end2end=self.views[task].end2end, fp16=False, format="pt", stride=int(self.views[task].stride.max()), dynamic=False)
            predictor.device = self.device
            predictor.imgsz = [640, 640]
            self.predictors[task] = predictor
        self.max_det = max_det
        self.last_audit = None

    def _tasks(self, tasks):
        tasks = tuple(tasks)
        if not tasks or len(set(tasks)) != len(tasks) or any(t not in TASK_TYPES for t in tasks):
            raise ValueError("Choose distinct stage/defect tasks")
        return tasks

    def predict_raw(self, rgb_tensor, *, tasks=("stage", "defect")):
        """Return actual native evaluation raw outputs keyed by source task."""
        tasks = self._tasks(tasks)
        if rgb_tensor.device != self.device or rgb_tensor.dtype != torch.float32 or rgb_tensor.ndim != 4 or tuple(rgb_tensor.shape[1:]) != (3, 640, 640) or not bool(torch.isfinite(rgb_tensor).all()) or float(rgb_tensor.min()) < 0 or float(rgb_tensor.max()) > 1:
            raise ValueError("Raw input must be normalized FP32 BCHW640 RGB on the model device")
        with preserve_eval_state(self.model, verify_values=self.audit_values) as guard:
            for task in tasks:
                self.views[task].set_head_attr(max_det=max(self.max_det, 300), agnostic_nms=False)
            output = self.model.predict_both(rgb_tensor) if len(tasks) == 2 else {tasks[0]: self.model.predict_source(rgb_tensor, tasks[0])}
            if set(output) != set(tasks):
                raise AssertionError("Raw task dictionary mismatch")
        self.last_audit = guard
        return output

    def predict(self, images_bgr, *, paths=None, tasks=("stage", "defect"), cpu=True):
        tasks = self._tasks(tasks)
        images = [images_bgr] if isinstance(images_bgr, np.ndarray) else list(images_bgr)
        if not images or any(not isinstance(im, np.ndarray) or im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3 or min(im.shape[:2]) < 1 for im in images):
            raise ValueError("Expected real BGR uint8 HWC images")
        paths = list(paths) if paths is not None else [f"frame_{i}" for i in range(len(images))]
        if len(paths) != len(images):
            raise ValueError("Path/image batch size mismatch")
        with preserve_eval_state(self.model, verify_values=self.audit_values) as guard:
            tensor = self.predictors[tasks[0]].preprocess(images)
            if tuple(tensor.shape[1:]) != (3, 640, 640):
                raise AssertionError("Common fixed-square preprocessing contract failed")
            for task in tasks:
                self.views[task].set_head_attr(max_det=max(self.max_det, 300), agnostic_nms=False)
            raw = self.model.predict_both(tensor) if len(tasks) == 2 else {tasks[0]: self.model.predict_source(tensor, tasks[0])}
            if set(raw) != set(tasks):
                raise AssertionError("Joint raw output keys mismatch")
            output = {}
            for task in tasks:
                predictor = self.predictors[task]
                predictor.batch = (paths, images, [""] * len(images))
                results = predictor.postprocess(raw[task], tensor, images)
                if len(results) != len(images):
                    raise AssertionError("Postprocess result batch count mismatch")
                output[task] = [r.cpu() for r in results] if cpu else results
        self.last_audit = guard
        return output
