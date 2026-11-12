"""One-session shared-trunk FP32 ONNX inference with frozen native postprocess.

Only CPU is exercised during engineering tests. CUDA is an explicit future
execution choice, requires CUDA EP, and never silently falls back. Provider
listing alone is not compute evidence: finish_profiling inspects Node events.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
JOINT_DIR = ROOT / "apple_research_v2/joint"
FROZEN_EVAL_FILE = JOINT_DIR / "joint_eval_v1.py"
INPUT_SHAPE = (1, 3, 640, 640)
OUTPUT_SHAPES = {"stage_detections": (1, 300, 38), "stage_prototypes": (1, 32, 160, 160), "defect_detections": (1, 300, 6)}
TASK_TYPES = {"stage": "segment", "defect": "detect"}
AUTHOR_NAMES = {"stage": {0: "immature apple", 1: "semi-mature apple", 2: "mature apple"}, "defect": {0: "class_0", 1: "class_1"}}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _f_path(path, *, within_root=False):
    p = Path(path).resolve()
    if p.drive.upper() != "F:" or (within_root and not p.is_relative_to(ROOT)):
        raise ValueError("All deployment paths must remain on F, outputs within the research root")
    return p


def _imports():
    # The frozen adapter routes all caches/settings to F before package imports.
    # No frozen file is changed, and no automatic installers are called here.
    sys.dont_write_bytecode = True
    if str(JOINT_DIR) not in sys.path:
        sys.path.insert(0, str(JOINT_DIR))
    import joint_eval_v1 as native
    import numpy as np
    import torch
    import onnxruntime as ort
    if ort.__version__ != "1.28.0":
        raise RuntimeError("This adapter is contracted against local ONNX Runtime 1.28.0")
    return native, np, torch, ort


def _decode_metadata(metadata):
    if metadata.get("schema") != "shared_dual_head_onnx_v1":
        raise ValueError("Unknown shared ONNX metadata schema")
    expected_text = {"shared_trunk": "true", "end2end": "true", "precision": "FP32", "imgsz": "640", "batch": "1", "extra_nms": "false"}
    if any(metadata.get(k) != v for k, v in expected_text.items()):
        raise ValueError("Only shared, native end2end, FP32 fixed640 batch1 without extra NMS is supported")
    parsed = dict(metadata)
    for key in ("task_names", "task_types", "input_shape", "output_names", "output_shapes"):
        try:
            parsed[key] = json.loads(metadata[key])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Missing/invalid JSON ONNX metadata: {key}") from exc
    names = {task: {int(k): str(v) for k, v in parsed["task_names"].get(task, {}).items()} for task in TASK_TYPES}
    if names != AUTHOR_NAMES or parsed["task_types"] != TASK_TYPES:
        raise ValueError("Task labels/types differ from the frozen source authorities")
    if parsed["input_shape"] != list(INPUT_SHAPE) or parsed["output_names"] != list(OUTPUT_SHAPES) or parsed["output_shapes"] != [list(v) for v in OUTPUT_SHAPES.values()]:
        raise ValueError("Fixed ONNX I/O metadata mismatch")
    if parsed.get("source_kind") not in {"canonical_untrained_fixture", "final40_ema"}:
        raise ValueError("Source must be an explicitly untrained fixture or completed final40 EMA")
    for key in ("source_sha256", "model_tensor_sha256", "frozen_joint_eval_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", parsed.get(key, "")):
            raise ValueError(f"Invalid source identity metadata: {key}")
    if parsed["frozen_joint_eval_sha256"] != sha_file(FROZEN_EVAL_FILE):
        raise ValueError("Frozen native preprocessing/postprocessing code differs from export contract")
    parsed["task_names"] = names
    return parsed


class ORTDualAdapter:
    """One common preprocess and one ORT invocation for both task outputs.

    predict accepts one real BGR uint8 image (or length-one list); predict_tensor
    accepts normalized contiguous FP32 RGB BCHW640. Results are independent task
    outputs, not matched fruit grades. Native empty-mask filtering is unchanged.
    CPU execution never requests CUDA or TensorRT. CUDA execution uses explicit
    device buffers with synchronization, and allows CPU graph shape operators
    while forbidding session/run fallback after an execution-provider failure.
    """
    def __init__(self, onnx_path, device="cpu", profile_dir=None, *, expected_sha256=None, conf=0.25, iou=0.7, max_det=300):
        self.path = _f_path(onnx_path)
        if not self.path.is_file() or self.path.suffix.lower() != ".onnx":
            raise ValueError("An existing local ONNX artifact is required")
        self.onnx_sha256 = sha_file(self.path)
        if expected_sha256 is not None and self.onnx_sha256 != expected_sha256:
            raise ValueError("ONNX artifact differs from the explicitly expected SHA")
        if not (0 <= conf <= 1 and 0 <= iou <= 1 and isinstance(max_det, int) and 1 <= max_det <= 300):
            raise ValueError("Invalid confidence/IoU/max_det inference settings")
        text_device = str(device)
        if text_device != "cpu" and not re.fullmatch(r"cuda:[0-9]+", text_device):
            raise ValueError("Choose cpu or an explicit cuda:<index>; no automatic provider selection")
        native, np, torch, ort = _imports()
        self._np, self._torch, self._ort, self._native = np, torch, ort, native
        self.device = torch.device(text_device)
        if self.device.type == "cpu" and torch.get_num_threads() > 2:
            torch.set_num_threads(2)
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.profile_dir = _f_path(profile_dir, within_root=True) if profile_dir is not None else None
        self._profile_closed = False
        self._profile_evidence = None
        self._lock = threading.RLock()
        self.last_raw = None
        self.last_preprocessed = None
        self.run_count = 0
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.profile_dir is not None:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            options.enable_profiling = True
            options.profile_file_prefix = str(self.profile_dir / "shared_ort_profile")
        if self.device.type == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("Explicit CUDA execution requested but CUDAExecutionProvider unavailable")
            providers = [("CUDAExecutionProvider", {"device_id": self.device.index, "use_tf32": 0}), "CPUExecutionProvider"]
        with native.offline_network():
            self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=providers, enable_fallback=False)
        self.session_identity = id(self.session)
        established = self.session.get_session_options()
        if established.intra_op_num_threads != 1 or established.inter_op_num_threads != 1:
            raise RuntimeError("Actual ORT CPU thread contract differs from requested 1/1")
        if self.device.type == "cpu" and torch.get_num_threads() > 2:
            raise RuntimeError("Torch postprocess exceeds the explicit two-thread CPU limit")
        self.actual_providers = self.session.get_providers()
        expected_providers = ["CPUExecutionProvider"] if self.device.type == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.actual_providers != expected_providers or getattr(self.session, "_enable_fallback", None) is not False:
            raise RuntimeError("Requested provider/fallback policy was not actually established")
        self.provider_options = self.session.get_provider_options()
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)" or tuple(inputs[0].shape) != INPUT_SHAPE:
            raise ValueError("Actual ONNX input is not fixed FP32 [1,3,640,640]")
        self.input_name = inputs[0].name
        if [o.name for o in outputs] != list(OUTPUT_SHAPES):
            raise ValueError("Actual ONNX output names/order mismatch")
        if any(o.type != "tensor(float)" or tuple(o.shape) != OUTPUT_SHAPES[o.name] for o in outputs):
            raise ValueError("Actual ONNX output shape/precision mismatch")
        self.metadata = _decode_metadata(self.session.get_modelmeta().custom_metadata_map)
        if self.device.type == "cuda":
            actual = self.provider_options.get("CUDAExecutionProvider", {})
            if str(actual.get("device_id")) != str(self.device.index) or str(actual.get("use_tf32")) != "0":
                raise RuntimeError("Actual CUDA provider index/TF32 options do not match the explicit FP32 contract")
        self.predictors = {}
        for task, cls in (("stage", native.SegmentationPredictor), ("defect", native.DetectionPredictor)):
            predictor = cls(overrides=dict(task=TASK_TYPES[task], mode="predict", imgsz=640, conf=conf, iou=iou, max_det=max_det, rect=False, quantize=32, retina_masks=True, save=False, save_txt=False, save_crop=False, show=False, verbose=False, augment=False, agnostic_nms=False, classes=None, compile=False, channels_last=False, project=str(ROOT / "runtime/deployment_shared_v1/predict"), name=task, exist_ok=True))
            # format=pt matches the frozen native preprocessing implementation;
            # rect=False makes the letterbox square and independent of backend.
            predictor.model = SimpleNamespace(names=self.metadata["task_names"][task], end2end=True, fp16=False, format="pt", stride=32, dynamic=False)
            predictor.device = self.device
            predictor.imgsz = [640, 640]
            self.predictors[task] = predictor

    def _assert_session(self):
        if self.device.type == "cpu" and self._torch.get_num_threads() > 2:
            raise RuntimeError("Torch CPU threads changed beyond the two-thread contract")
        if id(self.session) != self.session_identity or self.session.get_providers() != self.actual_providers or self.session.get_provider_options() != self.provider_options or getattr(self.session, "_enable_fallback", None) is not False:
            raise RuntimeError("Session identity/provider/fallback policy changed")

    def _validate_input(self, tensor):
        np, torch = self._np, self._torch
        if isinstance(tensor, np.ndarray):
            if self.device.type != "cpu" or tensor.dtype != np.float32 or not tensor.flags.c_contiguous:
                raise ValueError("NumPy input must be contiguous FP32 for explicit CPU mode")
            tensor = torch.from_numpy(tensor)
        if not isinstance(tensor, torch.Tensor) or tensor.device != self.device or tensor.dtype != torch.float32 or tuple(tensor.shape) != INPUT_SHAPE or not tensor.is_contiguous():
            raise ValueError("Expected contiguous normalized FP32 [1,3,640,640] RGB on the chosen device")
        if not bool(torch.isfinite(tensor).all()) or float(tensor.min()) < 0 or float(tensor.max()) > 1:
            raise ValueError("Input pixels must be finite and normalized to [0,1]")
        return tensor

    def _validate_outputs(self, outputs):
        torch = self._torch
        if list(outputs) != list(OUTPUT_SHAPES):
            raise AssertionError("ORT output mapping changed")
        for name, value in outputs.items():
            if value.dtype != torch.float32 or value.device != self.device or tuple(value.shape) != OUTPUT_SHAPES[name] or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid finite FP32 output: {name}")
        for name, nc in (("stage_detections", 3), ("defect_detections", 2)):
            pred = outputs[name]
            cls, scores = pred[..., 5], pred[..., 4]
            if not bool(((cls == cls.floor()) & (cls >= 0) & (cls < nc)).all()) or not bool(((scores >= 0) & (scores <= 1)).all()):
                raise ValueError("Output class IDs/scores violate the source task vocabulary")

    def predict_tensor(self, rgb_tensor):
        """Execute exactly one ORT call and return owned output tensors by name."""
        np, torch = self._np, self._torch
        with self._lock, torch.inference_mode():
            self._assert_session()
            tensor = self._validate_input(rgb_tensor)
            self.last_preprocessed = tensor
            if self.device.type == "cpu":
                arrays = self.session.run(list(OUTPUT_SHAPES), {self.input_name: tensor.detach().numpy()})
                output = {name: torch.from_numpy(array) for name, array in zip(OUTPUT_SHAPES, arrays)}
            else:
                # Explicit sync protects cross-library stream ownership. No CPU
                # input/output staging occurs; outputs are newly owned per call.
                torch.cuda.synchronize(self.device)
                binding = self.session.io_binding()
                binding.bind_input(name=self.input_name, device_type="cuda", device_id=self.device.index, element_type=np.float32, shape=INPUT_SHAPE, buffer_ptr=tensor.data_ptr())
                output = {name: torch.empty(shape, dtype=torch.float32, device=self.device) for name, shape in OUTPUT_SHAPES.items()}
                for name, value in output.items():
                    binding.bind_output(name=name, device_type="cuda", device_id=self.device.index, element_type=np.float32, shape=tuple(value.shape), buffer_ptr=value.data_ptr())
                binding.synchronize_inputs()
                self.session.run_with_iobinding(binding)
                binding.synchronize_outputs()
                torch.cuda.synchronize(self.device)
            self.run_count += 1
            self._validate_outputs(output)
            self._assert_session()
            self.last_raw = output
            return output

    def preprocess(self, images_bgr):
        """Public identical-to-native square640 BGR-to-RGB preprocessing."""
        np, torch = self._np, self._torch
        images = [images_bgr] if isinstance(images_bgr, np.ndarray) else list(images_bgr)
        if len(images) != 1 or any(not isinstance(im, np.ndarray) or im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3 or min(im.shape[:2]) < 1 for im in images):
            raise ValueError("Preprocess requires exactly one BGR uint8 HWC image")
        with self._lock, torch.inference_mode():
            tensor = self.predictors["stage"].preprocess(images)
            self.last_preprocessed = self._validate_input(tensor)
            return self.last_preprocessed

    def predict(self, images_bgr, *, paths=None, cpu=True):
        np, torch = self._np, self._torch
        images = [images_bgr] if isinstance(images_bgr, np.ndarray) else list(images_bgr)
        if len(images) != 1 or any(not isinstance(im, np.ndarray) or im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3 or min(im.shape[:2]) < 1 for im in images):
            raise ValueError("The exported graph accepts exactly one BGR uint8 HWC image")
        paths = ["frame_0"] if paths is None else ([str(paths)] if isinstance(paths, (str, Path)) else list(paths))
        if len(paths) != 1:
            raise ValueError("Exactly one path label is required")
        with self._lock, torch.inference_mode():
            tensor = self.preprocess(images)
            output = self.predict_tensor(tensor)
            # The exact native tuple structure is accepted by the official
            # postprocessor; raw training dictionaries are not consumed there.
            raw = {"stage": ((output["stage_detections"], output["stage_prototypes"]), {}), "defect": (output["defect_detections"], {})}
            results = {}
            for task in TASK_TYPES:
                predictor = self.predictors[task]
                predictor.batch = (paths, images, [""])
                value = predictor.postprocess(raw[task], tensor, images)
                if len(value) != 1:
                    raise AssertionError("Official postprocess returned wrong batch count")
                results[task] = [r.cpu() for r in value] if cpu else value
            return results

    def execution_record(self):
        self._assert_session()
        return {"onnx_path": str(self.path), "onnx_sha256": self.onnx_sha256, "onnxruntime_version": self._ort.__version__, "device": str(self.device), "actual_session_providers": self.actual_providers, "actual_provider_options": self.provider_options, "runtime_fallback_enabled": False, "tensorrt_requested": False, "run_count": self.run_count, "one_session_call_per_frame": True, "session_identity": self.session_identity, "intra_op_threads": self.session.get_session_options().intra_op_num_threads, "inter_op_threads": self.session.get_session_options().inter_op_num_threads, "torch_threads": self._torch.get_num_threads(), "precision": "FP32", "fixed_confidence": self.conf, "confidence_comparison": "native strict > threshold", "iou_argument": self.iou, "extra_nms": False, "retina_masks": True, "native_empty_mask_behavior": "unaltered official SegmentationPredictor filtering", "cuda_io_binding": self.device.type == "cuda", "cuda_sync_before_and_after_ort": self.device.type == "cuda", "source_kind": self.metadata["source_kind"], "source_sha256": self.metadata["source_sha256"], "model_tensor_sha256": self.metadata["model_tensor_sha256"], "frozen_joint_eval_sha256": self.metadata["frozen_joint_eval_sha256"], "profile_closed": self._profile_closed, "actual_cuda_compute_observed": bool(self._profile_evidence and self._profile_evidence["actual_cuda_compute_observed"]), "speed_benchmark_performed": False}

    def finish_profiling(self):
        """Stop profiling on this session and inspect actual execution events.

        The same session remains usable after profiling stops. CUDA succeeds as
        execution evidence only when real non-Memcpy CUDA Node kernels exist.
        """
        with self._lock:
            self._assert_session()
            if self.profile_dir is None:
                return {"status": "not_requested", "actual_cuda_compute_observed": False}
            if self._profile_closed:
                if self._profile_evidence is None:
                    raise RuntimeError("The previous profile verification failed; use a fresh profiled session")
                return self._profile_evidence
            profile = _f_path(self.session.end_profiling(), within_root=True)
            self._profile_closed = True
            if not profile.is_relative_to(self.profile_dir) or not profile.is_file():
                raise RuntimeError("ORT profile was not written inside the requested F directory")
            events = json.loads(profile.read_text(encoding="utf-8"))
            providers, cpu_ops, cuda_ops = Counter(), Counter(), Counter()
            kernel_rows = []
            for event in events:
                args = event.get("args", {})
                if event.get("cat") != "Node" or not str(event.get("name", "")).endswith("_kernel_time"):
                    continue
                provider, op = args.get("provider"), args.get("op_name")
                if not provider:
                    continue
                providers[provider] += 1
                if op and not str(op).lower().startswith("memcpy"):
                    if provider == "CUDAExecutionProvider":
                        cuda_ops[op] += 1
                    elif provider == "CPUExecutionProvider":
                        cpu_ops[op] += 1
                kernel_rows.append({"name": event.get("name"), "provider": provider, "op_name": op})
            cuda_observed = bool(cuda_ops)
            if self.device.type == "cuda" and not cuda_observed:
                raise RuntimeError("No actual CUDA compute kernels in profile; provider listing is insufficient")
            if self.device.type == "cpu" and any(p != "CPUExecutionProvider" for p in providers):
                raise RuntimeError("CPU-only session profile contains a non-CPU provider")
            if not kernel_rows or (self.device.type == "cpu" and not cpu_ops):
                raise RuntimeError("Profile contains no actual compute-node evidence")
            if sha_file(self.path) != self.onnx_sha256:
                raise RuntimeError("ONNX artifact changed while session ran")
            self._profile_evidence = {"status": "complete", "profile_path": str(profile), "profile_sha256": sha_file(profile), "session_identity": self.session_identity, "actual_providers": self.actual_providers, "provider_options": self.provider_options, "runtime_fallback_enabled": False, "kernel_events_by_provider": dict(providers), "cpu_compute_ops": dict(cpu_ops), "cuda_compute_ops": dict(cuda_ops), "actual_cuda_compute_observed": cuda_observed, "kernel_event_count": len(kernel_rows), "first_compute_events": kernel_rows[:20], "tensorrt_requested": False, "scope": "Provider execution evidence only; profile durations are not deployment throughput"}
            return self._profile_evidence
