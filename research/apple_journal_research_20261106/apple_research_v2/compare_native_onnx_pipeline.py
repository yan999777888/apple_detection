"""Fixed-threshold native-vs-ONNX output parity before synchronized pipeline timing.

All development images are compared; NO ground-truth labels are read. Native and
ONNX predictions use identical FP32 640 whole-frame preprocessing, confidence .25,
max_det 300, the checkpoint's original head, and native-resolution masks. Any count
change fails parity, including near-threshold changes. Tolerances are fixed below.
Timing runs only after parity passes. CPU mode performs parity only, no speed claim.
--check-only and --self-test do not import torch/ORT or run inference/write outputs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback

sys.dont_write_bytecode = True
from export_onnx_checked import (ROOT, ULTRALYTICS_VERSION, configure_runtime, fpath,
                                 normalized_names, now, require, sha, write_json)
from benchmark_yolo_pipeline import val_images, summarize

CONF = .25
MATCH_BOX_IOU = .99
BOX_ABS_TOL_PX = .1
SCORE_ABS_TOL = 1e-4
MIN_MASK_IOU = .999
BOUNDARY_SCORE_EPS = 1e-4
PROTOCOL = {"conf": CONF, "matching": "class-constrained Hungarian max box IoU; canonical coordinate/score/index order",
            "matching_box_iou_min": MATCH_BOX_IOU, "box_coordinate_absolute_tolerance_px": BOX_ABS_TOL_PX,
            "score_absolute_tolerance": SCORE_ABS_TOL, "mask_iou_min": MIN_MASK_IOU,
            "confidence_boundary_epsilon": BOUNDARY_SCORE_EPS, "count_changes_allowed": False,
            "class_count_changes_allowed": False, "threshold_adaptation_allowed": False,
            "imgsz": 640, "batch": 1, "quantize": 32, "retina_masks": True,
            "rect": False, "max_det": 300, "iou": .7, "augment": False,
            "timing_distinct_frames": 40, "timing_repeats": 3, "warmup_each_backend": 5}


def array_sha(array):
    return hashlib.sha256(memoryview(array)).hexdigest()


def box_ious(a, b, np):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    left = np.maximum(a[:, None, :2], b[None, :, :2])
    right = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    intersection = np.maximum(right-left, 0).prod(axis=2)
    area_a = np.maximum(a[:, 2:4]-a[:, :2], 0).prod(axis=1)
    area_b = np.maximum(b[:, 2:4]-b[:, :2], 0).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def compare_outputs(native, exported, np, linear_sum_assignment):
    a, am = native
    b, bm = exported
    issues, matches = [], []
    counts_a = Counter(int(v) for v in a[:, 5])
    counts_b = Counter(int(v) for v in b[:, 5])
    if len(a) != len(b):
        issues.append("detection_count_changed")
    if counts_a != counts_b:
        issues.append("class_counts_changed")
    def boundary(boxes):
        return [{"index": i, "class": int(row[5]), "score": float(row[4])}
                for i, row in enumerate(boxes) if abs(float(row[4])-CONF) <= BOUNDARY_SCORE_EPS]
    for cls in sorted(set(counts_a) | set(counts_b)):
        def ordered(boxes):
            return sorted([i for i, row in enumerate(boxes) if int(row[5]) == cls],
                          key=lambda i: (*[float(v) for v in boxes[i, :5]], i))
        ai, bi = ordered(a), ordered(b)
        if not ai or not bi:
            continue
        ious = box_ious(a[ai, :4].astype(np.float64), b[bi, :4].astype(np.float64), np)
        costs = np.where(ious >= MATCH_BOX_IOU, 1-ious, 1e6)
        rows, cols = linear_sum_assignment(costs)
        for r, c in zip(rows, cols):
            i, j = ai[int(r)], bi[int(c)]
            box_iou = float(ious[r, c])
            coord_error = float(np.max(np.abs(a[i, :4]-b[j, :4])))
            score_error = abs(float(a[i, 4])-float(b[j, 4]))
            mask_iou, xor_pixels = None, None
            if am is not None and bm is not None:
                av, bv = am[i].astype(bool), bm[j].astype(bool)
                union = int(np.logical_or(av, bv).sum())
                intersection = int(np.logical_and(av, bv).sum())
                mask_iou = intersection/union if union else 1.
                xor_pixels = union-intersection
            failures = []
            if box_iou < MATCH_BOX_IOU:
                failures.append("matching_iou")
            if coord_error > BOX_ABS_TOL_PX:
                failures.append("box_coordinate")
            if score_error > SCORE_ABS_TOL:
                failures.append("score")
            if mask_iou is not None and mask_iou < MIN_MASK_IOU:
                failures.append("mask_iou")
            matches.append({"native_index": i, "onnx_index": j, "class": cls,
                            "box_iou": box_iou, "max_box_abs_error_px": coord_error,
                            "score_abs_error": score_error, "mask_iou": mask_iou,
                            "mask_xor_pixels": xor_pixels, "failures": failures})
            if failures:
                issues.append("matched_output_tolerance_failed")
    return {"passed": not issues, "issues": sorted(set(issues)),
            "native_count": len(a), "onnx_count": len(b),
            "native_class_counts": dict(counts_a), "onnx_class_counts": dict(counts_b),
            "native_threshold_boundary": boundary(a), "onnx_threshold_boundary": boundary(b),
            "native_boxes_xyxy_score_class": a.tolist(), "onnx_boxes_xyxy_score_class": b.tolist(),
            "matches": matches}


def self_test():
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    boxes = np.array([[1, 1, 9, 9, .8, 0], [11, 11, 19, 19, .7, 0]], dtype=np.float32)
    masks = np.zeros((2, 24, 24), dtype=np.uint8)
    masks[0, 1:9, 1:9] = 1
    masks[1, 11:19, 11:19] = 1
    compare = lambda a, b: compare_outputs(a, b, np, linear_sum_assignment)
    require(compare((boxes, masks), (boxes[::-1], masks[::-1]))["passed"], "Permutation matching failed")
    bad = boxes.copy(); bad[0, 4] += .01
    require(not compare((boxes, masks), (bad, masks))["passed"], "Score tolerance failed to reject")
    bad = boxes.copy(); bad[0, 0] += .2
    require(not compare((boxes, masks), (bad, masks))["passed"], "Box tolerance failed to reject")
    bad = masks.copy(); bad[0, 1, 1] = 0
    require(not compare((boxes, masks), (boxes, bad))["passed"], "Mask tolerance failed to reject")
    bad = boxes.copy(); bad[0, 5] = 1
    require(not compare((boxes, masks), (bad, masks))["passed"], "Class mismatch failed to reject")
    boundary = boxes[:1].copy(); boundary[0, 4] = CONF + .00001
    result = compare((boundary, masks[:1]), (boxes[:0], masks[:0]))
    require(not result["passed"] and len(result["native_threshold_boundary"]) == 1,
            "Boundary count change was not explicitly failed/reported")
    require(compare((boxes[:0], masks[:0]), (boxes[:0], masks[:0]))["passed"], "Empty outputs should match")
    print(json.dumps({"phase": "self_test_passed", "checks": 7,
                      "torch_imported": "torch" in sys.modules, "gpu_used": False, "files_written": False}))


def preflight(args):
    import cv2
    import numpy as np
    receipt_path = fpath(args.export_receipt)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    require(receipt.get("phase") == "complete", "ONNX export receipt is not complete")
    native, onnx_path, data_path = map(fpath, (args.native, args.onnx, args.data))
    require(sha(native) == receipt["source_sha256"], "Native checkpoint does not match exported checkpoint SHA")
    require(sha(onnx_path) == receipt["onnx_sha256"], "ONNX bytes differ from checked export receipt")
    require(receipt["checkpoint_task"] in {"segment", "detect"}, "Unsupported checkpoint task")
    inspection = receipt["onnx_inspection"]
    require(inspection["checker_full_check_passed"] and inspection["input"]["shape"] == [1, 3, 640, 640],
            "Missing valid fixed-shape ONNX evidence")
    require(normalized_names(receipt["checkpoint_names"]) == normalized_names(
            __import__("ast").literal_eval(inspection["metadata"]["names"])), "Receipt class-name mismatch")
    output = fpath(args.output, exists=False)
    require(output.is_relative_to(ROOT) and output != ROOT and not output.exists(), "Fresh output under research root required")
    require(args.device == "cpu" or re.fullmatch(r"(?:cuda:)?\d+", args.device), "Device must be cpu or one CUDA ordinal")
    images = val_images(data_path)
    require(len(images) == args.expected_val_count, "Validation image count differs from frozen protocol")
    require(len(images) >= 40, "Need 40 distinct validation frames for timing")
    frames, records = [], []
    for path in images:
        raw = path.read_bytes()
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        require(frame is not None and frame.dtype == np.uint8 and frame.flags.c_contiguous,
                f"Invalid decoded BGR image: {path}")
        frames.append(frame)
        records.append({"path": str(path), "image_sha256": hashlib.sha256(raw).hexdigest(),
                        "decoded_bgr_sha256": array_sha(frame), "shape": list(frame.shape)})
    return native, onnx_path, output, receipt, frames, {
        "export_receipt": str(receipt_path), "export_receipt_sha256": sha(receipt_path),
        "native": str(native), "native_sha256": sha(native), "onnx": str(onnx_path), "onnx_sha256": sha(onnx_path),
        "data_yaml": str(data_path), "data_yaml_sha256": sha(data_path), "split": "val", "images": records,
        "ground_truth_labels_read": False, "all_validation_images_included": True,
        "source_sha256": {str(Path(__file__).resolve()): sha(__file__),
            str(Path(__file__).with_name("export_onnx_checked.py")): sha(Path(__file__).with_name("export_onnx_checked.py")),
            str(Path(__file__).with_name("benchmark_yolo_pipeline.py")): sha(Path(__file__).with_name("benchmark_yolo_pipeline.py"))}}


@contextmanager

def forced_ort_session(ort, onnx_path, output, device, sessions):
    original = ort.InferenceSession
    def create(weight, *positional, **kwargs):
        require(Path(weight).resolve() == onnx_path, "Unexpected model requested by ONNX backend")
        require(not positional, "Unexpected positional ORT session options")
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(output / "actual_wrapper_profile")
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"] if device == "cpu" else [
            ("CUDAExecutionProvider", {"device_id": int(device.split(":")[-1]), "use_tf32": 0}), "CPUExecutionProvider"]
        kwargs.update(sess_options=options, providers=providers, enable_fallback=False)
        session = original(weight, **kwargs)
        require(device == "cpu" or "CUDAExecutionProvider" in session.get_providers(), "CUDA provider initialization failed")
        session.disable_fallback()
        sessions.append(session)
        return session
    ort.InferenceSession = create
    try:
        yield
    finally:
        ort.InferenceSession = original


def materialize(model, frame, options, np):
    results = model.predict(source=frame, stream=False, **options)
    require(len(results) == 1 and results[0].boxes is not None, "Expected one frame with box output")
    result = results[0]
    boxes = np.array(result.boxes.data.detach().cpu().numpy(), copy=True, order="C")
    if model.task == "segment":
        masks = np.empty((0, *frame.shape[:2]), dtype=np.uint8) if result.masks is None else np.array(
            result.masks.data.detach().cpu().numpy(), copy=True, order="C")
    else:
        masks = None
    return boxes, masks


def validate_output(output, frame, np, names):
    boxes, masks = output
    require(boxes.ndim == 2 and boxes.shape[1] == 6 and np.isfinite(boxes).all(), "Invalid box output")
    require(all(float(row[5]).is_integer() and int(row[5]) in names for row in boxes), "Invalid class IDs")
    require(np.all(boxes[:, 4] >= CONF), "Returned detection below fixed confidence threshold")
    if masks is not None:
        require(masks.shape == (len(boxes), *frame.shape[:2]), "Native-resolution mask shape mismatch")
        require(np.isfinite(masks).all() and np.all((masks == 0) | (masks == 1)), "Masks must be finite binary planes")


def profile_evidence(session, output, cuda_expected):
    profile = fpath(session.end_profiling())
    events = json.loads(profile.read_text(encoding="utf-8"))
    rows = [e for e in events if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")
            and e.get("args", {}).get("provider") and e.get("args", {}).get("op_name")]
    providers, ops = Counter(), {}
    for e in rows:
        provider, op = e["args"]["provider"], e["args"]["op_name"]
        providers[provider] += 1
        ops.setdefault(provider, Counter()).update([op])
    cuda_rows = [e for e in rows if e["args"]["provider"] == "CUDAExecutionProvider"
                 and not e["args"]["op_name"].lower().startswith("memcpy")]
    result = {"session_object_id": id(session), "registered_providers": session.get_providers(),
        "provider_options": session.get_provider_options(), "executed_node_events_by_provider": dict(providers),
        "executed_ops_by_provider": {k: dict(v) for k, v in ops.items()},
        "actual_cuda_compute_observed": bool(cuda_rows), "all_compute_on_gpu_claimed": False,
        "session_fallback_enabled": False, "profile": str(profile), "profile_sha256": sha(profile),
        "profiling_ended_before_timing": True, "tensorrt_requested": False}
    write_json(output / "provider_execution_evidence.json", result)
    require(not cuda_expected or cuda_rows, "No actual CUDA computation in wrapper profile")
    return result


def run(args):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    native_path, onnx_path, output, receipt, frames, manifest = preflight(args)
    if args.check_only:
        print(json.dumps({"phase": "preflight_passed", "images": len(frames), "gpu_used": False,
            "torch_imported": "torch" in sys.modules, "files_written": False,
            "protocol": PROTOCOL, "source_binding": {k: manifest[k] for k in ("native_sha256", "onnx_sha256")}}))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    environment = configure_runtime(output)
    report = {"schema": "native_onnx_pipeline_parity_v1", "started_utc": now(), "protocol": PROTOCOL,
              "manifest": manifest, "environment": environment, "parity_rows": [], "timing_rows": [],
              "is_robot_end_to_end_measurement": False, "quality_against_ground_truth_evaluated": False,
              "parity_is_deployment_equivalence_not_accuracy": True}
    session, profiled = None, False
    try:
        import torch
        import ultralytics
        import onnxruntime as ort
        import scipy
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS, downloads
        require(ultralytics.__version__ == ULTRALYTICS_VERSION, "Unexpected Ultralytics version")
        SETTINGS.update({"sync": False, "clearml": False, "comet": False, "dvc": False, "mlflow": False,
            "neptune": False, "raytune": False, "tensorboard": False, "wandb": False,
            "weights_dir": str(output), "runs_dir": str(output / "runs"), "datasets_dir": str(output / "datasets")})
        downloads.attempt_download_asset = lambda file, *a, **kw: str(fpath(file))
        def no_download(*a, **kw):
            raise RuntimeError("Automatic download forbidden")
        downloads.safe_download = no_download
        torch.set_num_threads(4)
        cuda = args.device != "cpu"
        device = "cpu" if not cuda else "cuda:" + args.device.split(":")[-1]
        if cuda:
            require(torch.cuda.is_available(), "CUDA unavailable; no automatic CPU fallback")
            torch.cuda.set_device(torch.device(device))
        sync = (lambda: torch.cuda.synchronize(torch.device(device))) if cuda else (lambda: None)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        names = normalized_names(receipt["checkpoint_names"])
        native = YOLO(str(native_path))
        native.model.to(device).float().eval()
        exported = YOLO(str(onnx_path), task=receipt["checkpoint_task"])
        require(native.task == receipt["checkpoint_task"] and normalized_names(native.names) == names,
                "Native task/classes differ from export receipt")
        options = {"device": device, "imgsz": 640, "batch": 1, "quantize": 32, "conf": CONF,
            "iou": .7, "max_det": 300, "rect": False, "retina_masks": True,
            "end2end": bool(receipt["checkpoint_end2end"]), "augment": False,
            "verbose": False, "save": False, "save_txt": False, "save_conf": False,
            "save_crop": False, "show": False, "project": str(output / "predict"), "name": "unsaved"}
        sessions = []
        with torch.inference_mode():
            # Session initialization is outside both parity and timing statistics.
            with forced_ort_session(ort, onnx_path, output, device, sessions):
                initial = materialize(exported, frames[0], options, np)
            del initial
            require(len(sessions) == 1, "Unexpected ORT session count")
            session = sessions[0]
            backend = exported.predictor.model.backend
            require(backend.session is session and not backend.fp16, "Wrapper is not using the audited FP32 session")
            require(backend.device.type == ("cuda" if cuda else "cpu"), "Wrapper device differs from request")
            require(normalized_names(exported.names) == names, "ONNX wrapper class mapping changed")
            for index, frame in enumerate(frames):
                a = materialize(native, frame, options, np)
                b = materialize(exported, frame, options, np)
                sync()
                validate_output(a, frame, np, names)
                validate_output(b, frame, np, names)
                # Compare the actual predictor preprocessing tensors, outside timing.
                na = native.predictor.preprocess([frame]).detach().cpu().numpy()
                nb = exported.predictor.preprocess([frame]).detach().cpu().numpy()
                same_input = na.shape == nb.shape == (1, 3, 640, 640) and na.dtype == nb.dtype == np.float32 and np.array_equal(na, nb)
                row = compare_outputs(a, b, np, linear_sum_assignment)
                if not same_input:
                    row["passed"] = False
                    row["issues"].append("preprocessing_tensors_differ")
                row.update(frame_index=index, image=manifest["images"][index]["path"],
                           preprocessing_equal=bool(same_input), preprocessing_sha256=array_sha(na))
                report["parity_rows"].append(row)
                del a, b, na, nb
                if (index+1) % 12 == 0:
                    print(json.dumps({"phase": "parity_progress", "frames": index+1,
                        "failed_frames": sum(not r["passed"] for r in report["parity_rows"])}), flush=True)
            passed = all(row["passed"] for row in report["parity_rows"])
            report["parity_summary"] = {"all_images": len(frames), "passed": passed,
                "failed_frames": sum(not row["passed"] for row in report["parity_rows"]),
                "native_detections": sum(row["native_count"] for row in report["parity_rows"]),
                "onnx_detections": sum(row["onnx_count"] for row in report["parity_rows"]),
                "count_change_frames": sum(row["native_count"] != row["onnx_count"] for row in report["parity_rows"]),
                "native_boundary_detections": sum(len(row["native_threshold_boundary"]) for row in report["parity_rows"]),
                "onnx_boundary_detections": sum(len(row["onnx_threshold_boundary"]) for row in report["parity_rows"])}
            for model in (native, exported):
                require(not model.predictor.model.fp16 and model.predictor.args.quantize == 32, "FP32 prediction contract changed")
                require(bool(model.predictor.model.end2end) == bool(receipt["checkpoint_end2end"]), "Inference head changed")
            timing_allowed = passed and cuda and not args.parity_only
            if timing_allowed:
                for model in (native, exported):
                    for i in range(5):
                        warm = materialize(model, frames[i], options, np)
                        sync()
                        del warm
            evidence = profile_evidence(session, output, cuda)
            profiled = True
            report["provider_evidence"] = evidence
            if timing_allowed:
                selection = random.Random(20261106).sample(range(len(frames)), 40)
                report["timing_frame_indices"] = selection
                report["timing_scope"] = {
                    "included": "whole-frame YOLO.predict preprocessing/H2D/forward/postprocessing; all native-resolution masks and boxes copied to independent CPU arrays; CUDA synchronization",
                    "excluded": ["disk and image decode", "model/session initialization", "parity checks", "preprocessing parity copies", "profile collection", "drawing", "tracking", "camera", "robot control", "report serialization"],
                    "both_models_resident": True, "execution": "serial, alternating backend order by repetition",
                    "other_gpu_workloads_verified_absent": False, "capture_to_output_age": None,
                    "fps_definition": "frames / sum measured service wall times, not entire-loop throughput"}
                for repeat in range(3):
                    order = [("native", native), ("onnx", exported)]
                    if repeat % 2:
                        order.reverse()
                    for index in selection:
                        for kind, model in order:
                            sync()
                            started = time.perf_counter_ns()
                            prediction = materialize(model, frames[index], options, np)
                            sync()
                            elapsed = time.perf_counter_ns()-started
                            boxes, masks = prediction
                            report["timing_rows"].append({"backend": kind, "repeat": repeat,
                                "frame_index": index, "elapsed_ns": elapsed, "elapsed_ms": elapsed/1e6,
                                "detections": len(boxes), "cpu_output_bytes": int(boxes.nbytes+(masks.nbytes if masks is not None else 0))})
                            del prediction, boxes, masks
                    print(json.dumps({"phase": "timing_repeat_complete", "repeat": repeat+1}), flush=True)
                report["timing_summary"] = {kind: summarize([r["elapsed_ms"] for r in report["timing_rows"] if r["backend"] == kind])
                                             for kind in ("native", "onnx")}
                require(backend.session is session and id(session) == evidence["session_object_id"], "Timed ORT session changed")
                require(session.get_providers() == evidence["registered_providers"] and session.get_provider_options() == evidence["provider_options"], "Timed ORT providers changed")
                require(sha(evidence["profile"]) == evidence["profile_sha256"], "Profile changed after end_profiling")
            else:
                report["timing_not_run_reason"] = "parity_failed" if not passed else "CPU_or_explicit_parity_only"
        report["runtime"] = {"torch": str(torch.__version__), "ultralytics": ultralytics.__version__,
            "onnxruntime": ort.__version__, "scipy": scipy.__version__, "device": device,
            "ort_use_io_binding": backend.use_io_binding, "actual_onnx_providers": session.get_providers(),
            "predict_options": options, "native_parameter_dtypes": sorted({str(p.dtype) for p in native.model.parameters()}),
            "cuda_gpu": torch.cuda.get_device_name(torch.device(device)) if cuda else None}
        for key, hash_key in (("native", "native_sha256"), ("onnx", "onnx_sha256"), ("data_yaml", "data_yaml_sha256"), ("export_receipt", "export_receipt_sha256")):
            require(sha(manifest[key]) == manifest[hash_key], f"Input changed: {key}")
        for record, frame in zip(manifest["images"], frames):
            require(sha(record["path"]) == record["image_sha256"] and array_sha(frame) == record["decoded_bgr_sha256"], "Source image or decoded pixels changed")
        for path, digest in manifest["source_sha256"].items():
            require(sha(path) == digest, "Script/helper source changed")
        report.update(phase="complete" if passed else "parity_failed", completed_utc=now(), inputs_unchanged=True)
        write_json(output / "parity_benchmark.json", report)
        print(json.dumps({"phase": report["phase"], "report": str(output / "parity_benchmark.json"),
                          "parity": report["parity_summary"], "timing_performed": bool(report["timing_rows"])}, ensure_ascii=False))
        return 0 if passed else 2
    except BaseException as error:
        if session is not None and not profiled:
            try:
                report["failure_profile_path"] = session.end_profiling()
            except Exception:
                pass
        report.update(phase="failed", failed_utc=now(), error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        write_json(output / "failure.json", report)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path)
    p.add_argument("--onnx", type=Path)
    p.add_argument("--export-receipt", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--expected-val-count", type=int, default=72)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--parity-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    for name in ("native", "onnx", "export_receipt", "data", "output"):
        if getattr(args, name) is None:
            p.error("--"+name.replace("_", "-")+" is required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
