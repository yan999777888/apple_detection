"""Export a completed, SHA-bound YOLO checkpoint to checked FP32 ONNX on CPU.

The source checkpoint is copied into a fresh isolated F-drive directory. This
script never overwrites trained weights, installs packages, or downloads assets.
--check-only imports no torch/ONNX/ORT and writes nothing. Optional --verify-cuda
executes only a synthetic ORT provider smoke test, NOT a speed/accuracy benchmark.
Native-vs-ONNX boxes/masks and complete deployment timing require a separate test.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_VERSION = "8.4.115"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fpath(value, *, exists=True):
    path = Path(value).resolve()
    require(path.drive.upper() == "F:", f"All file inputs/outputs must stay on F: {path}")
    if exists:
        require(path.exists(), f"Missing local file: {path}")
    return path


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with fpath(path, exists=False).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def dependency_versions():
    result = {}
    for name in ("ultralytics", "torch", "numpy", "onnx", "onnxruntime", "onnxruntime-gpu"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    require(result["ultralytics"] == ULTRALYTICS_VERSION, "Export requires audited Ultralytics 8.4.115")
    for name in ("torch", "numpy", "onnx"):
        require(result[name] is not None, f"Missing {name}; automatic installation is forbidden")
    return result


def preflight(args):
    require(ROOT.drive.upper() == "F:", "Run the script from its F-drive project")
    source = fpath(args.weights)
    status_path = fpath(args.training_status)
    require(source.suffix.lower() == ".pt" and source.is_file(), "Existing local .pt required")
    require(source.is_relative_to(status_path.parent), "Checkpoint must belong to the completed training run")
    status = json.loads(status_path.read_text(encoding="utf-8-sig"))
    require(status.get("phase") == "complete", "Training is not complete; refuse live checkpoint export")
    selection_path = status_path.parent / "final_selection.json"
    require(selection_path.is_file(), "Completed run needs final_selection.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    digest = sha(source)
    require(digest == args.expected_weights_sha256.lower(), "Source checkpoint SHA256 mismatch")
    matched = [role for role in ("primary", "secondary")
               if selection.get(role, {}).get("sha256") == digest]
    require(matched, "Checkpoint is not SHA-bound to the completed final selection")
    output = fpath(args.output, exists=False)
    require(output.is_relative_to(ROOT) and output != ROOT, "Output must be a fresh child of the research root")
    require(not output.exists(), "Output already exists; refusing overwrite")
    require(0 <= args.cuda_device < 16, "Invalid CUDA ordinal")
    versions = dependency_versions()
    if args.verify_cuda:
        require(importlib.util.find_spec("onnxruntime") is not None, "ORT is not installed; no installation attempted")
    return source, output, {
        "source_checkpoint": str(source), "source_sha256": digest,
        "training_status": str(status_path), "training_status_sha256": sha(status_path),
        "final_selection": str(selection_path), "final_selection_sha256": sha(selection_path),
        "checkpoint_selection_roles": matched, "training_epochs_completed": status.get("epochs_completed"),
        "source_script_sha256": sha(__file__), "installed_versions": versions,
    }


def configure_runtime(output):
    runtime = output / "runtime"
    folders = {"TEMP": "tmp", "TMP": "tmp", "YOLO_CONFIG_DIR": "ultralytics",
               "MPLCONFIGDIR": "matplotlib", "TORCH_HOME": "torch", "HF_HOME": "hf",
               "XDG_CACHE_HOME": "cache", "CUDA_CACHE_PATH": "cuda",
               "TORCH_EXTENSIONS_DIR": "torch_extensions", "PIP_CACHE_DIR": "pip"}
    for key, suffix in folders.items():
        directory = runtime / suffix
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", YOLO_OFFLINE="1",
                      YOLO_AUTOINSTALL="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      WANDB_MODE="disabled", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    os.chdir(output)
    def no_network(*a, **kw):
        raise RuntimeError("Network disabled: this export uses installed packages and local weights only")
    socket.create_connection = no_network
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    return {key: os.environ[key] for key in folders}


def normalized_names(names):
    if isinstance(names, list):
        names = dict(enumerate(names))
    require(isinstance(names, dict) and names, "Missing class names")
    result = {int(key): str(value) for key, value in names.items()}
    require(sorted(result) == list(range(len(result))), "Noncontiguous class IDs")
    return result


def graph_tensors(onnx, graph, prefix="graph"):
    for tensor in graph.initializer:
        yield prefix + "/initializer/" + tensor.name, tensor
    for tensor in graph.sparse_initializer:
        yield prefix + "/sparse/" + tensor.values.name, tensor.values
    for index, node in enumerate(graph.node):
        for attribute in node.attribute:
            place = f"{prefix}/node{index}/{attribute.name}"
            if attribute.type == onnx.AttributeProto.TENSOR:
                yield place, attribute.t
            elif attribute.type == onnx.AttributeProto.TENSORS:
                for item in attribute.tensors:
                    yield place + "/" + item.name, item
            elif attribute.type == onnx.AttributeProto.GRAPH:
                yield from graph_tensors(onnx, attribute.g, place)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for number, subgraph in enumerate(attribute.graphs):
                    yield from graph_tensors(onnx, subgraph, place + f"/{number}")


def inspect_onnx(path, expected_task, expected_names, expected_end2end):
    import numpy as np
    import onnx
    graph_model = onnx.load(str(path), load_external_data=False)
    types = Counter()
    floating = 0
    for name, tensor in graph_tensors(onnx, graph_model.graph):
        require(not tensor.external_data and tensor.data_location != onnx.TensorProto.EXTERNAL,
                f"External tensor data is not allowed in this standalone export: {name}")
        require(tensor.data_type not in {onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16},
                f"Unexpected low precision tensor in FP32 export: {name}")
        array = onnx.numpy_helper.to_array(tensor)
        types[str(array.dtype)] += 1
        if np.issubdtype(array.dtype, np.floating):
            floating += 1
            require(np.isfinite(array).all(), f"Nonfinite initializer/constant: {name}")
    onnx.checker.check_model(graph_model, full_check=True)
    require(len(graph_model.graph.input) == 1, "Exactly one ONNX input expected")
    input_ = graph_model.graph.input[0]
    shape = [dimension.dim_value for dimension in input_.type.tensor_type.shape.dim]
    require(input_.name == "images" and shape == [1, 3, 640, 640], "Input must be fixed [1,3,640,640]")
    require(input_.type.tensor_type.elem_type == onnx.TensorProto.FLOAT, "Input must be FP32")
    metadata = {item.key: item.value for item in graph_model.metadata_props}
    require(metadata.get("task") == expected_task, "Exported task differs from checkpoint")
    require(normalized_names(ast.literal_eval(metadata["names"])) == expected_names,
            "Exported class-name mapping differs from checkpoint")
    require(ast.literal_eval(metadata["imgsz"]) == [640, 640] and int(metadata["batch"]) == 1,
            "Export metadata has wrong input size/batch")
    require(ast.literal_eval(metadata["end2end"]) == expected_end2end, "Export changed inference head")
    export_args = ast.literal_eval(metadata["args"])
    require(export_args.get("quantize") == 32 and export_args.get("simplify") is False
            and export_args.get("dynamic") is False and export_args.get("nms") is False,
            "Export metadata differs from fixed FP32/no-simplification recipe")
    outputs = graph_model.graph.output
    require(len(outputs) == (2 if expected_task == "segment" else 1), "Unexpected output arity")
    require(all(item.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for item in outputs),
            "All outputs must be FP32")
    opsets = {item.domain or "ai.onnx": item.version for item in graph_model.opset_import}
    require(opsets.get("ai.onnx") == 17, "Unexpected ONNX opset")
    return {"checker_full_check_passed": True, "all_floating_initializers_and_constants_finite": True,
            "floating_tensors_checked": floating, "tensor_dtype_counts": dict(types),
            "opsets": opsets, "ir_version": graph_model.ir_version, "metadata": metadata,
            "input": {"name": input_.name, "shape": shape, "dtype": "float32"},
            "output_names": [item.name for item in outputs], "node_count": len(graph_model.graph.node)}


def verify_cuda_execution(onnx_path, output, device_id=0):
    """Synthetic execution proof with raw profile; deliberately reports no FPS.

    Profiling session uses explicit CUDA + CPU EPs, disables session-level fallback
    at construction, and reports CPU node assignments. A provider list alone does
    not prove CUDA execution. No TensorRT provider is requested.
    """
    import numpy as np
    import torch  # load installed matching CUDA/cuDNN DLLs on Windows, no install
    import onnxruntime as ort
    require("CUDAExecutionProvider" in ort.get_available_providers(), "Installed ORT has no CUDA EP")
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(output / "ort_cuda_profile")
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options,
        providers=[("CUDAExecutionProvider", {"device_id": device_id, "use_tf32": 0}), "CPUExecutionProvider"],
        enable_fallback=False)
    registered = session.get_providers()
    require("CUDAExecutionProvider" in registered, "CUDA EP initialization did not succeed")
    session.disable_fallback()
    feed = np.random.default_rng(20261106).random((1, 3, 640, 640), dtype=np.float32)
    output_shapes = None
    try:
        for _ in range(3):
            outputs = session.run(None, {"images": feed})
            require(all(np.isfinite(value).all() for value in outputs), "Nonfinite ORT output")
            output_shapes = [list(value.shape) for value in outputs]
    finally:
        profile = fpath(session.end_profiling())
    events = json.loads(profile.read_text(encoding="utf-8"))
    nodes = [event for event in events if event.get("cat") == "Node"
             and event.get("name", "").endswith("_kernel_time")
             and event.get("args", {}).get("provider") and event.get("args", {}).get("op_name")]
    providers = Counter(event["args"]["provider"] for event in nodes)
    provider_ops = {}
    for event in nodes:
        provider_ops.setdefault(event["args"]["provider"], Counter()).update([event["args"]["op_name"]])
    cuda_compute = [event for event in nodes if event["args"]["provider"] == "CUDAExecutionProvider"
                    and not event["args"]["op_name"].lower().startswith("memcpy")]
    evidence = {"onnxruntime_version": ort.__version__, "torch_version": str(torch.__version__),
        "available_providers": ort.get_available_providers(), "registered_providers": registered,
        "provider_options": session.get_provider_options(), "session_fallback_enabled": False,
        "executed_node_events_by_provider": dict(providers),
        "executed_ops_by_provider": {key: dict(value) for key, value in provider_ops.items()},
        "actual_cuda_compute_observed": bool(cuda_compute),
        "all_compute_on_gpu_claimed": False, "cpu_partitioning_is_allowed_and_reported": True,
        "synthetic_input_only": True, "inference_runs": 3, "outputs_finite": True,
        "output_shapes": output_shapes, "profile": str(profile), "profile_sha256": sha(profile),
        "native_output_parity_evaluated": False, "speed_benchmark_performed": False,
        "tensorrt_provider_requested": False}
    write_json(output / "ort_cuda_execution_evidence.json", evidence)
    require(cuda_compute, "Profile contains no CUDA compute node; actual CUDA execution is unproven")
    return evidence


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--expected-weights-sha256", required=True)
    p.add_argument("--training-status", type=Path, required=True)
    p.add_argument("--task", choices=("detect", "segment"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--verify-cuda", action="store_true")
    p.add_argument("--cuda-device", type=int, default=0)
    args = p.parse_args(argv)
    source, output, receipt = preflight(args)
    if args.check_only:
        print(json.dumps({"phase": "preflight_passed", "gpu_used": False, "files_written": False,
                          "export_performed": False, **receipt}, ensure_ascii=False))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    receipt.update(started_utc=now(), environment=configure_runtime(output),
        export_recipe={"format": "onnx", "device": "cpu", "quantize": 32, "batch": 1,
                       "imgsz": 640, "dynamic": False, "simplify": False, "nms": False, "opset": 17},
        cuda_probe_requested=args.verify_cuda, native_output_parity_evaluated=False,
        speed_benchmark_performed=False, tensorrt_engine_built=False)
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS, downloads
        require(ultralytics.__version__ == ULTRALYTICS_VERSION, "Unexpected runtime Ultralytics version")
        SETTINGS.update({"weights_dir": str(output), "runs_dir": str(output / "runs"),
            "datasets_dir": str(output / "datasets"), "sync": False, "clearml": False,
            "comet": False, "dvc": False, "mlflow": False, "neptune": False, "raytune": False,
            "tensorboard": False, "wandb": False, "openvino_msg": False})
        def existing_asset(file, *a, **kw):
            return str(fpath(file))
        def no_download(*a, **kw):
            raise RuntimeError("Automatic download forbidden")
        downloads.attempt_download_asset = existing_asset
        downloads.safe_download = no_download
        torch.set_num_threads(4)
        snapshot = output / "source_checkpoint.pt"
        shutil.copyfile(source, snapshot)
        require(sha(snapshot) == receipt["source_sha256"], "Isolated checkpoint copy SHA mismatch")
        model = YOLO(str(snapshot))
        require(model.task == args.task, "Checkpoint task differs from requested task")
        model.model.float().cpu().eval()
        require(all(torch.isfinite(value).all().item() for value in model.model.state_dict().values()
                    if torch.is_tensor(value) and value.is_floating_point()), "Nonfinite checkpoint model")
        names = normalized_names(model.names)
        end2end = bool(getattr(model.model, "end2end", False))
        receipt.update(checkpoint_task=model.task, checkpoint_names=names, checkpoint_end2end=end2end,
                       snapshot=str(snapshot), snapshot_sha256=sha(snapshot))
        export_args = dict(receipt["export_recipe"], end2end=end2end, max_det=300,
                           agnostic_nms=False, verbose=False)
        exported = fpath(model.export(**export_args))
        require(exported.is_relative_to(output) and exported.suffix == ".onnx", "Export escaped isolated output")
        receipt["onnx_inspection"] = inspect_onnx(exported, args.task, names, end2end)
        receipt.update(onnx_path=str(exported), onnx_sha256=sha(exported), onnx_bytes=exported.stat().st_size)
        package_root = Path(ultralytics.__file__).resolve().parent
        receipt["export_source_sha256"] = {str(package_root / name): sha(package_root / name)
            for name in ("engine/exporter.py", "utils/export/engine.py", "nn/tasks.py", "nn/modules/head.py")}
        license_path = ROOT / "assets" / "ultralytics_yolo26_coco_20261106" / "LICENSE.AGPL-3.0.txt"
        if license_path.is_file():
            shutil.copyfile(license_path, output / "LICENSE.AGPL-3.0.txt")
            receipt["license_sha256"] = sha(license_path)
        if args.verify_cuda:
            receipt["cuda_execution"] = verify_cuda_execution(exported, output, args.cuda_device)
        else:
            receipt["cuda_execution"] = {"status": "not_run", "actual_cuda_compute_observed": False}
        require(sha(source) == receipt["source_sha256"] and sha(snapshot) == receipt["source_sha256"],
                "Source or snapshot checkpoint changed during export")
        require(sha(receipt["training_status"]) == receipt["training_status_sha256"]
                and sha(receipt["final_selection"]) == receipt["final_selection_sha256"],
                "Training completion/selection record changed during export")
        require(sha(__file__) == receipt["source_script_sha256"], "Export script changed while running")
        receipt.update(phase="complete", completed_utc=now(), source_checkpoint_unchanged=True)
        write_json(output / "export_receipt.json", receipt)
        print(json.dumps({"phase": "complete", "report": str(output / "export_receipt.json"),
                          "onnx": str(exported), "cuda_probe_executed": args.verify_cuda}, ensure_ascii=False))
        return 0
    except BaseException as error:
        receipt.update(phase="failed", failed_utc=now(), error_type=type(error).__name__,
                       error=str(error), traceback=traceback.format_exc())
        write_json(output / "failure.json", receipt)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
