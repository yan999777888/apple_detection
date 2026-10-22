"""Supervise the native, fixed-recipe query trainer with production limits.

Metadata is prepared once per invocation. Five-second telemetry reads only the
small training status; it never rehashes the RGB inventory or runs a model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPTS = ROOT / "scripts"
PYTHON = Path("F:/anaconda/envs/falldet/python.exe")
RUNNER = HERE / "query_dual_train.py"
RUNNER_SHA = "31a346471b2926cec7d822aa34dc04c41cf4e4ff67bb9e0acb7696a746d08d80"
NATIVE = HERE / "watch_query_dual_gpu_native.py"
NATIVE_SHA = "a0c2445a76104680221a65f18e9ffbea9f0ac5973a66a29b6b2779dda1e18686"
BASE = SCRIPTS / "watch_refinement.py"
BASE_SHA = "4f673f671cf1055408da2fd684b68dcbd15bb6f4f9606098e8a89c3b56248b60"
ATOMIC = SCRIPTS / "atomic_io.py"
ATOMIC_SHA = "3b5ca14efc337ce3f511f60f1829c14c24e5ab6fc9772714e051bf265e645380"
SMOKE_OUTPUT = HERE / "query_dual_gpu_native_smoke01"
SMOKE_REPORT = SMOKE_OUTPUT / "report.json"
SMOKE_SUPERVISOR = ROOT / "reports/query_dual_gpu_native_smoke01_supervisor.json"
SMOKE_IDENTITIES = ({"pid": 19024, "create_time": 1792596828.2607608},
                    {"pid": 27096, "create_time": 1792596831.7772})
STATE = {}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


SELF_SHA = sha(__file__)


def load_module(path, name, expected):
    if sha(path) != expected:
        raise RuntimeError("Pinned source changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if "torch" in sys.modules:
        raise RuntimeError("Supervisor must never import Torch")
    return module


def native():
    if "native" not in STATE:
        STATE["native"] = load_module(NATIVE, "query_train_pinned_native_guard", NATIVE_SHA)
    elif sha(NATIVE) != NATIVE_SHA:
        raise RuntimeError("Pinned native guard changed")
    return STATE["native"]


def lexical(value):
    path = Path(value).expanduser()
    path = Path(os.path.abspath(path if path.is_absolute() else ROOT / path))
    if path.drive.upper() != "F:" or path.resolve() != path:
        raise ValueError("Canonical F: paths are required")
    return path


def parse_child(arguments, root=ROOT):
    if Path(root) != ROOT:
        raise ValueError("Fixed project ROOT required")
    arguments = list(arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if not arguments or lexical(arguments[0]) != RUNNER:
        raise ValueError("Only the pinned query trainer is allowed")
    values = {}; resume = False; index = 1
    while index < len(arguments):
        flag = arguments[index]
        if flag == "--resume":
            if resume:
                raise ValueError("Duplicate --resume")
            resume = True; index += 1; continue
        if flag not in {"--run-dir", "--smoke-report"} or flag in values or index + 1 >= len(arguments):
            raise ValueError("Unknown, duplicate or incomplete trainer argument: " + flag)
        values[flag] = lexical(arguments[index + 1]); index += 2
    if set(values) != {"--run-dir", "--smoke-report"} or values["--smoke-report"] != SMOKE_REPORT:
        raise ValueError("Explicit run-dir and exact actual native smoke report required")
    run = values["--run-dir"]
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if (run.parent != ROOT / "runs" or not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run.name)
            or run.name.endswith(".") or run.name.split(".")[0].upper() in reserved):
        raise ValueError("Run must be directly named beneath ROOT/runs")
    if (not resume and run.exists()) or (resume and not run.is_dir()):
        raise ValueError("Fresh run required, or explicit --resume of this own run directory")
    command = [str(PYTHON), "-u", str(RUNNER), "--run-dir", str(run), "--smoke-report", str(SMOKE_REPORT)]
    if resume:
        command.append("--resume")
    return command, {"--output": str(run)}, {"run_dir": run, "smoke_report": SMOKE_REPORT, "resume": resume}


def load_runner():
    if sha(__file__) != SELF_SHA or sha(BASE) != BASE_SHA or sha(ATOMIC) != ATOMIC_SHA:
        raise RuntimeError("Pinned adapter/production/atomic source changed")
    return load_module(RUNNER, "query_train_metadata_only", RUNNER_SHA)


def validate_command(arguments, root):
    command, paths, values = parse_child(arguments, root)
    prepared = STATE.get("prepared")
    if prepared is not None:
        if prepared["command"] != command or prepared["paths"] != paths:
            raise RuntimeError("Fixed training command changed before launch")
        if sha(RUNNER) != RUNNER_SHA or sha(__file__) != SELF_SHA:
            raise RuntimeError("Pinned trainer/adapter changed before launch")
        return command, RUNNER, paths
    environment = native().configure_native_imports()
    metadata = load_runner().metadata(types.SimpleNamespace(**values))
    expected = metadata["source_sha256_expected"]
    if (not isinstance(expected, dict) or expected.get(str(RUNNER)) != RUNNER_SHA
            or any(lexical(path) != Path(path) or not re.fullmatch("[0-9a-f]{64}", digest)
                   for path, digest in expected.items())):
        raise ValueError("Actual fixed trainer source/data SHA map required")
    parent = {str(Path(__file__)): SELF_SHA, str(NATIVE): NATIVE_SHA, str(BASE): BASE_SHA, str(ATOMIC): ATOMIC_SHA}
    if any(path in expected and expected[path] != digest for path, digest in parent.items()):
        raise ValueError("Parent/child source pins conflict")
    STATE["prepared"] = {"command": command, "paths": paths, "values": values,
        "identity": metadata["identity"], "data_inventory": metadata["data_inventory"],
        "smoke_evidence": metadata["smoke_evidence"], "model_metadata": metadata["model_metadata"],
        "expected_child_sources": expected, "parent_only_sources": parent,
        "native_import_environment": environment}
    return command, RUNNER, paths


def smoke_complete(psutil):
    """Bind the completed native smoke without reading its unused partial weights."""
    smoke = read(SMOKE_SUPERVISOR)
    if (smoke.get("phase") != "complete" or smoke.get("exit_code") != 0
            or smoke.get("supervisor") != SMOKE_IDENTITIES[0] or smoke.get("child") != SMOKE_IDENTITIES[1]
            or smoke.get("output") != str(SMOKE_OUTPUT) or smoke.get("pinned_adapter_sha256") != NATIVE_SHA
            or smoke.get("pinned_probe_sha256") != native().RUNNER_SHA):
        raise RuntimeError("Exact native GPU smoke must first complete with exit0")
    box = load_module(native().BOX, "query_train_pinned_box_identity", native().BOX_SHA)
    for identity in SMOKE_IDENTITIES:
        if box.is_live(psutil, identity["pid"], identity["create_time"]):
            raise RuntimeError("Exact native smoke supervisor/child is still alive")
    final = smoke.get("final_evidence", {})
    report_digest = sha(SMOKE_REPORT)
    outputs = final.get("output_sha256", {})
    if (final.get("phase") != "verified"
            or final.get("scope") != "six_bounded_train_cases_and_original_coordinate_postprocess_only"
            or final.get("source_sha256_before") != final.get("source_sha256_after")
            or not final.get("source_sha256_before") or outputs.get(str(SMOKE_REPORT)) != report_digest):
        raise RuntimeError("Actual native smoke final verification/report SHA required")
    report = read(SMOKE_REPORT)
    if (sha(SMOKE_REPORT) != report_digest or report.get("phase") != "gpu_smoke_passed"
            or report.get("stages") != [512, 768] or report.get("completed_smoke_updates") != 6
            or report.get("checkpoint_roundtrip_passed") is not True or report.get("postprocess_bounds_passed") is not True
            or report.get("accuracy_reported") is not False or report.get("test_used") is not False
            or report.get("formal_training_resumable") is not False or report.get("future_formal_warm_start_allowed") is not False
            or report.get("source_before_after_identical") is not True
            or report.get("source_sha256_before") != report.get("source_sha256_after")
            or outputs.get(str(SMOKE_OUTPUT / "smoke.partial.pt")) != report.get("checkpoint_sha256")):
        raise RuntimeError("Both actual pixel stages, finite updates and strict partial mechanics must have passed")
    return {"supervisor_report": str(SMOKE_SUPERVISOR), "report": str(SMOKE_REPORT),
            "report_sha256": report_digest, "phase": "verified_complete_and_exact_processes_exited",
            "partial_weights_loaded_or_reused": False}


def preflight(psutil):
    module = native()
    environment = module.configure_native_imports()
    command = psutil.Process(os.getpid()).cmdline()
    supervisor = (any(command[i:i + 2] == ["-m", "watch_query_dual_train"] for i in range(len(command) - 1))
                  and not any(value.lower().endswith(".py") for value in command))
    child = False
    if len(command) >= 7 and lexical(command[0]) == PYTHON and command[1] == "-u":
        try:
            expected, _, _ = parse_child(command[2:])
            child = expected == command
        except ValueError:
            pass
    if not supervisor and not child:
        raise RuntimeError("Only native -m watch_query_dual_train or its exact own trainer CLI is allowed")
    if child:
        parent = psutil.Process(os.getppid()); argv = parent.cmdline()
        if (Path(parent.exe()).resolve() != PYTHON.resolve()
                or not any(argv[i:i + 2] == ["-m", "watch_query_dual_train"] for i in range(len(argv) - 1))
                or any(value.lower().endswith(".py") for value in argv)):
            raise RuntimeError("Trainer child must belong to its exact native module supervisor")
    smoke = smoke_complete(psutil)
    # Before Torch only: this retains the pinned PApple completion and original
    # WDDM baseline gate. It skips only this exact validated current trainer PID.
    guard = module.guard(psutil)
    return {"native_smoke": smoke, "existing_papple_and_WDDM_gate": guard,
            "native_import_environment": environment}


def guarded_spawn(command, *args, **kwargs):
    import psutil
    expected, _, _ = validate_command(command[2:], ROOT)
    if (command != expected or command[:2] != [str(PYTHON), "-u"]
            or Path(kwargs.get("cwd", "")) != ROOT or kwargs.get("shell", False) or kwargs.get("env") is not None):
        raise ValueError("Only the fixed native trainer command may spawn")
    STATE["guard_before_spawn"] = preflight(psutil)
    return subprocess.Popen(command, *args, **kwargs)


def verify_success(supervisor):
    prepared = STATE["prepared"]
    if (supervisor.get("exit_code") != 0 or not supervisor.get("child", {}).get("pid")
            or not supervisor["child"].get("create_time")):
        raise ValueError("Actual owned trainer exit0 required")
    run = Path(prepared["paths"]["--output"])
    signatures = {}
    def stable(path):
        path = Path(path)
        digest = sha(path); value = read(path)
        if sha(path) != digest:
            raise ValueError("Completed output changed during verification: " + str(path))
        signatures[str(path)] = digest
        return value
    status = stable(run / "training_status.json")
    summary = stable(run / "training_summary.json")
    config = stable(run / "run_config.json")
    index = stable(run / "checkpoint_index.json")
    schema = "query_dual_papple_only_full_epoch_training_v1"
    run_id = config.get("run_id")
    expected = prepared["expected_child_sources"]
    if (not run_id or config.get("schema") != schema or config.get("identity") != prepared["identity"]
            or config.get("source_sha256") != expected or config.get("fresh_official_initialization") is not True
            or config.get("smoke_evidence") != prepared["smoke_evidence"]):
        raise ValueError("Actual run config must bind fresh official initialization and exact source/data identity")
    if (summary.get("schema") != schema or summary.get("phase") != "complete"
            or summary.get("run_id") != run_id or summary.get("run_dir") != str(run)
            or summary.get("epochs_completed") != 6 or summary.get("recipe") != prepared["identity"]["recipe"]
            or summary.get("source_sha256_before") != expected or summary.get("source_sha256_after") != expected
            or summary.get("source_before_after_identical") is not True
            or summary.get("test_used") is not False or summary.get("smoke_weights_used_for_initialization") is not False):
        raise ValueError("Terminal actual complete six-epoch source/recipe evidence required")
    updates = summary.get("updates", {})
    if (set(updates) != {"attempted", "successful", "skipped"}
            or any(type(value) is not int or value < 0 for value in updates.values())
            or updates["attempted"] != 3456 or updates["successful"] < 1
            or updates["successful"] + updates["skipped"] != 3456):
        raise ValueError("Actual attempted/successful/skipped update accounting differs")
    if (status.get("schema") != schema or status.get("phase") != "complete" or status.get("run_id") != run_id
            or status.get("epoch") != 6 or status.get("image_in_epoch") != 2304 or status.get("completed_epochs") != 6
            or status.get("attempted_updates") != 3456 or status.get("successful_updates") != updates["successful"]
            or status.get("skipped_updates") != updates["skipped"] or status.get("source_before_after_identical") is not True
            or status.get("source_sha256_after") != expected):
        raise ValueError("Terminal actual trainer status differs")
    history = summary.get("history", [])
    if ([row.get("epoch") for row in history] != list(range(1, 7))
            or stable(run / "history.json") != history):
        raise ValueError("Six committed actual epoch records required")
    expected_names = {row["filename"] for row in prepared["data_inventory"]["train"]}
    for epoch, row in enumerate(history, 1):
        order = row.get("train_order", [])
        val = row.get("validation", {}); metric = val.get("summary", {})
        epoch_updates = row.get("updates", {})
        if (row.get("train_images") != 2304 or len(order) != 2304 or len(set(order)) != 2304
                or set(order) != expected_names or metric.get("images") != 814 or metric.get("gt_count") != 3449
                or metric.get("empty_gt_images") != 13 or val.get("ap_computed") is not False
                or not all(type(metric.get(key)) in (int, float) and math.isfinite(metric[key])
                           and 0 <= metric[key] <= 1 for key in ("precision", "recall", "f1"))
                or val.get("score") != min(metric["precision"], metric["recall"])
                or epoch_updates.get("attempted") != epoch * 576
                or epoch_updates.get("successful", -1) + epoch_updates.get("skipped", -1) != epoch * 576):
            raise ValueError("Actual full TRAIN2304/VAL814_GT3449 epoch scope or update evidence differs")
        folder = lexical(row["validation_directory"])
        if folder.parent != run / "validation" or not re.fullmatch(f"epoch_{epoch:03d}_attempt_[0-9]+", folder.name):
            raise ValueError("Each validation must belong to this run's exact epoch directory")
        files = row.get("validation_files_sha256", {})
        names = val.get("files", {})
        if not files or set(files) != {str(folder / name) for name in names.values()}:
            raise ValueError("Complete per-epoch actual validation-file inventory required")
        for path, digest in files.items():
            if lexical(path).parent != folder or not re.fullmatch("[0-9a-f]{64}", digest) or sha(path) != digest:
                raise ValueError("Actual validation output signature differs")
            signatures[path] = digest
        if read(folder / names["summary"]) != val:
            raise ValueError("Actual saved full VAL summary differs from epoch history")
        val_meta = read(folder / names["metadata"])
        if (val_meta.get("phase") != "complete" or val_meta.get("consumed_images") != 814
                or val_meta.get("consumed_GT") != 3449 or val_meta.get("no_GT_repair_or_exclusion") is not True
                or val_meta.get("preprocess", {}).get("input_hw") != [768, 768]
                or val_meta.get("test_data_read") is not False or val_meta.get("subset_filter") is not False
                or val_meta.get("all_RGB_before_after_verified") is not True
                or val_meta.get("all_original_GT_numeric_fields_and_mask_SHA256_verified") is not True
                or val_meta.get("source_sha256_before") != val_meta.get("source_sha256_after")
                or not val_meta.get("source_sha256_before")):
            raise ValueError("Actual full VAL consumption and true-mask source verification required")
    update_history = stable(run / "update_history.json")
    successful = skipped = 0
    if len(update_history) != 3456:
        raise ValueError("Actual accumulated update history is incomplete")
    for attempted, row in enumerate(update_history, 1):
        executed = row.get("optimizer_step_executed")
        if type(executed) is not bool:
            raise ValueError("Every AMP update must record whether AdamW actually executed")
        successful += int(executed); skipped += int(not executed)
        if (row.get("attempted") != attempted or row.get("successful") != successful or row.get("skipped") != skipped
                or row.get("epoch") != (attempted - 1) // 576 + 1
                or row.get("image_in_epoch") != ((attempted - 1) % 576 + 1) * 4):
            raise ValueError("Actual update counters/order differ")
    if successful != updates["successful"] or skipped != updates["skipped"]:
        raise ValueError("Terminal AMP accounting differs from actual update history")
    rank = lambda row: (row["validation"]["score"], row["validation"]["summary"]["f1"], row["validation"]["confidence"])
    best_epoch = max(history, key=rank)["epoch"]
    if (index.get("schema") != schema or index.get("run_id") != run_id
            or index.get("last") != summary.get("last") or index.get("best") != summary.get("best")
            or status.get("checkpoint_index") != index):
        raise ValueError("Authoritative own checkpoint commit index differs")
    for key, epoch in (("last", 6), ("best", best_epoch)):
        entry = summary[key]; path = lexical(entry["path"]); alias = run / f"{key}.pt"
        if (path.parent != run / "checkpoints" or not re.fullmatch(f"epoch_{epoch:03d}_attempt_[0-9]+[.]pt", path.name)
                or entry.get("epoch") != epoch or entry.get("run_id") != run_id
                or entry.get("score") != history[epoch - 1]["validation"]["score"]
                or summary.get(key + "_alias") != str(alias) or not re.fullmatch("[0-9a-f]{64}", entry["sha256"])
                or sha(path) != entry["sha256"] or sha(alias) != entry["sha256"]):
            raise ValueError("Own immutable last/best checkpoint and alias bytes must match the actual committed index")
        signatures[str(path)] = signatures[str(alias)] = entry["sha256"]
    all_sources = {**expected, **prepared["parent_only_sources"]}
    actual = {path: sha(path) for path in all_sources}
    if actual != all_sources or any(sha(path) != digest for path, digest in signatures.items()):
        raise ValueError("Consumed source/data or completed output bytes changed during final verification")
    return {"phase": "verified", "scope": "six_complete_public_PApple_TRAIN2304_and_VAL814_GT3449_epochs",
        "epochs_completed": 6, "updates": updates, "run_id": run_id, "last": summary["last"], "best": summary["best"],
        "quality_target_claimed": False, "test_used": False, "smoke_weights_used_for_initialization": False,
        "source_sha256_before": all_sources, "source_sha256_after": actual, "output_sha256": signatures}


def load_base():
    load_runner(); sys.path.insert(0, str(SCRIPTS))
    base = load_module(BASE, "query_train_pinned_production_supervisor", BASE_SHA)
    if list(inspect.signature(base.run).parameters) != ["args", "root", "psutil"]:
        raise RuntimeError("Pinned production supervisor run contract changed")
    atomic_json = base.atomic_json
    def validated_save(path, report):
        prepared = STATE.get("prepared")
        if prepared is not None and "log_name" in report and "monitor_summary" in report:
            report["pinned_adapter_sha256"] = SELF_SHA
            report["pinned_trainer_sha256"] = RUNNER_SHA
            report["native_import_environment"] = prepared["native_import_environment"]
            report["expected_actual_child_source_count"] = len(prepared["expected_child_sources"])
            report["query_training_guard"] = STATE.get("guard_before_spawn")
            report["accuracy_note"] = "Completion verifies the actual fixed full scope and own checkpoints; it makes no unmeasured95% claim."
            try:
                status = read(Path(prepared["paths"]["--output"]) / "training_status.json")
                report["runner_evidence"] = {key: status[key] for key in (
                    "phase", "epoch", "image_in_epoch", "attempted_updates", "successful_updates",
                    "skipped_updates", "completed_epochs", "error") if key in status}
            except (OSError, ValueError):
                report["runner_evidence"] = {"available": False}
            if report.get("phase") == "complete" and not STATE.get("final_verified"):
                try:
                    report["final_evidence"] = verify_success(report); STATE["final_verified"] = True
                except BaseException as error:
                    report.update(phase="failed", failure_reason="actual_query_training_evidence_rejected",
                                  final_evidence_error=repr(error))
        atomic_json(path, report)
    base.validate_command = validate_command
    base.SCRIPT_RULES = {RUNNER.name: (set(), set(), set(), "training_status.json")}
    base.atomic_json = validated_save
    base.subprocess = types.SimpleNamespace(run=subprocess.run, Popen=guarded_spawn,
        TimeoutExpired=subprocess.TimeoutExpired, CREATE_NO_WINDOW=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-name", required=True)
    parser.add_argument("--wall-time-seconds", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--smoke-report", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.wall_time_seconds <= 28800:
        parser.error("Explicit wall limit must be1..28800 seconds")
    command = [str(RUNNER), "--run-dir", args.run_dir, "--smoke-report", args.smoke_report]
    if args.resume:
        command.append("--resume")
    parse_child(command)
    import psutil
    preflight(psutil)
    base = load_base()
    from runtime import configure_paths
    if Path(configure_paths()).resolve() != ROOT:
        raise RuntimeError("Authorized runtime ROOT changed")
    return base.run(types.SimpleNamespace(log_name=args.log_name, wall_time_seconds=args.wall_time_seconds,
                                          command=command), ROOT, psutil)


if __name__ == "__main__":
    raise SystemExit(main())
