"""Supervise the native residual-geometry/deep-auxiliary twelve-epoch trial.

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
OLD = ROOT / "temp" / "query_dual_mask_candidate_20261021"
PYTHON = Path("F:/anaconda/envs/falldet/python.exe")
RUNNER = HERE / "train_residual_dual.py"
RUNNER_SHA = "40413ed9c97342bb6ed01806b34356d706d9a51a4a9d1f70dce2ebb1797519a0"
NATIVE = OLD / "watch_query_dual_gpu_native.py"
NATIVE_SHA = "a0c2445a76104680221a65f18e9ffbea9f0ac5973a66a29b6b2779dda1e18686"
BASE = SCRIPTS / "watch_refinement.py"
BASE_SHA = "4f673f671cf1055408da2fd684b68dcbd15bb6f4f9606098e8a89c3b56248b60"
ATOMIC = SCRIPTS / "atomic_io.py"
ATOMIC_SHA = "3b5ca14efc337ce3f511f60f1829c14c24e5ab6fc9772714e051bf265e645380"
SMOKE_OUTPUT = HERE / "residual_gpu_smoke01"
SMOKE_REPORT = SMOKE_OUTPUT / "report.json"
SMOKE_SUPERVISOR = ROOT / "reports/residual_gpu_smoke01_supervisor.json"
SMOKE_RUNNER = HERE / "probe_residual_dual_gpu.py"
SMOKE_RUNNER_SHA = "32c3804889afdafa13bff05461785970d89ee727ac51c580bef3da3f3b805b8a"
SMOKE_WATCH = HERE / "watch_residual_dual_gpu.py"
SMOKE_WATCH_SHA = "9e0f1141b6168e709a42bbbb54647cf0c91fe47ef5d1e848f7034a44ae8ca8f7"
PARENT_RUN = ROOT / "runs" / "query_dual_papple_train01_20261021"
PARENT_SUPERVISOR = ROOT / "reports" / "query_dual_papple_train01_20261021_supervisor.json"
PARENT_CHECKPOINT_SHA = "aac6f257bf43af3b945dd3469ba66214de99d7ae878e13e85ec7bef0e20b3227"
PARENT_IDENTITIES = ({"pid": 12600, "create_time": 1792597177.922958},
                     {"pid": 38160, "create_time": 1792597188.4334152})
DESKTOP = ROOT / "reports" / "wddm_desktop_gpu_baseline_20261022_01.json"
DESKTOP_SHA = "f9467bf7a991370b082e30700247ac44210c09bc0ecac0c682a0358ba8c973bc"
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
    if sha(DESKTOP) != DESKTOP_SHA:
        raise RuntimeError("The root-reviewed October22 desktop baseline changed")
    # Root reviewed all35 desktop identities. The only addition to the previous
    # snapshot is an exact WPS identity, with no Python process; retain every
    # original concurrency/STOP/PApple gate and override only the approved file.
    STATE["native"].DESKTOP = DESKTOP
    STATE["native"].DESKTOP_SHA = DESKTOP_SHA
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
    parent = {str(Path(__file__)): SELF_SHA, str(NATIVE): NATIVE_SHA, str(BASE): BASE_SHA,
              str(ATOMIC): ATOMIC_SHA, str(DESKTOP): DESKTOP_SHA}
    if any(path in expected and expected[path] != digest for path, digest in parent.items()):
        raise ValueError("Parent/child source pins conflict")
    STATE["prepared"] = {"command": command, "paths": paths, "values": values,
        "identity": metadata["identity"], "data_inventory": metadata["data_inventory"],
        "smoke_evidence": metadata["smoke_evidence"], "model_metadata": metadata["model_metadata"],
        "expected_child_sources": expected, "parent_only_sources": parent,
        "native_import_environment": environment}
    return command, RUNNER, paths


def parent_complete(psutil):
    """Require actual terminal-verified parent query6 and its exact processes gone."""
    supervisor = read(PARENT_SUPERVISOR)
    final = supervisor.get("final_evidence", {})
    if (supervisor.get("phase") != "complete" or supervisor.get("exit_code") != 0
            or supervisor.get("supervisor") != PARENT_IDENTITIES[0]
            or supervisor.get("child") != PARENT_IDENTITIES[1]
            or supervisor.get("output") != str(PARENT_RUN)
            or final.get("phase") != "verified" or final.get("epochs_completed") != 6
            or final.get("source_sha256_before") != final.get("source_sha256_after")
            or not final.get("source_sha256_before")):
        raise RuntimeError("The exact query parent must be complete/exit0/final verified")
    box = load_module(native().BOX, "residual_parent_identity_guard", native().BOX_SHA)
    for identity in PARENT_IDENTITIES:
        if box.is_live(psutil, identity["pid"], identity["create_time"]):
            raise RuntimeError("The exact parent query trainer/supervisor is still alive")
    summary_path = PARENT_RUN / "training_summary.json"
    summary = read(summary_path)
    last = summary.get("last", {})
    if (summary.get("phase") != "complete" or summary.get("epochs_completed") != 6
            or summary.get("test_used") is not False or last.get("epoch") != 6
            or last.get("sha256") != PARENT_CHECKPOINT_SHA or final.get("last") != last
            or final.get("output_sha256", {}).get(str(summary_path)) != sha(summary_path)):
        raise RuntimeError("Exact committed parent-six model and full training summary evidence required")
    return {"supervisor_report": str(PARENT_SUPERVISOR), "phase": "verified_complete_and_exact_processes_exited",
            "epochs_completed": 6, "parent_checkpoint": last, "optimizer_restored": False}


def smoke_complete(psutil):
    """Require actual residual1024 smoke completion and its exact process identities exited."""
    supervisor = read(SMOKE_SUPERVISOR)
    final = supervisor.get("final_evidence", {})
    if (supervisor.get("phase") != "complete" or supervisor.get("exit_code") != 0
            or supervisor.get("output") != str(SMOKE_OUTPUT)
            or supervisor.get("pinned_adapter_sha256") != SMOKE_WATCH_SHA
            or supervisor.get("pinned_probe_sha256") != SMOKE_RUNNER_SHA
            or sha(SMOKE_WATCH) != SMOKE_WATCH_SHA or sha(SMOKE_RUNNER) != SMOKE_RUNNER_SHA):
        raise RuntimeError("The exact actual residual1024 GPU smoke must first complete with exit0")
    box = load_module(native().BOX, "residual_smoke_identity_guard", native().BOX_SHA)
    identities = [supervisor.get(key, {}) for key in ("supervisor", "child")]
    for identity in identities:
        if (type(identity.get("pid")) is not int or identity["pid"] < 1
                or type(identity.get("create_time")) not in (int, float)):
            raise RuntimeError("Actual residual smoke process identity is missing")
        if box.is_live(psutil, identity["pid"], identity["create_time"]):
            raise RuntimeError("Exact residual smoke supervisor/child is still alive")
    digest = sha(SMOKE_REPORT)
    if (final.get("phase") != "verified"
            or final.get("source_sha256_before") != final.get("source_sha256_after")
            or not final.get("source_sha256_before")
            or final.get("output_sha256", {}).get(str(SMOKE_REPORT)) != digest):
        raise RuntimeError("Actual residual smoke source/output verification is required")
    report = read(SMOKE_REPORT)
    if (report.get("phase") != "residual_gpu_smoke_passed" or report.get("input_size") != 1024
            or report.get("completed_smoke_updates") != 3
            or report.get("strict_parent_model_load") is not True
            or report.get("checkpoint_roundtrip_passed") is not True
            or report.get("postprocess_bounds_passed") is not True
            or report.get("accuracy_reported") is not False or report.get("test_used") is not False
            or report.get("formal_training_resumable") is not False
            or report.get("source_before_after_identical") is not True
            or report.get("source_sha256_before") != report.get("source_sha256_after")
            or sha(SMOKE_REPORT) != digest):
        raise RuntimeError("Real1024 dual-mask/residual-box/deep-auxiliary updates and strict checkpoint mechanics required")
    return {"supervisor_report": str(SMOKE_SUPERVISOR), "report": str(SMOKE_REPORT),
            "report_sha256": digest, "phase": "verified_complete_and_exact_processes_exited",
            "actual_process_identities": identities, "partial_weights_loaded_or_reused": False}


def scan_other_model_commands(psutil):
    """Retain the native scan and explicitly cover this new probe/trainer pair."""
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid == os.getpid():
            continue
        try:
            scripts = [Path(value).name.lower() for value in process.info.get("cmdline") or []
                       if value.lower().endswith(".py")]
            if any(name in {RUNNER.name.lower(), SMOKE_RUNNER.name.lower()}
                   or re.search(r"(^|_)(train|training|eval|evaluate|inference|predict|smoke|verify)(_|\.)", name)
                   for name in scripts):
                raise RuntimeError(f"Another potentially GPU-using script is alive: PID {process.pid}, {scripts}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            if (process.info.get("name") or "").lower().startswith("python"):
                raise RuntimeError("Cannot inspect another Python process; concurrency check fails closed")


def preflight(psutil):
    module = native()
    environment = module.configure_native_imports()
    command = psutil.Process(os.getpid()).cmdline()
    supervisor = (any(command[i:i + 2] == ["-m", "watch_residual_dual_train"] for i in range(len(command) - 1))
                  and not any(value.lower().endswith(".py") for value in command))
    child = False
    if len(command) >= 7 and lexical(command[0]) == PYTHON and command[1] == "-u":
        try:
            expected, _, _ = parse_child(command[2:])
            child = expected == command
        except ValueError:
            pass
    if not supervisor and not child:
        raise RuntimeError("Only native -m watch_residual_dual_train or its exact own trainer CLI is allowed")
    if child:
        parent = psutil.Process(os.getppid()); argv = parent.cmdline()
        if (Path(parent.exe()).resolve() != PYTHON.resolve()
                or not any(argv[i:i + 2] == ["-m", "watch_residual_dual_train"] for i in range(len(argv) - 1))
                or any(value.lower().endswith(".py") for value in argv)):
            raise RuntimeError("Trainer child must belong to its exact native module supervisor")
    parent = parent_complete(psutil)
    smoke = smoke_complete(psutil)
    scan_other_model_commands(psutil)
    # Before Torch only: this retains the pinned PApple completion and original
    # WDDM baseline gate. It skips only this exact validated current trainer PID.
    guard = module.guard(psutil)
    return {"completed_query_parent": parent, "residual_1024_smoke": smoke,
            "existing_papple_and_WDDM_gate": guard,
            "desktop_baseline_override": {"path": str(DESKTOP), "sha256": DESKTOP_SHA,
                "basis": "Root-reviewed35/35 desktop identities; exact added WPS only, no Python"},
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
    schema = "residual_dual_papple_only_warmstart_full_epoch_training_v1"
    run_id = config.get("run_id")
    expected = prepared["expected_child_sources"]
    if (not run_id or config.get("schema") != schema or config.get("identity") != prepared["identity"]
            or config.get("source_sha256") != expected or config.get("fresh_official_initialization") is not False
            or config.get("parent_model_only_warmstart") is not True or config.get("new_optimizer_scaler_rng_schedule") is not True
            or config.get("parent_checkpoint") != prepared["model_metadata"]["parent_checkpoint"]
            or config.get("smoke_evidence") != prepared["smoke_evidence"]):
        raise ValueError("Actual run config must bind parent MODEL-only warm-start and new training state")
    if (summary.get("schema") != schema or summary.get("phase") != "complete"
            or summary.get("run_id") != run_id or summary.get("run_dir") != str(run)
            or summary.get("epochs_completed") != 12 or summary.get("recipe") != prepared["identity"]["recipe"]
            or summary.get("source_sha256_before") != expected or summary.get("source_sha256_after") != expected
            or summary.get("source_before_after_identical") is not True
            or summary.get("test_used") is not False or summary.get("smoke_weights_used_for_initialization") is not False
            or summary.get("parent_model_only_warmstart") is not True
            or summary.get("parent_checkpoint") != prepared["model_metadata"]["parent_checkpoint"]
            or summary.get("joint_recipe_change") != prepared["identity"]["recipe"]["joint_recipe_change"]
            or summary.get("causal_interpretation") != prepared["identity"]["recipe"]["causal_interpretation"]):
        raise ValueError("Terminal actual complete twelve-epoch source/recipe evidence required")
    updates = summary.get("updates", {})
    if (set(updates) != {"attempted", "successful", "skipped"}
            or any(type(value) is not int or value < 0 for value in updates.values())
            or updates["attempted"] != 6912 or updates["successful"] < 1
            or updates["successful"] + updates["skipped"] != 6912):
        raise ValueError("Actual attempted/successful/skipped update accounting differs")
    if (status.get("schema") != schema or status.get("phase") != "complete" or status.get("run_id") != run_id
            or status.get("epoch") != 12 or status.get("image_in_epoch") != 2304 or status.get("completed_epochs") != 12
            or status.get("attempted_updates") != 6912 or status.get("successful_updates") != updates["successful"]
            or status.get("skipped_updates") != updates["skipped"] or status.get("source_before_after_identical") is not True
            or status.get("source_sha256_after") != expected):
        raise ValueError("Terminal actual trainer status differs")
    history = summary.get("history", [])
    if ([row.get("epoch") for row in history] != list(range(1, 13))
            or stable(run / "history.json") != history):
        raise ValueError("Twelve committed actual epoch records required")
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
        if (val_meta.get("schema") != "residual_dual_full_val_evaluation_v1"
                or val_meta.get("phase") != "complete" or val_meta.get("consumed_images") != 814
                or val_meta.get("consumed_GT") != 3449 or val_meta.get("no_GT_repair_or_exclusion") is not True
                or val_meta.get("preprocess", {}).get("input_hw") != [1024, 1024]
                or val_meta.get("auxiliary_supervision_layers") != 1
                or val_meta.get("annotation_visible_area_exceeds_amodal_GT_retained") != 14
                or val_meta.get("postprocess_policy", {}).get("schema") != "residual_amodal_box_direct_postprocess_v1"
                or val_meta.get("test_data_read") is not False or val_meta.get("subset_filter") is not False
                or val_meta.get("all_RGB_before_after_verified") is not True
                or val_meta.get("all_original_GT_numeric_fields_and_mask_SHA256_verified") is not True
                or val_meta.get("source_sha256_before") != val_meta.get("source_sha256_after")
                or not val_meta.get("source_sha256_before")):
            raise ValueError("Actual full VAL consumption and true-mask source verification required")
    update_history = stable(run / "update_history.json")
    successful = skipped = 0
    if len(update_history) != 6912:
        raise ValueError("Actual accumulated update history is incomplete")
    for attempted, row in enumerate(update_history, 1):
        executed = row.get("optimizer_step_executed")
        if type(executed) is not bool:
            raise ValueError("Every AMP update must record whether AdamW actually executed")
        step = successful + 1
        factor = step / 100 if step <= 100 else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - 100) / 6812)))
        for field, base_lr in (("lr_backbone", 1e-5), ("lr_old_other", 5e-5), ("lr_new_box_head", 2e-4)):
            value = row.get(field)
            if type(value) not in (int, float) or not math.isclose(value, base_lr * factor, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("Recorded AdamW group LR differs from the successful-update-only schedule")
        if not math.isclose(row.get("lr_factor", -1), factor, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("AMP skips must not advance the learning-rate schedule")
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
    for key, epoch in (("last", 12), ("best", best_epoch)):
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
    return {"phase": "verified", "scope": "twelve_complete_public_PApple_TRAIN2304_and_VAL814_GT3449_residual_geometry_aux_epochs",
        "epochs_completed": 12, "updates": updates, "run_id": run_id, "last": summary["last"], "best": summary["best"],
        "quality_target_claimed": False, "test_used": False, "smoke_weights_used_for_initialization": False,
        "parent_model_only_warmstart": True, "parent_checkpoint": prepared["model_metadata"]["parent_checkpoint"],
        "input_size": 1024, "geometry_and_auxiliary_joint_recipe": True,
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
