"""Fixed six-epoch, public-PApple-only shared-query V/A training core.

Source preparation and metadata(args) use only the standard library. Runtime
is explicitly entered by train()/CLI, after a verified actual native 768 smoke.
Fresh runs always construct from verified official safetensors. Resume accepts
only this run's committed, complete TRAIN epoch plus full VAL checkpoint; smoke
weights and partial epochs have no loading path. This file is not a supervisor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import util
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
import traceback
import uuid


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPTS = ROOT / "scripts"
PAPPLE = ROOT / "data" / "papple" / "data"
PYTHON = Path("F:/anaconda/envs/falldet/python.exe")
SITE_PACKAGES = ROOT / "envs" / "query_dual_v1" / "Lib" / "site-packages"
SMOKE_REPORT = HERE / "query_dual_gpu_native_smoke01" / "report.json"
SMOKE_SUPERVISOR = ROOT / "reports" / "query_dual_gpu_native_smoke01_supervisor.json"
TRAIN_WATCH = HERE / "watch_query_dual_train.py"
SCHEMA = "query_dual_papple_only_full_epoch_training_v1"
PINS = {
    HERE / "query_model_runtime.py": "d3aae8748062bf76aca7639bf0c7e169c74cedde73269cedd30302e81df16d5e",
    HERE / "query_dual_mask.py": "cef59347445af673eceff65cd5d0ed7caa5769683c5e78eb7f4610deb2e48878",
    HERE / "query_dual_criterion.py": "6ceec79c357b0e30864dfd47e0113b574ce7b65eecd24c40ae193bdf9e5b74ee",
    HERE / "query_papple_data.py": "1fbb142f4710454be90833bdacc3ff4841c9339ca1d5332897f743da36be96df",
    HERE / "query_dual_postprocess.py": "d19cedb68be01e68957acff0026b3c50a6c6e4d7fe2bf176b317e37057cfbf9d",
    HERE / "query_dual_evaluate.py": "52ddf3972d650eeb9ab0ac666d17cbc4949ecbd3e4711f6ef78b8da3e548d629",
    HERE / "probe_query_dual_gpu_native.py": "2e83146d1b94f75cc221d156ee0a6b2aaf7c2c0494f08764e7bb2e0642066a34",
    HERE / "watch_query_dual_gpu_native.py": "a0c2445a76104680221a65f18e9ffbea9f0ac5973a66a29b6b2779dda1e18686",
    SCRIPTS / "dataset.py": "4d8fe0f1ac98e3b54dd4f029675ad240b621766140b259a537e99290ace9a0a0",
    SCRIPTS / "evaluate.py": "a42cd063ffe6aedfd28d2807b800fc368b0782b3073556cba2408d8159236492",
    SCRIPTS / "metrics.py": "1b0a7e385940fd0c6160cc08b75ddaf2e1f43f3ea00f5bd37d547fea6b08010d",
    SCRIPTS / "atomic_io.py": "3b5ca14efc337ce3f511f60f1829c14c24e5ab6fc9772714e051bf265e645380",
    SCRIPTS / "runtime.py": "a21eba633ba1d08cfe07b7695186d811f2e69ac560ce441a236ffce7b93fb64c",
    SCRIPTS / "train_refinement.py": "94dc54a5212b1e825295f1085bf6c6ce89dffaa10b331220afa08c20b573fc8f",
    SCRIPTS / "refinement_config.py": "20b31460a36b93d5816358c1f40169ed081d78577b095eb7033fc59cddecd364",
    ROOT / "reports" / "papple_val_provenance_20261021.json": "fcd96d08c75e1663c83efecb8eb321bbc10220dc74314c8227f1f8565a2ccb3a",
}
RECIPE = {
    "epochs": 6, "input_size": 768, "batch_size": 1, "accumulation_steps": 4,
    "train_images": 2304, "val_images": 814, "val_GT_instances": 3449, "val_empty_images": 13,
    "workers": 0, "pin_memory": False, "augment": False, "external_padding": False,
    "seed": 20261021, "optimizer": "AdamW", "backbone_module": "shared.pixel_level_module.encoder",
    "backbone_lr": 1e-5, "other_lr": 1e-4, "weight_decay": 0.05,
    "betas": [0.9, 0.999], "eps": 1e-8, "foreach": False,
    "gradient_clip_norm": 0.1, "amp": "float16", "GradScaler_init_scale": 128.0,
    "warmup_successful_updates": 100, "planned_updates": 3456,
    "schedule": "successful-update linear warmup then cosine to 5% at successful update 3456",
    "criterion": {"match_points": 12544, "loss_points": 12544, "oversample_ratio": 3.0,
                  "importance_ratio": 0.75, "class_weight": 2.0, "mask_weight": 5.0,
                  "dice_weight": 5.0, "no_object_weight": 0.1, "visible_weight": 0.5, "amodal_weight": 0.5},
    "GT_device": "cpu", "data": "public PApple TRAIN/VAL only", "test_used": False,
    "checkpoint_scope": "complete TRAIN epoch and complete full VAL only",
    "fresh_initialization": "verified official safetensors; never smoke or other-run weights",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _verify(expected):
    actual = {path: sha(path) for path in expected}
    if actual != expected:
        raise ValueError("Consumed source/data bytes changed: " + repr([p for p in expected if actual[p] != expected[p]]))
    return actual


def _paths(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    report = Path(args.smoke_report).expanduser().resolve()
    if (ROOT.drive.upper() != "F:" or run_dir.parent != (ROOT / "runs").resolve()
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_dir.name)
            or run_dir.name.endswith(".") or report != SMOKE_REPORT.resolve()):
        raise ValueError("Require a directly named F: local_amodal/runs run and the exact native smoke report")
    return run_dir, report


def _smoke_evidence(report_path):
    supervisor = read(SMOKE_SUPERVISOR)
    final = supervisor.get("final_evidence", {})
    report = read(report_path)
    report_sha = sha(report_path)
    if (supervisor.get("phase") != "complete" or supervisor.get("exit_code") != 0
            or final.get("phase") != "verified" or final.get("pixel_sizes") != [512, 768]
            or final.get("accuracy_reported") is not False or final.get("formal_training_resumable") is not False
            or final.get("output_sha256", {}).get(str(report_path)) != report_sha
            or final.get("source_sha256_before") != final.get("source_sha256_after")
            or not final.get("source_sha256_before")):
        raise ValueError("Actual completed native GPU supervisor with verified 768 evidence is required")
    source = report.get("source_sha256_before", {})
    if (report.get("phase") != "gpu_smoke_passed" or report.get("source_before_after_identical") is not True
            or source != report.get("source_sha256_after") or not source
            or report.get("completed_smoke_updates") != 6
            or report.get("strict_load") != {"strict": True, "missing_keys": [], "unexpected_keys": []}
            or report.get("initialization") != {"copied_classifier_rows": [47, 80],
                "AV_MLP_equal_and_independent": True, "shared_model_by_reference": True}
            or report.get("postprocess_bounds_passed") is not True or report.get("checkpoint_roundtrip_passed") is not True
            or report.get("accuracy_reported") is not False or report.get("test_used") is not False
            or report.get("formal_training_resumable") is not False or report.get("future_formal_warm_start_allowed") is not False
            or report.get("identity", {}).get("schema") != "query_dual_bounded_real_gpu_smoke_v1"
            or report.get("identity", {}).get("criterion_points") != 12544):
        raise ValueError("Same-source real dual-mask GPU smoke evidence is incomplete")
    stages = report.get("stage_results", [])
    if ([stage.get("input_size") for stage in stages] != [512, 768]
            or any([case.get("case_kind") for case in stage.get("cases", [])] != ["ordinary", "max_gt", "empty"] for stage in stages)):
        raise ValueError("Require all ordinary/max-GT/empty cases at 512 and 768")
    step = 0
    for stage in stages:
        for case in stage["cases"]:
            step += 1
            if (case.get("finite") is not True or case.get("GTunchanged") is not True
                    or case.get("optimizer_update_verified") is not True or case.get("optimizer_update_number") != step
                    or case.get("input_shape") != [1, 3, stage["input_size"], stage["input_size"]]
                    or case.get("GT_shapes") != [case.get("GT_count"), 1300, 1300]):
                raise ValueError("Actual finite 768 forward/backward/optimizer update and unchanged original GT required")
    combined = final["source_sha256_before"]
    for path, digest in source.items():
        if combined.get(path) != digest:
            raise ValueError("Child/supervisor smoke source identities disagree")
    for path in (HERE / "query_model_runtime.py", HERE / "query_dual_mask.py", HERE / "query_dual_criterion.py",
                 HERE / "query_papple_data.py", HERE / "query_dual_postprocess.py",
                 HERE / "probe_query_dual_gpu_native.py", HERE / "watch_query_dual_gpu_native.py"):
        if combined.get(str(path)) != PINS[path]:
            raise ValueError("Passed GPU smoke used a different frozen query implementation")
    _verify(combined)
    # Hash the evidence bytes only. No torch.load or warm-start path exists here.
    _verify(final["output_sha256"])
    return {"report": str(report_path), "report_sha256": report_sha,
            "supervisor": str(SMOKE_SUPERVISOR), "supervisor_sha256": sha(SMOKE_SUPERVISOR),
            "phase": "verified", "pixel_sizes": [512, 768], "completed_smoke_updates": 6,
            "source_sha256": combined, "output_sha256": final["output_sha256"],
            "official_source": report["identity"]["official_source"],
            "smoke_weights_used_for_initialization": False}


def metadata(args):
    """Read-only stdlib preparation for a production supervisor; no Torch/RGB decode.

    args needs run_dir and smoke_report. Result identity is deterministic and
    resume-independent. All official TRAIN/VAL image bytes are bound here;
    the runtime verifies the actual adapter inventory against these exact IDs.
    """
    if "torch" in sys.modules:
        raise RuntimeError("Prepare formal training metadata before importing Torch")
    run_dir, report = _paths(args)
    expected = {str(path): digest for path, digest in PINS.items()}
    expected[str(Path(__file__).resolve())] = sha(__file__)
    # The production adapter pins this trainer; record its actual bytes without
    # a circular pair of embedded hashes. It is verified again by the parent.
    expected[str(TRAIN_WATCH)] = sha(TRAIN_WATCH)
    _verify(expected)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from query_model_runtime import local_model_metadata
    from query_papple_data import ANNOTATION_SHA256
    model = local_model_metadata()
    smoke = _smoke_evidence(report)
    if smoke["official_source"] != model["source_identity"]:
        raise ValueError("Fresh official model identity differs from the actual passed GPU smoke")
    expected.update(model["source_sha256_expected"])
    expected.update(smoke["source_sha256"])
    expected.update(smoke["output_sha256"])
    expected[str(SMOKE_SUPERVISOR)] = smoke["supervisor_sha256"]
    inventory = {}
    provenance = read(ROOT / "reports" / "papple_val_provenance_20261021.json")
    frozen_val = {entry["image_id"]: entry for entry in provenance["image_metadata"]}
    for split, count in (("train", 2304), ("val", 814)):
        paired = {}
        for branch, filename in (("visible", "via_region_data_instance.json"), ("amodal", "via_region_data_amodal.json")):
            path = PAPPLE / "gt_json" / split / filename
            expected[str(path)] = ANNOTATION_SHA256[split][branch]
            if sha(path) != expected[str(path)]:
                raise ValueError("Fixed genuine public PApple polygon bytes changed")
            raw = read(path)
            raw = raw.get("_via_img_metadata", raw)
            rows = {}
            for value in raw.values():
                name = value["filename"]
                if name in rows or not re.fullmatch(r"_MG_[0-9]+_[0-9]+\.png", name):
                    raise ValueError("Duplicated or unexpected genuine PApple filename")
                regions = value.get("regions", {})
                values = regions.values() if isinstance(regions, dict) else regions
                ids = sorted({str(region["region_attributes"]["apple_ID"]).strip() for region in values})
                if any(not value or value == "0" for value in ids) or len(ids) > 100:
                    raise ValueError("Invalid IDs or GT count exceeds queries; no exclusion is allowed")
                rows[name] = ids
            paired[branch] = rows
        if paired["visible"] != paired["amodal"] or len(paired["visible"]) != count:
            raise ValueError("Complete public paired TRAIN2304/VAL814 inventory required")
        inventory[split] = [{"filename": name, "apple_ids": ids} for name, ids in sorted(paired["visible"].items())]
        for row in inventory[split]:
            image = PAPPLE / "images" / split / row["filename"]
            expected[str(image)] = frozen_val[row["filename"]]["RGB_sha256"] if split == "val" else sha(image)
    actual = _verify(expected)
    identity = {"schema": SCHEMA, "root": str(ROOT), "run_dir": str(run_dir), "data_root": str(PAPPLE),
                "recipe": RECIPE, "official_source": model["source_identity"],
                "smoke_report_sha256": smoke["report_sha256"], "smoke_supervisor_sha256": smoke["supervisor_sha256"],
                "data_inventory_sha256": hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    return {"identity": identity, "source_sha256_expected": actual, "data_inventory": inventory,
            "smoke_evidence": smoke, "model_metadata": model}


def _launch_guard():
    if (Path(sys.executable).resolve() != PYTHON.resolve()
            or os.environ.get("PYTHONPATH") != str(SITE_PACKAGES)):
        raise RuntimeError("Use real base F: Python with exact external query-dual site-packages PYTHONPATH")
    for name in ("transformers", "safetensors", "tokenizers", "huggingface_hub", "regex", "tqdm"):
        spec = util.find_spec(name)
        if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(SITE_PACKAGES.resolve()):
            raise RuntimeError("New isolated query dependency must resolve inside query_dual_v1: " + name)
    torch_spec = util.find_spec("torch")
    if (torch_spec is None or not torch_spec.origin
            or not Path(torch_spec.origin).resolve().is_relative_to((PYTHON.parent / "Lib" / "site-packages").resolve())):
        raise RuntimeError("Torch must remain the existing base-environment installation")


def _atomic_json(path, value, replace):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    replace(temporary, path)


def _alias_checkpoint(source, target, replace):
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    replace(temporary, target)


def _lr_factor(successful_updates):
    step = successful_updates + 1
    if step <= RECIPE["warmup_successful_updates"]:
        return step / RECIPE["warmup_successful_updates"]
    progress = (step - RECIPE["warmup_successful_updates"]) / (RECIPE["planned_updates"] - RECIPE["warmup_successful_updates"])
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _rng_state(torch, np, loader_generator, criterion_generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(), "loader_generator": loader_generator.get_state(),
            "criterion_generator": criterion_generator.get_state()}


def _restore_rng(state, torch, np, loader_generator, criterion_generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    loader_generator.set_state(state["loader_generator"])
    criterion_generator.set_state(state["criterion_generator"])


def _cpu_snapshot(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item, torch) for item in value)
    return value


def _dataset_check(dataset, split, inventory):
    config = dataset.configuration
    if (config != {"split": split, "input_size": 768, "batch_size": 1, "augment": False, "external_padding": False}
            or len(dataset) != len(inventory)):
        raise ValueError("Runtime adapter changed the fixed full official split recipe")
    for index, expected in enumerate(inventory):
        info = dataset.image_metadata(index)
        if (info["filename"] != expected["filename"] or list(info["apple_ids"]) != expected["apple_ids"]
                or tuple(info["original_size"]) != (1300, 1300) or tuple(info["input_size"]) != (768, 768)):
            raise ValueError("Runtime adapter filtered/reordered original official IDs or image geometry")


def _checkpoint_entry(run_dir, entry, run_id):
    path = Path(entry["path"]).resolve()
    if (path.parent != (run_dir / "checkpoints").resolve() or entry.get("run_id") != run_id
            or not re.fullmatch(fr"epoch_{entry['epoch']:03d}_attempt_[0-9]+\.pt", path.name)
            or sha(path) != entry["sha256"]):
        raise ValueError("Only this run's immutable, hash-verified complete epoch checkpoint is accepted")
    return path


def _commit_checkpoint(payload, index, run_dir, is_best, torch, replace):
    path = run_dir / "checkpoints" / f"epoch_{payload['completed_epoch']:03d}_attempt_{time.time_ns()}.pt"
    if path.exists():
        raise FileExistsError("Completed checkpoint evidence is immutable: " + str(path))
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    snapshot = _cpu_snapshot(payload, torch)
    with temporary.open("wb") as stream:
        torch.save(snapshot, stream)
        stream.flush()
        os.fsync(stream.fileno())
    del snapshot
    replace(temporary, path)
    entry = {"path": str(path), "sha256": sha(path), "epoch": payload["completed_epoch"],
             "run_id": payload["run_config"]["run_id"], "score": payload["validation"]["score"]}
    index["last"] = entry
    if is_best:
        index["best"] = entry
    # The index is the commit point; aliases can be recreated from immutable bytes.
    _atomic_json(run_dir / "checkpoint_index.json", index, replace)
    _alias_checkpoint(Path(index["last"]["path"]), run_dir / "last.pt", replace)
    _alias_checkpoint(Path(index["best"]["path"]), run_dir / "best.pt", replace)
    return index


def _resume_checkpoint(run_dir, config, evidence, torch):
    index = read(run_dir / "checkpoint_index.json")
    if index.get("schema") != SCHEMA or index.get("run_id") != config["run_id"]:
        raise ValueError("Resume index belongs to a different run/schema")
    path = _checkpoint_entry(run_dir, index["last"], config["run_id"])
    _checkpoint_entry(run_dir, index["best"], config["run_id"])
    # weights_only=False is restricted to our own committed file, never user paths.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    completed = checkpoint.get("completed_epoch", 0)
    if (checkpoint.get("schema") != SCHEMA or checkpoint.get("run_config") != config
            or checkpoint.get("source_sha256") != evidence["source_sha256_expected"]
            or checkpoint.get("complete_train_epoch") is not True or checkpoint.get("complete_full_val") is not True
            or checkpoint.get("micro_batches_in_accumulation") != 0 or completed != index["last"]["epoch"]
            or not 1 <= completed <= 6 or len(checkpoint.get("history", [])) != completed
            or checkpoint["updates"]["attempted"] != completed * 576
            or checkpoint["updates"]["successful"] + checkpoint["updates"]["skipped"] != checkpoint["updates"]["attempted"]
            or len(checkpoint.get("update_history", [])) != checkpoint["updates"]["attempted"]
            or checkpoint["validation"]["summary"]["images"] != 814
            or checkpoint["validation"]["summary"]["gt_count"] != 3449):
        raise ValueError("Only same-config/source, complete full-epoch/full-VAL own-run resume is supported")
    for row in checkpoint["history"]:
        if row["train_images"] != 2304 or row["validation"]["summary"]["images"] != 814 or row["validation"]["summary"]["gt_count"] != 3449:
            raise ValueError("Prior history contains a subset or incomplete epoch")
    _verify(checkpoint["validation_files_sha256"])
    return checkpoint, index


def train(args, evidence=None):
    """Execute the fixed recipe; caller/supervisor owns external monitoring."""
    _launch_guard()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(HERE))
    import psutil
    from watch_query_dual_train import preflight
    preflight(psutil)  # Unique actual-child gate, before metadata, mkdir or Torch.
    evidence = metadata(args) if evidence is None else evidence
    run_dir, _ = _paths(args)
    if evidence["identity"]["run_dir"] != str(run_dir) or evidence["identity"]["recipe"] != RECIPE:
        raise ValueError("Prepared training identity differs from the fixed runtime request")
    _verify(evidence["source_sha256_expected"])
    from runtime import configure_paths
    if Path(configure_paths()).resolve() != ROOT:
        raise RuntimeError("Fixed authorized F: project root required")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_CACHE=str(ROOT / "cache" / "query_hf" / "hub"),
                      HUGGINGFACE_HUB_CACHE=str(ROOT / "cache" / "query_hf" / "hub"),
                      TORCH_EXTENSIONS_DIR=str(ROOT / "cache" / "query_torch_extensions"))
    from train_refinement import TrainingLock
    from atomic_io import replace_with_retry
    with TrainingLock(SCRIPTS / "training.lock", run_dir, psutil):
        if args.resume:
            config = read(run_dir / "run_config.json")
            if config.get("identity") != evidence["identity"] or config.get("source_sha256") != evidence["source_sha256_expected"]:
                raise ValueError("Strict resume requires identical run config, source, official data and smoke evidence")
        else:
            run_dir.mkdir(exist_ok=False)
            (run_dir / "checkpoints").mkdir()
            (run_dir / "validation").mkdir()
            config = {"schema": SCHEMA, "run_id": str(uuid.uuid4()), "created_utc": utc_now(),
                      "identity": evidence["identity"], "source_sha256": evidence["source_sha256_expected"],
                      "smoke_evidence": evidence["smoke_evidence"], "fresh_official_initialization": True}
            _atomic_json(run_dir / "run_config.json", config, replace_with_retry)
        status = {"schema": SCHEMA, "phase": "initializing", "run_id": config["run_id"],
                  "run_dir": str(run_dir), "started_utc": utc_now(), "pid": os.getpid(),
                  "process_create_time": psutil.Process().create_time(), "resume": bool(args.resume),
                  "epoch": 0, "image_in_epoch": 0, "completed_epochs": 0,
                  "attempted_updates": 0, "successful_updates": 0, "skipped_updates": 0}
        _atomic_json(run_dir / "training_status.json", status, replace_with_retry)
        try:
            from runtime import authorized_guard
            torch = authorized_guard()
            import numpy as np
            from query_model_runtime import load_verified_query_model
            from query_papple_data import build_query_papple_dataset, collate_query_papple_batch
            from query_dual_criterion import build_query_dual_criterion
            from query_dual_evaluate import evaluate_full_papple
            versions = {name: version(name) for name in ("torch", "transformers", "scipy", "safetensors", "numpy", "Pillow")}
            if versions["torch"] != "2.12.1+cu132" or versions["transformers"] != "4.57.1":
                raise RuntimeError("Exact GPU-proven Torch/Transformers runtime required")
            torch.set_num_threads(2)
            torch.set_num_interop_threads(2)
            random.seed(RECIPE["seed"])
            np.random.seed(RECIPE["seed"])
            torch.manual_seed(RECIPE["seed"])
            torch.cuda.manual_seed_all(RECIPE["seed"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            train_data = build_query_papple_dataset(PAPPLE, "train", input_size=768)
            val_data = build_query_papple_dataset(PAPPLE, "val", input_size=768)
            _dataset_check(train_data, "train", evidence["data_inventory"]["train"])
            _dataset_check(val_data, "val", evidence["data_inventory"]["val"])
            model, loading = load_verified_query_model(device="cpu")
            if loading["source_identity"] != evidence["model_metadata"]["source_identity"]:
                raise ValueError("Formal initialization must come from the same verified official safetensors")
            model.cuda()
            criterion = build_query_dual_criterion().cuda()
            if criterion.configuration != RECIPE["criterion"]:
                raise ValueError("Keep the complete default 12544-point shared V/A criterion")
            backbone_ids = {id(parameter) for parameter in model.shared.pixel_level_module.encoder.parameters()}
            backbone, others = [], []
            for parameter in model.parameters():
                if not parameter.requires_grad:
                    raise ValueError("The fixed recipe trains every candidate parameter")
                (backbone if id(parameter) in backbone_ids else others).append(parameter)
            if not backbone or not others or len(backbone) + len(others) != len(list(model.parameters())):
                raise ValueError("Disjoint, complete encoder/other AdamW groups required")
            optimizer = torch.optim.AdamW([
                {"params": backbone, "lr": 1e-5, "base_lr": 1e-5, "name": "backbone"},
                {"params": others, "lr": 1e-4, "base_lr": 1e-4, "name": "other"}],
                betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05, foreach=False)
            scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
            loader_generator = torch.Generator(device="cpu").manual_seed(RECIPE["seed"])
            criterion_generator = torch.Generator(device="cuda").manual_seed(RECIPE["seed"])
            loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0,
                pin_memory=False, drop_last=False, generator=loader_generator, collate_fn=collate_query_papple_batch)
            history, update_history = [], []
            updates = {"attempted": 0, "successful": 0, "skipped": 0}
            index = {"schema": SCHEMA, "run_id": config["run_id"]}
            start_epoch = 1
            if args.resume:
                checkpoint, index = _resume_checkpoint(run_dir, config, evidence, torch)
                model.load_state_dict(checkpoint["model"], strict=True)
                optimizer.load_state_dict(checkpoint["optimizer"])
                scaler.load_state_dict(checkpoint["scaler"])
                if (checkpoint["model_config"] != loading["outer_config"]
                        or checkpoint["criterion_config"] != criterion.configuration
                        or checkpoint["runtime_versions"] != versions):
                    raise ValueError("Resume model/criterion/runtime configuration changed")
                history, update_history, updates = checkpoint["history"], checkpoint["update_history"], checkpoint["updates"]
                start_epoch = checkpoint["completed_epoch"] + 1
                # Loader is already constructed; no iterator has been created yet.
                _restore_rng(checkpoint["rng"], torch, np, loader_generator, criterion_generator)
                del checkpoint
                _alias_checkpoint(Path(index["last"]["path"]), run_dir / "last.pt", replace_with_retry)
                _alias_checkpoint(Path(index["best"]["path"]), run_dir / "best.pt", replace_with_retry)
            step_calls = {"count": 0}
            def did_optimizer_step(_optimizer, _args, _kwargs):
                step_calls["count"] += 1
            hook = optimizer.register_step_post_hook(did_optimizer_step)
            status.update(runtime_versions=versions, official_initialization=loading["initialization"])
            for epoch in range(start_epoch, 7):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                consumed, order, totals = 0, [], {}
                epoch_started = time.monotonic()
                status.update(phase="training", epoch=epoch, image_in_epoch=0, completed_epochs=epoch - 1)
                for batch in loader:
                    consumed += 1
                    name = batch["metadata"][0]["filename"]
                    order.append(name)
                    for key in ("labels", "mask_valid", "visible_masks", "amodal_masks"):
                        if batch["targets"][0][key].device.type != "cpu":
                            raise ValueError("Original target tensors stay on CPU; criterion owns point sampling transfer")
                    pixels = batch["pixel_values"].cuda()
                    pixel_mask = batch["pixel_mask"].cuda()
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(pixel_values=pixels, pixel_mask=pixel_mask)
                        losses = criterion(outputs, batch["targets"], generator=criterion_generator)
                        total = sum(losses.values())
                    if len(losses) != 5 or not bool(torch.isfinite(total)):
                        raise ValueError("Five finite weighted genuine paired-mask losses required")
                    for key, value in losses.items():
                        totals[key] = totals.get(key, 0.0) + float(value.detach())
                    scaler.scale(total / 4).backward()
                    if consumed % 4 == 0:
                        updates["attempted"] += 1
                        factor = _lr_factor(updates["successful"])
                        for group in optimizer.param_groups:
                            group["lr"] = group["base_lr"] * factor
                        scaler.unscale_(optimizer)
                        finite_gradients = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
                        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=True)) if finite_gradients else None
                        previous_steps, scale_before = step_calls["count"], scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        executed = step_calls["count"] - previous_steps
                        if executed not in (0, 1) or executed and not finite_gradients:
                            raise RuntimeError("AMP optimizer execution disagrees with finite unscaled gradients")
                        updates["successful"] += executed
                        updates["skipped"] += 1 - executed
                        update_history.append({"epoch": epoch, "image_in_epoch": consumed, **updates,
                            "optimizer_step_executed": bool(executed), "lr_factor": factor,
                            "lr_backbone": optimizer.param_groups[0]["lr"], "lr_other": optimizer.param_groups[1]["lr"],
                            "scale_before": scale_before, "scale_after": scaler.get_scale(), "unclipped_gradient_norm": norm})
                        optimizer.zero_grad(set_to_none=True)
                    status.update(image_in_epoch=consumed, attempted_updates=updates["attempted"],
                                  successful_updates=updates["successful"], skipped_updates=updates["skipped"])
                    if consumed % 25 == 0 or consumed == 2304:
                        _atomic_json(run_dir / "training_status.json", status, replace_with_retry)
                        print(f"QUERY TRAIN epoch={epoch}/6 images={consumed}/2304 loss={sum(totals.values())/consumed:.5f} "
                              f"updates={updates['successful']}/{updates['attempted']} skipped={updates['skipped']} scale={scaler.get_scale():.1f}", flush=True)
                    del batch, pixels, pixel_mask, outputs, losses, total
                expected_names = {row["filename"] for row in evidence["data_inventory"]["train"]}
                if consumed != 2304 or consumed % 4 or len(order) != len(set(order)) or set(order) != expected_names:
                    raise AssertionError("Every epoch must consume all 2304 official TRAIN images exactly once")
                status.update(phase="validating", image_in_epoch=2304)
                _atomic_json(run_dir / "training_status.json", status, replace_with_retry)
                validation_dir = run_dir / "validation" / f"epoch_{epoch:03d}_attempt_{time.time_ns()}"
                validation = evaluate_full_papple(model, val_data, "cuda:0", validation_dir)
                if (validation["summary"]["images"] != 814 or validation["summary"]["gt_count"] != 3449
                        or validation["summary"]["empty_gt_images"] != 13
                        or validation["score"] != min(validation["summary"]["precision"], validation["summary"]["recall"])):
                    raise AssertionError("Selection requires complete VAL814/GT3449 with one global BOX threshold")
                validation_files = {str(validation_dir / filename): sha(validation_dir / filename) for filename in validation["files"].values()}
                actual = _verify(evidence["source_sha256_expected"])
                row = {"epoch": epoch, "train_images": consumed, "train_order": order,
                       "train_loss_mean": {key: value / consumed for key, value in totals.items()},
                       "updates": dict(updates), "validation": validation, "validation_directory": str(validation_dir),
                       "validation_files_sha256": validation_files, "elapsed_seconds": time.monotonic() - epoch_started}
                history.append(row)
                ranking = lambda value: (value["validation"]["score"], value["validation"]["summary"]["f1"], value["validation"]["confidence"])
                is_best = len(history) == 1 or ranking(row) > max(ranking(previous) for previous in history[:-1])
                payload = {"schema": SCHEMA, "run_config": config, "model_config": loading["outer_config"],
                           "architecture_metadata": model.candidate_metadata, "criterion_config": criterion.configuration,
                           "runtime_versions": versions, "source_sha256": actual, "completed_epoch": epoch,
                           "complete_train_epoch": True, "complete_full_val": True, "micro_batches_in_accumulation": 0,
                           "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                           "rng": _rng_state(torch, np, loader_generator, criterion_generator),
                           "history": history, "updates": dict(updates), "update_history": update_history,
                           "validation": validation, "validation_directory": str(validation_dir),
                           "validation_files_sha256": validation_files}
                index = _commit_checkpoint(payload, index, run_dir, is_best, torch, replace_with_retry)
                del payload
                _atomic_json(run_dir / "history.json", history, replace_with_retry)
                _atomic_json(run_dir / "update_history.json", update_history, replace_with_retry)
                status.update(phase="epoch_complete", completed_epochs=epoch, checkpoint_index=index)
                _atomic_json(run_dir / "training_status.json", status, replace_with_retry)
                print(f"QUERY EPOCH {epoch}/6 BOX P={validation['summary']['precision']:.6f} R={validation['summary']['recall']:.6f} "
                      f"minPR={validation['score']:.6f} confidence={validation['confidence']:.6f} best_epoch={index['best']['epoch']}", flush=True)
            hook.remove()
            if len(history) != 6 or updates["attempted"] != 3456 or updates["successful"] + updates["skipped"] != 3456:
                raise AssertionError("Fixed six complete epochs and 3456 attempted accumulated updates required")
            actual = _verify(evidence["source_sha256_expected"])
            summary = {"schema": SCHEMA, "phase": "complete", "run_id": config["run_id"], "run_dir": str(run_dir),
                       "epochs_completed": 6, "recipe": RECIPE, "updates": updates, "history": history,
                       "last": index["last"], "best": index["best"],
                       "last_alias": str(run_dir / "last.pt"), "best_alias": str(run_dir / "best.pt"),
                       "source_sha256_before": evidence["source_sha256_expected"], "source_sha256_after": actual,
                       "source_before_after_identical": True, "smoke_weights_used_for_initialization": False,
                       "test_used": False, "completed_utc": utc_now()}
            _atomic_json(run_dir / "training_summary.json", summary, replace_with_retry)
            status.update(phase="complete", epoch=6, image_in_epoch=2304, completed_epochs=6, completed_utc=utc_now(),
                          attempted_updates=updates["attempted"], successful_updates=updates["successful"], skipped_updates=updates["skipped"],
                          source_sha256_after=actual, source_before_after_identical=True, checkpoint_index=index)
            _atomic_json(run_dir / "training_status.json", status, replace_with_retry)
            return summary
        except BaseException as error:
            status.update(phase="failed", error=repr(error), traceback=traceback.format_exc(), failed_utc=utc_now(),
                          no_partial_epoch_checkpoint_written=True)
            try:
                actual = {path: sha(path) for path in evidence["source_sha256_expected"]}
                status.update(source_sha256_after=actual, source_before_after_identical=actual == evidence["source_sha256_expected"])
            except BaseException as source_error:
                status["source_hash_failure"] = repr(source_error)
            if "updates" in locals():
                status.update(attempted_updates=updates["attempted"], successful_updates=updates["successful"], skipped_updates=updates["skipped"])
                _atomic_json(run_dir / "failed_update_history.json", update_history, replace_with_retry)
            _atomic_json(run_dir / "training_status.json", status, replace_with_retry)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
