"""Bounded, train-only shared-trunk BN diagnostic; never an optimizer run.

Default prepares a reproducible protocol without loading Torch or using a GPU.
--run recalibrates only shared-trunk BN statistics independently from final40
EMA for stage64, defect64 and balanced64+64, then validates both full dev sets.
Source-only variants are diagnostic interventions, not deployable oracle gates.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
JOINT_DIR = ROOT / "apple_research_v2/joint"
DEFAULT_RUN = ROOT / "runs/joint_shared_e40_s20261106_v1"
DEFAULT_MANIFEST = ROOT / "experiments/joint_training_v1/data_manifest.json"
SEED = 20261112
COUNT = 64
BATCH = 4
VARIANTS = ("stage_only64", "defect_only64", "balanced64_64")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def in_project(path):
    path = Path(path).resolve()
    require(path.drive.lower() == "f:" and path.is_relative_to(ROOT), "Path outside F research root")
    return path


def select_rows(records, source):
    rows = [r for r in records if r["source"] == source and r["split"] == "train"]
    require(len(rows) >= COUNT, "Too few train images")
    rows.sort(key=lambda r: (hashlib.sha256(f'{SEED}:{source}:{r["sample_id"]}'.encode()).hexdigest(), r["sample_id"]))
    selected = rows[:COUNT]
    require(len({r["image_path"] for r in selected}) == COUNT, "Duplicate calibration image")
    return [{k: r[k] for k in ("sample_id", "source", "split", "image_path", "image_sha256", "label_path", "label_sha256")} for r in selected]


def batch_plan(selected, variant):
    tasks = ("stage",) if variant == "stage_only64" else (("defect",) if variant == "defect_only64" else ("stage", "defect"))
    require(variant in VARIANTS, "Unknown diagnostic variant")
    return [selected[task][start:start + BATCH] for start in range(0, COUNT, BATCH) for task in tasks]


def prepare_protocol(run_dir, manifest_path):
    run_dir, manifest_path = in_project(run_dir), in_project(manifest_path)
    report, status, manifest = read(run_dir / "training_report.json"), read(run_dir / "status.json"), read(manifest_path)
    require(report["phase"] == status["phase"] == "complete" and report["epochs_completed"] == 40, "Incomplete source model")
    require(report["recipe"]["mode"] == "joint", "Shared model required")
    require(sha(manifest_path) == report["recipe"]["data_manifest_sha256"], "Manifest identity mismatch")
    checkpoint = in_project(report["final_checkpoint"]["file"])
    require(checkpoint.is_relative_to(run_dir) and checkpoint.name == "final40_complete.pt", "Wrong checkpoint")
    require(sha(checkpoint) == report["final_checkpoint"]["sha256"] == read(checkpoint.with_suffix(".sha256.json"))["sha256"], "Checkpoint identity mismatch")
    selected = {task: select_rows(manifest["records"], task) for task in ("stage", "defect")}
    for rows in selected.values():
        for row in rows:
            require(sha(in_project(row["image_path"])) == row["image_sha256"], "Calibration image changed")
            require(sha(in_project(row["label_path"])) == row["label_sha256"], "Calibration annotation changed")
    baseline = {v["task"]: {"metrics": v["receipt"]["metrics"], "per_class": v["receipt"]["per_class"], "seen": v["receipt"]["seen"]}
                for v in report["validation"] if v["epoch"] == 40 and v["variant"] == "ema" and v["primary"]}
    require(set(baseline) == {"stage", "defect"}, "Missing original final40 EMA results")
    return {
        "schema": "shared_trunk_bn_diagnostic_v1", "phase": "prepared_cpu_only", "script_sha256": sha(__file__),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "training_report": str(run_dir / "training_report.json"), "training_report_sha256": sha(run_dir / "training_report.json"),
        "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "seed": SEED,
        "samples_per_source": COUNT, "batch_size": BATCH, "imgsz": 640, "precision": "FP32", "optimizer": None,
        "scope": "Diagnostic only; development data are not physical-fruit-independent. No deployment promotion.",
        "normalization_scope": "Only BatchNorm modules under trunks.shared; head BN and all parameters frozen.",
        "bn_reset": "running_mean=0, running_var=1, num_batches_tracked=0; momentum=None (PyTorch cumulative per-batch mean/variance)",
        "preprocess": "Existing JointInferenceAdapter stage predictor: native BGR uint8 -> RGB FP32 square-letterbox640 /255, no augmentation.",
        "calibration_forward": "shared features only, no task head forward; all non-BN modules eval; no gradients",
        "selection": "SHA256(seed:source:sample_id) ordering; exactly64 TRAIN images/source; labels never used for statistics",
        "balanced_order": "Alternate one 4-image stage batch and one 4-image defect batch, 16 pairs",
        "evaluation": "Existing official evaluate_task, full stage72/defect113; FP32 imgsz640 conf.001 maxdet300; no validation images used for BN",
        "source_only_warning": "stage-only and defect-only variants diagnose domain bias. Choosing stats by true source is an oracle and not a deployed method.",
        "original_baseline": baseline, "selected": selected,
        "variants": {name: [[r["sample_id"] for r in batch] for batch in batch_plan(selected, name)] for name in VARIANTS},
    }


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def recalibrate_bn(model, target_module, batches, forward):
    """Update only selected BN buffers, prove every parameter/other buffer fixed."""
    import torch
    from torch import nn
    require(not any(p.requires_grad for p in model.parameters()), "Freeze all parameters before calibration")
    model.eval()
    selected = [(name, module) for name, module in model.named_modules()
                if isinstance(module, nn.modules.batchnorm._BatchNorm) and any(module is child for child in target_module.modules())]
    require(selected, "No target BN modules")
    require(all(m.track_running_stats and m.running_mean is not None for _, m in selected), "BN running statistics unavailable")
    bn_keys = {f"{name}.{suffix}" for name, _ in selected for suffix in ("running_mean", "running_var", "num_batches_tracked")}
    before_state = model.state_dict()
    before = {"parameters": state_hash(dict(model.named_parameters())),
              "bn_buffers": state_hash({k: v for k, v in before_state.items() if k in bn_keys}),
              "other_state": state_hash({k: v for k, v in before_state.items() if k not in bn_keys})}
    momentum = {name: module.momentum for name, module in selected}
    count = examples = 0
    try:
        for _, module in selected:
            module.reset_running_stats(); module.momentum = None; module.train()
        with torch.no_grad():
            for tensor in batches:
                require(tensor.ndim == 4 and len(tensor) == BATCH and bool(torch.isfinite(tensor).all()), "Invalid calibration batch")
                forward(tensor)
                count += 1; examples += len(tensor)
    finally:
        for name, module in selected:
            module.momentum = momentum[name]; module.eval()
    require(count > 0, "No calibration batches")
    for name, module in selected:
        require(int(module.num_batches_tracked) == count, "BN did not see each batch exactly once: " + name)
        require(bool(torch.isfinite(module.running_mean).all()) and bool(torch.isfinite(module.running_var).all()) and bool((module.running_var >= 0).all()), "Invalid BN statistics")
    after_state = model.state_dict()
    after = {"parameters": state_hash(dict(model.named_parameters())),
             "bn_buffers": state_hash({k: v for k, v in after_state.items() if k in bn_keys}),
             "other_state": state_hash({k: v for k, v in after_state.items() if k not in bn_keys})}
    require(before["parameters"] == after["parameters"] and before["other_state"] == after["other_state"], "Calibration mutated parameter or non-target buffer")
    require(not any(module.training for module in model.modules()), "Calibration failed to restore evaluation mode")
    require(all(p.grad is None for p in model.parameters()), "Calibration created gradients")
    return {"before": before, "after": after, "bn_module_count": len(selected), "bn_buffer_keys": sorted(bn_keys),
            "batches": count, "examples": examples, "parameters_unchanged": True, "non_target_state_unchanged": True,
            "no_gradients": True, "bn_changed": before["bn_buffers"] != after["bn_buffers"],
            "restored_momentum": momentum, "all_eval_after": True}


def cpu_self_test():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from torch import nn
    require(not torch.cuda.is_initialized(), "Self-test must remain CPU only")
    torch.manual_seed(SEED)
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(nn.Conv2d(3, 5, 1), nn.BatchNorm2d(5), nn.Dropout2d(.5))
            self.head = nn.Sequential(nn.Conv2d(5, 2, 1), nn.BatchNorm2d(2))
        def forward(self, value):
            require(not self.trunk[2].training and not self.head.training, "Non-target module entered training mode")
            return self.head(self.trunk(value))
    model = Tiny().eval().requires_grad_(False)
    rows = [torch.randn(BATCH, 3, 8, 8) + i + 2 for i in range(3)]
    report = recalibrate_bn(model, model.trunk, rows, model)
    require(report["batches"] == 3 and report["examples"] == 12 and report["bn_changed"], "Self-test failed")
    require(int(model.head[1].num_batches_tracked) == 0 and model.trunk[1].momentum == .1, "Head BN/momentum changed")
    model.trunk[0].weight.requires_grad_(True)
    try:
        recalibrate_bn(model, model.trunk, rows, model)
    except ValueError as error:
        require("Freeze" in str(error), "Unexpected parameter-freeze rejection")
    else:
        raise AssertionError("Unfrozen parameter accepted")
    require(not torch.cuda.is_initialized(), "Self-test initialized CUDA")
    return {"phase": "cpu_self_test_passed", "BN_only_change": True, "head_BN_unchanged": True,
            "all_parameters_unchanged": True, "non_BN_eval": True, "momentum_restored": True,
            "unfrozen_parameters_rejected": True, "CUDA_initialized": False}


def run_diagnostic(protocol, output, device):
    import cv2
    import torch
    from joint_state_v1 import atomic_json, load_checkpoint, seed_everything
    from joint_model_v1 import build_model, tensor_sha
    from joint_eval_v1 import JointInferenceAdapter, evaluate_task
    require(torch.device(device).type == "cuda" and torch.cuda.is_available(), "Explicit CUDA run required; use --self-test for CPU")
    seed_everything(SEED)
    checkpoint = Path(protocol["checkpoint"])
    identity = read(Path(protocol["training_report"]).parent / "identity.json")
    for source, expected in identity["source_code"].items():
        require(sha(source) == expected, "Frozen model/evaluation code changed: " + source)
    source_files = read(protocol["manifest"])["source_hashes"]
    for path, expected in source_files.items():
        require(sha(path) == expected, "Frozen source file changed: " + path)
    result = {"schema": protocol["schema"], "phase": "running", "protocol": protocol,
              "environment": {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(device)},
              "source_files_verified": len(source_files), "variants": {}}
    atomic_json(output / "status.json", {"phase": "running", "variant": None})
    try:
        for variant in VARIANTS:
            case_dir = output / variant
            case_dir.mkdir(exist_ok=False)
            atomic_json(output / "status.json", {"phase": "calibrating", "variant": variant})
            state = load_checkpoint(checkpoint)
            require(state["mode"] == "joint" and state["epoch_completed"] == 40, "Unexpected checkpoint contents")
            model = build_model("joint", device="cpu")
            model.load_state_dict(state["ema"]["model"], strict=True)
            require(tensor_sha(model.state_dict()) == state["ema_tensor_sha256"], "EMA tensor reconstruction mismatch")
            del state
            model = model.to(device).float().eval().requires_grad_(False)
            adapter = JointInferenceAdapter(model, audit_values=False)
            plan = batch_plan(protocol["selected"], variant)
            def batches():
                for batch in plan:
                    arrays = [cv2.imread(row["image_path"]) for row in batch]
                    require(all(array is not None for array in arrays), "Unreadable calibration image")
                    tensor = adapter.predictors["stage"].preprocess(arrays)
                    require(tuple(tensor.shape) == (BATCH, 3, 640, 640) and tensor.dtype == torch.float32, "Preprocess contract changed")
                    yield tensor
            audit = recalibrate_bn(model, model.trunks["shared"], batches(), lambda x: model.features(x, "stage"))
            require(audit["examples"] == (128 if variant == "balanced64_64" else 64), "Incomplete calibration")
            atomic_json(case_dir / "calibration_audit.json", audit)
            buffers = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in audit["bn_buffer_keys"]}
            with (case_dir / "bn_statistics_only.pt").open("xb") as stream:
                torch.save({"schema": "diagnostic_bn_buffers_only_v1", "NOT_full_or_deployment_checkpoint": True,
                            "source_checkpoint_sha256": protocol["checkpoint_sha256"], "variant": variant, "buffers": buffers}, stream)
            entry = {"calibration": audit, "bn_artifact_sha256": sha(case_dir / "bn_statistics_only.pt"), "validation": {}}
            for task in ("stage", "defect"):
                atomic_json(output / "status.json", {"phase": "validating", "variant": variant, "task": task})
                receipt = evaluate_task(model, task, output_dir=case_dir / ("validation_" + task), device=device, batch=1, imgsz=640)
                baseline = protocol["original_baseline"][task]["metrics"]
                entry["validation"][task] = {"metrics": receipt["metrics"], "per_class": receipt["per_class"], "seen": receipt["seen"],
                    "delta_from_original_EMA_pp": {key: 100 * (value - baseline[key]) for key, value in receipt["metrics"].items()},
                    "receipt": str(case_dir / ("validation_" + task) / "validation_receipt.json")}
                require(state_hash(dict(model.named_parameters())) == audit["after"]["parameters"], "Validation altered parameters")
            result["variants"][variant] = entry
            atomic_json(output / "results_partial.json", result)
            print(json.dumps({"event": "variant_complete", "variant": variant, "validation": entry["validation"]}, ensure_ascii=False), flush=True)
            del adapter, model, buffers
            gc.collect(); torch.cuda.empty_cache()
        require(sha(checkpoint) == protocol["checkpoint_sha256"], "Source checkpoint changed")
        result["phase"] = "complete"
        result["promotion"] = "None. These are diagnostic interventions only; retain original independent model baseline."
        atomic_json(output / "results.json", result)
        atomic_json(output / "status.json", {"phase": "complete", "variants_completed": list(VARIANTS)})
    except BaseException as error:
        atomic_json(output / "status.json", {"phase": "failed", "error": repr(error), "traceback": traceback.format_exc(),
                                            "completed_variants": list(result["variants"])})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(cpu_self_test(), indent=2)); return
    if args.output is None:
        parser.error("--output must be a new directory inside the F project")
    output = in_project(args.output)
    require(not output.exists(), "Refusing to overwrite prior diagnostic")
    protocol = prepare_protocol(args.run_dir, args.manifest)
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(JOINT_DIR))
    from joint_state_v1 import configure, atomic_json
    configure(output, args.device if args.run else "cpu")
    atomic_json(output / "protocol.json", protocol)
    if args.run:
        run_diagnostic(protocol, output, args.device)
    else:
        atomic_json(output / "status.json", {"phase": "prepared_cpu_only", "GPU_used": False, "training_performed": False})
        print(json.dumps({"phase": "prepared_cpu_only", "output": str(output), "variants": list(VARIANTS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
