"""CPU-only independent acceptance of a replayed formal PApple epoch.

Reads existing steps/retry/checkpoint evidence; never starts training or modifies
checkpoints. Exit 0 means all listed checks pass; exit 1 means incomplete/failed.
CUDA RNG bytes are validated structurally, never restored onto a GPU here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

SOURCE_SHA = "3b80d4a866ffa9394134d031edee95cc72f3e55afe6cf7e72bab69b6bbeffa01"
SCHEMA = "local_apple_epoch_recovery_acceptance_v1"
FAILURE_IMAGES = ["_MG_6940_22.png", "_MG_7008_14.png"]
NUMERICAL_POLICY = {"version": "same_batch_finite_update_v2", "amp_attempts": 3,
    "scale_backoff": 0.5, "fp32_fallback_attempts": 1,
    "restore_registered_buffers_and_rng_before_retry": True,
    "failed_batch_skip_allowed": False, "clip_error_if_nonfinite": True}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def object_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def finite_numbers(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_numbers(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_numbers(x) for x in value)
    return True


def walk_tensors(value: Any, torch: Any, prefix: str = ""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from walk_tensors(item, torch, prefix + "/" + str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from walk_tensors(item, torch, prefix + "/" + str(index))


def checkpoint_audit(path: Path, torch: Any, np: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    formal = checkpoint["formal_training"]
    model = list(walk_tensors(checkpoint["state_dict"], torch))
    optimizer = list(walk_tensors(formal["optimizer"], torch))
    rng = formal["rng"]
    rng_errors = []
    try:
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(rng["numpy"])
        torch.Generator(device="cpu").set_state(rng["torch_cpu"])
    except Exception as error:
        rng_errors.append(repr(error))
    cuda_states = rng.get("torch_cuda")
    cuda_shape_ok = (isinstance(cuda_states, list) and len(cuda_states) > 0 and all(
        isinstance(x, torch.Tensor) and x.device.type == "cpu" and x.dtype == torch.uint8
        and x.ndim == 1 and x.numel() > 0 for x in cuda_states))
    if not cuda_shape_ok:
        rng_errors.append("CUDA RNG is not a nonempty list of CPU uint8 vectors")
    report = {
        "path": str(path), "sha256": digest(path),
        "completed_epochs": formal["completed_epochs"], "global_step": formal["global_step"],
        "steps_per_epoch": formal["steps_per_epoch"], "at_epoch_boundary": formal["at_epoch_boundary"],
        "model_tensor_count": len(model),
        "model_nonfinite": [key for key, value in model if not bool(torch.isfinite(value).all())],
        "optimizer_tensor_count": len(optimizer),
        "optimizer_parameter_state_count": len(formal["optimizer"]["state"]),
        "optimizer_nonfinite": [key for key, value in optimizer if not bool(torch.isfinite(value).all())],
        "optimizer_step_values": sorted({float(v) for key, v in optimizer if key.endswith("/step")}),
        "learning_rates": [group["lr"] for group in formal["optimizer"]["param_groups"]],
        "criterion": formal["criterion"], "scaler": formal["scaler"],
        "rng_errors": rng_errors,
        "cuda_rng_validation": "structure only; no CUDA initialized or CUDA RNG restoration attempted",
        "cuda_rng_state_bytes": [x.numel() for x in cuda_states] if cuda_shape_ok else None,
        "all_saved_tensors_on_cpu": all(value.device.type == "cpu" for _, value in walk_tensors(checkpoint, torch)),
    }
    return report, checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", "--run-dir", dest="run_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("cache/datasets/local_apple_v1/train_manifest.json"))
    parser.add_argument("--source-checkpoint", type=Path, default=Path("runs/local_visible_papple_20261104/train_to_001/checkpoints/epoch_001.pt"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--retry-log", type=Path)
    parser.add_argument("--epoch", type=int, default=2)
    args = parser.parse_args()
    if args.epoch != 2:
        parser.error("This acceptance contract specifically audits the replay of epoch 2")
    repo = Path(__file__).resolve().parents[1]
    for name in ("run_dir", "output", "manifest", "source_checkpoint", "checkpoint", "retry_log"):
        value = getattr(args, name)
        if value is not None:
            value = (repo / value).resolve() if not value.is_absolute() else value.resolve()
            if value.drive.upper() != "F:":
                parser.error(f"All evidence/output paths must be on F: {value}")
            setattr(args, name, value)
    args.checkpoint = args.checkpoint or args.run_dir / "checkpoints/epoch_002.pt"
    args.retry_log = args.retry_log or args.run_dir / "numerical_retries.jsonl"
    protected = {args.manifest, args.source_checkpoint, args.source_checkpoint.with_suffix(".json"),
                 args.checkpoint, args.checkpoint.with_suffix(".json"), args.retry_log,
                 args.run_dir / "steps.jsonl", args.run_dir / "migration_receipt.json"}
    if args.output.suffix.lower() != ".json" or args.output in protected:
        parser.error("--output must be a new report JSON, not a checkpoint or input evidence file")
    # Configure temporary/cache locations before importing scientific libraries.
    for key in ("TEMP", "TMP", "TMPDIR"):
        os.environ[key] = str(repo / "cache/tmp")
    os.environ["TORCH_HOME"] = str(repo / "cache/torch")
    os.environ["MPLCONFIGDIR"] = str(repo / "cache/matplotlib")
    import numpy as np
    import torch
    torch.set_num_threads(2)
    checks = []
    report: dict[str, Any] = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "CPU-only independent evidence audit; checkpoint files are read-only",
        "run_dir": str(args.run_dir), "checks": checks, "passed": False}

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    try:
        manifest = read_json(args.manifest)
        rows = manifest["samples"]
        by_id = {x["image_id"]: x for x in rows}
        check("official_train_counts", len(rows) == 2304 and len(by_id) == 2304
              and sum(x["instance_count"] for x in rows) == 8579
              and sum(x["instance_count"] == 0 for x in rows) == 14)
        steps_all = read_jsonl(args.run_dir / "steps.jsonl")
        steps = [x for x in steps_all if x["epoch"] == 2]
        check("epoch2_batches_exactly_1_to_1152", [x["batch"] for x in steps] == list(range(1, 1153)))
        check("epoch2_global_steps_exactly_1153_to_2304", [x["global_step"] for x in steps] == list(range(1153, 2305)))
        image_ids = [image_id for x in steps for image_id in x["image_ids"]]
        check("all_images_exactly_once", Counter(image_ids) == Counter(by_id.keys()), {"images_seen": len(image_ids), "unique": len(set(image_ids))})
        check("instance_total_exactly_8579", sum(x["instances"] for x in steps) == 8579)
        check("each_batch_instance_counts_match_manifest", all(x["instances"] == sum(by_id[i]["instance_count"] for i in x["image_ids"]) for x in steps))
        check("one_successful_optimizer_update_per_step", all(x.get("optimizer_updates") == 1 for x in steps))
        check("all_recorded_step_numbers_finite", all(finite_numbers(x) for x in steps))
        check("loss_norm_scale_valid", all(x["total_loss"] >= 0 and x["grad_norm_before_clip"] >= 0 and x["amp_scale"] > 0 for x in steps))
        check("attempt_counts_consistent", all(1 <= x.get("numerical_attempts", 0) <= 4
              and x.get("numerical_retries") == x.get("numerical_attempts", 0) - 1
              and isinstance(x.get("fp32_fallback_used"), bool)
              and x["fp32_fallback_used"] == (x["numerical_attempts"] == 4) for x in steps))
        source_report, source = checkpoint_audit(args.source_checkpoint, torch, np)
        report["source_checkpoint"] = source_report
        check("source_epoch1_sha_unchanged", source_report["sha256"] == SOURCE_SHA)
        check("source_epoch1_state_finite", not source_report["model_nonfinite"] and not source_report["optimizer_nonfinite"])
        check("source_epoch1_boundary", source_report["completed_epochs"] == 1 and source_report["global_step"] == 1152 and source_report["at_epoch_boundary"])
        generator = torch.Generator().manual_seed(source["formal_training"]["recipe"]["seed"] + 1000003)
        permutation = torch.randperm(len(rows), generator=generator).tolist()
        expected_ids = [rows[i]["image_id"] for i in permutation]
        check("original_seeded_epoch2_order_preserved", image_ids == expected_ids)
        failure_steps = [x for x in steps if x["batch"] == 658]
        check("original_failure_batch658_updated_once", len(failure_steps) == 1
              and failure_steps[0]["global_step"] == 1810
              and failure_steps[0]["image_ids"] == FAILURE_IMAGES
              and failure_steps[0].get("optimizer_updates") == 1)
        report["former_failure_batch_success_record"] = failure_steps[0] if failure_steps else None
        retries_all = read_jsonl(args.retry_log)
        retries = [x for x in retries_all if x.get("epoch") == 2]
        retries_by_batch: dict[int, list[dict[str, Any]]] = {}
        for item in retries:
            retries_by_batch.setdefault(item["batch"], []).append(item)
        check("retry_count_matches_success_logs", len(retries) == sum(x.get("numerical_retries", 0) for x in steps))
        retry_errors = []
        expected_scale = source_report["scaler"]["scale"]
        for step in steps:
            found = retries_by_batch.get(step["batch"], [])
            if [x.get("attempt") for x in found] != list(range(1, step.get("numerical_attempts", 0))):
                retry_errors.append(f"batch {step['batch']}: retry attempt sequence mismatch")
            for item in found:
                if not (item.get("global_step_completed") == step["global_step"] - 1
                        and item.get("image_ids") == step["image_ids"]
                        and item.get("optimizer_updated") is False
                        and item.get("registered_buffers_and_rng_restored") is True
                        and item.get("will_retry_same_batch") is True
                        and item.get("amp_enabled") is True and item.get("fp32_fallback") is False
                        and item.get("old_scale") == expected_scale
                        and item.get("new_scale") == expected_scale * 0.5
                        and isinstance(item.get("gradient_diagnostics"), dict)):
                    retry_errors.append(f"batch {step['batch']}: retry safety fields invalid")
                expected_scale *= 0.5
            if step["amp_scale"] != expected_scale:
                retry_errors.append(f"batch {step['batch']}: successful scale differs from audited backoff sequence")
        if any(batch not in {x["batch"] for x in steps} for batch in retries_by_batch):
            retry_errors.append("Retry without a successful step in this epoch")
        check("same_batch_retry_safety_evidence", not retry_errors, retry_errors)
        report["retry_summary"] = {"path": str(args.retry_log), "failed_attempts": len(retries),
            "affected_batches": sorted(retries_by_batch),
            "fp32_fallback_success_batches": [x["batch"] for x in steps if x.get("fp32_fallback_used")],
            "first_retry_diagnostics": retries[0].get("gradient_diagnostics") if retries else None,
            "former_failure_batch_retries": retries_by_batch.get(658, []),
            "proof_limit": "Checks persisted safety fields and batch/update accounting; source review and engineering tests establish rollback implementation."}
        current_report, current = checkpoint_audit(args.checkpoint, torch, np)
        report["epoch2_checkpoint"] = current_report
        receipt = read_json(args.checkpoint.with_suffix(".json"))
        check("epoch2_checkpoint_sha_matches_sidecar", current_report["sha256"] == receipt["sha256"])
        check("epoch2_checkpoint_boundary", current_report["completed_epochs"] == 2 and current_report["global_step"] == 2304
              and current_report["steps_per_epoch"] == 1152 and current_report["at_epoch_boundary"])
        check("epoch2_checkpoint_all_tensors_finite", not current_report["model_nonfinite"] and not current_report["optimizer_nonfinite"])
        check("epoch2_optimizer_steps_exact", current_report["optimizer_step_values"] == [2304.0])
        check("checkpoint_state_tensor_counts_preserved", current_report["model_tensor_count"] == source_report["model_tensor_count"]
              and current_report["optimizer_parameter_state_count"] == source_report["optimizer_parameter_state_count"])
        check("epoch2_criterion_updates", current_report["criterion"]["updates"] == 2)
        check("checkpoint_rng_restorable_on_cpu_and_cuda_structurally_valid", not source_report["rng_errors"] and not current_report["rng_errors"],
              {"source": source_report["rng_errors"], "epoch2": current_report["rng_errors"]})
        check("checkpoint_scalar_state_finite", finite_numbers(current_report["scaler"]) and finite_numbers(current_report["criterion"])
              and current_report["scaler"]["scale"] > 0)
        check("learning_rate_unchanged", current_report["learning_rates"] == source_report["learning_rates"] == [0.0001])
        old_recipe = source["formal_training"]["recipe"]
        new_recipe = current["formal_training"]["recipe"]
        # Only these original fields may differ for the bounded numerical repair.
        allowed_changed_old_keys = {"sources", "amp_scale_policy"}
        differences = {key: {"before": value, "after": new_recipe.get(key)} for key, value in old_recipe.items()
                       if value != new_recipe.get(key)}
        check("original_training_recipe_unchanged_except_explicit_engineering_fields", set(differences) <= allowed_changed_old_keys, differences)
        check("only_bounded_numerical_policy_added_to_recipe", set(new_recipe) - set(old_recipe) == {"numerical_policy"}
              and new_recipe.get("numerical_policy") == NUMERICAL_POLICY
              and new_recipe.get("amp_scale_policy") == "no scale growth inside fixed budget; bounded same-batch retry then one FP32 fallback")
        check("data_adapter_source_unchanged", new_recipe["sources"]["local_training.py"] == old_recipe["sources"]["local_training.py"])
        check("only_numerics_and_artifact_io_sources_added", set(new_recipe["sources"]) == set(old_recipe["sources"]) | {"numerics.py", "artifact_io.py"})
        check("recovery_sources_match_current_files", all(digest(repo / "src/apple_local_rgbd" / name) == sha
              for name, sha in new_recipe["sources"].items()))
        check("manifest_matches_checkpoint", digest(args.manifest) == new_recipe["manifest_sha256"] == old_recipe["manifest_sha256"])
        migration_path = args.run_dir / "migration_receipt.json"
        report["migration_receipt"] = read_json(migration_path) if migration_path.exists() else None
        check("explicit_migration_receipt_present", migration_path.exists())
        migration = report["migration_receipt"] or {}
        check("migration_hashes_and_scope_match_actual_checkpoints", migration.get("schema") == "local_formal_numerical_migration_v1"
              and migration.get("checkpoint_sha256") == SOURCE_SHA
              and migration.get("old_recipe_sha256") == object_digest(old_recipe)
              and migration.get("new_recipe_sha256") == object_digest(new_recipe)
              and migration.get("old_sources") == old_recipe["sources"]
              and migration.get("new_sources") == new_recipe["sources"]
              and migration.get("numerical_policy") == NUMERICAL_POLICY
              and set(migration.get("allowed_changed_fields", [])) == {
                  "sources.formal_training.py", "sources.numerics.py (added)", "sources.artifact_io.py (added)",
                  "amp_scale_policy", "numerical_policy (added)"}
              and migration.get("all_other_recipe_fields_equal") is True
              and migration.get("dataset_and_optimizer_schedule_unchanged") is True)
        check("source_epoch1_file_unchanged_after_audit", digest(args.source_checkpoint) == SOURCE_SHA)
        check("cuda_not_initialized", not torch.cuda.is_initialized())
        report["evidence_sha256"] = {str(path): digest(path) for path in
            (args.run_dir / "steps.jsonl", args.manifest, args.retry_log, migration_path) if path.exists()}
    except Exception as error:
        check("audit_completed_without_missing_or_malformed_evidence", False, repr(error))
    report["passed"] = all(item["passed"] for item in checks)
    report["failed_checks"] = [x["name"] for x in checks if not x["passed"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(checks), "failed_checks": report["failed_checks"], "report": str(args.output)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
