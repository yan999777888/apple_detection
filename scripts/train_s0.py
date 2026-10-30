"""PApple-only S0 entry with a full-Q0 and real-pressure-test start gate."""
from pathlib import Path
import argparse
import json
import math
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_start_conditions(config_path):
    from apple_seg.training import load_config, sha
    config = load_config(config_path)
    raw, recipe = config["raw"], config["recipe"]
    if recipe["data_mode"] != "papple_only" or raw.get("experiment_arm") != "S0_GT_only":
        raise ValueError("This entry only starts the declared GT-only S0 arm")
    required = raw["prerequisites"]
    q0 = config["base"] / required["q0_directory"]
    pressure = config["base"] / required["pressure_directory"]
    baseline = read(q0 / "q0_run.json")
    if baseline["phase"] != "complete" or baseline["parent_sha256"] != raw["parent_sha256"] or baseline["input_size"] != recipe["input_size"]:
        raise ValueError("A complete Q0 for this exact parent and resolution is required")
    metric = baseline["summary"]
    if (metric["images"], metric["gt_count"], metric["empty_gt_images"]) != (814, 3449, 13):
        raise ValueError("Q0 full validation scope differs")
    tested = read(pressure / "training_summary.json")
    if tested["identity"]["parent_sha256"] != raw["parent_sha256"]:
        raise ValueError("Pressure test used a different parent checkpoint")
    if tested["phase"] != "smoke_complete" or not tested["smoke"] or tested["updates"]["successful"] < 2 or tested["updates"]["skipped"]:
        raise ValueError("At least two successful real pressure updates without AMP skips are required")
    if not tested["cuda_initialized"] or not tested["checkpoint"]["own_roundtrip_verified"] or not tested["source_before_after_identical"]:
        raise ValueError("Pressure test must verify CUDA, checkpoint roundtrip and stable source")
    if tested["updates"]["attempted"] != tested["updates"]["successful"] or len(tested["update_history"]) != tested["updates"]["successful"]:
        raise ValueError("Pressure history must cover every attempted/successful update")
    expected_sources = {"training.py", "model.py", "losses.py", "data.py", "evaluate.py"}
    if set(tested["source_sha256_after"]) != expected_sources or tested["source_sha256_before"] != tested["source_sha256_after"]:
        raise ValueError("Complete matching pressure source inventory required")
    for key in ("data_mode", "input_size", "accumulate", "workers", "backbone_lr", "head_lr", "gradient_clip_norm", "seed"):
        if tested["plan"][key] != recipe[key]:
            raise ValueError("Pressure/formal configuration mismatch: " + key)
    for filename, digest in tested["source_sha256_after"].items():
        if sha(ROOT / "src/apple_seg" / filename) != digest:
            raise ValueError("Source changed since pressure test: " + filename)
    expected_branches = {"amodal", "visible", "classifier", "shared_encoder"}
    for ordinal, row in enumerate(tested["update_history"], start=1):
        if not row["finite_gradients"] or not row["optimizer_step_executed"]:
            raise ValueError("Pressure gradient/optimizer check failed")
        if row["updates"] != {"attempted": ordinal, "successful": ordinal, "skipped": 0}:
            raise ValueError("Pressure update counters are not contiguous")
        if set(row["smoke_parameter_max_abs_change"]) != expected_branches or set(row["smoke_readout_gradient_abs_sum"]) != expected_branches:
            raise ValueError("Pressure evidence must cover A/V/class/shared branches")
        if not all(math.isfinite(value) and value > 0 for value in row["smoke_parameter_max_abs_change"].values()):
            raise ValueError("Pressure test did not update every checked model branch")
        if not all(math.isfinite(value) and value > 0 for value in row["smoke_readout_gradient_abs_sum"].values()):
            raise ValueError("Pressure test lacks nonzero gradients on all checked branches")
        if not (row["cuda_peak_reserved_bytes"] >= row["cuda_peak_allocated_bytes"] > 0):
            raise ValueError("Pressure test lacks CUDA memory measurements")
        if len(row["micro_batches"]) != 4:
            raise ValueError("Pressure evidence must contain four PApple microbatches per update")
        if any(m["source"]["dataset"] != "PApple" or m["source"]["split"] != "train" or not m["GT_unchanged"] for m in row["micro_batches"]):
            raise ValueError("Pressure data/GT contract failed")
    return {"Q0": str(q0), "pressure": str(pressure), "parent_sha256": raw["parent_sha256"],
            "pressure_summary_sha256": sha(pressure / "training_summary.json")}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args, _ = parser.parse_known_args()
    raw = read(args.config)
    if raw.get("training", {}).get("data_mode") != "papple_only":
        raise ValueError("train_s0 requires explicit papple_only")
    if not args.smoke:
        print(json.dumps({"event": "S0_start_gate_passed", **verify_start_conditions(args.config)}, ensure_ascii=False), flush=True)
    from train_mixed import main as run
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
