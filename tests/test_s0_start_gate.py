"""F-drive JSON fixtures for the S0 launch gate; no model or CUDA execution."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import pytest


REPO = Path(__file__).absolute().parents[1]
SPEC = importlib.util.spec_from_file_location("_test_s0_start_gate_entry", REPO / "scripts/train_s0.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
SOURCE_NAMES = ("training.py", "model.py", "losses.py", "data.py", "evaluate.py")
BRANCH_NAMES = ("amodal", "visible", "classifier", "shared_encoder")


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def fixture(monkeypatch):
    # Do not rely on a host-wide TEMP setting: fixture files always stay on F:.
    parent = REPO / "reports"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="s0_gate_fixture_", dir=parent) as temp:
        root = Path(temp)
        (root / "src/apple_seg").mkdir(parents=True)
        (root / "q0").mkdir()
        (root / "pressure").mkdir()
        sources = {}
        for name in SOURCE_NAMES:
            content = ("fixture source " + name).encode()
            (root / "src/apple_seg" / name).write_bytes(content)
            sources[name] = hashlib.sha256(content).hexdigest()
        raw = {
            "path_base": ".", "experiment_arm": "S0_GT_only", "parent_sha256": "a" * 64,
            "paths": {"papple_root": "assets/papple", "parent_checkpoint": "weights/parent.pt",
                      "papple_provenance": "assets/provenance.json", "output": "runs/s0"},
            "prerequisites": {"q0_directory": "q0", "pressure_directory": "pressure"},
            "training": {"data_mode": "papple_only", "input_size": 768, "accumulate": 4,
                         "workers": 0, "backbone_lr": 1e-6, "head_lr": 5e-6,
                         "gradient_clip_norm": .1, "seed": 20261030}}
        baseline = {"phase": "complete", "parent_sha256": "a" * 64, "input_size": 768,
                    "summary": {"images": 814, "gt_count": 3449, "empty_gt_images": 13},
                    "training_performed": False}
        row = {"finite_gradients": True, "optimizer_step_executed": True,
               "smoke_parameter_max_abs_change": dict.fromkeys(BRANCH_NAMES, 1e-6),
               "smoke_readout_gradient_abs_sum": dict.fromkeys(BRANCH_NAMES, .1),
               "cuda_peak_allocated_bytes": 1000000000, "cuda_peak_reserved_bytes": 1200000000,
               "micro_batches": [{"source": {"dataset": "PApple", "split": "train"},
                                  "GT_unchanged": True} for _ in range(4)]}
        tested = {"identity": {"parent_sha256": "a" * 64}, "phase": "smoke_complete", "smoke": True,
                  "updates": {"attempted": 2, "successful": 2, "skipped": 0},
                  "cuda_initialized": True, "checkpoint": {"own_roundtrip_verified": True},
                  "source_before_after_identical": True, "source_sha256_before": sources,
                  "source_sha256_after": sources, "plan": deepcopy(raw["training"]),
                  "update_history": [deepcopy(row), deepcopy(row)]}
        for ordinal, update in enumerate(tested["update_history"], start=1):
            update["updates"] = {"attempted": ordinal, "successful": ordinal, "skipped": 0}
        config = root / "s0.json"
        write(config, raw)
        write(root / "q0/q0_run.json", baseline)
        write(root / "pressure/training_summary.json", tested)
        monkeypatch.setattr(gate, "ROOT", root)
        yield root, config, baseline, tested


def test_complete_baseline_and_two_update_pressure_pass(fixture):
    root, config, _, _ = fixture
    result = gate.verify_start_conditions(config)
    assert Path(result["Q0"]) == root / "q0"
    assert Path(result["pressure"]) == root / "pressure"
    assert result["parent_sha256"] == "a" * 64
    assert len(result["pressure_summary_sha256"]) == 64


def test_missing_q0_is_rejected(fixture):
    root, config, _, _ = fixture
    (root / "q0/q0_run.json").unlink()
    with pytest.raises(FileNotFoundError):
        gate.verify_start_conditions(config)


@pytest.mark.parametrize("key,value", [("phase", "running"), ("parent_sha256", "b" * 64), ("input_size", 640)])
def test_incomplete_or_wrong_q0_is_rejected(fixture, key, value):
    root, config, baseline, _ = fixture
    baseline[key] = value
    write(root / "q0/q0_run.json", baseline)
    with pytest.raises(ValueError):
        gate.verify_start_conditions(config)


def test_partial_q0_scope_is_rejected(fixture):
    root, config, baseline, _ = fixture
    baseline["summary"]["images"] = 813
    write(root / "q0/q0_run.json", baseline)
    with pytest.raises(ValueError):
        gate.verify_start_conditions(config)


@pytest.mark.parametrize("case", ["wrong_parent", "one_update", "amp_skip", "no_cuda", "no_roundtrip",
                                  "source_changed", "non_papple", "val_sample", "gt_changed",
                                  "empty_history", "empty_source_inventory", "empty_gradient_branches",
                                  "empty_change_branches", "empty_micro_batches", "negative_vram"])
def test_invalid_pressure_evidence_is_rejected(fixture, case):
    root, config, _, tested = fixture
    row = tested["update_history"][0]
    if case == "wrong_parent": tested["identity"]["parent_sha256"] = "b" * 64
    elif case == "one_update": tested["updates"]["successful"] = 1
    elif case == "amp_skip": tested["updates"]["skipped"] = 1
    elif case == "no_cuda": tested["cuda_initialized"] = False
    elif case == "no_roundtrip": tested["checkpoint"]["own_roundtrip_verified"] = False
    elif case == "source_changed": (root / "src/apple_seg/model.py").write_text("changed source", encoding="utf-8")
    elif case == "non_papple": row["micro_batches"][0]["source"]["dataset"] = "MinneApple"
    elif case == "val_sample": row["micro_batches"][0]["source"]["split"] = "val"
    elif case == "gt_changed": row["micro_batches"][0]["GT_unchanged"] = False
    elif case == "empty_history": tested["update_history"] = []
    elif case == "empty_source_inventory": tested["source_sha256_after"] = {}
    elif case == "empty_gradient_branches": row["smoke_readout_gradient_abs_sum"] = {}
    elif case == "empty_change_branches": row["smoke_parameter_max_abs_change"] = {}
    elif case == "empty_micro_batches": row["micro_batches"] = []
    elif case == "negative_vram": row["cuda_peak_allocated_bytes"] = -1
    write(root / "pressure/training_summary.json", tested)
    with pytest.raises(ValueError):
        gate.verify_start_conditions(config)
