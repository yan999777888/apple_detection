"""CPU checks for pre-mask evaluator provenance and temporary-factory scope."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("premask_eval_under_test", ROOT / "scripts/evaluate_premask.py")
premask = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(premask)


def arguments(policy="policy.json", config="config.json", *extra):
    return premask.build_parser().parse_args([
        "--config", str(config), "--candidate-policy", str(policy),
        "--stage", "base", "--output", "out", *extra])


@pytest.mark.parametrize("value", [[], {"unknown": 64}, {"topk": 0}, {"topk": True},
                                  {"topk": 64.0}, {"score_threshold": float("nan")},
                                  {"box_nms_iou": float("inf")}, {"score_threshold": -1},
                                  {"box_nms_iou": 1.01}, {"score_threshold": False}])
def test_invalid_candidate_policy_is_rejected(value):
    with pytest.raises(ValueError):
        premask.normalize_policy(value)


def test_independent_controls_and_no_filter_reference():
    assert premask.normalize_policy({}) == dict(score_threshold=None, box_nms_iou=None, topk=None)
    assert premask.normalize_policy({"topk": 64}) == dict(score_threshold=None, box_nms_iou=None, topk=64)
    assert premask.normalize_policy({"box_nms_iou": .8})["topk"] is None
    args = arguments()
    assert args.pre_nms_threshold is None and args.roi_chunk_size == 64
    premask.validate_args(args)


@pytest.mark.parametrize("extra", [("--pre-nms-threshold", ".35"), ("--fixed-only",),
                                  ("--skip-dead-amodal",)])
def test_full_pr_and_original_amodal_branch_cannot_be_disabled(extra):
    with pytest.raises(ValueError):
        premask.validate_args(arguments("policy.json", "config.json", *extra))


@pytest.fixture
def files(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text('{"topk":64}', encoding="utf-8")
    config = tmp_path / "config.json"
    cfg = {"path_base": ".", "paths": {}}
    for name in ("base", "small"):
        path = tmp_path / (name + ".pt")
        path.write_bytes(("fixed-original-" + name).encode())
        cfg["paths"][name + "_checkpoint"] = path.name
        cfg[name + "_checkpoint_sha256"] = premask.fast.sha(path)
    config.write_text(json.dumps(cfg), encoding="utf-8")
    return arguments(policy, config), policy, config, cfg


def test_policy_is_bound_to_exact_bytes_and_rejects_duplicate_keys(files):
    _, path, _, _ = files
    value, binding = premask.load_policy(path)
    assert value["topk"] == 64 and binding["sha256"] == premask.fast.sha(path)
    path.write_text('{"topk":64,"topk":16}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        premask.load_policy(path)


@pytest.mark.parametrize("changed", ["policy", "base", "small"])
def test_policy_and_both_original_weights_checked_afterward(files, changed):
    args, policy, config, _ = files
    _, binding = premask.load_policy(policy)
    checkpoints = premask.checkpoint_bindings(config)
    premask.assert_bindings_unchanged(binding, checkpoints)
    path = policy if changed == "policy" else Path(checkpoints[changed]["path"])
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed during full VAL"):
        premask.assert_bindings_unchanged(binding, checkpoints)


def test_wrong_initial_checkpoint_is_rejected(files):
    _, _, config, _ = files
    (config.parent / "base.pt").write_bytes(b"different-checkpoint")
    with pytest.raises(ValueError, match="Original base checkpoint hash differs"):
        premask.checkpoint_bindings(config)


def test_temporary_factory_records_policy_and_restores_every_patch(files, monkeypatch):
    args, _, config, _ = files
    old_adapter = object()
    fake_module = SimpleNamespace(FastDeploymentAdapter=old_adapter)
    calls = []

    def fake_adapter(model, **kwargs):
        calls.append((model, kwargs))
        return kwargs

    old = {name: getattr(premask.fast, name) for name in
           ("build_parser", "deployment_spec", "source_inventory", "final_integrity")}
    # No dependency on the parallel adapter author's write timing.
    monkeypatch.setattr(premask, "EXTRA_SOURCES", ("scripts/evaluate_premask.py",))
    with pytest.raises(RuntimeError, match="test exception"):
        with premask.patched_evaluator(args, adapter_type=fake_adapter, adapter_module=fake_module):
            first = fake_module.FastDeploymentAdapter("model", roi_chunk_size=64)
            assert first["candidate_policy"]["topk"] == 64
            assert first["capture_candidates"] is False
            first["candidate_policy"]["topk"] = 1
            assert fake_module.FastDeploymentAdapter("model")["candidate_policy"]["topk"] == 64
            spec = premask.fast.deployment_spec(args)
            assert spec["candidate_policy"]["topk"] == 64
            assert spec["candidate_selection_uses_gt"] is False
            assert spec["automatic_promotion"] is False
            assert spec["complete_original_candidate_set"] is False
            assert spec["evaluation_policy"]["candidate_budget_restricts_recall"] is True
            assert spec["evaluation_policy"]["original_unfiltered_PR_claimed"] is False
            sources = premask.fast.source_inventory()
            assert str(Path("scripts/evaluate_premask.py")) in sources
            meta = {"config": str(config), "config_sha256": premask.fast.sha(config)}
            integrity = premask.fast.final_integrity(meta, sources)
            assert integrity["original_weights_before_after_identical"] is True
            assert set(integrity["original_weight_bindings"]) == {"base", "small"}
            parsed_again = premask.fast.build_parser().parse_args([
                "--config", str(config), "--candidate-policy", str(args.candidate_policy),
                "--stage", "base", "--output", "out"])
            assert parsed_again.roi_chunk_size == 64
            raise RuntimeError("test exception")
    assert fake_module.FastDeploymentAdapter is old_adapter
    assert all(getattr(premask.fast, name) is old[name] for name in old)
    assert len(calls) == 2


def test_full_source_inventory_includes_adapter_and_selector():
    assert set(premask.EXTRA_SOURCES) == {
        "scripts/evaluate_premask.py", "deployment/pre_mask.py", "deployment/candidate_policy.py"}


def test_fusion_rejects_another_candidate_policy_before_using_predictions(tmp_path):
    expected = {"views": "both", "candidate_policy": {"topk": 64}}
    wrong = deepcopy(expected)
    wrong["candidate_policy"]["topk"] = 96
    source = {"adapter": "pinned"}
    premask.fast.write(tmp_path / "base/prediction_metadata.json", {
        "phase": "complete", "deployment": wrong, "source_sha256": source,
        "config_sha256": "config", "checkpoint_sha256": "base-pinned"})
    with pytest.raises(ValueError, match="deployment spec"):
        premask.fast.run_fusion(SimpleNamespace(), {"base_checkpoint_sha256": "base-pinned"}, {},
                                source, tmp_path, {"deployment": expected, "config_sha256": "config"})
