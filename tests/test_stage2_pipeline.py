"""Small CPU orchestration tests; no CUDA, weights or full-VAL data loaded."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage2_pipeline_under_test", ROOT / "scripts/run_stage2_full_validation.py")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def point(precision=.9443430656934306, recall=.9002609452015077):
    row = {"precision": precision, "recall": recall}
    return {"at_confidence_0_75": dict(row), "at_recall90": dict(row)}


def test_explicit_single_configuration_and_frozen_full_stage_commands(tmp_path):
    selected, output = tmp_path / "selected.json", tmp_path / "run"
    for name in ("base", "small", "fuse"):
        args = pipeline.stage_arguments(name, selected, output)
        parsed = pipeline.evaluator.build_parser().parse_args(args[3:])
        pipeline.evaluator.validate_args(parsed)
        assert parsed.stage == name and parsed.stage2_config == selected
        assert parsed.pre_nms_threshold is None and parsed.views == "both"
        assert parsed.paired_backbone_batch2 and parsed.save_roi_masks
        assert parsed.precision == "fp16" and parsed.roi_chunk_size == 64
    with pytest.raises(ValueError):
        pipeline.stage_arguments("query", selected, output)


@pytest.mark.parametrize("precision,recall,expected", [(.939999, .91, True), (.94, .90, False),
    (.9443, .90, False), (.97, .899999, True)])
def test_abort_uses_only_complete_four_view_p90_floor(precision, recall, expected):
    assert pipeline.hard_abort(point(precision, recall)) is expected


def test_missing_recall90_aborts_and_never_promotes():
    value = point()
    value["at_recall90"] = None
    result = pipeline.acceptance(value, point())
    assert result["hard_abort"] and not result["maintains_original_P_at_R90"]
    assert not result["automatically_promoted"]


def test_94_floor_does_not_replace_original_baseline():
    result = pipeline.acceptance(point(.94, .91), point())
    assert not result["hard_abort"]
    assert not result["maintains_original_P_at_R90"]
    assert not result["floor_is_promotion_threshold"]
    assert result["original_baseline_preserved"] and not result["automatically_promoted"]


def test_precision_at_r90_and_strict_both_metrics_have_distinct_labels():
    result = pipeline.acceptance(point(.95, .90), point())
    assert result["maintains_original_P_at_R90"]
    assert not result["maintains_original_P_and_R_at_P90"]
    assert not result["automatically_promoted"]


def test_fresh_directory_claim_preserves_existing_results(tmp_path, monkeypatch):
    lock = tmp_path / "global.lock"
    monkeypatch.setattr(pipeline, "GLOBAL_LOCK", lock)
    output, report = tmp_path / "run", tmp_path / "report"
    output.mkdir()
    evidence = output / "existing.json"
    evidence.write_text("untouched", encoding="utf-8")
    with pytest.raises(FileExistsError):
        pipeline.claim_directories(output, report)
    assert evidence.read_text() == "untouched" and not lock.exists()


def test_shared_lock_blocks_second_explicit_pipeline(tmp_path, monkeypatch):
    lock = tmp_path / "global.lock"
    monkeypatch.setattr(pipeline, "GLOBAL_LOCK", lock)
    output, report = pipeline.claim_directories(tmp_path / "run", tmp_path / "report")
    first_receipt = json.loads(lock.read_text())
    assert (report / "pipeline.lock").is_file()
    with pytest.raises(FileExistsError):
        pipeline.claim_directories(tmp_path / "another_run", tmp_path / "another_report")
    assert json.loads(lock.read_text()) == first_receipt
    assert not (tmp_path / "another_run").exists()


def test_nested_output_report_rejected(tmp_path, monkeypatch):
    lock = tmp_path / "global.lock"
    monkeypatch.setattr(pipeline, "GLOBAL_LOCK", lock)
    with pytest.raises(ValueError, match="non-nested"):
        pipeline.claim_directories(tmp_path / "run", tmp_path / "run/report")
    assert not lock.exists()


def test_stage_receipt_requires_original_complete_uncut_configuration():
    selected = {"configuration": {"chosen": True}, "binding": {"path": "fixed", "sha256": "config"}}
    provenance = {"provenance_sha256": "truth", "provenance": {"binding": {"current_annotation_sha256": "annotation"}}}
    sources = {"source": "hash"}
    deployment = {"deployment_experiment": "stage2_original_strong_full_val_v1",
        "stage2_configuration": selected["configuration"], "stage2_config_file": selected["binding"],
        "pre_nms_threshold": None, "candidate_selection_uses_gt": False, "precision": "fp16",
        "roi_precision": "autocast", "roi_chunk_size": 64, "short_side": 1024, "max_size": 1280,
        "views": "both", "paired_backbone_batch2": True, "skip_dead_amodal": False,
        "evaluation_policy": {"fixed_only": False}}
    metadata = {"phase": "complete", "images": 814, "GT": 3449, "empty": 13,
        "model": "base", "source_sha256": sources, "config_sha256": pipeline.CONFIG_SHA,
        "provenance_sha256": "truth", "from_fresh_weight_inference": True,
        "original_strong_models_only": True, "training_performed": False,
        "source_before_after_identical": True, "config_before_after_identical": True,
        "stage2_configuration_before_after_identical": True, "original_weights_before_after_identical": True,
        "deployment": deployment, "checkpoint_sha256": pipeline.audit.ORIGINAL_CHECKPOINTS["base"],
        "annotation_sha256": "annotation", "roi_masks_saved": True,
        "constructor_config": {"backbone_name": "convnext_base"},
        "strict_restore": {"strict": True, "tensor_count": 434, "all_tensors_equal_to_checkpoint": True,
                           "missing_keys": [], "unexpected_keys": []}}
    pipeline.audit_stage(metadata, "base", selected, {}, provenance, sources)
    for path, value in [("pre_nms_threshold", .738), ("views", "original"), ("short_side", 512),
                        ("candidate_selection_uses_gt", True), ("paired_backbone_batch2", False)]:
        changed = deepcopy(metadata)
        changed["deployment"][path] = value
        with pytest.raises(ValueError):
            pipeline.audit_stage(changed, "base", selected, {}, provenance, sources)
