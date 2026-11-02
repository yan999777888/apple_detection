"""CPU-only checks of Stage 2 control isolation, endpoint gates and identity."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage2_screen_tests", ROOT / "scripts/run_stage2_screen.py")
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)


def args(*extra):
    return screen.build_parser().parse_args(["--output", "unused", *extra])


def arrays(scores=(.9, .74, .7)):
    n = len(scores)
    return {"boxes": np.arange(n*4, dtype=np.float32).reshape(n,4),
            "scores": np.asarray(scores, dtype=np.float32),
            "visible": np.arange(n*4, dtype=np.float32).reshape(n,2,2)/32,
            "amodal": np.arange(n*4, dtype=np.float32).reshape(n,2,2)/16}


def test_cases_change_one_control_against_f1_and_keep_original_nms_k():
    f1 = screen.case_spec("F1")
    expected = {"R0": "pre_nms_threshold", "T10": "score_threshold", "T15": "score_threshold",
                "G1": "preprocess_device", "L1": "channels_last", "B1": "cudnn_benchmark", "H1": "roi_align_precision"}
    for name, key in expected.items():
        candidate = screen.case_spec(name)
        assert {k for k in f1 if candidate[k] != f1[k]} == {key}
        assert candidate["box_nms_iou"] == .8 and candidate["topk"] == 64
    assert screen.case_spec("T10")["score_threshold"] == .10
    assert screen.case_spec("T15")["score_threshold"] == .15


@pytest.mark.parametrize("extra", [("--cases", "F1"), ("--cases", "R0", "T10"),
    ("--cases", "R0", "F1", "F1"), ("--models", "base", "base"), ("--limit", "0"),
    ("--benchmark-count", "1", "--diagnostic-count", "2"), ("--warmup", "-1")])
def test_invalid_execution_inventory_rejected_before_model_work(extra):
    with pytest.raises(ValueError):
        screen.validate_args(args(*extra))


def test_defaults_define_41_diagnostic_and_40_timed_frames():
    value = screen.validate_args(args())
    assert (value.limit, value.benchmark_count, value.warmup, value.rounds) == (41,8,5,5)
    assert value.diagnostic_count == 3
    assert screen.P90_CONFIDENCE == {"base": .7394223809242249, "small": .743447482585907}


def test_low_score_removal_preserves_every_high_endpoint_exact():
    old = arrays()
    new = {k: v[:2].copy() for k,v in old.items()}
    for threshold in (.738, .75, screen.P90_CONFIDENCE["base"]):
        result = screen.compare_arrays(old, new, threshold)
        assert result["exact"]
        screen.enforce_f1_gate({"point": result})


@pytest.mark.parametrize("field", ["boxes", "scores", "visible", "amodal"])
def test_f1_gate_rejects_even_one_high_confidence_scalar_change(field):
    old, new = arrays(), arrays()
    new[field].flat[0] += .00001
    result = screen.compare_arrays(old, new, .738)
    assert not result["exact"]
    assert result["fields"][field]["max_abs_same_output_order"] > 0
    with pytest.raises(RuntimeError, match="F1 exactness gate failed"):
        screen.enforce_f1_gate({"point": result})


def test_count_changes_never_report_fake_aligned_mask_error():
    old, new = arrays(), arrays((.9,))
    result = screen.compare_arrays(old,new,.738)
    assert not result["exact"]
    assert all(r["max_abs_same_output_order"] is None for r in result["fields"].values())
    empty = screen.compare_arrays(arrays(()), arrays(()), .75)
    assert empty["exact"] and empty["reference_count"] == 0


def test_same_size_reordering_and_dtype_changes_not_exact():
    old = arrays()
    reordered = {k:v[[1,0,2]] for k,v in old.items()}
    assert not screen.compare_arrays(old,reordered,.738)["exact"]
    converted = {k:v.astype(np.float64) for k,v in old.items()}
    assert not screen.compare_arrays(old,converted,.738)["exact"]


def test_archive_comparison_checks_identity_and_sha(tmp_path):
    references = []
    for name in ("old", "new"):
        path = tmp_path / (name + ".npz")
        np.savez(path,**arrays())
        references.append([{"image_id":"apple.png","path":str(path),"sha256":screen.sha(path)}])
    assert screen.compare_archive_sets(*references,{"fixed":.75})["fixed"]["exact"]
    Path(references[1][0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError,match="archive changed"):
        screen.compare_archive_sets(*references,{"fixed":.75})
    references[1][0]["image_id"] = "different.png"
    with pytest.raises(ValueError,match="inventory differs"):
        screen.compare_archive_sets(*references,{"fixed":.75})


def test_subset_keeps_manifest_order_empty_and_annotation_anomaly():
    images = [{"image_id":"a.png"},{"image_id":"b.png"},{"image_id":"c.png"}]
    truth = {"a.png":[{"occlusion_stratum":"heavy","apple_id":"1"}], "b.png":[],
             "c.png":[{"occlusion_stratum":"annotation_visible_area_exceeds_amodal","apple_id":"2"}]}
    contract = {"images":images,"truth":truth,"official_full_val":True,"scope":{}}
    selected = [{"image_index":2,"image_id":"c.png"},{"image_index":1,"image_id":"b.png"}]
    result = screen.frozen_subset(contract,selected)
    assert [i["image_id"] for i in result["images"]] == ["c.png","b.png"]
    assert result["scope"] == {"images":2,"GT_instances":1,"empty_GT_images":1}
    assert result["strata"]["annotation_visible_area_exceeds_amodal"] == 1
    assert contract["official_full_val"] is True and result["official_full_val"] is False
    selected[0]["image_id"] = "a.png"
    with pytest.raises(ValueError,match="does not match"):
        screen.frozen_subset(contract,selected)


def test_fingerprint_preserves_numbers_across_layout_and_detects_mutation():
    model = torch.nn.Conv2d(3,4,3)
    before = screen.tensor_fingerprint(model)
    model.to(memory_format=torch.channels_last)
    assert screen.tensor_fingerprint(model) == before
    with torch.no_grad():
        model.weight[0,0,0,0] += .001
    assert screen.tensor_fingerprint(model) != before


def test_full_validation_export_has_no_score_cutoff_or_promotion():
    exported = screen.full_validation_config(screen.case_spec("T10"))
    assert set(exported) == {"candidate_policy","preprocess_device","channels_last",
                             "cudnn_benchmark","cudnn_deterministic","roi_align_precision"}
    assert exported["candidate_policy"] == {"score_threshold":.1,"box_nms_iou":.8,"topk":64}
    assert "pre_nms_threshold" not in exported


def test_existing_output_fails_before_gpu_or_file_changes(tmp_path):
    # On Windows the configured pytest temp directory is on F:.
    if tmp_path.drive.upper() != "F:":
        pytest.skip("F-drive workspace required")
    marker = tmp_path / "do_not_replace.txt"
    marker.write_text("preserved")
    value = args("--output",str(tmp_path))
    with pytest.raises(FileExistsError):
        screen.run(value)
    assert marker.read_text() == "preserved"


def test_latency_summary_excludes_diagnostic_frames_by_explicit_input():
    result = screen.latency_summary([100,200,300])
    assert result["mean_ms"] == 200 and result["source_frames_per_second"] == 5
    assert result["p95_ms"] == 290
