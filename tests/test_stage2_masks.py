"""Tiny CPU tests; no real screen/GT mask recomputation and no GPU work."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage2_masks_test", ROOT / "scripts/audit_stage2_masks.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.mark.parametrize("cases", [[],["F1","F1"],["../F1"],["F1/path"],["../../else"]])
def test_invalid_cases_rejected(cases):
    with pytest.raises(ValueError):
        audit.normalize_cases(cases)


def test_f1_is_always_explicit_same_run_reference():
    assert audit.normalize_cases(["T15","C1"]) == ["F1","T15","C1"]
    assert audit.normalize_cases(["C1","F1"]) == ["F1","C1"]


def sample_contract():
    gt = [{"gt_index":0,"apple_id":"h","occlusion_stratum":"heavy_visible_lt_0.40","visible_area_fraction":.2},
          {"gt_index":1,"apple_id":"l","occlusion_stratum":"light","visible_area_fraction":.8}]
    return {"images":[{"image_id":"a.png"},{"image_id":"empty.png"}],
            "truth":{"a.png":gt,"empty.png":[]},"strata":{"heavy_visible_lt_0.40":1,"light":1}}


def diagnostic(visible, amodal, scores=(.9,.8)):
    return [{"image_id":"a.png","scores":list(scores),"visible_mask_gt_indices_by_prediction":visible,
             "amodal_mask_gt_indices_by_prediction":amodal},
            {"image_id":"empty.png","scores":[.95],"visible_mask_gt_indices_by_prediction":[-1],
             "amodal_mask_gt_indices_by_prediction":[-1]}]


def test_mask_branch_losses_independent_and_heavy_identity_retained():
    before = diagnostic([0,1],[-1,1])
    after = diagnostic([-1,1],[0,1])
    visible = audit.compare_mask_recall(before,after,sample_contract(),"visible",.75)
    amodal = audit.compare_mask_recall(before,after,sample_contract(),"amodal",.75)
    assert (visible["lost"],visible["recovered"]) == (1,0)
    assert visible["groups"]["heavy_visible_lt_0.40"]["lost"] == 1
    assert visible["ground_truth_instances"][0]["apple_id"] == "h"
    assert (amodal["lost"],amodal["recovered"]) == (0,1)


def test_confidence_filter_preserves_stable_greedy_high_prefix():
    from apple_seg.evaluate import _greedy_assignment, _mask_pr
    metrics = audit.load_metrics()
    # High-scoring duplicate gets the one GT; low score cannot steal it.
    matrix = np.asarray([[.8],[.95]],dtype=np.float64)
    assigned = _greedy_assignment(matrix,[.9,.74])
    assert assigned == [0,-1]
    rows = [{"scores":[.9,.74],"visible_mask_gt_indices_by_prediction":assigned},
            {"scores":[.99],"visible_mask_gt_indices_by_prediction":[-1]}]
    point = _mask_pr(rows,"visible_mask",.75,1,metrics)
    assert (point["tp"],point["fp"],point["fn"]) == (1,1,0)


def test_duplicate_gt_recall_and_image_order_are_rejected():
    invalid = diagnostic([0,0],[0,1])
    with pytest.raises(ValueError,match="same GT"):
        audit.point_assignments(invalid,"visible",.75)
    rows = diagnostic([0,1],[0,1])
    with pytest.raises(ValueError,match="image order"):
        audit.compare_mask_recall(rows,rows[::-1],sample_contract(),"visible",.75)


def test_frozen_subset_retains_empty_and_full_manifest_order(monkeypatch):
    monkeypatch.setattr(audit,"EXPECTED_SCOPE",(2,2,1))
    contract = sample_contract()
    samples = [{"image_index":1,"image_id":"empty.png"},{"image_index":0,"image_id":"a.png"}]
    result = audit.subset_contract(contract,samples)
    assert [r["image_id"] for r in result["images"]] == ["empty.png","a.png"]
    assert result["scope"]["empty_GT_images"] == 1
    assert result["strata"]["heavy_visible_lt_0.40"] == 1
    samples[0]["image_id"] = "wrong.png"
    with pytest.raises(ValueError,match="identity"):
        audit.subset_contract(contract,samples)


def archive_case(tmp_path):
    directory = tmp_path / "case"
    (directory / "roi_masks").mkdir(parents=True)
    path = directory / "roi_masks/000000.npz"
    boxes = np.asarray([[1,2,5,6]],dtype=np.float32)
    scores = np.asarray([.9],dtype=np.float32)
    np.savez(path,boxes=boxes,scores=scores,visible=np.ones((1,56,56),dtype=np.float32),
             amodal=np.ones((1,56,56),dtype=np.float32))
    record = {"image_id":"a.png","gt":{"boxes":[]},"pred":{"boxes":boxes.tolist(),"scores":scores.tolist()}}
    entry = {"path":str(path),"image_id":"a.png","manifest_index":5,"sha256":audit.sha(path),
             "bytes":path.stat().st_size,"dtype":"float32","shape":[1,56,56]}
    return directory,path,[entry],[record],[{"image_id":"a.png","image_index":5}]


def test_inference_bound_archive_sha_and_native_box_score_exact(tmp_path):
    directory,path,entries,records,samples = archive_case(tmp_path)
    evidence = {}
    assert audit.verify_archive_manifest(directory,entries,records,samples,evidence) == [path.resolve()]
    arrays = audit.load_roi_archive(path,records[0],evidence)
    assert arrays["visible"].shape == (1,56,56)
    changed = deepcopy(records[0]); changed["pred"]["boxes"][0][0] += .01
    with pytest.raises(ValueError,match="exactly match"):
        audit.load_roi_archive(path,changed,{})
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError,match="SHA/size"):
        audit.verify_archive_manifest(directory,entries,records,samples,{})


@pytest.mark.parametrize("change", ["path","index","shape","dtype"])
def test_manifest_rejects_mixed_view_or_wrong_tensor_metadata(tmp_path,change):
    directory,path,entries,records,samples = archive_case(tmp_path)
    if change == "path": entries[0]["path"] = str(tmp_path/"other.npz")
    if change == "index": entries[0]["manifest_index"] = 6
    if change == "shape": entries[0]["shape"] = [1,28,28]
    if change == "dtype": entries[0]["dtype"] = "float16"
    with pytest.raises(ValueError):
        audit.verify_archive_manifest(directory,entries,records,samples,{})


def test_native_mask_iou_matches_independently_of_box_iou():
    import torch
    from apple_seg.evaluate import _greedy_assignment
    from apple_seg.od_evaluate import _roi_mask_overlaps
    torch.set_num_threads(1)
    prediction = {"boxes":torch.tensor([[0.,0.,8.,8.]]),"scores":[.9],
                  "visible":torch.zeros(1,56,56),"amodal":torch.ones(1,56,56)}
    truth = {"boxes":[[0.,0.,8.,8.]],"visible_masks":np.ones((1,8,8),dtype=np.uint8),
             "amodal_masks":np.ones((1,8,8),dtype=np.uint8)}
    matrix = _roi_mask_overlaps(prediction,truth,(8,8),gt_chunk_size=1,tile_pixels=64)
    assert _greedy_assignment(matrix["visible_mask"],[.9]) == [-1]
    assert _greedy_assignment(matrix["amodal_mask"],[.9]) == [0]
    assert not torch.cuda.is_initialized()
