"""Tiny CPU fixtures test the real evaluation engine without formal VAL claims."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pytest
import torch

from apple_seg import evaluate as evaluation


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tensor_sha(mask):
    return hashlib.sha256(np.ascontiguousarray(mask.numpy(), dtype=np.uint8).tobytes()).hexdigest()


class TinyDataset:
    configuration = {"augment": False, "padding": False, "crop": False, "input_hw": [8, 8]}
    annotation_sha256 = {"visible": "1" * 64, "amodal": "2" * 64}

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class TinyModel(torch.nn.Module):
    """Synthetic logits only: no pretrained backbone or checkpoint access."""
    def __init__(self, predictions, on_call=None):
        super().__init__()
        self.predictions = predictions
        self.calls = 0
        self.on_call = on_call

    def forward(self, pixel_values, pixel_mask=None):
        assert pixel_values.device.type == "cpu"
        assert pixel_mask.device.type == "cpu"
        prediction = self.predictions[self.calls]
        self.calls += 1
        if self.on_call is not None:
            self.on_call(self.calls)
        return prediction


def logits_for(candidates):
    classes = torch.empty((1, 100, 2))
    classes[..., 0], classes[..., 1] = -12, 0
    visible = torch.full((1, 100, 8, 8), -12.0)
    amodal = torch.full_like(visible, -12.0)
    for query, score, v_mask, a_mask in candidates:
        classes[0, query, 0] = math.log(score / (1 - score))
        visible[0, query] = v_mask.float() * 24 - 12
        amodal[0, query] = a_mask.float() * 24 - 12
    return {"class_queries_logits": classes, "visible_masks_queries_logits": visible,
            "amodal_masks_queries_logits": amodal}


@pytest.fixture
def fixture(tmp_path):
    left, right = torch.zeros((8, 8), dtype=torch.uint8), torch.zeros((8, 8), dtype=torch.uint8)
    left[:, :4], right[:, 4:] = 1, 1
    empty = torch.zeros_like(left)
    images, rows, items = [], [], []
    for index, masks in enumerate((left[None], right[None], torch.empty((0, 8, 8), dtype=torch.uint8))):
        image_id = f"fixture_{index}.png"
        path = tmp_path / image_id
        path.write_bytes(b"synthetic-rgb-byte-identity-" + str(index).encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ids = [f"apple-{index}"] if len(masks) else []
        images.append({"image_id": image_id, "image_index": index, "gt_count": len(masks),
                       "apple_ids_in_target_order": ids, "height": 8, "width": 8,
                       "RGB_path": str(path), "RGB_bytes": path.stat().st_size, "RGB_sha256": digest})
        if len(masks):
            box = [0.0, 0.0, 4.0, 8.0] if index == 0 else [4.0, 0.0, 8.0, 8.0]
            rows.append({"image_id": image_id, "gt_index": 0, "apple_id": ids[0], "box": box,
                         "amodal_area_pixels": 32, "visible_area_pixels": 32, "visible_area_fraction": 1.0,
                         "visible_pixels_outside_amodal": 0, "occlusion_stratum": "mostly_visible_0.90_to_1.00",
                         "amodal_mask_uint8_sha256": tensor_sha(masks[0]), "visible_mask_uint8_sha256": tensor_sha(masks[0])})
        target = {"labels": torch.zeros(len(masks), dtype=torch.int64), "visible_masks": masks.clone(),
                  "amodal_masks": masks.clone(), "instance_ids": ids, "original_hw": [8, 8],
                  "annotation_validity": {"visible": True, "amodal": True},
                  "source": {"dataset": "PApple", "split": "val", "image_id": image_id,
                             "rgb_path": str(path), "rgb_sha256": digest}}
        items.append({"pixel_values": torch.zeros((3, 8, 8)), "pixel_mask": torch.ones((8, 8), dtype=torch.int64), "target": target})
    provenance = {"phase": "complete", "passed": True,
                  "scope": {"split": "val", "images": 3, "GT_instances": 2, "empty_GT_images": 1},
                  "image_metadata": images, "ground_truth_instances": rows,
                  "binding": {"current_annotation_sha256": TinyDataset.annotation_sha256},
                  "annotation_diagnostics_GT_only": {"strata_counts": {"mostly_visible_0.90_to_1.00": 2},
                      "instances_visible_area_exceeds_amodal": 0, "no_clipping_repair_or_GT_exclusion": True}}
    provenance_path = tmp_path / "fixture_provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    contract = evaluation._contract_from_provenance(provenance)
    contract.update(provenance_path=provenance_path, provenance_sha256=evaluation._sha(provenance_path))
    predictions = [logits_for([(0, .9, empty, left), (1, .9, left, left), (2, .4, right, right), (3, .95, left, empty)]),
                   logits_for([(0, .8, right, right)]), logits_for([])]
    return {"dataset": TinyDataset(items), "contract": contract, "predictions": predictions,
            "provenance": provenance, "output": tmp_path / "evaluation"}


def run_fixture(value, model=None):
    model = model or TinyModel(value["predictions"])
    return evaluation._evaluate_with_contract(model, value["dataset"], "cpu", value["output"], value["contract"])


@pytest.mark.parametrize("mutation", ["dataset_augment", "empty_flip", "source_crop"])
def test_rejects_augmentation_or_crop_even_when_empty_GT_hashes_cannot_detect_it(fixture, mutation):
    if mutation == "dataset_augment":
        fixture["dataset"].augment = True
    elif mutation == "empty_flip":
        fixture["dataset"].items[2]["target"]["source"]["horizontal_flip"] = True
    else:
        fixture["dataset"].items[0]["target"]["source"]["crop_xyxy"] = [0, 0, 4, 8]
    with pytest.raises(ValueError):
        run_fixture(fixture)
    assert not (fixture["output"] / "summary.json").exists()


def test_actual_engine_calibrates_all_distinct_scores_replays_and_matches_masks_independently(fixture):
    model = TinyModel(fixture["predictions"])
    model.train()
    original = fixture["dataset"].items[0]["target"]["amodal_masks"].clone()
    result = run_fixture(fixture, model)
    assert model.training and model.calls == 3
    assert result["model_selection_eligible"] is False and result["score"] is None
    summary = result["summary"]
    assert (summary["tp"], summary["fp"], summary["fn"], summary["images"], summary["empty_gt_images"]) == (2, 1, 0, 3, 1)
    assert summary["precision"] == 2 / 3 and summary["recall"] == 1
    assert result["confidence"] == pytest.approx(.8)
    raw = json.loads((fixture["output"] / "raw_predictions.json").read_text(encoding="utf-8"))
    curve = json.loads((fixture["output"] / "box_threshold_curve.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((fixture["output"] / "mask_pair_diagnostics.json").read_text(encoding="utf-8"))
    metadata = json.loads((fixture["output"] / "evaluation_metadata.json").read_text(encoding="utf-8"))
    metrics = evaluation._module(evaluation.METRICS_PATH, "_test_eval_metrics")
    replay = metrics.evaluate_at_threshold(raw, result["confidence"], .5)
    assert all(replay[key] == summary[key] for key in ("confidence", "tp", "fp", "fn", "precision", "recall"))
    assert len(curve) == len({score for row in raw for score in row["pred"]["scores"]}) == 3
    assert curve[0]["prediction_count"] == 2  # Exact .9 ties enter as one attainable threshold.
    assert diagnostics[0]["query_indices"] == [0, 1, 2]  # Highest-scoring empty A query 3 is dropped.
    assert diagnostics[0]["box_gt_indices_by_prediction"] == [0, -1, -1]
    assert diagnostics[0]["visible_mask_gt_indices_by_prediction"] == [-1, 0, -1]
    for branch in ("visible", "amodal"):
        point = summary[branch + "_instance_mask_pr_iou50"]
        assert point["confidence"] == result["confidence"] and point["is_mask_ap"] is False
        assert (point["tp"], point["fp"], point["fn"]) == (2, 1, 0)
    assert metadata["phase"] == "complete"
    assert metadata["source_sha256_before"] == metadata["source_sha256_after"]
    assert len(metadata["RGB_sha256_before_after"]) == 3
    assert torch.equal(original, fixture["dataset"].items[0]["target"]["amodal_masks"])


def test_partial_dataset_is_rejected_before_any_model_call(fixture):
    fixture["dataset"].items.pop()
    model = TinyModel(fixture["predictions"])
    with pytest.raises(ValueError, match="Full VAL dataset length differs"):
        run_fixture(fixture, model)
    assert model.calls == 0 and not (fixture["output"] / "summary.json").exists()


def test_public_api_rejects_unpinned_or_partial_provenance(fixture):
    with pytest.raises(ValueError, match="provenance SHA256 differs"):
        evaluation.evaluate_papple(TinyModel(fixture["predictions"]), fixture["dataset"], "cpu",
                                   fixture["output"], fixture["contract"]["provenance_path"])
    with pytest.raises(ValueError, match="814 images, 3449 GT and 13 empty"):
        evaluation._contract_from_provenance(fixture["provenance"], official=True)


@pytest.mark.parametrize("mutation", ["mask", "id_order", "boxes", "partial_annotation"])
def test_actual_gt_tampering_is_rejected_without_completed_summary(fixture, mutation):
    target = fixture["dataset"].items[0]["target"]
    if mutation == "mask":
        target["amodal_masks"][0, 0, 0] = 0
    elif mutation == "id_order":
        target["instance_ids"] = ["another-apple"]
    elif mutation == "boxes":
        target["boxes"] = [[0, 0, 8, 8]]
    else:
        target["annotation_validity"]["amodal"] = False
        target["amodal_masks"] = None
    model = TinyModel(fixture["predictions"])
    with pytest.raises(ValueError):
        run_fixture(fixture, model)
    assert model.calls == 0 and not (fixture["output"] / "summary.json").exists()


def test_rgb_byte_tampering_is_rejected(fixture):
    path = fixture["contract"]["images"][0]["RGB_path"]
    with open(path, "wb") as stream:
        stream.write(b"tampered RGB")
    model = TinyModel(fixture["predictions"])
    with pytest.raises(ValueError, match="RGB bytes/SHA256 differ"):
        run_fixture(fixture, model)
    assert model.calls == 0 and not (fixture["output"] / "summary.json").exists()


def test_gt_mutation_during_model_call_is_detected_and_training_mode_restored(fixture):
    def mutate(call):
        if call == 1:
            fixture["dataset"].items[0]["target"]["visible_masks"][0, 0, 0] = 0
    model = TinyModel(fixture["predictions"], on_call=mutate)
    model.train()
    with pytest.raises(ValueError, match="GT mask SHA256/area differs"):
        run_fixture(fixture, model)
    assert model.training and not (fixture["output"] / "summary.json").exists()
    metadata = json.loads((fixture["output"] / "evaluation_metadata.json").read_text(encoding="utf-8"))
    assert metadata["phase"] == "failed"


def test_provenance_change_during_evaluation_is_rejected(fixture):
    def mutate(call):
        if call == 1:
            fixture["contract"]["provenance_path"].write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="files changed during evaluation"):
        run_fixture(fixture, TinyModel(fixture["predictions"], on_call=mutate))
    assert not (fixture["output"] / "summary.json").exists()


def test_later_frame_cannot_change_an_already_evaluated_rgb(fixture):
    def mutate(call):
        if call == 2:
            path = fixture["contract"]["images"][0]["RGB_path"]
            with open(path, "wb") as stream:
                stream.write(b"late RGB mutation")
    with pytest.raises(ValueError, match="RGB bytes/SHA256 differ"):
        run_fixture(fixture, TinyModel(fixture["predictions"], on_call=mutate))
    assert not (fixture["output"] / "summary.json").exists()


def test_completed_evaluation_is_never_overwritten(fixture):
    run_fixture(fixture)
    summary_path = fixture["output"] / "summary.json"
    saved = summary_path.read_bytes()
    model = TinyModel(fixture["predictions"])
    with pytest.raises(ValueError, match="fresh evaluation directory"):
        run_fixture(fixture, model)
    assert model.calls == 0 and summary_path.read_bytes() == saved


def test_original_visible_outside_amodal_anomaly_is_retained(fixture):
    target = deepcopy(fixture["dataset"].items[0]["target"])
    rows = deepcopy(fixture["contract"]["truth"]["fixture_0.png"])
    target["visible_masks"].fill_(1)
    rows[0].update(visible_area_pixels=64, visible_area_fraction=2.0,
                   visible_pixels_outside_amodal=32, occlusion_stratum="annotation_visible_area_exceeds_amodal",
                   visible_mask_uint8_sha256=tensor_sha(target["visible_masks"][0]))
    actual = evaluation._audit_target(target, fixture["contract"]["images"][0], rows)
    assert len(actual["boxes"]) == 1
    assert np.count_nonzero(actual["visible_masks"]) == 64
    assert np.count_nonzero(actual["amodal_masks"]) == 32
