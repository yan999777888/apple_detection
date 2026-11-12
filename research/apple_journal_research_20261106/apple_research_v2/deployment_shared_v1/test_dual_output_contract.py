"""CPU contract tests, not trained-model evaluation or a speed benchmark."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if HERE.drive.upper() != "F:":
    raise RuntimeError("Tests must run from F")
sys.dont_write_bytecode = True
os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", YOLO_OFFLINE="1", YOLO_AUTOINSTALL="0", HF_HUB_OFFLINE="1", WANDB_MODE="disabled", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
for env in ("YOLO_CONFIG_DIR", "MPLCONFIGDIR", "TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR", "PIP_CACHE_DIR", "TEMP", "TMP", "TMPDIR"):
    path = HERE / "cpu_test_runtime" / env.lower()
    path.mkdir(parents=True, exist_ok=True)
    os.environ[env] = str(path)
import numpy as np
from dual_output_contract import *
import torch
from PIL import Image
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import ultralytics.models.yolo.segment.predict as segment_module

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
PREDICTOR_PATH = Path(segment_module.__file__).resolve()
PREDICTOR_SHA = hashlib.sha256(PREDICTOR_PATH.read_bytes()).hexdigest()
if PREDICTOR_SHA != NATIVE_SEGMENT_PREDICT_SHA256:
    raise RuntimeError("Native segmentation postprocessor changed; contract review required")
source = json.loads((ROOT / "experiments/joint_eval_contract_v1/contract_test_receipt.json").read_text(encoding="utf-8"))["real_images"]["stage"]
with Image.open(source) as image:
    REAL_BGR = np.asarray(image.convert("RGB"))[..., ::-1].copy()
REAL_SHA = hashlib.sha256(Path(source).read_bytes()).hexdigest()


def make_result(boxes, masks=None, task="stage", shape=(32, 32)):
    return NS(boxes=NS(data=np.array(boxes, dtype=np.float32, copy=True).reshape(-1, 6)),
              masks=None if masks is None else NS(data=np.array(masks, copy=True)),
              orig_shape=shape, names=NAMES[task].copy())


def stage_fixture():
    boxes = np.array([[1, 1, 11, 11, .8, 0], [15, 15, 25, 25, .7, 0]], dtype=np.float32)
    masks = np.zeros((2, 32, 32), dtype=np.uint8)
    masks[0, 1:11, 1:11] = 1; masks[1, 15:25, 15:25] = 1
    return boxes, masks


def native_segment_construct(count, masks):
    obj = object.__new__(SegmentationPredictor)
    obj.args = NS(retina_masks=True)
    obj.model = NS(names=NAMES["stage"])
    h, w = REAL_BGR.shape[:2]
    pred = torch.zeros((count, 38), dtype=torch.float32)
    for i in range(count):
        pred[i, :6] = torch.tensor([1 + i, 1, 15 + i, 15, .8, 0])
    tensor = torch.empty((1, 3, h, w), dtype=torch.float32)
    proto = torch.empty((32, 160, 160), dtype=torch.float32)
    with patch.object(segment_module.ops, "process_mask_native", return_value=masks):
        return obj.construct_result(pred, tensor, REAL_BGR, source, proto)


class OutputContractTests(unittest.TestCase):
    def test_raw_fixed_names_shapes_finite(self):
        output = {k: np.zeros(v, dtype=np.float32) for k, v in RAW_SHAPES.items()}
        self.assertTrue(validate_raw_outputs(output)["passed"])
        self.assertTrue(validate_raw_outputs(tuple(output.values()))["passed"])

    def test_raw_missing_extra_or_bad_container(self):
        output = {k: np.zeros(v, dtype=np.float32) for k, v in RAW_SHAPES.items()}
        self.assertFalse(validate_raw_outputs({**output, "extra": 1})["passed"])
        self.assertFalse(validate_raw_outputs({"stage_prototypes": output["stage_prototypes"]})["passed"])
        self.assertFalse(validate_raw_outputs([1, 2])["passed"])

    def test_raw_wrong_shape_dtype_rejected(self):
        output = {k: np.zeros(v, dtype=np.float32) for k, v in RAW_SHAPES.items()}
        output["stage_detections"] = output["stage_detections"].astype(np.float16)
        self.assertFalse(validate_raw_outputs(output)["passed"])
        output["stage_prototypes"] = np.zeros((1, 32, 80, 80), dtype=np.float32)
        self.assertFalse(validate_raw_outputs(output)["passed"])

    def test_raw_nonfinite_rejected(self):
        for name in RAW_SHAPES:
            output = {k: np.zeros(v, dtype=np.float32) for k, v in RAW_SHAPES.items()}
            output[name].flat[0] = np.nan
            report = validate_raw_outputs(output)
            self.assertFalse(report["passed"])
            json.dumps(report, allow_nan=False)

    def test_raw_invalid_scores_and_class_ids(self):
        output = {k: np.zeros(v, dtype=np.float32) for k, v in RAW_SHAPES.items()}
        output["stage_detections"][0, 0, 5] = 3
        output["defect_detections"][0, 0, 4] = 1.01
        self.assertFalse(validate_raw_outputs(output)["passed"])

    def test_raw_boundary_report_fixed_strict_operator(self):
        output = {k: np.zeros(v, dtype=np.float32) for k, v in RAW_SHAPES.items()}
        output["stage_detections"][0, :3, 4] = [.25 - 1e-5, .25, .25 + 1e-5]
        report = validate_raw_outputs(output)["outputs"]["stage_detections"]
        self.assertEqual(report["above_confidence_count"], 1)
        self.assertEqual(report["at_confidence_count"], 1)
        self.assertEqual(len(report["threshold_boundary_rows"]), 3)

    def test_permutation_matching_and_no_input_mutation(self):
        boxes, masks = stage_fixture(); original = boxes.copy()
        a = make_result(boxes, masks); b = make_result(boxes[::-1], masks[::-1])
        report = compare_results(a, b, "stage")
        self.assertTrue(report["passed"], report)
        self.assertTrue(np.array_equal(a.boxes.data, original))
        json.dumps(report, allow_nan=False)

    def test_equal_box_score_mask_tie_permutation(self):
        boxes, masks = stage_fixture(); boxes[1] = boxes[0]
        self.assertTrue(compare_results(make_result(boxes, masks), make_result(boxes[::-1], masks[::-1]), "stage")["passed"])

    def test_score_error_rejected(self):
        boxes, masks = stage_fixture(); b = boxes.copy(); b[0, 4] += .001
        self.assertFalse(compare_results(make_result(boxes, masks), make_result(b, masks), "stage")["passed"])

    def test_box_coordinate_and_iou_failures_reported(self):
        boxes, masks = stage_fixture(); b = boxes.copy(); b[0, 0] += .2
        result = compare_results(make_result(boxes, masks), make_result(b, masks), "stage")
        self.assertFalse(result["passed"])
        self.assertIn(0, result["unmatched_native_indices"])

    def test_mask_iou_failure(self):
        boxes, masks = stage_fixture(); bad = masks.copy(); bad[0, 1, 1] = 0
        self.assertFalse(compare_results(make_result(boxes, masks), make_result(boxes, bad), "stage")["passed"])

    def test_class_count_change_rejected(self):
        boxes, masks = stage_fixture(); b = boxes.copy(); b[0, 5] = 1
        result = compare_results(make_result(boxes, masks), make_result(b, masks), "stage")
        self.assertFalse(result["passed"]); self.assertIn("class_counts_changed", result["issues"])

    def test_boundary_count_change_never_waived(self):
        boxes, masks = stage_fixture(); boxes = boxes[:1]; masks = masks[:1]; boxes[0, 4] = .25 + 1e-5
        result = compare_results(make_result(boxes, masks), make_result([], None), "stage")
        self.assertFalse(result["passed"])
        self.assertIn("detection_count_changed", result["issues"])
        self.assertEqual(len(result["native_threshold_boundary"]), 1)

    def test_empty_none_and_empty_tensor_are_distinct(self):
        empty = make_result([], None)
        tensor_empty = make_result([], np.zeros((0, 32, 32), np.uint8))
        self.assertTrue(compare_results(empty, empty, "stage")["passed"])
        self.assertTrue(compare_results(tensor_empty, tensor_empty, "stage")["passed"])
        self.assertFalse(compare_results(empty, tensor_empty, "stage")["passed"])

    def test_invalid_native_mask_contract_rejected(self):
        boxes, masks = stage_fixture(); a = make_result(boxes, masks)
        self.assertFalse(compare_results(a, make_result(boxes, None), "stage")["passed"])
        self.assertFalse(compare_results(a, make_result(boxes, masks[:, :16]), "stage")["passed"])
        self.assertFalse(compare_results(a, make_result(boxes, np.zeros_like(masks)), "stage")["passed"])

    def test_detect_masks_and_wrong_metadata_rejected(self):
        boxes, masks = stage_fixture()
        self.assertFalse(compare_results(make_result(boxes, None, "defect"), make_result(boxes, masks, "defect"), "defect")["passed"])
        a = make_result([], None, "defect"); b = make_result([], None, "defect"); b.names = {0: "made_up", 1: "class_1"}
        self.assertFalse(compare_results(a, b, "defect")["passed"])

    def test_actual_native_no_detections_masks_none(self):
        result = native_segment_construct(0, None)
        self.assertEqual(len(result.boxes), 0); self.assertIsNone(result.masks)

    def test_actual_native_all_zero_masks_filtered(self):
        h, w = REAL_BGR.shape[:2]
        result = native_segment_construct(2, torch.zeros((2, h, w), dtype=torch.uint8))
        self.assertEqual(len(result.boxes), 0)
        self.assertIsNotNone(result.masks)
        self.assertEqual(tuple(result.masks.data.shape), (0, h, w))
        self.assertTrue(compare_results(result, result, "stage")["passed"])

    def test_actual_native_mixed_empty_masks_filtered(self):
        h, w = REAL_BGR.shape[:2]
        masks = torch.zeros((3, h, w), dtype=torch.uint8)
        masks[1, 1:15, 2:16] = 1
        result = native_segment_construct(3, masks)
        self.assertEqual(len(result.boxes), 1); self.assertEqual(len(result.masks), 1)
        self.assertEqual(float(result.boxes.xyxy[0, 0]), 2)
        self.assertTrue(compare_results(result, result, "stage")["passed"])

    def test_actual_native_confidence_strict_greater(self):
        obj = object.__new__(DetectionPredictor)
        obj.args = NS(conf=.25, iou=.7, classes=None, agnostic_nms=False, max_det=300, task="detect")
        obj.model = NS(names=NAMES["defect"], end2end=True)
        obj.batch = ([source], [REAL_BGR], [""])
        h, w = REAL_BGR.shape[:2]
        raw = torch.tensor([[[1, 1, 11, 11, .25 - 1e-5, 0], [1, 1, 11, 11, .25, 0], [1, 1, 11, 11, .25 + 1e-5, 0]]], dtype=torch.float32)
        result = obj.postprocess(raw, torch.empty((1, 3, h, w)), [REAL_BGR])[0]
        self.assertEqual(len(result.boxes), 1)
        self.assertGreater(float(result.boxes.conf[0]), .25)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OutputContractTests)
    test_names = [test.id() for test in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    assert torch.get_num_threads() <= 2 and torch.get_num_interop_threads() == 1
    assert not torch.cuda.is_initialized()
    receipt = {"schema": "dual_output_contract_cpu_tests_v1", "phase": "complete", "tests_passed": result.testsRun,
        "tests": test_names, "test_purpose": "Deterministic contract and actual native postprocess branch checks using controlled tensors; no trained-model quality or speed claim",
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256((HERE / "dual_output_contract.py").read_bytes()).hexdigest(),
        "native_segment_predict_path": str(PREDICTOR_PATH), "native_segment_predict_sha256": PREDICTOR_SHA,
        "real_original_image": str(source), "real_original_image_sha256": REAL_SHA, "real_original_hw": list(REAL_BGR.shape[:2]),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "cuda_initialized": torch.cuda.is_initialized(),
        "native_empty_mask_observations": {"no_input_detections": "zero boxes, masks=None", "all_masks_empty": "zero boxes, empty native-resolution masks tensor", "mixed_empty_masks": "empty masks and corresponding boxes filtered"}, "protocol": PROTOCOL}
    target = HERE / "dual_contract_cpu_receipt.json"
    with target.open("x", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"phase": "complete", "tests_passed": result.testsRun, "receipt": str(target), "cuda_initialized": False}))
