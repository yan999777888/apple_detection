"""CPU-only checks for Stage2 full-VAL scope, binding and reversible patches."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage2_eval_under_test", ROOT / "scripts/evaluate_stage2.py")
stage2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage2)


def configuration(**overrides):
    return {"candidate_policy": {"score_threshold": None, "box_nms_iou": .8, "topk": 64},
            "preprocess_device": "cpu", "channels_last": False, "cudnn_benchmark": False,
            "cudnn_deterministic": False, "roi_align_precision": "fp32", **overrides}


def arguments(config="baseline.json", stage="stage2.json", *extra):
    return stage2.build_parser().parse_args([
        "--config", str(config), "--stage2-config", str(stage), "--stage", "base",
        "--output", "unused", *extra])


@pytest.mark.parametrize("change", [{"preprocess_device": "gpu"}, {"channels_last": 1},
    {"cudnn_benchmark": "true"}, {"cudnn_deterministic": None}, {"roi_align_precision": "autocast"},
    {"candidate_policy": {"topk": True}}, {"candidate_policy": {"score_threshold": float("nan")}},
    {"extra": 1}])
def test_invalid_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        stage2.normalize_stage2_config(configuration(**change))


def test_missing_key_is_not_silently_defaulted():
    value = configuration()
    del value["cudnn_benchmark"]
    with pytest.raises(ValueError):
        stage2.normalize_stage2_config(value)


@pytest.mark.parametrize("extra", [("--pre-nms-threshold", ".738"), ("--pre-nms-threshold", ".35"),
    ("--fixed-only",), ("--skip-dead-amodal",), ("--precision", "fp32"),
    ("--roi-precision", "fp32"), ("--short-side", "512"), ("--roi-chunk-size", "32")])
def test_complete_pr_and_screened_settings_enforced(extra):
    with pytest.raises(ValueError):
        stage2.validate_args(arguments("baseline.json", "stage2.json", *extra))


def test_original_only_or_both_views_keep_complete_pr():
    for views in ("original", "both"):
        args = arguments("baseline.json", "stage2.json", "--views", views)
        stage2.validate_args(args)
        assert args.pre_nms_threshold is None and args.roi_chunk_size == 64


@pytest.fixture
def files(tmp_path):
    cfg = {"path_base": ".", "paths": {}}
    for name in ("base", "small"):
        checkpoint = tmp_path / (name + ".pt")
        checkpoint.write_bytes(("original_" + name).encode())
        cfg["paths"][name + "_checkpoint"] = checkpoint.name
        cfg[name + "_checkpoint_sha256"] = stage2.fast.sha(checkpoint)
    baseline, selected = tmp_path / "baseline.json", tmp_path / "selected.json"
    baseline.write_text(json.dumps(cfg), encoding="utf-8")
    selected.write_text(json.dumps(configuration(channels_last=True, cudnn_benchmark=True)), encoding="utf-8")
    return arguments(baseline, selected), baseline, selected


def test_stage2_file_hash_and_duplicate_rejection(files):
    _, _, path = files
    _, receipt = stage2.load_stage2_config(path)
    assert receipt["sha256"] == stage2.fast.sha(path)
    path.write_text('{"channels_last":true,"channels_last":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        stage2.load_stage2_config(path)


def test_backend_flags_restored_on_failure():
    before = stage2.backend_state(torch)
    with pytest.raises(RuntimeError, match="deliberate"):
        with stage2.scoped_backend(torch, configuration(cudnn_benchmark=not before["cudnn_benchmark"])):
            assert torch.backends.cudnn.benchmark is not before["cudnn_benchmark"]
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is True
            raise RuntimeError("deliberate")
    assert stage2.backend_state(torch) == before


class FakeAdapter:
    def __init__(self, model, **kwargs):
        self.model, self.kwargs = model, kwargs

    def predict_views(self, pixels, pixel_mask=None):
        return pixels, pixel_mask


class FakeModel:
    def to(self, **kwargs):
        self.kwargs = kwargs
        return self


@pytest.mark.parametrize("cuda_preprocess", [False, True])
def test_scoped_factory_layout_provenance_and_restoration(files, cuda_preprocess):
    args, baseline, selected = files
    config = configuration(channels_last=True, cudnn_benchmark=True,
                           preprocess_device="cuda" if cuda_preprocess else "cpu")
    selected.write_text(json.dumps(config), encoding="utf-8")
    adapter_module = SimpleNamespace(FastDeploymentAdapter=object())
    data_module = SimpleNamespace(ODInputDataset=type("OriginalDataset", (), {}))
    before_adapter, before_dataset = adapter_module.FastDeploymentAdapter, data_module.ODInputDataset
    functions = {key: getattr(stage2.fast, key) for key in
                 ("build_parser", "deployment_spec", "source_inventory", "final_integrity")}
    flags = stage2.backend_state(torch)
    with pytest.raises(RuntimeError, match="finish test"):
        with stage2.patched_evaluator(args, adapter_type=FakeAdapter, adapter_module=adapter_module,
                                     data_module=data_module, torch_module=torch,
                                     prepare_pixels=lambda *a, **k: None):
            assert (data_module.ODInputDataset is not before_dataset) is cuda_preprocess
            model = FakeModel()
            adapter = adapter_module.FastDeploymentAdapter(model, roi_precision="autocast")
            assert model.kwargs["memory_format"] == torch.channels_last
            assert adapter.kwargs["candidate_policy"] == config["candidate_policy"]
            assert adapter.kwargs["roi_align_precision"] == "fp32"
            assert not adapter.kwargs["capture_candidates"] and not adapter.kwargs["profile_pools"]
            pixels = torch.arange(2 * 3 * 7 * 9).reshape(2, 3, 7, 9).float()
            returned, mask = adapter.predict_views(pixels)
            assert returned.is_contiguous(memory_format=torch.channels_last)
            assert torch.equal(returned, pixels) and mask is None
            spec = stage2.fast.deployment_spec(args)
            assert spec["pre_nms_threshold"] is None
            assert spec["evaluation_policy"]["threshold_0738_full_PR_forbidden"] is True
            assert spec["user_hard_abort_precision_floor"] == .94
            assert spec["hard_abort_floor_is_promotion_threshold"] is False
            assert spec["protected_original_FP32_four_view_P90"]["precision"] > .9443
            assert spec["automatic_promotion"] is False
            sources = stage2.fast.source_inventory()
            assert str(Path("deployment/stage2.py")) in sources
            meta = {"config": str(baseline), "config_sha256": stage2.fast.sha(baseline), "deployment": spec}
            assert stage2.fast.final_integrity(meta, sources)["original_weights_before_after_identical"]
            parsed = stage2.fast.build_parser().parse_args([
                "--config", str(baseline), "--stage2-config", str(selected), "--stage", "fuse", "--output", "out"])
            assert parsed.pre_nms_threshold is None and parsed.stage2_config == selected
            selected.write_text(json.dumps(configuration()), encoding="utf-8")
            with pytest.raises(RuntimeError, match="changed during full VAL"):
                stage2.fast.final_integrity(meta, sources)
            raise RuntimeError("finish test")
    assert adapter_module.FastDeploymentAdapter is before_adapter
    assert data_module.ODInputDataset is before_dataset
    assert stage2.backend_state(torch) == flags
    assert all(getattr(stage2.fast, key) is value for key, value in functions.items())


def test_gpu_dataset_keeps_genuine_target_and_raw_crop_without_using_gt(tmp_path):
    """Mock CUDA output container; no CUDA operation or accuracy claim."""
    from apple_seg.od_data import ODInputDataset, legacy_rgb_tensor
    image = Image.fromarray(np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3))
    path = tmp_path / "rgb.png"
    image.save(path)
    visible, amodal = object(), object()
    original = {"original_hw": (6, 8), "visible_masks": visible, "amodal_masks": amodal,
                "source": {"rgb_path": str(path), "crop_xyxy": [1, 1, 9, 7]}, "instance_ids": [17]}

    class Native:
        augment, split, annotation_paths, annotation_sha256 = False, "val", {}, {}
        def __getitem__(self, index):
            return {"target": original}

    class MockCudaPixels:
        def __init__(self, tensor):
            self.tensor, self.shape, self.dtype = tensor, tensor.shape, tensor.dtype
            self.device = SimpleNamespace(type="cuda")
        def detach(self):
            return self
        def cpu(self):
            return self.tensor

    received = []
    def prepare(rgb, short, maximum, device, **kwargs):
        received.append((np.asarray(rgb).copy(), short, maximum, device, kwargs))
        return [MockCudaPixels(legacy_rgb_tensor(rgb, short, maximum))]

    dataset = stage2.gpu_dataset_type(ODInputDataset, prepare, torch)(Native(), short_sides=(6,), max_size=8)
    item = dataset[0]
    assert item["target"]["visible_masks"] is visible and item["target"]["amodal_masks"] is amodal
    assert item["target"]["instance_ids"] == [17]
    assert original["source"] == {"rgb_path": str(path), "crop_xyxy": [1, 1, 9, 7]}
    assert np.array_equal(received[0][0], np.asarray(image)[1:7, 1:9])
    assert received[0][1:] == (6, 8, "cuda", {"preprocess_device": "cuda", "channels_last": False})
    assert item["target"]["source"]["resize_hw"] == [6, 8]
    assert item["target"]["source"]["preprocessing_device"] == "cuda"
    assert len(item["target"]["source"]["normalized_input_sha256"]) == 64


def test_fusion_rejects_mixed_stage2_settings_before_loading_raw(tmp_path):
    expected = {"views": "both", "stage2_configuration": configuration()}
    wrong = deepcopy(expected)
    wrong["stage2_configuration"]["preprocess_device"] = "cuda"
    sources = {"stage2": "frozen"}
    stage2.fast.write(tmp_path / "base/prediction_metadata.json", {
        "phase": "complete", "deployment": wrong, "source_sha256": sources,
        "config_sha256": "config", "checkpoint_sha256": "base-sha"})
    with pytest.raises(ValueError, match="deployment spec"):
        stage2.fast.run_fusion(SimpleNamespace(), {"base_checkpoint_sha256": "base-sha"}, {}, sources,
                              tmp_path, {"deployment": expected, "config_sha256": "config"})
