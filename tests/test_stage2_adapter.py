"""CPU-only preprocessing and adapter lifecycle tests; no GPU validation."""
from collections import OrderedDict
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision.ops import MultiScaleRoIAlign

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "strong_runtime/src")]
from apple_seg.od_data import legacy_rgb_tensor
from deployment.pre_mask import PreMaskDeploymentAdapter
from deployment.stage2 import Stage2DeploymentAdapter, prepare_pixels


class ODRoIHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.box_roi_pool = MultiScaleRoIAlign(["0"], 3, 2)
        self.mask_roi_pool = MultiScaleRoIAlign(["0"], 5, 2)
        self.highres_roi_pool = None
        self.final_nms = "dual_mask"
        self.configuration = {}
        self.roi_chunk_size = 8

    def _roi_predictions(self, *args):
        raise AssertionError("CPU lifecycle test must not run a neural head")

    _decoded_boxes = _roi_predictions
    _final_nms = _roi_predictions
    _mask_predictions = _roi_predictions


class ODAmodalDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roi_heads = ODRoIHeads()
        self.weight = torch.nn.Parameter(torch.ones(1))


# Explicit tiny stand-ins only for existing adapter structural guards. No real
# checkpoints or model constructors are invoked by these lifecycle tests.
ODRoIHeads.__module__ = ODAmodalDetector.__module__ = "test_stage2.od_model"


def tiny_model():
    return ODAmodalDetector().eval()


def rgb(height=37, width=51):
    array = ((np.arange(height * width * 3).reshape(height, width, 3) * 17) % 256).astype(np.uint8)
    return Image.fromarray(array)


@pytest.mark.parametrize("short,max_size", [(24, 32), (64, 96), (37, 51)])
@pytest.mark.parametrize("channels_last", [False, True])
def test_cpu_preprocess_matches_frozen_legacy_exactly(short, max_size, channels_last):
    image = rgb()
    expected = legacy_rgb_tensor(image, short, max_size)[None]
    actual = prepare_pixels(image, short, max_size, "cpu", channels_last=channels_last)
    assert actual.dtype == torch.float32 and actual.device.type == "cpu"
    assert torch.equal(actual, expected)
    assert actual.is_contiguous(memory_format=torch.channels_last if channels_last else torch.contiguous_format)


def test_cpu_preprocess_retains_fp32_inside_outer_autocast():
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = prepare_pixels(rgb(), 24, 32, "cpu")
    assert actual.dtype == torch.float32
    assert torch.equal(actual, legacy_rgb_tensor(rgb(), 24, 32)[None])


@pytest.mark.parametrize("kwargs", [{"preprocess_device": "other"}, {"channels_last": 1},
                                    {"preprocess_device": "cuda"}])
def test_invalid_preprocessing_arguments_fail_without_gpu(kwargs):
    with pytest.raises(ValueError):
        prepare_pixels(rgb(), 24, 32, "cpu", **kwargs)


def test_preprocess_image_and_size_guards():
    with pytest.raises(TypeError):
        prepare_pixels(np.zeros((8, 8, 3)), 8, 8, "cpu")
    with pytest.raises(ValueError):
        prepare_pixels(rgb(), 0, 32, "cpu")


def test_default_prediction_delegates_to_existing_path(monkeypatch):
    image, mask, sentinel = torch.zeros(1, 3, 4, 4), torch.ones(1, 4, 4), object()
    def original(adapter, pixels, pixel_mask):
        assert pixels is image and pixel_mask is mask
        adapter.last_stats = {"original_path": True}
        return sentinel
    monkeypatch.setattr(PreMaskDeploymentAdapter, "predict_views", original)
    adapter = Stage2DeploymentAdapter(tiny_model())
    assert adapter.predict_views(image, mask) is sentinel
    assert adapter.last_stats == {"original_path": True, "roi_align_precision": "fp32"}


@pytest.mark.parametrize("profile", [False, True])
def test_fp32_pool_values_and_context_restoration(profile):
    model = tiny_model()
    pool = model.roi_heads.mask_roi_pool
    features = OrderedDict({"0": torch.arange(1 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)})
    boxes = [torch.tensor([[1.2, 2.3, 19.5, 25.4], [4., 4., 8., 8.]])]
    expected = pool(features, boxes, [(32, 32)])
    pool.scales, pool.map_levels = None, None
    version, weight = model.weight._version, model.weight.detach().clone()
    with Stage2DeploymentAdapter(model, roi_chunk_size=64, profile_pools=profile) as adapter:
        actual = pool(features, boxes, [(32, 32)])
        assert torch.equal(actual, expected)
        assert model.roi_heads.roi_chunk_size == 64
        if profile:
            row = adapter.last_stats["roi_pools"][0]
            assert row["pool"] == "mask_roi_pool" and row["roi_count"] == 2
            assert row["operator_input_dtype"] == row["output_dtype"] == "torch.float32"
            assert row["elapsed_ms"] >= 0 and row["cuda_synchronized"] is False
        else:
            assert "roi_pools" not in adapter.last_stats
    assert model.roi_heads.roi_chunk_size == 8
    assert pool.scales is None and pool.map_levels is None
    assert "forward" not in model.roi_heads.__dict__ and "forward" not in pool.__dict__
    assert "_strong_fast_adapter_active" not in model.roi_heads.__dict__
    assert model.weight._version == version and torch.equal(model.weight, weight)


def test_profile_toggle_works_without_reentering():
    model = tiny_model()
    with Stage2DeploymentAdapter(model) as adapter:
        pool = model.roi_heads.box_roi_pool
        args = ({"0": torch.ones(1, 2, 8, 8)}, [torch.tensor([[1., 1., 6., 6.]])], [(8, 8)])
        pool(*args)
        assert "roi_pools" not in adapter.last_stats
        adapter.profile_pools = True
        pool(*args)
        assert len(adapter.last_stats["roi_pools"]) == 1
        adapter.profile_pools = False
        pool(*args)
        assert len(adapter.last_stats["roi_pools"]) == 1


def test_exception_restores_pool_methods_and_shared_pool_alias():
    model = tiny_model()
    model.roi_heads.highres_roi_pool = model.roi_heads.mask_roi_pool
    with pytest.raises(RuntimeError, match="intentional"):
        with Stage2DeploymentAdapter(model, profile_pools=True):
            raise RuntimeError("intentional")
    assert all("forward" not in pool.__dict__ for pool in
               (model.roi_heads, model.roi_heads.box_roi_pool, model.roi_heads.mask_roi_pool))


@pytest.mark.parametrize("kwargs", [{"roi_align_precision": "bf16"}, {"profile_pools": 1},
                                    {"roi_align_precision": "fp16", "roi_precision": "fp32"}])
def test_invalid_adapter_options(kwargs):
    with pytest.raises(ValueError):
        Stage2DeploymentAdapter(tiny_model(), **kwargs)


def test_half_pool_mode_rejects_cpu_execution_without_fallback():
    model = tiny_model()
    with Stage2DeploymentAdapter(model, roi_align_precision="fp16", roi_precision="autocast") as adapter:
        with pytest.raises(ValueError, match="requires CUDA"):
            adapter.predict_views(torch.zeros(1, 3, 8, 8))
        with pytest.raises(ValueError, match="requires CUDA"):
            model.roi_heads.box_roi_pool({"0": torch.ones(1, 1, 8, 8)},
                                        [torch.tensor([[1., 1., 4., 4.]])], [(8, 8)])
    assert "forward" not in model.roi_heads.box_roi_pool.__dict__


def test_parent_eval_and_fp32_parameter_guards_still_apply():
    with pytest.raises(ValueError, match="eval"):
        with Stage2DeploymentAdapter(tiny_model().train()):
            pass
    with pytest.raises(ValueError, match="FP32"):
        with Stage2DeploymentAdapter(tiny_model().half()):
            pass


def test_cpu_suite_never_initializes_cuda():
    assert not torch.cuda.is_initialized()
