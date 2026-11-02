"""Isolated Stage 2 experiments on unchanged original strong-model weights.

FP32 pooling delegates prediction to the existing PreMaskDeploymentAdapter.
The opt-in FP16 path changes ROI feature/coordinate precision and requires
new output and accuracy checks. No speed or numerical equivalence is assumed.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from types import MethodType
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import MultiScaleRoIAlign

from deployment.pre_mask import PreMaskDeploymentAdapter


@torch.inference_mode()
def prepare_pixels(rgb, short_side, max_size, device, preprocess_device="cpu", channels_last=False):
    """Return FP32 normalized [1,3,H,W], preserving legacy operation order.

CPU mode uses the frozen function verbatim before transfer. CUDA mode sends
uint8 CHW to the target, then does float32 /255, normalization and bilinear
resize (no antialias). CPU/GPU kernels need independent numerical validation.
Only tensor layout is changed by channels_last; no model is accessed here.
"""
    from apple_seg.od_data import (
        legacy_rgb_tensor, _resize_scale, resized_hw, NORMALIZATION_MEAN, NORMALIZATION_STD,
    )
    if preprocess_device not in ("cpu", "cuda") or type(channels_last) is not bool:
        raise ValueError("Expected preprocess_device='cpu'|'cuda' and boolean channels_last")
    if not isinstance(rgb, Image.Image):
        raise TypeError("prepare_pixels expects a PIL image")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA output devices are supported")
    if preprocess_device == "cuda" and device.type != "cuda":
        raise ValueError("CUDA preprocessing requires a CUDA output device")
    rgb = rgb.convert("RGB")
    scale = _resize_scale(rgb.height, rgb.width, short_side, max_size)
    resized_hw(rgb.height, rgb.width, short_side, max_size)
    if preprocess_device == "cpu":
        with torch.autocast("cpu", enabled=False):
            pixels = legacy_rgb_tensor(rgb, short_side, max_size)[None].to(device)
    else:
        array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8).transpose(2, 0, 1))
        # Transfer byte input once; no hidden float32 CPU normalization/resize.
        byte_pixels = torch.from_numpy(array).to(device)
        with torch.autocast("cuda", enabled=False):
            image = byte_pixels.to(torch.float32).div_(255.)
            mean = torch.as_tensor(NORMALIZATION_MEAN, dtype=torch.float32, device=device)[:, None, None]
            std = torch.as_tensor(NORMALIZATION_STD, dtype=torch.float32, device=device)[:, None, None]
            image = (image - mean) / std
            pixels = F.interpolate(image[None], scale_factor=scale, mode="bilinear", align_corners=False,
                                   recompute_scale_factor=True, antialias=False)
    layout = torch.channels_last if channels_last else torch.contiguous_format
    return pixels.contiguous(memory_format=layout)


class Stage2DeploymentAdapter(PreMaskDeploymentAdapter):
    """Reversible ROI pool precision and diagnostic instrumentation.

Use FP32 checkpoint parameters, model.eval(), and caller CUDA autocast FP16
for roi_align_precision='fp16'. FP16 ROIAlign executes with autocast disabled
on half features AND half coordinates; this is a precision experiment, not
the installed torchvision autocast wrapper (which computes in FP32).

profile_pools can be toggled while active. Only when true do pool wrappers
synchronize or append last_stats['roi_pools'] records. These nested timings
are part of ROI head timings, and must not be added to parent timings.
"""
    def __init__(self, model, *, roi_align_precision="fp32", profile_pools=False, **kwargs):
        if roi_align_precision not in ("fp32", "fp16"):
            raise ValueError("roi_align_precision must be fp32 or fp16")
        if type(profile_pools) is not bool:
            raise ValueError("profile_pools must be boolean")
        super().__init__(model, **kwargs)
        if roi_align_precision == "fp16" and self.roi_precision != "autocast":
            raise ValueError("FP16 ROIAlign requires roi_precision='autocast'")
        self.roi_align_precision = roi_align_precision
        self.profile_pools = profile_pools

    def __enter__(self):
        # Capture genuine pool forwards before the parent installs its FP32
        # wrappers. Patching stacks are restored in reverse order by _restore.
        pools, seen = [], set()
        for name in ("box_roi_pool", "mask_roi_pool", "highres_roi_pool"):
            pool = getattr(self.model.roi_heads, name, None)
            if pool is None or id(pool) in seen:
                continue
            seen.add(id(pool))
            if self.roi_align_precision == "fp16" and type(pool) is not MultiScaleRoIAlign:
                raise NotImplementedError("FP16 pooling supports the original MultiScaleRoIAlign only")
            pools.append((name, pool, pool.forward))
        super().__enter__()
        try:
            for name, pool, original in pools:
                # MultiScaleRoIAlign lazily fills these caches. Restore them
                # as well as method overrides so even exceptions leave no edit.
                for attr in ("scales", "map_levels"):
                    if hasattr(pool, attr):
                        self._set(pool, attr, getattr(pool, attr))
                fp32_forward = pool.forward
                adapter = self

                def pool_forward(_pool, features, boxes, shapes, _raw=original,
                                 _fp32=fp32_forward, _name=name):
                    profiling = bool(adapter.profile_pools)
                    device = next(iter(features.values())).device
                    if profiling:
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        start = time.perf_counter()
                    if adapter.roi_align_precision == "fp32":
                        output = _fp32(features, boxes, shapes)
                        operator_dtype = "torch.float32"
                    else:
                        if device.type != "cuda":
                            raise ValueError("Experimental FP16 ROIAlign requires CUDA")
                        with torch.autocast("cuda", enabled=False):
                            half_features = OrderedDict((k, v.to(dtype=torch.float16)) for k, v in features.items())
                            half_boxes = [b.to(dtype=torch.float16) for b in boxes]
                            output = _raw(half_features, half_boxes, shapes)
                        if output.dtype != torch.float16:
                            raise RuntimeError("FP16 ROIAlign returned an unexpected dtype; no fallback")
                        operator_dtype = "torch.float16"
                    if profiling:
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        elapsed = (time.perf_counter() - start) * 1000
                        adapter.last_stats.setdefault("roi_pools", []).append({
                            "pool": _name, "roi_count": sum(len(b) for b in boxes),
                            "input_feature_dtypes": sorted({str(v.dtype) for v in features.values()}),
                            "input_box_dtypes": sorted({str(b.dtype) for b in boxes}),
                            "operator_input_dtype": operator_dtype, "operator_autocast_enabled": False,
                            "output_dtype": str(output.dtype), "elapsed_ms": elapsed,
                            "cuda_synchronized": device.type == "cuda",
                        })
                    return output

                self._set(pool, "forward", MethodType(pool_forward, pool))
            return self
        except BaseException:
            self._restore()
            raise

    @torch.inference_mode()
    def predict_views(self, images, pixel_mask=None):
        if self.roi_align_precision == "fp32":
            # Preserve the existing feature cast, validation and computation.
            outputs = super().predict_views(images, pixel_mask)
            self.last_stats["roi_align_precision"] = "fp32"
            return outputs
        return self._predict_half_pool_views(images, pixel_mask)

    def _predict_half_pool_views(self, images, pixel_mask):
        """Isolated copy of strong_fast.py:predict_views, FP16 FPN ROI cache.

The backbone/RPN and final FP32 decode/probability/NMS are unchanged. Only
the per-view ROI feature cache changes to half; raw pools convert coordinates
inside autocast-disabled scopes. No half checkpoint or CPU fallback exists.
"""
        if not self._active:
            raise RuntimeError("Use Stage2DeploymentAdapter as a context manager")
        if (not isinstance(images, torch.Tensor) or not images.is_floating_point()
                or images.ndim != 4 or images.shape[0] not in (1, 2)
                or images.shape[1] != 3 or min(images.shape[-2:]) < 1):
            raise ValueError("Expected normalized floating RGB[B,3,H,W], B=1 or 2")
        if images.device.type != "cuda":
            raise ValueError("Experimental FP16 ROIAlign prediction requires CUDA")
        if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16:
            raise ValueError("FP16 ROIAlign prediction requires caller CUDA autocast FP16")
        if pixel_mask is not None and (tuple(pixel_mask.shape) != (images.shape[0], *images.shape[-2:])
                                       or not bool(pixel_mask.all())):
            raise ValueError("Only unpadded all-one pixel_mask[B,H,W] is supported")
        if self.model.training or self.model.roi_heads.training:
            raise RuntimeError("Model was switched to training while adapter is active")
        self.traces = []
        batch, _, height, width = images.shape
        shape = (int(height), int(width))
        padded_h, padded_w = ((height + 31) // 32 * 32, (width + 31) // 32 * 32)
        image_list = ImageList(F.pad(images, (0, padded_w - width, 0, padded_h - height)), [shape] * batch)
        self.last_stats = {"batch": batch, "roi_precision": self.roi_precision,
                           "roi_align_precision": "fp16", "pre_nms_threshold": self.pre_nms_threshold,
                           "roi_chunk_size": self.model.roi_heads.roi_chunk_size,
                           "skip_dead_amodal": self.skip_dead_amodal,
                           "exact_original_dual_mask_nms": True, "nms": []}
        features = self.model.backbone(image_list.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        if not isinstance(features, Mapping) or any(v.shape[0] != batch for v in features.values()):
            raise ValueError("Backbone feature batch does not match view batch")
        proposals, losses = self.model.rpn(image_list, features, None)
        if losses or len(proposals) != batch:
            raise RuntimeError("RPN did not return independent eval proposals for each view")
        results = []
        for index in range(batch):
            current = OrderedDict((key, value[index:index + 1].to(dtype=torch.float16))
                                  for key, value in features.items())
            detections, detector_losses = self.model.roi_heads(current, [proposals[index].float()], [shape], None)
            if detector_losses or len(detections) != 1:
                raise RuntimeError("Original ROI head did not return exactly one eval view")
            result = {key: value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value
                      for key, value in detections[0].items()}
            if result.get("image_hw") != shape:
                raise RuntimeError("ROI result coordinates differ from the input view")
            results.append(result)
        return results
