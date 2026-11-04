"""Stateless pre-maxpool stride2 detail injection into the existing FPN P2."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import nn


def _feature(value, channels, name):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim != 4 or value.shape[1] != channels
            or min(value.shape[0], *value.shape[-2:]) < 1
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be finite floating [B,{channels},H,W]")


class _ExactResidualAdd(torch.autograd.Function):
    """Preserve base signed-zero bytes while keeping addition's derivative."""

    @staticmethod
    def forward(ctx, base, residual):
        return torch.where(residual == 0, base, base + residual)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient


class ShallowDetailFusion(nn.Module):
    """Learn stride2-to-stride4 detail; zero readout initially preserves P2.

    The encoder is initialized normally; only the final readout is zero.
    Its readout can learn immediately, then expose encoder/stem gradients.
    No normalization state, GT, interpolation, persistent features or hooks.
    """

    def __init__(self):
        super().__init__()
        self.downsample = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=True)
        self.activation = nn.ReLU()
        self.readout = nn.Conv2d(64, 256, 1, bias=True)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, stem, p2):
        _feature(stem, 64, "pre-maxpool stride2 stem")
        _feature(p2, 256, "P2")
        if stem.device != p2.device or stem.shape[0] != p2.shape[0]:
            raise ValueError("Stem and P2 must share device and batch")
        expected = tuple((side + 1) // 2 for side in stem.shape[-2:])
        if tuple(p2.shape[-2:]) != expected:
            raise ValueError("Native stem stride2-to-P2 stride4 shapes must match exactly")
        residual = self.readout(self.activation(self.downsample(stem)))
        # FrozenBN stem and autocast FPN can differ in dtype. Preserve P2 dtype.
        residual = residual.to(dtype=p2.dtype)
        if residual.shape != p2.shape or not bool(torch.isfinite(residual).all()):
            raise ValueError("Detail residual must be finite and exactly match P2")
        result = _ExactResidualAdd.apply(p2, residual)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Fused P2 must remain finite")
        return result


def forward_backbone_with_detail(backbone, images, detail_fusion):
    """Execute original R50 body/FPN once, without changing module ownership.

    This follows IntermediateLayerGetter.forward's existing ordered traversal.
    The pre-maxpool relu result lives only in this call. Original backbone
    state keys stay untouched, and only the returned P2 tensor is replaced.
    """
    _feature(images, 3, "images")
    body, fpn = getattr(backbone, "body", None), getattr(backbone, "fpn", None)
    names = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
    returns = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    if (not isinstance(body, nn.ModuleDict) or tuple(body.keys()) != names
            or getattr(body, "return_layers", None) != returns or not isinstance(fpn, nn.Module)
            or not isinstance(detail_fusion, ShallowDetailFusion)):
        raise ValueError("Detail path requires the native ordered R50 body/FPN layout")
    if (not isinstance(body["conv1"], nn.Conv2d) or body["conv1"].in_channels != 3
            or body["conv1"].out_channels != 64 or body["conv1"].stride != (2, 2)
            or not isinstance(body["maxpool"], nn.MaxPool2d) or body["maxpool"].stride != 2):
        raise ValueError("Detail path requires the native stride2 stem and stride2 maxpool")
    height, width = images.shape[-2:]
    features, stem, current = OrderedDict(), None, images
    for name, layer in body.items():
        current = layer(current)
        if name == "relu":
            stem = current
        if name in returns:
            features[returns[name]] = current
    _feature(stem, 64, "pre-maxpool stride2 stem")
    if tuple(stem.shape[-2:]) != ((height + 1) // 2, (width + 1) // 2):
        raise ValueError("Captured native stem must have stride2 spatial dimensions")
    for index, channels in enumerate((256, 512, 1024, 2048)):
        value = features[str(index)]
        _feature(value, channels, "native body C" + str(index + 2))
        stride = 4 * 2 ** index
        if value.shape[0] != images.shape[0] or tuple(value.shape[-2:]) != ((height + stride - 1) // stride, (width + stride - 1) // stride):
            raise ValueError("Native body feature scales are incompatible with detail fusion")
    outputs = fpn(features)
    if not isinstance(outputs, Mapping) or tuple(outputs) != ("0", "1", "2", "3", "pool"):
        raise ValueError("Detail path requires original P2-P6 FPN outputs")
    for index, value in enumerate(outputs.values()):
        _feature(value, 256, "native FPN output")
        stride = 4 * 2 ** index
        if value.shape[0] != images.shape[0] or tuple(value.shape[-2:]) != ((height + stride - 1) // stride, (width + stride - 1) // stride):
            raise ValueError("Native FPN feature scales are incompatible with detail fusion")
    result = OrderedDict(outputs)
    result["0"] = detail_fusion(stem, outputs["0"])
    return result
