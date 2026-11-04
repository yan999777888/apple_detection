"""Experimental query-relative affine correction of a frozen amodal mask.

This module changes mask geometry rather than confidence thresholds or the
metric's box extraction. Translation is a fraction of each predicted mask's
own width/height; positive log scale enlarges the mask about its own center.
Only a small prediction-only head is trainable. Effectiveness and novelty are
unproven and require complete-validation ablations.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn
import torch.nn.functional as F

from .quality import FEATURE_DIM, quality_features


def _validate_logits(logits: torch.Tensor) -> None:
    if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point()
            or logits.ndim != 4 or min(logits.shape) < 1):
        raise ValueError("Amodal logits must be floating[B,Q,H,W] with positive dimensions")
    if logits.device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA amodal logits are supported")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Amodal logits must be finite")


@torch.no_grad()
def predicted_mask_geometry(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached bbox centers/extents in grid_sample's [-1,1] units.

    A threshold of zero is the parent's sigmoid>=.5 support. Pixel *edges*
    define the box, including its last occupied pixel. Empty masks use the
    image center and one-pixel extents, without fabricating any foreground.
    """
    _validate_logits(logits)
    height, width = logits.shape[-2:]
    support = logits.detach() >= 0
    occupied_x, occupied_y = support.any(-2), support.any(-1)
    xs = torch.arange(width, device=logits.device, dtype=torch.float32)
    ys = torch.arange(height, device=logits.device, dtype=torch.float32)
    x1 = torch.where(occupied_x, xs, float(width)).amin(-1)
    y1 = torch.where(occupied_y, ys, float(height)).amin(-1)
    x2 = torch.where(occupied_x, xs + 1, 0.).amax(-1)
    y2 = torch.where(occupied_y, ys + 1, 0.).amax(-1)
    valid = support.flatten(-2).any(-1)
    center = torch.stack(((x1 + x2) / width - 1,
                          (y1 + y2) / height - 1), -1)
    extent = torch.stack(((x2 - x1).clamp_min(1) * (2. / width),
                          (y2 - y1).clamp_min(1) * (2. / height)), -1)
    center = torch.where(valid[..., None], center, torch.zeros_like(center))
    return center, extent


def affine_warp_amodal_logits(
    logits: torch.Tensor,
    parameters: torch.Tensor,
    *,
    background_logit: float = -20.,
    query_chunk_size: int = 8,
) -> torch.Tensor:
    """Correct detached A logits using [tx,ty,log(sx),log(sy)] per query.

    ``tx/ty`` are signed fractions of the original predicted box width/height.
    The source sampling grid is the inverse of the requested object transform:
    ``g = u + (u-c)*(exp(-log_scale)-1) - t*extent*exp(-log_scale)``.
    Thus positive tx moves foreground right and positive log(sx) widens it.

    All grids use pixel centers and ``align_corners=False`` explicitly. Shifted
    logits, rather than raw logits, are zero padded so off-image samples have
    the negative ``background_logit``. The residual is computed as
    ``A + (sample(A,g) - sample(A,identity))``. At zero parameters both samples
    are bit-identical, preserving the parent's exact values despite identity
    interpolation roundoff, while retaining gradients to the transform.

    No dense original-resolution masks are constructed. Computation is float32
    on the parent's low-resolution mask grid, in bounded query chunks, and the
    result retains the input dtype. The parent and bbox geometry are detached.
    """
    _validate_logits(logits)
    if (not isinstance(parameters, torch.Tensor) or not parameters.is_floating_point()
            or tuple(parameters.shape) != (*logits.shape[:2], 4)
            or parameters.device != logits.device):
        raise ValueError("Affine parameters must be floating[B,Q,4] on the logits device")
    if not bool(torch.isfinite(parameters).all()):
        raise ValueError("Affine parameters must be finite")
    if (isinstance(background_logit, bool) or not isinstance(background_logit, (int, float))
            or not math.isfinite(background_logit) or background_logit >= 0):
        raise ValueError("background_logit must be a finite negative number")
    if isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, int) or query_chunk_size < 1:
        raise ValueError("query_chunk_size must be a positive integer")
    batch, queries, height, width = logits.shape
    centers, extents = predicted_mask_geometry(logits)
    original = logits.detach()
    with torch.autocast(device_type=logits.device.type, enabled=False):
        values = original.float().reshape(batch * queries, 1, height, width)
        transforms = parameters.float().reshape(batch * queries, 4)
        centers, extents = centers.reshape(-1, 2), extents.reshape(-1, 2)
        ys = (torch.arange(height, device=logits.device, dtype=torch.float32) + .5) * (2. / height) - 1
        xs = (torch.arange(width, device=logits.device, dtype=torch.float32) + .5) * (2. / width) - 1
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        identity = torch.stack((xx, yy), -1)
        corrected = []
        for start in range(0, batch * queries, query_chunk_size):
            stop = min(start + query_chunk_size, batch * queries)
            base_grid = identity[None].expand(stop - start, -1, -1, -1)
            inverse_scale = (-transforms[start:stop, 2:]).exp()
            translation = transforms[start:stop, :2] * extents[start:stop]
            grid = (base_grid
                    + (base_grid - centers[start:stop, None, None])
                    * (inverse_scale[:, None, None] - 1)
                    - (translation * inverse_scale)[:, None, None])
            source = values[start:stop] - background_logit
            warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros",
                                   align_corners=False) + background_logit
            identity_sampled = F.grid_sample(source, base_grid, mode="bilinear", padding_mode="zeros",
                                             align_corners=False) + background_logit
            # Parenthesize the cancellation: (A + warped) - identity can round.
            residual = (warped - identity_sampled).to(dtype=original.dtype)
            base = original.reshape(batch * queries, 1, height, width)[start:stop]
            corrected.append(base + residual)
        return torch.cat(corrected, 0).reshape_as(original)


class QueryConditionedAmodalWarp(nn.Module):
    """Zero-identity affine head using detached 786-dimensional V/A evidence."""

    def __init__(
        self, *, width: int = 128, max_translation: float = .5,
        max_log_scale: float = math.log(2), background_logit: float = -20.,
        query_chunk_size: int = 8,
    ) -> None:
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        for name, value in (("max_translation", max_translation), ("max_log_scale", max_log_scale)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive number")
        # Avoid overflow in the float32 inverse scale, even for custom settings.
        if max_log_scale > math.log(16):
            raise ValueError("max_log_scale must not exceed log(16)")
        if (isinstance(background_logit, bool) or not isinstance(background_logit, (int, float))
                or not math.isfinite(background_logit) or background_logit >= 0):
            raise ValueError("background_logit must be a finite negative number")
        if isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, int) or query_chunk_size < 1:
            raise ValueError("query_chunk_size must be a positive integer")
        self.configuration = dict(width=width, max_translation=float(max_translation),
                                  max_log_scale=float(max_log_scale), background_logit=float(background_logit),
                                  query_chunk_size=query_chunk_size)
        # Optional modules must not change the parent's initialization stream.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            self.encoder = nn.Sequential(nn.Linear(FEATURE_DIM, width), nn.LayerNorm(width), nn.GELU())
            self.readout = nn.Linear(width, 4)
            nn.init.zeros_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)
        self.register_buffer("parameter_bounds", torch.tensor(
            [max_translation, max_translation, max_log_scale, max_log_scale], dtype=torch.float32))

    def predict_parameters(self, features: torch.Tensor) -> torch.Tensor:
        if (not isinstance(features, torch.Tensor) or not features.is_floating_point()
                or features.ndim != 3 or features.shape[-1] != FEATURE_DIM
                or min(features.shape[:2]) < 1 or features.device != self.readout.weight.device):
            raise ValueError("Prediction-only features must be floating[B,Q,786] on the head device")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("Prediction-only features must be finite")
        features = features.detach().to(dtype=self.readout.weight.dtype)
        return self.readout(self.encoder(features)).tanh() * self.parameter_bounds

    def forward(self, features: torch.Tensor, amodal_logits: torch.Tensor, *, targets=None) -> torch.Tensor:
        if targets is not None:
            raise ValueError("GT never enters amodal warp inference")
        return affine_warp_amodal_logits(amodal_logits, self.predict_parameters(features),
                                         background_logit=self.configuration["background_logit"],
                                         query_chunk_size=self.configuration["query_chunk_size"])

    def refine_outputs(self, outputs: dict, features: torch.Tensor, *, targets=None) -> dict:
        """Return a new dict; V/class/extra output tensors retain their identity."""
        if targets is not None:
            raise ValueError("GT never enters amodal warp inference")
        if not isinstance(outputs, dict) or "amodal_masks_queries_logits" not in outputs:
            raise ValueError("Prediction dict must contain amodal_masks_queries_logits")
        return {**outputs, "amodal_masks_queries_logits": self(features, outputs["amodal_masks_queries_logits"])}


class FrozenAmodalWarpModel(nn.Module):
    """One frozen DualMaskModel forward plus the trainable A-only warp head."""

    def __init__(self, parent: nn.Module, **configuration) -> None:
        super().__init__()
        if not isinstance(parent, nn.Module) or not hasattr(parent, "shared"):
            raise ValueError("A DualMaskModel-compatible parent with a shared backbone is required")
        self.parent = parent.requires_grad_(False).eval()
        self.amodal_warp = QueryConditionedAmodalWarp(**configuration)
        self.checkpoint_provenance = {
            **deepcopy(getattr(parent, "checkpoint_provenance", None) or {}),
            "experimental_model": "FrozenQueryRelativeAmodalAffineWarp",
            "amodal_warp": deepcopy(self.amodal_warp.configuration),
            "parent_frozen": True, "visible_readout_unchanged": True,
            "class_readout_unchanged": True, "direct_box_head": False,
            "geometry_source": "detached predicted A mask support; no GT in inference",
            "publication_novelty_proven": False,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.parent.eval()
        return self

    def forward(self, pixel_values, pixel_mask=None, *, targets=None):
        if targets is not None:
            raise ValueError("GT never enters amodal warp inference")
        captured = {}

        def capture(module, inputs, outputs):
            captured["queries"] = outputs.transformer_decoder_intermediate_states[-1].transpose(0, 1)
            captured["pixels"] = outputs.pixel_decoder_last_hidden_state

        hook = self.parent.shared.register_forward_hook(capture)
        try:
            with torch.no_grad():
                original = self.parent(pixel_values, pixel_mask=pixel_mask)
                features = quality_features(captured["queries"], captured["pixels"],
                                            original["visible_masks_queries_logits"],
                                            original["amodal_masks_queries_logits"],
                                            original["class_queries_logits"])
        finally:
            hook.remove()
        return self.amodal_warp.refine_outputs(original, features)
