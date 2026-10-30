"""Occlusion evidence and query competition for IoU-aware apple ranking.

Quality-aware scoring is established prior work (Mask Scoring R-CNN and GFL).
This experimental V/A adaptation requires measured ablation, not novelty claims.
"""
from copy import deepcopy
import math

import torch
from torch import nn
import torch.nn.functional as F

FEATURE_DIM = 786
QUALITY_BRANCHES = ("amodal_box", "visible_mask", "amodal_mask")


def _mask_statistics(logits):
    probability = logits.float().sigmoid().clamp(1e-6, 1-1e-6)
    binary = logits >= 0
    height, width = logits.shape[-2:]
    x = torch.arange(width, device=logits.device).float() / width
    y = torch.arange(height, device=logits.device).float() / height
    occupied_x, occupied_y = binary.any(-2), binary.any(-1)
    x1 = torch.where(occupied_x, x, 1.).amin(-1)
    y1 = torch.where(occupied_y, y, 1.).amin(-1)
    x2 = torch.where(occupied_x, x + 1/width, 0.).amax(-1)
    y2 = torch.where(occupied_y, y + 1/height, 0.).amax(-1)
    valid = binary.flatten(-2).any(-1)
    entropy = -(probability * probability.log() + (1-probability) * (1-probability).log()).mean((-1, -2))
    return torch.stack((probability.mean((-1,-2)), binary.float().mean((-1,-2)), entropy,
                        x1*valid, y1*valid, x2, y2), -1)


@torch.no_grad()
def quality_features(queries, pixels, visible, amodal, classes):
    """Prediction-only features; neither labels nor original targets enter here."""
    visible_p, amodal_p = visible.float().sigmoid(), amodal.float().sigmoid()
    hidden_p = amodal_p * (1-visible_p)
    pools = []
    for weights in (visible_p, hidden_p):
        pools.append(torch.einsum("bqhw,bchw->bqc", weights, pixels.float()) /
                     weights.sum((-1,-2)).clamp_min(1e-6).unsqueeze(-1))
    vstats, astats = _mask_statistics(visible), _mask_statistics(amodal)
    relation = torch.stack(((vstats[...,1] / astats[...,1].clamp_min(1e-6)).clamp_max(4),
                            hidden_p.mean((-1,-2)), (visible_p*(1-amodal_p)).mean((-1,-2)),
                            classes.float().softmax(-1)[...,0]), -1)
    features = torch.cat((queries.float(), *pools, vstats, astats, relation), -1)
    if features.shape[-1] != FEATURE_DIM or not bool(torch.isfinite(features).all()):
        raise ValueError("Invalid prediction-only quality features")
    return features


class DualEvidenceQualityHead(nn.Module):
    def __init__(self, *, competition=True, evidence=True, width=128):
        super().__init__()
        self.configuration = dict(competition=competition, evidence=evidence, width=width)
        self.encoder = nn.Sequential(nn.Linear(FEATURE_DIM, width), nn.LayerNorm(width), nn.GELU())
        self.competition = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, width*2, dropout=0., batch_first=True, norm_first=True),
            2, enable_nested_tensor=False) if competition else nn.Identity()
        self.readout = nn.Linear(width, 3)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, features):
        if not self.configuration["evidence"]:
            # Matched ablation: same input dimensionality and optimization;
            # remove mask/occlusion evidence, retain query and parent score.
            features = torch.cat((features[...,:256], torch.zeros_like(features[...,256:-1]), features[...,-1:]), -1)
        return self.readout(self.competition(self.encoder(features)))


class FrozenQualityModel(nn.Module):
    def __init__(self, parent, *, competition=True, evidence=True):
        super().__init__()
        self.parent = parent.requires_grad_(False).eval()
        self.quality_head = DualEvidenceQualityHead(competition=competition, evidence=evidence)
        self.checkpoint_provenance = {
            **deepcopy(parent.checkpoint_provenance),
            "experimental_model": "FrozenDualEvidenceQueryQuality",
            "quality_head": self.quality_head.configuration,
            "final_score": "sigmoid(parent apple/null log-odds + learned amodal-box quality residual)",
            "quality_supervision": "one-to-one IoU targets on TRAIN only; unknown A excluded",
            "parent_frozen": True, "publication_novelty_proven": False,
        }

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def extract(self, pixel_values, pixel_mask=None):
        captured = {}
        def capture(module, inputs, outputs):
            captured["queries"] = outputs.transformer_decoder_intermediate_states[-1].transpose(0,1)
            captured["pixels"] = outputs.pixel_decoder_last_hidden_state
        hook = self.parent.shared.register_forward_hook(capture)
        try:
            with torch.no_grad():
                original = self.parent(pixel_values, pixel_mask=pixel_mask)
                features = quality_features(captured["queries"], captured["pixels"],
                                            original["visible_masks_queries_logits"],
                                            original["amodal_masks_queries_logits"], original["class_queries_logits"])
        finally:
            hook.remove()
        return original, features

    def forward(self, pixel_values, pixel_mask=None, *, targets=None):
        if targets is not None:
            raise ValueError("GT never enters inference")
        original, features = self.extract(pixel_values, pixel_mask)
        residual = self.quality_head(features)
        parent_classes = original["class_queries_logits"].float()
        classes = parent_classes + torch.stack((residual[...,0], torch.zeros_like(residual[...,0])), -1)
        return {**original, "class_queries_logits": classes,
                "quality_logits": parent_classes[...,0,None]-parent_classes[...,1,None]+residual,
                "parent_class_queries_logits": parent_classes, "quality_features": features}


def _box(mask):
    xs, ys = torch.nonzero(mask.any(0)).flatten(), torch.nonzero(mask.any(1)).flatten()
    if not len(xs) or not len(ys):
        return (0.,0.,0.,0.)
    return (float(xs[0]),float(ys[0]),float(xs[-1])+1,float(ys[-1])+1)


def _box_iou(left, right):
    intersection = max(0.,min(left[2],right[2])-max(left[0],right[0])) * max(0.,min(left[3],right[3])-max(left[1],right[1]))
    union = (left[2]-left[0])*(left[3]-left[1])+(right[2]-right[0])*(right[3]-right[1])-intersection
    return intersection/union if union else 0.


@torch.no_grad()
def quality_targets(original, target, *, criterion, generator=None):
    """Exact original-resolution quality of the shared Hungarian assignment.

    All 100 queries remain; unmatched predictions have quality zero. The three
    columns are A-box / V-mask / A-mask. Minne has only the V-mask column.
    """
    from .evaluate import POSTPROCESS_PATH, _module
    classes, visible, amodal, _, count, known_amodal = criterion._validate(original, [target])
    branches = [(visible,"visible_masks",.5 if known_amodal else 1.)]
    if known_amodal:
        branches.append((amodal,"amodal_masks",.5))
    queries, targets = criterion._match(classes, branches, target, count, generator)
    result = torch.zeros((1,100,3), dtype=torch.float32)
    valid = torch.tensor([known_amodal,True,known_amodal], dtype=torch.bool)
    post = _module(POSTPROCESS_PATH, "_apple_quality_target_postprocess")
    qlist, tlist = queries.cpu().tolist(), targets.cpu().tolist()
    for prediction, key, column in ((visible,"visible_masks",1), (amodal,"amodal_masks",2)):
        if key == "amodal_masks" and not known_amodal:
            continue
        for start in range(0,len(qlist),8):
            qs, ts = qlist[start:start+8], tlist[start:start+8]
            masks = post._binary_chunk(prediction, qs, tuple(target["original_hw"]), torch, F.interpolate)
            for local,(query,truth_index) in enumerate(zip(qs,ts,strict=True)):
                predicted = masks[local]
                truth = target[key][truth_index].bool()
                union = torch.logical_or(predicted,truth).sum().item()
                intersection = torch.logical_and(predicted,truth).sum().item()
                result[0,query,column] = intersection/union if union else 0.
                if known_amodal and key == "amodal_masks":
                    result[0,query,0] = _box_iou(_box(predicted),_box(truth))
            del masks
    return {"quality_targets": result, "quality_valid": valid, "GT_count": count,
            "query_indices": qlist, "target_indices": tlist}


def quality_focal_loss(logits, targets, valid, counts):
    """Quality Focal Loss: BCE to continuous IoU, weighted by squared error.

    See GFL https://arxiv.org/abs/2006.04388. Losses for unknown branches have
    no graph edge. Normalize each image by its real GT count, retaining empties.
    """
    losses = {}
    for index,name in enumerate(QUALITY_BRANCHES):
        selected = valid[:,index]
        if bool(selected.any()):
            prediction, truth = logits[selected,:,index], targets[selected,:,index]
            weight = (prediction.sigmoid()-truth).abs().square()
            error = F.binary_cross_entropy_with_logits(prediction,truth,reduction="none")*weight
            losses[name] = (error.sum(-1)/counts[selected].clamp_min(1)).mean()
    return losses
