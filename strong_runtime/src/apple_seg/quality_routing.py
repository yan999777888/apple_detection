"""Prediction-only quality head with explicit annotation-aware gradient routing.

MinneApple V labels train only the V readout. PApple V/A labels train the shared
trunk and their own readouts. Use optimizer.zero_grad(set_to_none=True) before
each step: unused trunk/A parameters then retain grad=None, so AdamW skips
both their momentum update and decoupled weight decay on a Minne-only step.
"""
import torch
from torch import nn
import torch.nn.functional as F

from .quality import FEATURE_DIM, DualEvidenceQualityHead


class RoutedDualEvidenceQualityHead(nn.Module):
    """Same encoder/competition design and parameter count, separate readouts.

    forward(features) is GT-free and returns residuals[B,100,3] in the original
    A-box/V-mask/A-mask order. The default head has 366339 trainable parameters.
    Annotation-aware routing belongs only in routed_quality_losses.
    """
    def __init__(self, *, competition=True, evidence=True, width=128):
        super().__init__()
        original = DualEvidenceQualityHead(competition=competition, evidence=evidence, width=width)
        self.configuration = dict(original.configuration)
        self.encoder = original.encoder
        self.competition = original.competition
        self.amodal_box_readout = nn.Linear(width, 1)
        self.visible_mask_readout = nn.Linear(width, 1)
        self.amodal_mask_readout = nn.Linear(width, 1)
        for readout in (self.amodal_box_readout, self.visible_mask_readout, self.amodal_mask_readout):
            nn.init.zeros_(readout.weight)
            nn.init.zeros_(readout.bias)

    def encode(self, features):
        if (not isinstance(features, torch.Tensor) or not features.is_floating_point()
                or features.ndim != 3 or features.shape[0] < 1
                or tuple(features.shape[1:]) != (100, FEATURE_DIM)):
            raise ValueError("Expected floating prediction features[B,100,786]")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("Prediction features must be finite")
        if not self.configuration["evidence"]:
            features = torch.cat((features[..., :256], torch.zeros_like(features[..., 256:-1]),
                                  features[..., -1:]), -1)
        return self.competition(self.encoder(features))

    def forward(self, features):
        hidden = self.encode(features)
        return torch.cat((self.amodal_box_readout(hidden), self.visible_mask_readout(hidden),
                          self.amodal_mask_readout(hidden)), -1)


def _quality_branch_loss(logits, truth, counts):
    # Validate only the real supervised column; invalid unknown-A values are
    # deliberately never inspected or substituted with fabricated labels.
    if (not bool(torch.isfinite(truth).all()) or bool(((truth < 0) | (truth > 1)).any())):
        raise ValueError("Known quality targets must be finite in [0,1]")
    truth = truth.detach()
    weight = (logits.sigmoid() - truth).abs().square()
    error = F.binary_cross_entropy_with_logits(logits, truth, reduction="none") * weight
    return (error.sum(-1) / counts.clamp_min(1)).mean()


def routed_quality_losses(head, features, parent_margin, targets, valid, counts):
    """Return unweighted per-source QFL scalars, omitting absent source keys.

    Arguments use the cached-training shapes: features[B,100,786], frozen
    parent_margin[B,100], targets[B,100,3], bool valid[B,3], integer counts[B].
    All tensors share a device. Validity must be genuine PApple [T,T,T] or
    MinneApple [F,T,F]. Unknown A columns are not read by this loss function.

    The caller's balanced-source objective is:
      papple_amodal_box + .5*papple_amodal_mask
      + .25*papple_visible_mask + .25*minne_visible_mask
    Include only keys present in the returned dictionary. Parent margins are
    detached, and a Minne-only call invokes neither A readout nor A loss.
    """
    if not isinstance(head, RoutedDualEvidenceQualityHead):
        raise TypeError("Expected RoutedDualEvidenceQualityHead")
    if (not isinstance(features, torch.Tensor) or features.ndim != 3
            or tuple(features.shape[1:]) != (100, FEATURE_DIM) or features.shape[0] < 1):
        raise ValueError("Expected prediction features[B,100,786]")
    batch = features.shape[0]
    for name, value, shape in (("parent_margin", parent_margin, (batch, 100)),
                               ("targets", targets, (batch, 100, 3)),
                               ("valid", valid, (batch, 3)), ("counts", counts, (batch,))):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Expected {name}{shape}")
        if value.device != features.device:
            raise ValueError("All quality training tensors must share a device")
    if not parent_margin.is_floating_point() or not targets.is_floating_point():
        raise TypeError("Parent margins and quality targets must be floating tensors")
    if not bool(torch.isfinite(parent_margin).all()):
        raise ValueError("Parent margins must be finite")
    if valid.dtype != torch.bool:
        raise TypeError("Annotation validity must be bool[B,3]")
    if not bool(valid[:, 1].all()) or not torch.equal(valid[:, 0], valid[:, 2]):
        raise ValueError("Validity must be PApple [T,T,T] or MinneApple [F,T,F]")
    if counts.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        raise TypeError("GT counts must be integer tensors")
    if bool(((counts < 0) | (counts > 100)).any()):
        raise ValueError("Every original GT count must be in [0,100]")

    known = valid[:, 0]
    unknown = ~known
    losses = {}
    if bool(known.any()):
        hidden = head.encode(features[known])
        margin, real_counts = parent_margin[known].detach(), counts[known]
        for key, readout, column in (("papple_amodal_box", head.amodal_box_readout, 0),
                                     ("papple_visible_mask", head.visible_mask_readout, 1),
                                     ("papple_amodal_mask", head.amodal_mask_readout, 2)):
            logits = margin + readout(hidden).squeeze(-1)
            losses[key] = _quality_branch_loss(logits, targets[known, :, column], real_counts)
    if bool(unknown.any()):
        # Do not call forward() and select its V column: concatenating unused A
        # outputs would allocate zero A grads and let AdamW decay A parameters.
        with torch.no_grad():
            hidden = head.encode(features[unknown])
        logits = parent_margin[unknown].detach() + head.visible_mask_readout(hidden).squeeze(-1)
        losses["minne_visible_mask"] = _quality_branch_loss(logits, targets[unknown, :, 1], counts[unknown])
    return losses
