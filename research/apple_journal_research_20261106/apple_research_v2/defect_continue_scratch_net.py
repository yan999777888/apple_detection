"""Frozen insect detector and a separate scratch head sharing one feature extraction."""

from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou
from ultralytics.utils.tal import make_anchors
from class0_patch_refinement import ROOT, read, sha
from defect_continue_rank_net import flatten_quality
from defect_continue_warm_rank import initialized, unpack
import defect9485_small as original

WARM = ROOT / "experiments/defect_continue_warm_rank/weights.pt"
REPLAY = ROOT / "experiments/defect_continue_public_v1/replay/weights.pt"
WARM_SHA = "5e07009ec94310bd01bfd6c257ff8df853d3eef36c25de15e7f8483706fabd93"
REPLAY_SHA = "27fcf984acc450e6d09dc4f0e0a2810ed4daff0911668576b3b75e44feece0cc"


class LockedInsect(nn.Module):
    def __init__(self, device):
        super().__init__()
        if (
            sha(WARM) != WARM_SHA
            or read(WARM.parent / "receipt.json")["weight_sha"] != WARM_SHA
        ):
            raise ValueError("Unexpected frozen insect checkpoint")
        self.warm = initialized().eval()
        self.warm.load_state_dict(
            torch.load(WARM, map_location="cpu", weights_only=True), strict=True
        )
        self.warm = self.warm.to(device)
        self.warm.base.fuse(verbose=False)
        self.warm.requires_grad_(False)

    @torch.no_grad()
    def features(self, x):
        saved = []
        fine = None
        for i, layer in enumerate(self.warm.base.model[:-1]):
            if layer.f != -1:
                x = (
                    saved[layer.f]
                    if isinstance(layer.f, int)
                    else [x if j == -1 else saved[j] for j in layer.f]
                )
            x = layer(x)
            if i == 2:
                fine = x
            saved.append(x if layer.i in self.warm.base.save else None)
        return fine, [saved[i] for i in self.warm.head.f]

    @torch.no_grad()
    def detections(self, levels):
        q = flatten_quality(
            [b(f) for b, f in zip(self.warm.quality, levels, strict=True)]
        )
        raw = self.warm.head.forward_head(levels, **self.warm.head.one2one)
        return self.warm.export_raw(raw, q)


class ScratchHead(nn.Module):
    def __init__(self):
        super().__init__()
        if (
            sha(REPLAY) != REPLAY_SHA
            or read(REPLAY.parent / "receipt.json")["weight_sha"] != REPLAY_SHA
        ):
            raise ValueError("Unexpected high-recall replay checkpoint")
        model = original.architecture().eval()
        model.load_state_dict(
            torch.load(REPLAY, map_location="cpu", weights_only=True), strict=True
        )
        self.head = deepcopy(model.model[-1])
        self.head.nc = 1
        self.head.no = 4 * self.head.reg_max + 1
        for towers in (self.head.cv3, self.head.one2one_cv3):
            for tower in towers:
                source = tower[-1]
                dest = nn.Conv2d(source.in_channels, 1, 1)
                with torch.no_grad():
                    dest.weight.copy_(source.weight[:1])
                    dest.bias.copy_(source.bias[:1])
                tower[-1] = dest
        self.fine = nn.Sequential(
            nn.Conv2d(128, 24, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1, groups=24, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.Conv2d(24, 128, 1, bias=False),
        )
        nn.init.zeros_(self.fine[-1].weight)

    def forward(self, fine, pyramid):
        levels = [pyramid[0] + self.fine(fine), *pyramid[1:]]
        if self.training:
            # The whole backbone is frozen. Both native losses can train this specialist's new lateral branch.
            return dict(
                one2one=self.head.forward_head(levels, **self.head.one2one),
                one2many=self.head.forward_head(levels, **self.head.one2many),
            )
        raw = self.head.forward_head(levels, **self.head.one2one)
        decoded = torch.cat(
            [self.head._get_decode_boxes(raw), raw["scores"].sigmoid()], dim=1
        )
        return self.head.postprocess(decoded.transpose(1, 2))


def scratch_only(target):
    use = target["cls"].reshape(-1) == 0
    return {k: v[use] for k, v in target.items()}


def support(logits, boxes, annotations):
    positive, ranking = [], []
    for i, gt in enumerate(annotations):
        scratches = gt[gt[:, 4] == 0, :4]
        if not len(scratches):
            continue
        insects = gt[gt[:, 4] == 1, :4]
        with torch.no_grad():
            overlap = box_iou(boxes[i].detach().float(), scratches).nan_to_num()
            iou, owner = overlap.max(1)
            z = logits[i].detach()
            on_insect = (
                box_iou(boxes[i].detach().float(), insects).nan_to_num().max(1).values
                >= 0.4
                if len(insects)
                else torch.zeros_like(iou, dtype=torch.bool)
            )
            centers = (boxes[i, :, :2] + boxes[i, :, 2:]).detach() * 0.5
        for gi, ref in enumerate(scratches):
            with torch.no_grad():
                ids = torch.where((iou >= 0.5) & (owner == gi))[0]
                if not len(ids):
                    continue
                # Choose one already localized candidate per real scratch. Lower-ranked overlapping
                # anchors never become additional positives, preserving the one-to-one interpretation.
                pos = ids[(z[ids] + 0.5 * iou[ids]).argmax()]
                extent = (ref[2:] - ref[:2]).max().clamp(32, 112)
                nearby = (centers - (ref[:2] + ref[2:]) * 0.5).abs().amax(1) <= (
                    1.8 * extent
                )
                invalid = (iou < 0.10) & (z > -4.0)
                hard = torch.where(invalid & (nearby | on_insect))[0]
                if len(hard):
                    hard = hard[z[hard].topk(min(3, len(hard))).indices]
            s = logits[i, pos]
            positive.append(
                F.softplus(0.4055 - s)
            )  # protect a good single candidate through the fixed .40 gate
            if len(hard):
                ranking.append(F.softplus(0.65 + logits[i, hard] - s).mean())
    zero = logits.sum() * 0
    return (
        torch.stack(positive).mean() if positive else zero,
        torch.stack(ranking).mean() if ranking else zero,
        len(positive),
        len(ranking),
    )


def objective(spec, criterion, raw, target, hw, n):
    fp = {
        a: {
            k: [t.float() for t in v] if isinstance(v, list) else v.float()
            for k, v in branch.items()
        }
        for a, branch in raw.items()
    }
    native = criterion(fp, scratch_only(target))[0].sum() / n
    one = fp["one2one"]
    anchors, strides = make_anchors(one["feats"], spec.head.stride, 0.5)
    boxes = (
        spec.head.decode_bboxes(spec.head.dfl(one["boxes"]), anchors.T[None]).transpose(
            1, 2
        )
        * strides
    )
    pos, rank, npos, npairs = support(one["scores"][:, 0], boxes, unpack(target, hw, n))
    total = native + 0.30 * pos + 0.12 * rank
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite scratch objective")
    return (
        total,
        torch.stack([native.detach(), pos.detach(), rank.detach()]),
        npos,
        npairs,
    )


class Deployed(nn.Module):
    def __init__(self, locked, scratch):
        super().__init__()
        self.locked = locked
        self.scratch = scratch

    def forward(self, x):
        fine, levels = self.locked.features(x)
        return self.locked.detections(levels), self.scratch(fine, levels)


def boxes(model, im):
    rgb, (dx, dy, sx, sy) = original.square(im)
    x = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float() / 255
    bug_full, scratch = model(x)
    rows = []
    for raw, cls in ((bug_full, 1), (scratch, 0)):
        p = raw[0]
        p = p[p[:, 4] >= 0.001].cpu().numpy()
        h, w = im.shape[:2]
        p[:, [0, 2]] = ((p[:, [0, 2]] - dx) / sx).clip(0, w)
        p[:, [1, 3]] = ((p[:, [1, 3]] - dy) / sy).clip(0, h)
        # The original top-300 and both-class NMS run before taking the locked insect rows.
        p = original.nms(p)
        rows.append(p[p[:, 5] == cls])
    result = np.concatenate(rows)
    return result[np.argsort(-result[:, 4], kind="stable")]
