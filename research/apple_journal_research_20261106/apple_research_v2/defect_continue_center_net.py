"""Stride-4 class-specific center, offset, size and IoU network for surface defects."""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from torchvision.ops import batched_nms

from class0_patch_refinement import ROOT, sha

WEIGHTS = ROOT.parent / "apple_vision/local_amodal/cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
SIDE = 704
STRIDE = 4


class Lateral(nn.Sequential):
    def __init__(self, inputs):
        super().__init__(nn.Conv2d(inputs, 64, 1, bias=False), nn.BatchNorm2d(64))


class Mix(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
                         nn.BatchNorm2d(64), nn.SiLU(), nn.Conv2d(64, 64, 1, bias=False),
                         nn.BatchNorm2d(64), nn.SiLU())


class CenterDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        body = resnet50(weights=None)
        if pretrained:
            assert sha(WEIGHTS) == "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e"
            ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
            weights = {k.removeprefix("backbone.body."):v for k,v in ckpt.items()
                       if k.startswith("backbone.body.")}
            m = body.load_state_dict(weights, strict=False)
            assert set(m.missing_keys) == {"fc.weight", "fc.bias"} and not m.unexpected_keys
        self.stem = nn.Sequential(body.conv1, body.bn1, body.relu, body.maxpool)
        self.layer1, self.layer2, self.layer3 = body.layer1, body.layer2, body.layer3
        self.laterals = nn.ModuleList([Lateral(256), Lateral(512), Lateral(1024)])
        self.p3, self.p2 = Mix(), Mix()
        self.hm = nn.Sequential(Mix(), nn.Conv2d(64, 2, 1))
        self.reg = nn.Sequential(Mix(), nn.Conv2d(64, 8, 1))
        self.quality = nn.Sequential(Mix(), nn.Conv2d(64, 2, 1))
        nn.init.constant_(self.hm[-1].bias, -4.59)
        nn.init.normal_(self.reg[-1].weight, std=.001)
        with torch.no_grad():
            self.reg[-1].bias.copy_(torch.tensor([.5,.5,math.log(8),math.log(8)]*2))
        self.register_buffer("mean", torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def forward(self, images):
        x = (images.float().div(255) - self.mean) / self.std
        a = self.layer1(self.stem(x))
        b = self.layer2(a)
        c = self.layer3(b)
        p3 = self.p3(self.laterals[1](b) + F.interpolate(self.laterals[2](c), size=b.shape[-2:], mode="nearest"))
        p2 = self.p2(self.laterals[0](a) + F.interpolate(p3, size=a.shape[-2:], mode="nearest"))
        return self.hm(p2), self.reg(p2), self.quality(p2)


def heat_targets(labels, shape):
    batch = len(labels)
    h, w = shape
    hm = np.zeros((batch, 2, h, w), np.float32)
    assignments = {}
    for bi, boxes in enumerate(labels):
        for x1,y1,x2,y2,cls in boxes:
            c = int(cls)
            width, height = max((x2-x1)/STRIDE,.25), max((y2-y1)/STRIDE,.25)
            cx, cy = (x1+x2)/(2*STRIDE), (y1+y2)/(2*STRIDE)
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            ix, iy = int(cx), int(cy)
            sx, sy = max(.7,min(4., width*.14)), max(.7,min(4., height*.14))
            rx, ry = math.ceil(3*sx), math.ceil(3*sy)
            lx, hx = max(0, ix-rx), min(w, ix+rx+1)
            ly, hy = max(0, iy-ry), min(h, iy+ry+1)
            yy,xx=np.mgrid[ly:hy,lx:hx]
            target = np.exp(-.5*((xx-ix)**2/sx**2+(yy-iy)**2/sy**2))
            hm[bi,c,ly:hy,lx:hx] = np.maximum(target, hm[bi,c,ly:hy,lx:hx])
            for yy in range(max(0,iy-1),min(h,iy+2)):
                for xx in range(max(0,ix-1),min(w,ix+2)):
                    dist = (cx-(xx+.5))**2 + (cy-(yy+.5))**2
                    weight = 1. if (xx==ix and yy==iy) else .30
                    key = (bi,c,yy,xx)
                    old = assignments.get(key)
                    if old is None or (weight,-dist) > (old[0],-old[1]):
                        assignments[key] = (weight,dist,cx-xx,cy-yy,np.log(width),np.log(height))
    keys = np.asarray(list(assignments), np.int64).reshape(-1,4)
    vals = np.asarray(list(assignments.values()), np.float32).reshape(-1,6)
    return torch.from_numpy(hm), torch.from_numpy(keys), torch.from_numpy(vals)


def objective(outputs, target):
    logits, reg, qual = (v.float() for v in outputs)
    heat, keys, vals = (v.to(logits.device) for v in target)
    pos = heat == 1
    pred = logits.sigmoid()
    n = pos.sum().clamp_min(1)
    focal = ((F.softplus(-logits)*(1-pred).square()*pos).sum() +
             (F.softplus(logits)*pred.square()*(1-heat).pow(4)*~pos).sum())/n
    # Highest non-neighbour background activations in *the same frame* compete
    # with its labeled centers. Empty/absent classes train background directly.
    background = logits.masked_fill(heat > .025, -30).flatten(2)
    difficult, bg_index = background.topk(min(16, background.shape[-1]), dim=-1)
    hard_avg = torch.logsumexp(difficult, dim=2) - math.log(difficult.shape[-1])
    npos = pos.sum((2,3))
    pos_avg = (logits*pos).sum((2,3))/npos.clamp_min(1)
    ranking = torch.where(npos>0, F.softplus(1.5 + hard_avg - pos_avg),
                          F.softplus(hard_avg)).mean()
    qnegative = F.softplus(qual.flatten(2).gather(2,bg_index)).mean()
    if not len(keys):
        return focal + .22*ranking + .1*qnegative + reg.sum()*0
    bi, c, iy, ix = keys.unbind(1)
    raw = reg.reshape(reg.shape[0],2,4,*reg.shape[-2:])[bi,c,:,iy,ix]
    offsets, sizes = raw[:,:2], raw[:,2:].clamp(-2.5,6.5)
    weight, tgt = vals[:,0], vals[:,2:]
    loss_offset = F.smooth_l1_loss(offsets,tgt[:,:2],reduction="none",beta=.35).sum(1)
    loss_size = F.smooth_l1_loss(sizes,tgt[:,2:],reduction="none",beta=.22).sum(1)
    size, true_wh = sizes.exp(), tgt[:,2:].exp()
    box_lo, box_hi = offsets-.5*size, offsets+.5*size
    gt_lo, gt_hi = tgt[:,:2]-.5*true_wh, tgt[:,:2]+.5*true_wh
    intersect = (torch.minimum(box_hi,gt_hi)-torch.maximum(box_lo,gt_lo)).clamp_min(0).prod(1)
    union = size.prod(1)+true_wh.prod(1)-intersect
    iou = intersect/union.clamp_min(.0001)
    qual_loss = F.binary_cross_entropy_with_logits(qual[bi,c,iy,ix],iou.detach(),reduction="none")
    per_box = .75*loss_offset + .75*loss_size + 1.1*(1-iou) + .4*qual_loss
    return focal + .22*ranking + (per_box*weight).sum()/weight.sum() + .1*qnegative


@torch.inference_mode()
def decode(outputs, original_hw, threshold=.001):
    heat, reg, qual = (v.float() for v in outputs)
    heat = heat.sigmoid()
    confidence = heat * qual.sigmoid().sqrt()
    confidence *= heat.eq(F.max_pool2d(heat,3,1,1))
    _, classes, h, w = heat.shape
    result = []
    for i,(orig_h,orig_w) in enumerate(original_hw):
        scores, idx = confidence[i].flatten().topk(min(220,classes*h*w))
        mask = scores >= threshold
        scores, idx = scores[mask], idx[mask]
        cat, y, x = idx//(w*h), (idx//w)%h, idx%w
        geom = reg[i].reshape(2,4,h,w)[cat,:,y,x]
        center=torch.stack((x,y),1)+geom[:,:2]
        size=geom[:,2:].clamp(-2.5,6.5).exp()
        box=torch.cat((center-.5*size,center+.5*size),1) * geom.new_tensor([orig_w/w,orig_h/h]*2)
        box[:,::2].clamp_(0,orig_w)
        box[:,1::2].clamp_(0,orig_h)
        good=(box[:,2:]-box[:,:2]).min(1).values>0
        box, scores, cat=box[good],scores[good],cat[good]
        chosen=batched_nms(box,scores,cat,.5)
        result.append(torch.cat([box[chosen],scores[chosen,None],cat[chosen,None]],1).cpu().numpy())
    return result
