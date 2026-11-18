"""Single-pass YOLO26s with dense, GT-supervised localization and ranking quality."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou
from ultralytics.utils.tal import make_anchors
from defect9485_small import architecture


def flatten_quality(pyramid):
    """P3/P4/P5 row-major anchors, exactly the order used by Detect.forward_head."""
    return torch.cat([level.flatten(2) for level in pyramid],dim=2)


def fused_logits(logits, quality):
    # A common additive shift to all anchors carries no ranking information.
    # First subtract a common reference to ensure identical qualities give EXACT zero.
    delta=quality.float()-quality[:,:1,:1].float()
    relative=delta-delta.mean(dim=(1,2),keepdim=True)
    return logits.float()+3*torch.tanh(relative/3)


class QualityDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=architecture()  # Verified generic 80-class COCO checkpoint only.
        channels=[level[0].conv.in_channels for level in self.head.cv2]
        assert len(channels)==3 and self.head.end2end and self.head.nc==2
        self.quality=nn.ModuleList([
            nn.Sequential(nn.Conv2d(c,16,1,bias=False),nn.BatchNorm2d(16),nn.SiLU(),
                          nn.Conv2d(16,16,3,padding=1,groups=16,bias=False),nn.BatchNorm2d(16),nn.SiLU(),
                          nn.Conv2d(16,2,1))
            for c in channels])
        for branch in self.quality:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    @property
    def head(self):
        return self.base.model[-1]

    def features(self,x):
        cache=[]
        for layer in self.base.model[:-1]:
            if layer.f!=-1:
                x=cache[layer.f] if isinstance(layer.f,int) else [x if j==-1 else cache[j] for j in layer.f]
            x=layer(x)
            cache.append(x if layer.i in self.base.save else None)
        return [cache[i] for i in self.head.f]

    def forward(self,x):
        levels=self.features(x)
        q=flatten_quality([branch(f) for branch,f in zip(self.quality,levels,strict=True)])
        if self.training:
            return self.head(levels),q
        raw=self.head.forward_head(levels,**self.head.one2one)
        return self.export_raw(raw,q)

    def export_raw(self,raw,quality):
        # Fuse quality before the native top-k. This preserves anchor IDs across classes.
        boxes=self.head._get_decode_boxes(raw).float()
        p=torch.cat([boxes,fused_logits(raw["scores"],quality).sigmoid()],dim=1)
        return self.head.postprocess(p.transpose(1,2))


@torch.no_grad()
def assign(boxes,logits,annotations):
    """Group positive/hard-negative anchors per image and class, and boundary regression anchors per GT."""
    records=[]
    for image_id,gt in enumerate(annotations):
        gt=gt.to(device=boxes.device)
        gxy=gt[:,:4]
        iou=box_iou(boxes[image_id].detach().float(),gxy.float()).nan_to_num(0.).clamp_(0,1)
        classes=gt[:,4].long()
        best_any=iou.argmax(1) if len(gt) else None
        for cls in range(logits.shape[1]):
            scores=logits[image_id,cls].detach().float()
            ids=torch.where(classes==cls)[0]
            if len(ids):
                best_same,owner=iou[:,ids].max(1)
                owner=ids[owner]
            else:
                best_same=scores.new_zeros(scores.shape)
                owner=torch.full_like(scores,-1,dtype=torch.long)
            positives=[]
            border=[]
            for gi in ids:
                relevant=owner==gi
                good=torch.where(relevant & (best_same>=.50))[0]
                if len(good):
                    # Keep two best-localized anchors and the lowest-scoring well localized one.
                    selected=torch.cat([good[best_same[good].topk(min(2,len(good))).indices],
                                        good[scores[good].argmin()].reshape(1)]).unique()
                    positives.append(selected)
                near=torch.where(relevant & (best_same>=.22) & (best_same<.50))[0]
                if len(near):
                    # A box better explained by another GT receives no conflicting regression signal.
                    near=near[best_any[near]==gi]
                if len(near):
                    border.append(near[best_same[near].topk(min(3,len(near))).indices])
            empty=torch.empty(0,device=boxes.device,dtype=torch.long)
            pos=torch.cat(positives) if positives else empty
            near=torch.cat(border) if border else empty
            hard=torch.where((best_same<.10) & (scores >= -3.89))[0]  # sigmoid approximately 0.02
            if len(hard):
                hard=hard[scores[hard].topk(min(24,len(hard))).indices]
            records.append((image_id,cls,pos,near,hard,best_same,owner))
    return records


def auxiliary(boxes,logits,quality,annotations):
    joint=fused_logits(logits,quality)
    rows=assign(boxes,logits,annotations)
    calibration=[]
    ranking=[]
    border_terms=[]
    for image,cls,pos,near,hard,iou,owner in rows:
        valid=torch.cat([pos,near,hard])
        if len(valid):
            label=((iou[valid]-.22)/.58).clamp(0,1)
            label[torch.isin(valid,hard)]=0
            # Balance positive/ambiguous and annotated-negative populations within a source image.
            bce=F.binary_cross_entropy_with_logits(quality[image,cls,valid].float(),label,reduction="none")
            count=len(pos)+len(near)
            if count:calibration.append(bce[:count].mean())
            if len(hard):calibration.append(.55*bce[count:].mean())
        if len(pos) and len(hard):
            score=joint[image,cls]
            ranking.append(F.softplus(.40+score[hard][None,:]-score[pos][:,None]).mean())
        if len(near):
            tgt=annotations[image].to(boxes.device)[owner[near],:4].float()
            pred=boxes[image,near].float()
            # Raw regression distances can make reversed boxes early in training. Use smooth L1 normalized
            # by the real GT width/height until valid ordered boxes are learned by the native criterion.
            wh=(tgt[:,2:]-tgt[:,:2]).clamp_min(2.)
            border_terms.append(F.smooth_l1_loss((pred-tgt)/wh.repeat(1,2),torch.zeros_like(pred),beta=.35))
    zero=quality.float().sum()*0
    reduce=lambda items:torch.stack(items).mean() if items else zero
    return dict(quality=reduce(calibration),ranking=reduce(ranking),border=reduce(border_terms))


def batch_targets(annotations, hw, device):
    h,w=hw
    classes=[]
    xywh=[]
    indices=[]
    for i,arr in enumerate(annotations):
        b=arr.to(device=device,dtype=torch.float32)
        wh=b[:,2:4]-b[:,:2]
        if (wh<=0).any() or ((b[:,4]!=0)&(b[:,4]!=1)).any():
            raise ValueError("Invalid augmented target")
        whxy=torch.cat([b[:,:2]+.5*wh,wh],dim=1)/b.new_tensor([w,h,w,h])
        classes.append(b[:,4:5]);xywh.append(whxy)
        indices.append(torch.full((len(b),),i,device=device,dtype=torch.float))
    return dict(batch_idx=torch.cat(indices),cls=torch.cat(classes),bboxes=torch.cat(xywh))


def joint_loss(model,criterion,raw,q,annotations,hw):
    # Keep target assignment, distances, and score ranking in FP32 under AMP.
    pred={name:{k:([t.float() for t in v] if isinstance(v,list) else v.float())
                for k,v in branch.items()} for name,branch in raw.items()}
    native=criterion(pred,batch_targets(annotations,hw,q.device))[0].sum()/len(annotations)
    one=pred["one2one"]
    # The native inference cache can hold inference_mode strides after validation.
    # Build training anchors in the current grad context, at the same half-cell centers.
    anchors,strides=make_anchors(one["feats"],model.head.stride,.5)
    boxes=model.head.decode_bboxes(model.head.dfl(one["boxes"]),anchors.T.unsqueeze(0)).transpose(1,2)*strides
    components=auxiliary(boxes,one["scores"],q,annotations)
    total=native+.30*components["quality"]+.15*components["ranking"]+.65*components["border"]
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite quality/detection objective")
    return total,dict(native=native,**components)
