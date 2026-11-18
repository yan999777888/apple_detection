"""Positive-only detection supervision on the original photographed lesion and its real context."""
from contextlib import contextmanager
from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw
from torch import nn
from torch.nn import functional as F
from class0_patch_refinement import ROOT,read,sha
from defect_continue_public_transfer import SSDA, child,select_snapshot
from defect_continue_public_training import PARENT_SHA
from defect9485_small import SIZE

OUT=ROOT/"experiments/defect_continue_native_partial"
SNAP="75e57d602dc84195b8d705105909e37c"
MANIFEST="e37fa965b09dc6c1ba5839c88d781dfa785e70ceac837eba22d42859de464e75"
WINDOW=384
STRENGTH=.12
NORM_CAP=.10


def entries():
    snap=select_snapshot(name=SNAP)
    src=snap/"positives.json"
    assert sha(src)==MANIFEST
    receipt=read(snap/"receipt.json")
    assert receipt["manifest_sha256"]==MANIFEST and sha(SSDA)==receipt["SSDA_manifest_sha256"]
    approves={r["key"]:r for r in read(snap/"image_decisions.json")}
    specs={r["archive_sha256"]:r for r in receipt["source_specs"]}
    rows=read(src)
    for r in rows:
        decision=approves[r["source_id"]]
        spec=specs[r["archive_sha256"]]
        assert r["raw_split"]==decision["raw_split"]=="train"
        assert decision["decision"]=="export positive crops" and r["group_id"]==decision["group_id"]
        assert r["geometry"]=="polygon" and r["mask_supervision"] and r["background_supervision"] is False
        assert spec["category_map"][r["category_name"]]==r["target_class"] and spec["license"]==r["license"]
        assert spec["license_evidence_url"] and r["role"]=="training_only_local_positive"
        assert r["image_sha256"]==decision["sha256"]
    assert [sum(r["target_class"]==c for r in rows) for c in (0,1)]==[224,33]
    return snap,rows


@lru_cache(maxsize=10)
def scaled_source(path,expected):
    assert sha(path)==expected
    with Image.open(path) as frame:
        rgb=np.asarray(frame.convert("RGB"))
    H,W=rgb.shape[:2]
    ratio=SIZE/max(H,W)
    h,w=round(H*ratio),round(W*ratio)
    dx,dy=(SIZE-w)//2,(SIZE-h)//2
    board=np.full((SIZE,SIZE,3),114,np.uint8)
    board[dy:dy+h,dx:dx+w]=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_LINEAR)
    return rgb,board,(dx,dy,w,h)


def native_record(snap,r):
    # Resize the actual camera frame once at the usual pixel pitch. Cropping a
    # neighborhood never moves an annotation onto a different fruit.
    rgb,board,(dx,dy,w,h)=scaled_source(child(snap,r["full_image"]),r["image_sha256"])
    H,W=rgb.shape[:2]
    x0,y0,x1,y1=r["crop_xyxy"]
    mpath=child(snap,r["mask"])
    assert sha(mpath)==r["mask_sha256"]
    with Image.open(mpath) as f:
        small=np.asarray(f.convert("L"))>0
    reference=Image.new("L",(x1-x0,y1-y0))
    draw=ImageDraw.Draw(reference)
    for polygon in r["polygon_in_crop"]:
        draw.polygon([tuple(x) for x in polygon],fill=255)
    assert np.array_equal(np.asarray(reference)>0,small)
    full=np.zeros((H,W),np.uint8)
    full[y0:y1,x0:x1]=small
    mask=np.zeros(board.shape[:2],np.uint8)
    mask[dy:dy+h,dx:dx+w]=cv2.resize(full,(w,h),interpolation=cv2.INTER_NEAREST)
    box=np.asarray(r["box_xyxy"],np.float32)*[w/W,h/H,w/W,h/H]+[dx,dy,dx,dy]
    side=WINDOW
    if np.max(box[2:]-box[:2])>side-24:
        side=SIZE
    offset=np.clip(np.floor((box[:2]+box[2:]-side)*.5),0,SIZE-side).astype(int)
    l,t=offset
    pad=[l,t,l,t]
    crop_mask=np.ascontiguousarray(mask[t:t+side,l:l+side])
    cropped=board[t:t+side,l:l+side].copy()
    assert crop_mask.any() and crop_mask.sum()==mask.sum()
    return dict(rgb=cropped,mask=crop_mask,box=(box-pad).astype(np.float32),
                cls=r["target_class"],key=f"{r['source_id']}#{r['annotation']}",group=r["group_id"],
                original_hw=[H,W],offset=offset.tolist(),context_side=side)


def tensor(v,device):
    return torch.from_numpy(v["rgb"]).to(device).permute(2,0,1)[None].float()/255


def iou_giou(b,t):
    inter=(torch.minimum(b[:,2:],t[2:])-torch.maximum(b[:,:2],t[:2])).clamp_min(0).prod(1)
    area=(b[:,2:]-b[:,:2]).clamp_min(0).prod(1)
    union=area+(t[2:]-t[:2]).prod()-inter
    iou=inter/union.clamp_min(.001)
    encl=(torch.maximum(b[:,2:],t[2:])-torch.minimum(b[:,:2],t[:2])).clamp_min(0).prod(1)
    return iou,iou-(encl-union)/encl.clamp_min(.001)


def object_loss(pred,head,r):
    ref=pred["one2one"]["scores"]
    device=ref.device
    cls=r["cls"]
    mask=torch.as_tensor(r["mask"],device=device,dtype=torch.float32)[None,None]
    box=torch.as_tensor(r["box"],device=device)
    wh=(box[2:]-box[:2]).clamp_min(2.)
    center=(box[2:]+box[:2])*.5
    # A partially covered finest-level cell can carry a thin scratch signal.
    h,w=pred["one2one"]["feats"][0].shape[-2:]
    support=F.adaptive_avg_pool2d(mask,(h,w))[0,0].flatten()
    ids=torch.nonzero(support>0,as_tuple=False)[:,0]
    xpos=(ids%w+.5)*mask.shape[-1]/w
    ypos=(ids//w+.5)*mask.shape[-2]/h
    center_cost=(((torch.stack([xpos,ypos],1)-center)/wh).square()).sum(1)
    # Top few supporting cells by centrality and polygon coverage, independent
    # of current model score; background and unannotated classes receive zero loss.
    weights=support[ids].sqrt()*torch.exp(-3*center_cost)
    chosen=torch.topk(weights,min(3,len(ids))).indices
    sites=ids[chosen]
    weights=weights[chosen]
    weights=weights/weights.sum().clamp_min(1e-8)
    parts=[];details={}
    for mode in ("one2one","one2many"):
        raw=pred[mode]
        logits=raw["scores"][0,cls,sites].float()
        decoded=head._get_decode_boxes(raw)[0,:,sites].T.float()
        iou,giou=iou_giou(decoded,box)
        q=iou.detach().clamp(0,1)
        # No encouragement to raise a well-scored positive or an ill-localized
        # response. Geometry remains supervised at the same positive anchors.
        residual=(1.10-logits).clamp_min(0)
        score_loss=F.smooth_l1_loss(residual,torch.zeros_like(residual),reduction="none",beta=1.)
        score_loss*=((q-.15)/.45).clamp(0,1)
        offsets=(decoded-box)/wh.repeat(2)
        box_loss=1-giou+.25*F.smooth_l1_loss(offsets,torch.zeros_like(offsets),beta=.2,reduction="none").mean(1)
        parts.append(((.65*score_loss+box_loss)*weights).sum())
        details[mode]=dict(score=(logits.detach()*weights).sum(),iou=(q*weights).sum(),
                           valid_scores=((q>.15)&(logits<1.10)).sum().detach())
    return torch.stack(parts).mean(),dict(sites=sites,details=details,head_grid=h*w)


@contextmanager
def native_norm(model):
    head=model.model[-1]
    saved=(head.shape,head.anchors,head.strides)
    norms=[(m,m.training) for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
    try:
        for m,_ in norms:
            m.eval()
        yield
    finally:
        for m,state in norms:
            m.train(state)
        head.shape,head.anchors,head.strides=saved


def responsive_params(model):
    # For the external partial labels only the final class/box tower convolutions
    # can change. Exclude all biases, BN, backbone and every unrelated parameter.
    h=model.model[-1]
    params=[]
    for source in (h.one2one,h.one2many):
        for tower in (source["cls_head"],source["box_head"]):
            params.extend(v for v in tower[0].parameters() if v.ndim==4 and v.requires_grad)
    return params


def preserve_native_groups(model,parameters,x,labels):
    """Tangent directions of the native scores whose mean should stay unchanged.

    These constraints come from the supervised SSDA batch. They do not annotate any
    external location and do not assign a background target even to a native anchor.
    """
    branch=model(x)["one2one"]
    H,W=branch["feats"][0].shape[-2:]
    logits=branch["scores"][:,:,:H*W].float().reshape(len(x),2,H,W)
    with torch.no_grad():
        xyxy=labels["bboxes"].detach().cpu().numpy()
        ids=labels["batch_idx"].detach().cpu().numpy().astype(int)
        allow=np.ones((len(x),H,W),bool)
        for i,(cx,cy,w,h) in zip(ids,xyxy):
            l,t=np.floor(np.array([cx-w,cy-h])*[W,H]).astype(int).clip(0,[W,H])
            r,b=np.ceil(np.array([cx+w,cy+h])*[W,H]).astype(int).clip(0,[W,H])
            allow[i,t:b,l:r]=False
        allowed=torch.as_tensor(allow,device=x.device)
    means=[]
    for c in (0,1):
        z=logits[:,c][allowed]
        if not z.numel():
            continue
        ordinary=z[z.detach()<-2.2]
        if ordinary.numel():
            means.append(ordinary.mean())
        # Include each frame's strongest out-of-region responses, so changes
        # cannot hide in rare high-confidence sites by conserving only a global mean.
        tough=[]
        for i in range(len(x)):
            sites=logits[i,c][allowed[i]]
            if sites.numel():
                idx=sites.detach().topk(min(16,sites.numel())).indices
                tough.append(sites[idx].mean())
        if tough:
            means.append(torch.stack(tough).mean())
    basis=[]
    for i,m in enumerate(means):
        g=torch.autograd.grad(m,parameters,retain_graph=i+1<len(means),allow_unused=True)
        v=torch.cat([(t.detach().float() if t is not None else torch.zeros_like(p,dtype=torch.float32)).flatten()
                      for p,t in zip(parameters,g)])
        for q in basis:
            v-=torch.dot(v,q)*q
        norm=v.norm()
        if norm>1e-7:
            basis.append(v/norm)
    return basis


def remove_native_drift(parameters,grads,basis):
    vec=torch.cat([(g.detach().float() if g is not None else torch.zeros_like(p,dtype=torch.float32)).flatten()
                   for p,g in zip(parameters,grads)])
    original=vec.norm().clamp_min(1e-12)
    for _ in range(2):
        for q in basis:
            vec-=torch.dot(vec,q)*q
    ratio=(vec.norm()/original).item()
    parts=vec.split([p.numel() for p in parameters])
    return [v.view_as(p) for v,p in zip(parts,parameters)],ratio


def combine_grads(parameters,grads,apply=True):
    pairs=[(p,g) for p,g in zip(parameters,grads,strict=True) if p.grad is not None and g is not None]
    if not pairs:
        raise RuntimeError("No common primary and external head gradients")
    dot=sum((p.grad.float()*g.float()).sum() for p,g in pairs)
    nd=sum(p.grad.float().square().sum() for p,g in pairs).sqrt()
    ne=sum(g.float().square().sum() for p,g in pairs).sqrt()
    if not torch.isfinite(torch.stack([dot,nd,ne])).all():
        raise RuntimeError("Non-finite gradient comparison")
    cos=(dot/(nd*ne).clamp_min(1e-12)).item()
    gain=min(STRENGTH,NORM_CAP*nd.item()/max(ne.item(),1e-12)) if cos>0 and nd>0 and ne>0 else 0.
    if apply and gain:
        for p,g in pairs:
            p.grad.add_(g,alpha=gain)
    return dict(cos=cos,applied=gain>0,additional_over_native_norm=gain*ne.item()/max(nd.item(),1e-12))
