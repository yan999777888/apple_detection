"""Verify a sparse set of native fruit-surface patches against a frozen grader."""
from functools import lru_cache
from typing import NamedTuple
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import afruit_grade_model as grade
from grade_detail_patch import NativePatchGrade,pack,square


ARCH="native320_sparse_grade_pairs_20261119"
K=8
PATCH=80


class PairSide(int):
    """Flag for native patches and independent 224/320 whole-fruit inputs."""


class PairBatch(NamedTuple):
    parent: torch.Tensor
    patches: torch.Tensor
    valid: torch.Tensor


def map_hints(records,offset=(0,0),scale=(1.,1.),owner=None):
    result=[]
    for row in records:
        if owner is not None and (row["fruit_id"]!=owner or row["association"]!="unique"):
            continue
        x,y,r,b=row["bbox"]
        result.append(dict(bbox=[(x-offset[0])*scale[0],(y-offset[1])*scale[1],
                                 (r-offset[0])*scale[0],(b-offset[1])*scale[1]],score=float(row["score"])))
    return result


@lru_cache(maxsize=1)
def standalone_detector():
    from defect_blind_review_update import base_network
    return base_network()


@torch.inference_mode()
def standalone_hints(image):
    from afruit_visible_quality import prepare_input,defect_boxes,PRIMARY
    frame=prepare_input(image,1280)
    pred=defect_boxes(standalone_detector(),frame)
    pred=pred[pred[:,4]>=np.asarray(PRIMARY)[pred[:,5].astype(int)]]
    h,w=image.shape[:2];fh,fw=frame.shape[:2]
    return [dict(bbox=(p[:4]*[w/fw,h/fh,w/fw,h/fh]).tolist(),score=float(p[4])) for p in pred]


def extract_patch(frame,x,y,size):
    h,w=frame.shape[:2]
    size=min(max(5,int(round(size))),max(h,w))
    # Clip to the photo while keeping border structures available.
    x1=max(0,min(max(0,w-size),round(x-size/2)))
    y1=max(0,min(max(0,h-size),round(y-size/2)))
    return square(frame[y1:min(h,y1+size),x1:min(w,x1+size)],PATCH)


def proposals(frame,hints):
    h,w=frame.shape[:2]
    diameter=max(7,min(h,w))
    proposals=[]
    for row in sorted(hints,key=lambda z:-z["score"])[:4]:
        x,y,r,b=row["bbox"]
        if r>x and b>y and 0<=(x+r)/2<w and 0<=(y+b)/2<h:
            proposals.append(((x+r)/2,(y+b)/2,np.clip(max(r-x,b-y)*1.65,.12*diameter,.40*diameter),"detector"))
    # Low-contrast marks can be absent from an existing detector. Propose sparse
    # interior texture changes without assigning a lesion label to any peak.
    scale=176/max(h,w); ph,pw=max(1,round(h*scale)),max(1,round(w*scale))
    im=cv2.resize(frame,(pw,ph),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(np.float32)
    smooth=cv2.GaussianBlur(lab,(0,0),3.4)
    rgb=im.astype(np.int16)
    surface=(rgb.max(2)-rgb.min(2)>16)&(rgb.max(2)>65)
    inside=cv2.erode(surface.astype(np.uint8),np.ones((5,5),np.uint8))>0
    contrast=np.maximum(0,smooth[:,:,0]-lab[:,:,0])+.65*np.abs(smooth[:,:,1]-lab[:,:,1])+.45*np.abs(smooth[:,:,2]-lab[:,:,2])
    contrast=cv2.GaussianBlur(contrast,(3,3),.8)
    contrast[~inside]=-1.
    radius=max(3,round(.10*min(ph,pw)))
    for cx,cy,_,_ in proposals:
        cv2.circle(contrast,(round(cx*pw/w),round(cy*ph/h)),radius,-1.,-1)
    while len(proposals)<K:
        _,score,_,(cx,cy)=cv2.minMaxLoc(contrast)
        if score<=0:break
        proposals.append(((cx+.5)*w/pw,(cy+.5)*h/ph,.19*diameter,"texture"))
        cv2.circle(contrast,(cx,cy),radius,-1.,-1)
    return proposals


def patch_bank(frame,hints):
    spots=proposals(frame,hints)
    patches=[extract_patch(frame,*p[:3]) for p in spots]
    patches.extend([np.zeros((PATCH,PATCH,3),np.uint8)]*(K-len(patches)))
    return patches,np.arange(K)<len(spots),spots


def prepare(frames,device,hints=None):
    if hints is None:hints=[standalone_hints(im) for im in frames]
    if len(hints)!=len(frames):raise ValueError("Fruit and candidate counts differ")
    small=[grade.rgb_square(im) for im in frames]
    hi=[square(im,320) for im in frames]
    banks,valid=[],[]
    for im,rows in zip(frames,hints,strict=True):
        patches,mask,_=patch_bank(im,rows)
        banks.extend(patches);valid.append(mask)
    return PairBatch(pack(small,hi,device),grade.tensor_rgb(banks,device).reshape(-1,K,3,PATCH,PATCH),
                     torch.as_tensor(np.stack(valid),device=device))


class PairGrade(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=NativePatchGrade()
        self.patch_head=nn.Sequential(nn.LayerNorm(249),nn.Linear(249,64),nn.SiLU(),nn.Linear(64,1))

    def embedding(self,images):
        x=images
        summaries=[]
        for j,block in enumerate(self.base.parent.features):
            x=block(x)
            if j in (1,3,6):
                h,w=x.shape[-2:];core=x[:,:,h//4:max(h//4+1,h-h//4),w//4:max(w//4+1,w-w//4)]
                center=core.mean((2,3))
                summaries.extend([center,core.flatten(2).amax(2)-center,center-x.mean((2,3))])
            if j==6:break
        rgb=images.mean((2,3))
        chroma=images[:,:,20:60,20:60].mean((2,3))-rgb
        rough=(images-F.avg_pool2d(images,7,1,3,count_include_pad=False)).abs().mean((2,3))
        return torch.cat((*summaries,rgb,chroma,rough),1)

    @staticmethod
    def fruit_logits(base,evidence,valid):
        # Retain the certified severe branch. A mild decision requires spatial
        # support; baseline odds are a prior and can also rescue a normal calyx.
        maximum=evidence.masked_fill(~valid,-10.).amax(1)
        margin=base[:,1]-base[:,0]
        score=.35*margin+maximum
        decision=torch.where(valid.any(1),score,margin)
        safe=base.argmax(1)==2
        adjusted=torch.stack((base[:,0],base[:,0]+decision,torch.minimum(base[:,0],base[:,0]+decision)-1),1)
        return torch.where((safe|~valid.any(1))[:,None],base,adjusted)

    def forward(self,inputs):
        parent=self.base(inputs.parent)
        patch=inputs.patches.flatten(0,1)
        desc=self.embedding(patch)
        scores=self.patch_head(desc).reshape(parent.shape[0],K)
        return self.fruit_logits(parent,scores,inputs.valid)
