"""Condition native-patch damage evidence on its source and the surrounding fruit surface."""
from typing import NamedTuple
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import afruit_grade_model as grade
import grade_pairs_20261119 as paired


ARCH="native320_sparse_grounded_pairs_20261119"


class ContextSide(int):
    pass


class Batch(NamedTuple):
    parent:torch.Tensor
    patches:torch.Tensor
    geometry:torch.Tensor


def encode_input(frames,device,hints=None):
    if hints is None:hints=[paired.standalone_hints(im) for im in frames]
    all_pixels=[];all_geometry=[];locations=[]
    for frame,cues in zip(frames,hints,strict=True):
        h,w=frame.shape[:2]
        a=grade.rgb_square(frame);b=paired.square(frame,320)
        patches,valid,coords=paired.patch_bank(frame,cues)
        locations.append(coords)
        # Pack all pixels for one transfer and normalize all RGB channels once.
        all_pixels.append(np.concatenate([b.reshape(-1,3),a.reshape(-1,3),*(p.reshape(-1,3) for p in patches)]))
        geom=np.zeros((paired.K,10),np.float32)
        sh=max(1,round(h*320/max(h,w)));sw=max(1,round(w*320/max(h,w)))
        for i,(cx,cy,size,kind) in enumerate(coords):
            x=(sw*cx/w+(320-sw)//2)/320;y=(sh*cy/h+(320-sh)//2)/320
            ir=patches[i][::5,::5].astype(np.int16)
            hi=ir.max(2);lo=ir.min(2)
            skin=(hi-lo>16)&(hi>65)
            covered=skin[4:12,4:12].mean();outer=skin.mean()
            spec=((hi-lo<23)&(lo>185)).mean()
            bright=ir[4:12,4:12].mean()/255.
            geom[i]=[x,y,1.,float(kind=="detector"),size/min(h,w),covered,outer,spec,bright,min(x,y,1-x,1-y)]
        all_geometry.append(geom)
    buf=torch.from_numpy(np.stack(all_pixels)).to(device=device,dtype=torch.float32)
    buf=(buf/255.-buf.new_tensor(grade.MEAN))/buf.new_tensor(grade.STD)
    n=len(frames);k=paired.K;s=paired.PATCH;cut=320*320;end=cut+224*224
    hi=buf[:,:cut].reshape(n,320,320,3).permute(0,3,1,2)
    lo=buf[:,cut:end].reshape(n,224,224,3).permute(0,3,1,2)
    patches=buf[:,end:].reshape(n,k,s,s,3).permute(0,1,4,2,3)
    return Batch(torch.cat((hi,F.pad(lo,(0,96,0,96))),1),patches,
                 torch.as_tensor(np.stack(all_geometry),device=device)),locations


def prepare(frames,device,hints=None):
    return encode_input(frames,device,hints)[0]


class ContextGrade(paired.PairGrade):
    def __init__(self):
        super().__init__()
        self.context_head=nn.Sequential(nn.LayerNorm(249),nn.Linear(249,24),nn.SiLU())
        self.fusion_head=nn.Sequential(nn.Linear(24+13,24),nn.SiLU(),nn.Linear(24,1))
        self.clean=nn.Sequential(nn.Linear(7,12),nn.SiLU(),nn.Linear(12,1))
        nn.init.zeros_(self.fusion_head[-1].weight);nn.init.zeros_(self.fusion_head[-1].bias)

    def encode_batch(self,inputs):
        f,old=self.base.encode(inputs.parent)
        heat,strength=self.base.evidence(f)
        parent=self.base.repair(old,strength)
        xy=(2*inputs.geometry[:,:,:2]-1).unsqueeze(2)
        local=F.grid_sample(heat[:,None],xy,align_corners=False,padding_mode="border")[:,0,:,0]
        surround=F.avg_pool2d(heat[:,None],13,1,6,count_include_pad=False)
        ring=F.grid_sample(surround,xy,align_corners=False,padding_mode="border")[:,0,:,0]
        core=torch.cat((inputs.geometry,local[:,:,None],ring[:,:,None]),2)
        features=self.embedding(inputs.patches.reshape(-1,3,paired.PATCH,paired.PATCH)).reshape(parent.shape[0],paired.K,-1)
        return features,core,parent

    def verify(self,features,core):
        prior=self.patch_head(features)
        signals=torch.cat((self.context_head(features),core,prior),2)
        return prior[:,:,0]+self.fusion_head(signals)[:,:,0]

    def decisions(self,parent,scores,core):
        valid=core[:,:,2]>0
        v=valid.float();size=v.sum(1).clamp_min(1)
        scores=scores.masked_fill(~valid,-12.)
        m=scores.amax(1);m2=scores.topk(2,1).values[:,1]
        avg=(scores*v).sum(1)/size
        det=(core[:,:,3]*v).sum(1)
        skin=(core[:,:,5]*v).sum(1)/size
        area=(core[:,:,6]*v).sum(1)/size
        prior=parent[:,1]-parent[:,0]
        clean=self.clean(torch.stack((m,m2,avg,det/paired.K,skin,area,prior),1)).flatten()
        margin=.35*prior+m
        # An originally mild fruit may return to good only with affirmative
        # whole-face cleanliness and evidence from more than one patch.
        keep=(parent.argmax(1)==1)&((clean<=0)|(size<2))
        margin=torch.where(keep,margin.clamp_min(1.e-4),margin)
        repaired=torch.stack((parent[:,0],parent[:,0]+margin,torch.minimum(parent[:,0],parent[:,0]+margin)-1),1)
        safe=(parent.argmax(1)==2)|~valid.any(1)
        return torch.where(safe[:,None],parent,repaired),clean

    def forward(self,inputs):
        feat,core,parent=self.encode_batch(inputs)
        return self.decisions(parent,self.verify(feat,core),core)[0]
