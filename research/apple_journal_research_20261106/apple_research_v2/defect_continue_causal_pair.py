"""Recover the exact pasted polygon, replay the native geometric transform and train on paired differences."""
from contextlib import contextmanager
from copy import deepcopy

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from defect_continue_public_transfer import draw_transfer
from defect_continue_public_training import transfer_rng
from defect9485_small import SIZE

SNAP = "75e57d602dc84195b8d705105909e37c"
MANIFEST = "e37fa965b09dc6c1ba5839c88d781dfa785e70ceac837eba22d42859de464e75"
PATCH = 320


class RecordPlacement:
    """The last two integer draws of a successful native transfer are x/y and their admissible bounds."""
    def __init__(self,rng):
        self.rng=rng
        self.last=[]
    def integers(self,*args,**kwargs):
        value=self.rng.integers(*args,**kwargs)
        self.last=(self.last+[(args,int(value))])[-2:]
        return value
    def __getattr__(self,k):
        return getattr(self.rng,k)


def donor_lookup(bank):
    return {(c,d.group,d.key):d for c,groups in bank.items() for group in groups for d in group}


def prepared(images,labels,ids,bank,table,sizes,epoch):
    views,targets,pairs,additions=[],[],[],[]
    for local,i in enumerate(ids):
        im,lab=images[i],labels[i]
        trace=RecordPlacement(transfer_rng(epoch,i))
        after,target,info=draw_transfer(im,lab,bank,sizes,trace)
        views.append(after);targets.append(target)
        if info is None:
            continue
        (xb,x),(yb,y)=trace.last
        assert len(xb)==2 and len(yb)==2 and xb[0]==yb[0]
        w=im.shape[1]+1-xb[0]-xb[1]
        h=im.shape[0]+1-yb[0]-yb[1]
        donor=table[info["target_class"],info["group"],info["donor"]]
        weight=donor.mask.astype(np.float32)
        sampled=cv2.resize(weight,(w,h),interpolation=cv2.INTER_AREA if w<weight.shape[1] else cv2.INTER_LINEAR)
        patch=(cv2.resize(weight,(w,h),interpolation=cv2.INTER_NEAREST)>0)&(sampled>0)
        mask=np.zeros(im.shape[:2],np.uint8)
        mask[y:y+h,x:x+w]=patch
        yy,xx=np.nonzero(mask)
        if [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]!=info["box"]:
            raise RuntimeError("Recovered mask disagrees with native transfer receipt")
        if np.any((after!=im).any(2)&~mask.astype(bool)):
            raise RuntimeError("Changed pixels found off the native donor polygon")
        if not np.array_equal(target[:-1],lab):
            raise RuntimeError("Native transfer altered host labels")
        pairs.append(dict(i=local,host=int(i),before=im,mask=mask,cls=donor.cls))
        additions.append(dict(host=int(i),**info))
    return views,targets,pairs,additions


def replay_params(rng,n):
    """Uses a clone so the original augmentation consumes the exact same random sequence."""
    r=np.random.default_rng()
    r.bit_generator.state=deepcopy(rng.bit_generator.state)
    affines=[]
    for i in range(n):
        flip=r.random()<.5
        mat=cv2.getRotationMatrix2D((SIZE/2,SIZE/2),r.uniform(-12,12),r.uniform(.85,1.14))
        mat[:,2]+=r.uniform(-.035,.035,2)*SIZE
        affines.append((flip,mat))
    gain=(r.uniform(.82,1.18,(n,1,1,1))*r.uniform(.92,1.08,(n,3,1,1))).astype(np.float32)
    return affines,gain,r.bit_generator.state


def warped(frame,params,mask=False):
    flip,mat=params
    if flip:
        frame=frame[:,::-1]
    return cv2.warpAffine(frame,mat,(SIZE,SIZE),
                flags=cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR,
                borderValue=0 if mask else (114,114,114))


def before_tensor(im,gain,device):
    x=torch.from_numpy(np.ascontiguousarray(im)).to(device).permute(2,0,1).float()/255
    return (x*torch.as_tensor(gain,device=device)).clamp(0,1)


def map_pairs(pairs,params,post):
    aff,gain,_=params
    mapped=[]
    for v in pairs:
        i=v["i"]
        m=warped(v["mask"],aff[i],True)
        yy,xx=np.nonzero(m)
        if len(xx)<8:
            continue
        # Match the original transformed donor box keep test so rare border slivers
        # dropped by the detector never become auxiliary positive supervision.
        oy,ox=np.nonzero(v["mask"])
        corners=np.array([[ox.min(),oy.min()],[ox.max()+1,oy.min()],
                          [ox.min(),oy.max()+1],[ox.max()+1,oy.max()+1]],np.float64)
        flip,mat=aff[i]
        if flip:
            corners[:,0]=SIZE-corners[:,0]
        xy=corners@mat[:,:2].T+mat[:,2]
        raw=xy.max(0)-xy.min(0)
        vis=xy.max(0).clip(0,SIZE)-xy.min(0).clip(0,SIZE)
        if vis.min()<1 or np.prod(vis)<.30*np.prod(raw):
            continue
        # Local pair at native pixel pitch. Host lesions elsewhere never enter a loss.
        x=int(np.clip((xx.min()+xx.max()+1-PATCH)//2,0,SIZE-PATCH))
        y=int(np.clip((yy.min()+yy.max()+1-PATCH)//2,0,SIZE-PATCH))
        m_crop=m[y:y+PATCH,x:x+PATCH]
        if int(m_crop.sum())!=len(yy):
            continue
        orig=warped(v["before"],aff[i])[y:y+PATCH,x:x+PATCH]
        bg=before_tensor(orig,gain[i],post.device)
        fg=post[i,:,y:y+PATCH,x:x+PATCH]
        mapped.append(dict(host=v["host"],cls=v["cls"],
                      views=torch.stack([bg,fg]),mask=torch.as_tensor(m_crop,device=post.device)))
    return mapped


@contextmanager
def locked_norm(model):
    # An auxiliary pair cannot influence running stats or couple before/after through BN.
    norms=[(m,m.training) for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
    try:
        for m,_ in norms:
            m.eval()
        yield
    finally:
        for m,state in norms:
            m.train(state)


class DeltaMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Conv2d(128,2,1,bias=False)
    def forward(self,features,cls):
        p=self.proj(features)[:,cls].float()
        return p[1]-p[0]


def losses(prediction,aux,pair):
    feat=prediction["one2many"]["feats"][0]
    h,w=feat.shape[-2:]
    mask=pair["mask"].float()[None,None]
    footprint=F.adaptive_avg_pool2d(mask,(h,w))[0,0]
    support=footprint>0
    assert support.any()
    # Compare underlying class logits at the identical locations. Pre-existing
    # signals are baselines, not annotations of healthy fruit skin.
    weights=footprint[support]/footprint.sum().clamp_min(1e-5)
    rank=[]
    cls=pair["cls"]
    for mode in ("one2one","one2many"):
        raw=prediction[mode]["scores"][:,cls,:h*w].float().reshape(2,h,w)[:,support]
        after=(raw[1]*weights).sum()
        before=torch.logsumexp(raw[0]+weights.clamp_min(1e-9).log(),dim=0)
        rank.append(F.softplus(1.+before-after))
    delta=aux(feat,cls)
    positive=(F.softplus(1.-delta[support])*weights).sum()
    # Only a known unchanged local ring is supervised to have zero *response difference*;
    # neither absolute appearance nor the rest of the fruit is labelled negative.
    dilated=F.max_pool2d(support[None,None].float(),5,1,2)[0,0].bool()
    inner=F.max_pool2d(support[None,None].float(),3,1,1)[0,0].bool()
    ring=dilated & ~inner
    unchanged=F.smooth_l1_loss(delta[ring],torch.zeros_like(delta[ring])) if ring.any() else delta.sum()*0
    return torch.stack(rank).mean(),positive+.5*unchanged


def metric_pair(model,aux,pair):
    pred=model(pair["views"])
    return losses(pred,aux,pair)
