"""Estimate fruit severity from distinct verified surface traces in fruit coordinates."""
from typing import NamedTuple
import cv2
import numpy as np
import torch
from torch import nn
import grade_pairs_context_20261119 as ctx
import grade_pairs_20261119 as pair


ARCH="native320_sparse_surface_burden_20261119"
SIDE=36


class BurdenSide(int):pass


class Inputs(NamedTuple):
    context:ctx.Batch
    footprints:torch.Tensor


def fruit_box(frame):
    h,w=frame.shape[:2];scale=144/max(h,w)
    ph,pw=max(1,round(h*scale)),max(1,round(w*scale))
    reduced=cv2.resize(frame,(pw,ph),interpolation=cv2.INTER_AREA)
    v=reduced.astype(np.int16)
    possible=((v.max(2)-v.min(2)>17)&(v.max(2)>59)).astype(np.uint8)
    possible=cv2.morphologyEx(possible,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    n,lab,stats,centers=cv2.connectedComponentsWithStats(possible)
    if n>1:
        j=1+np.argmax(stats[1:,4])
        x,y,r,b,a=stats[j]
        if a>.16*pw*ph and min(r,b)>=.22*min(pw,ph):
            return x*w/pw,y*h/ph,r*w/pw,b*h/ph
    return 0.,0.,float(w),float(h)


def footprints(frame,coords,patches):
    # Estimate traces in a shared fruit coordinate frame; the complete apple
    # face is represented by a fixed grid for small and large camera inputs.
    ox,oy,fw,fh=fruit_box(frame)
    full=np.zeros((pair.K,SIDE,SIDE),np.float32);h,w=frame.shape[:2]
    for j,(cx,cy,s,kind) in enumerate(coords):
        im=patches[j]
        lab=cv2.cvtColor(im,cv2.COLOR_RGB2LAB).astype(np.float32)
        ref=cv2.GaussianBlur(lab,(0,0),5.)
        diff=np.maximum(0.,ref[:,:,0]-lab[:,:,0])+.70*np.abs(ref[:,:,1]-lab[:,:,1])+.35*np.abs(ref[:,:,2]-lab[:,:,2])
        diff=cv2.GaussianBlur(diff,(3,3),.7)
        median=float(np.median(diff));mad=float(np.median(np.abs(diff-median)))
        support=np.clip((diff-median-1.6*max(1.,mad))/(4.+2*mad),0.,1.)
        # The verifier is trained on the central structure. Border pixels in
        # each crop remain context and do not contribute to occupied area.
        support[:16]=0;support[-16:]=0;support[:,:16]=0;support[:,-16:]=0
        size=min(max(5,round(s)),max(h,w))
        x0=max(0,min(max(0,w-size),round(cx-size/2)));y0=max(0,min(max(0,h-size),round(cy-size/2)))
        x1=min(w,x0+size);y1=min(h,y0+size)
        lx,ly=round((x0-ox)/fw*SIDE),round((y0-oy)/fh*SIDE)
        rx,by=round((x1-ox)/fw*SIDE),round((y1-oy)/fh*SIDE)
        l,t=max(0,lx),max(0,ly);r,b=min(SIDE,rx),min(SIDE,by)
        if r<=l or b<=t:continue
        shrunk=cv2.resize(support,(max(1,rx-lx),max(1,by-ly)),interpolation=cv2.INTER_AREA)
        full[j,t:b,l:r]=shrunk[t-ly:b-ly,l-lx:r-lx]
    return full


def prepare(frames,device,hints=None):
    maps=[]
    def observe(frame,coords,patches):maps.append(footprints(frame,coords,patches))
    values,_=ctx.encode_input(frames,device,hints,observe)
    return Inputs(values,torch.as_tensor(np.stack(maps),device=device))


def features(parent,scores,core,maps):
    eligible=core[:,:,2]>0
    prob=scores.sigmoid()*eligible
    sorted_indices=prob.argsort(dim=1,descending=True,stable=True)
    a=maps.flatten(2).gather(1,sorted_indices[:,:,None].expand(-1,-1,SIDE*SIDE))
    p=prob.gather(1,sorted_indices)
    weighted=a*p[:,:,None]
    running=weighted.cummax(1).values
    # The per-pixel maximum exactly removes repeated observations of the same
    # same footprint. Fractional new coverage approximates distinct trace count.
    prev=torch.cat((torch.zeros_like(running[:,:1]),running[:,:-1]),1)
    increment=(weighted-prev).clamp_min(0).sum(2)
    novel=increment/(weighted.sum(2)+1e-6)
    area=running[:,-1];mass=area.sum(1).clamp_min(1e-5)
    grid=torch.linspace(-1,1,SIDE,device=maps.device)
    yy,xx=torch.meshgrid(grid,grid,indexing="ij");x=xx.flatten();y=yy.flatten()
    mx=(area*x).sum(1)/mass;my=(area*y).sum(1)/mass
    spread=((area*((x-mx[:,None]).square()+(y-my[:,None]).square())).sum(1)/mass).sqrt()
    types=core[:,:,3].gather(1,sorted_indices)
    count=(p*novel).sum(1)
    # The area is an appearance proxy and does not claim an absolute wound size.
    frac=area.mean(1)
    independent=p*novel
    second=independent.topk(2,dim=1).values[:,1]
    skin=core[:,:,5].gather(1,sorted_indices)
    return torch.stack((p[:,0],second,frac*100.,weighted.mean(2).amax(1)*100.,count,
        (types*p*novel).sum(1),((1-types)*p*novel).sum(1),
        spread,(parent[:,1]-parent[:,0]).clamp(-10,10)/4.,
        (independent*skin).sum(1)/independent.sum(1).clamp_min(1e-5)),1)


class BurdenGrade(ctx.ContextGrade):
    def __init__(self):
        super().__init__()
        self.burden=nn.Linear(10,1)
        self.register_buffer("mean",torch.zeros(10));self.register_buffer("scale",torch.ones(10))

    def classify_burden(self,parent,feat,core):
        z=self.burden((feat-self.mean)/self.scale).flatten()
        # Positive/negative log-odds come from the whole-face grade estimator.
        # Demotion needs multiple inspected sites; the severe decision is immutable.
        count=(core[:,:,2]>0).sum(1)
        z=torch.where((parent.argmax(1)==1)&(count<2),z.clamp_min(1.e-4),z)
        out=torch.stack((parent[:,0],parent[:,0]+z,torch.minimum(parent[:,0],parent[:,0]+z)-1),1)
        return torch.where(((count==0)|(parent.argmax(1)==2))[:,None],parent,out)

    def forward(self,batch):
        f,c,p=self.encode_batch(batch.context)
        e=self.verify(f,c);desc=features(p,e,c,batch.footprints)
        return self.classify_burden(p,desc,c)
