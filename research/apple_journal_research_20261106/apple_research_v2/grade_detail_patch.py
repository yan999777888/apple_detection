"""Mobile verifier of small visible surface marks from a native 320px fruit crop."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import afruit_grade_model as grade


ARCH = "mobilenet_v3_small_native320_patch_v1"


class DetailSide(int):
    """Numeric side in reports, carrying the exact two-input ABI to preprocessing."""


def square(frame, side):
    h,w=frame.shape[:2]
    scale=side/max(h,w)
    sh,sw=max(1,round(h*scale)),max(1,round(w*scale))
    resized=cv2.resize(frame,(sw,sh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    canvas=np.full((side,side,3),(124,116,104),np.uint8)
    x,y=(side-sw)//2,(side-sh)//2
    canvas[y:y+sh,x:x+sw]=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
    return canvas


def pack(small,detail,device):
    native=grade.tensor_rgb(detail,device)
    parent=grade.tensor_rgb(small,device)
    return torch.cat((native,F.pad(parent,(0,96,0,96))),dim=1)


class NativePatchGrade(nn.Module):
    def __init__(self):
        super().__init__()
        self.parent=grade.network()
        self.verifier=nn.Sequential(nn.Conv2d(49,32,1),nn.GroupNorm(4,32),nn.SiLU(),
            nn.Conv2d(32,32,3,padding=1,groups=32),nn.SiLU(),nn.Conv2d(32,1,1))
        nn.init.constant_(self.verifier[-1].bias,-1.)

    def encode(self,x):
        fine=x[:,:3]
        coarse=self.parent(x[:,3:,:224,:224].contiguous(memory_format=torch.channels_last))
        early=self.parent.features[1](self.parent.features[0](fine))
        middle=self.parent.features[3](self.parent.features[2](early))
        rgb=F.avg_pool2d(fine,4,4)
        hi=rgb-F.avg_pool2d(rgb,5,1,2,count_include_pad=False)
        wide=rgb-F.avg_pool2d(rgb,13,1,6,count_include_pad=False)
        features=torch.cat((early,F.interpolate(middle,early.shape[-2:],mode="bilinear",
                            align_corners=False),rgb,hi,wide),1)
        return features,coarse

    def evidence(self,features):
        heat=self.verifier(features)[:,0]
        # Score one small contiguous patch, then require corroborating adjacent pixels.
        local=F.avg_pool2d(heat[:,None],3,1,1,count_include_pad=False)[:,0]
        return local,local.flatten(1).topk(6,1).values.mean(1)

    @staticmethod
    def repair(old,e):
        evidence=e-.2*(old[:,0]-old[:,1]).clamp_min(0)
        switch=(old.argmax(1)==0)&(evidence>0)
        mild=torch.where(switch,old[:,0]+evidence,old[:,1])
        return torch.stack((old[:,0],mild,old[:,2]),1)

    def forward(self,x):
        features,old=self.encode(x)
        _,e=self.evidence(features)
        return self.repair(old,e)
