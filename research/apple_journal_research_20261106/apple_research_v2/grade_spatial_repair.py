"""Shared mobile features with a supervised surface mark map for mild fruit damage."""
import torch
from torch import nn
from torch.nn import functional as F

from afruit_grade_model import network


class SpatialRepairGrade(nn.Module):
    def __init__(self):
        super().__init__()
        self.parent=network()
        self.mark=nn.Sequential(nn.Conv2d(64,24,1),nn.GroupNorm(4,24),nn.SiLU(),nn.Conv2d(24,1,1))
        self.context=nn.Linear(64,1)

    def encode(self,x):
        for i,block in enumerate(self.parent.features):
            x=block(x)
            if i==3:
                first=x
            elif i==6:
                middle=x
        shared=torch.cat([first,F.interpolate(middle,size=first.shape[-2:],mode="bilinear",align_corners=False)],dim=1)
        old=self.parent.classifier(self.parent.avgpool(x).flatten(1))
        return shared,old

    def local_scores(self, shared, old):
        lesion=self.mark(shared)[:,0]
        # Require a neighbourhood of responses. Normal structures at the calyx and stem
        # provide explicit negative supervision for this map.
        tiles=F.avg_pool2d(lesion[:,None],3,stride=1)[:,0].flatten(1)
        strength=tiles.topk(5,dim=1).values.mean(1)
        context=self.context(shared[:,:,2:-2,2:-2].mean((2,3))).flatten()
        return lesion,strength+context+.25*(old[:,1]-old[:,0])

    @staticmethod
    def combine(old, mild_evidence):
        change=(old.argmax(1)==0)&(mild_evidence>0)
        mild=torch.where(change,old[:,0]+mild_evidence,old[:,1])
        return torch.stack([old[:,0],mild,old[:,2]],dim=1)

    def forward(self,x):
        shared,old=self.encode(x)
        _,evidence=self.local_scores(shared,old)
        return self.combine(old,evidence)
