"""Direct shallow mark versus normal-structure evidence. The parent supplies only severe status."""
import torch
from torch import nn
from torch.nn import functional as F
from afruit_grade_model import network


class MarkContrastGrade(nn.Module):
    def __init__(self):
        super().__init__()
        self.parent=network()
        self.mark=nn.Sequential(nn.Conv2d(40,32,1),nn.SiLU(),
                                nn.Conv2d(32,32,3,padding=1,groups=32),nn.SiLU(),
                                nn.Conv2d(32,1,1))
        nn.init.constant_(self.mark[-1].bias,-1.)

    def encode(self,x):
        for i,block in enumerate(self.parent.features):
            x=block(x)
            if i==1:early=x
            elif i==3:middle=x
        pooled=self.parent.avgpool(x).flatten(1)
        coarse=self.parent.classifier(pooled)
        # Feature at each point is compared only to its own small surrounding neighbourhood.
        early=early-F.avg_pool2d(early,9,1,4,count_include_pad=False)
        middle=middle-F.avg_pool2d(middle,5,1,2,count_include_pad=False)
        field=torch.cat((early,F.interpolate(middle,early.shape[-2:],mode="bilinear",align_corners=False)),1)
        return field,coarse

    def evidence(self,field):
        score=self.mark(field)
        # Two adjacent pixels in each 4-by-4 source region are evidence for a small trace.
        score=F.avg_pool2d(score,2,2)[:,0]
        core=score[:,2:-2,2:-2]
        smooth=F.avg_pool2d(core[:,None],2,1)[:,0].flatten(1)
        strength=smooth.topk(3,1).values.mean(1)
        return score,strength

    @staticmethod
    def combine(coarse,e):
        strongest=coarse[:,:2].amax(1)
        margin=coarse[:,2]-strongest
        severe=torch.where(coarse.argmax(1)==2,margin.clamp_min(1.e-4),margin.clamp_max(-1.e-4))
        return torch.stack((-F.relu(e),-F.relu(-e),severe),1)

    def forward(self,x):
        field,coarse=self.encode(x)
        _,e=self.evidence(field)
        return self.combine(coarse,e)

