"""Grade an apple using a fixed Woo grader and capture-paired appearance evidence."""
import torch
from torch import nn

import afruit_grade_model as grade
from defect9485_repr_encoder import Encoder


ARCH = "woo_dinov2_capture_contrast_v1"


class CaptureContrastGrade(nn.Module):
    def __init__(self):
        super().__init__()
        self.woo = grade.network()
        self.appearance = Encoder()
        self.adjust = nn.Sequential(nn.LayerNorm(2304), nn.Linear(2304,48),
                                    nn.GELU(), nn.Dropout(.2), nn.Linear(48,3))
        nn.init.zeros_(self.adjust[-1].bias)
        nn.init.zeros_(self.adjust[-1].weight)

    def encode(self,x):
        with torch.no_grad():
            original=self.woo(x).float()
            rgb=((x*self.appearance.rgb_std+self.appearance.rgb_mean)*255).round().clamp(0,255)
            with torch.autocast("cuda",dtype=torch.float16,enabled=x.is_cuda):
                evidence=self.appearance(rgb.permute(0,2,3,1))
        return original,evidence.float()

    def combine(self,old,embedding):
        prior=(old-old.mean(1,keepdim=True)).clamp(-6,6)
        return .5*prior+self.adjust(embedding)

    def forward(self,x):
        return self.combine(*self.encode(x))
