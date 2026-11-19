"""Use the two existing whole-fruit graders to check severity disagreements."""
import torch
from torch import nn
import afruit_grade_model as grade
from grade_detail_patch import NativePatchGrade


ARCH="original_native320_severity_agreement_v1"


class SeverityAgreement(nn.Module):
    def __init__(self):
        super().__init__()
        self.native=NativePatchGrade()
        self.published=grade.network()

    def forward(self, inputs):
        native=self.native(inputs)
        published=self.published(inputs[:,3:,:224,:224].contiguous(memory_format=torch.channels_last))
        a=native.argmax(dim=1)
        b=published.argmax(dim=1)
        moderate=((a==2)&(b!=2))|((a==0)&(b==2))
        # A disputed severe call may only be routed to the middle grade.
        # Encoded vote scores serve the deployed argmax API, not calibration.
        middle=native[:,1]
        highest=native.amax(dim=1)
        return torch.stack((native[:,0],torch.where(moderate,highest+1.,middle),native[:,2]),dim=1)
