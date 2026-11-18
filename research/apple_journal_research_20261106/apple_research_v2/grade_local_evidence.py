"""A same-backbone good/minor head that can only downgrade a predicted good apple."""
import torch
from torch import nn
from torch.nn import functional as F

from afruit_grade_model import network


class LocalEvidenceGrade(nn.Module):
    def __init__(self):
        super().__init__()
        self.parent = network()
        self.local = nn.Linear(2 * (40 + 48 + 96), 1)

    def features_and_grade(self, tensor):
        pooled = []
        for i, block in enumerate(self.parent.features):
            tensor = block(tensor)
            if i in (6, 8, 11):
                # Avoid letterbox-dominated borders and pool neighbouring responses.
                center = tensor[:, :, 1:-1, 1:-1]
                tiles = F.adaptive_avg_pool2d(center, (4, 4))
                pooled.extend((tiles.mean((2, 3)), tiles.amax((2, 3))))
        parent_scores = self.parent.classifier(self.parent.avgpool(tensor).flatten(1))
        return torch.cat(pooled, dim=1), parent_scores

    def forward(self, tensor):
        features, old = self.features_and_grade(tensor)
        evidence = self.local(features).squeeze(1)
        promote_minor = (old.argmax(1) == 0) & (evidence > 0)
        mild = torch.where(promote_minor, old[:, 0] + evidence, old[:, 1])
        return torch.stack((old[:, 0], mild, old[:, 2]), dim=1)


def architecture_model(name):
    if name == "mobilenet_v3_small_native320_patch_v1":
        from grade_detail_patch import NativePatchGrade
        return NativePatchGrade()
    if name == "mobilenet_v3_small_confirmed_mark_contrast_v1":
        from grade_mark_contrast import MarkContrastGrade
        return MarkContrastGrade()
    if name == "mobilenet_v3_small_supervised_surface_v1":
        from grade_spatial_repair import SpatialRepairGrade
        return SpatialRepairGrade()
    if name == "mobilenet_v3_small_local_evidence_v1":
        return LocalEvidenceGrade()
    if name in (None, "MobileNet V3 Small", "MobileNetV3-Small"):
        return network()
    raise ValueError(f"Unrecognized fruit grading model: {name}")
