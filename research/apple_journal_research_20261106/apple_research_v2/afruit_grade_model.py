"""Three AFruitDB apple appearance grades, in order from good to severe."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/afruit_grade_v1"
SIDE = 224
LABELS = ("good", "medium", "bad")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def network(pretrained=False):
    init = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=init)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(LABELS))
    return model


def rgb_square(frame):
    if frame is None or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a nonempty uint8 BGR image or fruit crop")
    h, w = frame.shape[:2]
    if min(h, w) < 1:
        raise ValueError("Empty fruit crop")
    scale = SIDE / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((SIDE, SIDE, 3), (124, 116, 104), np.uint8)
    y, x = (SIDE - nh) // 2, (SIDE - nw) // 2
    canvas[y:y + nh, x:x + nw] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return canvas


def tensor_rgb(squares, device):
    x = torch.from_numpy(np.stack(squares)).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    return (x / 255 - x.new_tensor(MEAN)[None, :, None, None]) / x.new_tensor(STD)[None, :, None, None]


def batch_preprocess(crops, device):
    return tensor_rgb([rgb_square(im) for im in crops], device)


def decide(logits, threshold):
    probabilities = logits.float().softmax(dim=1)
    grades = probabilities.argmax(dim=1)
    # If the best class is good but there is substantial severe evidence,
    # route it to the middle grade. Its observed grade is not forced to severe.
    return torch.where((grades == 0) & (probabilities[:, 2] >= threshold), 1, grades)


def load_model(device):
    receipt = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    weights = OUT / "weights.pt"
    if file_sha(weights) != receipt["weights_sha256"]:
        raise ValueError("Grade weights differ from the validation-selected checkpoint")
    model = network()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.to(device).eval().requires_grad_(False), float(receipt["threshold"])
