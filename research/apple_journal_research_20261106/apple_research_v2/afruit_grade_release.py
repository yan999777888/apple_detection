"""Frozen winner used by serial deployment and the held-out grading benchmark."""
import json
import torch

from afruit_grade_model import ROOT, batch_preprocess, decide, file_sha, network

OUT = ROOT / "experiments/afruit_grade_release_v1"


def load_model(device):
    cfg = json.loads((OUT/"selection.json").read_text(encoding="utf-8"))
    weight = OUT/"weights.pt"
    if file_sha(weight)!=cfg["weights_sha256"]:
        raise ValueError("Selected deployment weights changed")
    model=network()
    model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=True))
    return model.to(device).eval().requires_grad_(False),float(cfg["threshold"])
