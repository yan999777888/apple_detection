"""Strict inference for the separate frozen-parent quality experiment schema."""
from copy import deepcopy
from pathlib import Path
import re
import math

import torch

from .inference import file_sha
from .model import from_checkpoint, _strict_load_model_state
from .quality import FrozenQualityModel

SCHEMA = "apple_dual_evidence_quality_training_v1"


def load_quality_checkpoint(path,expected_sha256,parent_checkpoint,*,require_calibration=True):
    if not isinstance(expected_sha256,str) or re.fullmatch("[0-9a-f]{64}",expected_sha256) is None:
        raise ValueError("An explicit checkpoint SHA256 is required")
    path=Path(path).absolute()
    if file_sha(path)!=expected_sha256:
        raise ValueError("Quality checkpoint SHA256 differs")
    payload=torch.load(path,map_location="cpu",weights_only=False,mmap=False)
    if payload.get("schema")!=SCHEMA or payload.get("smoke") is not False:
        raise ValueError("Only formal quality-training checkpoints may be used")
    identity=payload["identity"]
    for name in ("quality.py","model.py"):
        if file_sha(Path(__file__).with_name(name))!=identity["source_sha256"][name]:
            raise ValueError("Quality architecture source differs from this checkpoint: "+name)
    recipe=identity["recipe"]
    calibration=[p for p in payload["validations"] if p["update"]==payload["updates"]]
    point=calibration[-1] if calibration else None
    if point is not None and (point["summary"]["images"],point["summary"]["gt_count"],point["summary"]["empty_gt_images"])!=(814,3449,13):
        raise ValueError("Quality calibration requires the complete PApple validation scope")
    if point is not None:
        values = [point["confidence"],point["score"],point["summary"]["precision"],point["summary"]["recall"]]
        if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=1 for v in values):
            raise ValueError("Invalid quality calibration values")
        if not math.isclose(point["score"],min(point["summary"]["precision"],point["summary"]["recall"]),abs_tol=1e-12,rel_tol=0):
            raise ValueError("Quality calibration score differs from P/R")
    if require_calibration and point is None:
        raise ValueError("These quality weights have no matching completed validation")
    parent,_=from_checkpoint(parent_checkpoint,identity["parent_sha256"])
    model=FrozenQualityModel(parent,competition=recipe["competition"],evidence=recipe["evidence"])
    strict=_strict_load_model_state(model.quality_head,payload["quality_head"])
    model.eval()
    provenance={**deepcopy(model.checkpoint_provenance),"checkpoint_sha256":expected_sha256,
                "successful_updates":payload["updates"],"input_size":recipe["input_size"],
                "calibration":deepcopy(point),"strict_quality_load":strict,"code_version":payload["code_version"],
                "optimizer_restored":False,"checkpoint_RNG_restored":False,
                "parent_checkpoint_sha256":identity["parent_sha256"]}
    model.checkpoint_provenance=deepcopy(provenance)
    del payload
    if file_sha(path)!=expected_sha256:
        raise ValueError("Quality checkpoint changed while loading")
    return model,provenance
