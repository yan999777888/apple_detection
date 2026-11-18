"""Load an opt-in public-data grade candidate without altering frozen model files."""
from pathlib import Path

import cv2
import numpy as np
import torch

import afruit_grade_model as original
from afruit_grade_train import read
from grade_local_evidence import architecture_model


def tensor_at_size(frames, device, side):
    from grade_detail_patch import DetailSide,pack,square
    if isinstance(side,DetailSide):
        return pack([original.rgb_square(im) for im in frames],[square(im,320) for im in frames],device)
    if side == original.SIDE:
        return original.batch_preprocess(frames, device)
    squares=[]
    for frame in frames:
        height,width=frame.shape[:2]
        if min(height,width)<1 or frame.ndim!=3 or frame.shape[2]!=3 or frame.dtype!=np.uint8:
            raise ValueError("Expected a decoded BGR fruit crop")
        ratio=side/max(height,width)
        scaled_h,scaled_w=max(1,round(height*ratio)),max(1,round(width*ratio))
        im=cv2.resize(frame,(scaled_w,scaled_h),interpolation=cv2.INTER_AREA if ratio<1 else cv2.INTER_LINEAR)
        canvas=np.full((side,side,3),(124,116,104),np.uint8)
        y,x=(side-scaled_h)//2,(side-scaled_w)//2
        canvas[y:y+scaled_h,x:x+scaled_w]=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        squares.append(canvas)
    return original.tensor_rgb(squares,device)


def load(directory, device):
    folder=Path(directory)
    metadata=read(folder/"selection.json")
    actual=original.file_sha(folder/"weights.pt")
    if actual!=metadata["weights_sha256"]:
        raise ValueError("Candidate does not match its selection receipt")
    side=int(metadata.get("input_side",224))
    if side not in (224,320):
        raise ValueError("Unknown grade input size")
    if metadata.get("architecture") == "mobilenet_v3_small_native320_patch_v1":
        from grade_detail_patch import DetailSide
        if side!=320:
            raise ValueError("Native local verifier requires dual 320/224 input")
        side=DetailSide(320)
    model=architecture_model(metadata.get("architecture"))
    model.load_state_dict(torch.load(folder/"weights.pt",map_location="cpu",weights_only=True),strict=True)
    return model.to(device).eval().requires_grad_(False),float(metadata["threshold"]),side


def install(pipe, directory):
    pipe.grade,pipe.grade_threshold,side=load(directory,"cuda")
    pipe.preprocess_grade=lambda frames,device:tensor_at_size(frames,device,side)
    return side
