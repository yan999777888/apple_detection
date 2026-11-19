"""Replay frozen local patch scores on co-photographed cosmetic spots and adjacent skin."""
import cv2
import numpy as np
import torch

import afruit_grade_model as grade
from afruit_grade_train import read, atomic
from matched_grade_loader import load
from grade_pairs_20261119 import extract_patch


@torch.inference_mode()
def main():
    root=grade.ROOT
    work=root/"experiments/grade_pairs_20261119"
    source=root/"datasets/harvard_grade_severe_20261119/manifest.json"
    model,_,_=load(work/"sparse_verifier_v1","cpu")
    ledger=read(source)["local_pairs"]
    patches=[]
    for r in ledger:
        path=root/r["source_path"]
        if grade.file_sha(path)!=r["sha256"]:raise ValueError("Original photo changed")
        frame=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
        for side in ("positive","control"):
            x,y,b,t=r[side+"_xyxy_raw"]
            patches.append(extract_patch(frame,(x+b)/2,(y+t)/2,max(b-x,t-y)))
    scores=model.patch_head(model.embedding(grade.tensor_rgb(patches,"cpu"))).flatten().tolist()
    rows=[dict(id=r["number"],group=r["group_id"],spot=s,normal=n,spot_above_normal=s>n)
          for r,s,n in zip(ledger,scores[::2],scores[1::2],strict=True)]
    atomic(work/"harvard_cosmetic_pairs.json",dict(rows=rows,model_sha=grade.file_sha(work/"sparse_verifier_v1/weights.pt"),
        source_sha=grade.file_sha(source),scope="Three newly audited training-source spot/control pairs from a single white-studio session. Scores are fixed pre-fruit aggregation patch logits, never whole-fruit grades. No fitting on these pairs in this replay."))
    print(rows,flush=True)


if __name__=="__main__":
    torch.set_num_threads(2);cv2.setNumThreads(1);main()
