"""Reproduce grading errors and locate the checkpoint and input change at which they arise."""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from afruit_grade_model import ROOT, decide, file_sha
from afruit_grade_train import atomic, read
from afruit_visible_quality import prepare_input
from matched_grade_loader import load, tensor_at_size


OUT=ROOT/"experiments/grade_error49_v1/checkpoints"
PREVIOUS=ROOT/"experiments/grade_matched_review_v1"
MODELS=dict(original="afruit_grade_release_v1",crop_aligned="afruit_grade_crop_alignment_v1",
            replay="afruit_grade_matched_v1/control",woo="afruit_grade_matched_v1/paired")


def edge_background(frame):
    # Alters only nearly white, low-chroma pixels connected to an image boundary.
    low=frame.min(axis=2).astype(np.int16)
    high=frame.max(axis=2).astype(np.int16)
    neutral=(low>=230)&(high-low<=20)
    _, labels=cv2.connectedComponents(neutral.astype(np.uint8),connectivity=8)
    border=np.unique(np.r_[labels[0,:],labels[-1,:],labels[:,0],labels[:,-1]])
    mask=np.isin(labels,border[border>0])
    fraction=float(mask.mean())
    if not (.05<=fraction<=.8):
        return None,fraction
    out=frame.copy()
    out[mask]=128
    return out,fraction


def gather():
    frames=[]; rows=[]
    def add(im,panel,source,r,ref,**other):
        frames.append(im);rows.append(dict(panel=panel,source=source,path=r["path"],target=ref,**other))
    for r in read(PREVIOUS/"external/predictions.json"):
        if r["panel"]!="HDF":
            continue
        with Image.open(ROOT/r["path"]) as image:
            im=np.ascontiguousarray(np.asarray(ImageOps.exif_transpose(image).convert("RGB"))[:,:,::-1])
        gray,fraction=edge_background(im)
        info=dict(expected=r["candidate"],reference=r["original"],group=r["group"],border_fraction=fraction)
        add(im,"hdf","original",r,r["target"],**info)
        if gray is not None:
            add(gray,"hdf","gray_edge",r,r["target"],**info)
    before={r["path"]:r for r in read(ROOT/"experiments/afruit_visible_quality_v1/test_predictions_1280.json")}
    for r in read(PREVIOUS/"field/paired_predictions.json"):
        if r["linked_grade"] is None:
            continue
        im=cv2.imdecode(np.fromfile(ROOT/r["path"],np.uint8),cv2.IMREAD_COLOR)
        im=prepare_input(im,1280)
        x,y,right,bottom=before[r["path"]]["fruits"][0]["crop"]
        for source,image,key,ref in (("crop",im[y:bottom,x:right],"linked_grade","previous_linked"),("whole",im,"direct_grade","previous_direct")):
            add(image,"af",source,r,r["truth"],expected=r[key],reference=r[ref],group=r["group"])
    return frames,rows


@torch.inference_mode()
def run():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    frames,rows=gather()
    # Transform once; using CPU for HDF (original protocol), CUDA for AF (deployment).
    sets={device:[i for i,r in enumerate(rows) if (r["panel"]=="af")== (device=="cuda")] for device in ("cuda","cpu")}
    tensors={device:[tensor_at_size([frames[i] for i in ids[j:j+12]],"cpu",224) for j in range(0,len(ids),12)] for device,ids in sets.items()}
    del frames
    def infer(key,body,cut):
        for device,ids in sets.items():
            model=body.to(device).eval()
            pos=0
            for batch in tensors[device]:
                logits=model(batch.to(device))
                preds=decide(logits,cut).cpu().tolist()
                prob=logits.softmax(1).cpu().tolist()
                for cls,vec in zip(preds,prob,strict=True):
                    rec=rows[ids[pos]]
                    rec[key]=cls
                    if key=="woo":
                        rec["woo_scores"]=vec
                    pos+=1
    hashes={}
    for key,path in MODELS.items():
        d=ROOT/"experiments"/path
        m,cut,side=load(d,"cpu")
        if side!=224:
            raise ValueError("Expected matched 224-input checkpoints")
        hashes[key]=file_sha(d/"weights.pt")
        infer(key,m,cut)
    differences={}
    for donor in MODELS:
        model,cut,_=load(ROOT/"experiments"/MODELS["woo"],"cpu")
        other,_,_=load(ROOT/"experiments"/MODELS[donor],"cpu")
        src=dict(other.named_buffers())
        differences[donor]=sum(not torch.equal(v,src[k]) for k,v in model.named_buffers()
                               if k.endswith(("running_mean","running_var")))
        if donor not in ("replay","crop_aligned"):
            continue
        for k,v in model.named_buffers():
            if k.endswith(("running_mean","running_var")):
                v.copy_(src[k])
        infer("woo_bn_from_"+donor,model,cut)
    h=[r for r in rows if r["panel"]=="hdf" and r["source"]=="original" and r["target"]=="fresh"]
    a=[r for r in rows if r["panel"]=="af" and r["source"]=="crop" and r["target"]==1]
    sets=dict(hdf_all_fresh=h,hdf_36=[r for r in h if r["expected"]==2],af_resolved_mild=a,af_13=[r for r in a if r["expected"]==0])
    if len(sets["hdf_36"])!=36 or len(sets["af_13"])!=13:
        raise ValueError("Error population mismatch")
    keys=[*MODELS,"woo_bn_from_replay","woo_bn_from_crop_aligned"]
    counts={s:{k:[sum(r[k]==i for r in data) for i in range(3)] for k in keys} for s,data in sets.items()}
    gray={r["path"]:r for r in rows if r["source"]=="gray_edge"}
    for name,data in sets.items():
        if name.startswith("hdf"):
            counts[name]["woo_gray_edge"]=[sum(r["path"] in gray and gray[r["path"]]["woo"]==i for r in data) for i in range(3)]
    match={}
    for s in ("hdf","af"):
        main=[r for r in rows if r["panel"]==s and r["source"]!="gray_edge"]
        match[s]=dict(n=len(main),matched_candidate=sum(r["expected"]==r["woo"] for r in main),matched_original=sum(r["reference"]==r["original"] for r in main))
    atomic(OUT/"ledger.json",rows)
    atomic(OUT/"summary.json",dict(checkpoint_sha256=hashes,reproduced=match,counts=counts,
                                   batch_norm_buffer_differences_vs_woo=differences,
                                   scope="Diagnostic counterfactuals, no parameter optimization. Synthetic gray edges illustrate sensitivity, and do not establish the origin of an individual mistake."))
    print(match,counts,flush=True)


if __name__=="__main__":
    run()
