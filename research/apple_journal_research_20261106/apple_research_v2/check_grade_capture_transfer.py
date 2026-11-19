"""Fit independent, training-only apple grade probes while withholding whole capture groups."""
from collections import Counter
import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

import afruit_grade_crop_alignment as align
import afruit_grade_model as grade
import afruit_grade_train as io
from defect9485_repr import encoder, META_SHA


OUT=grade.ROOT/"experiments/grade_pairs_20261119/capture_holdout"
WOO=grade.ROOT/"datasets/apple_matched_quality_v1/manifest.json"


@torch.inference_mode()
def main():
    if (OUT/"result.json").exists():raise FileExistsError(OUT)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    io.atomic(OUT/"protocol.json",dict(feature_encoder=META_SHA,af_train=grade.file_sha(io.MANIFEST),
        woo=grade.file_sha(WOO),scope="Hold out each entire AF train capture group containing grade 0 or 1. Use automatically located single AF fruit, no second view of a photo included. Woo uses reviewed training crops.",
        classifier="Fixed L2 logistic estimator C=12 on L2-normalized, frozen generic DINO features. Balance each source/class cell, fixed 0.5 decision. Only contrast adding the unchanged 58 Woo training views to AF.",
        independence="The feature encoder never fitted to this project's AF training labels. Capture group is a proxy, physical individual identities unknown. No AF development, history or independent orchard evaluation."))
    rows=align.split_rows("train");crops=align.load_cache("train",rows)
    meta=[];images=[]
    for r,crop in zip(rows,crops,strict=True):
        if r["grade"] in (0,1) and crop is not None:
            meta.append(("AF",r["group_id"],r["grade"]));images.append(crop)
    for r in io.read(WOO)["samples"]:
        if r["split"]=="train" and r["grade"] in (0,1):
            meta.append(("Woo",r["condition_id"],r["grade"]));images.append(grade.rgb_square(align.read_frame(r)))
    enc=encoder().cuda().eval()
    feat=[]
    for start in range(0,len(images),12):
        with torch.autocast("cuda",dtype=torch.float16):
            z=enc(torch.as_tensor(np.stack(images[start:start+12]),device="cuda"))
        feat.append(z.float().cpu().numpy())
    x=normalize(np.concatenate(feat));y=np.array([r[2] for r in meta])
    folds={}
    for group in sorted({g for s,g,t in meta if s=="AF"}):
        val=np.asarray([s=="AF" and g==group for s,g,t in meta])
        arms={}
        for woo in (False,True):
            train=np.asarray([s=="Woo" and woo or s=="AF" and g!=group for s,g,t in meta])
            cells=Counter((s,t) for allowed,(s,g,t) in zip(train,meta) if allowed)
            if set(y[train])!={0,1}:raise ValueError("Missing class in training fold")
            weight=np.array([1/cells[s,t] for allowed,(s,g,t) in zip(train,meta) if allowed])
            weight*=len(weight)/weight.sum()
            reg=LogisticRegression(C=12.,random_state=61119,max_iter=500).fit(x[train],y[train],sample_weight=weight)
            pred=reg.predict(x[val]);mat=np.zeros((2,2),int);np.add.at(mat,(y[val],pred),1)
            arms["af_plus_woo" if woo else "af_only"]=mat.tolist()
        folds[group]=arms
    by_arm={name:np.sum([np.asarray(f[name]) for f in folds.values()],axis=0).tolist() for name in ("af_only","af_plus_woo")}
    io.atomic(OUT/"result.json",dict(folds=folds,pooled=by_arm,source_class_counts={f"{s}_{t}":n for (s,t),n in Counter((s,t) for s,g,t in meta).items()}))
    print("folds",folds,"pooled",by_arm,flush=True)


if __name__=="__main__":main()
