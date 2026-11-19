"""Test how informative the space outside an automatically detected fruit is for grade."""
from collections import Counter

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import afruit_grade_crop_alignment as af
import afruit_grade_train as io
from afruit_grade_model import file_sha


SOURCE = af.grade.ROOT / "datasets/grade_pairs_sources_20261119"
OUT = af.grade.ROOT / "experiments/grade_pairs_20261119/background_control"


def describe(frame, inside):
    rgb=frame[inside]
    return np.concatenate((np.quantile(rgb, [.1,.25,.5,.75,.9], axis=0).reshape(-1),
                           rgb.std(0),np.mean(rgb>240,axis=0), np.mean(rgb<30,axis=0)))


def partition(split):
    rows=[]
    excluded=Counter()
    for r in io.read(SOURCE/f"af_proposals_{split}.json")["samples"]:
        if r["grade"] not in (0,1):
            continue
        if r["crop"] is None or len(r["fruits"])!=1:
            excluded["no_unique_fruit"]+=1
            continue
        image=af.read_frame(r);h,w=r["hw"]
        frame=cv2.resize(image,(160,160),interpolation=cv2.INTER_AREA)
        x,y,right,bottom=r["crop"]
        # Enlarge the original padded fruit crop again before admitting outside
        # pixels, and round outward to reject mixed pixels at the boundary.
        padx=max(4,int(.03*(right-x)));pady=max(4,int(.03*(bottom-y)))
        left=int(np.floor((x-padx)/w*160));top=int(np.floor((y-pady)/h*160))
        right=int(np.ceil((right+padx)/w*160));bottom=int(np.ceil((bottom+pady)/h*160))
        outside=np.ones(frame.shape[:2],dtype=bool)
        outside[max(0,top):min(160,bottom),max(0,left):min(160,right)]=False
        if outside.sum()<256:
            excluded["insufficient_outside_pixels"]+=1
            continue
        rows.append((r,describe(frame,outside)))
    return rows,dict(excluded)


def main():
    if (OUT/"result.json").exists():raise FileExistsError(OUT)
    cv2.setNumThreads(1)
    protocol=dict(scope="Grade 1 and grade 2 photos with one resolved fruit, using only pixels outside its expanded padded fruit box. Background proxy; no claim that unseen adjacent fruit are absent.",
        split_sha={s:file_sha(SOURCE/f"af_proposals_{s}.json") for s in ("train","val")},
        input="24 per-channel colour summaries of exterior pixels; no position, time, image names, dimensions, grade probabilities, boxes or fruit-pixel features fed to model.",
        fit="One fixed class-balanced standardized logistic regression, C=1, on AF training photos. Validation decision fixed at 0.5.",
        limitations="Condition and grade can be correlated. A predictive exterior alone does not prove that a deployed neural model uses it; poor discrimination does not rule out source cues in fruit appearance.")
    io.atomic(OUT/"protocol.json",protocol)
    train,train_excluded=partition("train");val,val_excluded=partition("val")
    groups=lambda data:{r["group_id"] for r,_ in data}
    if groups(train)&groups(val):raise ValueError("Train and development share a recorded capture group")
    x=np.stack([f for _,f in train]);y=[r["grade"] for r,_ in train]
    learner=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight="balanced",random_state=119,max_iter=1000))
    learner.fit(x,y)
    score=learner.predict_proba(np.stack([f for _,f in val]))[:,1]
    matrix=np.zeros((2,2),int);rows=[]
    for (r,_),p in zip(val,score,strict=True):
        g=r["grade"];label=int(p>=.5);matrix[g,label]+=1
        rows.append(dict(path=r["path"],photo=r["sha256"],group=r["group_id"],truth=g,prediction=label))
    result=dict(training_groups=len(groups(train)),dev_groups=len(groups(val)),train_class_counts=dict(Counter(y)),
        excluded=dict(train=train_excluded,val=val_excluded),dev_confusion=matrix.tolist(),
        balanced_accuracy=float(np.mean(np.diag(matrix)/matrix.sum(1))),
        roc_auc=float(roc_auc_score([r["truth"] for r in rows],score)),rows=rows)
    io.atomic(OUT/"result.json",result)
    print({k:v for k,v in result.items() if k!="rows"},flush=True)


if __name__=="__main__":
    main()
