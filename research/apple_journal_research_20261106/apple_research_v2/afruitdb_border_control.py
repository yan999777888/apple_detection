"""Counterfactual input diagnostic using peripheral pixels only; no model selection."""
import argparse
import hashlib
import json

import cv2
import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

from afruitdb_grade_source import ROOT, OUT, dump

RUN = ROOT/"experiments"/"afruitdb_border_control_v1"
SEED = 20261118


def feature(row):
    with Image.open(ROOT/row["path"]) as img:
        img.draft("RGB",(384,384))
        rgb=np.asarray(ImageOps.exif_transpose(img).convert("RGB").resize((128,128),Image.Resampling.BILINEAR))
    features=[]
    # Equally divided into four sides. No central pixels enter a feature.
    for side in (rgb[:8],rgb[-8:],rgb[8:-8,:8],rgb[8:-8,-8:]):
        pix=side.reshape(-1,3)/255.
        features.extend(np.quantile(pix,[.10,.25,.50,.75,.90],axis=0).flatten())
        features.extend(pix.std(0))
        gray=cv2.cvtColor(side,cv2.COLOR_RGB2GRAY)
        edge=np.abs(cv2.Laplacian(gray,cv2.CV_32F))
        features.extend(np.quantile(edge,[.25,.50,.75,.95]))
    return features


def appearance(model, examples, part):
    rows=[r for r in examples if r["split"]==part]
    x=np.array([feature(r) for r in rows])
    pred=model.predict(x)
    cm=confusion_matrix([r['grade'] for r in rows],pred,labels=[0,1,2])
    return dict(split=part,images=len(rows),confusion_true_by_pred=cm.tolist(),
                accuracy=float(cm.trace()/cm.sum()),balanced_accuracy=float(np.mean(cm.diagonal()/cm.sum(1))),
                severe_as_good=int(cm[2,0]),severe_images=int(cm[2].sum()))


def run(part):
    manifest=OUT/"manifest.json"
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows=json.loads(manifest.read_bytes())["samples"]
    if part=="fit":
        if (RUN/"validation.json").exists(): raise FileExistsError("Already frozen")
        train=[r for r in rows if r["split"]=="train"]
        model=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=3,class_weight="balanced",random_state=SEED,n_jobs=1)
        model.fit([feature(r) for r in train],[r["grade"] for r in train])
        RUN.mkdir(parents=True,exist_ok=True)
        joblib.dump(model,RUN/"trees.joblib")
        receipt=dict(manifest_sha256=digest,protocol="128 RGB; only outer 8-pixel perimeter (23.44% of pixels); fixed 200 ExtraTrees, 3 minimum samples per leaf. No tuning. Peripheral fruit may be present.",
                     train_images=len(train),train_groups=len({r["group_id"] for r in train}),
                     model_sha256=hashlib.sha256((RUN/"trees.joblib").read_bytes()).hexdigest(),
                     results=appearance(model,rows,"val"))
        dump(RUN/"validation.json",receipt)
    else:
        if (RUN/"test.json").exists(): raise FileExistsError("Test already evaluated")
        receipt=json.loads((RUN/"validation.json").read_bytes())
        assert receipt["manifest_sha256"] == digest
        assert receipt["model_sha256"] == hashlib.sha256((RUN/"trees.joblib").read_bytes()).hexdigest()
        receipt["results"]=appearance(joblib.load(RUN/"trees.joblib"),rows,"test")
        dump(RUN/"test.json",receipt)
    print(receipt,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase",choices=("fit","test"))
    cv2.setNumThreads(1)
    run(ap.parse_args().phase)
