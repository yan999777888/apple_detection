"""Compare fixed grading models on external labelled fruit views kept out of fitting."""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

import afruit_grade_model as grade
from afruit_grade_train import atomic, read
from matched_grade_loader import load, tensor_at_size


def photos_and_labels():
    root = grade.ROOT
    fruit_dir = root / "datasets/public_defect_intake/fruitlife_v1"
    owners = read(fruit_dir / "physical_id_split.json")["assignment"]
    fruit = read(fruit_dir / "apple_manifest.json")["samples"]
    allowed = {r["specimen_id"] for r in fruit if owners[r["specimen_id"]] == "external_test"}
    external = [dict(**r, panel="FruitLife", target=f'stage_{r["author_stage"]}',
                     group=r["specimen_id"]) for r in fruit if r["specimen_id"] in allowed]
    if len(allowed) != 3 or len(external) != 180 or len({r["sha256"] for r in external}) != 180:
        raise ValueError("Physical-fruit test panel changed")
    declared=read(root/"datasets/apple_matched_fruitlife_woo_v1/external_test.json")["samples"]
    if {r["sha256"] for r in declared}!={r["sha256"] for r in external}:
        raise ValueError("External photos do not match the permanently reserved set")
    historical = read(root / "experiments/afruit_external_appearance_v1/protocol.json")["cases"]
    external.extend(dict(**r, panel="HDF",target=r["source_label"]) for r in historical)
    penn = read(root / "datasets/penn_surface_challenge_v1/manifest.json")["samples"]
    external.extend(dict(**r,panel="PennState",target=f'visible_grade_{r["grade"]}',group=r["group_id"]) for r in penn)
    return external


@torch.inference_mode()
def evaluate(models, rows):
    crop_names = [r["sha256"] for r in rows]
    if len(set(crop_names)) != len(rows):
        raise ValueError("Duplicate evaluation images")
    for i in range(0,len(rows),16):
        batch=rows[i:i+16]
        images=[]
        for r in batch:
            path=grade.ROOT/r["path"]
            if grade.file_sha(path)!=r["sha256"]:
                raise ValueError(f"Source changed: {path.name}")
            if r["panel"]=="HDF":
                with Image.open(path) as im:
                    frame=np.asarray(ImageOps.exif_transpose(im).convert("RGB"))[:,:,::-1].copy()
            else:
                frame=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
            images.append(frame)
        for name,(model,cutoff,side) in models.items():
            inp=tensor_at_size(images,"cpu",side)
            labels=grade.decide(model(inp),cutoff).tolist()
            for r,y in zip(batch,labels,strict=True):
                r[name]=int(y)


def summarize(rows, names):
    by_panel={}
    for panel in sorted({r["panel"] for r in rows}):
        subset=[r for r in rows if r["panel"]==panel]
        report={}
        for name in names:
            totals=defaultdict(lambda:[0,0,0])
            per_fruit=defaultdict(lambda:defaultdict(lambda:[0,0,0]))
            for r in subset:
                totals[r["target"]][r[name]]+=1
                if panel=="FruitLife":
                    per_fruit[r["group"]][r["target"]][r[name]]+=1
            report[name]=dict(totals=totals,per_fruit=per_fruit)
        by_panel[panel]=report
    return by_panel


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    if (args.output/"results.json").exists():
        raise ValueError("Completed panel already exists")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    sources=dict(original=grade.ROOT/"experiments/afruit_grade_release_v1",candidate=args.candidate)
    models={name:load(path,"cpu") for name,path in sources.items()}
    rows=photos_and_labels()
    evaluate(models,rows)
    report=dict(models={k:grade.file_sha(v/"weights.pt") for k,v in sources.items()},
                scope="CPU FP32 whole provided fruit inputs; FruitLife physical holdout crops by author masks; HDF uses historical real development panel and binary source labels.",
                prediction_columns=["good","mild","severe"],panels=summarize(rows,sources))
    atomic(args.output/"predictions.json",rows)
    atomic(args.output/"results.json",report)
    for name,panel in report["panels"].items():
        print(name,{k:v["totals"] for k,v in panel.items()})
