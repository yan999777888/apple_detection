"""Read curator-confirmed native coordinates; retain matching source/capture IDs."""
from collections import defaultdict
import cv2
import numpy as np

import afruit_grade_model as grade
import afruit_grade_train as io
from grade_pairs_20261119 import extract_patch


FILE=grade.ROOT/"datasets/grade_pairs_sources_20261119/regions.json"


def read_patches():
    table=io.read(FILE)["regions"]
    by=defaultdict(list)
    targets={"surface_defect":1,"confirmed_normal":0}
    for r in table:
        if r["target"] in targets and r["split"] in ("train","val","dev"):
            by[r["path"]].append(r)
    pics=[];labels=[];meta=[]
    partitions={}
    for path,annotations in sorted(by.items()):
        full=grade.ROOT/path
        if any(r["image_sha256"]!=grade.file_sha(full) for r in annotations):
            raise ValueError("Native patch source changed")
        image=cv2.imdecode(np.fromfile(full,np.uint8),cv2.IMREAD_COLOR)
        for r in annotations:
            h,w=image.shape[:2]
            if list((h,w))!=r["image_hw"]:raise ValueError("Native patch grid changed")
            x,y,right,bottom=r["source_image_xyxy"]
            if not (0<=x<right<=w and 0<=y<bottom<=h):raise ValueError(f"Bad patch {r['id']}")
            area=r.get("parent_fruit_xyxy") or [0,0,w,h]
            diameter=max(7,min(area[2]-area[0],area[3]-area[1]))
            span=float(np.clip(1.65*max(right-x,bottom-y),.12*diameter,.40*diameter))
            crop=extract_patch(image,(x+right)/2,(y+bottom)/2,span)
            subset="train" if r["split"]=="train" else "dev"
            group=r["source"],r["group_id"]
            if partitions.setdefault(group,subset)!=subset:raise ValueError("Physical/capture group overlaps patch development")
            label=targets[r["target"]]
            for flip in (False,True) if subset=="train" else (False,):
                pics.append(np.ascontiguousarray(crop[:,::-1]) if flip else crop)
                labels.append(label)
                meta.append(dict(id=r["id"],split=subset,source=r["source"],label=label,subtype=r["subtype"],
                                 group=r["group_id"],pair_group=path,path=path,original_xyxy=r["source_image_xyxy"],flipped=flip))
    return pics,labels,meta
