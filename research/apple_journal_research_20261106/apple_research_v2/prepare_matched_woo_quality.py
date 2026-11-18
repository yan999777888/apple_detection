"""Prepare manually screened single-face grade proxies from Woo's real, camera-matched apples."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets/apple_matched_quality_v1"
ZIP = ROOT / "datasets/public_defect_intake/orchard-apple-defect-segmentatio.v1-orchard_apple_defect_seg_v1.coco-segmentation.zip"
SHA = "b32b8203a7b92e2149dddbe5c20038b9a292b7c711f166b6a90b66ef54ac02c8"
PRIOR = ROOT / "experiments/defect_public_intake/snapshots/75e57d602dc84195b8d705105909e37c/image_decisions.json"
SOURCE = "https://universe.roboflow.com/woo-wwyog/orchard-apple-defect-segmentatio/dataset/1"
# Reviewed on the camera views and fruit close-ups, before any grade model inference.
# Indices resolve against sorted (train, timestamp, camera, filename) of train-only timestamps.
# Grade 0 requires no author lesion AND a visibly intact face. Grade 1 requires only
# limited superficial scratches; holes/calyces, open cuts, bruising and ambiguous cases are omitted.
CLEAN = {0,2,3,5,6,9,11,13,14,18,19,21,24,26,28,29,30,32,33,36,37,38,39,40,41,43,44,
         67,68,69,71,74,77,78,82,87,93,94,95,99,105,*range(109,139)}
MILD = {45,46,48,49,50,51,54,56,57,60,61,62,63,86,89,92,96,98,101,104}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(name, data):
    file = OUT / name
    temp = file.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(file)


def descriptor(frame):
    """Color-select the isolated fruit. The crop retains 10% context, including its real background."""
    h,w=frame.shape[:2]
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=((hsv[:,:,1]>32)&((hsv[:,:,0]<46)|(hsv[:,:,0]>158))&(frame[:,:,2]>58)).astype("uint8")
    mask[:round(h*.13)]=0
    mask[round(h*.89):]=0
    mask[:,:round(w*.04)]=0
    mask[:,round(w*.96):]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    n,_,st,_=cv2.connectedComponentsWithStats(mask)
    if n<2:
        return None
    x,y,b,t,area=map(int,st[1+np.argmax(st[1:,4])])
    if not (.19*w<b<.58*w and .26*h<t<.85*h and area>30000):
        return None
    box=(max(0,int(x-.1*b)),max(0,int(y-.1*t)),min(w,int(x+1.1*b)),min(h,int(y+1.1*t)))
    patch=frame[box[1]:box[3],box[0]:box[2]]
    key=cv2.resize(frame[y:y+t,x:x+b],(40,40),interpolation=cv2.INTER_AREA).astype(np.float32)
    return patch,key,box,area


def duplicate(image, other):
    # Tight spatial appearance check within an identical camera in addition to ancestry screening.
    for test in (other,other[:, ::-1]):
        mae=np.abs(image-test).mean()
        if mae<7.0 and np.corrcoef(image.ravel(),test.ravel())[0,1]>.982:
            return True
    return False


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    photo_dir=OUT/"woo_crops"
    photo_dir.mkdir(exist_ok=True)
    if digest(ZIP.read_bytes())!=SHA:
        raise ValueError("Unexpected author archive")
    prior=json.loads(PRIOR.read_bytes())
    prior={r["member"]:r for r in prior if r["project_id"].startswith("woo-")}
    held={r["group_id"] for r in prior.values() if r["raw_split"]!="train"}
    afruit={r["sha256"] for r in json.loads((ROOT/"datasets/afruitdb_apples_v1/manifest.json").read_bytes())["samples"]}
    frames=[]
    partitions=defaultdict(set)
    holdout_features=defaultdict(list)
    excluded=Counter()
    notes=[]
    samples=[]
    previous=defaultdict(list)
    with ZipFile(ZIP) as z:
        for split in ("train","valid","test"):
            coco=json.loads(z.read(split+"/_annotations.coco.json"))
            an=defaultdict(list)
            for ann in coco["annotations"]:
                an[ann["image_id"]].append(ann)
            for im in coco["images"]:
                name=im["file_name"]
                stamp,side=name.split("_jpg.rf.")[0].rsplit("_",1)
                member=split+"/"+name
                partitions[stamp].add(split)
                frames.append((split,stamp,side,name,member,an[im["id"]]))
                if split!="train":
                    f=cv2.imdecode(np.frombuffer(z.read(member),np.uint8),cv2.IMREAD_COLOR)
                    result=descriptor(f)
                    if result is not None:
                        holdout_features[side].append(result[1])
        pure=sorted(row for row in frames if partitions[row[1]]=={"train"})
        if len(pure)!=139 or CLEAN & MILD:
            raise ValueError("The reviewed, ordered original set changed")
        for i,(split,stamp,side,name,member,annotations) in enumerate(pure):
            label=0 if i in CLEAN else 1 if i in MILD else None
            if label is None:
                excluded["visual_review_ambiguous_or_not_selected"]+=1
                continue
            old=prior[member]
            if old["group_id"] in held or "overlap" in old["decision"]:
                excluded["earlier_heldout_visual_or_name_family"]+=1
                continue
            if {ann["category_id"] for ann in annotations}!=({3} if label else set()):
                raise ValueError("Review conflicts with original annotation")
            payload=z.read(member)
            sha=digest(payload)
            if sha in afruit:
                excluded["AFruitDB_same_image"]+=1
                continue
            result=descriptor(cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR))
            if result is None:
                excluded["automatic_fruit_crop_geometry"]+=1
                continue
            crop,feature,box,area=result
            if any(duplicate(feature,ref) for ref in holdout_features[side]):
                excluded["fruit_crop_near_author_holdout"]+=1
                continue
            if any(duplicate(feature,ref) for ref in previous[side]):
                excluded["fruit_crop_near_accepted_train_view"]+=1
                continue
            fraction=sum(x["area"] for x in annotations)/area
            if label and fraction>.065:
                excluded["marked_region_over_6p5_percent"]+=1
                continue
            ok,img=cv2.imencode(".png",crop)
            if not ok:
                raise IOError(member)
            data=img.tobytes()
            path=photo_dir/(sha[:16]+".png")
            path.write_bytes(data)
            previous[side].append(feature)
            samples.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(data),grade=label,split="train",
                                source="Woo_CC_BY_4.0_visible_surface_proxy",source_member=member,source_sha256=sha,
                                condition_id="Woo_20260814_station_"+side,group_id="Woo_20260814_session",
                                capture_key=stamp,source_box=box,source_marked_fraction=fraction,
                                original_labels=["scratch"] if label else [],review_index=i,
                                label_basis="small author scratch polygons plus visible-face review" if label else
                                            "no author polygons plus visible-face review; no claim about unseen side"))
            notes.append([i,side,label])
    count=Counter((r["condition_id"],r["grade"]) for r in samples)
    # Only stations represented on both sides of the quality distinction may enter the matched contrast.
    samples=[r for r in samples if count[r["condition_id"],1-r["grade"]]]
    scope={"source_url":SOURCE,"license":"CC BY 4.0","attribution":"Woo (orchard-apple-defect-segmentatio v1), Roboflow Universe",
           "source_sha256":SHA,"prior_audit_sha256":digest(PRIOR.read_bytes()),
           "transform":"Color-localized real fruit crop with 10% context. Source lesion polygons support a proxy face grade; all 3 native views use the same station.",
           "heldout":"Author val/test and their timestamp families excluded; prior visual/name families and a new crop similarity screen excluded. Physical fruit identities were not released, apparent repeated apples appear across multiple timestamps. All accepted Woo examples belong to a single training session. Independent grading evaluation uses other data.",
           "review":"Author gives lesion types, not whole-fruit grades. Two-stage nonexpert AI visual review on full scene and unmarked enlarged crop. Intact-face is limited to the visible surface. No test predictions or grade model were used in selection.",
           "n_train_only_timestamp_views":len(pure),"n_train_only_timestamps":len({r[1] for r in pure}),
           "excluded":dict(excluded),"samples":samples}
    write_json("manifest.json",scope)
    summary={k:v for k,v in scope.items() if k!="samples"}
    summary.update(samples=len(samples),grade_counts=dict(Counter(r["grade"] for r in samples)),
                   by_camera={c:dict(Counter(r["grade"] for r in samples if r["condition_id"]==c)) for c in sorted({r["condition_id"] for r in samples})},
                   distinct_capture_keys=len({r["capture_key"] for r in samples}),
                   manifest_sha256=digest((OUT/"manifest.json").read_bytes()))
    write_json("source_audit.json",summary)
    print(summary["by_camera"],summary["excluded"],flush=True)


if __name__=="__main__":
    cv2.setNumThreads(1)
    run()
