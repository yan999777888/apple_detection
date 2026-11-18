"""Freeze out-of-training-day PSU orchard views with visible small lesions or pronounced rot."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"datasets/penn_surface_challenge_v1"
ARCHIVE=ROOT/"datasets/public_defect_intake/Fruit_defects.v2-severity_val.coco-segmentation.zip"
ARCHIVE_SHA="14f3879657d6a251a6d8707a421b8e91d5c33fa8a05ad6c940a296dff0daa56a"
SNAP=ROOT/"experiments/defect_public_intake/snapshots"
VISIBLE_MINOR={9,12,14,16,29,36,42,43,52,53,56,57,58,59,60,62,65,66,67}
VISIBLE_ADVANCED_ROT={0,1,2,5,8}
MILD_TYPES={"Sting","Plum Curculio","Apple Scab"}


def read(path):
    return json.loads(path.read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sample_time(member):
    return datetime.strptime(Path(member).name[:19],"IMG_%Y%m%d_%H%M%S")


def polygon_mask(polygons,h,w):
    result=np.zeros((h,w),np.uint8)
    cv2.fillPoly(result,[np.array(p).reshape(-1,2).round().astype(np.int32) for p in polygons],1)
    return result


def candidates(z,used_days):
    d=json.loads(z.read("train/_annotations.coco.json"))
    cat={c["id"]:c["name"] for c in d["categories"]}
    grouped=defaultdict(list)
    for a in d["annotations"]:
        grouped[a["image_id"]].append(a)
    accepted=[]
    idx=-1
    for im in sorted(d["images"],key=lambda r:r["file_name"]):
        member="train/"+im["file_name"]
        ts=sample_time(member)
        for fruit in [a for a in grouped[im["id"]] if cat[a["category_id"]]=="Apple"]:
            idx+=1
            if ts.date().isoformat() in used_days or idx not in VISIBLE_MINOR|VISIBLE_ADVANCED_ROT:
                continue
            fg=polygon_mask(fruit["segmentation"],im["height"],im["width"])
            paired=[]
            for lesion in grouped[im["id"]]:
                if cat[lesion["category_id"]]=="Apple":
                    continue
                m=polygon_mask(lesion["segmentation"],*fg.shape)
                if (fg*m).sum()>.5*m.sum():
                    paired.append(lesion)
            category={cat[a["category_id"]] for a in paired}
            area=sum(a["area"] for a in paired)/fruit["area"]
            grade=1 if idx in VISIBLE_MINOR else 2
            if grade==1 and (not category or not category<=MILD_TYPES or len(paired)>2 or area>.04):
                continue
            if grade==2 and (category!={"Rot"} or area<.10):
                continue
            h,w=fg.shape
            x,y,b,t=fruit["bbox"]
            box=[max(0,int(x-.1*b)),max(0,int(y-.1*t)),min(w,int(x+1.1*b)),min(h,int(y+1.1*t))]
            accepted.append(dict(original_member=member,grade=grade,timestamp=ts.isoformat(),group_id="PSU_day_"+ts.date().isoformat(),
                                 original_fruit_annotation=fruit["id"],source_box=box,
                                 original_lesions=[dict(id=a["id"],name=cat[a["category_id"]],area=a["area"]) for a in paired],
                                 marked_fraction=area,review_index=idx))
    if idx!=75:
        raise ValueError("Unexpected original fruit count")
    # One chronologically first eligible view in any same-day contiguous 120-second burst.
    accepted.sort(key=lambda r:(r["timestamp"],r["original_member"]))
    kept=[]
    previous=None
    for r in accepted:
        tm=datetime.fromisoformat(r["timestamp"])
        if previous is None or (tm-previous).total_seconds()>120:
            kept.append(r)
        previous=tm
    return kept


def signature(image):
    rgb=cv2.resize(image,(32,32),interpolation=cv2.INTER_AREA)
    dct=cv2.dct(cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY).astype(np.float32))[:8,:8]
    return int.from_bytes(np.packbits(dct>np.median(dct)).tobytes(),"big"),rgb


def two_views(image):
    h,w=image.shape[:2]
    return [image,image[h//6:h-h//6,w//6:w-w//6]]


def reference_paths():
    base=ROOT/"apple_research_v2/datasets"
    for name in ("mendeley_stage_seg_v1","papple_visible_stage_seg_v1"):
        for p in sorted((base/name/"images/train").iterdir()):
            if p.suffix.lower() in (".jpg",".jpeg",".png"):
                yield name,p
    for path in (base/"ssda_date_v2/splits/train.txt").read_text(encoding="utf-8-sig").splitlines():
        if path.strip():
            yield "ssda_date_v2",Path(path)
    hdf=read(ROOT/"diagnostics/hdf_peripheral_v1_s20261106/frozen_split_manifest.json")["representatives"]
    for r in hdf:
        if r["split"]=="train" and not r["quarantined"]:
            yield "HDF",Path(r["image_path"])
    for name in ("afruitdb_apples_v1","apple_matched_quality_v1"):
        for r in read(ROOT/"datasets"/name/"manifest.json")["samples"]:
            if r["split"]=="train":
                yield name,ROOT/r["path"]
    for p in (SNAP/"75e57d602dc84195b8d705105909e37c/source_images").glob("*"):
        if p.is_file():
            yield "published_defect_intake",p


def screen(rows,images):
    signatures=[]
    hash_to_index={}
    for i,(source,fruit,raw,crop_hash) in enumerate(images):
        hash_to_index[raw]=hash_to_index[crop_hash]=i
        for im in (source,fruit):
            for view in two_views(im):
                for k in range(4):
                    rot=np.rot90(view,k)
                    for tr in (rot,rot[:,::-1]):
                        sig,pixels=signature(tr)
                        signatures.append((i,sig,pixels))
    counters=Counter()
    excluded=set()
    candidate_edges=[]
    for origin,p in reference_paths():
        counters[origin]+=1
        data=p.read_bytes()
        same=hash_to_index.get(sha(data))
        if same is not None:
            excluded.add(same)
        frame=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_REDUCED_COLOR_4)
        if frame is None:
            raise ValueError(f"Unreadable prior input {p}")
        for view in two_views(frame):
            hx,rgb=signature(view)
            for i,ref,ref_pixels in signatures:
                if (hx^ref).bit_count()<=8:
                    err=np.abs(rgb.astype(np.float32)-ref_pixels).mean()
                    if err<18.0:
                        candidate_edges.append([i,origin,p.name,float(err)])
                        excluded.add(i)
    eligible=[(r,im) for i,(r,im) in enumerate(zip(rows,images,strict=True)) if i not in excluded]
    return eligible,dict(references=dict(counters),denied_source_or_crop_matches=candidate_edges,
                         excluded_count=len(excluded),method="SHA-256 plus full/central RGB thumbnails; mirrored and rotated source + fruit crop, pHash<=8 and mean 8-bit color difference<18. Also exclude every source date previously used by the defect project.")


def run():
    if sha(ARCHIVE.read_bytes())!=ARCHIVE_SHA:
        raise ValueError("Unexpected official archive")
    previous_files=sorted(SNAP.glob("*/positives.json"))
    previous={r["image_member"] for f in previous_files for r in read(f)
              if "penn-state-university" in r["source_id"]}
    used_days={sample_time(p).date().isoformat() for p in previous}
    with ZipFile(ARCHIVE) as z:
        rows=candidates(z,used_days)
        images=[]
        for r in rows:
            data=z.read(r["original_member"])
            source=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            x,y,u,v=r["source_box"]
            crop=source[y:v,x:u]
            ok,png=cv2.imencode(".png",crop)
            if not ok:
                raise IOError(r["original_member"])
            images.append((source,crop,sha(data),sha(png.tobytes())))
    accepted,audit=screen(rows,images)
    (OUT/"images").mkdir(parents=True,exist_ok=True)
    final=[]
    for r,(_,crop,original,hs) in accepted:
        path=OUT/"images"/(hs[:18]+".png")
        data=cv2.imencode(".png",crop)[1].tobytes()
        if sha(data)!=hs:
            raise IOError("Non-reproducible crop")
        path.write_bytes(data)
        final.append(dict(**r,path=path.relative_to(ROOT).as_posix(),sha256=hs,source_sha256=original,
                          source="PSU_orchard_visible_lesion_proxy",condition_id=r["group_id"],split="external_test"))
    record=dict(source="Penn State University, Fruit_defects v2 severity_val; CC BY 4.0",
                source_url="https://universe.roboflow.com/penn-state-university/fruit_defects/dataset/2",
                archive_sha256=ARCHIVE_SHA,unrelated_to_grade_classifier_training=True,
                earlier_defect_training_related_source_images=len(previous),excluded_complete_dates=sorted(used_days),
                prior_snapshots={f.parent.name:sha(f.read_bytes()) for f in previous_files},screen=audit,
                severity="Image crop with original fruit and lesion polygons. Small superficial visible lesion or obvious rot confirmed by nonexpert visual inspection. These degree mappings are project proxies, not author's grades. No healthy comparator.",
                independence="Apple grade training used other sources. Original detector previously learned from parts of this PSU dataset. Source dates and close appearance matches against known prior imagery are excluded. Orchard/site/physical identities are not established as independent.",
                transform="Original author Apple box plus 10% each side clipped to frame; author area fraction uses polygon area. Retain first qualifying image per <=120s contiguous temporal component.",samples=final)
    (OUT/"manifest.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf8")
    print(Counter(r["grade"] for r in final),audit,used_days,flush=True)


if __name__=="__main__":
    cv2.setNumThreads(1)
    run()
