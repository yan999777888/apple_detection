"""Prepare previously unused-by-grader HDF captures for a guarded grading continuation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"datasets/grade_guard_hdf_v1"
MAN=ROOT/"diagnostics/hdf_peripheral_v1_s20261106/frozen_split_manifest.json"
SOURCE_TYPES=("fresh","bruise_defect","scab_defect","rot_defect")
SEED="hdf-grade-guard-20261118-v1"


def read(path):
    return json.loads(path.read_bytes())


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf8")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def rank(row):
    return digest(SEED+"|"+row["recommended_proxy_group_id"])


def image(row):
    with Image.open(row["image_path"]) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def signature(im):
    w,h=im.size
    frames=(im,im.crop((w//9,h//9,w-w//9,h-h//9)))
    parts=[]
    for view in frames:
        a=np.asarray(view.resize((16,16),Image.Resampling.BILINEAR),np.uint8)
        for k in range(4):
            v=np.rot90(a,k)
            parts.extend([v.copy(),v[:,::-1].copy()])
    return np.stack(parts)


def close(sig,bank):
    if not bank:
        return False
    # All orientations are stored on the reference side; test undistorted original and central field.
    for i in (0,8):
        a=np.asarray(bank,dtype=np.uint8)
        for start in range(0,len(a),96):
            dist=np.abs(a[start:start+96].astype(np.int16)-sig[i]).mean(axis=(-1,-2,-3))
            if np.min(dist)<11:
                return True
    return False


def prepare():
    rows=[r for r in read(MAN)["representatives"] if not r["quarantined"] and not r["source_label_conflict_family"]]
    denied_groups={r["recommended_proxy_group_id"] for r in rows if r["split"]!="train"}
    denied={}
    for r in sorted(rows,key=lambda x:x["sha256"]):
        if r["split"]!="train":
            denied.setdefault(r["recommended_proxy_group_id"],r)
    bank=[signature(image(r)) for r in denied.values()]
    selected=[]
    seen_groups=set()
    stats={}
    for name,limit in zip(SOURCE_TYPES,(132,150,150,100),strict=True):
        candidates=sorted((r for r in rows if r["author_class"]==name and r["split"]=="train"),key=lambda r:(rank(r),r["sha256"]))
        counts=Counter()
        for r in candidates:
            g=r["recommended_proxy_group_id"]
            if g in denied_groups or g in seen_groups:
                continue
            seen_groups.add(g)
            sig=signature(image(r))
            if close(sig,bank):
                counts["quarantined_visually_close"]+=1
                continue
            bank.append(sig)
            selected.append(dict(path=Path(r["image_path"]).relative_to(ROOT).as_posix(),sha256=r["sha256"],
                                 group_id=g,source_label=name,index=counts["saved"]))
            counts["saved"]+=1
            if counts["saved"]==limit:
                break
        stats[name]=dict(counts)
    save(OUT/"screen.json",dict(previous_heldout_proxy_groups=len(denied_groups),rules="Group-disjoint, lowest sha representative, then d4 original/central 16x16 RGB MAE<11 excluded in both older heldout and newly selected crops; unavailable physical fruit IDs.",counts=stats,rows=selected))
    sheet()


def sheet():
    rows=read(OUT/"screen.json")["rows"]
    for name in SOURCE_TYPES:
        subset=[x for x in rows if x["source_label"]==name]
        for start in range(0,len(subset),35):
            block=subset[start:start+35]
            canvas=Image.new("RGB",(7*182,5*166),(240,240,240));draw=ImageDraw.Draw(canvas)
            for i,r in enumerate(block):
                with Image.open(ROOT/r["path"]) as orig:
                    im=ImageOps.exif_transpose(orig).convert("RGB"); im.thumbnail((178,144))
                x=i%7*182;y=i//7*166;canvas.paste(im,(x,y))
                draw.text((x,y+144),f"{name} {r['index']:03d}",fill="black")
            canvas.save(OUT/f"{name}_{start:03d}.jpg",quality=89)


def finish():
    """Curate visible extents before viewing any new model predictions."""
    import penn_surface_challenge as penn
    from zipfile import ZipFile
    screen=read(OUT/"screen.json")["rows"]
    fresh={
        "pale_lenticels":("train",[0,1,5,6,9,12,18,24,26]),
        "burgundy_cool":("train",[2,3,15,19,20,21,22,27,28,30,34,38]),
        "pale_gradient":("train",[7,10,31,33,45,49]),
        "upright_stripes":("train",[11,14,37,47]),
        "horizontal_stripes":("val",[4,32,43]),
        "smooth_bicolour":("val",[8,13,23,25]),
        "smooth_burgundy":("test",[29,35,50]),
        "calyx_bi_red":("test",[16,36,40,41,42,44,46])}
    # Keep only one view per recognisable public stock-photograph family.
    # Bruise, scab and rot folders originally describe defect types without a severity grade.
    mild={"bruise_defect":[15,24,98,130],"scab_defect":[11,33,34,54,116,120,122,123]}
    advanced=[0,1,2,3,4,5,7,8,9,10,13,15,16,18,20,21,22,24,25,26,29,30,32,33,34,40,41,42,43,
              44,46,48,49,50,51,52,56,57,59,60,61,62,68,71,74,75,78,84,86,87,88,90,91,94,95]
    # A coarse visual family ID prevents obviously derived views entering different folds.
    duplicates=[(13,20),(15,36),(21,79),(22,40),(25,10),(30,46,49),(34,52),(42,70),(48,3,74),
                (50,51),(56,87),(59,85),(13,92),(24,2,44),(5,78),(71,84)]
    rows=[]
    for source in screen:
        name,i=source["source_label"],source["index"]
        row=dict(source); row["source"]="HDF_real"; row["prior_group_id"]=row["group_id"]
        if name=="fresh":
            assignment=[(family,sp) for family,(sp,ids) in fresh.items() if i in ids]
            if not assignment:
                continue
            family,sp=assignment[0]
            row.update(grade=0,split=sp,group_id="hdf_fresh_family_"+family,train_target="author_fresh_visible")
        elif name=="rot_defect" and i in advanced:
            family=next((min(t) for t in duplicates if i in t),i)
            row.update(grade=2,group_id=f"hdf_rot_family_{family:03d}",train_target="author_rot_plus_visible_extensive_affected_or_shrunken_surface")
        elif i in mild.get(name,[]):
            row.update(grade=1,group_id=f"hdf_{name}_view_{i:03d}",train_target="author_defect_plus_nonexpert_localised_shallow_visible_proxy")
        else:
            continue
        rows.append(row)
    # Take one audited example per family to avoid inflating apparent severity support.
    distinct={}
    for r in rows:
        if r["grade"] in (1,2):distinct.setdefault(r["group_id"],r)
    rows=[r for r in rows if r["grade"]==0]+list(distinct.values())
    for grade in (1,2):
        part=sorted([r for r in rows if r["grade"]==grade],key=lambda r:digest(SEED+"|holdout|"+r["group_id"]))
        n=max(2,round(len(part)/5))
        for j,row in enumerate(part):row["split"]="val" if j<n else "test" if j<2*n else "train"
    old_psu=read(penn.OUT/"manifest.json")
    previous_days=set(old_psu["excluded_complete_dates"])
    old_holdout_days={r["timestamp"][:10] for r in old_psu["samples"]}
    if previous_days & old_holdout_days:
        raise ValueError("PSU grading test dates would be reused")
    with ZipFile(penn.ARCHIVE) as z:
        all_days={penn.sample_time(r.filename).date().isoformat() for r in z.infolist() if r.filename.endswith("jpg")}
        for r in penn.candidates(z,all_days-previous_days):
            if r["grade"]!=1:continue
            raw=z.read(r["original_member"])
            frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            x,y,right,bottom=r["source_box"]; crop=frame[y:bottom,x:right]
            blob=cv2.imencode(".png",crop)[1].tobytes(); hs=hashlib.sha256(blob).hexdigest()
            path=OUT/"penn"/(hs[:20]+".png");path.parent.mkdir(exist_ok=True);path.write_bytes(blob)
            day=r["timestamp"][:10]
            sp={"2024-10-10":"val","2024-10-30":"test"}.get(day,"train")
            rows.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=hs,group_id="psu_"+day,split=sp,
                             source_label="author_fruit_and_lesion_polygons",source="PSU_training_dates",grade=1,
                             train_target="visually_confirmed_small_lesion_proxy",source_sha256=hashlib.sha256(raw).hexdigest(),
                             original_member=r["original_member"],source_box=r["source_box"],review_index=r["review_index"]))
    splits=defaultdict(set)
    for r in rows:splits[r["group_id"]].add(r["split"])
    if any(len(x)!=1 for x in splits.values()) or len(rows)!=len({r["sha256"] for r in rows}):
        raise ValueError("Duplicate content or split leakage")
    audit=dict(source_manifest_sha256=hashlib.sha256(MAN.read_bytes()).hexdigest(),existing_hdf_eval_allowed=False,
               rule="HDF train-only coarse visual family split plus previously-used PSU detector dates. Previous Penn challenge dates excluded. AI visual mapping of damage extent, no expert quality ground truth. Source-stock identities beyond visual screening unknown.",
               counts={sp:{str(c):sum(r["split"]==sp and r["grade"]==c for r in rows) for c in range(3)} for sp in ("train","val","test")},
               samples=rows)
    target=OUT/"manifest.json"
    if target.exists() and read(target)!=audit:raise ValueError("Frozen guard manifest differs")
    save(target,audit)
    print(audit["counts"])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","sheet","finish"])
    args=parser.parse_args()
    cv2.setNumThreads(1)
    {"prepare":prepare,"sheet":sheet,"finish":finish}[args.action]()
