"""Nearest annotated examples and pixel-scale case sheets for the fixed arbitration head."""

from collections import Counter
from pathlib import Path
from functools import lru_cache
import json
import math
import numpy as np
from PIL import Image,ImageDraw
from class0_patch_refinement import ROOT, OLD, manifest, configure, read, atomic_json, load_image
from compare_defect_target9485 import PHASE, indexed, load_candidates
from defect_continue_teacher_proxy import OUT as HEAD
from defect_continue_teacher import describe
from defect_coordinate_vote_20261118 import vote
from audit_defect_causes_20261112 import ious, match

OUT = ROOT/"experiments/defect_continue_stratified"


def data(sid, b, cls, index, role, im):
    w, h = np.maximum(1,np.asarray(b)[2:4]-b[:2])
    # Subtract the ring means already encoded by describe(). Omit absolute position
    # and the two log-size features from the photometric nearest-neighbor distance.
    return dict(sid=sid, camera=sid.split("_")[0], box=list(map(float,b[:4])), cls=int(cls),idx=int(index),
                px=float(np.sqrt(w*h)),kind=role,feat=describe(im,b) [2:].astype(np.float32))


def closest(q, pool, cls, same_image=False):
    eligible = [r for r in pool if r["camera"]==q["camera"] and r["cls"]==cls and (same_image or r["sid"]!=q["sid"])]
    if not eligible:
        return None
    # Both box spans must match. Use the first supported caliper and disclose expansions.
    w0 = np.maximum(1,np.subtract(q["box"][2:],q["box"][:2]))
    wh = np.maximum(1,np.asarray([np.subtract(r["box"][2:],r["box"][:2]) for r in eligible]))
    delta = abs(np.log(wh / w0)).max(1)
    for factor in (1.35,1.6,2.,3.,float("inf")):
        good = np.where(delta <= np.log(factor))[0]
        if len(good):break
    cand = [eligible[i] for i in good]
    zv = np.stack([r["z"] for r in cand])
    d = np.mean(np.minimum((zv-q["z"])**2,12.),axis=1)
    # Limited geometry contribution even after the caliper.
    d = np.sqrt(d+.15*delta[good]**2)
    i = int(np.argmin(d))
    v = cand[i]
    return dict(sid=v["sid"],idx=v["idx"],box=v["box"],cls=v["cls"],distance=float(d[i]),
                span_ratio=float(np.exp(delta[good[i]])),eligible=len(good))


def summary_cases():
    from defect_algorithm_modes_20261112 import nms
    import cv2
    cv2.setNumThreads(1)
    files = manifest()
    train = []
    for sid,r in files.items():
        if r["partition"]!="train":continue
        im = load_image(r["rgb_path"])
        for i,v in enumerate(r["instances"]):
            x,y,w,h = v["xywh_normalized"]
            b = np.array([x-w/2,y-h/2,x+w/2,y+h/2])*1024
            train.append(data(sid,b,v["class_id"],i,"train",im))
    assert len(train)==1276
    src = indexed(read(PHASE))
    head = load_candidates(HEAD/"tradeoff_predictions.json",src)
    small = load_candidates(ROOT/"experiments/defect9485_small_v1/low_predictions.json",src)
    prior = read(HEAD/"error_ledger.json")["original_class0_misses"]
    six = {(r["image"],r["index"]):r["code"] for r in prior if r["small_hit"] and not r["new_fixed_hit"]}
    assert len(six)==6
    queries = []
    fps,tps = 0,0
    for sid in sorted(src):
        r=src[sid]
        im=load_image(files[sid]["rgb_path"])
        prim=vote(r["sources"],.5,[.008,.008])
        pairs,ov=match(head[sid],r["GT"])
        for i,b in enumerate(head[sid]):
            def supported(rows):
                pool=rows[(rows[:,5]==b[5]) & (rows[:,4]>=.08)]
                return len(pool) and ious([b[:4]],pool[:,:4]).max()>=.5
            if not supported(prim) or not supported(small[sid]):continue
            ok=i in pairs
            new=data(sid,b[:4],int(b[5]),i,"shared_TP" if ok else "shared_FP",im)
            if ok:tps+=1
            else:fps+=1
            if len(r["GT"]):
                near=int(ov[i].argmax())
                new.update(nearest_gt_iou=float(ov[i,near]),nearest_gt_class=r["GT"][near]["class"])
            else:new.update(nearest_gt_iou=0.,nearest_gt_class=None)
            new.update(code=(f"T{tps:03}" if ok else f"F{fps:02}"),hit_gt=pairs.get(i),score=float(b[4]))
            queries.append(new)
        for j,g in enumerate(r["GT"]):
            if (sid,j) in six:
                query=data(sid,g["box"],g["class"],j,"miss",im)
                query.update(code=six[sid,j],hit_gt=j)
                queries.append(query)
    assert (fps,tps)==(50,187)
    mat=np.stack([v["feat"] for v in train])
    med=np.median(mat,axis=0)
    scale=np.maximum(np.quantile(mat,.75,axis=0)-np.quantile(mat,.25,axis=0),1e-3)
    for r in train+queries:r["z"]=np.clip((r["feat"]-med)/scale,-7,7)
    true=[r for r in queries if r["kind"]=="shared_TP"]
    for r in queries:
        c=r["cls"]
        r.update(annotated_same=closest(r,train,c),annotated_other=closest(r,train,1-c),correct_dev=closest(r,true,c))
    for r in queries:
        for k in ("feat","z"):del r[k]
    atomic_json(OUT/"cases.json",queries)
    return queries


@lru_cache(maxsize=18)
def photo(path):
    with Image.open(path) as im:return im.convert("RGB")


def crop(im,box,side):
    x=int(round((box[0]+box[2])/2));y=int(round((box[1]+box[3])/2))
    # Pad outside of an image in neutral gray, center the query in each crop.
    l=x-side//2;t=y-side//2
    xy=(max(0,l),max(0,t),min(im.width,l+side),min(im.height,t+side))
    tile=Image.new("RGB",(side,side),(130,130,130))
    tile.paste(im.crop(xy),(xy[0]-l,xy[1]-t))
    return tile.resize((128,128),Image.Resampling.LANCZOS)


def gallery(queries):
    files=manifest()
    ordering=sorted([q for q in queries if q["kind"]!="shared_TP"],key=lambda r:(r["kind"]!="miss",r["code"]))
    for start in range(0,len(ordering),7):
        group=ordering[start:start+7]
        page=Image.new("RGB",(676,26+155*len(group)),(248,248,248));d=ImageDraw.Draw(page)
        for j,title in enumerate(("Whole + box","Dev query","Train same ID","Train OTHER ID","Dev correct same")):
            d.text((3+134*j,5),title,fill="black")
        for i,q in enumerate(group):
            top=27+i*155
            sid=q["sid"];im=photo(files[sid]["rgb_path"])
            box=q["box"];whole=im.resize((128,128))
            ImageDraw.Draw(whole).rectangle([round(v/8) for v in box],outline="#00e5ff",width=2)
            page.paste(whole,(2,top))
            side=max(88,round(max(box[2]-box[0],box[3]-box[1])*1.65))
            for j,r in enumerate([q,q["annotated_same"],q["annotated_other"],q["correct_dev"]],1):
                if r:
                    tile=crop(photo(files[r["sid"]]["rgb_path"]),r["box"],side)
                    left=2+134*j
                    page.paste(tile,(left,top))
                    if j>=2:d.text((left,top+130),f"C{r['cls']} span x{r['span_ratio']:.2f}",fill="black")
            d.text((2,top+130),f"{q['code']} C{q['cls']} cam{sid[6:8]}",fill="black")
            d.text((136,top+130),f"width {side}px",fill="black")
        page.save(OUT/f"sheet_{1+start//7:02}.jpg",quality=93)


def threshold_diagnostic(rows):
    result={}
    for name,ff in [("positive_distance",lambda r:r["annotated_same"]["distance"]),
                    ("other_vs_positive",lambda r:r["annotated_same"]["distance"]-r["annotated_other"]["distance"])]:
        a=sorted([(ff(r),r["kind"]) for r in rows if r["kind"]!="miss"],reverse=True)
        tp=fp=0;details=[]
        for i,(val,label) in enumerate(a):
            tp+=label=="shared_TP";fp+=label=="shared_FP"
            if i+1==len(a) or a[i+1][0]!=val:details.append((int(fp),int(tp)))
        result[name]={"most_FP_removed_losing_at_most_3_TP":max(f for f,t in [(0,0)]+details if t<=3),
                      "fewest_TP_lost_removing_at_least_30_FP":min(t for f,t in details if f>=30)}
    return result


if __name__=="__main__":
    configure(OUT,"cpu")
    rows=summary_cases()
    gallery(rows)
    stats=threshold_diagnostic(rows)
    atomic_json(OUT/"appearance_diagnostic.json",stats)
    print(stats,flush=True)
