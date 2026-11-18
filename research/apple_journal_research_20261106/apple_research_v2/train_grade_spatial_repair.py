"""Fit a localized mild-defect branch using fruit-level labels and training-only mark locations."""
import argparse
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import json

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import afruit_grade_crop_alignment as align
import afruit_grade_model as grade
import afruit_grade_train as base
from grade_spatial_repair import SpatialRepairGrade
from matched_grade_loader import load
from prepare_matched_woo_quality import ZIP


ROOT=grade.ROOT
WOO=ROOT/"datasets/apple_matched_quality_v1/manifest.json"
SEED=42091


def picture(row):
    p=ROOT/row["path"]
    if grade.file_sha(p)!=row["sha256"]:
        raise ValueError("Source photo changed")
    image=cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(str(p))
    return image


def map_region(polygons, box):
    x,y,r,b=box
    side=grade.SIDE
    scale=side/max(r-x,b-y)
    w,h=max(1,round((r-x)*scale)),max(1,round((b-y)*scale))
    full=np.zeros((side,side),np.uint8)
    for pts in polygons:
        points=(np.array(pts,dtype=float).reshape(-1,2)-[x,y])*[w/(r-x),h/(b-y)]+[(side-w)//2,(side-h)//2]
        cv2.fillPoly(full,[np.rint(points).astype(np.int32)],1)
    return full.reshape(28,8,28,8).max((1,3))


def rectangles(boxes):
    return [[x,y,r,y,r,b,x,b] for x,y,r,b in boxes if x<r and y<b]


def woo_marks():
    polygons=defaultdict(list)
    with ZipFile(ZIP) as z:
        ann=json.loads(z.read("train/_annotations.coco.json"))
        names={r["id"]:"train/"+r["file_name"] for r in ann["images"]}
        for a in ann["annotations"]:
            if a["category_id"]==3:
                polygons[names[a["image_id"]]].extend(a["segmentation"])
    return polygons


def dataset(train, annotations, guard):
    split="train" if train else "val"
    rows,squares,_=base.open_rows((split,))[split]
    crops=align.load_cache(split,rows)
    regions=base.read(align.OUT/f"{split}_regions.json")
    photos=[]; targets=[]; meta=[]
    empty=np.full((28,28),-1,np.int8)
    def add(square,grade_id,domain,group,view,key,mask=None):
        photos.append(square)
        targets.append(empty.copy() if mask is None else mask.copy())
        meta.append(dict(grade=int(grade_id),domain=domain,group=group,view=view,key=key))
    for r,im,fruit,region in zip(rows,squares,crops,regions,strict=True):
        h,w=region["hw"]
        annotation=annotations.get(r["path"])
        for name,square in (("whole",im),("auto",fruit)):
            if square is None:
                continue
            mask=None
            if train and annotation:
                sh,sw=annotation["source_hw"]
                roi=region["crop"] if name=="auto" else (0,0,w,h)
                original=np.array(roi)*[sw/w,sh/h,sw/w,sh/h]
                pos=map_region(rectangles(annotation["positives_xyxy"]),original)
                neg=map_region(rectangles(annotation["negatives_xyxy"]),original)
                mask=empty.copy()
                mask[neg>0]=0
                mask[pos>0]=1
            add(square,r["grade"],"af",r["group_id"],name,r["path"],mask)
    for r in base.read(guard)["samples"]:
        if r["split"]==split:
            if train and (r.get("source")=="HDF_real" or "hdf_real_only" in str(r["path"])):
                raise ValueError("HDF cannot be used for fitting; repeated physical fruit are uncertain")
            add(grade.rgb_square(picture(r)),r["grade"],"guard",r["group_id"],"provided",r["path"])
    if train:
        markings=woo_marks()
        for r in base.read(WOO)["samples"]:
            if r["split"]!="train":
                continue
            pos=map_region(markings[r["source_member"]],r["source_box"])
            if bool(pos.any())!=bool(r["grade"]):
                raise ValueError("Woo annotation/crop correspondence lost")
            add(grade.rgb_square(picture(r)),r["grade"],"woo",r["condition_id"],"provided",r["path"],pos)
    return photos,np.stack(targets),meta


@torch.inference_mode()
def cache(model, photos):
    maps=[];scores=[]
    for start in range(0,len(photos),24):
        feats,old=model.encode(grade.tensor_rgb(photos[start:start+24],"cuda"))
        maps.append(feats.half());scores.append(old)
    return torch.cat(maps),torch.cat(scores)


def evaluate(pred,meta):
    lookup={}; tables={}
    for name,domain,view in (("auto","af","auto"),("whole","af","whole"),("guard","guard","provided")):
        matrix=np.zeros((3,4),int)
        for i,r in enumerate(meta):
            if r["domain"]==domain and r["view"]==view:
                matrix[r["grade"],pred[i]]+=1
                lookup[(name,r["key"])]=int(pred[i])
        if name=="auto":
            matrix[:,3]=[25,15,17]-matrix.sum(1)
        tables[name]=matrix.tolist()
    return tables,lookup


def order(tables,reference):
    a,h=np.array(tables["auto"]),np.array(tables["guard"])
    p,q=np.array(reference["auto"]),np.array(reference["guard"])
    whole=np.array(tables["whole"])
    checks=[a[0,0]>=p[0,0],h[0,0]>=q[0,0],a[2,0]<=p[2,0],h[2,0]<=q[2,0],
            a[2,2]>=p[2,2],a[1,1]>p[1,1],whole[2,0]==0]
    return (all(checks),int(sum(checks)),int(a[1,1]),int(h[1,1]),int(a[0,0]),int(h[0,0]))


def run(args):
    torch.manual_seed(SEED);cv2.setNumThreads(1);torch.set_num_threads(2)
    rng=np.random.default_rng(SEED)
    if (args.out/"selection.json").exists():
        raise ValueError("Completed branch already exists")
    ann=base.read(args.annotations)
    annotation_rows=ann["samples"] if isinstance(ann,dict) else ann
    local={r["path"]:r for r in annotation_rows}
    manifest=base.read(base.MANIFEST)["samples"]
    if not set(local)<=set(r["path"] for r in manifest if r["split"]=="train"):
        raise ValueError("New lesion boxes may only originate from training images")
    parent,cut,side=load(args.parent,"cuda")
    if cut!=1.0 or side!=224:
        raise ValueError("A fixed 224 argmax parent is required")
    net=SpatialRepairGrade().cuda()
    net.parent.load_state_dict(parent.state_dict())
    net.parent.eval().requires_grad_(False)
    protocol=dict(start=grade.file_sha(args.parent/"weights.pt"),architecture="mobilenet_v3_small_supervised_surface_v1",
        af_annotations=grade.file_sha(args.annotations),woo=grade.file_sha(WOO),guard=grade.file_sha(args.guard),
        image_training="balanced by domain, grade and capture group, train only",
        location_training="COCO local scratch mask; AF confirmed shallow marks and normal calyx/texture boxes only",
        limits="The new branch may only move parent-good to mild; fixed boundary at zero. Run exactly 240 optimizer updates, check development every 20.",
        seed=SEED,steps=240,script_sha256=grade.file_sha(__file__))
    base.atomic(args.out/"protocol.json",protocol)
    sets=[]
    for train in (True,False):
        pictures,masks,meta=dataset(train,local,args.guard)
        feats,old=cache(net,pictures)
        sets.append((feats,old,torch.tensor(masks,device="cuda"),meta))
        del pictures
    xt,zt,mask,meta=sets[0]; xv,zv,_,vmeta=sets[1]
    stock,_=evaluate(zv.argmax(1).cpu().numpy(),vmeta)
    buckets=defaultdict(lambda:defaultdict(list)); annot=[]
    for i,r in enumerate(meta):
        buckets[(r["domain"],r["grade"])][r["group"]].append(i)
        if mask[i].eq(1).any():
            annot.append(i)
    if not annot:
        raise ValueError("Missing training mark positives")
    groups={k:list(g.values()) for k,g in buckets.items()}
    params=list(net.mark.parameters())+list(net.context.parameters())
    optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=.01)
    y=torch.tensor([r["grade"] for r in meta],device="cuda")
    history=[];best=None
    for step in range(1,241):
        ids=[]
        for domain,count in (("af",8),("guard",5),("woo",5)):
            for label in (0,1):
                pools=groups[domain,label]
                for _ in range(count):
                    bucket=pools[rng.integers(len(pools))]
                    ids.append(bucket[rng.integers(len(bucket))])
        ids.extend(rng.choice(annot,6,replace=True).tolist())
        maps=xt[ids].float(); labels=mask[ids]
        if rng.random()<.5:
            maps=maps.flip((-1,));labels=labels.flip((-1,))
        pred,score=net.local_scores(maps,zt[ids])
        pos=labels==1;neg=labels==0
        loc=-F.logsigmoid(pred[pos]).mean()-F.logsigmoid(-pred[neg]).mean()
        loss=F.binary_cross_entropy_with_logits(score,y[ids].float())+.28*loc
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(params,4.,error_if_nonfinite=True);optimizer.step()
        if step%20==0:
            with torch.inference_mode():
                outputs=[]
                for j in range(0,len(vmeta),24):
                    old=zv[j:j+24]
                    _,s=net.local_scores(xv[j:j+24].float(),old)
                    outputs.extend(net.combine(old,s).argmax(1).cpu().tolist())
                result,_=evaluate(outputs,vmeta)
            rank=order(result,stock)
            history.append(dict(step=step,metrics=result,rank=list(rank)))
            if best is None or rank>best:
                best=rank;chosen=history[-1]
                base.atomic(args.out/"weights.pt",net.state_dict(),tensor=True)
    base.atomic(args.out/"history.json",history)
    receipt=dict(architecture=protocol["architecture"],input_side=224,threshold=1.0,weights_sha256=grade.file_sha(args.out/"weights.pt"),
                 step=chosen["step"],passed=bool(best[0]),metrics=chosen["metrics"],parent_validation=stock,
                 original_annotations=len(local),positive_supervision_views=len(annot),protocol_sha256=grade.file_sha(args.out/"protocol.json"))
    base.atomic(args.out/"selection.json",receipt);print(receipt,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent",type=Path)
    parser.add_argument("annotations",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("--guard",type=Path,required=True)
    run(parser.parse_args())
