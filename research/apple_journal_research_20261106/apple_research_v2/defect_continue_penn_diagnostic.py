"""CPU-only annotated-region audit of the fixed SSDA Small detector on Penn State."""
from collections import Counter,defaultdict
import json
import zipfile

import cv2
import numpy as np
import torch
from class0_patch_refinement import ROOT,configure,atomic_json,sha,read
from defect9485_small import architecture,square,PRIMARY,OUT as SMALL
from defect_algorithm_modes_20261112 import nms
from audit_defect_causes_20261112 import ious

OUT=ROOT/"experiments/defect_continue_penn_diagnostic"
ARCHIVE=ROOT/"datasets/public_defect_intake/Fruit_defects.v2-severity_val.coco-segmentation.zip"
INSECT=("Codling Moth Entry","Plum Curculio","Stink Bug")
OTHER=("Bitter Pit","Flyspeck","Apple Scab","Sunburn","Rot")
UNKNOWN=("Sting",)
KINDS=INSECT+OTHER+UNKNOWN


@torch.inference_mode()
def predict(model,im):
    rgb,(dx,dy,sx,sy)=square(im)
    x=torch.from_numpy(rgb).permute(2,0,1)[None].float()/255
    out=model(x)
    p=out[0][0] if isinstance(out,tuple) else out[0]
    pred=p[p[:,4]>=.001].cpu().numpy()
    h,w=im.shape[:2]
    pred[:,[0,2]]=np.clip((pred[:,[0,2]]-dx)/sx,0,w)
    pred[:,[1,3]]=np.clip((pred[:,[1,3]]-dy)/sy,0,h)
    pred=nms(pred)
    return pred[pred[:,4]>=np.asarray(PRIMARY)[pred[:,5].astype(int)]]


def mask_of(ann,shape):
    mask=np.zeros(shape,np.uint8)
    contours=[np.rint(np.array(v).reshape(-1,2)).astype(np.int32) for v in ann["segmentation"]]
    assert contours and all(len(c)>=3 for c in contours)
    cv2.fillPoly(mask,contours,1)
    return mask


def clip_rect(p,hw):
    h,w=hw
    x,y=np.floor(p[:2]).astype(int).clip([0,0],[w-1,h-1])
    X,Y=np.ceil(p[2:4]).astype(int).clip([x+1,y+1],[w,h])
    return x,y,X,Y


def audit_one(im,preds,anns):
    shape=im.shape[:2]
    masks=[mask_of(a,shape) for a in anns]
    areas=np.array([np.sum(m) for m in masks])
    boxes=np.asarray([a["bbox"] for a in anns],float).reshape(-1,4)
    boxes[:,2:]+=boxes[:,:2]
    ov=ious(preds[:,:4],boxes)
    pan=[]
    for i,p in enumerate(preds):
        if int(p[5])!=1:
            continue
        x,y,X,Y=clip_rect(p,shape)
        cx,cy=(int(v) for v in ((p[:2]+p[2:4])*.5))
        cx,cy=min(shape[1]-1,cx),min(shape[0]-1,cy)
        matches=[]
        for k,(a,m) in enumerate(zip(anns,masks)):
            inter=int(m[y:Y,x:X].sum())
            if not inter:
                continue
            inside=bool(m[cy,cx])
            coverage=inter/int(areas[k])
            strict=bool(ov[i,k]>=.5 and inside and coverage>=.5)
            # Permissive *only for ambiguity*, e.g. a small insect lesion nested in scab.
            plausible=bool(strict or inside or (coverage>=.35 and ov[i,k]>=.08))
            matches.append(dict(index=k,id=a["id"],cat=a["name"],IoU=float(ov[i,k]),
                                center_inside=inside,coverage=coverage,strict=strict,plausible=plausible))
        pan.append(dict(pred_id=i,pred=p.tolist(),matches=matches))
    records=[]
    for k,a in enumerate(anns):
        risks=[]
        for j,other in enumerate(anns):
            if j==k or a["name"]==other["name"]:
                continue
            shared=int(np.sum(masks[j]&masks[k]))
            if shared/min(areas[j],areas[k])>=.1:
                risks.append(other["id"])
        considered=[(v,m) for v in pan for m in v["matches"] if m["index"]==k]
        strict=[(v,m) for v,m in considered if m["strict"]]
        clean=[]
        mixed=[]
        for v,m in strict:
            confounds=[j["id"] for j in v["matches"] if j["plausible"] and j["cat"]!=a["name"]]
            target=mixed if risks or confounds else clean
            target.append(dict(pred_id=v["pred_id"],score=v["pred"][4],IoU=m["IoU"],
                               coverage=m["coverage"],confound_ids=sorted(set(risks+confounds))))
        touch=any(m["center_inside"] for v,m in considered)
        records.append(dict(id=a["id"],category=a["name"],bbox=boxes[k].tolist(),
                            strict_clean=clean,strict_mixed=mixed,
                            overlapping_other_category_ids=risks,
                            center_in_polygon_any=touch))
    return records,pan


def panel(im,a,pred,cat):
    mask=mask_of(a,im.shape[:2])
    x,y,w,h=a["bbox"]
    size=int(min(700,max(96,1.6*max(w,h))))
    cx,cy=int(x+w/2),int(y+h/2)
    x1,y1=int(np.clip(cx-size//2,0,im.shape[1]-size)),int(np.clip(cy-size//2,0,im.shape[0]-size))
    original=im[y1:y1+size,x1:x1+size].copy()
    colored=im.copy()
    cv2.drawContours(colored,cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,225,0),2)
    if pred is not None:
        u,v,U,V=map(int,pred[:4]);cv2.rectangle(colored,(u,v),(U,V),(0,100,255),2)
    overlay=colored[y1:y1+size,x1:x1+size]
    tiles=np.hstack([cv2.resize(v,(148,148)) for v in (original,overlay)])
    label=np.full((22,296,3),245,np.uint8)
    cv2.putText(label,f'{cat} #{a["id"]}',(3,15),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,20,20),1)
    return np.vstack([label,tiles])


def main():
    configure(OUT,"cpu")
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    assert sha(ARCHIVE)=="14f3879657d6a251a6d8707a421b8e91d5c33fa8a05ad6c940a296dff0daa56a"
    wp=SMALL/"weights.pt"
    assert sha(wp)==read(SMALL/"receipt.json")["weight_sha"]
    model=architecture().eval()
    model.load_state_dict(torch.load(wp,map_location="cpu",weights_only=True))
    model=model.fuse(verbose=False).requires_grad_(False)
    total=defaultdict(Counter)
    records=[]
    tiles=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        src=json.loads(z.read("train/_annotations.coco.json"))
        names={a["id"]:a["name"] for a in src["categories"]}
        assert len(src["images"])==70
        for image in src["images"]:
            im=cv2.imdecode(np.frombuffer(z.read("train/"+image["file_name"]),np.uint8),cv2.IMREAD_COLOR)
            assert im.shape[:2]==(image["height"],image["width"])
            anns=[dict(a,name=names[a["category_id"]]) for a in src["annotations"]
                  if a["image_id"]==image["id"] and names[a["category_id"]] in KINDS]
            pred=predict(model,im)
            loc,pan=audit_one(im,pred,anns)
            records.append(dict(image_id=image["id"],filename=image["file_name"],
                                native_shape=im.shape[:2],predictions=pred.tolist(),lesions=loc,pred_overlaps=pan))
            for a,r in zip(anns,loc):
                q=total[r["category"]]
                q["author_regions"]+=1
                q["any_class1_center_inside"]+=r["center_in_polygon_any"]
                q["strict_clean"]+=bool(r["strict_clean"])
                q["strict_mixed"]+=bool(r["strict_mixed"])
                q["overlap_other_diagnosis"]+=bool(r["overlapping_other_category_ids"])
                if r["category"] in OTHER and (r["strict_clean"] or r["strict_mixed"] or r["center_in_polygon_any"]):
                    match=r["strict_clean"] or r["strict_mixed"]
                    p=pred[match[0]["pred_id"]] if match else None
                    tiles.append((not bool(r["strict_clean"]),-max([v["score"] for v in match] or [0.]),
                                  r["id"],panel(im,a,p,r["category"])))
    if tiles:
        tiles.sort(key=lambda x:x[:3])
        blocks=[t[-1] for t in tiles]
        for ix in range(0,len(blocks),12):
            chunk=blocks[ix:ix+12]
            while len(chunk)%3:
                chunk.append(np.zeros_like(chunk[0]))
            canvas=np.vstack([np.hstack(chunk[j:j+3]) for j in range(0,len(chunk),3)])
            (OUT/f"pathology_review_{ix//12+1}.jpg").write_bytes(cv2.imencode(".jpg",canvas)[1])
    strict=[(r["image_id"],a["id"],a["category"]) for r in records for a in r["lesions"] if a["strict_clean"] and a["category"] in OTHER]
    atomic_json(OUT/"diagnostic.json",dict(image_count=70,thresholds=PRIMARY,
            archive_sha=sha(ARCHIVE),weights_sha=sha(wp),device="CPU fp32 768 single view, fused, two CPU threads",
            rules=dict(strict="author lesion box IoU >= .5, prediction center inside author polygon, >= .5 lesion polygon pixels covered",
                       mixed="other diagnosis polygon intersection/min area >= .10, or another diagnosis plausible for same box",
                       plausible="center in other mask or >=.35 of other polygon covered and bbox IoU>=.08",
                       exclusions="Apple ignored; Sting unknown; annotated regions only; unlabeled fruit skin is not verified negative"),
            by_category=total,strict_other_regions=strict,records=records))
    print(dict(total),len(strict),len({k[0] for k in strict}))


if __name__=="__main__":
    main()
