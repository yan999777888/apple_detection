"""Prepare registered stage features and fit one grade head on same-condition AF/Woo pairs."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
import grade_stage_neck as neck
import afruit_grade_crop_alignment as af

OUT=neck.OUT
REGIONS=neck.ROOT/"datasets/grade_pairs_sources_20261119/regions.json"
SEED=19061141
STEPS=420
G=neck.GRID


def recipe():
    protocol=dict(stage=neck.visible.STAGE_WEIGHT_SHA,parent=neck.grade.file_sha(neck.PARENT/"weights.pt"),AF=neck.grade.file_sha(neck.io.MANIFEST),
                  Woo=neck.grade.file_sha(neck.WOO),regions=neck.grade.file_sha(REGIONS),source=neck.grade.file_sha(__file__),
                  network=neck.grade.file_sha(Path(neck.__file__)),seed=SEED,steps=STEPS,
                  mechanism="Passive read of the original stage head P3/P4/P5 (strides 8,16,32). ROIAlign 10x10 uses actual rounded content/padding. Original visible mask support. The full supplied-photo mode pools its whole rectangle regardless of native instance count; use a unique native mask only when exactly one is present.",
                  training="Use original AF training split and the previously checked matched Woo 48 good + 10 minor training-only surface views. For AF use frozen auto crop only when exactly one predicted fruit, plus supplied whole view. Woo uses supplied face crops. No HDF/AF-test/field/source holdouts.",
                  objective="Frozen native320 and detectors. Only train 3 level projections, mark convolution and pooled appearance readout. Preserve every native320 severe argmax. Other fruit combine .35 native mild-good with learned local evidence. AF/ Woo condition-matched good versus light pairs; fixed 420 AdamW updates. True mark and explicit normal cells alone supply location loss.",
                  selection="Development every 30 steps. Gate all 57 auto: good>=20/25, mild>=6/15,severe>=16/17,none severe->good. Whole zero severe->good. Prioritize all safety gates, then mild, then good+severe, then whole mild. First strict tie wins.",
                  detection_control="Direct comparison of full frozen baseline outputs with and without passive head hooks for two train and at least six development in each pathway. Check all source-grid sizes on synthetic padding fill. Underlying models/stage are never optimized.")
    p=OUT/"protocol.json"
    if p.exists() and neck.io.read(p)!=protocol:raise ValueError("Frozen recipe changed")
    neck.io.atomic(p,protocol)


def mapped_labels(r,box,scale,marks):
    out=np.full((G,G),-1,np.int8)
    for subset,label in (("confirmed_normal",0),("surface_defect",1)):
        for ob in marks.get(r["path"],[]):
            if ob["target"]!=subset:continue
            a=np.array(ob["source_image_xyxy"])*scale
            x=(a[0]+a[2])*.5;y=(a[1]+a[3])*.5
            if box[0]<=x<box[2] and box[1]<=y<box[3]:
                j=min(G-1,int(G*(x-box[0])/(box[2]-box[0])));i=min(G-1,int(G*(y-box[1])/(box[3]-box[1])))
                out[i,j]=label
    return out


def cache():
    recipe();rt=neck.Runtime()
    marks=defaultdict(list)
    for r in neck.io.read(REGIONS)["regions"]:
        if r["split"]=="train":marks[r["path"]].append(r)
    owner=neck.io.read(neck.WOO)["samples"]
    # Expanded Woo uses the native full frame, while the grade manifest stores a cropped face.
    # Reproject only exact same-camera, same-original source marks.
    bysha={r["source_sha256"]:r for r in owner}
    for r in neck.io.read(REGIONS)["regions"]:
        if r["source"]!="Woo" or r["split"]!="train" or r["image_sha256"] not in bysha:continue
        own=bysha[r["image_sha256"]];x,y=own["source_box"][:2]
        xy=np.asarray(r["source_image_xyxy"])-[x,y,x,y];marks[own["path"]].append(dict(r,source_image_xyxy=xy.tolist()))
    audits=[]
    for split in ("train","val"):
        p=OUT/(split+"_features.pt")
        if p.exists():raise ValueError("Frozen feature cache already exists")
        source=[dict(r,domain="AF") for r in af.split_rows(split)]
        regions={v["sha256"]:v for v in neck.io.read(af.OUT/(split+"_regions.json"))}
        if split=="train":source+= [dict(r,domain="Woo") for r in owner if r["split"]=="train"]
        cells=[];support=[];old=[];labels=[];records=[];checks=[];unresolved=[]
        for n,r in enumerate(source):
            raw=af.read_frame(r);frame=neck.visible.prepare_input(raw,1280);h,w=frame.shape[:2]
            scale=np.array([w/raw.shape[1],h/raw.shape[0]]*2)
            check=n<(2 if split=="train" else 6)
            res,ms,ids,data=rt.observe(frame,known=False,verify=check,both=True)
            f,m,z,cs=data
            if check:
                given,_,_,other=rt.observe(frame,known=True,verify=True)
                if not torch.allclose(f[-1],other[0][0],rtol=1e-5,atol=1e-5) or not torch.allclose(z[-1],other[2][0],atol=1e-4,rtol=1e-4):
                    raise ValueError("Supplied full-plane score maps disagree with standalone path")
                checks.append(dict(sha=r["sha256"],frozen_scene_and_supplied_unchanged=True))
            use=[]
            if r["domain"]=="AF":
                expected=regions[r["sha256"]]
                if len(ms)!=expected["count"]:raise ValueError("Changed original stage count")
                if len(ids)==1 and len(ms)==1:
                    box=res["fruits"][ids[0]]["crop"]
                    if box!=expected["crop"]:raise ValueError("Original registered GT/crop moved")
                    use.append((0,"auto",box))
                else:unresolved.append(dict(path=r["path"],grade=r["grade"],count=len(ms)))
            use.append((-1,"whole",[0,0,w,h]))
            for i,v,box in use:
                cells.append(f[i].half().cpu());support.append(m[i].half().cpu());old.append(z[i].cpu())
                labels.append(mapped_labels(r,box,scale,marks) if split=="train" else np.full((G,G),-1,np.int8))
                records.append(dict(path=r["path"],sha=r["sha256"],domain=r["domain"],grade=r["grade"],view=v,
                                    group=r.get("condition_id",r["group_id"]),capture=r.get("capture_key"),box=box))
            if n%80==0:print(split,n,len(source),flush=True)
        packed=dict(grids=torch.stack(cells),coverage=torch.stack(support),parent=torch.stack(old),labels=torch.tensor(np.stack(labels)),
                    records=records,unresolved=unresolved,channels=cs,protocol_sha=neck.grade.file_sha(OUT/"protocol.json"))
        neck.io.atomic(p,packed,tensor=True)
        audits+=checks
    neck.io.atomic(OUT/"alignment_and_detection_audit.json",dict(controls=audits,grid_shapes={str(k):v for k,v in rt.shapes.items()},channels=cs,feature_layers=[16,19,22],strides=[8,16,32]))


def evaluate(pred,data):
    tables={}
    for v in ("auto","whole"):
        counts=np.zeros((3,4),int)
        for i,r in enumerate(data["records"]):
            if r["view"]==v:counts[r["grade"],int(pred[i])]+=1
        if v=="auto":
            for r in data["unresolved"]:counts[r["grade"],3]+=1
        if counts.sum(1).tolist()!=[25,15,17]:raise ValueError("All validation photos required")
        tables[v]=dict(n=57,confusion=counts.tolist(),good=int(counts[0,0]),mild=int(counts[1,1]),severe=int(counts[2,2]),severe_good=int(counts[2,0]))
    return tables


def order(m):
    a=m["auto"];w=m["whole"]
    guards=[a["good"]>=20,a["severe"]>=16,a["severe_good"]==0,w["severe_good"]==0]
    return (int(all(guards) and a["mild"]>=6),sum(guards),a["mild"],a["good"]+a["severe"],w["mild"])


def fit():
    recipe()
    if (OUT/"selection.json").exists():raise ValueError("Single training already completed")
    torch.manual_seed(SEED);np_rng=np.random.default_rng(SEED)
    pack=[torch.load(OUT/(k+"_features.pt"),map_location="cpu",weights_only=True) for k in ("train","val")]
    tr,val=pack
    for d in pack:
        for k in ("grids","coverage","parent","labels"):d[k]=d[k].cuda()
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    by_domain=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(tr["records"]):
        groups[r["domain"],r["grade"]][r["group"]][r["path"]].append(i)
        by_domain[r["domain"],r["grade"]][r["path"]].append(i)
    def pick(gr,domain,grade):
        g=groups[domain,grade]
        if gr is None:gr=sorted(g)[np_rng.integers(len(g))]
        p=g[gr];sel=sorted(p)[np_rng.integers(len(p))];return int(np_rng.choice(p[sel]))
    model=neck.NeckGrade(tr["channels"]).cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.02)
    baseline=evaluate(val["parent"].argmax(1).tolist(),val)
    history=[];best=None
    for step in range(1,STEPS+1):
        ids=[];couples=[]
        for domain,n in (("AF",8),("Woo",6)):
            cond=sorted(set(groups[domain,0])&set(groups[domain,1]))
            for _ in range(n):
                group=cond[np_rng.integers(len(cond))];couples.append(len(ids))
                ids.extend([pick(group,domain,0),pick(group,domain,1)])
        for y in (0,1):
            ids += [pick(None,"AF",y) for _ in range(6)]
        x,m,z,y=[tr[k][ids] for k in ("grids","coverage","parent","labels")]
        x=x.float();m=m.float()
        if np_rng.random()<.5:x=x.flip([-1]);m=m.flip([-1]);y=y.flip([-1])
        target=torch.tensor([tr["records"][i]["grade"] for i in ids],device="cuda",dtype=torch.float32)
        _,d,heat=model(x,m,z)
        e=d-.35*(z[:,1]-z[:,0])
        ci=torch.tensor(couples,device="cuda")
        loss=F.binary_cross_entropy_with_logits(d,target)+.58*F.binary_cross_entropy_with_logits(e,target)
        loss=loss+.47*F.softplus(.65-e[ci+1]+e[ci]).mean()
        heat=heat[:,0]
        for t,sgn in ((0,1),(1,-1)):
            active=(y==t)&(heat>-25)
            if active.any():loss=loss+.12*F.softplus(heat[active]*sgn).mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),4,error_if_nonfinite=True);opt.step()
        if step%30:continue
        with torch.no_grad():
            logits,_,_=model(val["grids"].float(),val["coverage"].float(),val["parent"])
            pred=logits.argmax(1).cpu().tolist()
        metrics=evaluate(pred,val);k=order(metrics);entry=dict(step=step,rank=k,metrics=metrics);history.append(entry)
        if best is None or k>best:
            best=k;chosen=entry;chosen_pred=pred;neck.io.atomic(OUT/"weights.pt",model.state_dict(),tensor=True)
        print(step,metrics["auto"]["confusion"],flush=True)
    neck.io.atomic(OUT/"history.json",history)
    receipt=dict(architecture="stage_neck_native320_v1",entry="grade_stage_neck.Runtime(selection).observe / __call__(1280px BGR,known_single_fruit=...)",
                 step=chosen["step"],stage_sha256=neck.visible.STAGE_WEIGHT_SHA,native320_sha256=neck.grade.file_sha(neck.PARENT/"weights.pt"),
                 weights_sha256=neck.grade.file_sha(OUT/"weights.pt"),channels=tr["channels"],metrics=chosen["metrics"],parent=baseline,
                 passed=bool(best[0]),protocol_sha256=neck.grade.file_sha(OUT/"protocol.json"))
    neck.io.atomic(OUT/"selection.json",receipt)
    neck.io.atomic(OUT/"development_predictions.json",dict(records=[dict(r,baseline=int(val["parent"][i].argmax()),candidate=chosen_pred[i]) for i,r in enumerate(val["records"])],unresolved=val["unresolved"]))
    print(receipt,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("action",choices=("cache","fit","all"))
    arg=parser.parse_args();cv2.setNumThreads(2);torch.set_num_threads(2)
    if arg.action in ("cache","all"):cache()
    if arg.action in ("fit","all"):fit()

