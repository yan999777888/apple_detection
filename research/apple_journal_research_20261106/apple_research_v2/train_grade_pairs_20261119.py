"""Fit and evaluate a sparse native-patch verifier, using matched positive and normal cues."""
from collections import Counter,defaultdict
import json
import argparse
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import torch
from torch.nn import functional as F

import afruit_grade_model as grade
import afruit_grade_train as io
import afruit_grade_crop_alignment as align
from afruit_visible_quality import prepare_input
from matched_grade_loader import load
import select_grade_repair as sel
import grade_pairs_20261119 as net


ROOT=grade.ROOT
SOURCE=ROOT/"datasets/grade_pairs_sources_20261119"
OUT=ROOT/"experiments/grade_pairs_20261119/sparse_verifier_v1"
PARENT=ROOT/"experiments/grade_error49_v1/border_native320_woo_ref"
GUARD=sel.guard.INPUT
SEED=551901


def raw(row):
    return align.read_frame(row)


def views(split):
    cache=io.read(SOURCE/f"af_proposals_{split}.json")
    rows=align.split_rows(split)
    if cache["split"]!=split or [r["sha256"] for r in cache["samples"]]!=[r["sha256"] for r in rows]:
        raise ValueError("Detector data does not match AF split/order")
    for row,result in zip(rows,cache["samples"],strict=True):
        if split=="train" and row["grade"]==2:continue
        im=raw(row);frame=prepare_input(im,1280)
        sh,sw=im.shape[:2];fh,fw=frame.shape[:2]
        if [fh,fw]!=result["hw"]:raise ValueError("Detection grid changed")
        metadata=dict(path=row["path"],sha256=row["sha256"],grade=row["grade"],group_id=row["group_id"])
        yield dict(metadata,panel="whole"),im,net.map_hints(result["defects"],scale=(sw/fw,sh/fh))
        if result["crop"] is None:
            if split=="val":yield dict(metadata,panel="auto"),None,None
        else:
            x,y,r,b=result["crop"]
            if len(result["fruits"])!=1:raise ValueError("Ambiguous fruit crop")
            hints=net.map_hints(result["defects"],offset=(x,y),owner=result["fruits"][0]["fruit_id"])
            yield dict(metadata,panel="auto"),frame[y:b,x:r],hints
    for row in io.read(GUARD)["samples"]:
        if row["split"]!=split:continue
        if split=="train" and (row["grade"]!=0 or row["source"]!="Harvard_Dataverse_Good_Apple_CC0"):continue
        if "hdf" in row["source"].lower():raise ValueError("HDF holdout excluded")
        im=raw(row)
        yield dict(panel="guard",path=row["path"],sha256=row["sha256"],grade=row["grade"],group_id=row["group_id"]),im,net.standalone_hints(im)


@torch.inference_mode()
def panel_embeddings(model,woofer,split):
    feats=[];old=[];woo=[];valid=[];meta=[];images=[];boxes=[];locations=[];rows=[]
    def flush():
        if not images:return
        x=net.prepare(images,"cuda",boxes)
        desc=model.embedding(x.patches.flatten(0,1)).reshape(len(images),net.K,-1)
        feats.append(desc);old.append(model.base(x.parent))
        w=woofer(x.parent[:,3:,:224,:224].contiguous(memory_format=torch.channels_last))
        woo.append(w);valid.append(x.valid)
        for row,im,hints in zip(rows,images,boxes,strict=True):
            meta.append(row);locations.append(net.proposals(im,hints))
        rows.clear();boxes.clear();images.clear()
    unresolved=[]
    for row,im,hints in views(split):
        if im is None:
            unresolved.append(row);continue
        rows.append(row);images.append(im);boxes.append(hints)
        if len(images)==5:flush()
    flush()
    return torch.cat(feats),torch.cat(old),torch.cat(woo),torch.cat(valid),meta,locations,unresolved


def cached_panels(model,woofer,split):
    receipt=dict(architecture_source=grade.file_sha(net.__file__),training_source=grade.file_sha(__file__),
        parent=grade.file_sha(PARENT/"weights.pt"),reference=grade.file_sha(sel.REFERENCE/"weights.pt"),
        af=grade.file_sha(SOURCE/f"af_proposals_{split}.json"),guard=grade.file_sha(GUARD),split=split)
    path=OUT/f"{split}_frozen_patch_features.pt"
    if not path.exists():
        values=panel_embeddings(model,woofer,split)
        io.atomic(path,dict(binding=receipt,tensors=[v.cpu() for v in values[:4]],
                            metadata=json.dumps(values[4:],ensure_ascii=False)),tensor=True)
    data=torch.load(path,map_location="cpu",weights_only=True)
    if data["binding"]!=receipt:raise ValueError("Stale dev/train patch features")
    return (*[v.cuda() for v in data["tensors"]],*json.loads(data["metadata"]))


def open_parents():
    model=net.PairGrade().cuda().eval()
    base,_,_=load(PARENT,"cuda")
    model.base.load_state_dict(base.state_dict());model.base.requires_grad_(False).eval()
    woo,_,_=load(sel.REFERENCE,"cuda")
    return model,woo


def counts_for(predicted,meta,unresolved):
    result={}
    for p in sel.PANELS:
        ids=[i for i,r in enumerate(meta) if r["panel"]==p]
        missing=[r for r in unresolved if r["panel"]==p]
        result[p]=sel.counts([meta[i] for i in ids]+missing,[int(predicted[i]) for i in ids]+[None]*len(missing))
    return result


def sample_bank(meta):
    by=defaultdict(lambda:defaultdict(list))
    for i,row in enumerate(meta):
        by[(row["grade"],row["group_id"]=="burst_0100")][row["path"]].append(i)
    return {k:list(v.values()) for k,v in by.items()}


def draw(bank,n,rng):
    out=[]
    for _ in range(n):
        candidates=bank[rng.integers(len(bank))]
        out.append(candidates[rng.integers(len(candidates))])
    return out


@torch.inference_mode()
def embeddings(net_model,patches):
    vals=[]
    for start in range(0,len(patches),48):
        x=grade.tensor_rgb(patches[start:start+48],"cuda")
        vals.append(net_model.embedding(x))
    return torch.cat(vals)


def patch_report(scores,target,metadata):
    scores=np.asarray(scores);target=np.asarray(target)
    labels=(scores>0).astype(int)
    result={"n":len(scores),"true_positive":int(np.sum((target==1)&(labels==1))),
            "false_positive":int(np.sum((target==0)&(labels==1))),"positive":int(sum(target)),"normal":int(sum(1-target))}
    grouped=defaultdict(list)
    for i,r in enumerate(metadata):grouped[r["pair_group"]].append(i)
    outcomes=[]
    for indices in grouped.values():
        pos=[scores[i] for i in indices if target[i]==1];neg=[scores[i] for i in indices if target[i]==0]
        outcomes.extend(float(a>b)+.5*float(a==b) for a in pos for b in neg)
    result["matched_pairs"]=len(outcomes)
    result["matched_positive_rank_fraction"]=float(np.mean(outcomes)) if outcomes else None
    return result


def benchmark(model,dev):
    data=[]
    for row,im,hints in views("val"):
        if row["panel"]=="auto" and im is not None:data.append((im,hints))
        if len(data)==12:break
    times=[]
    with torch.inference_mode():
        for i in range(5+len(data)*8):
            im,hints=data[i%len(data)];torch.cuda.synchronize();s=perf_counter()
            out=model(net.prepare([im],"cuda",[hints])).argmax(1).cpu().tolist()
            torch.cuda.synchronize()
            if i>=5:times.append(1000*(perf_counter()-s))
    return dict(n=len(times),mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),fps=1000/float(np.mean(times)),
                scope="One real detected fruit and its existing hints: grade + preparation only; no detector, maturity, overlay, source I/O or 3D.")


def train():
    from prepare_grade_pair_patches_20261119 import read_patches
    dest=OUT/"selection.json"
    if dest.exists():raise FileExistsError("Sparse verifier already completed")
    OUT.mkdir(exist_ok=True,parents=True)
    torch.manual_seed(SEED);torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    rng=np.random.default_rng(SEED)
    model,woo=open_parents()
    spec=dict(seed=SEED,arch=net.ARCH,parent=grade.file_sha(PARENT/"weights.pt"),woo=grade.file_sha(sel.REFERENCE/"weights.pt"),
              steps=480,check_every=60,choices="Common November 19 gate: auto mild >=6, good >=20, severe correct >=16, severe_good==0 and whole severe_good==0. Rank by pass, auto mild count, fewer lost good and severe cases versus native320 and Woo, then held-out-source patch proof. Chronologically first on complete ties.",
              detector="Top four frozen deployed instance-associated defect boxes + separated interior texture peaks, eight slots, native-resolution 80px patch inputs. New confirmed patch/face classification, frozen parent and local feature encoder.",
              objective="Class-balanced paired region BCE plus within-photo positive versus normal ranking; weak fruit-level normal/mild samples balance the matched burst_0100 group against alternate AF train setups plus Harvard clean train. Development-region labels are never used for fitting.",
              scene=grade.file_sha(SOURCE/"af_proposals_train.json"),val_scene=grade.file_sha(SOURCE/"af_proposals_val.json"),regions=grade.file_sha(SOURCE/"regions.json"),
              source=grade.file_sha(__file__),network=grade.file_sha(net.__file__))
    io.atomic(OUT/"protocol.json",spec)
    patches,targets,local=read_patches()
    px=embeddings(model,patches).clone();py=torch.tensor(targets,dtype=torch.float32,device="cuda")
    ids=[i for i,r in enumerate(local) if r["split"]=="train"]
    test=[i for i,r in enumerate(local) if r["split"]=="dev"]
    if not test or {targets[i] for i in test}!={0,1}:raise ValueError("Need both classes for a reserved position check")
    region_groups=defaultdict(lambda:defaultdict(list));paired=defaultdict(lambda:defaultdict(list))
    for i in ids:
        r=local[i]
        region_groups[r["label"]][r["pair_group"]].append(i)
        paired[r["pair_group"]][r["label"]].append(i)
    region_groups={k:list(g.values()) for k,g in region_groups.items()}
    pair_groups=[g for g in paired.values() if set(g)=={0,1}]
    if not pair_groups:raise ValueError("Missing matched damage/normal patch evidence")
    print("Anchored patch counts",Counter((r["split"],r["label"],r["source"]) for r in local),flush=True)
    xt,zt,_,mt,train_rows,_,_=cached_panels(model,woo,"train")
    xt,zt,mt=xt.clone(),zt.clone(),mt.clone()
    xv,zv,wv,mv,dev_rows,where,missing=cached_panels(model,woo,"val")
    old_z=counts_for(zv.argmax(1).tolist(),dev_rows,missing)
    old_w=counts_for(wv.argmax(1).tolist(),dev_rows,missing)
    banks=sample_bank(train_rows);yt=torch.tensor([r["grade"] for r in train_rows],device="cuda")
    print("Cached training and dev",len(train_rows),len(dev_rows),"Woo auto",old_w["auto"]["confusion"],flush=True)
    opt=torch.optim.AdamW(model.patch_head.parameters(),lr=.0008,weight_decay=.02)
    history=[];best=None
    for step in range(1,481):
        idx=draw(region_groups[0],18,rng)+draw(region_groups[1],18,rng)
        z=model.patch_head(px[idx]).flatten()
        local_loss=F.binary_cross_entropy_with_logits(z,py[idx])
        ps=[];ns=[]
        for j in rng.integers(len(pair_groups),size=12):
            g=pair_groups[j];ps.append(rng.choice(g[1]));ns.append(rng.choice(g[0]))
        p=model.patch_head(px[ps]).flatten();n=model.patch_head(px[ns]).flatten()
        rank_loss=F.softplus(1.-p+n).mean()
        photos=sum((draw(banks[label,matched],5,rng) for label in (0,1) for matched in (True,False)),[])
        e=model.patch_head(xt[photos]).squeeze(-1)
        pred=model.fruit_logits(zt[photos],e,mt[photos])
        photo_loss=F.cross_entropy(pred,yt[photos])
        loss=local_loss+.5*rank_loss+.65*photo_loss
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.patch_head.parameters(),3.,error_if_nonfinite=True);opt.step()
        if step%60:continue
        with torch.inference_mode():
            e=model.patch_head(xv).squeeze(-1)
            result=model.fruit_logits(zv,e,mv)
            pred=result.argmax(1).tolist()
            evidence=model.patch_head(px[test]).flatten().cpu().tolist()
        metrics=counts_for(pred,dev_rows,missing)
        local_check=patch_report(evidence,[targets[i] for i in test],[local[i] for i in test])
        a=metrics["auto"]
        passed=a["mild"]>=6 and a["good"]>=20 and a["severe"]>=16 and a["severe_good"]==0 and metrics["whole"]["severe_good"]==0
        lost=sum((ref[i]==row["grade"] and new!=row["grade"]) for ref in (wv.argmax(1).tolist(),zv.argmax(1).tolist())
                 for i,(row,new) in enumerate(zip(dev_rows,pred,strict=True)) if row["panel"]=="auto" and row["grade"] in (0,2))
        rank=(not passed,-a["mild"],lost,-(local_check["matched_positive_rank_fraction"] or 0))
        h=dict(step=step,loss=float(loss.detach()),metrics=metrics,passed=passed,patch_check=local_check,relative_lost_good_severe=lost)
        history.append(h);io.atomic(OUT/"history.json",history)
        if best is None or rank<best:
            best=rank;chosen=h
            io.atomic(OUT/"weights.pt",model.state_dict(),tensor=True)
            paired_rows=[]
            for i,(row,loc) in enumerate(zip(dev_rows,where,strict=True)):
                j=int(e[i].masked_fill(~mv[i],-1e4).argmax())
                paired_rows.append(dict(**row,woo=int(wv[i].argmax()),native320=int(zv[i].argmax()),
                    proposed=pred[i],best_proposal=loc[j] if loc else None,sparse_scores=e[i,:len(loc)].tolist()))
            io.atomic(OUT/"development_predictions.json",dict(rows=paired_rows,unresolved=missing,woo=old_w,native320=old_z,
                patch_dev=[dict(**local[i],logit=float(v)) for i,v in zip(test,evidence,strict=True)],patch_training_counts=Counter(str((r["source"],r["label"])) for r in local if r["split"]=="train")))
        print(step,metrics["auto"]["confusion"],"local",local_check,"best",chosen["step"],flush=True)
    model.load_state_dict(torch.load(OUT/"weights.pt",map_location="cuda",weights_only=True))
    io.atomic(OUT/"speed_dev_single_grade.json",benchmark(model,dev_rows))
    io.atomic(dest,dict(architecture=net.ARCH,input_side=320,threshold=1.,weights_sha256=grade.file_sha(OUT/"weights.pt"),
        protocol_sha256=grade.file_sha(OUT/"protocol.json"),metrics=chosen["metrics"],step=chosen["step"],passed=chosen["passed"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-only",action="store_true")
    args=parser.parse_args()
    if args.cache_only:
        torch.set_num_threads(2);cv2.setNumThreads(1)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        model,woo=open_parents()
        for split in ("val","train"):
            data=cached_panels(model,woo,split)
            print(split,len(data[4]),len(data[6]),flush=True)
    else:train()
