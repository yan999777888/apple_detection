"""Hard-negative and source-conditioned training for the November sparse verifier."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
import afruit_grade_model as grade
import afruit_grade_train as io
import grade_pairs_20261119 as paired
import grade_pairs_context_20261119 as ctx
import train_grade_pairs_20261119 as prior
from matched_grade_loader import load


OUT=prior.OUT.parent/"grounded_sparse_v2"
SEED=561901


def region_views(split):
    table=io.read(prior.SOURCE/"regions.json")["regions"]
    by=defaultdict(list)
    for r in table:
        if r["split"]==split and r["target"] in ("surface_defect","confirmed_normal"):
            by[r["path"]].append(r)
    afcache={r["path"]:r for r in io.read(prior.SOURCE/f"af_proposals_{split}.json")["samples"]}
    for row,frame,hints in prior.views(split):
        records=by.get(row["path"],[]);labels=[]
        for ref in records:
            box=list(ref["source_image_xyxy"])
            if row["panel"]=="auto" and frame is not None:
                item=afcache[row["path"]];dy,dx=np.array(item["hw"])/np.array(ref["image_hw"])
                x,y=item["crop"][:2];box=(np.array(box)*[dx,dy,dx,dy]-[x,y,x,y]).tolist()
            labels.append((ref["target"],box))
        yield row,frame,hints,labels
    if split=="train":
        for path,records in sorted(by.items()):
            r=records[0]
            if r["source"] not in ("Woo","PennState"):continue
            area=r["parent_fruit_xyxy"]
            if not area:continue
            x,y,right,bottom=map(round,area)
            frame=prior.raw(dict(path=path,sha256=r["image_sha256"]))[y:bottom,x:right]
            if not frame.size:raise ValueError("Empty annotated source fruit")
            labels=[(ref["target"],(np.array(ref["source_image_xyxy"])-[x,y,x,y]).tolist()) for ref in records]
            row=dict(path=path,sha256=r["image_sha256"],grade=-1,group_id=r["group_id"],panel=r["source"])
            yield row,frame,paired.standalone_hints(frame),labels


def anchored_label(spot,labels):
    x,y,size=spot[:3]
    votes=set()
    for target,(a,b,c,d) in labels:
        inter=max(0,min(c,x+size/2)-max(a,x-size/2))*max(0,min(d,y+size/2)-max(b,y-size/2))
        covered=inter/max(1,(c-a)*(d-b))
        central=(abs(x-(a+c)/2)<=.28*size and abs(y-(b+d)/2)<=.28*size)
        if covered>=.5 and central:votes.add(int(target=="surface_defect"))
    return next(iter(votes)) if len(votes)==1 else -1


@torch.inference_mode()
def collect(model,split):
    meta=[];spots=[];ys=[];fs=[];cs=[];ps=[];missing=[];items=[]
    def flush():
        if not items:return
        data,coords=ctx.encode_input([r[1] for r in items],"cuda",[r[2] for r in items])
        f,c,p=model.encode_batch(data);fs.append(f);cs.append(c);ps.append(p)
        for (row,_,_,labels),loc in zip(items,coords,strict=True):
            lab=[anchored_label(s,labels) for s in loc]
            meta.append(row);spots.append(loc);ys.append(lab+[-1]*(paired.K-len(lab)))
        items.clear()
    for r in region_views(split):
        if r[1] is None:missing.append(r[0]);continue
        items.append(r)
        if len(items)==6:flush()
    flush()
    return torch.cat(fs),torch.cat(cs),torch.cat(ps),torch.tensor(ys,device="cuda"),meta,spots,missing


def caches(model,split):
    file=OUT/f"{split}_context_features.pt"
    receipt=dict(fit=grade.file_sha(__file__),arch=grade.file_sha(ctx.__file__),previous=grade.file_sha(prior.OUT/"weights.pt"),
                 scene=grade.file_sha(prior.SOURCE/f"af_proposals_{split}.json"),regions=grade.file_sha(prior.SOURCE/"regions.json"))
    if not file.exists():
        a=collect(model,split)
        io.atomic(file,dict(receipt=receipt,values=[v.cpu() for v in a[:4]],other=json.dumps(a[4:])),tensor=True)
    data=torch.load(file,map_location="cpu",weights_only=True)
    if data["receipt"]!=receipt:raise ValueError("Changed context features")
    return (*[v.cuda() for v in data["values"]],*json.loads(data["other"]))


def describe(meta,labels,core,old):
    banks=defaultdict(lambda:defaultdict(list))
    eligible=core[:,:,2].cpu().numpy()>0;score=old.cpu().numpy();truth=labels.cpu().numpy()
    for i,r in enumerate(meta):
        for j in np.flatnonzero(eligible[i]&(truth[i]>=0)):
            banks["mark" if truth[i,j] else "normal"][r["path"]].append((i,int(j)))
        if r["grade"]==0:
            for flag in (False,True):
                ids=np.flatnonzero(eligible[i] & ((core[i,:,3].cpu().numpy()>0)==flag))
                if len(ids):
                    hard=sorted(ids,key=lambda j:-score[i,j])[:2]
                    banks["hard_detector" if flag else "hard_texture"][r["path"]].extend((i,int(j)) for j in hard)
    return {k:list(g.values()) for k,g in banks.items()}


def rank(metrics,dev_rows,n,w,labels,pred,local):
    a=metrics["auto"]
    ok=(a["mild"]>=6 and a["good"]>=20 and a["severe"]>=16 and a["severe_good"]==0 and metrics["whole"]["severe_good"]==0)
    lost=sum(int(ref[i])==r["grade"] and new!=r["grade"] for ref in (n,w) for i,(r,new) in enumerate(zip(dev_rows,pred,strict=True))
             if r["panel"]=="auto" and r["grade"] in (0,2))
    paired=local[labels==1].mean()-local[labels==0].mean() if (labels==0).any() and (labels==1).any() else local.new_tensor(0.)
    return (not ok,-a["mild"],lost,-float(paired)),ok


@torch.inference_mode()
def speed(model):
    photos=[]
    for r,im,h in prior.views("val"):
        if r["panel"]=="auto" and im is not None:photos.append((im,h))
        if len(photos)>=12:break
    times=[]
    for i in range(101):
        im,hints=photos[i%12]
        torch.cuda.synchronize();start=time.perf_counter()
        selected=model(ctx.prepare([im],"cuda",[hints])).argmax(1).cpu().item()
        torch.cuda.synchronize()
        if i>=5:times.append((time.perf_counter()-start)*1000)
    return dict(n=96,mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),fps=float(1000/np.mean(times)))


def run():
    if (OUT/"selection.json").is_file():raise FileExistsError("Completed context branch")
    torch.manual_seed(SEED);torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    rng=np.random.default_rng(SEED);model=ctx.ContextGrade().cuda().eval()
    base,_,_=load(prior.OUT,"cuda")
    state=model.load_state_dict(base.state_dict(),strict=False)
    if state.unexpected_keys or any(not k.startswith(("context_head.","fusion_head.","clean.")) for k in state.missing_keys):raise ValueError("Incomplete verifier initializer")
    model.base.requires_grad_(False);model.patch_head.requires_grad_(False)
    spec=dict(architecture=ctx.ARCH,parent_sparse=grade.file_sha(prior.OUT/"weights.pt"),reference_native=grade.file_sha(prior.PARENT/"weights.pt"),
        steps=360,checkpoint_every=30,seed=SEED,initial="previous learned verifier and native320 parent frozen",
        method="One calibrated local head uses candidate type, scale, local/ring skin occupancy, reflection, border support, native parent mark-map responses and frozen patch descriptor. Train true centered regions and same-domain annotated normal regions, plus highest-scoring false locations from each source on each labeled clean training face.",
        decisions="Prior severe protected. Return a mild fruit to good only when separate whole-face visual cleanliness trained on actual grades supports release and two valid patches exist.",
        inputs="AF train/val and already approved existing train or source validation only; no access to field outputs.",
        policy_sha256=grade.file_sha(OUT.parent/"protocol.json"),regions=grade.file_sha(prior.SOURCE/"regions.json"),
        code=grade.file_sha(__file__),model=grade.file_sha(ctx.__file__))
    io.atomic(OUT/"protocol.json",spec)
    f,c,p,y,meta,spots,_=caches(model,"train")
    vx,vc,vp,vy,vm,where,missing=caches(model,"val")
    frozen=io.read(prior.OUT/"development_predictions.json")
    vwoo={ (r["path"],r["panel"]):r["woo"] for r in frozen["rows"]}
    w=[vwoo[r["path"],r["panel"]] for r in vm]
    g=prior.sample_bank([r for r in meta if r["grade"] in (0,1)])
    ir=[i for i,r in enumerate(meta) if r["grade"] in (0,1)]
    whole=torch.tensor([r["grade"] for r in meta],device="cuda")
    with torch.inference_mode():old=model.patch_head(f).squeeze(-1)
    bags=describe(meta,y,c,old)
    io.atomic(OUT/"anchored_bank_audit.json",dict(unique_sources={k:len(v) for k,v in bags.items()},
         slots_by_type={k:sum(map(len,v)) for k,v in bags.items()},dev_confirmed_loc_samples=Counter(vy[vy>=0].tolist())))
    print("Real mapped training/clean-source groups", {k:len(v) for k,v in bags.items()},flush=True)
    params=[q for q in model.parameters() if q.requires_grad]
    opt=torch.optim.AdamW(params,lr=.0012,weight_decay=.018)
    history=[];best=None
    for step in range(1,361):
        bank=[];truth=[]
        for key,n in (("mark",16),("normal",8),("hard_detector",5),("hard_texture",5)):
            sampled=prior.draw(bags[key],n,rng);bank+=sampled;truth += [int(key=="mark")]*n
        ii,jj=map(list,zip(*bank))
        loc=model.verify(f[ii],c[ii])[torch.arange(len(ii)),jj]
        target=torch.tensor(truth,device="cuda",dtype=torch.float)
        lesion=F.binary_cross_entropy_with_logits(loc,target)
        ranking=F.softplus(.8-loc[:16,None]+loc[None,16:]).mean()
        samples=sum((prior.draw(g[label,matched],5,rng) for label in (0,1) for matched in (False,True)),[])
        pics=[ir[j] for j in samples]
        z=model.verify(f[pics],c[pics])
        output,clean=model.decisions(p[pics],z,c[pics])
        fruit=F.cross_entropy(output,whole[pics])
        normal=F.binary_cross_entropy_with_logits(clean,(whole[pics]==0).float())
        loss=.65*lesion+.22*ranking+fruit+.65*normal
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(params,3.,error_if_nonfinite=True);opt.step()
        if step%30:continue
        with torch.inference_mode():
            scores=model.verify(vx,vc)
            logits,q=model.decisions(vp,scores,vc);pred=logits.argmax(1).tolist()
            metrics=prior.counts_for(pred,vm,missing)
            ordering,passed=rank(metrics,vm,vp.argmax(1).tolist(),w,vy,pred,scores)
        entry=dict(step=step,loss=float(loss.detach()),metrics=metrics,passed=passed,rank=ordering)
        history.append(entry);io.atomic(OUT/"history.json",history)
        if best is None or ordering<best:
            best=ordering;chosen=entry;io.atomic(OUT/"weights.pt",model.state_dict(),tensor=True)
            rows=[]
            for i,(r,sites) in enumerate(zip(vm,where,strict=True)):
                best_site=int(scores[i,:len(sites)].argmax()) if sites else -1
                rows.append(dict(**r,woo=w[i],native320=int(vp[i].argmax()),proposed=pred[i],clean_margin=float(q[i]),
                  best_proposal=sites[best_site] if sites else None,sparse_scores=scores[i,:len(sites)].tolist(),
                  labels_on_proposals=vy[i,:len(sites)].tolist()))
            io.atomic(OUT/"development_predictions.json",dict(rows=rows,unresolved=missing,woo=frozen["woo"],native320=frozen["native320"]))
        print("step",step,metrics["auto"]["confusion"],"chosen",chosen["step"],flush=True)
    model.load_state_dict(torch.load(OUT/"weights.pt",map_location="cuda",weights_only=True))
    io.atomic(OUT/"speed_dev_single_grade.json",speed(model))
    io.atomic(OUT/"selection.json",dict(architecture=ctx.ARCH,input_side=320,threshold=1.,weights_sha256=grade.file_sha(OUT/"weights.pt"),
         protocol_sha256=grade.file_sha(OUT/"protocol.json"),metrics=chosen["metrics"],step=chosen["step"],passed=chosen["passed"]))


if __name__=="__main__":
    run()
