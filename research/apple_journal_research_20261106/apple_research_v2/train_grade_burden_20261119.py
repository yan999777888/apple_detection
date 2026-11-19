"""Fit a single paired, regularized fruit severity readout on CPU."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
import afruit_grade_model as grade
import afruit_grade_train as io
import grade_pairs_20261119 as sparse
import grade_pairs_burden_20261119 as b
import train_grade_context_20261119 as grounded
import train_grade_pairs_20261119 as prior
from matched_grade_loader import load


OUT=prior.OUT.parent/"fruit_surface_burden_v1"
WOO=grade.ROOT/"datasets/apple_matched_quality_v1/manifest.json"


def descriptors(model,split):
    file=grounded.OUT/f"{split}_context_features.pt"
    bank=torch.load(file,weights_only=True,map_location="cpu")
    original=io.read(grounded.OUT/"protocol.json")
    if bank["receipt"]["arch"]!=original["model"] or bank["receipt"]["scene"]!=grade.file_sha(prior.SOURCE/f"af_proposals_{split}.json"):
        raise ValueError("Different archived patch evidence")
    feat,core,parent,_=bank["values"];meta,coords,miss=json.loads(bank["other"])
    cache={r["path"]:r for r in io.read(prior.SOURCE/f"af_proposals_{split}.json")["samples"]}
    table=io.read(prior.SOURCE/"regions.json")["regions"]
    by={}
    for r in table:
        if r["split"]==split:by.setdefault(r["path"],r)
    results=[];held=None;raw=None;working=None
    for row,sites in zip(meta,coords,strict=True):
        path=row["path"]
        if held!=path:
            raw=prior.raw(row);working=None;held=path
        panel=row["panel"]
        if panel=="auto":
            working=prior.prepare_input(raw,1280);x,y,r,bt=cache[path]["crop"];frame=working[y:bt,x:r]
        elif panel in ("PennState","Woo"):
            x,y,r,bt=map(round,by[path]["parent_fruit_xyxy"]);frame=raw[y:bt,x:r]
        else:frame=raw
        patches=[sparse.extract_patch(frame,*s[:3]) for s in sites]
        results.append(b.footprints(frame,sites,patches))
    masks=torch.from_numpy(np.stack(results))
    with torch.inference_mode():
        scores=model.verify(feat,core)
        desc=b.features(parent,scores,core,masks)
        ref,_=model.decisions(parent,scores,core)
    return desc.clone(),parent,core,meta,miss,ref.argmax(1).tolist(),scores,coords


def tradeoffs(meta,samples,parent):
    choices=[];groups=defaultdict(lambda:defaultdict(list));pairs=defaultdict(lambda:defaultdict(list))
    woo={}
    for r in io.read(WOO)["samples"]:
        if r["split"]=="train":
            for k in ("sha256","source_sha256"):woo[r[k]]=r
    for i,row in enumerate(meta):
        g=row["grade"];site=row["group_id"];domain=row["panel"];condition=site
        if parent[i].argmax()==2:continue
        if domain in ("whole","auto"):
            source="matched_AF" if site=="burst_0100" else "other_AF"
            condition=(site,domain)
        elif row["panel"]=="Woo" and row["sha256"] in woo:
            ref=woo[row["sha256"]];source="Woo";g=ref["grade"];condition=ref["condition_id"]
        elif row["panel"]=="guard" and row["grade"]==0:
            source="healthy_train"
        else:continue
        if g not in (0,1):continue
        k=len(choices);choices.append((i,g,source,condition,row["path"]))
        groups[source][g].append(k)
        if source in ("matched_AF","Woo"):pairs[(source,condition)][g].append(k)
    weights=np.zeros(len(choices))
    for source,fraction in (("matched_AF",.50),("Woo",.25),("other_AF",.20),("healthy_train",.05)):
        groups_y=groups[source]
        if not groups_y:raise ValueError(f"Missing training domain {source}")
        for targets in groups_y.values():
            # Counterbalance the two views of a photographed apple.
            by_path=defaultdict(list)
            for k in targets:by_path[choices[k][4]].append(k)
            for views in by_path.values():
                for k in views:weights[k]=fraction/(len(groups_y)*len(by_path)*len(views))
    positive=[];negative=[];pw=[]
    active=[g for g,v in pairs.items() if set(v)=={0,1}]
    for group in active:
        by_y=pairs[group]
        for pos in by_y[1]:
            for neg in by_y[0]:
                positive.append(pos);negative.append(neg)
                pw.append(1./(len(active)*len(by_y[0])*len(by_y[1])))
    if not positive:raise ValueError("No condition-matched grade pairs")
    return choices,torch.tensor(weights,dtype=torch.float64),positive,negative,torch.tensor(pw,dtype=torch.float64)


def run():
    if (OUT/"selection.json").exists():raise FileExistsError("Completed fruit-level calibration")
    cv2.setNumThreads(1);torch.set_num_threads(2)
    model=b.BurdenGrade().eval()
    base,_,_=load(grounded.OUT,"cpu")
    loaded=model.load_state_dict(base.state_dict(),strict=False)
    if set(loaded.missing_keys)!={"mean","scale","burden.bias","burden.weight"} or loaded.unexpected_keys:
        raise ValueError("Unexpected frozen encoder")
    spec=dict(architecture=b.ARCH,prior=grade.file_sha(grounded.OUT/"weights.pt"),common=grade.file_sha(OUT.parent/"protocol.json"),
        design="Estimate severity from a fruit-normalized 36x36 map of local color-contrast traces and a frozen per-site verifier. Posterior-weighted per-pixel maximum avoids adding the same wound twice. Report largest/total fractional area, fractional independent trace count and source counts, spatial spread, strongest supports, skin evidence and parent prior. Physical diameter is not inferred.",
        training="One convex readout, grade BCE + one same-setup, cross-grade pairwise ordering objective + fixed L2=0.08. Across AF/Woo/otherAF/healthy weight .50/.25/.20/.05; normalized by source, class, original photo and its duplicate views. No threshold grid or dev checkpoint selection.",
        groups="Matched pairs within AF mixed-grade burst and same view mode, or Woo same station/camera. Only manually graded Woo examples become fruit-level samples; marks alone do not imply a fruit grade.",
        source=grade.file_sha(__file__),network=grade.file_sha(b.__file__),train_sha=grade.file_sha(grounded.OUT/"train_context_features.pt"))
    saved=(OUT/"weights.pt").exists()
    if saved:
        locked=io.read(OUT/"protocol.json")
        if any(v!=locked.get(k) for k,v in spec.items() if k!="source"):
            raise ValueError("Existing fitted weights follow another protocol")
    else:io.atomic(OUT/"protocol.json",spec)
    xt,pt,ct,train_meta,_,_,_,_=descriptors(model,"train")
    rows,wt,pos,neg,pw=tradeoffs(train_meta,xt,pt)
    xx=xt[[v[0] for v in rows]].double()
    y=torch.tensor([v[1] for v in rows],dtype=torch.float64)
    center=(xx*wt[:,None]).sum(0);scale=((xx-center).square()*wt[:,None]).sum(0).sqrt().clamp_min(.06)
    norm=(xx-center)/scale
    layer=torch.nn.Linear(norm.shape[1],1,dtype=torch.float64)
    torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
    opt=torch.optim.LBFGS(layer.parameters(),lr=1.,max_iter=120,tolerance_grad=1.e-8,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad();z=layer(norm).flatten()
        lo=(wt*F.binary_cross_entropy_with_logits(z,y,reduction="none")).sum()
        lo=lo+(pw*F.softplus(-z[pos]+z[neg])).sum()+.08*layer.weight.square().sum()
        lo.backward();return lo
    if saved:
        model.load_state_dict(torch.load(OUT/"weights.pt",map_location="cpu",weights_only=True))
        if not torch.equal(model.mean,center.float()) or not torch.equal(model.scale,scale.float()):
            raise ValueError("Saved readout does not match the registered training descriptors")
        layer.load_state_dict({k:v.double() for k,v in model.burden.state_dict().items()})
    else:
        opt.step(closure)
        model.mean.copy_(center.float());model.scale.copy_(scale.float())
        model.burden.load_state_dict({k:v.float() for k,v in layer.state_dict().items()})
        io.atomic(OUT/"weights.pt",model.state_dict(),tensor=True)
    vx,vp,vc,meta,missing,old_predictions,scores,coords=descriptors(model,"val")
    with torch.inference_mode():pred=model.classify_burden(vp,vx,vc).argmax(1).tolist()
    frozen=io.read(grounded.OUT/"development_predictions.json")
    historic={(r["path"],r["panel"]):r for r in frozen["rows"]}
    metrics=prior.counts_for(pred,meta,missing);a=metrics["auto"]
    passed=(a["mild"]>=6 and a["good"]>=20 and a["severe"]>=16 and a["severe_good"]==0 and metrics["whole"]["severe_good"]==0)
    results=[]
    for i,row in enumerate(meta):
        priorrow=historic[row["path"],row["panel"]]
        results.append(dict(priorrow,grounded=old_predictions[i],proposed=pred[i],normalized_burden=vx[i].tolist()))
    io.atomic(OUT/"development_predictions.json",dict(rows=results,unresolved=missing,woo=frozen["woo"],native320=frozen["native320"]))
    io.atomic(OUT/"fit.json",dict(supervised_views=Counter(str((x[2],x[1])) for x in rows),
        unique_matched_ordering_pairs=len(pos),learned=layer.weight.detach().flatten().tolist(),
        mean=center.tolist(),scale=scale.tolist(),solution_loss=float(closure().detach())))
    io.atomic(OUT/"selection.json",dict(architecture=b.ARCH,input_side=320,threshold=1.,weights_sha256=grade.file_sha(OUT/"weights.pt"),
         protocol_sha256=grade.file_sha(OUT/"protocol.json"),metrics=metrics,passed=passed))
    print("Development",metrics["auto"]["confusion"],metrics["whole"]["confusion"],metrics["guard"]["confusion"],passed,flush=True)


if __name__=="__main__":run()
