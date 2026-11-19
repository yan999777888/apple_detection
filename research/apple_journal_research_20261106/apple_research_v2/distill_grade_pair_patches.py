"""Train a fast proposal verifier using out-of-source DINO patch predictions on AF training fruit."""
from collections import Counter

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from torch.nn import functional as F

import afruit_grade_model as grade
import afruit_grade_train as io
import grade_pairs_20261119 as net
import train_grade_pairs_20261119 as sparse
from defect9485_repr import encoder, META_SHA
from prepare_grade_pair_patches_20261119 import FILE, read_patches
from train_grade_capture_contrast import ranking


OUT=grade.ROOT/"experiments/grade_pairs_20261119/patch_distill"
SEED=721119


@torch.inference_mode()
def teacher_features(teacher,patches):
    groups=[]
    for j in range(0,len(patches),64):
        u=np.stack([cv2.resize(x,(84,84),interpolation=cv2.INTER_LINEAR) for x in patches[j:j+64]])
        with torch.autocast("cuda",dtype=torch.float16):
            f=teacher(torch.as_tensor(u,device="cuda")).float().cpu().numpy()
        groups.append(f)
    return normalize(np.concatenate(groups))


def teach(x,y,mask,local):
    # Count each source and label cell once, regardless of repeated views or flips.
    ii=np.flatnonzero(mask)
    cells=Counter((local[i]["source"],int(y[i])) for i in ii)
    w=np.array([1/cells[local[i]["source"],int(y[i])] for i in ii]);w=w*len(w)/sum(w)
    return LogisticRegression(C=12,max_iter=500,random_state=SEED).fit(x[ii],y[ii],sample_weight=w)


def prepare_scores(teacher,head,ixrows):
    scores=[];keys=[];pending=[]
    def flush():
        if pending:
            scores.extend(head.predict_proba(teacher_features(teacher,pending))[:,1].tolist());pending.clear()
    for row,im,hints in sparse.views("train"):
        if im is None:continue
        keys.append((row["panel"],row["sha256"]))
        bank,_,_=net.patch_bank(im,hints)
        pending.extend(bank)
        if len(pending)>=128:flush()
    flush()
    assert keys==[(r["panel"],r["sha256"]) for r in ixrows]
    return torch.tensor(scores,device="cuda").reshape(len(keys),net.K)


def run():
    if (OUT/"summary.json").exists():raise FileExistsError(OUT)
    cv2.setNumThreads(2);torch.set_num_threads(2);torch.manual_seed(SEED)
    model,woo=sparse.open_parents()
    initial=sparse.OUT/"weights.pt"
    assert grade.file_sha(initial)==io.read(sparse.OUT/"selection.json")["weights_sha256"]
    model.load_state_dict(torch.load(initial,map_location="cpu",weights_only=True))
    io.atomic(OUT/"protocol.json",dict(source=grade.file_sha(FILE),dino=META_SHA,parent=grade.file_sha(initial),
            seed=SEED,steps=240,lr=.0004,coef=[0,.9],decision="Fixed argmax and existing 8-proposal architecture. Both arms start from the same frozen sparse run, same train image exposures. The teacher for proposals on AF and Harvard excludes all AF/Harvard patch labels. No synthetic grade assigned to uncertified apple areas. Distill only soft probability on proposals and keep native supervised patches/fruit loss in both arms. Common development gate and automatic mild primary."))
    px,y,local=read_patches()
    target=np.asarray(y);train=np.array([r["split"]=="train" for r in local]);val=~train
    teacher=encoder().cuda().eval()
    dinox=teacher_features(teacher,px)
    no_af=train & np.array([r["source"] not in ("AFruitDB","Harvard_Dataverse_Good_Apple_CC0") for r in local])
    head=teach(dinox,target,no_af,local)
    vprob=head.predict_proba(dinox[val])[:,1]
    io.atomic(OUT/"teacher.json",dict(excluded="All AFruitDB and Harvard annotations",fit_sources=dict(Counter(local[i]["source"] for i in np.flatnonzero(no_af & np.array([not r["flipped"] for r in local])))),
       dev=sparse.patch_report(np.log((vprob+1e-6)/(1-vprob+1e-6)),target[val],[r for r,b in zip(local,val) if b])))
    tx,tz,_,valid,rows,_,_=sparse.cached_panels(model,woo,"train")
    vx,vz,_,vm,vr,_,missing=sparse.cached_panels(model,woo,"val")
    soft=prepare_scores(teacher,head,rows)
    del teacher
    lx=sparse.embeddings(model,px).clone()
    ty=torch.tensor([r["grade"] for r in rows],device="cuda")
    ly=torch.tensor(y,dtype=torch.float,device="cuda")
    by_class=[np.flatnonzero(train & (target==k)) for k in (0,1)]
    banks=sparse.sample_bank(rows)
    for arm,coef in (("matched_replay",0.),("soft_teacher",.9)):
        rng=np.random.default_rng(SEED);model.load_state_dict(torch.load(initial,map_location="cpu",weights_only=True))
        opt=torch.optim.AdamW(model.patch_head.parameters(),lr=.0004,weight_decay=.02)
        history=[];best=None
        for step in range(1,241):
            ids=np.concatenate([rng.choice(ix,20,replace=True) for ix in by_class])
            which=sum([sparse.draw(banks[k,matched],5,rng) for k in (0,1) for matched in (True,False)],[])
            logits=model.patch_head(tx[which]).squeeze(-1);mask=valid[which]
            image_loss=F.cross_entropy(model.fruit_logits(tz[which],logits,mask),ty[which])
            spot_loss=F.binary_cross_entropy_with_logits(model.patch_head(lx[ids]).flatten(),ly[ids])
            kd=F.binary_cross_entropy_with_logits(logits[mask],soft[which][mask])
            loss=spot_loss+.65*image_loss+coef*kd
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.patch_head.parameters(),3.,error_if_nonfinite=True);opt.step()
            if step%30:continue
            with torch.no_grad():
                logits=model.patch_head(vx).squeeze(-1)
                pred=model.fruit_logits(vz,logits,vm).argmax(1).tolist()
            metrics=sparse.counts_for(pred,vr,missing);rank=ranking(metrics)
            history.append(dict(step=step,metrics=metrics,rank=list(rank)))
            if best is None or rank>best:
                best=rank;choice=history[-1];io.atomic(OUT/arm/"weights.pt",model.state_dict(),tensor=True)
        io.atomic(OUT/arm/"history.json",history)
        io.atomic(OUT/arm/"selection.json",dict(architecture=net.ARCH,input_side=320,threshold=1.,
                 weights_sha256=grade.file_sha(OUT/arm/"weights.pt"),step=choice["step"],passed=bool(best[0]),metrics=choice["metrics"]))
        print(arm,choice["metrics"]["auto"]["confusion"],flush=True)
    io.atomic(OUT/"summary.json",{a:io.read(OUT/a/"selection.json") for a in ("matched_replay","soft_teacher")})


if __name__=="__main__":run()
