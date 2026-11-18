"""Paired 224px grade continuation using curated healthy appearance, surface lesions and visible rot."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import math

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import afruit_grade_model as grade
import afruit_grade_train as base
import afruit_grade_matched as matched
import afruit_grade_crop_alignment as align


OUT=grade.ROOT/"experiments/grade_guard_healthy_v1"
INPUT=grade.ROOT/"datasets/grade_guard_hdf_v1/manifest.json"
START=grade.ROOT/"experiments/afruit_grade_matched_v1/paired"
SEED=18112607
EPOCHS,STEPS=18,18


def read_rows(split):
    if split not in ("train","val"):
        raise ValueError("New test is excluded from all training and model selection")
    rows=[r for r in base.read(INPUT)["samples"] if r["split"]==split]
    squares=[]
    for r in rows:
        p=grade.ROOT/r["path"]
        if grade.file_sha(p)!=r["sha256"]:
            raise ValueError("Curated data changed")
        sq=grade.rgb_square(cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_COLOR))
        squares.append(sq)
    return rows,squares


def random_background(square,rng):
    """Vary the white backdrop and black transform triangles without editing interior lesions."""
    hi=square.max(2);lo=square.min(2)
    possible=((lo>=227)|((hi<=23))|((square==[124,116,104]).all(2))).astype(np.uint8)
    _,labels=cv2.connectedComponents(possible)
    edge=np.concatenate([labels[0,:],labels[-1,:],labels[:,0],labels[:,-1]])
    edge=edge[edge>0]
    if not edge.size:
        return square
    mask=np.isin(labels,np.unique(edge))
    shade=rng.uniform(80,235)
    color=np.clip(shade+rng.uniform(-18,18,3),0,255).astype(np.uint8)
    other=square.copy();other[mask]=color
    return other


def runtime():
    model=grade.network().cuda()
    checkpoint=START/"weights.pt"
    if grade.file_sha(checkpoint)!=base.read(START/"selection.json")["weights_sha256"]:
        raise ValueError("Prior matched-grade candidate changed")
    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location="cpu"))
    return model


def get_validation():
    af,images,y=base.open_rows(("val",))["val"]
    c=align.load_cache("val",af)
    rows,photos=read_rows("val")
    return (af,images,y,c,[i for i,v in enumerate(c) if v is not None]),(rows,photos)


def assess(model,validation):
    af,hdf=validation
    out=matched.evaluate(model,af)
    rows,photos=hdf
    logits=base.logits_for(model,photos)
    y=np.asarray([r["grade"] for r in rows],dtype=int)
    out["new_dev"]=align.score(y,logits.argmax(1).numpy())
    return out


def checks(m,parent):
    a,p=m["auto"]["confusion"],parent["auto"]["confusion"]
    d,dp=m["new_dev"]["confusion"],parent["new_dev"]["confusion"]
    return dict(auto_mild=a[1][1]>p[1][1],auto_good=a[0][0]>=p[0][0],
                auto_severe_good=a[2][0]==0,auto_severe_correct=a[2][2]>=p[2][2]-1,
                whole_severe_good=m["whole"]["severe_to_good"]==0,
                new_fresh_severe=d[0][2]<=dp[0][2] and (dp[0][2]==0 or d[0][2]<dp[0][2]),
                new_fresh_good=d[0][0]>=dp[0][0],
                new_rot_good=d[2][0]<=dp[2][0])


def ordering(m,parent):
    c=checks(m,parent); a=m["auto"]["confusion"]; d=m["new_dev"]["confusion"]
    safety=("auto_good","auto_severe_good","whole_severe_good","new_rot_good","auto_severe_correct")
    return (all(c.values()),sum(c[k] for k in safety),sum(c.values()),d[0][0]-d[0][2],
            a[1][1],d[2][2],a[0][0],a[2][2],d[1][1],-m["nll"])


def freeze():
    protocol=dict(start=grade.file_sha(START/"weights.pt"),curated=grade.file_sha(INPUT),
          af_manifest=grade.file_sha(base.MANIFEST),woo_manifest=grade.file_sha(matched.AUX),
          code=grade.file_sha(__file__),seed=SEED,epochs=EPOCHS,steps=STEPS,input_side=224,threshold=1.,
          control="Previous matched Woo model; real AF grade/group-balanced crop and full replay, known Woo camera pairs, distill prior AF severity. 18x18 AdamW updates; BN frozen.",
          intervention="Same replay/update order plus 4 grouped examples in each curated grade per step, full 3-class ground/proxy supervision, additional healthy-to-severe and severe-to-healthy exclusion costs, randomized border-connected studio background/black wedges on half of HDF training views.",
          selection="AF 57 untouched labels + newly reserved 18 grouped validation views. Fixed argmax. Prioritise gate, then safety and count of checks, healthy retention, mild recall. New test excluded.",
          groups=base.read(INPUT)["counts"],proxy="HDF and PSU visible severity labels derived from source labels plus pre-fit visual review; no expert whole-fruit grades.")
    lock=OUT/"protocol.json"
    if lock.is_file() and base.read(lock)!=protocol:raise ValueError("Protocol changed")
    if not lock.is_file():base.atomic(lock,protocol)


def train(arm):
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    af_rng=np.random.default_rng(SEED+1);ext_rng=np.random.default_rng(SEED+2);supp_rng=np.random.default_rng(SEED+3)
    freeze()
    v=get_validation()
    reference=runtime().eval().requires_grad_(False);before=assess(reference,v)
    rec=OUT/"parent_validation.json"
    if not rec.exists():base.atomic(rec,before)
    out=OUT/arm
    if (out/"selection.json").exists():raise ValueError("Completed arm already exists")
    tr,ims,y=base.open_rows(("train",))["train"];crops=align.load_cache("train",tr)
    af_groups=[matched.groups(tr,np.flatnonzero(y==c)) for c in range(3)]
    ext,conditions=matched.auxiliary()
    guard,gimg=read_rows("train")
    supp_groups=[matched.groups(guard,[i for i,r in enumerate(guard) if r["grade"]==c]) for c in range(3)]
    model=runtime();opt=torch.optim.AdamW([{"params":model.features.parameters(),"lr":.000065},{"params":model.classifier.parameters(),"lr":.00027}],weight_decay=.008)
    ay=torch.arange(3,device="cuda").repeat_interleave(6);sy=torch.arange(3,device="cuda").repeat_interleave(4)
    xy=torch.arange(2,device="cuda").repeat(5)
    best=None;history=[];af_sha=hashlib.sha256();aux_sha=hashlib.sha256()
    for ep in range(1,EPOCHS+1):
        model.train()
        for mod in model.modules():
            if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):mod.eval()
        losses=[]
        for step in range(STEPS):
            idx=[matched.pick(g,af_rng) for g in af_groups for _ in range(6)]
            use=af_rng.random(len(idx))<.64
            af_sha.update(np.asarray(idx,dtype='<i4').tobytes()+use.tobytes())
            af_x=grade.tensor_rgb([base.augment(crops[i] if u and crops[i] is not None else ims[i],af_rng) for u,i in zip(use,idx,strict=True)],"cuda")
            z=model(af_x)
            with torch.no_grad():target_prob=reference(af_x).softmax(1)[:,2]
            loss=F.cross_entropy(z,ay,label_smoothing=.02)+.32*F.binary_cross_entropy_with_logits(z[:,2]-z[:,:2].logsumexp(1),target_prob)
            pairs=[conditions[ext_rng.integers(len(conditions))] for _ in range(5)]
            ix=[matched.pick(g,ext_rng) for pair in pairs for g in pair]
            aux_sha.update(np.asarray(ix,dtype='<i4').tobytes())
            ex=grade.tensor_rgb([base.augment(ext[i],ext_rng) for i in ix],"cuda")
            zz=model(ex); ev=zz[:,1]-zz[:,0]
            loss=loss+.45*F.cross_entropy(zz[:,:2],xy)+.12*F.softplus(.5+ev[::2]-ev[1::2]).mean()
            if arm=="grounded":
                ids=[matched.pick(g,supp_rng) for g in supp_groups for _ in range(4)]
                sq=[]
                for i in ids:
                    im=gimg[i]
                    if guard[i]["source"]=="HDF_real" and supp_rng.random()<.55:
                        im=random_background(im,supp_rng)
                    sq.append(base.augment(im,supp_rng))
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.cuda.manual_seed(SEED+ep*STEPS+step)
                    scores=model(grade.tensor_rgb(sq,"cuda"))
                loss=loss+.68*F.cross_entropy(scores,sy,label_smoothing=.025)
                healthy_severe=scores[:4,2]-scores[:4,:2].logsumexp(1)
                severe_healthy=scores[-4:,0]-scores[-4:,1:].logsumexp(1)
                loss=loss+.14*(F.softplus(healthy_severe).mean()+F.softplus(severe_healthy).mean())
            t=((ep-1)*STEPS+step)/(EPOCHS*STEPS-1)
            for group,lr in zip(opt.param_groups,(.000065,.00027),strict=True):group["lr"]=lr*(.10+.90*(1+math.cos(math.pi*t))/2)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();losses.append(loss.item())
        m=assess(model,v); row=dict(epoch=ep,metrics=m,loss=float(np.mean(losses)),checks=checks(m,before))
        history.append(row)
        if best is None or ordering(m,before)>ordering(best["metrics"],before):
            best=row;base.atomic(out/"weights.pt",model.state_dict(),tensor=True)
        base.atomic(out/"history.json",history)
        print(arm,ep,m["auto"]["confusion"],m["new_dev"]["confusion"],"selected",best["epoch"],flush=True)
    receipt=dict(arm=arm,epoch=best["epoch"],updates=EPOCHS*STEPS,metrics=best["metrics"],checks=best["checks"],
                 passed=all(best["checks"].values()),af_exposures=af_sha.hexdigest(),woo_exposures=aux_sha.hexdigest(),
                 input_side=224,threshold=1.,weights_sha256=grade.file_sha(out/"weights.pt"),protocol_sha256=grade.file_sha(OUT/"protocol.json"))
    base.atomic(out/"selection.json",receipt)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("action",choices=["baseline","control","grounded"])
    a=parser.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if a.action=="baseline":
        freeze();r=assess(runtime().eval(),get_validation());base.atomic(OUT/"parent_validation.json",r)
        print(r["auto"]["confusion"],r["new_dev"]["confusion"],flush=True)
    else:train(a.action)
