"""Colour/texture-preserving boundary-invariance control for the apple appearance grader."""
from collections import Counter
import argparse
import hashlib
import math
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import afruit_grade_model as grade
import afruit_grade_matched as prior
import afruit_grade_train as io
import afruit_grade_crop_alignment as align
import guard_grade_sources as sources

OUT=grade.ROOT/"experiments/grade_guard_sources_v2"
INPUT=sources.OUT/"manifest.json"
START=prior.OUT/"paired"
SEED=18112617
EPOCHS,STEPS=14,18


def support(split):
    if split not in ("train","val"):raise ValueError("Only fit and development rows may be opened")
    rows=[r for r in io.read(INPUT)["samples"] if r["split"]==split]
    images=[]
    for r in rows:
        path=grade.ROOT/r["path"]
        if grade.file_sha(path)!=r["sha256"]:raise ValueError("Support image changed")
        image=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
        images.append(grade.rgb_square(image))
    return rows,images


def boundary_view(frame):
    """Replace only the thin contact band of a border-connected light neutral background."""
    rgb=frame.astype(np.int16); lo=rgb.min(2); hi=rgb.max(2)
    pad=(frame==[124,116,104]).all(2)
    neutral=((lo>=200) & (hi-lo<=42)) | ((hi-lo<=6)&(lo>=109)&(hi<=118))
    connected=(neutral|pad).astype(np.uint8)
    _,labels=cv2.connectedComponents(connected)
    edges=np.unique(np.concatenate((labels[0],labels[-1],labels[:,0],labels[:,-1])))
    bg=np.isin(labels,edges[edges>0])&~pad
    if bg.mean()<.035 or bg.mean()>.93:return frame,False
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    band=cv2.morphologyEx(bg.astype(np.uint8),cv2.MORPH_GRADIENT,kernel).astype(bool)&~pad
    if band.mean()<.012:return frame,False
    out=frame.copy()
    smoother=cv2.GaussianBlur(frame,(9,9),1.9)
    out[band]=smoother[band]
    return out,True


def validation():
    af,images,y=io.open_rows(("val",))["val"];crops=align.load_cache("val",af)
    return (af,images,y,crops,[i for i,v in enumerate(crops) if v is not None]),support("val")


def assess(model,v):
    af,(rows,ims)=v
    m=prior.evaluate(model,af)
    logits=io.logits_for(model,ims)
    y=np.array([r["grade"] for r in rows])
    m["new_dev"]=align.score(y,logits.argmax(1).numpy())
    return m


def checks(m,p):
    a,ref=m["auto"]["confusion"],p["auto"]["confusion"]
    d,old=m["new_dev"]["confusion"],p["new_dev"]["confusion"]
    return dict(auto_mild=a[1][1]>ref[1][1],auto_good=a[0][0]>=ref[0][0],
                auto_severe_correct=a[2][2]>=ref[2][2]-1,auto_severe_good=a[2][0]==0,
                whole_severe_good=m["whole"]["severe_to_good"]==0,
                auxiliary_good=d[0][0]>=old[0][0],auxiliary_extreme=d[0][2]<=old[0][2])


def order(m,p):
    c=checks(m,p); a=m["auto"]["confusion"];d=m["new_dev"]["confusion"]
    safe=sum(v for k,v in c.items() if k!="auto_mild")
    return (all(c.values()),safe,a[1][1],d[0][0]-d[0][2],a[0][0],a[2][2],
            m["whole"]["medium_correct"],d[1][1],-m["nll"])


def load():
    p=START/"weights.pt"
    if grade.file_sha(p)!=io.read(START/"selection.json")["weights_sha256"]:raise ValueError("Prior weights changed")
    m=grade.network()
    m.load_state_dict(torch.load(p,map_location="cpu",weights_only=True))
    return m.cuda()


def freeze():
    d=dict(start=grade.file_sha(START/"weights.pt"),support=grade.file_sha(INPUT),af=grade.file_sha(io.MANIFEST),
           woo=grade.file_sha(prior.AUX),code=grade.file_sha(__file__),seed=SEED,epochs=EPOCHS,steps=STEPS,
           arm="grounded: fixed original Woo-selected MobileNet 224 state_dict, fixed training/validation. Border arm adds paired geometry-preserving narrow mobile-band blur consistency once per update from same pictures; no inference changes. Original arm uses unmodified picture pair only.",
           strategy="Accumulate AF canonical ordered CE with original severity and conditional good/minor margins (paired teacher raw, no threshold change); paired Woo supervision; same real three-grade support per step plus grade-extreme cost. Pretrained stages 0:8 and BN stats frozen. Source groups balanced. Borders audited only with frozen rules on all available grades. Training AF only; AF development 57, independently grouped aux validation. Historical HDF and all final photo sets excluded.",
           gate="Improve development AF minor above 3, good >=20, severe>=16 and none to good (also whole view); no new external good losses or extreme errors compared to starting Woo. Prioritize all gates, then safety, minor count, external good, good and severe. No unseen-final data.",
           output="Unmodified MobileNetV3-Small 224, three logits, argmax.",epochs_selected_by="development only")
    p=OUT/"protocol.json"
    if p.exists() and io.read(p)!=d:raise ValueError("Changed locked recipe")
    if not p.exists():io.atomic(p,d)


def run(arm):
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    freeze()
    teacher=load().eval().requires_grad_(False)
    v=validation();parent=assess(teacher,v)
    if not (OUT/"parent_validation.json").is_file():io.atomic(OUT/"parent_validation.json",parent)
    if arm=="baseline":return
    dest=OUT/arm
    if (dest/"selection.json").exists():raise ValueError("Already done")
    af,photos,y=io.open_rows(("train",))["train"];crops=align.load_cache("train",af)
    g_af=[prior.groups(af,np.flatnonzero(y==k)) for k in range(3)]
    imgs,conds=prior.auxiliary()
    aux,external=support("train")
    g_aux=[prior.groups(aux,[i for i,r in enumerate(aux) if r["grade"]==k]) for k in range(3)]
    ar,wr,sr,br=[np.random.default_rng(SEED+k) for k in range(1,5)]
    af_y=torch.arange(3,device="cuda").repeat_interleave(6)
    wo_y=torch.arange(2,device="cuda").repeat(4)
    sx_y=torch.arange(3,device="cuda").repeat_interleave(4)
    model=load()
    for layer in list(model.features)[:8]:layer.requires_grad_(False)
    opt=torch.optim.AdamW([dict(params=[p for p in model.features.parameters() if p.requires_grad],lr=4e-5),
                           dict(params=model.classifier.parameters(),lr=.00015)],weight_decay=.008)
    best=None;history=[];exposures=hashlib.sha256();perturb=Counter()
    for ep in range(1,EPOCHS+1):
        model.train()
        for layer in model.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):layer.eval()
        losslog=[]
        for st in range(STEPS):
            ai=[prior.pick(g,ar) for g in g_af for _ in range(6)]
            actual=ar.random(len(ai))<.60
            wi=[prior.pick(g,wr) for _ in range(4) for g in conds[wr.integers(len(conds))]]
            si=[prior.pick(g,sr) for g in g_aux for _ in range(4)]
            exposures.update(np.array(ai+wi+si,dtype="<i4").tobytes()+actual.tobytes())
            raw=[crops[i] if c and crops[i] is not None else photos[i] for c,i in zip(actual,ai,strict=True)]
            raw+= [imgs[i] for i in wi]+[external[i] for i in si]
            # Produce paired content before photometric augmentation, identically across both arms.
            pair_ix=np.sort(br.choice(len(raw),size=12,replace=False))
            paired=[]
            for i in pair_ix:
                src=raw[i]; aug,ok=boundary_view(src)
                perturb[str(i//6 if i<18 else (i-18)%2 if i<26 else (i-26)//4)]+=int(ok)
                if ok:paired.append((src,aug))
            train_x=[io.augment(im,ar if i<18 else wr if i<26 else sr) for i,im in enumerate(raw)]
            tensor=grade.tensor_rgb(train_x,"cuda")
            logits=model(tensor);az,wz,sz=logits.split((18,8,12))
            with torch.no_grad():tz=teacher(tensor[:18])
            sev=F.binary_cross_entropy_with_logits(az[:,2]-az[:,:2].logsumexp(1),tz.softmax(1)[:,2])
            # T=2 makes the good/minor signal informative even on memorised training photos.
            teacher_minor=(tz[:12,1]-tz[:12,0])/2
            rel=4*F.binary_cross_entropy_with_logits((az[:12,1]-az[:12,0])/2,teacher_minor.sigmoid())
            margin=wz[:,1]-wz[:,0]
            loss=F.cross_entropy(az,af_y,label_smoothing=.02)+.32*sev+.38*rel
            loss=loss+.45*F.cross_entropy(wz[:,:2],wo_y)+.12*F.softplus(.5+margin[::2]-margin[1::2]).mean()
            loss=loss+.48*F.cross_entropy(sz,sx_y,label_smoothing=.02)
            loss=loss+.1*(F.softplus(sz[:4,2]-sz[:4,:2].logsumexp(1)).mean()+F.softplus(sz[-4:,0]-sz[-4:,1:].logsumexp(1)).mean())
            if arm=="border" and paired:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    state=model.training
                    model.eval() # disable stochastic dropout for the geometric-only consistency signal
                    with torch.no_grad():ref=model(grade.tensor_rgb([p[0] for p in paired],"cuda"))
                    zz=model(grade.tensor_rgb([p[1] for p in paired],"cuda"))
                    model.train(state)
                    for bn in model.modules():
                        if isinstance(bn, nn.modules.batchnorm._BatchNorm):bn.eval()
                    loss=loss+.5*4*F.kl_div((zz/2).log_softmax(1),(ref/2).softmax(1),reduction="batchmean")
            t=((ep-1)*STEPS+st)/(EPOCHS*STEPS-1)
            for g,lr in zip(opt.param_groups,(4e-5,.00015),strict=True):g["lr"]=lr*(.1+.9*(1+math.cos(math.pi*t))/2)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();losslog.append(loss.item())
        m=assess(model,v);entry=dict(epoch=ep,metrics=m,checks=checks(m,parent),loss=float(np.mean(losslog)))
        history.append(entry)
        if best is None or order(m,parent)>order(best["metrics"],parent):
            best=entry;io.atomic(dest/"weights.pt",model.state_dict(),tensor=True)
        io.atomic(dest/"history.json",history)
        print(arm,ep,m["auto"]["confusion"],m["new_dev"]["confusion"],flush=True)
    rec=dict(arm=arm,epoch=best["epoch"],updates_completed=STEPS*EPOCHS,threshold=1.,input_side=224,
             metrics=best["metrics"],checks=best["checks"],passed=all(best["checks"].values()),
             weights_sha256=grade.file_sha(dest/"weights.pt"),protocol_sha256=grade.file_sha(OUT/"protocol.json"),
             image_exposures=exposures.hexdigest(),opportunity_count_by_class=dict(perturb),state_dict="ordinary MobileNetV3-Small")
    io.atomic(dest/"selection.json",rec)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm",choices=("baseline","grounded","border"))
    run(parser.parse_args().arm)

